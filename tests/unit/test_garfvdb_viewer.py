# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GARfVDB overlay viewer core (no live Vulkan viewer required)."""

import numpy as np
import torch

from unittest import mock

from fvdb_reality_capture.instance_segmentation.util import apply_pca_projection, fit_pca_projection
from fvdb_reality_capture.instance_segmentation.viewer import (
    _LOCK_WIDGET_NAME,
    _OPACITY_WIDGET_NAME,
    _SCALE_WIDGET_NAME,
    _SHOW_WIDGET_NAME,
    GARfVDBOverlayViewer,
)


class _FakeWidget:
    def __init__(self, value):
        self.value = value


class _FakeImageView:
    def __init__(self, name, scene_name):
        self.name = name
        self.scene_name = scene_name
        self.last_update = None
        self.update_count = 0

    def update(self, rgba_image):
        self.last_update = np.asarray(rgba_image)
        self.update_count += 1


class _FakeScene:
    """Minimal stand-in for fvdb.viz.Scene exercising the widgets + overlay code paths."""

    _SCENE_NAME = "fake-scene"

    def __init__(self):
        self.widgets = {}
        self.image_view = None
        self.added_image_kwargs = None
        self.added_image_count = 0
        # Orbit camera state read by the viewer each frame.
        self.camera_orbit_center = torch.tensor([0.0, 0.0, 0.0])
        self.camera_orbit_direction = torch.tensor([0.0, 0.0, 1.0])
        self.camera_orbit_radius = 3.0
        self.camera_up_direction = torch.tensor([0.0, 1.0, 0.0])
        self.camera_fov = 1.0

    def add_slider(self, name, min, max, initial=None, step=0.01):
        self.widgets[name] = _FakeWidget(float(initial if initial is not None else min))
        return self.widgets[name]

    def add_checkbox(self, name, initial=False):
        self.widgets[name] = _FakeWidget(bool(initial))
        return self.widgets[name]

    def add_image(self, name, width, height, rgba_image):
        self.added_image_kwargs = dict(name=name, width=width, height=height)
        self.added_image_count += 1
        self.image_view = _FakeImageView(name=name, scene_name=self._SCENE_NAME)
        self.image_view.update(rgba_image)
        return self.image_view


class _FakeProduct:
    """Stand-in for GARfVDB that records the raw scale passed to render_features."""

    def __init__(self, max_grouping_scale=4.0, render_wh=(8, 4)):
        self.max_grouping_scale = max_grouping_scale
        self.reconstruction_metadata = {}
        self.render_calls = []
        self._render_wh = render_wh

    def render_features(self, c2w, projection, image_size, scale):
        self.render_calls.append(scale)
        w, h = self._render_wh
        torch.manual_seed(0)
        features = torch.rand(1, h, w, 4)
        alpha = torch.ones(1, h, w, 1)
        return features, alpha


def _make_viewer(scene, product, **kwargs):
    return GARfVDBOverlayViewer(
        scene,
        product,
        device="cpu",
        overlay_width=16,
        overlay_height=8,
        overlay_downsample=2,
        add_carrier=False,
        set_initial_camera=False,
        **kwargs,
    )


def test_widgets_registered_with_expected_names_and_initials():
    scene = _FakeScene()
    _make_viewer(
        scene, _FakeProduct(), initial_scale_fraction=0.25, initial_mask_blend=0.6, lock_pca_colors=True
    )
    assert set(scene.widgets) == {
        _SCALE_WIDGET_NAME,
        _OPACITY_WIDGET_NAME,
        _SHOW_WIDGET_NAME,
        _LOCK_WIDGET_NAME,
    }
    assert scene.widgets[_SCALE_WIDGET_NAME].value == 0.25
    assert scene.widgets[_OPACITY_WIDGET_NAME].value == 0.6
    assert scene.widgets[_SHOW_WIDGET_NAME].value is True
    assert scene.widgets[_LOCK_WIDGET_NAME].value is True


def test_normalized_scale_maps_to_raw_grouping_scale():
    scene = _FakeScene()
    product = _FakeProduct(max_grouping_scale=4.0)
    viewer = _make_viewer(scene, product, initial_scale_fraction=0.25)

    viewer.render_once()
    assert product.render_calls == [0.25 * 4.0]  # normalized 0.25 -> raw 1.0
    assert scene.image_view is not None
    assert scene.added_image_kwargs == {"name": "GARfVDB Features", "width": 16, "height": 8}


def test_no_rerender_when_nothing_changes():
    scene = _FakeScene()
    product = _FakeProduct()
    viewer = _make_viewer(scene, product)

    assert viewer.render_once() is True  # first call always renders
    assert viewer.render_once() is False  # camera + widgets unchanged -> no work, caller can idle-sleep
    assert len(product.render_calls) == 1

    scene.camera_orbit_radius = 7.0  # camera moved -> renders again immediately (no delay)
    assert viewer.render_once() is True
    assert len(product.render_calls) == 2


def test_scale_slider_change_triggers_rerender():
    scene = _FakeScene()
    product = _FakeProduct(max_grouping_scale=10.0)
    viewer = _make_viewer(scene, product, initial_scale_fraction=0.1)

    viewer.render_once()
    scene.widgets[_SCALE_WIDGET_NAME].value = 0.5
    viewer.render_once()
    assert product.render_calls == [0.1 * 10.0, 0.5 * 10.0]


def test_hide_overlay_removes_view_and_re_adds_on_show(monkeypatch):
    # Hiding removes the overlay view outright (a transparent frame still occludes the carrier),
    # so the underlying Gaussian carrier scene becomes visible again.
    import fvdb.viz._viewer_server as vs

    removed: list[tuple[str, str]] = []
    monkeypatch.setattr(vs, "remove_view", lambda scene_name, name: removed.append((scene_name, name)))

    scene = _FakeScene()
    product = _FakeProduct()
    viewer = _make_viewer(scene, product)

    viewer.render_once()  # overlay shown -> image view created
    assert scene.added_image_count == 1
    assert len(product.render_calls) == 1

    scene.widgets[_SHOW_WIDGET_NAME].value = False
    viewer.render_once()
    # The view was removed (not just cleared) and no new render happened while hidden.
    assert removed == [(_FakeScene._SCENE_NAME, "GARfVDB Features")]
    assert viewer._image_view is None
    assert len(product.render_calls) == 1

    # Re-enabling the overlay recreates the image view.
    scene.widgets[_SHOW_WIDGET_NAME].value = True
    viewer.render_once()
    assert scene.added_image_count == 2
    assert len(product.render_calls) == 2


def _fake_apply(features, state, mask=None):
    # Stand-in for apply_pca_projection with the shape render_once expects ([B, H, W, 3]).
    return torch.zeros(features.shape[0], features.shape[1], features.shape[2], 3)


def _patch_pca():
    fit = mock.patch(
        "fvdb_reality_capture.instance_segmentation.viewer.fit_pca_projection",
        return_value=object(),
    )
    apply = mock.patch(
        "fvdb_reality_capture.instance_segmentation.viewer.apply_pca_projection",
        side_effect=_fake_apply,
    )
    return fit, apply


def test_locked_pca_reuses_projection_across_camera_moves():
    scene = _FakeScene()
    viewer = _make_viewer(scene, _FakeProduct())
    scene.widgets[_LOCK_WIDGET_NAME].value = True

    fit_patch, apply_patch = _patch_pca()
    with fit_patch as fit, apply_patch:
        viewer.render_once()
        scene.camera_orbit_radius = 5.0  # simulate a camera move
        viewer.render_once()
        scene.camera_orbit_radius = 9.0
        viewer.render_once()

    # PCA is fit once and reused across camera moves -> stable colors.
    assert fit.call_count == 1


def test_unlocked_pca_refits_each_frame():
    scene = _FakeScene()
    viewer = _make_viewer(scene, _FakeProduct())  # lock off by default

    fit_patch, apply_patch = _patch_pca()
    with fit_patch as fit, apply_patch:
        viewer.render_once()
        scene.camera_orbit_radius = 5.0
        viewer.render_once()

    assert fit.call_count == 2


def test_lock_refits_when_grouping_scale_changes():
    scene = _FakeScene()
    viewer = _make_viewer(scene, _FakeProduct(), initial_scale_fraction=0.1)
    scene.widgets[_LOCK_WIDGET_NAME].value = True

    fit_patch, apply_patch = _patch_pca()
    with fit_patch as fit, apply_patch:
        viewer.render_once()
        scene.widgets[_SCALE_WIDGET_NAME].value = 0.5  # different grouping scale -> different feature field
        viewer.render_once()

    assert fit.call_count == 2


def test_locked_projection_maps_same_feature_to_same_color():
    torch.manual_seed(0)
    feats_a = torch.randn(1, 10, 10, 6)
    state = fit_pca_projection(feats_a, 3)

    # A different "frame" whose overall distribution differs, but which contains one identical feature vector.
    feats_b = torch.randn(1, 10, 10, 6) * 4.0
    feats_b[0, 0, 0] = feats_a[0, 5, 5]

    rgb_a = apply_pca_projection(feats_a, state)  # [100, 3] (no mask)
    rgb_b = apply_pca_projection(feats_b, state)
    # The frozen transform maps the identical feature to the identical color across frames.
    assert torch.allclose(rgb_a[5 * 10 + 5], rgb_b[0], atol=1e-5)
