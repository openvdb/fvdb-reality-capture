# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Interactive GARfVDB visualization."""

from __future__ import annotations

import logging
import pathlib
import time

import cv2
import fvdb.viz as fviz
import numpy as np
import torch
from fvdb.types import to_Mat33fBatch, to_Mat44fBatch, to_Vec2iBatch

from .garfvdb import GARfVDB
from .util import apply_pca_projection, fit_pca_projection

# Widget labels shown in the viewer's "Scene Params" window.
_SCALE_WIDGET_NAME = "Grouping scale (normalized)"
_OPACITY_WIDGET_NAME = "Overlay opacity"
_SHOW_WIDGET_NAME = "Show segmentation overlay"
_LOCK_WIDGET_NAME = "Lock PCA colors"


def _camera_tuple_to_c2w(
    center: np.ndarray,
    eye_direction: np.ndarray,
    radius: float,
    up_world: np.ndarray,
    device: torch.device,
) -> torch.Tensor | None:
    position = center - eye_direction * radius
    eye_norm = np.linalg.norm(eye_direction)
    if eye_norm < 1e-8:
        return None
    forward = eye_direction / eye_norm
    right = np.cross(forward, up_world)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        return None
    right /= right_norm
    up = np.cross(right, forward)
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-8:
        return None
    up /= up_norm

    c2w_gl = np.eye(4, dtype=np.float32)
    c2w_gl[:3, 0] = right
    c2w_gl[:3, 1] = up
    c2w_gl[:3, 2] = -forward
    c2w_gl[:3, 3] = position
    opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return torch.from_numpy(c2w_gl @ opengl_to_opencv).to(device)


class GARfVDBOverlayViewer:
    """Drives a GARfVDB feature overlay in an fvdb ``Scene`` with interactive controls.

    Registers three "Scene Params" widgets — a normalized grouping-scale slider (``[0, 1]`` mapped to
    ``[0, max_grouping_scale]``), an overlay-opacity slider, and a show/hide checkbox — and renders one
    overlay frame per :meth:`render_once` call whenever the orbit camera or any widget changes.

    The same object is used by the offline ``frgs show`` viewer (which owns a blocking poll loop) and by
    the live training viewer (which pumps :meth:`render_once` from the training loop). It reads widget
    ``.value`` attributes directly each call, matching the existing camera-polling pattern, so no
    ``poll_widgets``/``on_update`` callback wiring is required.
    """

    def __init__(
        self,
        scene: "fviz.Scene",
        product: GARfVDB,
        *,
        device: str | torch.device = "cuda:0",
        overlay_width: int = 1440,
        overlay_height: int = 720,
        overlay_downsample: int = 2,
        initial_scale_fraction: float = 0.1,
        initial_mask_blend: float = 0.5,
        lock_pca_colors: bool = False,
        add_gaussians: bool = True,
        set_initial_camera: bool = True,
    ) -> None:
        if not 0.0 <= initial_scale_fraction <= 1.0:
            raise ValueError(f"initial_scale_fraction must be in [0, 1], got {initial_scale_fraction}")
        if not 0.0 <= initial_mask_blend <= 1.0:
            raise ValueError(f"initial_mask_blend must be in [0, 1], got {initial_mask_blend}")
        if overlay_width <= 0 or overlay_height <= 0 or overlay_downsample <= 0:
            raise ValueError("Overlay dimensions and downsample must be positive")
        if overlay_width < overlay_downsample or overlay_height < overlay_downsample:
            raise ValueError(
                f"overlay_downsample ({overlay_downsample}) must not exceed overlay_width ({overlay_width}) "
                f"or overlay_height ({overlay_height}); the downsampled render target would be empty."
            )

        self._logger = logging.getLogger(__name__)
        self._scene = scene
        self._product = product
        self._device = torch.device(device)
        self._overlay_width = overlay_width
        self._overlay_height = overlay_height
        self._overlay_downsample = overlay_downsample
        self._render_width = overlay_width // overlay_downsample
        self._render_height = overlay_height // overlay_downsample

        if add_gaussians:
            scene.add_gaussian_splat_3d("Gaussian Splats", product.gaussians)
        if set_initial_camera:
            self._add_cameras_and_lookat()

        # Interactive controls (see class docstring for semantics).
        self._scale_widget = scene.add_slider(_SCALE_WIDGET_NAME, 0.0, 1.0, initial_scale_fraction, 0.01)
        self._opacity_widget = scene.add_slider(_OPACITY_WIDGET_NAME, 0.0, 1.0, initial_mask_blend, 0.01)
        self._show_widget = scene.add_checkbox(_SHOW_WIDGET_NAME, True)
        self._lock_widget = scene.add_checkbox(_LOCK_WIDGET_NAME, lock_pca_colors)

        self._image_view = None
        self._last_camera: tuple[np.ndarray, np.ndarray, float, np.ndarray, float] | None = None
        self._last_widgets: tuple[float, float, bool, bool] | None = None
        # Frozen PCA->RGB transform used while "Lock PCA colors" is on (and the scale it was fit at).
        self._pca_state = None
        self._pca_state_scale: float | None = None

    def _add_cameras_and_lookat(self) -> None:
        """Add training cameras (if present in metadata) and set the initial orbit view."""
        metadata = self._product.reconstruction_metadata
        camera_to_world = metadata.get("camera_to_world_matrices")
        projection_matrices = metadata.get("projection_matrices")
        image_sizes = metadata.get("image_sizes")
        if camera_to_world is not None:
            camera_to_world = to_Mat44fBatch(camera_to_world).cpu()
        if projection_matrices is not None:
            projection_matrices = to_Mat33fBatch(projection_matrices).cpu()
        if image_sizes is not None:
            image_sizes = to_Vec2iBatch(image_sizes).cpu()
        if camera_to_world is not None and projection_matrices is not None:
            self._scene.add_cameras(
                name="Training Cameras",
                camera_to_world_matrices=camera_to_world,
                projection_matrices=projection_matrices,
                image_sizes=image_sizes,
            )

        gaussians = self._product.gaussians
        centroid = gaussians.means.mean(dim=0).detach().cpu().numpy()
        if camera_to_world is not None:
            initial_eye = camera_to_world[0, :3, 3].numpy()
        else:
            extent = gaussians.means.max(dim=0).values - gaussians.means.min(dim=0).values
            initial_eye = centroid + np.ones(3) * float(extent.max().item() / 2.0)
        self._scene.set_camera_lookat(eye=initial_eye, center=centroid, up=[0, 0, 1])

    def _read_camera(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
        scene = self._scene
        return (
            scene.camera_orbit_center.cpu().numpy(),
            scene.camera_orbit_direction.cpu().numpy(),
            float(scene.camera_orbit_radius),
            scene.camera_up_direction.cpu().numpy(),
            float(scene.camera_fov),
        )

    @staticmethod
    def _camera_unchanged(a: tuple, b: tuple) -> bool:
        return (
            np.allclose(a[0], b[0])
            and np.allclose(a[1], b[1])
            and np.isclose(a[2], b[2])
            and np.allclose(a[3], b[3])
            and np.isclose(a[4], b[4])
        )

    def _hide_overlay(self) -> None:
        """Remove the overlay image view so the underlying Gaussian scene is visible again.

        ImageView exposes no visibility toggle, and drawing a fully-transparent frame still draws an
        opaque quad that occludes the 3D scene. So we remove the view outright; render_once re-creates it
        (via ``add_image``) the next time the overlay is enabled.
        """
        if self._image_view is None:
            return
        try:
            fviz._viewer_server.remove_view(self._image_view.scene_name, self._image_view.name)
        except Exception as exc:  # keep the viewer alive even if removal is unavailable
            self._logger.warning("Could not remove GARfVDB overlay view: %s", exc)
            return
        self._image_view = None

    def render_once(self) -> bool:
        """Render/refresh the overlay if the camera or any widget changed since the last call.

        Returns:
            ``True`` if the camera or a widget changed since the previous call (i.e. work was done),
            ``False`` if nothing changed. Callers use this to render continuously while the camera is
            moving and idle-sleep only when there is nothing to do.
        """
        camera = self._read_camera()
        widgets = (
            float(self._scale_widget.value),
            float(self._opacity_widget.value),
            bool(self._show_widget.value),
            bool(self._lock_widget.value),
        )
        if (
            self._last_camera is not None
            and self._last_widgets is not None
            and self._camera_unchanged(camera, self._last_camera)
            and widgets == self._last_widgets
        ):
            return False
        self._last_camera = camera
        self._last_widgets = widgets

        scale_fraction, mask_blend, show_overlay, lock_pca = widgets
        if not show_overlay:
            self._hide_overlay()
            return True

        c2w = _camera_tuple_to_c2w(*camera[:4], self._device)
        if c2w is None:
            return True
        focal = self._render_height / (2.0 * np.tan(camera[4] / 2.0))
        projection = torch.tensor(
            [
                [focal, 0.0, self._render_width / 2.0],
                [0.0, focal, self._render_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self._device,
        )
        scale = scale_fraction * self._product.max_grouping_scale
        try:
            features, alpha = self._product.render_features(
                c2w,
                projection,
                (self._render_width, self._render_height),
                scale,
            )
            if not torch.isfinite(features).all():
                self._logger.warning("Skipping GARfVDB frame containing non-finite features")
                return True
            if features.shape[-1] < 3:
                self._logger.warning(
                    "Skipping GARfVDB overlay: PCA projection requires at least 3 feature channels, got %d",
                    features.shape[-1],
                )
                return True
            pca_mask = alpha.squeeze(-1) > 0
            # When locked, reuse the frozen PCA->RGB transform so colors don't flicker as the camera
            # moves. Refit when unlocked, or when locking for the first time / after the scale changes
            # (a different grouping scale produces a different feature field).
            if lock_pca:
                if self._pca_state is None or self._pca_state_scale != scale_fraction:
                    self._pca_state = fit_pca_projection(features, 3, mask=pca_mask)
                    self._pca_state_scale = scale_fraction
                pca_state = self._pca_state
            else:
                self._pca_state = None
                self._pca_state_scale = None
                pca_state = fit_pca_projection(features, 3, mask=pca_mask)
            rgb = apply_pca_projection(features, pca_state, mask=pca_mask)[0]
            rgba = torch.cat([rgb, alpha[0] * mask_blend], dim=-1).clamp(0, 1)
            rgba_image = (rgba.detach().cpu().numpy() * 255).astype(np.uint8)
            if self._overlay_downsample > 1:
                rgba_image = cv2.resize(
                    rgba_image,
                    (self._overlay_width, self._overlay_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            flat_rgba = rgba_image.flatten()
            if self._image_view is None:
                self._image_view = self._scene.add_image(
                    name="GARfVDB Features",
                    width=self._overlay_width,
                    height=self._overlay_height,
                    rgba_image=flat_rgba,
                )
            else:
                self._image_view.update(flat_rgba)
        except (RuntimeError, ValueError) as exc:
            self._logger.warning("Could not render GARfVDB overlay: %s", exc)
        return True


def show_garfvdb_bundle(
    path: str | pathlib.Path,
    *,
    viewer_port: int = 8080,
    viewer_ip_address: str = "127.0.0.1",
    verbose: bool = False,
    device: str | torch.device = "cuda:0",
    scale_fraction: float = 0.1,
    mask_blend: float = 0.5,
    lock_pca_colors: bool = False,
    overlay_width: int = 1440,
    overlay_height: int = 720,
    overlay_downsample: int = 2,
    idle_poll_interval: float = 1.0 / 120.0,
) -> None:
    """Load a GARfVDB bundle and display its gaussians with a live, interactive feature overlay.

    ``scale_fraction`` and ``mask_blend`` seed the initial positions of the interactive grouping-scale and
    overlay-opacity sliders; both can be adjusted live in the viewer, along with a show/hide checkbox and a
    "Lock PCA colors" checkbox (seeded by ``lock_pca_colors``) that freezes the feature coloring so it does
    not flicker as the camera moves.

    ``idle_poll_interval`` is how long to sleep between camera polls **only when nothing changed**; while
    the camera is moving the overlay re-renders back-to-back with no added delay.
    """
    if not 0.0 <= scale_fraction <= 1.0:
        raise ValueError(f"scale_fraction must be in [0, 1], got {scale_fraction}")
    if not 0.0 <= mask_blend <= 1.0:
        raise ValueError(f"mask_blend must be in [0, 1], got {mask_blend}")
    if overlay_width <= 0 or overlay_height <= 0 or overlay_downsample <= 0:
        raise ValueError("Overlay dimensions and downsample must be positive")

    logger = logging.getLogger(__name__)
    # Vulkan must be initialized before loading CUDA payloads.
    fviz.init(ip_address=viewer_ip_address, port=viewer_port, verbose=verbose)
    product = GARfVDB.load(path, device=device)
    scene = fviz.get_scene("GARfVDB Instance Segmentation")
    viewer = GARfVDBOverlayViewer(
        scene,
        product,
        device=device,
        overlay_width=overlay_width,
        overlay_height=overlay_height,
        overlay_downsample=overlay_downsample,
        initial_scale_fraction=scale_fraction,
        initial_mask_blend=mask_blend,
        lock_pca_colors=lock_pca_colors,
    )
    fviz.show()
    logger.info("GARfVDB viewer running at http://%s:%d", viewer_ip_address, viewer_port)

    try:
        while True:
            # Render as fast as possible while the camera/widgets are changing (render_once returns True),
            # and only sleep when there is nothing to do, so updates track camera motion in real time
            # instead of waiting for the mouse to stop.
            if not viewer.render_once():
                time.sleep(idle_poll_interval)
    except KeyboardInterrupt:
        logger.info("Shutting down GARfVDB viewer")


__all__ = ["show_garfvdb_bundle", "GARfVDBOverlayViewer"]
