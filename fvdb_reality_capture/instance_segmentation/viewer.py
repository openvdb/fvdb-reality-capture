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
from .util import pca_projection_fast


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


def show_garfvdb_bundle(
    path: str | pathlib.Path,
    *,
    viewer_port: int = 8080,
    viewer_ip_address: str = "127.0.0.1",
    verbose: bool = False,
    device: str | torch.device = "cuda",
    scale_fraction: float = 0.1,
    mask_blend: float = 0.5,
    overlay_width: int = 1440,
    overlay_height: int = 720,
    overlay_downsample: int = 2,
    camera_check_interval: float = 0.5,
) -> None:
    """Load a GARfVDB bundle and display its carrier with a live feature overlay."""
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
    device_obj = torch.device(device)
    scene = fviz.get_scene("GARfVDB Instance Segmentation")
    scene.add_gaussian_splat_3d("Gaussian Carrier", product.carrier)

    metadata = product.reconstruction_metadata
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
        scene.add_cameras(
            name="Training Cameras",
            camera_to_world_matrices=camera_to_world,
            projection_matrices=projection_matrices,
            image_sizes=image_sizes,
        )

    centroid = product.carrier.means.mean(dim=0).detach().cpu().numpy()
    if camera_to_world is not None:
        initial_eye = camera_to_world[0, :3, 3].numpy()
    else:
        extent = product.carrier.means.max(dim=0).values - product.carrier.means.min(dim=0).values
        initial_eye = centroid + np.ones(3) * float(extent.max().item() / 2.0)
    scene.set_camera_lookat(eye=initial_eye, center=centroid, up=[0, 0, 1])
    fviz.show()

    render_width = overlay_width // overlay_downsample
    render_height = overlay_height // overlay_downsample
    scale = scale_fraction * product.max_grouping_scale
    image_view = None
    last_camera: tuple[np.ndarray, np.ndarray, float, np.ndarray, float] | None = None
    logger.info("GARfVDB viewer running at http://%s:%d", viewer_ip_address, viewer_port)

    try:
        while True:
            time.sleep(camera_check_interval)
            camera = (
                scene.camera_orbit_center.cpu().numpy(),
                scene.camera_orbit_direction.cpu().numpy(),
                float(scene.camera_orbit_radius),
                scene.camera_up_direction.cpu().numpy(),
                float(scene.camera_fov),
            )
            if last_camera is not None:
                unchanged = (
                    np.allclose(camera[0], last_camera[0])
                    and np.allclose(camera[1], last_camera[1])
                    and np.isclose(camera[2], last_camera[2])
                    and np.allclose(camera[3], last_camera[3])
                    and np.isclose(camera[4], last_camera[4])
                )
                if unchanged:
                    continue
            last_camera = camera
            c2w = _camera_tuple_to_c2w(*camera[:4], device_obj)
            if c2w is None:
                continue
            focal = render_height / (2.0 * np.tan(camera[4] / 2.0))
            projection = torch.tensor(
                [
                    [focal, 0.0, render_width / 2.0],
                    [0.0, focal, render_height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
                device=device_obj,
            )
            try:
                features, alpha = product.render_features(
                    c2w,
                    projection,
                    (render_width, render_height),
                    scale,
                )
                if not torch.isfinite(features).all():
                    logger.warning("Skipping GARfVDB frame containing non-finite features")
                    continue
                rgb = pca_projection_fast(features, 3, mask=alpha.squeeze(-1) > 0)[0]
                rgba = torch.cat([rgb, alpha[0] * mask_blend], dim=-1).clamp(0, 1)
                rgba_image = (rgba.detach().cpu().numpy() * 255).astype(np.uint8)
                if overlay_downsample > 1:
                    rgba_image = cv2.resize(
                        rgba_image,
                        (overlay_width, overlay_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                flat_rgba = rgba_image.flatten()
                if image_view is None:
                    image_view = scene.add_image(
                        name="GARfVDB Features",
                        width=overlay_width,
                        height=overlay_height,
                        rgba_image=flat_rgba,
                    )
                else:
                    image_view.update(flat_rgba)
            except (RuntimeError, ValueError) as exc:
                logger.warning("Could not render GARfVDB overlay: %s", exc)
    except KeyboardInterrupt:
        logger.info("Shutting down GARfVDB viewer")


__all__ = ["show_garfvdb_bundle"]
