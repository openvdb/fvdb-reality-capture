# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Adapters from reality-capture's Gaussian conventions to gsplat."""

from __future__ import annotations

import gsplat
import torch
from gsplat.cuda._wrapper import UnscentedTransformParameters

from fvdb.jagged_tensor import JaggedTensor

from ..enums import CameraModel


_OPENCV_CAMERA_MODELS = (
    CameraModel.OPENCV_RADTAN_5,
    CameraModel.OPENCV_RATIONAL_8,
    CameraModel.OPENCV_RADTAN_THIN_PRISM_9,
    CameraModel.OPENCV_THIN_PRISM_12,
)


def camera_model_for_gsplat(camera_model: CameraModel) -> str:
    if camera_model == CameraModel.ORTHOGRAPHIC:
        return "ortho"
    if camera_model == CameraModel.PINHOLE or camera_model in _OPENCV_CAMERA_MODELS:
        return "pinhole"
    raise ValueError(f"Unsupported camera model: {camera_model}")


def distortion_coeffs_for_gsplat(
    distortion_coeffs: torch.Tensor | None,
    camera_model: CameraModel,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if camera_model in (CameraModel.PINHOLE, CameraModel.ORTHOGRAPHIC):
        return None, None, None
    if camera_model not in _OPENCV_CAMERA_MODELS:
        raise ValueError(f"Unsupported camera model: {camera_model}")
    if distortion_coeffs is None or distortion_coeffs.ndim != 2 or distortion_coeffs.shape[1] != 12:
        raise ValueError("OpenCV distortion_coeffs must have shape [C, 12]")
    return (
        distortion_coeffs[:, :6].contiguous(),
        distortion_coeffs[:, 6:8].contiguous(),
        distortion_coeffs[:, 8:12].contiguous(),
    )


def project_gaussians_analytic(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    image_width: int,
    image_height: int,
    eps2d: float,
    near: float,
    far: float,
    radius_clip: float,
    calc_compensations: bool,
    camera_model: CameraModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return gsplat.fully_fused_projection(
        means,
        None,
        quats,
        log_scales.exp(),
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d=eps2d,
        near_plane=near,
        far_plane=far,
        radius_clip=radius_clip,
        packed=False,
        calc_compensations=calc_compensations,
        camera_model=camera_model_for_gsplat(camera_model),
    )


def project_gaussians_unscented(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    image_width: int,
    image_height: int,
    eps2d: float,
    near: float,
    far: float,
    radius_clip: float,
    calc_compensations: bool,
    camera_model: CameraModel,
    distortion_coeffs: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    radial, tangential, thin_prism = distortion_coeffs_for_gsplat(distortion_coeffs, camera_model)
    return gsplat.fully_fused_projection_with_ut(
        means,
        quats,
        log_scales.exp(),
        None,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d=eps2d,
        near_plane=near,
        far_plane=far,
        radius_clip=radius_clip,
        calc_compensations=calc_compensations,
        camera_model=camera_model_for_gsplat(camera_model),
        ut_params=UnscentedTransformParameters(0.1, 2.0, 0.0, 0.1, True),
        radial_coeffs=radial,
        tangential_coeffs=tangential,
        thin_prism_coeffs=thin_prism,
        rolling_shutter=gsplat.RollingShutterType.GLOBAL,
    )


def evaluate_spherical_harmonics(
    degree: int,
    means: torch.Tensor,
    viewmats: torch.Tensor,
    coeffs: torch.Tensor,
    radii: torch.Tensor,
    camera_ids: torch.Tensor | None = None,
    gaussian_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    masks = (radii[..., 0] > 0) & (radii[..., 1] > 0)
    batch_ids = None
    if camera_ids is not None or gaussian_ids is not None:
        if camera_ids is None or gaussian_ids is None:
            raise ValueError("camera_ids and gaussian_ids must be provided together")
        camera_ids = camera_ids.to(torch.int64)
        gaussian_ids = gaussian_ids.to(torch.int64)
        batch_ids = torch.zeros_like(camera_ids)
    colors = gsplat.spherical_harmonics(
        degree,
        means,
        viewmats,
        coeffs,
        masks,
        batch_ids=batch_ids,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    return torch.where(masks.unsqueeze(-1), colors + 0.5, 0.0)


def intersect_tiles(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    num_images: int,
    tile_size: int,
    tile_height: int,
    tile_width: int,
    *,
    conics: torch.Tensor | None = None,
    opacities: torch.Tensor | None = None,
    image_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = means2d.ndim == 2
    gaussian_ids = None
    if packed:
        gaussian_ids = torch.arange(means2d.shape[0], dtype=torch.int64, device=means2d.device)
        if image_ids is None:
            raise ValueError("image_ids are required for packed tile intersection")
        image_ids = image_ids.to(torch.int64)
    _, isect_ids, flatten_ids = gsplat.isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=num_images if packed else None,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
        conics=conics,
        opacities=opacities,
    )
    offsets = gsplat.isect_offset_encode(isect_ids, num_images, tile_width, tile_height)
    return offsets.to(torch.int64).contiguous(), flatten_ids.contiguous()


def jagged_image_ids(pixels: JaggedTensor) -> torch.Tensor:
    if pixels.jidx.numel() == 0:
        return torch.zeros(pixels.jdata.shape[0], dtype=torch.int32, device=pixels.device)
    return pixels.jidx.to(torch.int32).contiguous()


def intersect_tiles_sparse(
    pixels: JaggedTensor,
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    num_images: int,
    tile_size: int,
    tile_height: int,
    tile_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_ids = jagged_image_ids(pixels)
    active_tiles, active_tile_mask, tile_pixel_mask, tile_pixel_cumsum, pixel_map = gsplat.build_sparse_tile_layout(
        pixels.jdata.to(torch.int32), image_ids, num_images, tile_size, tile_width, tile_height
    )
    tile_offsets, flatten_ids = gsplat.isect_tiles_sparse(
        means2d,
        radii,
        depths,
        active_tile_mask,
        active_tiles,
        num_images,
        tile_size,
        tile_width,
        tile_height,
    )
    return (
        tile_offsets.to(torch.int64).contiguous(),
        flatten_ids,
        active_tiles,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
    )
