# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""``torch.autograd.Function`` wrappers over the Gaussian splatting kernels in :mod:`fvdb.functional`.

``fvdb.functional`` exposes each kernel's forward and backward pass as separate, non-differentiable
functions. The classes here pair them up so that gradients flow through the composable stages in
this package. They are private; use the stage functions instead.
"""

from __future__ import annotations

from typing import Any

import torch
from fvdb import JaggedTensor
from fvdb import functional as F

# ---------------------------------------------------------------------------
#  Projection (analytic)
# ---------------------------------------------------------------------------


class _ProjectGaussiansFn(torch.autograd.Function):
    """Analytic (EWA) projection of 3D Gaussians to 2D with gradients to the 3D parameters."""

    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        world_to_cam: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        eps2d: float,
        near: float,
        far: float,
        min_radius_2d: float,
        calc_compensations: bool,
        ortho: bool,
        accum_grad_norms: torch.Tensor | None = None,
        accum_step_counts: torch.Tensor | None = None,
        accum_max_radii: torch.Tensor | None = None,
    ):
        radii, means2d, depths, conics, compensations = F.project_gaussians_analytic_fwd(
            means,
            quats,
            log_scales,
            world_to_cam,
            projection_matrices,
            image_width,
            image_height,
            eps2d,
            near,
            far,
            min_radius_2d,
            calc_compensations,
            ortho,
        )
        if not calc_compensations:
            compensations = None

        to_save = [means, quats, log_scales, world_to_cam, projection_matrices, radii, conics]
        if compensations is not None:
            to_save.append(compensations)
        ctx.save_for_backward(*to_save)

        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.eps2d = eps2d
        ctx.calc_compensations = calc_compensations
        ctx.ortho = ortho
        ctx.accum_grad_norms = accum_grad_norms
        ctx.accum_step_counts = accum_step_counts
        ctx.accum_max_radii = accum_max_radii

        if compensations is not None:
            return radii, means2d, depths, conics, compensations
        return radii, means2d, depths, conics

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        grad_means2d = grad_outputs[1]
        grad_depths = grad_outputs[2]
        grad_conics = grad_outputs[3]
        maybe_grad_comp = grad_outputs[4:]
        if grad_means2d is not None:
            grad_means2d = grad_means2d.contiguous()
        if grad_depths is not None:
            grad_depths = grad_depths.contiguous()
        if grad_conics is not None:
            grad_conics = grad_conics.contiguous()

        grad_compensations: torch.Tensor | None = None
        if ctx.calc_compensations and maybe_grad_comp:
            gc = maybe_grad_comp[0]
            if gc is not None:
                grad_compensations = gc.contiguous()

        saved = ctx.saved_tensors
        means, quats, log_scales, world_to_cam, projection_matrices, radii, conics = saved[:7]
        compensations = saved[7] if ctx.calc_compensations else None

        assert grad_means2d is not None
        assert grad_depths is not None
        assert grad_conics is not None
        d_means, _, d_quats, d_scales, d_w2c = F.project_gaussians_analytic_bwd(
            means,
            quats,
            log_scales,
            world_to_cam,
            projection_matrices,
            compensations,
            ctx.image_width,
            ctx.image_height,
            ctx.eps2d,
            radii,
            conics,
            grad_means2d,
            grad_depths,
            grad_conics,
            grad_compensations,
            ctx.needs_input_grad[3],
            ctx.ortho,
            ctx.accum_grad_norms,
            ctx.accum_max_radii,
            ctx.accum_step_counts,
        )

        return (d_means, d_quats, d_scales, d_w2c) + (None,) * 12


# ---------------------------------------------------------------------------
#  Projection (analytic, jagged batches of scenes)
# ---------------------------------------------------------------------------


class _ProjectGaussiansJaggedFn(torch.autograd.Function):
    """Analytic projection for a batch of scenes with varying Gaussian and camera counts."""

    @staticmethod
    def forward(
        ctx,
        g_sizes: torch.Tensor,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        c_sizes: torch.Tensor,
        world_to_cam: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        eps2d: float,
        near: float,
        far: float,
        min_radius_2d: float,
        ortho: bool,
    ):
        radii, means2d, depths, conics, compensations = F.project_gaussians_analytic_jagged_fwd(
            g_sizes,
            means,
            quats,
            scales,
            c_sizes,
            world_to_cam,
            projection_matrices,
            image_width,
            image_height,
            eps2d,
            near,
            far,
            min_radius_2d,
            ortho,
        )

        ctx.save_for_backward(g_sizes, means, quats, scales, c_sizes, world_to_cam, projection_matrices, radii, conics)
        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.eps2d = eps2d
        ctx.ortho = ortho

        return radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        grad_means2d = grad_outputs[1]
        grad_depths = grad_outputs[2]
        grad_conics = grad_outputs[3]
        # grad_outputs[4] is grad_compensations, which the jagged backward does not consume.
        if grad_means2d is not None:
            grad_means2d = grad_means2d.contiguous()
        if grad_depths is not None:
            grad_depths = grad_depths.contiguous()
        if grad_conics is not None:
            grad_conics = grad_conics.contiguous()

        g_sizes, means, quats, scales, c_sizes = ctx.saved_tensors[:5]
        world_to_cam, projection_matrices, radii, conics = ctx.saved_tensors[5:]

        assert grad_means2d is not None
        assert grad_depths is not None
        assert grad_conics is not None
        d_means, _, d_quats, d_scales, d_w2c = F.project_gaussians_analytic_jagged_bwd(
            g_sizes,
            means,
            quats,
            scales,
            c_sizes,
            world_to_cam,
            projection_matrices,
            ctx.image_width,
            ctx.image_height,
            ctx.eps2d,
            radii,
            conics,
            grad_means2d,
            grad_depths,
            grad_conics,
            ctx.needs_input_grad[5],
            ctx.ortho,
        )

        return (None, d_means, d_quats, d_scales, None, d_w2c) + (None,) * 8


# ---------------------------------------------------------------------------
#  Spherical harmonics evaluation
# ---------------------------------------------------------------------------


class _EvaluateGaussianSHFn(torch.autograd.Function):
    """Spherical-harmonics feature evaluation with gradients to the coefficients, means and cameras."""

    @staticmethod
    def forward(
        ctx,
        sh_degree_to_use: int,
        num_cameras: int,
        means: torch.Tensor,  # [N, 3]
        world_to_cam: torch.Tensor,  # [C, 4, 4]
        camera_ids: torch.Tensor,  # empty (dense) or [M] int32 (packed)
        gaussian_ids: torch.Tensor,  # empty (dense) or [M] int32 (packed)
        sh0_coeffs: torch.Tensor,  # [N, 1, D] (dense) or [M, 1, D] (packed)
        shN_coeffs: torch.Tensor,  # [N, K-1, D] (dense) or [M, K-1, D] (packed)
        radii: torch.Tensor,  # [C, N, 2] (dense) or [1, M, 2] (packed)
    ) -> torch.Tensor:
        features = F.evaluate_spherical_harmonics_fwd(
            sh_degree_to_use,
            num_cameras,
            means,
            world_to_cam,
            camera_ids,
            gaussian_ids,
            sh0_coeffs,
            shN_coeffs,
            radii,
        )

        ctx.save_for_backward(means, world_to_cam, camera_ids, gaussian_ids, shN_coeffs, radii)
        ctx.sh_degree_to_use = sh_degree_to_use
        ctx.num_cameras = num_cameras
        ctx.num_gaussians = sh0_coeffs.size(0)

        return features

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_loss_d_colors = grad_outputs[0]
        if d_loss_d_colors is None:
            return (None,) * 9
        d_loss_d_colors = d_loss_d_colors.contiguous()

        means, world_to_cam, camera_ids, gaussian_ids, shN_coeffs, radii = ctx.saved_tensors

        d_sh0, d_shN, d_means, d_w2c = F.evaluate_spherical_harmonics_bwd(
            ctx.sh_degree_to_use,
            ctx.num_cameras,
            ctx.num_gaussians,
            means,
            world_to_cam,
            camera_ids,
            gaussian_ids,
            shN_coeffs,
            d_loss_d_colors,
            radii,
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[3],
        )

        return (None, None, d_means, d_w2c, None, None, d_sh0, d_shN, None)


# ---------------------------------------------------------------------------
#  Dense screen-space rasterization
# ---------------------------------------------------------------------------


def _save_optional(ctx, to_save: list[torch.Tensor], backgrounds: torch.Tensor | None, masks: torch.Tensor | None):
    """Append the optional rasterization inputs to ``to_save`` and record which were present."""
    ctx.has_backgrounds = backgrounds is not None
    ctx.has_masks = masks is not None
    if backgrounds is not None:
        to_save.append(backgrounds)
    if masks is not None:
        to_save.append(masks)


def _load_optional(ctx, saved: tuple[torch.Tensor, ...], first: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Read back the optional rasterization inputs saved by :func:`_save_optional`."""
    backgrounds = saved[first] if ctx.has_backgrounds else None
    masks = saved[first + int(ctx.has_backgrounds)] if ctx.has_masks else None
    return backgrounds, masks


class _RasterizeScreenSpaceGaussiansFn(torch.autograd.Function):
    """Dense alpha-blending of projected Gaussians with gradients to the 2D quantities."""

    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        features: torch.Tensor,
        opacities: torch.Tensor,
        image_width: int,
        image_height: int,
        image_origin_w: int,
        image_origin_h: int,
        tile_size: int,
        tile_offsets: torch.Tensor,
        tile_gaussian_ids: torch.Tensor,
        absgrad: bool,
        backgrounds: torch.Tensor | None,
        masks: torch.Tensor | None,
    ):
        rendered, rendered_alphas, last_ids = F.rasterize_screen_space_gaussians_fwd(
            means2d,
            conics,
            features,
            opacities,
            image_width,
            image_height,
            image_origin_w,
            image_origin_h,
            tile_size,
            tile_offsets,
            tile_gaussian_ids,
            -1,
            backgrounds,
            masks,
        )

        to_save = [means2d, conics, features, opacities, tile_offsets, tile_gaussian_ids, rendered_alphas, last_ids]
        _save_optional(ctx, to_save, backgrounds, masks)
        ctx.save_for_backward(*to_save)

        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.image_origin_w = image_origin_w
        ctx.image_origin_h = image_origin_h
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        return rendered, rendered_alphas

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_rendered, d_alphas = grad_outputs[0], grad_outputs[1]
        if d_rendered is not None:
            d_rendered = d_rendered.contiguous()
        if d_alphas is not None:
            d_alphas = d_alphas.contiguous()

        saved = ctx.saved_tensors
        means2d, conics, features, opacities, tile_offsets, tile_gaussian_ids, rendered_alphas, last_ids = saved[:8]
        backgrounds, masks = _load_optional(ctx, saved, 8)

        assert d_rendered is not None
        assert d_alphas is not None
        _, d_means2d, d_conics, d_features, d_opacities = F.rasterize_screen_space_gaussians_bwd(
            means2d,
            conics,
            features,
            opacities,
            ctx.image_width,
            ctx.image_height,
            ctx.image_origin_w,
            ctx.image_origin_h,
            ctx.tile_size,
            tile_offsets,
            tile_gaussian_ids,
            rendered_alphas,
            last_ids,
            d_rendered,
            d_alphas,
            ctx.absgrad,
            -1,
            backgrounds,
            masks,
        )

        return (d_means2d, d_conics, d_features, d_opacities) + (None,) * 10


# ---------------------------------------------------------------------------
#  Sparse screen-space rasterization
# ---------------------------------------------------------------------------


class _RasterizeScreenSpaceGaussiansSparseFn(torch.autograd.Function):
    """Alpha-blending at an arbitrary set of pixels with gradients to the 2D quantities.

    The forward pass takes and returns flat per-pixel tensors; the jagged structure of the pixel
    selection is saved so the backward pass can rebuild the JaggedTensors the kernel expects.
    """

    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        features: torch.Tensor,
        opacities: torch.Tensor,
        pixels_to_render: JaggedTensor,
        image_width: int,
        image_height: int,
        image_origin_w: int,
        image_origin_h: int,
        tile_size: int,
        tile_offsets: torch.Tensor,
        tile_gaussian_ids: torch.Tensor,
        active_tiles: torch.Tensor,
        tile_pixel_mask: torch.Tensor,
        tile_pixel_cumsum: torch.Tensor,
        pixel_map: torch.Tensor,
        absgrad: bool,
        backgrounds: torch.Tensor | None,
        masks: torch.Tensor | None,
    ):
        rendered_jt, alphas_jt, last_ids_jt = F.rasterize_screen_space_gaussians_sparse_fwd(
            pixels_to_render,
            means2d,
            conics,
            features,
            opacities,
            image_width,
            image_height,
            image_origin_w,
            image_origin_h,
            tile_size,
            tile_offsets,
            tile_gaussian_ids,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            -1,
            backgrounds,
            masks,
        )

        to_save = [
            means2d,
            conics,
            features,
            opacities,
            tile_offsets,
            tile_gaussian_ids,
            pixels_to_render.jdata,
            pixels_to_render.joffsets,
            pixels_to_render.jlidx,
            alphas_jt.jdata,
            last_ids_jt.jdata,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
        ]
        _save_optional(ctx, to_save, backgrounds, masks)
        ctx.save_for_backward(*to_save)

        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.image_origin_w = image_origin_w
        ctx.image_origin_h = image_origin_h
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        return rendered_jt.jdata, alphas_jt.jdata

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_rendered, d_alphas = grad_outputs[0], grad_outputs[1]
        if d_rendered is not None:
            d_rendered = d_rendered.contiguous()
        if d_alphas is not None:
            d_alphas = d_alphas.contiguous()

        saved = ctx.saved_tensors
        means2d, conics, features, opacities, tile_offsets, tile_gaussian_ids = saved[:6]
        pixels_jdata, joffsets, jlidx, alphas_jdata, last_ids_jdata = saved[6:11]
        active_tiles, tile_pixel_mask, tile_pixel_cumsum, pixel_map = saved[11:15]
        backgrounds, masks = _load_optional(ctx, saved, 15)

        pixels_jt = JaggedTensor.from_data_offsets_and_list_ids(pixels_jdata, joffsets, jlidx)
        assert d_rendered is not None
        assert d_alphas is not None
        _, d_means2d, d_conics, d_features, d_opacities = F.rasterize_screen_space_gaussians_sparse_bwd(
            pixels_jt,
            means2d,
            conics,
            features,
            opacities,
            ctx.image_width,
            ctx.image_height,
            ctx.image_origin_w,
            ctx.image_origin_h,
            ctx.tile_size,
            tile_offsets,
            tile_gaussian_ids,
            pixels_jt.jagged_like(alphas_jdata),
            pixels_jt.jagged_like(last_ids_jdata),
            pixels_jt.jagged_like(d_rendered),
            pixels_jt.jagged_like(d_alphas),
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            ctx.absgrad,
            -1,
            backgrounds,
            masks,
        )

        return (d_means2d, d_conics, d_features, d_opacities) + (None,) * 15


# ---------------------------------------------------------------------------
#  World-space rasterization
# ---------------------------------------------------------------------------


class _RasterizeWorldSpaceGaussiansFn(torch.autograd.Function):
    """Ray-based rasterization of 3D Gaussians with gradients to the 3D parameters."""

    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        features: torch.Tensor,
        opacities: torch.Tensor,
        world_to_cam_start: torch.Tensor,
        world_to_cam_end: torch.Tensor,
        projection_matrices: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        rolling_shutter_type: int,
        camera_model: int,
        image_width: int,
        image_height: int,
        image_origin_w: int,
        image_origin_h: int,
        tile_size: int,
        tile_offsets: torch.Tensor,
        tile_gaussian_ids: torch.Tensor,
        backgrounds: torch.Tensor | None,
        masks: torch.Tensor | None,
    ):
        rendered, rendered_alphas, last_ids = F.rasterize_world_space_gaussians_fwd(
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_cam_start,
            world_to_cam_end,
            projection_matrices,
            distortion_coeffs,
            rolling_shutter_type,
            camera_model,
            image_width,
            image_height,
            image_origin_w,
            image_origin_h,
            tile_size,
            tile_offsets,
            tile_gaussian_ids,
            backgrounds,
            masks,
        )

        to_save = [
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_cam_start,
            world_to_cam_end,
            projection_matrices,
            distortion_coeffs,
            tile_offsets,
            tile_gaussian_ids,
            rendered_alphas,
            last_ids,
        ]
        _save_optional(ctx, to_save, backgrounds, masks)
        ctx.save_for_backward(*to_save)

        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.image_origin_w = image_origin_w
        ctx.image_origin_h = image_origin_h
        ctx.tile_size = tile_size
        ctx.rolling_shutter_type = rolling_shutter_type
        ctx.camera_model = camera_model

        return rendered, rendered_alphas

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_rendered, d_alphas = grad_outputs[0], grad_outputs[1]
        if d_rendered is not None:
            d_rendered = d_rendered.contiguous()
        if d_alphas is not None:
            d_alphas = d_alphas.contiguous()

        saved = ctx.saved_tensors
        means, quats, log_scales, features, opacities = saved[:5]
        world_to_cam_start, world_to_cam_end, projection_matrices, distortion_coeffs = saved[5:9]
        tile_offsets, tile_gaussian_ids, rendered_alphas, last_ids = saved[9:13]
        backgrounds, masks = _load_optional(ctx, saved, 13)

        assert d_rendered is not None
        assert d_alphas is not None
        d_means, d_quats, d_log_scales, d_features, d_opacities = F.rasterize_world_space_gaussians_bwd(
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_cam_start,
            world_to_cam_end,
            projection_matrices,
            distortion_coeffs,
            ctx.rolling_shutter_type,
            ctx.camera_model,
            ctx.image_width,
            ctx.image_height,
            ctx.image_origin_w,
            ctx.image_origin_h,
            ctx.tile_size,
            tile_offsets,
            tile_gaussian_ids,
            rendered_alphas,
            last_ids,
            d_rendered,
            d_alphas,
            backgrounds,
            masks,
        )

        return (d_means, d_quats, d_log_scales, d_features, d_opacities) + (None,) * 15
