# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
``fvdb_reality_capture.functional`` -- composable, differentiable Gaussian splatting.

Rendering is split into four stages that pass small frozen dataclasses between them, so a custom
pipeline can insert its own logic anywhere:

1. :func:`project_gaussians` projects the 3D Gaussians into every camera.
2. :func:`evaluate_gaussian_sh` turns spherical-harmonics coefficients into per-camera features.
3. :func:`intersect_gaussian_tiles` or :func:`intersect_gaussian_tiles_sparse` bins Gaussians into tiles.
4. :func:`rasterize_screen_space_gaussians`, :func:`rasterize_world_space_gaussians` or
   :func:`rasterize_screen_space_gaussians_sparse` alpha-blends the features.

Stages 3 and 4 take per-camera opacities, ``[C, N]``. Compute them once per render with
:func:`compute_gaussian_opacities` and pass the same tensor to every stage.

Every stage except tile intersection is differentiable. The kernels themselves live in
:mod:`fvdb.functional`; :class:`~fvdb_reality_capture.GaussianSplat3d` composes these stages.
"""

from ._analysis import (
    rasterize_contributing_gaussian_ids,
    rasterize_contributing_gaussian_ids_sparse,
    rasterize_num_contributing_gaussians,
    rasterize_num_contributing_gaussians_sparse,
)
from ._opacity import compute_gaussian_opacities
from ._projection import (
    check_distortion_coeffs,
    project_gaussians,
    requires_distortion_coeffs,
    resolve_projection_method,
)
from ._rasterization import (
    Crop,
    apply_crop,
    apply_pixel_mask,
    pad_crop,
    pixel_mask_to_tile_mask,
    rasterize_screen_space_gaussians,
    rasterize_screen_space_gaussians_sparse,
    rasterize_world_space_gaussians,
    validate_crop,
)
from ._spherical_harmonics import evaluate_gaussian_sh, sh_degree_from_coefficients
from ._tile_intersection import (
    as_pixel_jagged,
    check_tiles_match,
    deduplicate_pixels,
    intersect_gaussian_tiles,
    intersect_gaussian_tiles_sparse,
)
from ._types import GaussianTileIntersection, ProjectedGaussians, SparseGaussianTileIntersection

__all__ = [
    # Types
    "ProjectedGaussians",
    "GaussianTileIntersection",
    "SparseGaussianTileIntersection",
    # Stage 1: projection
    "project_gaussians",
    "resolve_projection_method",
    "check_distortion_coeffs",
    "requires_distortion_coeffs",
    # Stage 2: features
    "evaluate_gaussian_sh",
    "sh_degree_from_coefficients",
    # Stage 3: tile intersection
    "intersect_gaussian_tiles",
    "intersect_gaussian_tiles_sparse",
    "deduplicate_pixels",
    "as_pixel_jagged",
    "check_tiles_match",
    # Stage 4: rasterization
    "rasterize_screen_space_gaussians",
    "rasterize_world_space_gaussians",
    "rasterize_screen_space_gaussians_sparse",
    "compute_gaussian_opacities",
    "pixel_mask_to_tile_mask",
    "Crop",
    "validate_crop",
    "apply_crop",
    "apply_pixel_mask",
    "pad_crop",
    # Analysis
    "rasterize_num_contributing_gaussians",
    "rasterize_num_contributing_gaussians_sparse",
    "rasterize_contributing_gaussian_ids",
    "rasterize_contributing_gaussian_ids_sparse",
]
