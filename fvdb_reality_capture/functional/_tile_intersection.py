# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Stage 3 of the composable pipeline: bin projected Gaussians into image tiles."""

from __future__ import annotations

import math

import torch
from fvdb import JaggedTensor
from fvdb import functional as F
from fvdb.functional import as_pixel_jagged

from ._opacity import check_opacities
from ._types import GaussianTileIntersection, ProjectedGaussians, SparseGaussianTileIntersection


def _tile_grid(image_width: int, image_height: int, tile_size: int) -> tuple[int, int]:
    return math.ceil(image_height / tile_size), math.ceil(image_width / tile_size)


def check_tiles_match(
    tiles: GaussianTileIntersection | SparseGaussianTileIntersection, projected: ProjectedGaussians
) -> None:
    """Raise if ``tiles`` were not intersected for ``projected``.

    Tile intersections index the kernels by camera, image tile and Gaussian. Stale ones, from a projection
    of a different camera batch or image size, or of the same cameras before an optimizer step or a
    refinement moved, added or removed Gaussians, would read out of range or blend the wrong Gaussians
    rather than raise, so the stages check this up front. Every projection carries a token that its tile
    intersections record; ``dataclasses.replace`` on the projection keeps the token, so a projection
    rebuilt with ``replace`` (for example with some fields detached) still matches its tiles.

    Args:
        tiles (GaussianTileIntersection | SparseGaussianTileIntersection): The tile intersection to check.
        projected (ProjectedGaussians): The projection the tiles are about to be rasterized with.
    """
    if (tiles.image_width, tiles.image_height) != (projected.image_width, projected.image_height):
        raise ValueError(
            f"tiles were intersected for a {tiles.image_width}x{tiles.image_height} image but the projection is "
            f"{projected.image_width}x{projected.image_height}"
        )
    if isinstance(tiles, GaussianTileIntersection):
        num_cameras = tiles.tile_offsets.shape[0]
    else:
        num_cameras = tiles.pixels_to_render.num_tensors
    if num_cameras != projected.num_cameras:
        raise ValueError(f"tiles cover {num_cameras} cameras but the projection has {projected.num_cameras}")
    if tiles.projection_token is not projected.token:
        raise ValueError(
            "tiles were intersected for a different projection; recompute them after the Gaussians or cameras change"
        )


def _culling_inputs(
    projected: ProjectedGaussians, opacities: torch.Tensor | None
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Conics and opacities for the tighter iso-contour tile test, or ``None`` for the bounding-box test."""
    if opacities is None:
        return None, None
    return projected.conics, check_opacities(opacities, projected).detach()


def intersect_gaussian_tiles(
    projected: ProjectedGaussians,
    opacities: torch.Tensor | None = None,
    tile_size: int = 16,
) -> GaussianTileIntersection:
    """Bin projected Gaussians into the tiles of the full image, sorted by camera, tile and depth.

    Not differentiable. Passing ``opacities`` enables a tighter per-tile culling test based on each
    Gaussian's iso-contour at the opacity threshold instead of its bounding box, which reduces the
    work of every later stage.

    Args:
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        opacities (torch.Tensor | None): Per-camera opacities, ``[C, N]``, from
            :func:`compute_gaussian_opacities`, for tighter culling. ``None`` culls by bounding box.
        tile_size (int): Tile side length in pixels.

    Returns:
        tiles (GaussianTileIntersection): The tile intersections.
    """
    num_tiles_h, num_tiles_w = _tile_grid(projected.image_width, projected.image_height, tile_size)
    conics, opacities = _culling_inputs(projected, opacities)
    tile_offsets, tile_gaussian_ids = F.intersect_gaussian_tiles(
        projected.means2d,
        projected.radii,
        projected.depths,
        projected.num_cameras,
        tile_size,
        num_tiles_h,
        num_tiles_w,
        conics=conics,
        opacities=opacities,
    )
    return GaussianTileIntersection(
        tile_offsets=tile_offsets,
        tile_gaussian_ids=tile_gaussian_ids,
        tile_size=tile_size,
        image_width=projected.image_width,
        image_height=projected.image_height,
        projection_token=projected.token,
    )


def deduplicate_pixels(
    pixels_to_render: JaggedTensor, image_width: int, image_height: int
) -> tuple[JaggedTensor, torch.Tensor, bool]:
    """Remove pixels that appear more than once within a camera.

    Each pixel's first occurrence is kept, the unique pixels are in the order they were first requested,
    and pixels outside the image are never merged with anything.

    Args:
        pixels_to_render (JaggedTensor): ``(row, col)`` integer pixels, one list per camera.
        image_width (int): Image width, used to linearize pixel coordinates.
        image_height (int): Image height, used to linearize pixel coordinates.

    Returns:
        unique_pixels (JaggedTensor): The pixels with duplicates removed.
        inverse_indices (torch.Tensor): For each flat requested pixel, its index into the flat unique pixels.
            Empty when there are no duplicates, since nothing needs reordering then.
        has_duplicates (bool): Whether anything was removed. When ``False`` the input is returned as is.
    """
    jdata = pixels_to_render.jdata
    total_pixels = jdata.shape[0]
    device = jdata.device
    if total_pixels == 0:
        return pixels_to_render, torch.empty(0, dtype=torch.long, device=device), False

    jidx = pixels_to_render.jidx
    single_list = jidx.shape[0] == 0
    num_lists = pixels_to_render.num_tensors
    rows = jdata[:, 0].long()
    cols = jdata[:, 1].long()
    keys = rows * image_width + cols
    if not single_list:
        keys = keys + jidx.long() * (image_height * image_width)
    # Linearizing (row, col) aliases pixels outside the image onto valid ones, e.g. (0, W) onto (1, 0).
    # Give each such pixel a key of its own so it is never merged into a valid pixel's output and the
    # layout kernel's bounds check still sees it.
    in_image = (rows >= 0) & (rows < image_height) & (cols >= 0) & (cols < image_width)
    out_of_image_keys = num_lists * image_height * image_width + torch.arange(total_pixels, device=device)
    keys = torch.where(in_image, keys, out_of_image_keys)

    # A stable sort puts each pixel's first occurrence first within its group of duplicates.
    sorted_keys, sort_perm = keys.sort(stable=True)
    is_group_start = torch.ones(total_pixels, dtype=torch.bool, device=device)
    if total_pixels > 1:
        is_group_start[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_ids = is_group_start.long().cumsum(0) - 1
    num_unique = int(group_ids[-1].item()) + 1
    if num_unique == total_pixels:
        return pixels_to_render, torch.empty(0, dtype=torch.long, device=device), False

    # Order the unique pixels by first request rather than by sorted key, and remap the groups to match.
    unique_orig_indices, order = sort_perm[is_group_start].sort()
    rank = torch.empty(num_unique, dtype=torch.long, device=device)
    rank[order] = torch.arange(num_unique, device=device)
    inverse_indices = torch.empty(total_pixels, dtype=torch.long, device=device)
    inverse_indices[sort_perm] = rank[group_ids]
    unique_jdata = jdata[unique_orig_indices]

    if single_list:
        unique_batch_idx = torch.zeros(num_unique, dtype=torch.long, device=device)
    else:
        unique_batch_idx = jidx.long()[unique_orig_indices]
    counts_per_list = torch.bincount(unique_batch_idx, minlength=num_lists)
    new_offsets = torch.zeros(num_lists + 1, dtype=torch.long, device=device)
    new_offsets[1:] = counts_per_list.cumsum(0)
    unique_pixels = JaggedTensor.from_data_and_offsets(unique_jdata, new_offsets)
    return unique_pixels, inverse_indices, True


def intersect_gaussian_tiles_sparse(
    pixels_to_render: JaggedTensor | torch.Tensor,
    projected: ProjectedGaussians,
    opacities: torch.Tensor | None = None,
    tile_size: int = 16,
) -> SparseGaussianTileIntersection:
    """Bin projected Gaussians into only the tiles that contain requested pixels.

    Not differentiable. Requested pixels are deduplicated per camera before the layout is built;
    the result records how to expand per-pixel outputs back to the requested order.

    Args:
        pixels_to_render (JaggedTensor | torch.Tensor): ``(row, col)`` integer pixels, one list per
            camera, or a ``[C, P, 2]`` tensor.
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        opacities (torch.Tensor | None): Per-camera opacities, ``[C, N]``, from
            :func:`compute_gaussian_opacities`, for tighter culling. ``None`` culls by bounding box.
        tile_size (int): Tile side length in pixels. The sparse kernels require ``16``.

    Returns:
        sparse_tiles (SparseGaussianTileIntersection): The sparse tile intersections.
    """
    pixels = as_pixel_jagged(pixels_to_render)
    num_tiles_h, num_tiles_w = _tile_grid(projected.image_width, projected.image_height, tile_size)
    unique_pixels, inverse_indices, has_duplicates = deduplicate_pixels(
        pixels, projected.image_width, projected.image_height
    )
    active_tiles, active_tile_mask, tile_pixel_mask, tile_pixel_cumsum, pixel_map = F.build_sparse_gaussian_tile_layout(
        tile_size,
        num_tiles_h,
        num_tiles_w,
        unique_pixels,
        image_width=projected.image_width,
        image_height=projected.image_height,
    )
    conics, opacities = _culling_inputs(projected, opacities)
    tile_offsets, tile_gaussian_ids = F.intersect_gaussian_tiles_sparse(
        projected.means2d,
        projected.radii,
        projected.depths,
        active_tile_mask,
        active_tiles,
        projected.num_cameras,
        tile_size,
        num_tiles_h,
        num_tiles_w,
        conics=conics,
        opacities=opacities,
    )
    return SparseGaussianTileIntersection(
        tile_offsets=tile_offsets,
        tile_gaussian_ids=tile_gaussian_ids,
        pixels_to_render=pixels,
        unique_pixels=unique_pixels,
        inverse_indices=inverse_indices,
        has_duplicates=has_duplicates,
        active_tiles=active_tiles,
        active_tile_mask=active_tile_mask,
        tile_pixel_mask=tile_pixel_mask,
        tile_pixel_cumsum=tile_pixel_cumsum,
        pixel_map=pixel_map,
        tile_size=tile_size,
        image_width=projected.image_width,
        image_height=projected.image_height,
        projection_token=projected.token,
    )
