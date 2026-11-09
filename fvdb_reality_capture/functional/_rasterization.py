# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Stage 4 of the composable pipeline: alpha-blend features into images or pixel sets."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as nnf
from fvdb import JaggedTensor

from ..enums import RollingShutterType
from ._autograd import (
    _RasterizeScreenSpaceGaussiansFn,
    _RasterizeScreenSpaceGaussiansSparseFn,
    _RasterizeWorldSpaceGaussiansFn,
)
from ._opacity import check_opacities
from ._projection import check_distortion_coeffs
from ._tile_intersection import check_tiles_match
from ._types import GaussianTileIntersection, ProjectedGaussians, SparseGaussianTileIntersection

Crop = tuple[int, int, int, int]
"""A crop as ``(origin_w, origin_h, width, height)`` in pixels."""


def validate_crop(crop: Crop, image_width: int, image_height: int) -> Crop:
    """Check a crop against an image and clip it to the image bounds.

    A crop that runs past the image edge is clipped; one that lies entirely outside the image clips to a
    zero-size crop, so rendering it yields an empty ``[C, 0, 0, D]`` result rather than an error, and
    :func:`pad_crop` can grow that back to the requested size.

    Args:
        crop (tuple[int, int, int, int]): ``(origin_w, origin_h, width, height)`` in pixels.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.

    Returns:
        crop (tuple[int, int, int, int]): The crop with its size clipped so it lies inside the image.

    Raises:
        ValueError: If the origin is negative or the size is not positive.
    """
    origin_w, origin_h, width, height = crop
    if origin_w < 0 or origin_h < 0:
        raise ValueError(f"Crop origin must be non-negative, got ({origin_w}, {origin_h})")
    if width <= 0 or height <= 0:
        raise ValueError(f"Crop size must be positive, got ({width}, {height})")
    width = max(0, min(width, image_width - origin_w))
    height = max(0, min(height, image_height - origin_h))
    if width == 0 or height == 0:
        return origin_w, origin_h, 0, 0
    return origin_w, origin_h, width, height


def _empty_render(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``[C, 0, 0, D]`` render of a crop clipped to nothing, as a slice of ``features`` so it stays connected."""
    empty = features[:, :0].reshape(features.shape[0], 0, 0, features.shape[-1])
    return empty, empty[..., :1]


def _finish_dense_render(
    images: torch.Tensor,
    alphas: torch.Tensor,
    crop: Crop | None,
    masks: torch.Tensor | None,
    backgrounds: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice a full-size render to its crop, then apply the per-pixel mask, already in crop coordinates."""
    images, alphas = apply_crop(images, alphas, crop)
    if masks is not None:
        images, alphas = apply_pixel_mask(images, alphas, masks, backgrounds)
    return images, alphas


def _render_masks(
    crop: Crop | None,
    masks: torch.Tensor | None,
    crop_masks: torch.Tensor | None,
    projected: ProjectedGaussians,
    tiles: GaussianTileIntersection,
    device: torch.device,
) -> tuple[Crop | None, torch.Tensor | None, torch.Tensor | None]:
    """Resolve the crop and masks into what the rasterizer and the post-pass need.

    Checks that ``tiles`` belong to ``projected`` first. Returns the clipped crop, the per-pixel mask in
    the output's coordinates to apply after rendering and slicing (``None`` when no pixel mask was given,
    since slicing to the crop already discards out-of-crop pixels), and the per-tile mask that lets the
    rasterizer skip tiles outside the crop or fully masked out. ``masks`` is always in image coordinates
    and ``crop_masks`` always in crop coordinates (the crop as requested or as clipped to the image), so
    a shape never has to be guessed; the tile mask is pooled from whichever was given, without building a
    full-image mask. The tile grid comes from ``tiles`` so it cannot drift from the intersection's.
    """
    check_tiles_match(tiles, projected)
    num_cameras = projected.num_cameras
    tile_size = tiles.tile_size
    full_shape = (num_cameras, tiles.image_height, tiles.image_width)
    if masks is not None and crop_masks is not None:
        raise ValueError("pass either masks (image coordinates) or crop_masks (crop coordinates), not both")
    if crop_masks is not None and crop is None:
        raise ValueError("crop_masks needs a crop")
    for name, tensor in (("masks", masks), ("crop_masks", crop_masks)):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if masks is not None and tuple(masks.shape) != full_shape:
        raise ValueError(f"masks must be a full-image [C, H, W] mask of shape {full_shape}, got {tuple(masks.shape)}")
    if crop is None:
        if masks is None:
            return None, None, None
        masks = masks.bool()
        return None, masks, pixel_mask_to_tile_mask(masks, tile_size)
    requested_shape = (num_cameras, crop[3], crop[2])
    origin_w, origin_h, width, height = clipped = validate_crop(crop, tiles.image_width, tiles.image_height)
    tile_y0, tile_x0 = origin_h // tile_size, origin_w // tile_size
    tile_y1, tile_x1 = -(-(origin_h + height) // tile_size), -(-(origin_w + width) // tile_size)
    tile_window = torch.zeros(num_cameras, tiles.num_tiles_h, tiles.num_tiles_w, dtype=torch.bool, device=device)
    tile_window[:, tile_y0:tile_y1, tile_x0:tile_x1] = True
    if masks is None and crop_masks is None:
        return clipped, None, tile_window
    crop_shape = (num_cameras, height, width)
    if masks is not None:
        crop_mask = _window(masks.bool(), clipped)
    elif tuple(crop_masks.shape) in (crop_shape, requested_shape):
        # A mask of the requested crop size covers the clipped part in its top-left corner.
        crop_mask = crop_masks[:, :height, :width].bool()
    else:
        raise ValueError(
            f"crop_masks must match the crop {requested_shape} or its clipped size {crop_shape}, got {tuple(crop_masks.shape)}"
        )
    # Pool the crop mask over the crop's tiles. The crop need not start on a tile boundary, so the mask is
    # placed at its offset within the first tile before pooling.
    aligned = torch.zeros(
        num_cameras, origin_h % tile_size + height, origin_w % tile_size + width, dtype=torch.bool, device=device
    )
    aligned[:, origin_h % tile_size :, origin_w % tile_size :] = crop_mask
    tile_masks = torch.zeros_like(tile_window)
    if height > 0 and width > 0:
        tile_masks[:, tile_y0:tile_y1, tile_x0:tile_x1] = pixel_mask_to_tile_mask(aligned, tile_size)
    return clipped, crop_mask, tile_masks & tile_window


def apply_crop(images: torch.Tensor, alphas: torch.Tensor, crop: Crop | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice rendered images and alphas to a crop window.

    Args:
        images (torch.Tensor): Rendered images, ``[C, H, W, D]``.
        alphas (torch.Tensor): Rendered alphas, ``[C, H, W, 1]``.
        crop (tuple[int, int, int, int] | None): ``(origin_w, origin_h, width, height)`` window, already
            validated against the image; ``None`` returns the inputs unchanged.

    Returns:
        images (torch.Tensor): The window of ``images``, ``[C, height, width, D]``.
        alphas (torch.Tensor): The window of ``alphas``, ``[C, height, width, 1]``.
    """
    if crop is None:
        return images, alphas
    return _window(images, crop), _window(alphas, crop)


def _window(tensor: torch.Tensor, crop: Crop) -> torch.Tensor:
    origin_w, origin_h, width, height = crop
    return tensor[:, origin_h : origin_h + height, origin_w : origin_w + width]


def _background_like(images: torch.Tensor, backgrounds: torch.Tensor | None) -> torch.Tensor:
    """Per-camera background features as ``[C, 1, 1, D]``, broadcastable against ``images``; black if ``None``."""
    if backgrounds is None:
        return torch.zeros(images.shape[0], 1, 1, images.shape[-1], device=images.device, dtype=images.dtype)
    return backgrounds.to(images)[:, None, None, :]


def pad_crop(
    images: torch.Tensor,
    alphas: torch.Tensor,
    height: int,
    width: int,
    backgrounds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extend a rendered crop to ``height`` by ``width``, filling the added area with the background at zero alpha.

    This is how a crop that runs past the image edge keeps the size that was asked for: the part inside
    the image is rendered, the rest is background. Inputs already of the target size are returned as is.

    Args:
        images (torch.Tensor): Rendered crop, ``[C, h, w, D]`` with ``h <= height`` and ``w <= width``.
        alphas (torch.Tensor): Its alphas, ``[C, h, w, 1]``.
        height (int): Target height in pixels.
        width (int): Target width in pixels.
        backgrounds (torch.Tensor | None): Per-camera background features, ``[C, D]``. Black if ``None``.

    Returns:
        images (torch.Tensor): ``[C, height, width, D]`` with the input in its top-left corner.
        alphas (torch.Tensor): ``[C, height, width, 1]``, zero outside the input.

    Raises:
        ValueError: If the input is larger than the target size in either dimension.
    """
    num_cameras, current_h, current_w, channels = images.shape
    if (current_h, current_w) == (height, width):
        return images, alphas
    if current_h > height or current_w > width:
        raise ValueError(f"pad_crop cannot shrink a {(current_h, current_w)} render to {(height, width)}")
    padded = _background_like(images, backgrounds).expand(num_cameras, height, width, channels).clone()
    padded[:, :current_h, :current_w] = images
    padded_alphas = alphas.new_zeros(num_cameras, height, width, 1)
    padded_alphas[:, :current_h, :current_w] = alphas
    return padded, padded_alphas


def pixel_mask_to_tile_mask(pixel_mask: torch.Tensor, tile_size: int) -> torch.Tensor:
    """Mark a tile as rendered if any of its pixels is.

    Args:
        pixel_mask (torch.Tensor): Boolean per-pixel mask, ``[C, H, W]``.
        tile_size (int): Tile side length in pixels.

    Returns:
        tile_mask (torch.Tensor): Boolean per-tile mask, ``[C, ceil(H / tile_size), ceil(W / tile_size)]``.
    """
    pooled = nnf.max_pool2d(
        pixel_mask.bool().unsqueeze(1).float(), kernel_size=tile_size, stride=tile_size, ceil_mode=True
    )
    return pooled.bool().squeeze(1)


def apply_pixel_mask(
    images: torch.Tensor, alphas: torch.Tensor, pixel_mask: torch.Tensor, backgrounds: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill masked-out pixels of a render with the background at zero alpha.

    The rasterizers skip whole tiles; this is the per-pixel pass that follows, and it works on any
    render, so a crop can be masked after slicing without building a full-image mask.

    Args:
        images (torch.Tensor): Rendered features, ``[C, H, W, D]``.
        alphas (torch.Tensor): Rendered alphas, ``[C, H, W, 1]``.
        pixel_mask (torch.Tensor): Boolean mask, ``[C, H, W]``; ``True`` keeps the rendered pixel.
        backgrounds (torch.Tensor | None): Per-camera background features, ``[C, D]``. Black if ``None``.

    Returns:
        images (torch.Tensor): ``[C, H, W, D]`` with masked-out pixels set to the background.
        alphas (torch.Tensor): ``[C, H, W, 1]`` with masked-out pixels set to zero.
    """
    keep = pixel_mask.unsqueeze(-1).to(images.dtype)
    background = _background_like(images, backgrounds)
    return images * keep + background * (1.0 - keep), alphas * keep


def rasterize_screen_space_gaussians(
    projected: ProjectedGaussians,
    features: torch.Tensor,
    opacities: torch.Tensor,
    tiles: GaussianTileIntersection,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
    crop: Crop | None = None,
    crop_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alpha-blend projected Gaussians into dense images.

    Differentiable with respect to ``features``, ``opacities`` and, for an ``ANALYTIC`` projection,
    the projection itself. The unscented projection is forward-only, so through this function the 3D
    parameters receive no gradient; :func:`rasterize_world_space_gaussians` is the training path for
    it. A ``crop`` selects a window of the images: tiles outside it are skipped and the result is
    exactly the corresponding region of the uncropped render. The output buffers are still allocated
    at full image size before slicing.

    Args:
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        features (torch.Tensor): Per-camera, per-Gaussian features, ``[C, N, D]``.
        opacities (torch.Tensor): Per-camera opacities, ``[C, N]``, from :func:`compute_gaussian_opacities`.
            Compute them once per render and pass the same tensor to every stage.
        tiles (GaussianTileIntersection): Output of :func:`intersect_gaussian_tiles` for ``projected``.
        backgrounds (torch.Tensor | None): Per-camera background features, ``[C, D]``. Black if ``None``.
        masks (torch.Tensor | None): Boolean per-pixel render mask in image coordinates, ``[C, H, W]``.
            Masked-out pixels receive the background with zero alpha and no gradient.
        crop (tuple[int, int, int, int] | None): ``(origin_w, origin_h, width, height)`` window to keep,
            clipped to the image; a crop entirely outside it yields an empty ``[C, 0, 0, D]`` render, which
            :meth:`~fvdb_reality_capture.GaussianSplat3d.render_from_projected_gaussians` pads back to the requested size.
        crop_masks (torch.Tensor | None): Boolean per-pixel render mask in crop coordinates, of the crop's
            requested ``[C, height, width]`` or clipped size; an alternative to ``masks`` when a crop is given.

    Returns:
        images (torch.Tensor): Blended features, ``[C, H, W, D]`` (or the crop size).
        alphas (torch.Tensor): Accumulated alpha in ``[0, 1)``, ``[C, H, W, 1]`` (or the crop size).
    """
    opacities = check_opacities(opacities, projected)
    crop, masks, tile_masks = _render_masks(crop, masks, crop_masks, projected, tiles, opacities.device)
    if crop is not None and (crop[2] == 0 or crop[3] == 0):
        return _empty_render(features)
    images, alphas = cast(
        tuple[torch.Tensor, torch.Tensor],
        _RasterizeScreenSpaceGaussiansFn.apply(
            projected.means2d,
            projected.conics,
            features,
            opacities,
            tiles.image_width,
            tiles.image_height,
            0,
            0,
            tiles.tile_size,
            tiles.tile_offsets,
            tiles.tile_gaussian_ids,
            False,
            backgrounds,
            tile_masks,
        ),
    )
    return _finish_dense_render(images, alphas, crop, masks, backgrounds)


def rasterize_world_space_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    projected: ProjectedGaussians,
    features: torch.Tensor,
    opacities: torch.Tensor,
    world_to_camera_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    tiles: GaussianTileIntersection,
    distortion_coeffs: torch.Tensor | None = None,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
    crop: Crop | None = None,
    crop_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alpha-blend 3D Gaussians into dense images by evaluating them along per-pixel rays.

    Differentiable with respect to the 3D parameters, ``features`` and ``opacities``, which makes it
    the training path for the unscented projection. The projection supplies the tile intersections
    and the camera model; the 3D parameters are evaluated directly.

    Args:
        means (torch.Tensor): Gaussian centers in world space, ``[N, 3]``.
        quats (torch.Tensor): Gaussian rotations as quaternions, ``[N, 4]``.
        log_scales (torch.Tensor): Natural-log scale factors, ``[N, 3]``.
        projected (ProjectedGaussians): Output of :func:`project_gaussians` for these Gaussians and cameras.
        features (torch.Tensor): Per-camera, per-Gaussian features, ``[C, N, D]``.
        opacities (torch.Tensor): Per-camera opacities, ``[C, N]``, from :func:`compute_gaussian_opacities`.
        world_to_camera_matrices (torch.Tensor): World-to-camera transforms, ``[C, 4, 4]``.
        projection_matrices (torch.Tensor): Camera intrinsics, ``[C, 3, 3]``.
        tiles (GaussianTileIntersection): Output of :func:`intersect_gaussian_tiles` for ``projected``.
        distortion_coeffs (torch.Tensor | None): Packed OpenCV distortion coefficients, ``[C, 12]``.
            Required for the OpenCV camera models; ``None`` is allowed for pinhole and orthographic cameras.
        backgrounds (torch.Tensor | None): Per-camera background features, ``[C, D]``. Black if ``None``.
        masks (torch.Tensor | None): Boolean per-pixel render mask in image coordinates, ``[C, H, W]``.
        crop (tuple[int, int, int, int] | None): ``(origin_w, origin_h, width, height)`` window to keep,
            clipped to the image; a crop entirely outside it yields an empty ``[C, 0, 0, D]`` render, which
            :meth:`~fvdb_reality_capture.GaussianSplat3d.render_from_projected_gaussians` pads back to the requested size.
        crop_masks (torch.Tensor | None): Boolean per-pixel render mask in crop coordinates, of the crop's
            requested ``[C, height, width]`` or clipped size; an alternative to ``masks`` when a crop is given.

    Returns:
        images (torch.Tensor): Blended features, ``[C, H, W, D]`` (or the crop size).
        alphas (torch.Tensor): Accumulated alpha in ``[0, 1)``, ``[C, H, W, 1]`` (or the crop size).
    """
    opacities = check_opacities(opacities, projected)
    distortion_coeffs = check_distortion_coeffs(
        distortion_coeffs, projected.camera_model, projected.num_cameras, opacities.device
    )
    if distortion_coeffs is None:
        distortion_coeffs = torch.zeros(
            projected.num_cameras, 12, device=world_to_camera_matrices.device, dtype=world_to_camera_matrices.dtype
        )
    crop, masks, tile_masks = _render_masks(crop, masks, crop_masks, projected, tiles, opacities.device)
    if crop is not None and (crop[2] == 0 or crop[3] == 0):
        return _empty_render(features)
    images, alphas = cast(
        tuple[torch.Tensor, torch.Tensor],
        _RasterizeWorldSpaceGaussiansFn.apply(
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_camera_matrices,
            world_to_camera_matrices,
            projection_matrices,
            distortion_coeffs,
            int(RollingShutterType.NONE),
            int(projected.camera_model),
            tiles.image_width,
            tiles.image_height,
            0,
            0,
            tiles.tile_size,
            tiles.tile_offsets,
            tiles.tile_gaussian_ids,
            backgrounds,
            tile_masks,
        ),
    )
    return _finish_dense_render(images, alphas, crop, masks, backgrounds)


def rasterize_screen_space_gaussians_sparse(
    projected: ProjectedGaussians,
    features: torch.Tensor,
    opacities: torch.Tensor,
    sparse_tiles: SparseGaussianTileIntersection,
    backgrounds: torch.Tensor | None = None,
    tile_masks: torch.Tensor | None = None,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Alpha-blend projected Gaussians at the requested pixels only.

    Differentiable with respect to ``features``, ``opacities`` and, for an ``ANALYTIC`` projection, the
    projection itself. Results are returned in the order of ``sparse_tiles.pixels_to_render``,
    duplicates included.

    Args:
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        features (torch.Tensor): Per-camera, per-Gaussian features, ``[C, N, D]``.
        opacities (torch.Tensor): Per-camera opacities, ``[C, N]``, from :func:`compute_gaussian_opacities`.
        sparse_tiles (SparseGaussianTileIntersection): Output of :func:`intersect_gaussian_tiles_sparse`.
        backgrounds (torch.Tensor | None): Per-camera background features, ``[C, D]``. Black if ``None``.
        tile_masks (torch.Tensor | None): Boolean per-tile render mask, ``[C, num_tiles_h, num_tiles_w]``.
            Unlike the dense rasterizers this is per tile, since the pixels to render are explicit;
            use :func:`pixel_mask_to_tile_mask` to derive it from a per-pixel mask.

    Returns:
        features (JaggedTensor): Blended features per requested pixel, one ``[P_c, D]`` list per camera.
        alphas (JaggedTensor): Accumulated alpha per requested pixel, one ``[P_c, 1]`` list per camera.
    """
    opacities = check_opacities(opacities, projected)
    check_tiles_match(sparse_tiles, projected)
    if tile_masks is not None:
        expected = tuple(sparse_tiles.active_tile_mask.shape)
        if tuple(tile_masks.shape) != expected:
            raise ValueError(
                f"tile_masks must be a per-tile [C, tiles_h, tiles_w] mask of shape {expected}, got {tuple(tile_masks.shape)}"
            )
        if tile_masks.device != opacities.device:
            raise ValueError(f"tile_masks must be on {opacities.device}, got {tile_masks.device}")
    rendered, alphas = cast(
        tuple[torch.Tensor, torch.Tensor],
        _RasterizeScreenSpaceGaussiansSparseFn.apply(
            projected.means2d,
            projected.conics,
            features,
            opacities,
            sparse_tiles.unique_pixels,
            sparse_tiles.image_width,
            sparse_tiles.image_height,
            0,
            0,
            sparse_tiles.tile_size,
            sparse_tiles.tile_offsets,
            sparse_tiles.tile_gaussian_ids,
            sparse_tiles.active_tiles,
            sparse_tiles.tile_pixel_mask,
            sparse_tiles.tile_pixel_cumsum,
            sparse_tiles.pixel_map,
            False,
            backgrounds,
            None if tile_masks is None else tile_masks.bool(),
        ),
    )
    requested = sparse_tiles.pixels_to_render
    return (
        requested.jagged_like(sparse_tiles.expand_to_requested(rendered)),
        requested.jagged_like(sparse_tiles.expand_to_requested(alphas)),
    )
