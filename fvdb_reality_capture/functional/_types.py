# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Frozen dataclasses passed between the stages of the composable Gaussian splatting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from fvdb import JaggedTensor

from ..enums import CameraModel, ProjectionMethod


@dataclass(frozen=True, eq=False)
class ProjectedGaussians:
    """
    Output of :func:`project_gaussians`: the 2D footprint of every Gaussian in every camera.

    ``C`` is the number of cameras and ``N`` the number of Gaussians. Gaussians that were culled
    (outside the near/far planes or too small) have both radii set to zero; their other entries are
    undefined and are ignored by every downstream stage.
    """

    radii: torch.Tensor
    """Per-axis projected radii in pixels, ``[C, N, 2]``, ``int32``."""

    means2d: torch.Tensor
    """Projected centers in pixel coordinates, ``[C, N, 2]``."""

    depths: torch.Tensor
    """View-space depths, ``[C, N]``."""

    conics: torch.Tensor
    """Inverse 2D covariances as ``(a, b, c)`` of ``a x^2 + 2 b x y + c y^2``, ``[C, N, 3]``."""

    compensations: torch.Tensor | None
    """Anti-aliasing opacity compensation factors, ``[C, N]``, or ``None`` when ``antialias`` was off."""

    image_width: int
    """Width in pixels of the images the Gaussians were projected into."""

    image_height: int
    """Height in pixels of the images the Gaussians were projected into."""

    camera_model: CameraModel
    """Camera model used for the projection."""

    projection_method: ProjectionMethod
    """The projection method actually used, with :attr:`~fvdb_reality_capture.ProjectionMethod.AUTO` resolved."""

    token: object = field(default_factory=object, repr=False)
    """Identity of this projection. Tile intersections record it so stale tiles are caught; ``replace`` keeps it."""

    @property
    def num_cameras(self) -> int:
        """Number of cameras ``C``."""
        return self.means2d.shape[0]

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians ``N``."""
        return self.means2d.shape[1]

    @property
    def is_differentiable(self) -> bool:
        """Whether gradients flow from the 2D quantities back to the 3D Gaussian parameters.

        This is ``True`` for the analytic projection and ``False`` for the unscented transform, whose
        kernel has no backward pass. World-space rasterization differentiates through the 3D
        parameters directly and does not need it.
        """
        return self.projection_method == ProjectionMethod.ANALYTIC


@dataclass(frozen=True, eq=False)
class GaussianTileIntersection:
    """Output of :func:`intersect_gaussian_tiles`: which Gaussians touch which image tile, depth sorted."""

    tile_offsets: torch.Tensor
    """Start index into :attr:`tile_gaussian_ids` for each tile, ``[C, num_tiles_h, num_tiles_w]``."""

    tile_gaussian_ids: torch.Tensor
    """Flattened Gaussian index of every tile intersection, ``[num_intersections]``."""

    tile_size: int
    """Tile side length in pixels."""

    image_width: int
    """Image width in pixels."""

    image_height: int
    """Image height in pixels."""

    projection_token: object = field(repr=False)
    """The :attr:`ProjectedGaussians.token` of the projection these tiles were intersected for."""

    @property
    def num_tiles_w(self) -> int:
        """Number of tiles along the image width."""
        return self.tile_offsets.shape[2]

    @property
    def num_tiles_h(self) -> int:
        """Number of tiles along the image height."""
        return self.tile_offsets.shape[1]


@dataclass(frozen=True, eq=False)
class SparseGaussianTileIntersection:
    """
    Output of :func:`intersect_gaussian_tiles_sparse`: tile bookkeeping for rendering an arbitrary
    set of pixels.

    The sparse kernels require each pixel to appear once per camera, so the requested pixels are
    deduplicated here. Downstream stages render :attr:`unique_pixels` and use
    :attr:`inverse_indices` to expand their results back to the order of :attr:`pixels_to_render`.
    """

    tile_offsets: torch.Tensor
    """Start index into :attr:`tile_gaussian_ids` per active tile plus a trailing end, ``[AT + 1]``."""

    tile_gaussian_ids: torch.Tensor
    """Flattened Gaussian index of every tile intersection, ``[num_intersections]``."""

    pixels_to_render: JaggedTensor
    """The requested ``(row, col)`` pixels, one list per camera, normalized to a JaggedTensor by :func:`as_pixel_jagged`."""

    unique_pixels: JaggedTensor
    """The requested pixels with per-camera duplicates removed. Equal to :attr:`pixels_to_render` if none."""

    inverse_indices: torch.Tensor
    """Index into the flat unique pixels for each flat requested pixel, ``[num_requested]``. Empty when
    :attr:`has_duplicates` is ``False``, since nothing needs reordering then."""

    has_duplicates: bool
    """Whether :attr:`pixels_to_render` contained duplicate pixels within a camera."""

    active_tiles: torch.Tensor
    """Flattened ids of the tiles that contain at least one requested pixel, ``[AT]``."""

    active_tile_mask: torch.Tensor
    """Boolean mask of active tiles, ``[C, num_tiles_h, num_tiles_w]``."""

    tile_pixel_mask: torch.Tensor
    """Per active tile bitmask of the requested pixels within it, ``[AT, words_per_tile]``, ``uint64``."""

    tile_pixel_cumsum: torch.Tensor
    """Inclusive cumulative count of requested pixels over the active tiles, ``[AT]``."""

    pixel_map: torch.Tensor
    """Output slot of the ``k``-th requested pixel of each active tile, ``[num_unique]``."""

    tile_size: int
    """Tile side length in pixels."""

    image_width: int
    """Image width in pixels."""

    image_height: int
    """Image height in pixels."""

    projection_token: object = field(repr=False)
    """The :attr:`ProjectedGaussians.token` of the projection these tiles were intersected for."""

    @property
    def num_cameras(self) -> int:
        """Number of cameras ``C``."""
        return self.active_tile_mask.shape[0]

    def expand_to_requested(self, per_unique_pixel: torch.Tensor) -> torch.Tensor:
        """Reorder a flat per-unique-pixel result into the order of :attr:`pixels_to_render`.

        Args:
            per_unique_pixel (torch.Tensor): Tensor whose first dimension runs over the unique pixels.

        Returns:
            per_requested_pixel (torch.Tensor): The same values indexed by the requested pixels.
        """
        if not self.has_duplicates:
            return per_unique_pixel
        return per_unique_pixel.index_select(0, self.inverse_indices)
