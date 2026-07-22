# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import NamedTuple

import torch


def compute_mask_cdf(pixel_to_mask_id: torch.Tensor) -> torch.Tensor:
    """Compute the per-pixel mask-selection CDF from a pixel-to-mask-id tensor.

    ``mask_cdf`` is fully derived from ``pixel_to_mask_id`` (it depends only on the areas of the
    masks intersecting each pixel), so it is recomputed at load time rather than stored on disk.
    A full ``[H, W, MM]`` float32 CDF is by far the largest field in the mask cache (~155 MB for a
    ~2K image), and dropping it from the cache roughly quarters the on-disk artifact and the (disk
    bound) write time.

    Each mask is weighted by ``-log`` of its sampling probability (its area over the total masked
    area) so that small masks are more likely to be selected for supervision; the ``-1`` padding is
    excluded. Pixels whose only mask has probability 1 (a lone mask) or which intersect no mask at
    all fall back to a uniform distribution so the CDF always terminates at 1 on a real mask rather
    than on padding.

    Args:
        pixel_to_mask_id: ``[H, W, MM]`` integer tensor mapping each pixel to the ids of the masks
            that intersect it (``-1`` padding), packed in order of decreasing mask area.

    Returns:
        ``[H, W, MM]`` float32 CDF used to sample a mask per pixel during training.
    """
    # Shift ids by +1 so the -1 "no mask" padding maps to index 0 of the per-id tables.
    shifted = pixel_to_mask_id.reshape(-1).long() + 1  # [H*W*MM]
    num_ids = int(shifted.max().item()) + 1 if shifted.numel() > 0 else 1

    # Per-mask area = number of pixel slots each mask id occupies, indexed by (id + 1).
    area_per_id = torch.bincount(shifted, minlength=num_ids).to(torch.float32)
    area_per_id[0] = 0.0  # drop the -1 padding so it contributes no probability

    # Probability of sampling each mask = its area over the total masked area.
    probs = area_per_id / area_per_id.sum().clamp(min=1e-12)  # indexed by (id + 1)
    mask_probs = probs[shifted].view(pixel_to_mask_id.shape)  # [H, W, MM]

    # Weight each valid mask by -log(prob) (>= 0 for prob in (0, 1]): smaller masks (smaller prob)
    # get more weight, biasing sampling toward them. Padding / never-masked slots have prob 0
    # (-log -> +inf) and are excluded.
    weights = -torch.log(mask_probs)
    valid = torch.isfinite(weights)
    weights = torch.where(valid, weights, torch.zeros_like(weights))

    # The per-pixel weight sum is 0 when a pixel's only mask has prob 1 (a lone mask, -log(1) = 0)
    # or when a pixel intersects no mask at all. In the lone-mask case fall back to a uniform
    # distribution over the pixel's valid masks so the CDF still reaches 1 on a real mask instead of
    # collapsing to all zeros (which would make sampling select padding).
    weight_sum = weights.sum(dim=-1, keepdim=True)
    degenerate = weight_sum <= 0.0
    valid_weights = valid.to(weights.dtype)
    weights = torch.where(degenerate, valid_weights, weights)
    weight_sum = torch.where(degenerate, valid_weights.sum(dim=-1, keepdim=True), weight_sum)

    mask_pdf = weights / weight_sum.clamp(min=1e-12)
    mask_cdf = torch.cumsum(mask_pdf, dim=-1)  # [H, W, MM]
    # Push padding slots (and fully-unmasked pixels, whose pdf is all zero) to 1 so a uniform sample
    # never selects padding when at least one real mask exists.
    mask_cdf[~valid] = 1.0
    return mask_cdf


def center_features(features: torch.Tensor) -> torch.Tensor:
    """Center features by subtracting the mean across samples.

    Args:
        features: Tensor of shape [N, C] where N is the number of samples and C is the feature dimension.

    Returns:
        Zero-mean features with the same shape as input.
    """
    mean = torch.mean(features, dim=0, keepdim=True)
    return features - mean


def calculate_pca_projection(features: torch.Tensor, n_components: int = 3, center: bool = True) -> torch.Tensor:
    """Calculate the PCA projection matrix from feature data.

    Computes the principal components of the input features using low-rank SVD.

    Args:
        features: Feature tensor of shape ``[B, H, W, C]`` or ``[N, C]``.
        n_components: Number of principal components to compute.
        center: If True, center features before computing PCA.

    Returns:
        Projection matrix of shape ``[C, n_components]`` containing the
        principal component vectors.
    """
    features_flat = features.reshape(-1, features.shape[-1])

    # Center the data
    if center:
        features_centered = center_features(features_flat)
    else:
        features_centered = features_flat

    _, _, V = torch.pca_lowrank(features_centered, q=n_components, center=False)

    return V


def pca_projection_fast(
    features: torch.Tensor,
    n_components: int = 3,
    V: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project features to a lower dimension using PCA.

    Projects high-dimensional features onto the first few principal components
    and normalizes the result to [0, 1] range for visualization.

    Args:
        features: Feature tensor of shape ``[B, H, W, C]``.
        n_components: Number of principal components to project onto.
        V: Optional pre-computed projection matrix of shape ``[C, n_components]``.
            If None, PCA is computed from the input features.
        mask: Optional boolean mask of shape ``[B, H, W]`` indicating valid
            features. Invalid features are set to zero in the output.

    Returns:
        Projected features of shape ``[B, H, W, n_components]`` normalized
        to [0, 1] range.
    """
    B, H, W, C = features.shape

    if mask is not None:
        features = features[mask]
    features_flat = features.reshape(-1, C)

    # Center the data
    features_centered = center_features(features_flat)

    if V is None:
        V = calculate_pca_projection(features_centered, n_components, center=False)

    # Project data onto principal components
    projected = torch.mm(features_centered, V.to(features.device))

    # Normalize to [0, 1] range
    mins = projected.min(dim=0, keepdim=True)[0]
    maxs = projected.max(dim=0, keepdim=True)[0]
    projected_normalized = (projected - mins) / (maxs - mins + 1e-8)

    if mask is not None:
        result = torch.zeros(B, H, W, n_components, device=features.device)
        result[mask] = projected_normalized
    else:
        result = projected_normalized

    return result


class PCAProjectionState(NamedTuple):
    """A frozen PCA-to-RGB transform.

    Captures every per-frame quantity that :func:`pca_projection_fast` recomputes — the centering
    ``mean``, the principal-component ``basis``, and the ``mins``/``maxs`` used for [0, 1] normalization
    — so the same feature vector maps to the same color across frames. Used to "lock" the feature
    visualization so colors do not flicker when the camera moves.
    """

    mean: torch.Tensor  # [1, C]
    basis: torch.Tensor  # [C, n_components]
    mins: torch.Tensor  # [1, n_components]
    maxs: torch.Tensor  # [1, n_components]


def fit_pca_projection(
    features: torch.Tensor,
    n_components: int = 3,
    mask: torch.Tensor | None = None,
) -> PCAProjectionState:
    """Fit a reusable PCA-to-RGB transform from a single frame of features.

    Args:
        features: Feature tensor of shape ``[B, H, W, C]``.
        n_components: Number of principal components to project onto.
        mask: Optional boolean mask of shape ``[B, H, W]`` selecting valid features.

    Returns:
        A :class:`PCAProjectionState` that can be reused via :func:`apply_pca_projection`.
    """
    if mask is not None:
        features = features[mask]
    features_flat = features.reshape(-1, features.shape[-1])
    mean = torch.mean(features_flat, dim=0, keepdim=True)
    features_centered = features_flat - mean
    basis = calculate_pca_projection(features_centered, n_components, center=False)
    projected = torch.mm(features_centered, basis)
    mins = projected.min(dim=0, keepdim=True)[0]
    maxs = projected.max(dim=0, keepdim=True)[0]
    return PCAProjectionState(mean=mean, basis=basis, mins=mins, maxs=maxs)


def apply_pca_projection(
    features: torch.Tensor,
    state: PCAProjectionState,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project features to RGB using a previously fitted :class:`PCAProjectionState`.

    Mirrors :func:`pca_projection_fast` but with the transform frozen, so unseen features are placed
    consistently (and may fall outside [0, 1] — callers should clamp).

    Args:
        features: Feature tensor of shape ``[B, H, W, C]``.
        state: A transform produced by :func:`fit_pca_projection`.
        mask: Optional boolean mask of shape ``[B, H, W]`` selecting valid features.

    Returns:
        Projected features of shape ``[B, H, W, n_components]`` (invalid pixels set to zero).
    """
    B, H, W, C = features.shape
    n_components = state.basis.shape[-1]
    device = features.device
    selected = features[mask] if mask is not None else features
    features_flat = selected.reshape(-1, C)
    projected = torch.mm(features_flat - state.mean.to(device), state.basis.to(device))
    mins = state.mins.to(device)
    maxs = state.maxs.to(device)
    projected_normalized = (projected - mins) / (maxs - mins + 1e-8)
    if mask is not None:
        result = torch.zeros(B, H, W, n_components, device=device)
        result[mask] = projected_normalized
    else:
        result = projected_normalized
    return result


def unique_values_to_colors(tensor: torch.Tensor) -> torch.Tensor:
    """Map unique integer values to distinct RGB colors.

    Generates evenly-spaced hues in HSV space for each unique value and
    converts to RGB for visualization of segmentation masks or labels.

    Args:
        tensor: Integer tensor of shape ``[H, W]`` containing label values.

    Returns:
        RGB color tensor of shape ``[H, W, 3]`` with values in [0, 1].
    """
    # Get unique values and their indices
    unique_values, inverse_indices = torch.unique(tensor, return_inverse=True)
    num_unique = len(unique_values)

    # Generate distinct colors using HSV color space with evenly spaced hues and full
    # saturation/value. Exclude the endpoint (linspace of num_unique + 1, drop the last) so hue 0
    # and hue 1 -- which are both red -- don't collide when the wheel is sampled.
    hues = torch.linspace(0, 1, num_unique + 1, device=tensor.device)[:-1]  # [num_unique]
    saturation = torch.ones_like(hues)
    value = torch.ones_like(hues)

    # HSV -> RGB (standard six-sector conversion): c is the chroma, x the second-largest component,
    # and m the achromatic offset added to every channel. Which of (c, x, 0) each of R, G, B takes
    # depends on the 60-degree hue sector, so the assignment must be permuted per sector rather than
    # fixed to (R, G, B) = (c, x, m).
    c = value * saturation  # [num_unique]
    h6 = hues * 6.0
    x = c * (1 - torch.abs(h6 % 2 - 1))
    m = value - c
    sector = torch.floor(h6).to(torch.long) % 6

    zero = torch.zeros_like(c)
    r = torch.where((sector == 0) | (sector == 5), c, torch.where((sector == 1) | (sector == 4), x, zero))
    g = torch.where((sector == 1) | (sector == 2), c, torch.where((sector == 0) | (sector == 3), x, zero))
    b = torch.where((sector == 3) | (sector == 4), c, torch.where((sector == 2) | (sector == 5), x, zero))
    color_map = torch.stack((r + m, g + m, b + m), dim=-1)  # [num_unique, 3]

    # Create output tensor by mapping indices to colors
    output = color_map[inverse_indices.reshape(tensor.shape)]  # [H, W, 3]

    return output


_SH_SCALE = 1.0 / 0.28209479177387814  # ≈ 3.5449077018110318
_SH_OFFSET = -0.5 * _SH_SCALE  # ≈ -1.7724538509055159


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB values to spherical harmonics coefficients.

    Args:
        rgb: [N, 3] Tensor of RGB values

    Returns:
        [N, 3] Tensor of spherical harmonics coefficients
    """
    return rgb * _SH_SCALE + _SH_OFFSET


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    """Convert degree-zero spherical harmonics coefficients to RGB values.

    Args:
        sh: Tensor of spherical harmonics coefficients.

    Returns:
        Tensor of RGB values with the same shape.
    """
    return (sh - _SH_OFFSET) / _SH_SCALE
