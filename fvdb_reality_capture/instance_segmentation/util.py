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
    bound) write time. This must stay byte-for-byte consistent with the value that used to be
    written, so the implementation matches the original inline computation exactly.

    Args:
        pixel_to_mask_id: ``[H, W, MM]`` integer tensor mapping each pixel to the ids of the masks
            that intersect it (``-1`` padding), packed in order of decreasing mask area.

    Returns:
        ``[H, W, MM]`` float32 CDF used to sample a mask per pixel during training.
    """
    # Get the unique ids of each mask, and the number of pixels each mask occupies (area)
    mask_ids, num_pix_per_mask = torch.unique(pixel_to_mask_id, return_counts=True)  # [N], [N]

    # Sort masks by their area
    mask_area_sort_ids = torch.argsort(num_pix_per_mask)
    mask_ids, num_pix_per_mask = mask_ids[mask_area_sort_ids], num_pix_per_mask[mask_area_sort_ids]  # [N], [N]
    num_pix_per_mask[0] = 0  # Remove the -1 mask which corresponds to no mask, [N]

    # The probability of any pixel landing in a mask is just the area of the mask over the area of the image
    probs = num_pix_per_mask / num_pix_per_mask.sum()  # [N]

    # Gather the probability values into pixel_to_mask_id, which produces a tensor where
    # each pixel has a list of probabilities that correspond to the masks that intersect that pixel
    mask_probs = torch.gather(probs, 0, pixel_to_mask_id.reshape(-1).long() + 1).view(
        pixel_to_mask_id.shape
    )  # [H, W, MM]

    # Compute a CDF for each pixel (which sums to 1) which weighs each mask by its log probability of being sampled
    # i.e. mask_cdf[i, j, k] is a cumulative probability weight used to select mask k for pixel [i, j]
    mask_cdf = torch.log(mask_probs)
    never_masked = mask_cdf.isinf()
    mask_cdf[never_masked] = 0.0
    mask_cdf = mask_cdf / (mask_cdf.sum(dim=-1, keepdim=True) + 1e-6)
    mask_cdf = torch.cumsum(mask_cdf, dim=-1)  # [H, W, MM]
    mask_cdf[never_masked] = 1.0
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

    # Generate distinct colors using HSV color space
    # We'll use evenly spaced hues and full saturation/value
    hues = torch.linspace(0, 1, num_unique, device=tensor.device)
    saturation = torch.ones_like(hues)
    value = torch.ones_like(hues)

    # Convert HSV to RGB
    h = hues.unsqueeze(1).unsqueeze(2)  # [num_unique, 1, 1]
    s = saturation.unsqueeze(1).unsqueeze(2)  # [num_unique, 1, 1]
    v = value.unsqueeze(1).unsqueeze(2)  # [num_unique, 1, 1]

    # HSV to RGB conversion
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c

    # Create RGB components
    rgb = torch.zeros((num_unique, 1, 1, 3), device=tensor.device)
    rgb[:, 0, 0, 0] = c.squeeze()  # R
    rgb[:, 0, 0, 1] = x.squeeze()  # G
    rgb[:, 0, 0, 2] = m.squeeze()  # B

    # Map each unique value to its color
    color_map = rgb.squeeze(1).squeeze(1)  # [num_unique, 3]

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
