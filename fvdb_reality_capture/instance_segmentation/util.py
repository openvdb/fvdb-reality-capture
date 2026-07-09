# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch


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
