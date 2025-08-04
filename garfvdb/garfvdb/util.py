# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Optional

import torch


def center_features(features: torch.Tensor) -> torch.Tensor:
    """Center the features."""
    mean = torch.mean(features, dim=0, keepdim=True)
    return features - mean


def calculate_pca_projection(features: torch.Tensor, n_components: int = 3, center: bool = True) -> torch.Tensor:
    """Calculate the PCA projection matrix.

    Args:
        features: A 4D tensor of shape [B, H, W, C] containing features to project
        n_components: The number of principal components to use

    Returns:
        A 2D tensor of shape [C, n_components] containing the PCA projection matrix
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
    features: torch.Tensor, n_components: int = 3, V: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Project features using PCA to a lower dimension.

    Args:
        features: A 4D tensor of shape [B, H, W, C] containing features to project
        n_components: The number of principal components to use
        mask: A tensor of shape [B, H, W] containing a mask of valid features

    Returns:
        A 4D tensor of shape [B, H, W, n_components] containing the projected features
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
    projected = torch.mm(features_flat, V.to(features.device))

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
    """Convert a 2D tensor to a 3D RGB tensor with distinct colors.

    Args:
        tensor: A 2D tensor of shape [H, W]

    Returns:
        A 3D tensor of shape [H, W, 3] where each unique value is mapped to a distinct RGB color
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


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB values to spherical harmonics coefficients.

    Args:
        rgb: [N, 3] Tensor of RGB values

    Returns:
        [N, 3] Tensor of spherical harmonics coefficients
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
