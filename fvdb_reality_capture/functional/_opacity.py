# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Per-camera opacity computation shared by the rasterization and analysis stages."""

from __future__ import annotations

import torch

from ._types import ProjectedGaussians


def compute_gaussian_opacities(logit_opacities: torch.Tensor, projected: ProjectedGaussians) -> torch.Tensor:
    """Turn logit opacities into the per-camera opacities the rasterization kernels consume.

    Applies the sigmoid, repeats the result for every camera, and multiplies in the anti-aliasing
    compensation factors when the projection computed them.

    Args:
        logit_opacities (torch.Tensor): Logit opacities, shape ``[N]``.
        projected (ProjectedGaussians): Projection whose camera count and compensations to use.

    Returns:
        opacities (torch.Tensor): Opacities in ``[0, 1]``, shape ``[C, N]``, contiguous.
    """
    # The kernels require a contiguous [C, N] tensor, so the per-camera copy is materialized.
    opacities = torch.sigmoid(logit_opacities).repeat(projected.num_cameras, 1)
    if projected.compensations is not None:
        opacities = opacities * projected.compensations
    return opacities


def check_opacities(opacities: torch.Tensor, projected: ProjectedGaussians) -> torch.Tensor:
    """Validate per-camera opacities against a projection and return them ready for the kernels.

    Args:
        opacities (torch.Tensor): Opacities from :func:`compute_gaussian_opacities`, shape ``[C, N]``.
        projected (ProjectedGaussians): Projection the opacities were computed for.

    Returns:
        opacities (torch.Tensor): The same values, contiguous and on the projection's device.
    """
    if tuple(opacities.shape) != (projected.num_cameras, projected.num_gaussians):
        raise ValueError(
            f"opacities must have shape [{projected.num_cameras}, {projected.num_gaussians}], got {tuple(opacities.shape)}"
        )
    if opacities.device != projected.means2d.device:
        raise ValueError(f"opacities must be on {projected.means2d.device}, got {opacities.device}")
    # The kernels index a dense [C, N] buffer, so an expanded or strided view is materialized here
    # rather than failing inside the kernel.
    return opacities.contiguous()
