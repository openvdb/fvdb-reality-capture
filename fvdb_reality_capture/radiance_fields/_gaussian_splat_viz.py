# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from fvdb.viz import GaussianSplatViewData, ShOrderingMode

from .gaussian_splatting import GaussianSplat3d


def gaussian_splat_to_view_data(model: GaussianSplat3d) -> GaussianSplatViewData:
    """Adapt a Gaussian splat model to the data contract used by ``fvdb.viz``.

    The returned object retains references to the model's tensors; it does not copy or
    detach them.

    Args:
        model (GaussianSplat3d): The Gaussian splat model to adapt.

    Returns:
        view_data (GaussianSplatViewData): Viewer data referencing the model's tensors.
    """
    return GaussianSplatViewData(
        means=model.means,
        quats=model.quats,
        log_scales=model.log_scales,
        logit_opacities=model.logit_opacities,
        sh0=model.sh0,
        shN=model.shN,
        sh_ordering=ShOrderingMode.RGB_RGB_RGB,
    )


__all__ = ["gaussian_splat_to_view_data"]
