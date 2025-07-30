# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch


# TODO: Turn into operation on Checkpoint
def _filter_splat_means(splats, percentile=[0.98, 0.98, 0.98, 0.98, 0.98, 0.98], decimate=4):
    raise NotImplementedError(
        "This function is not used and is here for reference during development. It will be removed soon."
    )
    """
    Remove all gaussians with locations falling outside the provided percentile ranges
    Args:
        splats: dictionary containing splat info to filter
        percentile: drop all splats with locations outside the percentiles (minx, maxx, miny, maxy, minz, maxz)
        decimate: decimate the number of splats by this factor when calculating the percentile range

    Returns:
        dictionary of splats after removal of gaussians outside bounds
    """
    points = splats["means"]

    lower_boundx = torch.quantile(points[::decimate, 0], 1.0 - percentile[0])
    upper_boundx = torch.quantile(points[::decimate, 0], percentile[1])

    lower_boundy = torch.quantile(points[::decimate, 1], 1.0 - percentile[2])
    upper_boundy = torch.quantile(points[::decimate, 1], percentile[3])

    lower_boundz = torch.quantile(points[::decimate, 2], 1.0 - percentile[4])
    upper_boundz = torch.quantile(points[::decimate, 2], percentile[5])

    good_inds = torch.logical_and(points[:, 0] > lower_boundx, points[:, 0] < upper_boundx)
    good_inds = torch.logical_and(good_inds, points[:, 1] > lower_boundy)
    good_inds = torch.logical_and(good_inds, points[:, 1] < upper_boundy)
    good_inds = torch.logical_and(good_inds, points[:, 2] > lower_boundz)
    good_inds = torch.logical_and(good_inds, points[:, 2] < upper_boundz)

    splats["means"] = splats["means"][good_inds, :]
    splats["logit_opacities"] = splats["logit_opacities"][good_inds]
    splats["quats"] = splats["quats"][good_inds, :]
    splats["log_scales"] = splats["log_scales"][good_inds, :]
    splats["sh0"] = splats["sh0"][good_inds, :]
    splats["shN"] = splats["shN"][good_inds, :]
    splats["accumulated_gradient_step_counts_for_grad"] = splats["accumulated_gradient_step_counts_for_grad"][good_inds]
    splats["accumulated_mean_2d_gradient_norms_for_grad"] = splats["accumulated_mean_2d_gradient_norms_for_grad"][
        good_inds
    ]

    return splats


# TODO: Turn into operation on Checkpoint
def _prune_large(splats, prune_scale3d_threshold=0.05):
    raise NotImplementedError(
        "This function is not used and is here for reference during development. It will be removed soon."
    )
    """
    Remove all gaussians with sizes larger than provided percent threshold (relative to scene scale)
    Args:
        splats: dictionary containing splat info to filter
        percentile: drop all spats with opacities outside this percentile
        decimate: decimate the number of splats by this factor when calculating the percentile range

    Returns:
        dictionary of splats after removal of gaussians outside bounds
    """

    points = splats["means"]
    scene_center = torch.mean(points, dim=0)
    dists = torch.linalg.norm(points - scene_center, dim=1)
    scene_scale = torch.max(dists) * 1.1
    good_inds = torch.exp(splats["log_scales"]).max(dim=-1).values < prune_scale3d_threshold * scene_scale

    splats["means"] = splats["means"][good_inds, :]
    splats["logit_opacities"] = splats["logit_opacities"][good_inds]
    splats["quats"] = splats["quats"][good_inds, :]
    splats["log_scales"] = splats["log_scales"][good_inds, :]
    splats["sh0"] = splats["sh0"][good_inds, :]
    splats["shN"] = splats["shN"][good_inds, :]
    splats["accumulated_gradient_step_counts_for_grad"] = splats["accumulated_gradient_step_counts_for_grad"][good_inds]
    splats["accumulated_mean_2d_gradient_norms_for_grad"] = splats["accumulated_mean_2d_gradient_norms_for_grad"][
        good_inds
    ]

    return splats


# TODO: Turn into operation on Checkpoint
def _filter_splat_opacities(splats, percentile=0.98, decimate=4):
    raise NotImplementedError(
        "This function is not used and is here for reference during development. It will be removed soon."
    )
    """
    Remove all gaussians falling outside provided percentile range for logit_opacities.
    Args:
        splats: dictionary containing splat info to filter
        percentile: drop all spats with opacities outside this percentile range
        decimate: decimate the number of splats by this factor when calculating the percentile range

    Returns:
        dictionary of splats after removal of gaussians outside bounds
    """
    lower_bound = torch.quantile(splats["logit_opacities"][::decimate], 1.0 - percentile)
    good_inds = splats["logit_opacities"] > lower_bound

    splats["means"] = splats["means"][good_inds, :]
    splats["logit_opacities"] = splats["logit_opacities"][good_inds]
    splats["quats"] = splats["quats"][good_inds, :]
    splats["log_scales"] = splats["log_scales"][good_inds, :]
    splats["sh0"] = splats["sh0"][good_inds, :]
    splats["shN"] = splats["shN"][good_inds, :]
    splats["accumulated_gradient_step_counts_for_grad"] = splats["accumulated_gradient_step_counts_for_grad"][good_inds]
    splats["accumulated_mean_2d_gradient_norms_for_grad"] = splats["accumulated_mean_2d_gradient_norms_for_grad"][
        good_inds
    ]

    return splats
