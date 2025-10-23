# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import fvdb
import torch


def filter_splats_by_mean_percentile(
    splats: fvdb.GaussianSplat3d, percentile=[0.98, 0.98, 0.98, 0.98, 0.98, 0.98], decimate=4
) -> fvdb.GaussianSplat3d:
    """
    Remove all gaussians with locations falling outside the provided percentile ranges

    Args:
        splats (fvdb.GaussianSplat3d): The :class:`~fvdb.GaussianSplat3d` to filter.
        percentile (NumericMaxRank1): The percentiles to use for filtering. The percentiles are in the order of (minx, maxx, miny, maxy, minz, maxz).
        decimate (int): Decimate the number of splats by this factor when calculating the percentile range.

    Returns:
        filtered_splats (fvdb.GaussianSplat3d): The :class:`~fvdb.GaussianSplat3d` after removal of gaussians outside percentile bounds
    """
    points = splats.means

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

    return splats[good_inds]


def filter_splats_by_opacity_percentile(
    splats: fvdb.GaussianSplat3d, percentile=0.98, decimate=4
) -> fvdb.GaussianSplat3d:
    """
    Remove all gaussians falling outside provided percentile range for logit_opacities.

    Args:
        splats (fvdb.GaussianSplat3d): The :class:`~fvdb.GaussianSplat3d` to filter.
        percentile (float): The percentile to use for filtering. The percentile is the percentile of the logit_opacities to use for filtering.
        decimate (int): Decimate the number of splats by this factor when calculating the percentile range.

    Returns:
        filtered_splats (fvdb.GaussianSplat3d): The :class:`~fvdb.GaussianSplat3d` after removal of gaussians outside opacity percentile range
    """
    lower_bound = torch.quantile(splats.logit_opacities[::decimate], 1.0 - percentile)
    good_inds = splats.logit_opacities > lower_bound

    return splats[good_inds]


def filter_splats_above_scale(splats: fvdb.GaussianSplat3d, prune_scale3d_threshold=0.05) -> fvdb.GaussianSplat3d:
    """
    Remove all gaussians with sizes larger than provided percent threshold (relative to scene scale)

    Args:
        splats (fvdb.GaussianSplat3d): The :class:`~fvdb.GaussianSplat3d` to filter.
        prune_scale3d_threshold (float): Drop all spats with scales larger than this threshold (relative to scene scale).

    Returns:
        filtered_splats (fvdb.GaussianSplat3d): The :class:`~fvdb.GaussianSplat3d` after removal of gaussians outside threshold
    """

    points = splats.means
    scene_center = torch.mean(points, dim=0)
    dists = torch.linalg.norm(points - scene_center, dim=1)
    scene_scale = torch.max(dists) * 1.1
    good_inds = torch.exp(splats.log_scales).max(dim=-1).values < prune_scale3d_threshold * scene_scale

    return splats[good_inds]


def filter_splats_below_scale(splats: fvdb.GaussianSplat3d, prune_scale3d_threshold=0.05) -> fvdb.GaussianSplat3d:
    """
    Remove all gaussians with sizes smaller than provided percent threshold (relative to scene scale)

    Args:
        splats (fvdb.GaussianSplat3d): The :class:`~fvdb.GaussianSplat3d` to filter.
        prune_scale3d_threshold (float): Drop all spats with scales smaller than this threshold (relative to scene scale).

    Returns:
        filtered_splats (fvdb.GaussianSplat3d): The :class:`~fvdb.GaussianSplat3d` after removal of gaussians outside threshold
    """

    points = splats.means
    scene_center = torch.mean(points, dim=0)
    dists = torch.linalg.norm(points - scene_center, dim=1)
    scene_scale = torch.max(dists) * 1.1
    good_inds = torch.exp(splats.log_scales).max(dim=-1).values > prune_scale3d_threshold * scene_scale

    return splats[good_inds]
