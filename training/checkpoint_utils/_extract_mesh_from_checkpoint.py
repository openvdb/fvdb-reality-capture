# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch

from ..checkpoint import Checkpoint
from ._extract_tsdf_from_checkpoint import extract_tsdf_from_checkpoint
from ._extract_tsdf_from_checkpoint_dlnr import extract_tsdf_from_checkpoint_dlnr


@torch.no_grad()
def extract_mesh_from_checkpoint(
    checkpoint: Checkpoint,
    truncation_margin: float,
    near: float = 0.1,
    far: float = 1e10,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract a mesh from a Gaussian splat using Truncated Signed Distance Field (TSDF) fusion and marching cubes.

    The TSDF fusion algorithm is based on the paper:
    "KinectFusion: Real-Time Dense Surface Mapping and Tracking"
    (https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/)

    Args:
        checkpoint (Checkpoint): A checkpoint containing the Gaussian splat model and camera parameters.
        truncation_margin (float): Margin for truncating the TSDF, in world units.
        near (float): Near plane distance below which to ignore depth samples (default is 0.0).
        far (float): Far plane distance above which to ignore depth samples (default is 1e10).
        device (torch.device | str): Device to use (default is "cuda").
        dtype: Data type for the TSDF and weights. Default is torch.float16.
        feature_dtype: Data type for the features (default is torch.uint8 which is good for RGB colors).
        show_progress (bool): Whether to show a progress bar (default is True).

    Returns:
        mesh_vertices (torch.Tensor): Vertices of the extracted mesh.
        mesh_faces (torch.Tensor): Faces of the extracted mesh.
        mesh_colors (torch.Tensor): Colors of the extracted mesh vertices.
    """

    accum_grid, tsdf, colors = extract_tsdf_from_checkpoint(
        checkpoint,
        truncation_margin,
        near=near,
        far=far,
        device=device,
        dtype=dtype,
        feature_dtype=feature_dtype,
        show_progress=show_progress,
    )

    mesh_vertices, mesh_faces, _ = accum_grid.marching_cubes(tsdf, 0.0)
    mesh_colors = accum_grid.sample_trilinear(mesh_vertices, colors.to(dtype)) / 255.0
    mesh_colors.clip_(min=0.0, max=1.0)

    return mesh_vertices, mesh_faces, mesh_colors


@torch.no_grad()
def extract_mesh_from_checkpoint_dlnr(
    checkpoint: Checkpoint,
    truncation_margin: float,
    baseline: float = 0.07,
    near: float = 4.0,
    far: float = 20.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    dlnr_backbone: str = "middleburry",
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract a mesh from a Gaussian splat using Truncated Signed Distance Field (TSDF) fusion and
    marching cubes where depth maps for TSDF fusion are estimated using the DNLR model.

    The mesh extraction algorithm is based on the paper:
    "GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views"
    (https://arxiv.org/abs/2404.01810)

    The DLNR model is a high-frequency stereo matching network that computes optical flow and disparity maps
    between two images. The DLNR model is described in the paper "High-Frequency Stereo Matching Network"
    (https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf).

    The TSDF fusion algorithm is based on the paper:
    "KinectFusion: Real-Time Dense Surface Mapping and Tracking"
    (https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/)

    Args:
        checkpoint (Checkpoint): A checkpoint containing the Gaussian splat model and camera parameters.
        truncation_margin (float): Margin for truncating the TSDF, in world units.
        baseline (float): Baseline for the DLNR model as a percentage of the scene scale (default is 0.07).
            The scene scale is defined as the median distance from the camera origins to their mean.
        near (float): Near plane distance as a multiple of the baseline below which to ignore depth samples (default is 4.0).
        far (float): Far plane distance as a multiple of the baseline above which to ignore depth samples (default is 20.0).
        device (torch.device | str): Device to use (default is "cuda").
        dtype: Data type for the TSDF and weights. Default is torch.float16.
        feature_dtype: Data type for the features (default is torch.uint8 which is good for RGB colors).
        dlnr_backbone (str): Backbone to use for the DLNR model, either "middleburry" or "sceneflow".
            Default is "middleburry".
        show_progress (bool): Whether to show a progress bar (default is True).
    Returns:
        mesh_vertices (torch.Tensor): Vertices of the extracted mesh.
        mesh_faces (torch.Tensor): Faces of the extracted mesh.
        mesh_colors (torch.Tensor): Colors of the extracted mesh vertices.
    """

    baseline_rescaled = baseline * checkpoint.scene_scale
    near_rescaled = near * baseline_rescaled
    far_rescaled = far * baseline_rescaled

    accum_grid, tsdf, colors = extract_tsdf_from_checkpoint_dlnr(
        checkpoint=checkpoint,
        truncation_margin=truncation_margin,
        baseline=baseline_rescaled,
        near=near_rescaled,
        far=far_rescaled,
        device=device,
        dtype=dtype,
        feature_dtype=feature_dtype,
        dlnr_backbone=dlnr_backbone,
        show_progress=show_progress,
    )

    mesh_vertices, mesh_faces, _ = accum_grid.marching_cubes(tsdf, 0.0)
    mesh_colors = accum_grid.sample_trilinear(mesh_vertices, colors.to(dtype)) / 255.0
    mesh_colors.clip_(min=0.0, max=1.0)

    return mesh_vertices, mesh_faces, mesh_colors
