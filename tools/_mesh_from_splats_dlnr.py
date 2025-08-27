# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch

from fvdb import GaussianSplat3d

from ._tsdf_from_splats_dlnr import tsdf_from_splats_dlnr


@torch.no_grad()
def mesh_from_splats_dlnr(
    model: GaussianSplat3d,
    camera_to_world_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    image_sizes: torch.Tensor,
    truncation_margin: float,
    baseline: float = 0.07,
    near: float = 4.0,
    far: float = 20.0,
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
        model (GaussianSplat3d): The Gaussian splat model to extract a mesh from
        camera_to_world_matrices (torch.Tensor): A (C, 4, 4)-shaped Tensor containing the camera to world
            matrices to render depth images from for mesh extraction where C is the number of camera views.
        projection_matrices (torch.Tensor): A (C, 3, 3)-shaped Tensor containing the perspective projection matrices
            used to render images for mesh extraction where C is the number of camera views.
        image_sizes (torch.Tensor): A (C, 2)-shaped Tensor containing the width and height of each image to extract
            from the Gaussian splat where C is the number of camera views.
        truncation_margin (float): Margin for truncating the TSDF, in world units.
        baseline (float): Baseline for the DLNR model as a percentage of the scene scale (default is 0.07).
            The scene scale is defined as the median distance from the camera origins to their mean.
        near (float): Near plane distance as a multiple of the baseline below which to ignore depth samples (default is 4.0).
        far (float): Far plane distance as a multiple of the baseline above which to ignore depth samples (default is 20.0).
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

    near_rescaled = near * baseline
    far_rescaled = far * baseline

    accum_grid, tsdf, colors = tsdf_from_splats_dlnr(
        model=model,
        camera_to_world_matrices=camera_to_world_matrices,
        projection_matrices=projection_matrices,
        image_sizes=image_sizes,
        truncation_margin=truncation_margin,
        baseline=baseline,
        near=near_rescaled,
        far=far_rescaled,
        dtype=dtype,
        feature_dtype=feature_dtype,
        dlnr_backbone=dlnr_backbone,
        show_progress=show_progress,
    )

    mesh_vertices, mesh_faces, _ = accum_grid.marching_cubes(tsdf, 0.0)
    mesh_colors = accum_grid.sample_trilinear(mesh_vertices, colors.to(dtype)) / 255.0
    mesh_colors.clip_(min=0.0, max=1.0)

    return mesh_vertices, mesh_faces, mesh_colors
