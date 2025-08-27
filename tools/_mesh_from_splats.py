# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch

from fvdb import GaussianSplat3d

from ._tsdf_from_splats import tsdf_from_splats


@torch.no_grad()
def mesh_from_splats(
    model: GaussianSplat3d,
    camera_to_world_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    image_sizes: torch.Tensor,
    truncation_margin: float,
    near: float = 0.1,
    far: float = 1e10,
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
        model (GaussianSplat3d): The Gaussian splat model to extract a mesh from
        camera_to_world_matrices (torch.Tensor): A (C, 4, 4)-shaped Tensor containing the camera to world
            matrices to render depth images from for mesh extraction where C is the number of camera views.
        projection_matrices (torch.Tensor): A (C, 3, 3)-shaped Tensor containing the perspective projection matrices
            used to render images for mesh extraction where C is the number of camera views.
        image_sizes (torch.Tensor): A (C, 2)-shaped Tensor containing the width and height of each image to extract
            from the Gaussian splat where C is the number of camera views.
        truncation_margin (float): Margin for truncating the TSDF, in world units.
        near (float): Near plane distance below which to ignore depth samples (default is 0.0).
        far (float): Far plane distance above which to ignore depth samples (default is 1e10).
        dtype: Data type for the TSDF and weights. Default is torch.float16.
        feature_dtype: Data type for the features (default is torch.uint8 which is good for RGB colors).
        show_progress (bool): Whether to show a progress bar (default is True).

    Returns:
        mesh_vertices (torch.Tensor): Vertices of the extracted mesh.
        mesh_faces (torch.Tensor): Faces of the extracted mesh.
        mesh_colors (torch.Tensor): Colors of the extracted mesh vertices.
    """

    accum_grid, tsdf, colors = tsdf_from_splats(
        model,
        camera_to_world_matrices,
        projection_matrices,
        image_sizes,
        truncation_margin,
        near=near,
        far=far,
        dtype=dtype,
        feature_dtype=feature_dtype,
        show_progress=show_progress,
    )

    mesh_vertices, mesh_faces, _ = accum_grid.marching_cubes(tsdf, 0.0)
    mesh_colors = accum_grid.sample_trilinear(mesh_vertices, colors.to(dtype)) / 255.0
    mesh_colors.clip_(min=0.0, max=1.0)

    return mesh_vertices, mesh_faces, mesh_colors
