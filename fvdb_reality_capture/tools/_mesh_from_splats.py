# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch
from fvdb import GaussianSplat3d
from fvdb.types import NumericMaxRank2, NumericMaxRank3

from ._common import validate_camera_matrices_and_image_sizes
from ._tsdf_from_splats import tsdf_from_splats


@torch.no_grad()
def mesh_from_splats(
    model: GaussianSplat3d,
    camera_to_world_matrices: NumericMaxRank3,
    projection_matrices: NumericMaxRank3,
    image_sizes: NumericMaxRank2,
    truncation_margin: float,
    grid_shell_thickness: float = 3.0,
    near: float = 0.1,
    far: float = 1e10,
    alpha_threshold: float = 0.1,
    image_downsample_factor: int = 1,
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract a triangle mesh from a `fvdb.GaussianSplat3d` using TSDF fusion
    from depth maps rendered from the Gaussian splat model.

    In short, this algorithm works by rendering images and depth maps from multiple views of the Gaussian splat model,
    and then integrating these depth maps and images into a sparse `fvdb.Grid` in a narrow band around the surface using a weighted averaging scheme.
    The algorithm returns this grid along with signed distance values and colors (or other features) at each voxel.

    The algorithm then extracts a mesh using the marching cubes algorithm implemented in `fvdb.marching_cubes.marching_cubes`
    over the Grid and TSDF values.

    The TSDF fusion algorithm is a method for integrating multiple depth maps into a single volumetric representation of a scene encoding a
    truncated signed distance field (_i.e._ a signed distance field in a narrow band around the surface). TSDF fusion was first described in the paper
    "KinectFusion: Real-Time Dense Surface Mapping and Tracking"
    (https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf).
    We use a modified version of this algorithm which only allocates voxels in a narrow band around the surface of the model
    to reduce memory usage and speed up computation.

    Args:
        model (GaussianSplat3d): The Gaussian splat model to extract a mesh from
        camera_to_world_matrices (torch.Tensor): A (C, 4, 4)-shaped Tensor containing the camera to world
            matrices to render depth images from for mesh extraction where C is the number of camera views.
        projection_matrices (torch.Tensor): A (C, 3, 3)-shaped Tensor containing the perspective projection matrices
            used to render images for mesh extraction where C is the number of camera views.
        image_sizes (torch.Tensor): A (C, 2)-shaped Tensor containing the width and height of each image to extract
            from the Gaussian splat where C is the number of camera views.
        truncation_margin (float): Margin for truncating the TSDF, in world units.
        grid_shell_thickness (float): Thickness of the TSDF grid shell in multiples of the truncation margin (default is 3.0).
            _i.e_. if truncation_margin is 0.1 and grid_shell_thickness is 3.0, the TSDF grid will extend 0.3 world units
            from the surface of the model.
        near (float): Near plane distance below which to ignore depth samples (default is 0.1).
        far (float): Far plane distance above which to ignore depth samples (default is 1e10).
        alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model is transparent
            (usually indicating the background). Default is 0.1.
        image_downsample_factor (int): Factor by which to downsample the rendered images for depth estimation (default is 1, i.e. no downsampling).
        dtype: Data type for the TSDF and weights. Default is torch.float16.
        feature_dtype: Data type for the features (default is torch.uint8 which is good for RGB colors).
        show_progress (bool): Whether to show a progress bar (default is True).

    Returns:
        mesh_vertices (torch.Tensor): Vertices of the extracted mesh.
        mesh_faces (torch.Tensor): Faces of the extracted mesh.
        mesh_colors (torch.Tensor): Colors of the extracted mesh vertices.
    """

    camera_to_world_matrices, projection_matrices, image_sizes = validate_camera_matrices_and_image_sizes(
        camera_to_world_matrices, projection_matrices, image_sizes
    )

    accum_grid, tsdf, colors = tsdf_from_splats(
        model,
        camera_to_world_matrices,
        projection_matrices,
        image_sizes,
        truncation_margin,
        grid_shell_thickness=grid_shell_thickness,
        near=near,
        far=far,
        alpha_threshold=alpha_threshold,
        image_downsample_factor=image_downsample_factor,
        dtype=dtype,
        feature_dtype=feature_dtype,
        show_progress=show_progress,
    )

    mesh_vertices, mesh_faces, _ = accum_grid.marching_cubes(tsdf, 0.0)
    mesh_colors = accum_grid.sample_trilinear(mesh_vertices, colors.to(dtype)) / 255.0
    mesh_colors.clip_(min=0.0, max=1.0)

    return mesh_vertices, mesh_faces, mesh_colors
