# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch
from fvdb import GaussianSplat3d
from fvdb.types import NumericMaxRank2, NumericMaxRank3

from ._common import validate_camera_matrices_and_image_sizes
from ._tsdf_from_splats_dlnr import tsdf_from_splats_dlnr


@torch.no_grad()
def mesh_from_splats_dlnr(
    model: GaussianSplat3d,
    camera_to_world_matrices: NumericMaxRank3,
    projection_matrices: NumericMaxRank3,
    image_sizes: NumericMaxRank2,
    truncation_margin: float,
    grid_shell_thickness: float = 3.0,
    baseline: float = 0.07,
    near: float = 4.0,
    far: float = 20.0,
    disparity_reprojection_threshold: float = 3.0,
    alpha_threshold: float = 0.1,
    image_downsample_factor: int = 1,
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    dlnr_backbone: str = "middleburry",
    use_absolute_baseline: bool = False,
    show_progress: bool = True,
    num_workers: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract a triangle mesh from a `fvdb.GaussianSplat3d` using TSDF fusion
    from depth maps predicted from the Gaussian splat model and the DLNR foundation model. DLNR is a
    high-frequency stereo matching network that computes optical flow and disparity maps between two images.

    In short, this algorithm works by rendering stereo pairs of images from multiple views of the Gaussian splat model, and
    using DLNR to compute depth maps from these stereo pairs.
    The depth maps and images are then integrated into a sparse `fvdb.Grid` in a narrow band around the surface using a weighted averaging scheme.
    The algorithm returns this grid along with signed distance values and colors (or other features) at each voxel.

    The algorithm then extracts a mesh using the marching cubes algorithm implemented in `fvdb.marching_cubes.marching_cubes`
    over the Grid and TSDF values.

    The TSDF extraction algorithm is based on the paper
    "GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views"
    (https://arxiv.org/abs/2404.01810). We make key improvements to the method by using a more robust
    stereo baseline estimation method and by using a more efficient TSDF fusion implementation.

    The TSDF fusion algorithm is a method for integrating multiple depth maps into a single volumetric representation of a scene encoding a
    truncated signed distance field (_i.e._ a signed distance field in a narrow band around the surface). TSDF fusion was first described in the paper
    "KinectFusion: Real-Time Dense Surface Mapping and Tracking"
    (https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/).
    We use a modified version of this algorithm which only allocates voxels in a narrow band around the surface of the model
    to reduce memory usage and speed up computation.

    The DLNR model is a high-frequency stereo matching network that computes optical flow and disparity maps
    between two images. The DLNR model is described in the paper "High-Frequency Stereo Matching Network"
    (https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf).

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
        baseline (float): Baseline distance for stereo depth estimation.
            If use_absolute_baseline is False, this is interpreted as a fraction of the mean depth of each image (default is 0.07).
            Otherwise, it is interpreted as an absolute distance in world units.
        near (float): Near plane distance below which to ignore depth samples, as a multiple of the baseline.
        far (float): Far plane distance above which to ignore depth samples, as a multiple of the baseline.
        disparity_reprojection_threshold (float): Reprojection error threshold for occlusion masking in pixels (default is 3.0).
        alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model is transparent
            (usually indicating the background) . Default is 0.1.
        image_downsample_factor (int): Factor by which to downsample the rendered images for depth estimation.
            Default is 1, _i.e._ no downsampling.
        dtype (torch.dtype): Data type for the TSDF grid (default is torch.float16).
        feature_dtype (torch.dtype): Data type for the color features (default is torch.uint8).
        dlnr_backbone (str): Backbone to use for the DLNR model, either "middleburry" or "sceneflow".
        use_absolute_baseline (bool): If True, use the provided baseline as an absolute distance in world units (default is False).
        show_progress (bool): Whether to show a progress bar (default is True).
        num_workers (int): Number of workers to use for loading data generated by DLNR (default is 8).

    Returns:
        mesh_vertices (torch.Tensor): Vertices of the extracted mesh.
        mesh_faces (torch.Tensor): Faces of the extracted mesh.
        mesh_colors (torch.Tensor): Colors of the extracted mesh vertices.
    """

    camera_to_world_matrices, projection_matrices, image_sizes = validate_camera_matrices_and_image_sizes(
        camera_to_world_matrices, projection_matrices, image_sizes
    )
    accum_grid, tsdf, colors = tsdf_from_splats_dlnr(
        model=model,
        camera_to_world_matrices=camera_to_world_matrices,
        projection_matrices=projection_matrices,
        image_sizes=image_sizes,
        truncation_margin=truncation_margin,
        grid_shell_thickness=grid_shell_thickness,
        baseline=baseline,
        near=near,
        far=far,
        disparity_reprojection_threshold=disparity_reprojection_threshold,
        alpha_threshold=alpha_threshold,
        image_downsample_factor=image_downsample_factor,
        dtype=dtype,
        feature_dtype=feature_dtype,
        dlnr_backbone=dlnr_backbone,
        use_absolute_baseline=use_absolute_baseline,
        show_progress=show_progress,
        num_workers=num_workers,
    )

    mesh_vertices, mesh_faces, _ = accum_grid.marching_cubes(tsdf, 0.0)
    mesh_colors = accum_grid.sample_trilinear(mesh_vertices, colors.to(dtype)) / 255.0
    mesh_colors.clip_(min=0.0, max=1.0)

    return mesh_vertices, mesh_faces, mesh_colors
