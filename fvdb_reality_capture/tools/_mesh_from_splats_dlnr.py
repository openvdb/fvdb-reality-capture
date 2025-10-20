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
    Extract a triangle mesh from a :class:`fvdb.GaussianSplat3d` using TSDF fusion from depth maps predicted from the Gaussian splat radiance field and the
    `DLNR foundation model <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf>`_.
    DLNR is a high-frequency stereo matching network that computes optical flow and disparity maps between two images, which can be used to compute depth.

    This algorithm proceeds in three steps:

    1. First, it renders stereo pairs of images from the Gaussian splat radiance field, and uses
       DLNR to compute depth maps from these stereo pairs in the frame of the first image in the pair.
       The result is a set of depth maps aligned with the rendered images.

    2. Second, it integrates the depths and colors/features into a sparse :class:`fvdb.Grid` in a narrow band
       around the surface using sparse truncated signed distance field (TSDF) fusion.
       The result is a sparse voxel grid representation of the scene where each voxel stores a signed distance
       value and color (or other features).

    3. Third, it extracts a mesh using the sparse marching cubes algorithm implemented in :class:`fvdb.Grid.marching_cubes`
       over the Grid and TSDF values. This step produces a triangle mesh with vertex colors sampled from the
       colors/features stored in the Grid.

    .. note::

        If you want to extract the TSDF grid and colors/features without extracting a mesh,
        you can use :func:`fvdb_reality_capture.tools.tsdf_from_splats_dlnr` directly.

    .. note::

        If you want to extract a point cloud from a Gaussian splat model instead of a mesh,
        consider using :func:`fvdb_reality_capture.tools.point_cloud_from_splats` which
        extracts a point cloud directly from depth images rendered from the Gaussian splat model.

    .. note::

        This algorithm implemented is based on the paper
        `"GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views" <https://arxiv.org/abs/2404.01810>`_.
        We make key improvements to the method by using a more robust stereo baseline estimation method and by using a much
        more efficient sparse TSDF fusion implementation built on `fVDB <https://openvdb.github.io/fvdb>`_.

    .. note::

        The TSDF fusion algorithm is a method for integrating multiple depth maps into a single volumetric representation of a scene encoded a
        truncated signed distance field (*i.e.* a signed distance field in a narrow band around the surface).
        TSDF fusion was first described in the paper
        `"KinectFusion: Real-Time Dense Surface Mapping and Tracking" <https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/>`_.
        We use a modified version of this algorithm which only allocates voxels in a narrow band around the surface of the model
        to reduce memory usage and speed up computation.

    .. note::

        The DLNR model is a high-frequency stereo matching network that computes optical flow and disparity maps
        between two images. The DLNR model is described in the paper
        `"High-Frequency Stereo Matching Network" <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf>`_.

    Args:
        model (GaussianSplat3d): The Gaussian splat radiance field to extract a mesh from.
        camera_to_world_matrices (NumericMaxRank3): A ``(C, 4, 4)``-shaped Tensor containing the camera to world
            matrices to render depth images from for mesh extraction where ``C`` is the number of camera views.
        projection_matrices (NumericMaxRank3): A ``(C, 3, 3)``-shaped Tensor containing the perspective projection matrices
            used to render images for mesh extraction where ``C`` is the number of camera views.
        image_sizes (NumericMaxRank2): A ``(C, 2)``-shaped Tensor containing the height and width of each image to extract
            from the Gaussian splat where ``C`` is the number of camera views. *i.e.*, ``image_sizes[c] = (height_c, width_c)``.
        truncation_margin (float): Margin for truncating the TSDF, in world units. This defines the half-width of the band around the surface
            where the TSDF is defined in world units.
        grid_shell_thickness (float): The number of voxels along each axis to include in the TSDF volume.
            This defines the resolution of the Grid around narrow band around the surface.
            Default is 3.0.
        baseline (float): Baseline distance for stereo depth estimation.
            If ``use_absolute_baseline`` is ``False``, this is interpreted as a fraction of
            the mean depth of each image. Otherwise, it is interpreted as an absolute distance in world units.
            Default is 0.07.
        near (float): Near plane distance below which to ignore depth samples, as a multiple of the baseline.
        far (float): Far plane distance above which to ignore depth samples, as a multiple of the baseline.
        disparity_reprojection_threshold (float): Reprojection error threshold for occlusion
            masking (in pixels units). Default is 3.0.
        alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model is transparent
            (usually indicating the background). Default is 0.1.
        image_downsample_factor (int): Factor by which to downsample the rendered images for depth estimation.
            Default is 1, *i.e.* no downsampling.
        dtype (torch.dtype): Data type for the TSDF grid values. Default is ``torch.float16``.
        feature_dtype (torch.dtype): Data type for the color features. Default is ``torch.uint8``.
        dlnr_backbone (str): Backbone to use for the DLNR model, either ``"middleburry"`` or ``"sceneflow"``.
            Default is ``"middleburry"``.
        use_absolute_baseline (bool): If ``True``, treat the provided baseline as an absolute distance in world units.
            If ``False``, treat the baseline as a fraction of the mean depth of each image estimated using the
            Gaussian splat radiance field. Default is ``False``.
        show_progress (bool): Whether to show a progress bar during processing. Default is ``True``.
        num_workers (int): Number of workers to use for loading data generated by DLNR. Default is 8.

    Returns:
        mesh_vertices (torch.Tensor): A ``(V, 3)``-shaped tensor of mesh vertices of the extracted mesh.
        mesh_faces (torch.Tensor): A ``(F, 3)``-shaped tensor of faces of the extracted mesh.
        mesh_colors (torch.Tensor): A ``(V, D)``-shaped tensor of colors of the extracted mesh vertices
            where ``D`` is the number of channels encoded by the Gaussian Splat model (usually 3 for RGB colors).
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
