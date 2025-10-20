# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch
import tqdm
from fvdb import GaussianSplat3d, Grid
from fvdb.types import (
    NumericMaxRank1,
    NumericMaxRank2,
    NumericMaxRank3,
    is_NumericScalar,
    to_FloatingScalar,
    to_VecNf,
)

from ._common import validate_camera_matrices_and_image_sizes


@torch.no_grad()
def tsdf_from_splats(
    model: GaussianSplat3d,
    camera_to_world_matrices: NumericMaxRank3,
    projection_matrices: NumericMaxRank3,
    image_sizes: NumericMaxRank2,
    truncation_margin: float,
    grid_shell_thickness: float = 3.0,
    near: NumericMaxRank1 = 0.1,
    far: NumericMaxRank1 = 1e10,
    alpha_threshold: float = 0.1,
    image_downsample_factor: int = 1,
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    show_progress: bool = True,
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """
    Extract a Truncated Signed Distance Field (TSDF) from a :class:`fvdb.GaussianSplat3d` using TSDF fusion
    from depth maps rendered from the Gaussian splat model.

    The algorithm proceeds in two steps:

    1. First, it renders depth and color/feature images from the Gaussian splat radiance field at each of the specified
       camera views.

    2. Second, it integrates the depths and colors/features into a sparse :class:`fvdb.Grid` in a narrow band
       around the surface using sparse truncated signed distance field (TSDF) fusion.
       The result is a sparse voxel grid representation of the scene where each voxel stores a signed distance
       value and color (or other features).


    .. note::

        You can extract a mesh from the TSDF using the marching cubes algorithm implemented in
        :class:`fvdb.Grid.marching_cubes`. If your goal is to extract a mesh from a Gaussian splat model,
        consider using :func:`fvdb_reality_capture.tools.mesh_from_splats_dlnr` which combines this function
        with marching cubes to directly extract a mesh.

    .. note::

        For higher quality TSDFs, consider using :func:`fvdb_reality_capture.tools.tsdf_from_splats_dlnr`
        which uses depth maps estimated using a deep learning-based depth estimation approach instead of the raw
        depth maps rendered from the Gaussian splat model.

    .. note::

        The TSDF fusion algorithm is a method for integrating multiple depth maps into a single volumetric representation of a scene encoded a
        truncated signed distance field (*i.e.* a signed distance field in a narrow band around the surface).
        TSDF fusion was first described in the paper
        `"KinectFusion: Real-Time Dense Surface Mapping and Tracking" <https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/>`_.
        We use a modified version of this algorithm which only allocates voxels in a narrow band around the surface of the model
        to reduce memory usage and speed up computation.


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
        near (NumericMaxRank1): Near plane distance below which to ignore depth samples. Can be a scalar to use a
            single value for all images or a tensor-like object of shape ``(C,)`` to use a different value for each
            image. Default is 0.1.
        far (NumericMaxRank1): Far plane distance above which to ignore depth samples. Can be a scalar to use a
            single value for all images or a tensor-like object of shape ``(C,)`` to use a different value for each
            image. Default is 1e10.
        alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model is transparent
            (usually indicating the background). Default is 0.1.
        image_downsample_factor (int): Factor by which to downsample the rendered images for depth estimation.
            Default is 1, *i.e.* no downsampling.
        dtype (torch.dtype): Data type for the TSDF grid values. Default is ``torch.float16``.
        feature_dtype (torch.dtype): Data type for the color features. Default is ``torch.uint8``.
        show_progress (bool): Whether to show a progress bar during processing. Default is ``True``.

    Returns:
        accum_grid (Grid): The accumulated :class:`fvdb.Grid` representing the voxels in the TSDF volume.
        tsdf (torch.Tensor): The TSDF values for each voxel in the grid.
        colors (torch.Tensor): The colors/features for each voxel in the grid.
    """

    if grid_shell_thickness <= 1.0:
        raise ValueError("grid_shell_thickness must be greater than 1.0")

    device = model.device

    camera_to_world_matrices, projection_matrices, image_sizes = validate_camera_matrices_and_image_sizes(
        camera_to_world_matrices, projection_matrices, image_sizes
    )

    voxel_size = truncation_margin / grid_shell_thickness
    accum_grid = Grid.from_zero_voxels(voxel_size=voxel_size, origin=0.0, device=model.device)
    tsdf = torch.zeros(accum_grid.num_voxels, device=model.device, dtype=dtype)
    weights = torch.zeros(accum_grid.num_voxels, device=model.device, dtype=dtype)
    features = torch.zeros((accum_grid.num_voxels, model.num_channels), device=model.device, dtype=feature_dtype)

    num_cameras = len(camera_to_world_matrices)
    enumerator = (
        tqdm.tqdm(range(num_cameras), unit="imgs", desc="Extracting TSDF") if show_progress else range(num_cameras)
    )

    if image_downsample_factor > 1:
        image_sizes = image_sizes // image_downsample_factor
        projection_matrices = projection_matrices.clone()
        projection_matrices[:, :2, :] /= image_downsample_factor

    # You can pass in per-image near and far planes as tensors of shape (C,)
    # or single scalar values to use the same near and far planes for all images.
    # We convert these to the appropriate types here (either a tensor of shape (C,) or a tensor with a single value).
    if is_NumericScalar(near):
        near = to_FloatingScalar(near)
        near_is_scalar = True
    else:
        near = to_VecNf(near, num_cameras)
        near_is_scalar = False

    if is_NumericScalar(far):
        far = to_FloatingScalar(far)
        far_is_scalar = True
    else:
        far = to_VecNf(far, num_cameras)
        far_is_scalar = False

    for i in enumerator:
        cam_to_world_matrix = camera_to_world_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        world_to_cam_matrix = torch.linalg.inv(cam_to_world_matrix).contiguous().to(dtype=torch.float32, device=device)
        projection_matrix = projection_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        image_size = image_sizes[i]

        # We set near and far planes to 0.0 and 1e10 respectively to avoid clipping
        # in the rendering process. Instead, we will use the provided near and far planes
        # to filter the depth images after rendering so pixels out of range will not be integrated
        # into the TSDF.
        feature_and_depth, alpha = model.render_images_and_depths(
            world_to_camera_matrices=world_to_cam_matrix.unsqueeze(0),
            projection_matrices=projection_matrix.unsqueeze(0),
            image_width=int(image_size[1].item()),
            image_height=int(image_size[0].item()),
            near=0.0,
            far=1e10,
        )

        if feature_dtype == torch.uint8:
            feature_images = (feature_and_depth[..., : model.num_channels].clip_(min=0.0, max=1.0) * 255.0).to(
                feature_dtype
            )
        else:
            feature_images = feature_and_depth[..., : model.num_channels].to(feature_dtype)

        # Get the near and far planes for this image.
        near_i = near.item() if near_is_scalar else near[i]
        far_i = far.item() if far_is_scalar else far[i]

        alpha = alpha[0].clamp(min=1e-10).squeeze(-1)
        feature_images = feature_images.squeeze(0)
        depth_images = (feature_and_depth[0, ..., -1] / alpha).to(dtype)
        if alpha_threshold > 0.0:
            alpha_mask = alpha > alpha_threshold
            weight_images = ((depth_images > near_i) & (depth_images < far_i) & alpha_mask).to(dtype).squeeze(0)
        else:
            weight_images = ((depth_images > near_i) & (depth_images < far_i)).to(dtype).squeeze(0)
        accum_grid, tsdf, weights, features = accum_grid.integrate_tsdf_with_features(
            truncation_margin,
            projection_matrix.to(dtype),
            cam_to_world_matrix.to(dtype),
            tsdf,
            features,
            weights,
            depth_images,
            feature_images,
            weight_images,
        )

        if show_progress:
            assert isinstance(enumerator, tqdm.tqdm)
            enumerator.set_postfix({"accumulated_voxels": accum_grid.num_voxels})

        # TSDF fusion is a bit of a torture case for the PyTorch memory allocator since
        # it progressively allocates bigger tensors which don't fit in the memory pool,
        # causing the pool to grow larger and larger.
        # To avoid this, we synchronize the CUDA device and empty the cache after each image.
        del feature_images, depth_images, weight_images
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # After integrating all the images, we prune the grid to remove empty voxels which have no weights.
    # This is done to reduce the size of the grid and speed up the marching cubes algorithm
    # which will be used to extract the mesh.
    new_grid = accum_grid.pruned_grid(weights > 0.0)
    filter_tsdf = new_grid.inject_from(accum_grid, tsdf)
    filter_colors = new_grid.inject_from(accum_grid, features)

    return new_grid, filter_tsdf, filter_colors
