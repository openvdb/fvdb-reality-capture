# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch
import tqdm
from fvdb import GaussianSplat3d, Grid
from fvdb.types import NumericMaxRank2, NumericMaxRank3

from ._common import validate_camera_matrices_and_image_sizes


@torch.no_grad()
def tsdf_from_splats(
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
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """
    Extract a Truncated Signed Distance Field (TSDF) from a `fvdb.GaussianSplat3d` using TSDF fusion
    from depth maps rendered from the Gaussian splat model.

    In short, this algorithm works by rendering images and depth maps from multiple views of the Gaussian splat model,
    and then integrating these depth maps and images into a sparse `fvdb.Grid` in a narrow band around the surface using a weighted averaging scheme.
    The algorithm returns this grid along with signed distance values and colors (or other features) at each voxel.

    A mesh can be extracted from the TSDF using the marching cubes algorithm implemented in `fvdb.marching_cubes.marching_cubes`.

    The TSDF fusion algorithm is a method for integrating multiple depth maps into a single volumetric representation of a scene encoding a
    truncated signed distance field (_i.e._ a signed distance field in a narrow band around the surface). TSDF fusion was first described in the paper
    "KinectFusion: Real-Time Dense Surface Mapping and Tracking"
    (https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/).
    We use a modified version of this algorithm which only allocates voxels in a narrow band around the surface of the model
    to reduce memory usage and speed up computation.

    Args:
        model (GaussianSplat3d): The Gaussian splat model to extract a mesh from
        camera_to_world_matrices (torch.Tensor): A (C, 4, 4)-shaped Tensor containing the camera to world
            matrices to render depth images from for mesh extraction where C is the number of camera views.
        projection_matrices (torch.Tensor): A (C, 3, 3)-shaped Tensor containing the perspective projection matrices
            used to render images for mesh extraction where C is the number of camera views.
        image_sizes (NumericMaxRank2): A (C, 2)-shaped Tensor containing the height and width of each image to extract
            from the Gaussian splat where C is the number of camera views.
        truncation_margin (float): Margin for truncating the TSDF, in world units.
        grid_shell_thickness (float): Thickness of the TSDF grid shell in multiples of the truncation margin (default is 3.0).
            _i.e_. if truncation_margin is 0.1 and grid_shell_thickness is 3.0, the TSDF grid will extend 0.3 world units
            from the surface of the model. This value must be greater than 1.0.
        near (float): Near plane distance below which to ignore depth samples (default is 0.0).
        far (float): Far plane distance above which to ignore depth samples (default is 1e10).
        alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model
            is transparent, which usually indicates the pixel is part of the background. (default is 0.1).
        image_downsample_factor (int): Factor by which to downsample the rendered images for depth estimation.
            A downsample factor of N means the rendered images will be 1/N the width and height of the input image size.
            (default is 1, _i.e._ no downsampling).
        dtype: Data type for the TSDF and weights. Default is torch.float16.
        feature_dtype: Data type for the features (default is torch.uint8 which is good for RGB colors).
        show_progress (bool): Whether to show a progress bar (default is True).

    Returns:
        grid (Grid): A Grid object encoding the topology of the TSDF.
        tsdf (torch.Tensor): A tensor of TSDF values indexed by the grid.
        features (torch.Tensor): A tensor of features (e.g., colors) indexed by the grid.
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

    enumerator = (
        tqdm.tqdm(range(len(camera_to_world_matrices)), unit="imgs", desc="Extracting TSDF")
        if show_progress
        else range(len(camera_to_world_matrices))
    )

    if image_downsample_factor > 1:
        image_sizes = image_sizes // image_downsample_factor
        projection_matrices = projection_matrices.clone()
        projection_matrices[:, :2, :] /= image_downsample_factor

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

        alpha = alpha[0].clamp(min=1e-10).squeeze(-1)
        feature_images = feature_images.squeeze(0)
        depth_images = (feature_and_depth[0, ..., -1] / alpha).to(dtype)
        if alpha_threshold > 0.0:
            alpha_mask = alpha > alpha_threshold
            weight_images = ((depth_images > near) & (depth_images < far) & alpha_mask).to(dtype).squeeze(0)
        else:
            weight_images = ((depth_images > near) & (depth_images < far)).to(dtype).squeeze(0)
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
