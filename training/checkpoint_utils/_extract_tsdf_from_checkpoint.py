# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch
import tqdm

from fvdb import GaussianSplat3d, Grid

from ..checkpoint import Checkpoint


@torch.no_grad()
def extract_tsdf_from_checkpoint(
    checkpoint: Checkpoint,
    truncation_margin: float,
    near: float = 0.1,
    far: float = 1e10,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    show_progress: bool = True,
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """
    Extract a Truncated Signed Distance Field (TSDF) using TSDF fusion from depth maps rendered from a Gaussian splat.
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
        grid (Grid): A Grid object encoding the topology of the TSDF.
        tsdf (torch.Tensor): A tensor of TSDF values indexed by the grid.
        features (torch.Tensor): A tensor of features (e.g., colors) indexed by the grid.
    """

    model: GaussianSplat3d = checkpoint.splats.to(device)

    camera_to_world_matrices = checkpoint.camera_to_world_matrices
    projection_matrices = checkpoint.projection_matrices
    image_sizes = checkpoint.image_sizes

    if camera_to_world_matrices is None:
        raise ValueError("Camera to world matrices are not available in the checkpoint.")
    if projection_matrices is None:
        raise ValueError("Projection matrices are not available in the checkpoint.")
    if image_sizes is None:
        raise ValueError("Image sizes are not available in the checkpoint.")

    voxel_size = truncation_margin / 2.0
    accum_grid = Grid.from_dense(dense_dims=1, ijk_min=0, voxel_size=voxel_size, origin=0.0, device=model.device)
    tsdf = torch.zeros(accum_grid.num_voxels, device=model.device, dtype=dtype)
    weights = torch.zeros(accum_grid.num_voxels, device=model.device, dtype=dtype)
    features = torch.zeros((accum_grid.num_voxels, model.num_channels), device=model.device, dtype=feature_dtype)

    enumerator = (
        tqdm.tqdm(range(len(camera_to_world_matrices)), unit="imgs", desc="Extracting TSDF")
        if show_progress
        else range(len(camera_to_world_matrices))
    )

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
        feature_images = feature_images.squeeze(0)
        depth_images = (feature_and_depth[..., -1].unsqueeze(-1) / alpha.clamp(min=1e-10)).to(dtype).squeeze(0)
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
