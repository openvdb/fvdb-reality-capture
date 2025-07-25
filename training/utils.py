# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch
import tqdm
from skimage import feature, morphology

from fvdb import GaussianSplat3d, Grid

from .checkpoint import Checkpoint


@torch.no_grad()
def extract_tsdf_from_checkpoint(
    checkpoint: Checkpoint,
    trunctation_margin: float,
    near: float = 0.1,
    far: float = 1e10,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    show_progress: bool = True,
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """
    Extract a TSDF using TSDF fusion from depth maps rendered from a Gaussian splat.
    The TSDF fusion algorithm is based on the paper:
    "KinectFusion: Real-Time Dense Surface Mapping and Tracking"
    (https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/)

    Args:
        checkpoint (Checkpoint): A checkpoint containing the Gaussian splat model and camera parameters.
        truncation_margin (float): Margin for truncating the TSDF, in world units.
        near (float): Near plane distance below which we'll ignore depth samples (default is 0.0).
        far (float): Far plane distance above which we'll ignore depth samples (default is 1e10).
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

    voxel_size = trunctation_margin / 2.0
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
            trunctation_margin,
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
    Extract a mesh from a Gaussian splat using TSDF fusion and marching cubes.
    The TSDF fusion algorithm is based on the paper:
    "KinectFusion: Real-Time Dense Surface Mapping and Tracking"
    (https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/)

    Args:
        checkpoint (Checkpoint): A checkpoint containing the Gaussian splat model and camera parameters.
        truncation_margin (float): Margin for truncating the TSDF, in world units.
        near (float): Near plane distance below which we'll ignore depth samples (default is 0.0).
        far (float): Far plane distance above which we'll ignore depth samples (default is 1e10).
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
def extract_point_cloud_from_checkpoint(
    checkpoint: Checkpoint,
    near: float = 0.1,
    far: float = 1e10,
    depth_image_downsample_factor: int = 1,
    canny_edge_std: float = 1.0,
    canny_mask_dilation: int = 5,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cuda",
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a point cloud from a Gaussian splat using depth rendering, possibly filtering points
    using Canny edge detection on the depth images.

    Args:
        checkpoint (Checkpoint): A checkpoint containing the Gaussian splat model and camera parameters.
        near (float): Near plane distance below which we'll ignore depth samples (default is 0.1).
        far (float): Far plane distance above which we'll ignore depth samples (default is 1e10).
        depth_image_downsample_factor (int): Factor by which to downsample the depth images before extracting points
            (default is 1, no downsampling). This is useful to reduce the number of points extracted from the point cloud
            and speed up the extraction process. A value of 2 will downsample the depth images by a factor of 2 in both dimensions,
            resulting in a point cloud with approximately 1/4 the number of points compared to the original depth images.
        quantization (float): Quantization step for the point cloud (default is 0.0, no quantization).
        canny_edge_std (float): Standard deviation for the Gaussian filter applied to the depth image
            before Canny edge detection (default is 1.0). Set to 0.0 to disable canny edge filtering.
        canny_mask_dilation (int): Dilation size for the Canny edge mask (default is 5).
        dtype (torch.dtype): Data type for the point cloud and colors (default is torch.float16).
        device (torch.device | str): Device to use (default is "cuda").
        show_progress (bool): Whether to show a progress bar (default is True).

    Returns:
        points (torch.Tensor): A [num_points, 3] shaped tensor of points in camera space.
        colors (torch.Tensor): A [num_points, 3] shaped tensor of RGB colors for the points.
    """

    model = checkpoint.splats.to(device)
    camera_to_world_matrices = checkpoint.camera_to_world_matrices
    projection_matrices = checkpoint.projection_matrices
    image_sizes = checkpoint.image_sizes

    if camera_to_world_matrices is None:
        raise ValueError("Camera to world matrices are not available in the checkpoint.")
    if projection_matrices is None:
        raise ValueError("Projection matrices are not available in the checkpoint.")
    if image_sizes is None:
        raise ValueError("Image sizes are not available in the checkpoint.")

    points_list = []
    colors_list = []

    enumerator = (
        tqdm.tqdm(range(len(camera_to_world_matrices)), unit="imgs", desc="Extracting Point Cloud")
        if show_progress
        else range(len(camera_to_world_matrices))
    )

    total_points = 0
    for i in enumerator:
        cam_to_world_matrix = camera_to_world_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        world_to_cam_matrix = torch.linalg.inv(cam_to_world_matrix).contiguous()
        projection_matrix = projection_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        inv_projection_matrix = torch.linalg.inv(projection_matrix).contiguous()

        image_size = image_sizes[i]
        image_width = int(image_size[1].item())
        image_height = int(image_size[0].item())

        # We set near and far planes to 0.0 and 1e10 respectively to avoid clipping
        # in the rendering process. Instead, we will use the provided near and far planes
        # to filter the depth images after rendering so pixels out of range will not be accumulated
        feature_and_depth, alpha = model.render_images_and_depths(
            world_to_camera_matrices=world_to_cam_matrix.unsqueeze(0),
            projection_matrices=projection_matrix.unsqueeze(0),
            image_width=image_width,
            image_height=image_height,
            near=0.0,
            far=1e10,
        )

        feature_image = feature_and_depth[..., : model.num_channels].squeeze(0)
        depth_image = (feature_and_depth[..., -1].unsqueeze(-1) / alpha.clamp(min=1e-10)).squeeze()  # [H, W]

        assert feature_image.shape == (image_height, image_width, model.num_channels)
        assert depth_image.shape == (image_height, image_width)

        mask = ((depth_image > near) & (depth_image < far)).squeeze(-1)  # [H, W]
        # TODO: Add GPU Canny edge detection
        if canny_edge_std > 0.0:
            canny_mask = torch.tensor(
                morphology.dilation(
                    feature.canny(depth_image.squeeze(-1).cpu().numpy(), sigma=canny_edge_std),
                    footprint=np.ones((canny_mask_dilation, canny_mask_dilation)),
                )
                == 0,
                device=device,
            )
            mask = mask & canny_mask

        # Unproject depth image to camera space coordinates
        row, col = torch.meshgrid(
            torch.arange(0, image_height, device=device, dtype=torch.float32),
            torch.arange(0, image_width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        cam_pts = torch.stack([col, row, torch.ones_like(row)])  # [3, H, W]
        cam_pts = inv_projection_matrix @ cam_pts.view(3, -1)  # [3, H, W]
        cam_pts = cam_pts.view(3, image_height, image_width) * depth_image.unsqueeze(0)  # [3, H, W]

        # Transform camera space coordinates to world coordinates
        world_pts = torch.cat(
            [cam_pts, torch.ones(1, cam_pts.shape[1], cam_pts.shape[2]).to(cam_pts)], dim=0
        )  # [4, H, W]
        world_pts = cam_to_world_matrix @ world_pts.view(4, -1)  # [4, H, W]
        world_pts = world_pts[:3] / world_pts[3].unsqueeze(0)  # [3, H * W]
        world_pts = world_pts.view(3, image_height, image_width).permute(1, 2, 0)  # [H, W, 3]

        # Optionally downsample the world points and feature image
        world_pts = world_pts[::depth_image_downsample_factor, ::depth_image_downsample_factor, :]
        feature_image = feature_image[::depth_image_downsample_factor, ::depth_image_downsample_factor, :]
        mask = mask[::depth_image_downsample_factor, ::depth_image_downsample_factor]

        world_pts = world_pts[mask].view(-1, 3)  # [num_points, 3]
        features = feature_image[mask]  # [num_points, C]

        if world_pts.numel() == 0:
            continue

        assert world_pts.shape[0] == features.shape[0], "Number of points and features must match."

        if show_progress:
            assert isinstance(enumerator, tqdm.tqdm)
            enumerator.set_postfix({"total_points": total_points})

        points_list.append(world_pts.to(dtype))
        colors_list.append(features.to(dtype))
        total_points += points_list[-1].shape[0]

    return torch.cat(points_list, dim=0).to(dtype), torch.cat(colors_list, dim=0).to(dtype).clip_(min=0.0, max=1.0)
