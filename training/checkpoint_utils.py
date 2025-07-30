# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import uuid
from typing import Sequence

import numpy as np
import torch
import tqdm
from datasets import SfmDataset
from skimage import feature, morphology

from fvdb import GaussianSplat3d, Grid

from .checkpoint import Checkpoint
from .scene_optimization_runner import Config


# TODO: Turn into operation on Checkpoint
def _filter_splat_means(splats, percentile=[0.98, 0.98, 0.98, 0.98, 0.98, 0.98], decimate=4):
    raise NotImplementedError(
        "This function is not used and is here for reference during development. It will be removed soon."
    )
    """
    Remove all gaussians with locations falling outside the provided percentile ranges
    Args:
        splats: dictionary containing splat info to filter
        percentile: drop all splats with locations outside the percentiles (minx, maxx, miny, maxy, minz, maxz)
        decimate: decimate the number of splats by this factor when calculating the percentile range

    Returns:
        dictionary of splats after removal of gaussians outside bounds
    """
    points = splats["means"]

    lower_boundx = torch.quantile(points[::decimate, 0], 1.0 - percentile[0])
    upper_boundx = torch.quantile(points[::decimate, 0], percentile[1])

    lower_boundy = torch.quantile(points[::decimate, 1], 1.0 - percentile[2])
    upper_boundy = torch.quantile(points[::decimate, 1], percentile[3])

    lower_boundz = torch.quantile(points[::decimate, 2], 1.0 - percentile[4])
    upper_boundz = torch.quantile(points[::decimate, 2], percentile[5])

    good_inds = torch.logical_and(points[:, 0] > lower_boundx, points[:, 0] < upper_boundx)
    good_inds = torch.logical_and(good_inds, points[:, 1] > lower_boundy)
    good_inds = torch.logical_and(good_inds, points[:, 1] < upper_boundy)
    good_inds = torch.logical_and(good_inds, points[:, 2] > lower_boundz)
    good_inds = torch.logical_and(good_inds, points[:, 2] < upper_boundz)

    splats["means"] = splats["means"][good_inds, :]
    splats["logit_opacities"] = splats["logit_opacities"][good_inds]
    splats["quats"] = splats["quats"][good_inds, :]
    splats["log_scales"] = splats["log_scales"][good_inds, :]
    splats["sh0"] = splats["sh0"][good_inds, :]
    splats["shN"] = splats["shN"][good_inds, :]
    splats["accumulated_gradient_step_counts_for_grad"] = splats["accumulated_gradient_step_counts_for_grad"][good_inds]
    splats["accumulated_mean_2d_gradient_norms_for_grad"] = splats["accumulated_mean_2d_gradient_norms_for_grad"][
        good_inds
    ]

    return splats


# TODO: Turn into operation on Checkpoint
def _prune_large(splats, prune_scale3d_threshold=0.05):
    raise NotImplementedError(
        "This function is not used and is here for reference during development. It will be removed soon."
    )
    """
    Remove all gaussians with sizes larger than provided percent threshold (relative to scene scale)
    Args:
        splats: dictionary containing splat info to filter
        percentile: drop all spats with opacities outside this percentile
        decimate: decimate the number of splats by this factor when calculating the percentile range

    Returns:
        dictionary of splats after removal of gaussians outside bounds
    """

    points = splats["means"]
    scene_center = torch.mean(points, dim=0)
    dists = torch.linalg.norm(points - scene_center, dim=1)
    scene_scale = torch.max(dists) * 1.1
    good_inds = torch.exp(splats["log_scales"]).max(dim=-1).values < prune_scale3d_threshold * scene_scale

    splats["means"] = splats["means"][good_inds, :]
    splats["logit_opacities"] = splats["logit_opacities"][good_inds]
    splats["quats"] = splats["quats"][good_inds, :]
    splats["log_scales"] = splats["log_scales"][good_inds, :]
    splats["sh0"] = splats["sh0"][good_inds, :]
    splats["shN"] = splats["shN"][good_inds, :]
    splats["accumulated_gradient_step_counts_for_grad"] = splats["accumulated_gradient_step_counts_for_grad"][good_inds]
    splats["accumulated_mean_2d_gradient_norms_for_grad"] = splats["accumulated_mean_2d_gradient_norms_for_grad"][
        good_inds
    ]

    return splats


# TODO: Turn into operation on Checkpoint
def _filter_splat_opacities(splats, percentile=0.98, decimate=4):
    raise NotImplementedError(
        "This function is not used and is here for reference during development. It will be removed soon."
    )
    """
    Remove all gaussians falling outside provided percentile range for logit_opacities.
    Args:
        splats: dictionary containing splat info to filter
        percentile: drop all spats with opacities outside this percentile range
        decimate: decimate the number of splats by this factor when calculating the percentile range

    Returns:
        dictionary of splats after removal of gaussians outside bounds
    """
    lower_bound = torch.quantile(splats["logit_opacities"][::decimate], 1.0 - percentile)
    good_inds = splats["logit_opacities"] > lower_bound

    splats["means"] = splats["means"][good_inds, :]
    splats["logit_opacities"] = splats["logit_opacities"][good_inds]
    splats["quats"] = splats["quats"][good_inds, :]
    splats["log_scales"] = splats["log_scales"][good_inds, :]
    splats["sh0"] = splats["sh0"][good_inds, :]
    splats["shN"] = splats["shN"][good_inds, :]
    splats["accumulated_gradient_step_counts_for_grad"] = splats["accumulated_gradient_step_counts_for_grad"][good_inds]
    splats["accumulated_mean_2d_gradient_norms_for_grad"] = splats["accumulated_mean_2d_gradient_norms_for_grad"][
        good_inds
    ]

    return splats


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
        near (float): Near plane distance below which to ignore depth samples (default is 0.1).
        far (float): Far plane distance above which to ignore depth samples (default is 1e10).
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


def merge_checkpoints(
    checkpoint_list: Sequence[Checkpoint],
    run_name: str | None = None,
    config: Config | None = None,
    train_dataset: SfmDataset | None = None,
    eval_dataset: SfmDataset | None = None,
) -> Checkpoint:
    """
    Merge a list of checkpoints into a single checkpoint.
    The merged checkpoint will contain the splats, camera matrices, and other metadata from all checkpoints.

    Note: If you pass in a train_dataset and eval_dataset, they will be used as the datasets for
        the merged checkpoint. If you pass in None, the merged checkpoint will not contain any
        datasets, and the camera poses will simply be concatenated.

    Note: If you pass in a config, it will be used as the configuration for the merged checkpoint,
        otherwise the checkpoint will have no config.

    Note: If you pass `run_name=None`, the merged checkpoint will have a default run name of
        "merged_UUID" where UUID is a generated UUID.

    Note: The merged checkpoint will not contain the optimizer state, so it is not
        suitable for resuming training.

    Args:
        checkpoint_list (Sequence[Checkpoint]): List of Checkpoint objects to merge.
        run_name (str | None): Name for the merged checkpoint. If None, a default name will be generated
            of the form "merged_UUID" where UUID is a generated UUID.
        config (Config | None): Configuration for the merged checkpoint. If None, the merged checkpoint will not
            contain any configuration.
        train_dataset (SfmDataset | None): Training dataset for the merged checkpoint. If None, the merged
            checkpoint will not contain a training dataset and the dataset metadata (camera poses, etc.) will
            just be the concatenation of the dataset metadata in the input checkpoint list.
        eval_dataset (SfmDataset | None): Evaluation dataset for the merged checkpoint. If None, the merged
            checkpoint will not contain an evaluation dataset and the dataset metadata (camera poses, etc.) will
            just be the concatenation of the dataset metadata in the input checkpoint list.

    Returns:
        Checkpoint: A new Checkpoint object containing the merged data.
    """
    if not checkpoint_list:
        raise ValueError("The checkpoint list is empty.")

    new_splat: GaussianSplat3d = GaussianSplat3d.cat([cp.splats for cp in checkpoint_list])

    if train_dataset is None and eval_dataset is not None:
        raise ValueError("If eval_dataset is provided, train_dataset must also be provided.")
    if train_dataset is not None and eval_dataset is None:
        raise ValueError("If train_dataset is provided, eval_dataset must also be provided.")

    if run_name is None:
        run_name = f"merged_{uuid.uuid4()}"

    if train_dataset is not None and eval_dataset is not None:
        merged_checkpoint = Checkpoint.make_checkpoint(
            step=-1,
            run_name=run_name,
            model=new_splat,
            config=vars(config),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizer=None,
            pose_adjust_model=None,
            pose_adjust_optimizer=None,
            pose_adjust_scheduler=None,
        )
    else:
        camera_origins = torch.stack([cp.camera_to_world_matrices[:, :3, 3] for cp in checkpoint_list], dim=0)
        mean_origin = camera_origins.mean(dim=0, keepdim=True)
        dists = torch.norm(camera_origins - mean_origin, dim=-1)
        scene_scale = torch.max(dists).item()

        final_train_c2w = torch.cat([cp.final_training_poses for cp in checkpoint_list], dim=0)
        train_proj_matrices = torch.cat([cp.training_projection_matrices for cp in checkpoint_list], dim=0)
        train_image_sizes = torch.cat([cp.training_image_sizes for cp in checkpoint_list], dim=0)
        eval_c2w = torch.cat([cp.eval_poses for cp in checkpoint_list], dim=0)
        eval_proj_matrices = torch.cat([cp.eval_projection_matrices for cp in checkpoint_list], dim=0)
        eval_image_sizes = torch.cat([cp.eval_image_sizes for cp in checkpoint_list], dim=0)

        merged_checkpoint = Checkpoint.make_minimal_checkpoint(
            run_name=run_name,
            model=new_splat,
            scene_scale=scene_scale,
            training_camera_to_world_matrices=final_train_c2w,
            training_projection_matrices=train_proj_matrices,
            training_image_sizes=train_image_sizes,
            eval_camera_to_world_matrices=eval_c2w,
            eval_projection_matrices=eval_proj_matrices,
            eval_image_sizes=eval_image_sizes,
        )

    return merged_checkpoint
