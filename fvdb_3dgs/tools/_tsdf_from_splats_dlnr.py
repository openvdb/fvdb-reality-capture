# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import torch
import tqdm

from fvdb import GaussianSplat3d, Grid

from ..foundation_models.dlnr import DLNRModel


def debug_plot(
    disparity_l2r: torch.Tensor,
    disparity_r2l: torch.Tensor,
    depth: torch.Tensor,
    image_l: torch.Tensor,
    image_r: torch.Tensor,
    occlusion_mask: torch.Tensor,
    out_filename: str,
) -> None:
    """
    Debug plotting. Plots the disparity maps, depth map, left and right images,
    and the occlusion mask.

    Args:
        disparity_l2r (torch.Tensor): Left-to-right disparity map.
        disparity_r2l (torch.Tensor): Right-to-left disparity map.
        depth (torch.Tensor): Depth map.
        image_l (torch.Tensor): Left image.
        image_r (torch.Tensor): Right image.
        occlusion_mask (torch.Tensor): Occlusion mask.
        out_filename (str): Output filename for the plot.
    """
    import cv2
    import matplotlib.pyplot as plt

    depth_np = depth.squeeze().cpu().numpy()
    occlusion_mask_np = occlusion_mask.cpu().numpy()

    # Shade the depth map by 1 / norm(gradient(depth_np) + shading_eps)
    shading_eps = 1e-6
    g_x = cv2.Sobel(depth_np, cv2.CV_64F, 1, 0)
    g_y = cv2.Sobel(depth_np, cv2.CV_64F, 0, 1)
    shading = 1 / (np.sqrt((g_x**2) + (g_y**2) + shading_eps))
    shading[~occlusion_mask_np] = shading.max()  # Highlight occluded areas

    depth_np[~occlusion_mask_np] = depth_np.max()  # Highlight occluded areas in depth map

    plt.figure(figsize=(10, 20))
    plt.subplot(4, 2, 1)
    plt.title("Depth Map")
    plt.imshow(depth_np, cmap="turbo")
    plt.colorbar()

    plt.subplot(4, 2, 2)
    plt.title("Shaded Depth Map")
    plt.imshow(shading, cmap="turbo")

    plt.subplot(4, 2, 3)
    plt.title("Disparity L2R")
    plt.imshow(disparity_l2r.squeeze().cpu().numpy(), cmap="jet")
    plt.colorbar()

    plt.subplot(4, 2, 4)
    plt.title("Disparity R2L")
    plt.imshow(disparity_r2l.squeeze().cpu().numpy(), cmap="jet")
    plt.colorbar()

    plt.subplot(4, 2, 5)
    plt.title("Left Image")
    plt.imshow(image_l.squeeze().cpu().numpy() / 255.0)

    plt.subplot(4, 2, 6)
    plt.title("Right Image")
    plt.imshow(image_r.squeeze().cpu().numpy() / 255.0)

    plt.savefig(out_filename, bbox_inches="tight")
    plt.close()


def render_stereo_pair(
    model: GaussianSplat3d,
    baseline: float,
    world_to_camera_matrix: torch.Tensor,
    projection_matrix: torch.Tensor,
    image_width: int,
    image_height: int,
):
    """
    Render a pair of stereo images from a Gaussian Splat model.

    The pair of images is rendered by shifting the camera position by a baseline distance along
    the camera's -x axis to simulate stereo vision.

    Args:
        model (GaussianSplat3d): The Gaussian Splat model to render from.
        baseline (float): The distance between the two camera positions along the camera -x axis.
        world_to_camera_matrix (torch.Tensor): The camera_to_world transformation matrix for the first
            image in the stereo pair. The second image has the same transformation but with the x position
            shifted by the negative baseline.
        projection_matrix (torch.Tensor): The projection matrix for the camera.
        image_width (int): The width of the rendered images.
        image_height (int): The height of the rendered images.

    Returns:
        image_1 (torch.Tensor): The first rendered image whose camera to world matrix is the same as the input.
        image_2 (torch.Tensor): The second rendered image whose camera to world matrix is the same as the input but
            with the x position shifted by the negative baseline.
    """
    # Compute the left and right camera poses
    world_to_camera_matrix_left = world_to_camera_matrix.clone()
    world_to_camera_matrix_right = world_to_camera_matrix.clone()
    world_to_camera_matrix_right[0, 3] -= baseline

    world_to_camera_matrix = torch.stack([world_to_camera_matrix_left, world_to_camera_matrix_right], dim=0)
    projection_matrix = torch.stack([projection_matrix, projection_matrix], dim=0)

    images, _ = model.render_images(
        world_to_camera_matrices=world_to_camera_matrix,
        projection_matrices=projection_matrix,
        image_width=image_width,
        image_height=image_height,
        near=0.0,
        far=1e10,
    )
    return images[0], images[1]


def compute_occlusion_mask(l2r_disparity, r2l_disparity, reprojection_threshold):
    """
    Compute an occlusion mask using the disparity maps by filtering pixels where the
    reprojection error exceeds the reprojection threshold.

    Given a point in space, and a stereo pair of images, disparity maps are computed as the
    difference in pixel coordinates between the projection of that point in the left and right images.

    The occlusion mask is computed by using the left-to-right disparity map to project pixels from the left image
    to the right image, and then using the right-to-left disparity map to reproject those pixels back to the left image.
    If the reprojection error exceeds the reprojection threshold, the pixel is considered occluded.

    Args:
        l2r_disparity (np.ndarray): Left-to-right disparity map.
        r2l_disparity (np.ndarray): Right-to-left disparity map.
        reprojection_threshold (int): Threshold on the reprojection error.

    Returns:
        torch.Tensor: Binary occlusion mask where 0 indicates occluded pixels and 1 indicates visible pixels.
    """

    height, width = l2r_disparity.shape

    x_values = torch.arange(width, device=l2r_disparity.device)
    y_values = torch.arange(height, device=l2r_disparity.device)
    x_grid, y_grid = torch.meshgrid(x_values, y_values, indexing="xy")

    x_projected = (x_grid - l2r_disparity).to(torch.int32)
    x_projected_clipped = torch.clamp(x_projected, 0, width - 1)

    x_reprojected = x_projected_clipped + r2l_disparity[y_grid, x_projected_clipped]
    x_reprojected_clipped = torch.clamp(x_reprojected, 0, width - 1)

    disparity_difference = torch.abs(x_grid - x_reprojected_clipped)

    occlusion_mask = disparity_difference > reprojection_threshold

    occlusion_mask[(x_projected < 0) | (x_projected >= width)] = True

    return ~occlusion_mask


def compute_disparities_and_depth(
    image_l: torch.Tensor,  # [H, W, C]
    image_r: torch.Tensor,  # [H, W, C]
    dlnr_model: DLNRModel,
    projection_matrix: torch.Tensor,  # [3, 3]
    baseline: float,
):
    _, disparity_l2r = dlnr_model.predict_flow(
        images1=image_l.unsqueeze(0),  # [1, H, W, C]
        images2=image_r.unsqueeze(0),  # [1, H, W, C]
        flow_init=None,
    )
    disparity_l2r = -disparity_l2r[0]  # [H, W]

    image_l_flip = torch.flip(image_l, dims=[1])
    image_r_flip = torch.flip(image_r, dims=[1])
    _, disparity_r2l = dlnr_model.predict_flow(
        images1=image_r_flip.unsqueeze(0),  # [1, H, W, C]
        images2=image_l_flip.unsqueeze(0),  # [1, H, W, C]
        flow_init=None,
    )
    disparity_r2l = -torch.flip(disparity_r2l[0], dims=[1])  # [H, W]

    fx = projection_matrix[0, 0].item()
    depth = (fx * baseline) / disparity_l2r

    return disparity_l2r, disparity_r2l, depth


def get_images_depth_and_weights(
    model: GaussianSplat3d,
    baseline: float,
    world_to_cam_matrix: torch.Tensor,
    projection_matrix: torch.Tensor,
    image_width: int,
    image_height: int,
    near: float,
    far: float,
    dlnr_model: DLNRModel,
    save_debug_images_to: str | None = None,
):
    """
    Compute rgb images, depths, and weights for TSDF fusion using a Gaussian splat model for images
    and DLNR for depth, and occlusion masking. This algorithm is roughly based on the GS2Mesh algorithm
    described in https://arxiv.org/abs/2404.01810.

    The alorithm renders a stereo pair of images from the Gaussian splat model, computes disparities
    using DLNR, computes an occlusion mask based on the disparities, and then computes a
    near/far mask based on the depth. The final weights are a combination of the near/far mask and the occlusion mask.

    Args:
        model (GaussianSplat3d): The Gaussian splat model to render from.
        baseline (float): The distance between the two camera positions along the camera -x axis.
        world_to_cam_matrix (torch.Tensor): The camera_to_world transformation matrix for the first
            image in the stereo pair. The second image has the same transformation but with the x position
            shifted by the negative baseline.
        projection_matrix (torch.Tensor): The projection matrix for the camera.
        image_width (int): The width of the rendered images.
        image_height (int): The height of the rendered images.
        near (float): Near plane distance below which to ignore depth samples.
        far (float): Far plane distance above which to ignore depth samples.
            Units are in multiples of the scene scale (variance in distance from camera positions around their mean).
        dlnr_model (DLNRModel): The DLNR model to compute optical flow and disparity.
        save_debug_images_to (str | None): If provided, saves debug images to this path.

    """
    # Render the stereo pair of images and clip to [0, 1]
    image_l, image_r = render_stereo_pair(
        model, baseline, world_to_cam_matrix, projection_matrix, image_width, image_height
    )
    image_l.clip_(min=0.0, max=1.0)
    image_r.clip_(min=0.0, max=1.0)

    # Compute left-to-right and right-to-left disparities and depth using DLNR
    disparity_l2r, disparity_r2l, depth = compute_disparities_and_depth(
        image_l=image_l,
        image_r=image_r,
        dlnr_model=dlnr_model,
        projection_matrix=projection_matrix,
        baseline=baseline,
    )

    # Compute an occlusion mask based on the reprojection error of the disparities
    occlusion_mask = compute_occlusion_mask(
        disparity_l2r,
        disparity_r2l,
        reprojection_threshold=3.0,
    )

    # Create masks using the near and far values
    near_far_mask = (depth > near) & (depth < far)

    # The final weights are a combination of the near/far mask and the occlusion mask
    weights = near_far_mask & occlusion_mask

    if save_debug_images_to is not None:
        debug_plot(
            disparity_l2r=disparity_l2r,
            disparity_r2l=disparity_r2l,
            depth=depth,
            image_l=image_l,
            image_r=image_r,
            occlusion_mask=occlusion_mask,
            out_filename=save_debug_images_to,
        )

    return image_l, depth, weights


@torch.no_grad()
def tsdf_from_splats_dlnr(
    model: GaussianSplat3d,
    camera_to_world_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    image_sizes: torch.Tensor,
    truncation_margin: float,
    baseline: float = 0.1,
    near: float = 0.1,
    far: float = 1e10,
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    dlnr_backbone: str = "middleburry",
    show_progress: bool = True,
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """
    Extract a TSDF grid from a checkpoint using DLNR for depth estimation.

    Args:
        model (GaussianSplat3d): The Gaussian splat model to extract a mesh from
        camera_to_world_matrices (torch.Tensor): A (C, 4, 4)-shaped Tensor containing the camera to world
            matrices to render depth images from for mesh extraction where C is the number of camera views.
        projection_matrices (torch.Tensor): A (C, 3, 3)-shaped Tensor containing the perspective projection matrices
            used to render images for mesh extraction where C is the number of camera views.
        image_sizes (torch.Tensor): A (C, 2)-shaped Tensor containing the width and height of each image to extract
            from the Gaussian splat where C is the number of camera views.
        truncation_margin (float): Margin for truncating the TSDF, in world units.
        baseline (float): Baseline distance for stereo depth estimation, in world units.
        near (float): Near plane distance below which to ignore depth samples, in world units.
        far (float): Far plane distance above which to ignore depth samples, in world units.
        dtype (torch.dtype): Data type for the TSDF grid (default is torch.float16).
        feature_dtype (torch.dtype): Data type for the color features (default is torch.uint8).
        dlnr_backbone (str): Backbone to use for the DLNR model, either "middleburry" or "sceneflow".
        show_progress (bool): Whether to show a progress bar (default is True).

    Returns:
        accum_grid (Grid): The accumulated grid containing the TSDF.
        tsdf (torch.Tensor): The TSDF values in the grid.
        colors (torch.Tensor): The color features in the grid.
    """

    if model.num_channels != 3:
        raise ValueError(f"Expected model with 3 channels, got {model.num_channels} channels.")

    device = model.device

    voxel_size = truncation_margin / 2.0
    accum_grid = Grid.from_dense(dense_dims=1, ijk_min=0, voxel_size=voxel_size, origin=0.0, device=model.device)
    tsdf = torch.zeros(accum_grid.num_voxels, device=model.device, dtype=dtype)
    weights = torch.zeros(accum_grid.num_voxels, device=model.device, dtype=dtype)
    colors = torch.zeros((accum_grid.num_voxels, model.num_channels), device=model.device, dtype=feature_dtype)

    enumerator = (
        tqdm.tqdm(range(len(camera_to_world_matrices)), unit="imgs", desc="Extracting TSDF")
        if show_progress
        else range(len(camera_to_world_matrices))
    )

    dlnr_model = DLNRModel(backbone=dlnr_backbone, device=model.device)

    for i in enumerator:
        cam_to_world_matrix = camera_to_world_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        world_to_cam_matrix = torch.linalg.inv(cam_to_world_matrix).contiguous().to(dtype=torch.float32, device=device)
        projection_matrix = projection_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        image_height, image_width = int(image_sizes[i][0].item()), int(image_sizes[i][1].item())

        rgb_image, depth_image, weight_image = get_images_depth_and_weights(
            model=model,
            baseline=baseline,
            world_to_cam_matrix=world_to_cam_matrix,
            projection_matrix=projection_matrix,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            dlnr_model=dlnr_model,
            save_debug_images_to=None,  # Set to a path if you want to save debug images
        )
        if feature_dtype == torch.uint8:
            rgb_image = (rgb_image * 255).to(feature_dtype)
        else:
            rgb_image = rgb_image.to(feature_dtype)
        depth_image = depth_image.to(dtype)
        weight_image = weight_image.to(dtype)

        accum_grid, tsdf, weights, colors = accum_grid.integrate_tsdf_with_features(
            truncation_margin,
            projection_matrix.to(dtype),
            cam_to_world_matrix.to(dtype),
            tsdf,
            colors,
            weights,
            depth_image,
            rgb_image,
            weight_image,
        )

        if show_progress:
            assert isinstance(enumerator, tqdm.tqdm)
            enumerator.set_postfix({"accumulated_voxels": accum_grid.num_voxels})

        # Prune out zero weight voxels to save memory
        new_grid = accum_grid.pruned_grid(weights > 0.0)
        tsdf = new_grid.inject_from(accum_grid, tsdf)
        colors = new_grid.inject_from(accum_grid, colors)
        weights = new_grid.inject_from(accum_grid, weights)
        accum_grid = new_grid

        # TSDF fusion is a bit of a torture case for the PyTorch memory allocator since
        # it progressively allocates bigger tensors which don't fit in the memory pool,
        # causing the pool to grow larger and larger.
        # To avoid this, we synchronize the CUDA device and empty the cache after each image.
        del rgb_image, depth_image, weight_image
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # After integrating all the images, we prune the grid to remove empty voxels which have no weights.
    # This is done to reduce the size of the grid and speed up the marching cubes algorithm
    # which will be used to extract the mesh.
    new_grid = accum_grid.pruned_grid(weights > 0.0)
    filter_tsdf = new_grid.inject_from(accum_grid, tsdf)
    filter_colors = new_grid.inject_from(accum_grid, colors)

    return new_grid, filter_tsdf, filter_colors
