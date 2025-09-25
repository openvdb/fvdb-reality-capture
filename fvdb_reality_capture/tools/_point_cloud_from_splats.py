# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import torch
import tqdm
from skimage import feature, morphology

from fvdb import GaussianSplat3d


@torch.no_grad()
def point_cloud_from_splats(
    model: GaussianSplat3d,
    camera_to_world_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    image_sizes: torch.Tensor,
    near: float = 0.1,
    far: float = 1e10,
    depth_image_downsample_factor: int = 1,
    canny_edge_std: float = 1.0,
    canny_mask_dilation: int = 5,
    dtype: torch.dtype = torch.float16,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a point cloud from a Gaussian splat using depth rendering, possibly filtering points
    using Canny edge detection on the depth images.

    Args:
        model (GaussianSplat3d): The Gaussian splat model to extract a mesh from
        camera_to_world_matrices (torch.Tensor): A (C, 4, 4)-shaped Tensor containing the camera to world
            matrices to render depth images from for mesh extraction where C is the number of camera views.
        projection_matrices (torch.Tensor): A (C, 3, 3)-shaped Tensor containing the perspective projection matrices
            used to render images for mesh extraction where C is the number of camera views.
        image_sizes (torch.Tensor): A (C, 2)-shaped Tensor containing the width and height of each image to extract
            from the Gaussian splat where C is the number of camera views.
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
        show_progress (bool): Whether to show a progress bar (default is True).

    Returns:
        points (torch.Tensor): A [num_points, 3] shaped tensor of points in camera space.
        colors (torch.Tensor): A [num_points, 3] shaped tensor of RGB colors for the points.
    """

    device = model.device

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
        image_height = int(image_size[0].item())
        image_width = int(image_size[1].item())

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
