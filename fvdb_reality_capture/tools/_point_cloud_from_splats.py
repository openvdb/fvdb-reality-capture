# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import torch
import tqdm
from fvdb import GaussianSplat3d
from fvdb.types import (
    NumericMaxRank1,
    NumericMaxRank2,
    NumericMaxRank3,
    is_NumericScalar,
    to_FloatingScalar,
    to_VecNf,
)
from skimage import feature, morphology

from ._common import validate_camera_matrices_and_image_sizes


@torch.no_grad()
def point_cloud_from_splats(
    model: GaussianSplat3d,
    camera_to_world_matrices: NumericMaxRank3,
    projection_matrices: NumericMaxRank3,
    image_sizes: NumericMaxRank2,
    near: NumericMaxRank1 = 0.1,
    far: NumericMaxRank1 = 1e10,
    alpha_threshold: float = 0.1,
    image_downsample_factor: int = 1,
    canny_edge_std: float = 1.0,
    canny_mask_dilation: int = 5,
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a point cloud with colors/features from a Gaussian splat radiance field by unprojecting depth images rendered from it.

    This algorithm can optionally filter out points near depth discontinuities using the following heurstic:

    1. Apply a small Gaussian filter to the depth images to reduce noise.
    2. Run a `Canny edge detector <https://en.wikipedia.org/wiki/Canny_edge_detector>`_ on the depth immage to find
       depth discontinuities. The result is an image mask where pixels near depth edges are marked.
    3. Dilate the edge mask to remove depth samples near edges.
    4. Remove points from the point cloud where the corresponding depth pixel is marked in the dilated edge mask.

    Args:
        model (GaussianSplat3d): The Gaussian splat radiance field to extract a mesh from.
        camera_to_world_matrices (NumericMaxRank3): A ``(C, 4, 4)``-shaped Tensor containing the camera to world
            matrices to render depth images from for mesh extraction where ``C`` is the number of camera views.
        projection_matrices (NumericMaxRank3): A ``(C, 3, 3)``-shaped Tensor containing the perspective projection matrices
            used to render images for mesh extraction where ``C`` is the number of camera views.
        image_sizes (NumericMaxRank2): A ``(C, 2)``-shaped Tensor containing the height and width of each image to extract
            from the Gaussian splat where ``C`` is the number of camera views. *i.e.*, ``image_sizes[c] = (height_c, width_c)``.
        near (NumericMaxRank1): Near plane distance below which to ignore depth samples. Can be a scalar to use a
            single value for all images or a tensor-like object of shape ``(C,)`` to use a different value for each
            image. Default is 0.1.
        far (NumericMaxRank1): Far plane distance above which to ignore depth samples. Can be a scalar to use a
            single value for all images or a tensor-like object of shape ``(C,)`` to use a different value for each
            image. Default is 1e10.
        alpha_threshold (float): Alpha threshold to mask pixels rendered by the Gaussian splat model that are transparent
            (usually indicating the background). Default is 0.1.
        image_downsample_factor (int): Factor by which to downsample the rendered images before extracting points
            This is useful to reduce the number of points extracted from the point cloud
            and speed up the extraction process. A value of 2 will downsample the rendered images by a factor of 2 in both dimensions,
            resulting in a point cloud with approximately 1/4 the number of points compared to the original rendered images.
        canny_edge_std (float): Standard deviation (in pixel units) for the Gaussian filter applied to the depth image
            before Canny edge detection. Set to 0.0 to disable canny edge filtering.
            Default is 1.0.
        canny_mask_dilation (int): Dilation size (in pixels) for the Canny edge mask. Default is 5.
        dtype (torch.dtype): Data type to store the point cloud positions. Default is ``torch.float16``.
        feature_dtype (torch.dtype): Data type to store per-point colors/features.
            Default is ``torch.uint8`` which is good for RGB colors.
        show_progress (bool): Whether to show a progress bar. Default is ``True``.

    Returns:
        points (torch.Tensor): A ``(num_points, 3)``-shaped tensor of point positions in world space.
        colors (torch.Tensor): A ``(num_points, D)``-shaped tensor of colors/features per point where
            ``D`` is the number of channels encoded by the Gaussian Splat model (usually 3 for RGB colors).
    """

    device = model.device

    camera_to_world_matrices, projection_matrices, image_sizes = validate_camera_matrices_and_image_sizes(
        camera_to_world_matrices, projection_matrices, image_sizes
    )

    points_list = []
    colors_list = []

    num_cameras = len(camera_to_world_matrices)
    enumerator = (
        tqdm.tqdm(range(num_cameras), unit="imgs", desc="Extracting Point Cloud")
        if show_progress
        else range(num_cameras)
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

    total_points = 0
    for i in enumerator:
        cam_to_world_matrix = camera_to_world_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        world_to_cam_matrix = torch.linalg.inv(cam_to_world_matrix).contiguous().to(dtype=torch.float32, device=device)
        projection_matrix = projection_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        inv_projection_matrix = torch.linalg.inv(projection_matrix).to(dtype=torch.float32, device=device)
        image_size = image_sizes[i]
        image_width = int(image_size[1].item())
        image_height = int(image_size[0].item())

        # We set near and far planes to 0.0 and 1e10 respectively to avoid clipping
        # in the rendering process. Instead, we will use the provided near and far planes
        # to filter the depth images after rendering so pixels out of range will not add points
        # to the point cloud.
        feature_and_depth, alpha = model.render_images_and_depths(
            world_to_camera_matrices=world_to_cam_matrix.unsqueeze(0),
            projection_matrices=projection_matrix.unsqueeze(0),
            image_width=image_width,
            image_height=image_height,
            near=0.0,
            far=1e10,
        )

        if feature_dtype == torch.uint8:
            feature_images = (feature_and_depth[..., : model.num_channels].clip_(min=0.0, max=1.0) * 255.0).to(
                feature_dtype
            )
        else:
            feature_images = feature_and_depth[..., : model.num_channels].to(feature_dtype)
        alpha = alpha[0].clamp(min=1e-10).squeeze(-1)  # [H, W]
        feature_images = feature_images.squeeze(0)  # [H, W, C]
        depth_images = (feature_and_depth[0, ..., -1] / alpha).to(dtype)

        # Get the near and far planes for this image.
        near_i = near.item() if near_is_scalar else near[i]
        far_i = far.item() if far_is_scalar else far[i]

        if alpha_threshold > 0.0:
            alpha_mask = alpha > alpha_threshold
            mask = ((depth_images > near_i) & (depth_images < far_i) & alpha_mask).squeeze(-1)  # [H, W]
        else:
            mask = ((depth_images > near_i) & (depth_images < far_i)).squeeze(-1)  # [H, W]

        # TODO: Add GPU Canny edge detection
        if canny_edge_std > 0.0:
            canny_mask = torch.tensor(
                morphology.dilation(
                    feature.canny(depth_images.squeeze(-1).cpu().numpy(), sigma=canny_edge_std),
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
        cam_pts = cam_pts.view(3, image_height, image_width) * depth_images.unsqueeze(0)  # [3, H, W]

        # Transform camera space coordinates to world coordinates
        world_pts = torch.cat(
            [cam_pts, torch.ones(1, cam_pts.shape[1], cam_pts.shape[2]).to(cam_pts)], dim=0
        )  # [4, H, W]
        world_pts = cam_to_world_matrix @ world_pts.view(4, -1)  # [4, H, W]
        world_pts = world_pts[:3] / world_pts[3].unsqueeze(0)  # [3, H * W]
        world_pts = world_pts.view(3, image_height, image_width).permute(1, 2, 0)  # [H, W, 3]

        # Optionally downsample the world points and feature image
        world_pts = world_pts[mask].view(-1, 3)  # [num_points, 3]
        features = feature_images[mask]  # [num_points, C]

        if world_pts.numel() == 0:
            continue

        assert world_pts.shape[0] == features.shape[0], "Number of points and features must match."

        if show_progress:
            assert isinstance(enumerator, tqdm.tqdm)
            enumerator.set_postfix({"total_points": total_points})

        points_list.append(world_pts.to(dtype))
        colors_list.append(features)
        total_points += points_list[-1].shape[0]

    return torch.cat(points_list, dim=0).to(dtype), torch.cat(colors_list, dim=0)
