# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
import tempfile

import numpy as np
import torch
import tqdm
from fvdb import GaussianSplat3d, Grid
from fvdb.types import NumericMaxRank2, NumericMaxRank3

from fvdb_reality_capture.foundation_models.dlnr import DLNRModel
from fvdb_reality_capture.sfm_scene import SfmCache

from ._common import validate_camera_matrices_and_image_sizes


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
    plt.imshow(image_l.squeeze().cpu().numpy())

    plt.subplot(4, 2, 6)
    plt.title("Right Image")
    plt.imshow(image_r.squeeze().cpu().numpy())

    plt.savefig(out_filename, bbox_inches="tight")
    plt.close()


class TSDFInputDataset(torch.utils.data.Dataset):
    """
    A torch Dataset that computes RGB images, depths, and weights for TSDF fusion using a Gaussian splat model for images
    and DLNR for depth, and occlusion masking. The dataset caches the results to disk to avoid recomputing them
    when running TSDF fusion.
    """

    def __init__(
        self,
        cache_path: pathlib.Path,
        model: GaussianSplat3d,
        camera_to_world_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_sizes: torch.Tensor,
        baseline: float,
        near: float,
        far: float,
        reprojection_threshold: float,
        alpha_threshold: float,
        dlnr_model: DLNRModel,
        use_absolute_baseline: bool,
        show_progress: bool,
    ):
        """
        Create a TSDFInputDataset by precomputing and caching the RGB images, depths, and weights for TSDF fusion.

        Args:
            cache_path (pathlib.Path): Path to the directory to use for caching the results.
            model (GaussianSplat3d): The Gaussian splat model to render from.
            camera_to_world_matrices (torch.Tensor): A (C, 4, 4)-shaped Tensor containing the camera to world
                matrices to render depth images from for mesh extraction where C is the number of camera views.
            projection_matrices (torch.Tensor): A (C, 3, 3)-shaped Tensor containing the perspective projection matrices
                used to render images for mesh extraction where C is the number of camera views.
            image_sizes (torch.Tensor): A (C, 2)-shaped Tensor containing the width and height of each image to extract
                from the Gaussian splat where C is the number of camera views.
            baseline (float): The distance between the two camera positions along the camera -x axis.
                If use_absolute_baseline is False, this is interpreted as a fraction of the mean depth of each image.
            near (float): Near plane distance below which to ignore depth samples, as a multiple of the baseline.
            far (float): Far plane distance above which to ignore depth samples, as a multiple of the baseline.
            reprojection_threshold (float): Reprojection error threshold for occlusion masking in pixels.
            alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model is transparent
                (usually indicating the background).
            dlnr_model (DLNRModel): The DLNR model to compute optical flow and disparity.
            use_absolute_baseline (bool): If True, use the provided baseline as an absolute distance in world units.
            show_progress (bool): Whether to show a progress bar (default is True).
        """
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)

        self.cache = SfmCache.get_cache(cache_path, "TSDFInputs", "Cache for TSDF inputs")
        self.num_images = camera_to_world_matrices.shape[0]
        self.model = model
        self.baseline_fraction_of_depth_or_absolute = baseline
        self.use_absolute_baseline = use_absolute_baseline
        self.near = near
        self.far = far
        self.reprojection_threshold = reprojection_threshold
        self.alpha_threshold = alpha_threshold
        self.dlnr_model = dlnr_model

        device = model.device

        enumerator = (
            tqdm.tqdm(range(self.num_images), unit="imgs", desc="Generating DLNR Depths")
            if show_progress
            else range(self.num_images)
        )

        for i in enumerator:
            cam_to_world_matrix = camera_to_world_matrices[i].to(dtype=torch.float32, device=device)
            world_to_cam_matrix = (
                torch.linalg.inv(cam_to_world_matrix).contiguous().to(dtype=torch.float32, device=device)
            )
            projection_matrix = projection_matrices[i].to(dtype=torch.float32, device=device)
            image_height, image_width = int(image_sizes[i][0].item()), int(image_sizes[i][1].item())

            # debug_img_name = f"debug_image_{i:04d}.png"
            rgb_image, depth_image, weight_image = self.extract_single_tsdf_input(
                world_to_cam_matrix=world_to_cam_matrix,
                projection_matrix=projection_matrix,
                image_width=image_width,
                image_height=image_height,
                save_debug_images_to=None,  # Set to a path if you want to save debug images
            )

            self.cache.write_file(f"rgb_{i}", rgb_image.cpu().numpy(), data_type="npy")
            self.cache.write_file(f"depth_{i}", depth_image.cpu().numpy(), data_type="npy")
            self.cache.write_file(f"weight_{i}", weight_image.cpu().numpy(), data_type="npy")

    def extract_single_tsdf_input(
        self,
        world_to_cam_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
        image_width: int,
        image_height: int,
        save_debug_images_to: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute an RGB image, depth map, and weight image to be used as input to TSDF fusion for a given camera pose.
        This function uses the Gaussian splat model for images and the DLNR model for depth, and occlusion masking.
        This algorithm is roughly based on the GS2Mesh algorithm described in https://arxiv.org/abs/2404.01810.

        The algorithm renders a stereo pair of images from the Gaussian splat model for each camera position, computes disparities
        using DLNR, computes an occlusion mask based on the disparities, and then computes a
        near/far mask based on the depth. The final weights are a combination of the near/far mask and the occlusion mask.

        Args:
            world_to_cam_matrix (torch.Tensor): The camera_to_world transformation matrix for the first
                image in the stereo pair. The second image has the same transformation but with the x position
                shifted by the negative baseline.
            projection_matrix (torch.Tensor): The projection matrix for the camera.
            image_width (int): The width of the rendered images in pixels.
            image_height (int): The height of the rendered images in pixels.
            alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model is not confident.
            save_debug_images_to (str | None): If provided, saves debug images to this path.

        Returns:
            image_l (torch.Tensor): The first rendered image whose camera to world matrix is the same as the input.
            depth (torch.Tensor): The rendered depth image for the first camera.
            weights (torch.Tensor): The computed weight image for the first camera.
        """

        if not self.use_absolute_baseline:
            baseline = self.estimate_baseline_from_depth(
                world_to_camera_matrix=world_to_cam_matrix,
                projection_matrix=projection_matrix,
                image_width=image_width,
                image_height=image_height,
            )
        else:
            baseline = self.baseline_fraction_of_depth_or_absolute

        near = self.near * baseline
        far = self.far * baseline

        # Render the stereo pair of images and clip to [0, 1]
        image_l, image_r, alpha_mask = self.render_stereo_pair(
            baseline, world_to_cam_matrix, projection_matrix, image_width, image_height
        )
        image_l.clip_(min=0.0, max=1.0)
        image_r.clip_(min=0.0, max=1.0)

        # Compute left-to-right and right-to-left disparities and depth using DLNR
        disparity_l2r, disparity_r2l, depth = self.compute_disparities_and_depth(
            image_l=image_l,
            image_r=image_r,
            projection_matrix=projection_matrix,
            baseline=baseline,
        )

        # Compute an occlusion mask based on the reprojection error of the disparities
        occlusion_mask = self.compute_occlusion_mask(
            disparity_l2r,
            disparity_r2l,
        )

        # Create masks using the near and far values
        near_far_mask = (depth > near) & (depth < far)

        # The final weights are a combination of the near/far mask and the occlusion mask
        if alpha_mask is not None:
            weights = near_far_mask & occlusion_mask & alpha_mask
        else:
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

    def estimate_baseline_from_depth(
        self,
        world_to_camera_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> float:
        """
        Estimate a baseline distance as a percentage of the mean depth of an image rendered from a Gaussian Splat model.

        We want to choose a baseline that is wide enough to get good depth estimates from stereo matching,
        but not so wide that the two images have little overlap (and thus have high error in the disparity estimate).
        A common heuristic is to set the baseline to be a small percentage of the mean depth of the scene.

        Args:
            world_to_camera_matrix (torch.Tensor): The camera_to_world transformation matrix for the image.
            projection_matrix (torch.Tensor): The projection matrix for the camera.
            image_width (int): The width of the rendered image.
            image_height (int): The height of the rendered image.

        Returns:
            float: The estimated baseline distance in world units.
        """

        depth_0, alpha_0 = self.model.render_depths(
            world_to_camera_matrices=world_to_camera_matrix.unsqueeze(0),
            projection_matrices=projection_matrix.unsqueeze(0),
            image_width=image_width,
            image_height=image_height,
            near=0.0,
            far=1e10,
        )
        depth_0 = depth_0 / alpha_0.clamp(min=1e-10)
        baseline = self.baseline_fraction_of_depth_or_absolute * depth_0.mean().item()
        return baseline

    def render_stereo_pair(
        self,
        baseline: float,
        world_to_camera_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Render a pair of stereo images from a Gaussian Splat model.

        The pair of images is rendered by shifting the camera position by a baseline distance along
        the camera's -x axis to simulate stereo vision.

        Args:
            baseline (float): The distance between the two camera positions along the camera -x axis (in world units).
            world_to_camera_matrix (torch.Tensor): The camera_to_world transformation matrix for the first
                image in the stereo pair. The second image has the same transformation but with the x position
                shifted by the negative baseline.
            projection_matrix (torch.Tensor): The projection matrix for the camera.
            image_width (int): The width of the rendered images in pixels.
            image_height (int): The height of the rendered images in pixels.

        Returns:
            image_1 (torch.Tensor): The first rendered image whose camera to world matrix is the same as the input.
            image_2 (torch.Tensor): The second rendered image whose camera to world matrix is the same as the input but
                with the x position shifted by the negative baseline.
            alpha_mask (torch.Tensor): A binary mask indicating pixels where the alpha value exceeds the alpha
                threshold if self._alpha_threshold > 0.0, else None.
        """
        # Compute the left and right camera poses
        world_to_camera_matrix_left = world_to_camera_matrix.clone()
        world_to_camera_matrix_right = world_to_camera_matrix.clone()
        world_to_camera_matrix_right[0, 3] -= baseline

        world_to_camera_matrix = torch.stack([world_to_camera_matrix_left, world_to_camera_matrix_right], dim=0)
        projection_matrix = torch.stack([projection_matrix, projection_matrix], dim=0)

        images, alphas = self.model.render_images(
            world_to_camera_matrices=world_to_camera_matrix,
            projection_matrices=projection_matrix,
            image_width=image_width,
            image_height=image_height,
            near=0.0,
            far=1e10,
        )

        alpha_mask = alphas[0].squeeze(-1) > self.alpha_threshold if self.alpha_threshold > 0.0 else None

        return images[0], images[1], alpha_mask

    def compute_occlusion_mask(self, l2r_disparity: torch.Tensor, r2l_disparity: torch.Tensor) -> torch.Tensor:
        """
        Compute an occlusion mask using the disparity maps by filtering pixels where the
        reprojection error exceeds the reprojection threshold.

        Given a point in space, and a stereo pair of images, disparity maps are computed as the
        difference in pixel coordinates between the projection of that point in the left and right images.

        The occlusion mask is computed by using the left-to-right disparity map to project pixels from the left image
        to the right image, and then using the right-to-left disparity map to reproject those pixels back to the left image.
        If the reprojection error exceeds the reprojection threshold, the pixel is considered occluded.

        Args:
            l2r_disparity (torch.Tensor): Left-to-right disparity map.
            r2l_disparity (torch.Tensor): Right-to-left disparity map.

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

        occlusion_mask = disparity_difference > self.reprojection_threshold

        occlusion_mask[(x_projected < 0) | (x_projected >= width)] = True

        return ~occlusion_mask

    def compute_disparities_and_depth(
        self,
        image_l: torch.Tensor,  # [H, W, C]
        image_r: torch.Tensor,  # [H, W, C]
        projection_matrix: torch.Tensor,  # [3, 3]
        baseline: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute left-to-right and right-to-left disparities and depth using the DLNR model, and use
        the pinhole camera model to convert the left-to-right disparity to depth.

        Args:
            image_l (torch.Tensor): The left image.
            image_r (torch.Tensor): The right image.
            projection_matrix (torch.Tensor): The projection matrix for the camera.
            baseline (float): The distance between the two camera positions along the camera -x axis (in world units).

        Returns:
            disparity_l2r (torch.Tensor): Left-to-right disparity map.
            disparity_r2l (torch.Tensor): Right-to-left disparity map.
            depth (torch.Tensor): Depth map computed from the left-to-right disparity.
        """
        _, disparity_l2r = self.dlnr_model.predict_flow(
            images1=image_l.unsqueeze(0),  # [1, H, W, C]
            images2=image_r.unsqueeze(0),  # [1, H, W, C]
            flow_init=None,
        )
        disparity_l2r = -disparity_l2r[0]  # [H, W]

        image_l_flip = torch.flip(image_l, dims=[1])
        image_r_flip = torch.flip(image_r, dims=[1])
        _, disparity_r2l = self.dlnr_model.predict_flow(
            images1=image_r_flip.unsqueeze(0),  # [1, H, W, C]
            images2=image_l_flip.unsqueeze(0),  # [1, H, W, C]
            flow_init=None,
        )
        disparity_r2l = -torch.flip(disparity_r2l[0], dims=[1])  # [H, W]

        fx = projection_matrix[0, 0].item()
        depth = (fx * baseline) / disparity_l2r

        return disparity_l2r, disparity_r2l, depth

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        _, rgb = self.cache.read_file(f"rgb_{idx}")
        _, depth = self.cache.read_file(f"depth_{idx}")
        _, weight = self.cache.read_file(f"weight_{idx}")
        return torch.from_numpy(rgb), torch.from_numpy(depth), torch.from_numpy(weight)


@torch.no_grad()
def tsdf_from_splats_dlnr(
    model: GaussianSplat3d,
    camera_to_world_matrices: NumericMaxRank3,
    projection_matrices: NumericMaxRank3,
    image_sizes: NumericMaxRank2,
    truncation_margin: float,
    grid_shell_thickness: float | int = 3.0,
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
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """
    Extract a Truncated Signed Distance Field (TSDF) from a `fvdb.GaussianSplat3d` using TSDF fusion from depth maps
    predicted from the Gaussian splat radiance field and the
    `DLNR foundation model <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf>`_.
    DLNR is a high-frequency stereo matching network that computes optical flow and disparity maps between two images, which can be used to compute depth.

    This algorithm proceeds in two steps:

    1. First, it renders stereo pairs of images from the Gaussian splat radiance field, and uses
       DLNR to compute depth maps from these stereo pairs in the frame of the first image in the pair.
       The result is a set of depth maps aligned with the rendered images.

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
        accum_grid (Grid): The accumulated :class:`fvdb.Grid` representing the voxels in the TSDF volume.
        tsdf (torch.Tensor): The TSDF values for each voxel in the grid.
        colors (torch.Tensor): The colors/features for each voxel in the grid.
    """

    if model.num_channels != 3:
        raise ValueError(f"Expected model with 3 channels, got {model.num_channels} channels.")

    if grid_shell_thickness <= 1.0:
        raise ValueError("grid_shell_thickness must be greater than 1.0")

    camera_to_world_matrices, projection_matrices, image_sizes = validate_camera_matrices_and_image_sizes(
        camera_to_world_matrices, projection_matrices, image_sizes
    )

    if image_downsample_factor > 1:
        image_sizes = image_sizes // image_downsample_factor
        projection_matrices = projection_matrices.clone()
        projection_matrices[:, :2, :] /= image_downsample_factor

    with tempfile.TemporaryDirectory() as cache_path:
        dataset = TSDFInputDataset(
            cache_path=pathlib.Path(cache_path),
            model=model,
            camera_to_world_matrices=camera_to_world_matrices,
            projection_matrices=projection_matrices,
            image_sizes=image_sizes,
            baseline=baseline,
            near=near,
            far=far,
            reprojection_threshold=disparity_reprojection_threshold,
            alpha_threshold=alpha_threshold,
            dlnr_model=DLNRModel(backbone=dlnr_backbone, device=model.device),
            use_absolute_baseline=use_absolute_baseline,
            show_progress=show_progress,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

        device = model.device
        # The voxel size is set by dividing the truncation margin by the grid shell thickness.
        # This ensures that the truncation margin spans 'grid_shell_thickness' number of voxels,
        # controlling the grid resolution and mesh quality. Adjusting grid_shell_thickness changes
        # how many voxels fit within the truncation margin, affecting surface detail.
        voxel_size = truncation_margin / grid_shell_thickness
        accum_grid = Grid.from_dense(dense_dims=1, ijk_min=0, voxel_size=voxel_size, origin=0.0, device=device)
        tsdf = torch.zeros(accum_grid.num_voxels, device=device, dtype=dtype)
        weights = torch.zeros(accum_grid.num_voxels, device=device, dtype=dtype)
        colors = torch.zeros((accum_grid.num_voxels, model.num_channels), device=device, dtype=feature_dtype)

        enumerator = tqdm.tqdm(dataloader, unit="imgs", desc="Extracting TSDF") if show_progress else dataloader

        for i, tsdf_input in enumerate(enumerator):
            cam_to_world_matrix = camera_to_world_matrices[i].to(dtype=torch.float32, device=device)
            projection_matrix = projection_matrices[i].to(dtype=torch.float32, device=device)

            rgb_image, depth_image, weight_image = tsdf_input
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
                depth_image.squeeze(0).to(device),
                rgb_image.squeeze(0).to(device),
                weight_image.squeeze(0).to(device),
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
