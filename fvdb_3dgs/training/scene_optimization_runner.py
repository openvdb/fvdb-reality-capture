# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import json
import logging
import pathlib
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm
from fvdb import GaussianSplat3d
from fvdb.utils.metrics import psnr, ssim
from scipy.spatial import cKDTree
from torch.utils.tensorboard import SummaryWriter

from ..sfm_scene import SfmScene
from ..transforms import (
    BaseTransform,
    Compose,
    CropScene,
    DownsampleImages,
    FilterImagesWithLowPoints,
    NormalizeScene,
    PercentileFilterPoints,
)
from ..viewer import Viewer
from .camera_pose_adjust import CameraPoseAdjustment
from .checkpoint import Checkpoint
from .gaussian_splat_optimizer import GaussianSplatOptimizer
from .lpips import LPIPSLoss
from .sfm_dataset import SfmDataset
from .utils import make_unique_name_directory_based_on_time


@dataclass
class Config:
    # Random seed
    seed: int = 42

    #
    # Training duration and evaluation parameters
    #

    # Number of training epochs -- i.e. number of times we will visit each image in the dataset
    max_epochs: int = 200
    # Optional maximum number of training steps (overrides max_epochs * dataset_size if set)
    max_steps: int | None = None
    # Percentage of total epochs at which we perform evaluation on the validation set. i.e. 10 means perform evaluation after 10% of the epochs.
    eval_at_percent: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 75, 100])
    # Percentage of total epochs at which we save the model checkpoint. i.e. 10 means save a checkpoint after 10% of the epochs.
    save_at_percent: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 75, 100])

    #
    # Gaussian Optimization Parameters
    #

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # If you're using very large images, run the forward pass on crops and accumulate gradients
    crops_per_image: int = 1
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this many epochs
    increase_sh_degree_every_epoch: int = 5
    # Initial opacity of each Gaussian
    initial_opacity: float = 0.1
    # Initial scale of each Gaussian
    initial_covariance_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Which network to use for LPIPS loss
    lpips_net: Literal["vgg", "alex"] = "alex"
    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0
    # Use random background for training to discourage transparency
    random_bkgd: bool = False
    # When to start refining (split/duplicate/merge) Gaussians during optimization
    refine_start_epoch: int = 3
    # When to stop refining (split/duplicate/merge) Gaussians during optimization
    refine_stop_epoch: int = 100
    # How often to refine (split/duplicate/merge) Gaussians during optimization
    refine_every_epoch: float = 0.75
    # How often to reset the opacities of the Gaussians during optimization
    reset_opacities_every_epoch: int = 16
    # When to stop using the 2d projected scale for refinement (default of 0 is to never use it)
    refine_using_scale2d_stop_epoch: int = 0
    # Whether to ignore masks during training
    ignore_masks: bool = False
    # Whether to remove Gaussians that fall outside the scene bounding box
    remove_gaussians_outside_scene_bbox: bool = False

    #
    # Pose optimization parameters
    #

    # Flag to enable camera pose optimization.
    optimize_camera_poses: bool = True
    # Learning rate for camera pose optimization.
    pose_opt_lr: float = 1e-5
    # Weight for regularization of camera pose optimization.
    pose_opt_reg: float = 1e-6
    # Learning rate decay factor for camera pose optimization (will decay to this fraction of initial lr)
    pose_opt_lr_decay: float = 1.0
    # Which epoch to stop optimizing camera postions. Default matches max training epochs.
    pose_opt_stop_epoch: int = max_epochs
    # Standard devation for the normal distribution used for camera pose optimization's random iniitilaization
    pose_opt_init_std: float = 1e-4

    #
    # Gaussian Rendering Parameters
    #

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10
    # Minimum screen space radius below which Gaussians are ignored after projection
    min_radius_2d: float = 0.0
    # Blur amount for anti-aliasing
    eps_2d: float = 0.3
    # Whether to use anti-aliasing or not
    antialias: bool = False
    # Size of tiles to use during rasterization
    tile_size: int = 16


def crop_image_batch(image: torch.Tensor, mask: torch.Tensor | None, ncrops: int):
    """
    Generator to iterate a minibatch of images (B, H, W, C) into disjoint patches patches (B, H_patch, W_patch, C).
    We use this function when training on very large images so that we can accumulate gradients over
    crops of each image.

    Args:
        image (torch.Tensor): Image minibatch (B, H, W, C)
        mask (torch.Tensor | None): Optional mask of shape (B, H, W) to apply to the image.
        ncrops (int): Number of chunks to split the image into (i.e. each crop will have shape (B, H/ncrops x W/ncrops, C).

    Yields: A crop of the input image and its coordinate
        image_patch (torch.Tensor): the patch with shape (B, H/ncrops, W/ncrops, C)
        mask_patch (torch.Tensor | None): the mask patch with shape (B, H/ncrops, W/ncrops) or None if no mask is provided
        crop (tuple[int, int, int, int]): the crop coordinates (x, y, w, h),
        is_last (bool): is true if this is the last crop in the iteration
    """
    h, w = image.shape[1:3]
    patch_w, patch_h = w // ncrops, h // ncrops
    patches = np.array(
        [
            [i * patch_w, j * patch_h, (i + 1) * patch_w, (j + 1) * patch_h]
            for i, j in itertools.product(range(ncrops), range(ncrops))
        ]
    )
    for patch_id in range(patches.shape[0]):
        x1, y1, x2, y2 = patches[patch_id]
        image_patch = image[:, y1:y2, x1:x2]
        mask_patch = None
        if mask is not None:
            mask_patch = mask[:, y1:y2, x1:x2]

        crop = (x1, y1, (x2 - x1), (y2 - y1))
        assert (x2 - x1) == patch_w and (y2 - y1) == patch_h
        is_last = patch_id == (patches.shape[0] - 1)
        yield image_patch, mask_patch, crop, is_last


class TensorboardLogger:
    """
    A utility class to log training metrics to TensorBoard.
    """

    def __init__(self, log_dir: pathlib.Path, log_every_step: int = 100, log_images_to_tensorboard: bool = False):
        """
        Create a new `TensorboardLogger` instance which is used to track training and evaluation progress in tensorboard.

        Args:
            log_dir (pathlib.Path): Directory to save TensorBoard logs.
            log_every_step (int): Log every `log_every_step` steps.
            log_images_to_tensorboard (bool): Whether to log images to TensorBoard.
        """
        self._log_every_step = log_every_step
        self._log_dir = log_dir
        self._log_images_to_tensorboard = log_images_to_tensorboard
        self._tb_writer = SummaryWriter(log_dir=log_dir)

    def log_training_iteration(
        self,
        step: int,
        num_gaussians: int,
        loss: float,
        l1loss: float,
        ssimloss: float,
        mem: float,
        gt_img: torch.Tensor,
        pred_img: torch.Tensor,
        pose_loss: float | None,
    ):
        """
        Log training metrics to TensorBoard.

        Args:
            step: Current training step.
            num_gaussians: Number of Gaussians in the model.
            loss: Total loss value.
            l1loss: L1 loss value.
            ssimloss: SSIM loss value.
            mem: Maximum GPU memory allocated in GB.
            pose_loss: Pose optimization loss, if applicable.
            gt_img: Ground truth image for visualization.
            pred_img: Predicted image for visualization.
        """
        if self._log_every_step > 0 and step % self._log_every_step == 0 and self._tb_writer is not None:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            self._tb_writer.add_scalar("train/loss", loss, step)
            self._tb_writer.add_scalar("train/l1loss", l1loss, step)
            self._tb_writer.add_scalar("train/ssimloss", ssimloss, step)
            self._tb_writer.add_scalar("train/num_gaussians", num_gaussians, step)
            self._tb_writer.add_scalar("train/mem", mem, step)
            # Log pose optimization metrics
            if pose_loss is not None:
                # Log individual components of pose parameters
                self._tb_writer.add_scalar("train/pose_reg_loss", pose_loss, step)
            if self._log_images_to_tensorboard:
                canvas = torch.cat([gt_img, pred_img], dim=2).detach().cpu().numpy()
                canvas = canvas.reshape(-1, *canvas.shape[2:])
                self._tb_writer.add_image("train/render", canvas, step)
            self._tb_writer.flush()

    def log_evaluation_iteration(
        self,
        step: int,
        psnr: float,
        ssim: float,
        lpips: float,
        avg_time_per_image: float,
        num_gaussians: int,
    ):
        """
        Log evaluation metrics to TensorBoard.

        Args:
            step: The training step after which the evaluation was performed.
            psnr: Peak Signal-to-Noise Ratio for the evaluation (averaged over all images in the validation set).
            ssim: Structural Similarity Index Measure for the evaluation (averaged over all images in the validation set).
            lpips: Learned Perceptual Image Patch Similarity for the evaluation (averaged over all images in the validation set).
            avg_time_per_image: Average time taken to evaluate each image.
            num_gaussians: Number of Gaussians in the model at this evaluation step.
        """

        self._tb_writer.add_scalar("eval/psnr", psnr, step)
        self._tb_writer.add_scalar("eval/ssim", ssim, step)
        self._tb_writer.add_scalar("eval/lpips", lpips, step)
        self._tb_writer.add_scalar("eval/avg_time_per_image", avg_time_per_image, step)
        self._tb_writer.add_scalar("eval/num_gaussians", num_gaussians, step)


class ViewerLogger:
    """
    A utility class to visualize the scene being trained and log training statistics and model state to the viewer.
    """

    def __init__(
        self,
        splat_scene: GaussianSplat3d,
        train_dataset: SfmDataset,
        viewer_port: int = 8080,
        verbose: bool = False,
    ):
        """
        Create a new `ViewerLogger` instance which is used to track training and evaluation progress through the viewer.

        Args:
            splat_scene: The GaussianSplat3d scene to visualize.
            train_dataset: The dataset containing camera frames and images.
            viewer_port: The port on which the viewer will run.
            verbose: If True, print additional information about the viewer.
        """

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self.viewer = Viewer(port=viewer_port, verbose=verbose)
        bbmin, bbmax = train_dataset.points.min(axis=0), train_dataset.points.max(axis=0)
        bbox_diagonal_length = float(np.linalg.norm(bbmax - bbmin))
        self.viewer.camera_far = 3.0 * bbox_diagonal_length
        self._splat_model_view = self.viewer.register_gaussian_splat_3d(name="Model", gaussian_scene=splat_scene)

        sfm_scene = train_dataset.sfm_scene
        scene_extent = sfm_scene.points.max(0) - sfm_scene.points.min(0)
        axis_scale = 0.01 * float(np.linalg.norm(scene_extent))
        self._logger.info(f"Using scene extent = {scene_extent} for viewer. Scaling camera view axis by {axis_scale}.")

        self._train_camera_view = self.viewer.register_camera_view(
            name="Training Cameras",
            cam_to_world_matrices=train_dataset.camera_to_world_matrices,
            projection_matrices=train_dataset.projection_matrices,
            image_sizes=train_dataset.image_sizes,
            images=None,
            frustum_line_width=2.0,
            frustum_scale=1.0 * axis_scale,
            axis_length=2.0 * axis_scale,
            axis_thickness=0.1 * axis_scale,
            show_images=False,
            enabled=False,
        )

        self._training_metrics_view = self.viewer.register_dictionary_label(
            "Training Metrics",
            {
                "Current Iteration": 0,
                "Current SH Degree": 0,
                "Num Gaussians": 0,
                "Loss": 0.0,
                "SSIM Loss": 0.0,
                "L1 Loss": 0.0,
                "GPU Memory Usage": 0,
                "Pose Regularization": 0.0,
            },
        )

        self._evaluation_metrics_view = self.viewer.register_dictionary_label(
            "Evaluation Metrics",
            {
                "Last Evaluation Step": 0,
                "PSNR": 0.0,
                "SSIM": 0.0,
                "LPIPS": 0.0,
                "Evaluation Time": 0.0,
                "Num Gaussians": 0,
            },
        )

    @torch.no_grad
    def pause_for_eval(self):
        self._splat_model_view.allow_enable_in_viewer = False
        self._splat_model_view.enabled = False
        self._training_metrics_view["Status"] = "**Paused for Evaluation**"

    @torch.no_grad
    def resume_after_eval(self):
        self._splat_model_view.allow_enable_in_viewer = True
        self._splat_model_view.enabled = True
        del self._training_metrics_view["Status"]

    @torch.no_grad
    def set_sh_basis_to_view(self, sh_degree: int):
        """
        Set the degree of the spherical harmonics to use in the viewer.

        Args:
            sh_degree: The spherical harmonics degree to view.
        """
        self._splat_model_view.sh_degree = sh_degree

    @torch.no_grad
    def update_camera_poses(self, cam_to_world_matrices: torch.Tensor, image_ids: torch.Tensor):
        """
        Update camera poses in the viewer corresponding to the given image IDs

        Args:
            cam_to_world_matrices: A tensor of shape (B, 4, 4) containing camera-to-world matrices.
            image_ids: A tensor of shape (B,) containing image IDs of the cameras in the training set to update.
        """
        for i in range(len(cam_to_world_matrices)):
            cam_to_world_matrix = cam_to_world_matrices[i].cpu().numpy()
            image_id = int(image_ids[i].item())
            self._train_camera_view[image_id].cam_to_world_matrix = cam_to_world_matrix

    @torch.no_grad
    def log_evaluation_iteration(
        self, step: int, psnr: float, ssim: float, lpips: float, average_time_per_img: float, num_gaussians: int
    ):
        """
        Log data for a single evaluation step to the viewer.

        Args:
            step: The training step after which the evaluation was performed.
            psnr: Peak Signal-to-Noise Ratio for the evaluation (averaged over all images in the validation set).
            ssim: Structural Similarity Index Measure for the evaluation (averaged over all images in the validation set).
            lpips: Learned Perceptual Image Patch Similarity for the evaluation (averaged over all images in the validation set).
            average_time_per_img: Average time taken to evaluate each image.
            num_gaussians: Number of Gaussians in the model at this evaluation step.
        """
        self._evaluation_metrics_view["Last Evaluation Step"] = step
        self._evaluation_metrics_view["PSNR"] = psnr
        self._evaluation_metrics_view["SSIM"] = ssim
        self._evaluation_metrics_view["LPIPS"] = lpips
        self._evaluation_metrics_view["Average Time Per Image (s)"] = average_time_per_img
        self._evaluation_metrics_view["Num Gaussians"] = num_gaussians

    @torch.no_grad
    def log_training_iteration(
        self,
        step: int,
        loss: float,
        l1loss: float,
        ssimloss: float,
        mem: float,
        num_gaussians: int,
        current_sh_degree: int,
        pose_regulation: float | None,
    ):
        """
        Log data for a single training step to the viewer.

        Args:
            step: The current training step.
            loss: Total loss value for the training step.
            l1loss: L1 loss value for the training step.
            ssimloss: SSIM loss value for the training step.
            mem: Maximum GPU memory allocated in GB during this step.
            num_gaussians: Number of Gaussians in the model at this training step.
            current_sh_degree: Current degree of spherical harmonics used in the
            pose_regulation: Pose optimization regularization loss, if applicable.
        """

        self._training_metrics_view["Current Iteration"] = step
        self._training_metrics_view["Current SH Degree"] = current_sh_degree
        self._training_metrics_view["Num Gaussians"] = num_gaussians
        self._training_metrics_view["Loss"] = loss
        self._training_metrics_view["SSIM Loss"] = ssimloss
        self._training_metrics_view["L1 Loss"] = l1loss
        self._training_metrics_view["GPU Memory Usage"] = f"{mem:3.2f} GiB"
        if pose_regulation is not None:
            self._training_metrics_view["Pose Regularization"] = f"{pose_regulation:.3e}"
        else:
            if "Pose Regularization" in self._training_metrics_view:
                # Remove the pose regularization key if it was previously set
                del self._training_metrics_view["Pose Regularization"]


class SceneOptimizationRunner:
    """Engine for training and testing."""

    __PRIVATE__ = object()

    def _save_statistics(self, step: int, stage: str, stats: dict) -> None:
        """
        Save statistics in a dict to a JSON file.

        Args:
            step: The current training step.
            stage: The stage of training (e.g., "train", "eval").
            stats: A dictionary containing statistics to save.
        """
        if self._stats_path is None:
            self._logger.info("No stats path specified, skipping statistics save.")
            return
        stats_path = self._stats_path / pathlib.Path(f"stats_{stage}_{step:04d}.json")

        self._logger.info(f"Saving {stage} statistics at step {step} to path {stats_path}.")

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

    def _save_rendered_image(
        self, step: int, stage: str, image_name: str, predicted_image: torch.Tensor, ground_truth_image: torch.Tensor
    ):
        """
        Save a rendered image and its ground truth image to the evaluation renders directory.

        The rendered image and ground truth image are concatenated horizontally and saved as a single image file.

        Args:
            step: The current training step.
            stage: The stage of training (e.g., "train", "eval").
            image_name: The name of the image file to save.
            predicted_image: The predicted image tensor to save.
            ground_truth_image: The ground truth image tensor to save.
        """
        if self._image_render_path is None:
            self._logger.debug("No image render path specified, skipping image save.")
            return
        eval_render_directory_path = self._image_render_path / pathlib.Path(f"{stage}_{step:04d}")
        eval_render_directory_path.mkdir(parents=True, exist_ok=True)
        image_path = eval_render_directory_path / pathlib.Path(image_name)
        self._logger.info(f"Saving {stage} image at step {step} to {image_path}")
        canvas = torch.cat([predicted_image, ground_truth_image], dim=2).squeeze(0).cpu().numpy()
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(image_path),
            (canvas * 255).astype(np.uint8),
        )

    @staticmethod
    def _make_or_get_results_directories(
        run_name: str | None,
        save_results: bool,
        results_base_path: pathlib.Path,
        save_eval_images: bool,
        exists_ok: bool = True,
    ) -> tuple[str, pathlib.Path | None, pathlib.Path | None, pathlib.Path | None, pathlib.Path | None]:
        """
        Create or get the paths to the results directories for the training run.

        Args:
            run_name (str | None): The name of the run. If None, a unique name will be generated.
            save_results (bool): Whether to save results to disk.
            results_base_path (pathlib.Path): The base path where results will be saved.
            save_eval_images (bool): Whether to save evaluation images during training.
            exists_ok (bool): If True, will not raise an error if the run name already exists.

        Returns:
            run_name (str): The name of the run.
            eval_render_path (pathlib.Path | None): Path to save evaluation renders, or None if not saving.
            stats_path (pathlib.Path): Path to save statistics.
            checkpoints_path (pathlib.Path): Path to save model checkpoints.
            tensorboard_path (pathlib.Path): Path to save TensorBoard logs.
        """
        logger = logging.getLogger(f"{__name__}.SceneOptimizationRunner")

        if not save_results:
            logger.info("No results will be saved. You can set `save_results=True` to save the training run.")
            # If no results are saved and you didn't pass a run name, we'll generate a unique one
            if run_name is None:
                run_name = str(uuid.uuid4())
                logger.info(f"Generated a unique run name '{run_name}' for this run.")
            return run_name, None, None, None, None

        results_base_path.mkdir(exist_ok=True)

        if run_name is None:
            logger.info("No run name provided. Creating a new run directory.")
            run_name, results_path = make_unique_name_directory_based_on_time(results_base_path, prefix="run")
        else:
            results_path = results_base_path / pathlib.Path(run_name)
            if not results_path.exists():
                logger.info(
                    f"Run name {run_name} does not exist in results path {results_base_path}. Creating new run directory."
                )
                results_path.mkdir(exist_ok=True)
            else:
                if not exists_ok:
                    raise FileExistsError(
                        f"Run name {run_name} already exists in results path {results_base_path}. "
                        "Please provide a different run name or set exists_ok=True."
                    )
                logger.info(f"Using existing run name {run_name} in results path {results_base_path}.")
                logger.info(f"Results will be saved to {results_path.absolute()}.")

        eval_render_path = None
        if save_eval_images:
            eval_render_path = results_path / pathlib.Path("eval_renders")
            eval_render_path.mkdir(exist_ok=True)

        stats_path = results_path / pathlib.Path("stats")
        stats_path.mkdir(exist_ok=True)

        checkpoints_path = results_path / pathlib.Path("checkpoints")
        checkpoints_path.mkdir(exist_ok=True)

        tensorboard_path = results_path / pathlib.Path("tb")
        tensorboard_path.mkdir(exist_ok=True)

        return run_name, eval_render_path, stats_path, checkpoints_path, tensorboard_path

    @property
    def checkpoint(self):
        """
        Return a Checkpoint object containing the current training state.

        This includes the model, optimizer, configuration, pose adjustment model, and datasets.

        Returns:
            Checkpoint: A Checkpoint object containing the current training state.
        """
        return Checkpoint(
            step=self._global_step,
            run_name=self.run_name,
            model=self.model,
            dataset_transform=self._dataset_transform,
            dataset_path=self._dataset_path,
            dataset_splits={"train": self._training_dataset.indices, "val": self._validation_dataset.indices},
            optimizer=self.optimizer,
            config=vars(self.config),
            pose_adjust_model=self.pose_adjust_model,
            pose_adjust_optimizer=self.pose_adjust_optimizer,
            pose_adjust_scheduler=self.pose_adjust_scheduler,
        )

    @torch.no_grad()
    def _splat_metadata(self) -> dict[str, torch.Tensor | float | int | str]:
        training_camera_to_world_matrices = torch.from_numpy(self._training_dataset.camera_to_world_matrices).to(
            dtype=torch.float32, device=self.device
        )
        if self.pose_adjust_model is not None:
            training_camera_to_world_matrices = self.pose_adjust_model(
                training_camera_to_world_matrices, torch.arange(len(self.training_dataset), device=self.device)
            )

        # Save projection parameters as a per-camera tuple (fx, fy, cx, cy, h, w)
        training_projection_matrices = torch.from_numpy(self._training_dataset.projection_matrices.astype(np.float32))
        training_image_sizes = torch.from_numpy(self._training_dataset.image_sizes.astype(np.int32))
        normalization_transform = torch.from_numpy(self.training_dataset.sfm_scene.transformation_matrix).to(
            torch.float32
        )

        return {
            "normalization_transform": normalization_transform,
            "camera_to_world_matrices": training_camera_to_world_matrices,
            "projection_matrices": training_projection_matrices,
            "image_sizes": training_image_sizes,
            "scene_scale": SceneOptimizationRunner._compute_scene_scale(self.training_dataset.sfm_scene),
            "eps2d": self.config.eps_2d,
            "near_plane": self.config.near_plane,
            "far_plane": self.config.far_plane,
            "min_radius_2d": self.config.min_radius_2d,
            "antialias": int(self.config.antialias),
            "tile_size": self.config.tile_size,
        }

    @torch.no_grad()
    def _save_checkpoint_and_ply(self, ckpt_path: pathlib.Path, ply_path: pathlib.Path):
        """
        Saves a checkpoint and a PLY file to disk
        """
        if self._checkpoints_path is None:
            return

        self.checkpoint.save(ckpt_path)

        self.model.save_ply(
            ply_path,
            metadata=self._splat_metadata(),
        )

    @property
    def config(self) -> Config:
        """
        Get the configuration object for the current training run.

        Returns:
            Config: The configuration object containing all parameters for the training run.
        """
        return self._cfg

    @property
    def run_name(self) -> str:
        """
        Get the name of the current run.

        Returns:
            str | None: The name of the run, or None if no run name is set.
        """
        return self._run_name

    @property
    def model(self) -> GaussianSplat3d:
        """
        Get the Gaussian Splatting model being trained.

        Returns:
            GaussianSplat3d: The model instance.
        """
        return self._model

    @property
    def optimizer(self) -> GaussianSplatOptimizer:
        """
        Get the optimizer used for training the Gaussian Splatting model.

        Returns:
            GaussianSplatOptimizer: The optimizer instance.
        """
        return self._optimizer

    @property
    def pose_adjust_model(self) -> CameraPoseAdjustment | None:
        """
        Get the camera pose adjustment model used for optimizing camera poses during training.

        Returns:
            CameraPoseAdjustment | None: The pose adjustment model instance, or None if not used.
        """
        return self._pose_adjust_model

    @property
    def pose_adjust_optimizer(self) -> torch.optim.Adam | None:
        """
        Get the optimizer used for adjusting camera poses during training.

        Returns:
            torch.optim.Optimizer | None: The pose adjustment optimizer instance, or None if not used.
        """
        return self._pose_adjust_optimizer

    @property
    def pose_adjust_scheduler(self) -> torch.optim.lr_scheduler.ExponentialLR | None:
        """
        Get the learning rate scheduler used for adjusting camera poses during training.

        Returns:
            torch.optim.lr_scheduler.ExponentialLR | None: The pose adjustment scheduler instance, or None if not used.
        """
        return self._pose_adjust_scheduler

    @property
    def training_dataset(self) -> SfmDataset:
        """
        Get the training dataset used for training the Gaussian Splatting model.

        Returns:
            SfmDataset: The training dataset instance.
        """
        return self._training_dataset

    @property
    def validation_dataset(self) -> SfmDataset:
        """
        Get the validation dataset used for evaluating the Gaussian Splatting model.

        Returns:
            SfmDataset: The validation dataset instance.
        """
        return self._validation_dataset

    @property
    def stats_path(self) -> pathlib.Path | None:
        """
        Get the path where training statistics are saved.

        Returns:
            pathlib.Path | None: The path to the statistics directory, or None if not set.
        """
        return self._stats_path

    @property
    def image_render_path(self) -> pathlib.Path | None:
        """
        Get the path where rendered images are saved during evaluation.

        Returns:
            pathlib.Path | None: The path to the evaluation renders directory, or None if not set.
        """
        return self._image_render_path

    @property
    def checkpoints_path(self) -> pathlib.Path | None:
        """
        Get the path where model checkpoints are saved.

        Returns:
            pathlib.Path | None: The path to the checkpoints directory, or None if not set.
        """
        return self._checkpoints_path

    @staticmethod
    def _init_model(
        config: Config,
        device: torch.device | str,
        training_dataset: SfmDataset,
    ):
        """
        Initialize the Gaussian Splatting model with random parameters based on the training dataset.

        Args:
            config: Configuration object containing model parameters.
            device: The device to run the model on (e.g., "cuda" or "cpu").
            training_dataset: The dataset used for training, which provides the initial points and RGB values
                            for the Gaussians.
        """

        def _knn(x_np: np.ndarray, k: int = 4) -> torch.Tensor:
            kd_tree = cKDTree(x_np)
            distances, _ = kd_tree.query(x_np, k=k)
            return torch.from_numpy(distances).to(device=device, dtype=torch.float32)

        def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
            C0 = 0.28209479177387814
            return (rgb - 0.5) / C0

        num_gaussians = training_dataset.points.shape[0]

        dist2_avg = (_knn(training_dataset.points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        log_scales = torch.log(dist_avg * config.initial_covariance_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

        means = torch.from_numpy(training_dataset.points).to(device=device, dtype=torch.float32)  # [N, 3]
        quats = torch.rand((num_gaussians, 4), device=device)  # [N, 4]
        logit_opacities = torch.logit(torch.full((num_gaussians,), config.initial_opacity, device=device))  # [N,]

        rgbs = torch.from_numpy(training_dataset.points_rgb / 255.0).to(device=device, dtype=torch.float32)  # [N, 3]
        sh_0 = _rgb_to_sh(rgbs).unsqueeze(1)  # [N, 1, 3]

        sh_n = torch.zeros((num_gaussians, (config.sh_degree + 1) ** 2 - 1, 3), device=device)  # [N, K-1, 3]

        model = GaussianSplat3d(means, quats, log_scales, logit_opacities, sh_0, sh_n, True)
        model.requires_grad = True

        if config.refine_using_scale2d_stop_epoch > 0:
            model.accumulate_max_2d_radii = True

        return model

    @staticmethod
    def _compute_scene_scale(sfm_scene: SfmScene, use_sfm_depths=True) -> float:
        """
        Compute a measure of the "scale" of a scene. I.e. how far away objects of interest are from
        the cameras in the capture.

        Args:
            sfm_scene (SfmScene): The scene loaded from an structure-from-motion (SfM) pipeline.
            use_sfm_depths (bool): Whether to use the SfM depths for scale estimation (True by default).

        Returns:
            scene_scale (float): An estimate of how far objects in the scene are from the cameras that captured them
        """
        if use_sfm_depths:
            # Estimate the scene scale as the median across the median distances from cameras to the
            # sfm points they see. If there is not too much variance in how far the cameras are from the scene
            # this gives a rough estimate of the scene scale.
            median_depth_per_camera = []
            for image_meta in sfm_scene.images:
                # Don't use cameras that don't see any points in the estimate
                if len(image_meta.point_indices) == 0:
                    continue
                points = sfm_scene.points[image_meta.point_indices]
                dist_to_points = np.linalg.norm(points - image_meta.origin, axis=1)
                median_dist = np.median(dist_to_points)
                median_depth_per_camera.append(median_dist)
            return float(np.median(median_depth_per_camera))
        else:
            # The old way used the maximum distance from any camera to the centroid of all cameras
            # which worked well for orbit scans with a central point of interest but not so much
            # for other types of capture (e.g. drone footage).
            # This code is around as a reference and so we can compare the new behavior to the old
            # but is not used
            origins = np.stack([cam.origin for cam in sfm_scene.images], axis=0)
            centroid = np.mean(origins, axis=0)
            dists = np.linalg.norm(origins - centroid, axis=1)
            return np.max(dists)

    @staticmethod
    def new_run(
        dataset_path: str | pathlib.Path,
        config: Config = Config(),
        run_name: str | None = None,
        image_downsample_factor: int = 4,
        points_percentile_filter: float = 0.0,
        normalization_type: Literal["none", "pca", "ecef2enu", "similarity"] = "pca",
        crop_bbox: tuple[float, float, float, float, float, float] | None = None,
        min_points_per_image: int = 5,
        results_path: str | pathlib.Path = pathlib.Path("results"),
        device: str | torch.device = "cuda",
        use_every_n_as_val: int = 100,
        disable_viewer: bool = False,
        log_tensorboard_every: int = 100,
        log_images_to_tensorboard: bool = False,
        save_eval_images: bool = False,
        save_results: bool = True,
    ) -> "SceneOptimizationRunner":
        """
        Create a `Runner` instance for a new training run.

        Args:
            dataset_path (str | pathlib.Path): Path to the dataset directory containing the SFM data.
            config (Config): Configuration object containing model parameters.
            run_name (str | None): Optional name for the run. If None, a unique name will be generated.
                If a run with the same name already exists, an exception will be raised.
            image_downsample_factor (int): Factor by which to downsample the images for training.
            points_percentile_filter (float): Percentile filter to apply to the points in the dataset (in [0, 100]).
            normalization_type (Literal["none", "pca", "ecef2enu", "similarity"]): Type of normalization to apply to the scene data.
            crop_bbox (tuple[float, float, float, float, float, float] | None): Optional bounding box to crop the scene data.
                In the form [x_min, y_min, z_min, x_max, y_max, z_max].
                If None, no cropping will be applied.
            min_points_per_image (int): Minimum number of points that must be visible in an image for it to be included in the dataset.
            results_path (str | pathlib.Path): Base path where results will be saved.
            device (str | torch.device): The device to run the model on (e.g., "cuda" or "cpu").
            use_every_n_as_val (int): How often to use a training image as a validation image
            disable_viewer (bool): Whether to disable the viewer for this run.
            log_tensorboard_every (int): How often to log metrics to TensorBoard.
            log_images_to_tensorboard (bool): Whether to log images to TensorBoard.
            save_eval_images (bool): Whether to save evaluation images during training.
            save_results (bool): Whether to save results to disk.

        Returns:
            Runner: A `Runner` instance initialized with the specified configuration and datasets.
        """
        if isinstance(dataset_path, str):
            dataset_path = pathlib.Path(dataset_path)

        if isinstance(results_path, str):
            results_path = pathlib.Path(results_path)

        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        logger = logging.getLogger(f"{__name__}.SceneOptimizationRunner")

        # Dataset transform
        transforms = [
            NormalizeScene(normalization_type=normalization_type),
            PercentileFilterPoints(
                percentile_min=np.full((3,), points_percentile_filter),
                percentile_max=np.full((3,), 100.0 - points_percentile_filter),
            ),
            DownsampleImages(
                image_downsample_factor=image_downsample_factor,
            ),
            FilterImagesWithLowPoints(min_num_points=min_points_per_image),
        ]
        if crop_bbox is not None:
            transforms.append(CropScene(crop_bbox))
        transform = Compose(*transforms)

        sfm_scene: SfmScene = SfmScene.from_colmap(dataset_path)
        sfm_scene = transform(sfm_scene)

        indices = np.arange(sfm_scene.num_images)
        if use_every_n_as_val > 0:
            mask = np.ones(len(indices), dtype=bool)
            mask[::use_every_n_as_val] = False
            train_indices = indices[mask]
            val_indices = indices[~mask]
        else:
            train_indices = indices
            val_indices = np.array([], dtype=int)

        train_dataset = SfmDataset(sfm_scene, train_indices)
        val_dataset = SfmDataset(sfm_scene, val_indices)

        logger.info(
            f"Created dataset training and test datasets with {len(train_dataset)} training images and {len(val_dataset)} test images."
        )

        # Initialize model
        model = SceneOptimizationRunner._init_model(config, device, train_dataset)
        logger.info(f"Model initialized with {model.num_gaussians:,} Gaussians")

        # Initialize optimizer
        max_steps = config.max_epochs * len(train_dataset)
        optimizer = GaussianSplatOptimizer(
            model,
            scene_scale=SceneOptimizationRunner._compute_scene_scale(train_dataset.sfm_scene) * 1.1,
            mean_lr_decay_exponent=0.01 ** (1.0 / max_steps),
        )

        # Optional camera position optimizer
        pose_adjust_optimizer = None
        pose_adjust_model = None
        pose_adjust_scheduler = None
        if config.optimize_camera_poses:
            # Module to adjust camera poses during training
            pose_adjust_model = CameraPoseAdjustment(len(train_dataset), init_std=config.pose_opt_init_std).to(device)

            # Increase learning rate for pose optimization and add gradient clipping
            pose_adjust_optimizer = torch.optim.Adam(
                pose_adjust_model.parameters(),
                lr=config.pose_opt_lr * 100.0,
                weight_decay=config.pose_opt_reg,
            )

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(pose_adjust_model.parameters(), max_norm=1.0)

            # Add learning rate scheduler for pose optimization
            pose_adjust_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                pose_adjust_optimizer, gamma=config.pose_opt_lr_decay ** (1.0 / max_steps)
            )

        # Setup output directories.
        run_name, image_render_path, stats_path, checkpoints_path, tensorboard_path = (
            SceneOptimizationRunner._make_or_get_results_directories(
                run_name=run_name,
                results_base_path=results_path,
                save_results=save_results,
                save_eval_images=save_eval_images,
                exists_ok=False,
            )
        )

        return SceneOptimizationRunner(
            config=config,
            dataset_path=dataset_path,
            sfm_scene=sfm_scene,
            dataset_transform=transform,
            train_indices=train_indices,
            val_indices=val_indices,
            model=model,
            optimizer=optimizer,
            pose_adjust_model=pose_adjust_model,
            pose_adjust_optimizer=pose_adjust_optimizer,
            pose_adjust_scheduler=pose_adjust_scheduler,
            start_step=0,
            run_name=run_name,
            image_render_path=image_render_path,
            stats_path=stats_path,
            checkpoints_path=checkpoints_path,
            tensorboard_path=tensorboard_path,
            log_tensorboard_every=log_tensorboard_every,
            log_images_to_tensorboard=log_images_to_tensorboard,
            disable_viewer=disable_viewer,
            _private=SceneOptimizationRunner.__PRIVATE__,
        )

    @staticmethod
    def from_checkpoint(
        checkpoint: Checkpoint,
        results_path: pathlib.Path | str = pathlib.Path("results"),
        disable_viewer: bool = False,
        log_tensorboard_every: int = 100,
        log_images_to_tensorboard: bool = False,
        save_eval_images: bool = False,
        save_results: bool = True,
    ) -> "SceneOptimizationRunner":
        """
        Create a `Runner` instance from a saved checkpoint.

        Args:
            checkpoint (Checkpoint): The checkpoint to load from.
            results_path (pathlib.Path | str): Base path where results will be saved.
            disable_viewer (bool): Whether to disable the viewer for this run.
            log_tensorboard_every (int): How often to log metrics to TensorBoard.
            log_images_to_tensorboard (bool): Whether to log images to TensorBoard.
            save_results (bool): Whether to save results to disk.
            save_eval_images (bool): Whether to save evaluation images during training.
        """
        if isinstance(results_path, str):
            results_path = pathlib.Path(results_path)

        logger = logging.getLogger(f"{__name__}.SceneOptimizationRunner")
        config = Config(**checkpoint.config)

        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        if not checkpoint.dataset_path.exists():
            raise FileNotFoundError(f"Checkpoint dataset path {checkpoint.dataset_path} does not exist.")

        sfm_scene: SfmScene = SfmScene.from_colmap(checkpoint.dataset_path)
        sfm_scene = checkpoint.dataset_transform(sfm_scene)

        if "train" not in checkpoint.dataset_splits:
            raise ValueError("Checkpoint does not have 'train' split")
        if "val" not in checkpoint.dataset_splits:
            raise ValueError("Checkpoint does not have 'val' split")

        train_indices = checkpoint.dataset_splits["train"]
        val_indices = checkpoint.dataset_splits["val"]

        logger.info(f"Loaded checkpoint with {checkpoint.splats.num_gaussians:,} Gaussians.")

        # Setup output directories.
        run_name, image_render_path, stats_path, checkpoints_path, tensorboard_path = (
            SceneOptimizationRunner._make_or_get_results_directories(
                run_name=checkpoint.run_name,
                save_results=save_results,
                results_base_path=results_path,
                save_eval_images=save_eval_images,
            )
        )

        return SceneOptimizationRunner(
            config=config,
            dataset_path=checkpoint.dataset_path,
            sfm_scene=sfm_scene,
            dataset_transform=checkpoint.dataset_transform,
            train_indices=train_indices,
            val_indices=val_indices,
            model=checkpoint.splats,
            optimizer=checkpoint.optimizer,
            pose_adjust_model=checkpoint.pose_adjust_model,
            pose_adjust_optimizer=checkpoint.pose_adjust_optimizer,
            pose_adjust_scheduler=checkpoint.pose_adjust_scheduler,
            start_step=checkpoint.step,
            run_name=run_name,
            image_render_path=image_render_path,
            stats_path=stats_path,
            checkpoints_path=checkpoints_path,
            tensorboard_path=tensorboard_path,
            log_tensorboard_every=log_tensorboard_every,
            log_images_to_tensorboard=log_images_to_tensorboard,
            disable_viewer=disable_viewer,
            _private=SceneOptimizationRunner.__PRIVATE__,
        )

    def __init__(
        self,
        config: Config,
        dataset_path: pathlib.Path,
        sfm_scene: SfmScene,
        dataset_transform: BaseTransform,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        model: GaussianSplat3d,
        optimizer: GaussianSplatOptimizer,
        pose_adjust_model: CameraPoseAdjustment | None,
        pose_adjust_optimizer: torch.optim.Adam | None,
        pose_adjust_scheduler: torch.optim.lr_scheduler.ExponentialLR | None,
        start_step: int,
        run_name: str,
        image_render_path: pathlib.Path | None,
        stats_path: pathlib.Path | None,
        checkpoints_path: pathlib.Path | None,
        tensorboard_path: pathlib.Path | None,
        log_tensorboard_every: int,
        log_images_to_tensorboard: bool,
        disable_viewer: bool,
        _private: object | None = None,
    ) -> None:
        """
        Initialize the Runner with the provided configuration, model, optimizer, datasets, and paths.

        Note: This constructor should only be called by the `new_run` or `resume_from_checkpoint` methods.

        Args:
            config (Config): Configuration object containing model parameters.
            sfm_scene (SfmScene): The Structure-from-Motion scene.
            dataset_transform (BaseTransform): The transform used to normalize/scale/resample the SfmScene.
            train_indices (np.ndarray): The indices for the training set.
            val_indices (np.ndarray): The indices for the validation set.
            model (GaussianSplat3d): The Gaussian Splatting model to train.
            optimizer (GaussianSplatOptimizer | None): The optimizer for the model if training.
                Note: You can pass in a None optimizer if you want to use the model only for evaluation.
            pose_adjust_model (CameraPoseAdjustment | None): The camera pose adjustment model, if used
            pose_adjust_optimizer (torch.optim.Adam | None): The optimizer for camera pose adjustment, if used.
            pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None): The learning rate scheduler
                for camera pose adjustment, if used.
            start_step (int): The step to start training from (useful for resuming training
                from a checkpoint).
            run_name (str | None): The name of the training run or None for an un-named run.
            image_render_path (pathlib.Path | None): Path to save rendered images during evaluation.
            stats_path (pathlib.Path | None): Path to save training statistics
            checkpoints_path (pathlib.Path | None): Path to save model checkpoints.
            tensorboard_path (pathlib.Path | None): Path to save TensorBoard logs.
            log_tensorboard_every (int): How often to log metrics to TensorBoard.
            log_images_to_tensorboard (bool): Whether to log images to TensorBoard.
            disable_viewer (bool): Whether to disable the viewer for this run.
            _private (object | None): Private object to ensure this class is only initialized through `new_run` or `resume_from_checkpoint`.
        """
        if _private is not SceneOptimizationRunner.__PRIVATE__:
            raise ValueError("Runner should only be initialized through `new_run` or `resume_from_checkpoint`.")

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._cfg = config
        self._model = model
        self._optimizer = optimizer
        self._pose_adjust_model = pose_adjust_model
        self._pose_adjust_optimizer = pose_adjust_optimizer
        self._pose_adjust_scheduler = pose_adjust_scheduler
        self._start_step = start_step

        self._sfm_scene = sfm_scene
        self._dataset_transform = dataset_transform
        self._dataset_path = dataset_path
        self._training_dataset = SfmDataset(sfm_scene=sfm_scene, dataset_indices=train_indices)
        self._validation_dataset = SfmDataset(sfm_scene=sfm_scene, dataset_indices=val_indices)

        self.device = model.device

        self._run_name = run_name
        self._image_render_path = image_render_path
        self._stats_path = stats_path
        self._checkpoints_path = checkpoints_path

        self._global_step: int = 0

        # Tensorboard
        self._tensorboard_logger = None
        if tensorboard_path is not None and optimizer is not None:
            self._tensorboard_logger = TensorboardLogger(
                log_dir=tensorboard_path,
                log_every_step=log_tensorboard_every,
                log_images_to_tensorboard=log_images_to_tensorboard,
            )

        # Viewer
        self._viewer = ViewerLogger(self.model, self._training_dataset) if not disable_viewer else None

        # Losses & Metrics.
        if self.config.lpips_net == "alex":
            self._lpips = LPIPSLoss(backbone="alex").to(model.device)
        elif self.config.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self._lpips = LPIPSLoss(backbone="vgg").to(model.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {self.config.lpips_net}")

    def train(self) -> tuple[GaussianSplat3d, dict[str, torch.Tensor | float | int | str]]:
        """
        Run the training loop for the Gaussian Splatting model.

        This method initializes the training data loader, sets up the training loop, and performs optimization steps
        for the model. It also handles camera pose optimization if enabled, and logs training metrics to
        TensorBoard and the viewer.

        The training loop iterates over the training dataset, computes losses, updates model parameters,
        and logs metrics at each step. It also handles progressive refinement of the model based on the
        configured epochs and steps.

        The training process includes:
        - Loading training data in batches.
        - Performing camera pose optimization if enabled.
        - Rendering images from the model's projected Gaussians.
        - Computing losses (L1, SSIM, LPIPS) and updating model parameters.
        - Logging training metrics to TensorBoard and the viewer.

        Returns:
            Checkpoint: A checkpoint object containing the current training state, including the model, optimizer,
            and training configuration. This can be used to save the current state of the training process
            or resume training later.
        """
        if self.optimizer is None:
            raise ValueError("This runner was not created with an optimizer. Cannot run training.")

        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )

        # Calculate total steps, allowing max_steps to override the computed value
        computed_total_steps: int = int(self.config.max_epochs * len(self.training_dataset))
        total_steps: int = self.config.max_steps if self.config.max_steps is not None else computed_total_steps

        refine_start_step: int = int(self.config.refine_start_epoch * len(self.training_dataset))
        refine_stop_step: int = int(self.config.refine_stop_epoch * len(self.training_dataset))
        refine_every_step: int = int(self.config.refine_every_epoch * len(self.training_dataset))
        reset_opacities_every_step: int = int(self.config.reset_opacities_every_epoch * len(self.training_dataset))
        refine_using_scale2d_stop_step: int = int(
            self.config.refine_using_scale2d_stop_epoch * len(self.training_dataset)
        )
        increase_sh_degree_every_step: int = int(
            self.config.increase_sh_degree_every_epoch * len(self.training_dataset)
        )
        pose_opt_stop_step: int = int(self.config.pose_opt_stop_epoch * len(self.training_dataset))

        # Progress bar to track training progress
        if self.config.max_steps is not None:
            self._logger.info(
                f"Using max_steps={self.config.max_steps} (overriding computed {computed_total_steps} steps)"
            )
        pbar = tqdm.tqdm(range(0, total_steps), unit="imgs", desc="Training")

        # Flag to break out of outer epoch loop when max_steps is reached
        reached_max_steps = False

        # Zero out gradients before training in case we resume training
        self.optimizer.zero_grad()
        if self.pose_adjust_optimizer is not None:
            self.pose_adjust_optimizer.zero_grad()

        for epoch in range(self.config.max_epochs):
            for minibatch in trainloader:
                batch_size = minibatch["image"].shape[0]

                # Skip steps before the start step
                if self._global_step < self._start_step:
                    pbar.set_description(
                        f"Skipping step {self._global_step:,} (before start step {self._start_step:,})"
                    )
                    pbar.update(batch_size)
                    self._global_step = pbar.n
                    continue
                if self._viewer is not None:
                    self._viewer.viewer.acquire_lock()

                cam_to_world_mats: torch.Tensor = minibatch["camera_to_world"].to(self.device)  # [B, 4, 4]
                world_to_cam_mats: torch.Tensor = minibatch["world_to_camera"].to(self.device)  # [B, 4, 4]

                # Camera pose optimization
                image_ids = minibatch["image_id"].to(self.device)  # [B]
                if self.pose_adjust_model is not None:
                    if self._global_step < pose_opt_stop_step:
                        cam_to_world_mats = self.pose_adjust_model(cam_to_world_mats, image_ids)
                    else:
                        # After pose_opt_stop_iter, don't track gradients through pose adjustment
                        with torch.no_grad():
                            cam_to_world_mats = self.pose_adjust_model(cam_to_world_mats, image_ids)

                projection_mats = minibatch["projection"].to(self.device)  # [B, 3, 3]
                image = minibatch["image"]  # [B, H, W, 3]
                mask = minibatch["mask"] if "mask" in minibatch and not self.config.ignore_masks else None
                image_height, image_width = image.shape[1:3]

                # Progressively use higher spherical harmonic degree as we optimize
                sh_degree_to_use = min(self._global_step // increase_sh_degree_every_step, self.config.sh_degree)
                projected_gaussians = self.model.project_gaussians_for_images(
                    world_to_cam_mats,
                    projection_mats,
                    image_width,
                    image_height,
                    self.config.near_plane,
                    self.config.far_plane,
                    "perspective",
                    sh_degree_to_use,
                    self.config.min_radius_2d,
                    self.config.eps_2d,
                    self.config.antialias,
                )

                # If you have very large images, you can iterate over disjoint crops and accumulate gradients
                # If cfg.crops_per_image is 1, then this just returns the image
                for pixels, mask_pixels, crop, is_last in crop_image_batch(image, mask, self.config.crops_per_image):
                    # Actual pixels to compute the loss on, normalized to [0, 1]
                    pixels = pixels.to(self.device) / 255.0  # [1, H, W, 3]

                    # Render an image from the gaussian splats
                    # possibly using a crop of the full image
                    crop_origin_w, crop_origin_h, crop_w, crop_h = crop
                    colors, alphas = self.model.render_from_projected_gaussians(
                        projected_gaussians, crop_w, crop_h, crop_origin_w, crop_origin_h, self.config.tile_size
                    )
                    # If you want to add random background, we'll mix it in here
                    if self.config.random_bkgd:
                        bkgd = torch.rand(1, 3, device=self.device)
                        colors = colors + bkgd * (1.0 - alphas)

                    if mask_pixels is not None:
                        # set the ground truth pixel values to match render, thus loss is zero at mask pixels and not updated
                        mask_pixels = mask_pixels.to(self.device)
                        pixels[~mask_pixels] = colors.detach()[~mask_pixels]

                    # Image losses
                    l1loss = F.l1_loss(colors, pixels)
                    ssimloss = 1.0 - ssim(
                        colors.permute(0, 3, 1, 2).contiguous(),
                        pixels.permute(0, 3, 1, 2).contiguous(),
                    )
                    loss = l1loss * (1.0 - self.config.ssim_lambda) + ssimloss * self.config.ssim_lambda

                    # Rgularize opacity to ensure Gaussian's don't become too opaque
                    if self.config.opacity_reg > 0.0:
                        loss = loss + self.config.opacity_reg * torch.abs(self.model.opacities).mean()

                    # Regularize scales to ensure Gaussians don't become too large
                    if self.config.scale_reg > 0.0:
                        loss = loss + self.config.scale_reg * torch.abs(self.model.scales).mean()

                    # If you're optimizing poses, regularize the pose parameters so the poses
                    # don't drift too far from the initial values
                    if self.pose_adjust_model is not None and self._global_step < pose_opt_stop_step:
                        pose_params = self.pose_adjust_model.pose_embeddings(image_ids)
                        pose_reg = torch.mean(torch.abs(pose_params))
                        loss = loss + self.config.pose_opt_reg * pose_reg

                    # If we're splitting into crops, accumulate gradients, so pass retain_graph=True
                    # for every crop but the last one
                    loss.backward(retain_graph=not is_last)

                # Update the log in the progress bar
                pbar.set_description(
                    f"loss={loss.item():.3f}| "
                    f"sh degree={sh_degree_to_use}| "
                    f"num gaussians={self.model.num_gaussians:,}"
                )

                # Refine the gaussians via splitting/duplication/pruning
                if (
                    self._global_step > refine_start_step
                    and self._global_step % refine_every_step == 0
                    and self._global_step < refine_stop_step
                ):
                    num_gaussians_before: int = self.model.num_gaussians
                    use_scales_for_refinement: bool = self._global_step > reset_opacities_every_step
                    use_screen_space_scales_for_refinement: bool = self._global_step < refine_using_scale2d_stop_step
                    if not use_screen_space_scales_for_refinement:
                        self.model.accumulate_max_2d_radii = False
                    num_dup, num_split, num_prune = self.optimizer.refine_gaussians(
                        use_scales=use_scales_for_refinement,
                        use_screen_space_scales=use_screen_space_scales_for_refinement,
                    )
                    self._logger.debug(
                        f"Step {self._global_step:,}: Refinement: {num_dup:,} duplicated, {num_split:,} split, {num_prune:,} pruned. "
                        f"Num Gaussians: {self.model.num_gaussians:,} (before: {num_gaussians_before:,})"
                    )
                    # If you specified a crop bounding box, clip the Gaussians that are outside the crop
                    # bounding box. This is useful if you want to train on a subset of the scene
                    # and don't want to waste resources on Gaussians that are outside the crop.
                    if self.config.remove_gaussians_outside_scene_bbox:
                        bbox_min, bbox_max = self.training_dataset.scene_bbox
                        ng_prior = self.model.num_gaussians
                        points = self.model.means

                        outside_mask = torch.logical_or(points[:, 0] < bbox_min[0], points[:, 0] > bbox_max[0])
                        outside_mask = torch.logical_or(outside_mask, points[:, 1] < bbox_min[1])
                        outside_mask = torch.logical_or(outside_mask, points[:, 1] > bbox_max[1])
                        outside_mask = torch.logical_or(outside_mask, points[:, 2] < bbox_min[2])
                        outside_mask = torch.logical_or(outside_mask, points[:, 2] > bbox_max[2])

                        self.optimizer.remove_gaussians(outside_mask)
                        ng_post = self.model.num_gaussians
                        nclip = ng_prior - ng_post
                        self._logger.debug(
                            f"Clipped {nclip:,} Gaussians outside the crop bounding box min={bbox_min}, max={bbox_max}."
                        )

                # Reset the opacity parameters every so often
                if self._global_step % reset_opacities_every_step == 0 and self._global_step > 0:
                    self.optimizer.reset_opacities()

                # Step the Gaussian optimizer
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # If you enabled pose optimization, step the pose optimizer if we performed a
                # pose update this iteration
                if self.config.optimize_camera_poses and self._global_step < pose_opt_stop_step:
                    assert (
                        self.pose_adjust_optimizer is not None
                    ), "Pose optimizer should be initialized if pose optimization is enabled."
                    assert (
                        self.pose_adjust_scheduler is not None
                    ), "Pose scheduler should be initialized if pose optimization is enabled."
                    self.pose_adjust_optimizer.step()
                    self.pose_adjust_optimizer.zero_grad(set_to_none=True)
                    self.pose_adjust_scheduler.step()

                # Log to tensorboard if you requested it
                if self._tensorboard_logger is not None:
                    self._tensorboard_logger.log_training_iteration(
                        self._global_step,
                        self.model.num_gaussians,
                        loss.item(),
                        l1loss.item(),
                        ssimloss.item(),
                        torch.cuda.max_memory_allocated() / 1024**3,
                        pose_loss=pose_reg.item() if self.config.optimize_camera_poses else None,
                        gt_img=pixels,
                        pred_img=colors,
                    )

                # Update the viewer
                if self._viewer is not None:
                    self._viewer.viewer.release_lock()
                    self._viewer.log_training_iteration(
                        self._global_step,
                        loss=loss.item(),
                        l1loss=l1loss.item(),
                        ssimloss=ssimloss.item(),
                        mem=torch.cuda.max_memory_allocated() / 1024**3,
                        num_gaussians=self.model.num_gaussians,
                        current_sh_degree=sh_degree_to_use,
                        pose_regulation=pose_reg.item() if self.config.optimize_camera_poses else None,
                    )
                    if self.config.optimize_camera_poses:
                        self._viewer.update_camera_poses(cam_to_world_mats, image_ids)
                    if (
                        self._global_step % increase_sh_degree_every_step == 0
                        and sh_degree_to_use < self.config.sh_degree
                    ):
                        self._viewer.set_sh_basis_to_view(sh_degree_to_use)

                pbar.update(batch_size)
                self._global_step = pbar.n

                # Check if we've reached max_steps and break out of training
                if self.config.max_steps is not None and self._global_step >= self.config.max_steps:
                    reached_max_steps = True
                    break

            # Check if we've reached max_steps and break out of outer epoch loop
            if reached_max_steps:
                break

            # Save the model if we've reached a percentage of the total epochs specified in save_at_percent
            if epoch in [(pct * self.config.max_epochs // 100) - 1 for pct in self.config.save_at_percent]:
                if self._global_step <= self._start_step and self._checkpoints_path is not None:
                    self._logger.info(
                        f"Skipping checkpoint save at epoch {epoch + 1} (before start step {self._start_step})."
                    )
                    continue
                if self._checkpoints_path is not None:
                    ckpt_path = self._checkpoints_path / pathlib.Path(f"ckpt_{self._global_step:04d}.pt")
                    self._logger.info(f"Saving checkpoint at epoch {epoch + 1} to {ckpt_path}.")
                    ply_path = self._checkpoints_path / pathlib.Path(f"ckpt_{self._global_step:04d}.ply")
                    self._logger.info(f"Saving PLY file at epoch {epoch + 1} to {ply_path}.")
                    self._save_checkpoint_and_ply(ckpt_path, ply_path)

            # Run evaluation if we've reached a percentage of the total epochs specified in eval_at_percent
            if epoch in [(pct * self.config.max_epochs // 100) - 1 for pct in self.config.eval_at_percent]:
                if len(self.validation_dataset) == 0:
                    continue
                if self._global_step <= self._start_step:
                    self._logger.info(
                        f"Skipping evaluation at epoch {epoch + 1} (before start step {self._start_step})."
                    )
                    continue
                if self._viewer is not None:
                    self._viewer.pause_for_eval()
                self.eval()
                if self._viewer is not None:
                    self._viewer.resume_after_eval()

        if self._checkpoints_path is not None and 100 in self.config.save_at_percent:
            # If we already saved the final checkpoint at 100%, create a symlink to it so there is always a ckpt_final.pt
            final_ckpt_path = self._checkpoints_path / pathlib.Path(f"ckpt_{self._global_step:04d}.pt")
            final_ckpt_symlink_path = self._checkpoints_path / pathlib.Path("ckpt_final.pt")
            final_ply_path = self._checkpoints_path / pathlib.Path(f"ckpt_{self._global_step:04d}.ply")
            final_ply_symlink_path = self._checkpoints_path / pathlib.Path("ckpt_final.ply")
            self._logger.info(
                f"Training completed. Creating symlink {final_ckpt_symlink_path} pointing to final checkpoint at {final_ckpt_path}."
            )
            final_ckpt_symlink_path.symlink_to(final_ckpt_path.absolute())
            final_ply_symlink_path.symlink_to(final_ply_path.absolute())
        elif self._checkpoints_path is not None and 100 not in self.config.save_at_percent:
            ckpt_path = self._checkpoints_path / pathlib.Path(f"ckpt_final.pt")
            self._logger.info(f"Saving checkpoint at epoch {epoch + 1} to {ckpt_path}.")
            ply_path = self._checkpoints_path / pathlib.Path(f"ckpt_final.ply")
            self._logger.info(f"Saving PLY file at epoch {epoch + 1} to {ply_path}.")
            self._save_checkpoint_and_ply(ckpt_path, ply_path)
        else:
            self._logger.info("Training completed. No checkpoints path specified, not saving final checkpoint.")

        return self._model, self._splat_metadata()

    @torch.no_grad()
    def eval(self, stage: str = "val"):
        """
        Run evaluation of the Gaussian Splatting model on the validation dataset.

        This method evaluates the model by rendering images from the projected Gaussians and computing
        various image quality metrics.

        Args:
            stage (str): The name of the evaluation stage used for logging.
        """
        self._logger.info("Running evaluation...")
        device = self.device

        valloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=1, shuffle=False, num_workers=1)
        evaluation_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            world_to_cam_matrices = data["world_to_camera"].to(device)
            projection_matrices = data["projection"].to(device)
            ground_truth_image = data["image"].to(device) / 255.0
            mask_pixels = data["mask"] if "mask" in data and not self.config.ignore_masks else None

            height, width = ground_truth_image.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()

            predicted_image, _ = self.model.render_images(
                world_to_cam_matrices,
                projection_matrices,
                width,
                height,
                self.config.near_plane,
                self.config.far_plane,
                "perspective",
                self.config.sh_degree,
                self.config.tile_size,
                self.config.min_radius_2d,
                self.config.eps_2d,
                self.config.antialias,
            )
            predicted_image = torch.clamp(predicted_image, 0.0, 1.0)
            # depths = colors[..., -1:] / alphas.clamp(min=1e-10)
            # depths = (depths - depths.min()) / (depths.max() - depths.min())
            # depths = depths / depths.max()

            torch.cuda.synchronize()

            evaluation_time += time.time() - tic

            if mask_pixels is not None:
                # set the ground truth pixel values to match render, thus loss is zero at mask pixels and not updated
                mask_pixels = mask_pixels.to(self.device)
                ground_truth_image[~mask_pixels] = predicted_image.detach()[~mask_pixels]

            # write images
            self._save_rendered_image(
                self._global_step, stage, f"image_{i:04d}.jpg", predicted_image, ground_truth_image
            )

            ground_truth_image = ground_truth_image.permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
            predicted_image = predicted_image.permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
            metrics["psnr"].append(psnr(predicted_image, ground_truth_image))
            metrics["ssim"].append(ssim(predicted_image, ground_truth_image))
            metrics["lpips"].append(self._lpips(predicted_image, ground_truth_image))

        evaluation_time /= len(valloader)

        psnr_mean = torch.stack(metrics["psnr"]).mean()
        ssim_mean = torch.stack(metrics["ssim"]).mean()
        lpips_mean = torch.stack(metrics["lpips"]).mean()
        self._logger.info(f"Evaluation for stage {stage} completed. Average time per image: {evaluation_time:.3f}s")
        self._logger.info(f"PSNR: {psnr_mean.item():.3f}, SSIM: {ssim_mean.item():.4f}, LPIPS: {lpips_mean.item():.3f}")

        # Save stats as json
        stats = {
            "psnr": psnr_mean.item(),
            "ssim": ssim_mean.item(),
            "lpips": lpips_mean.item(),
            "evaluation_time": evaluation_time,
            "num_gaussians": self.model.num_gaussians,
        }
        self._save_statistics(self._global_step, stage, stats)

        # Log to tensorboard if enabled
        if self._tensorboard_logger is not None:
            self._tensorboard_logger.log_evaluation_iteration(
                self._global_step,
                psnr_mean.item(),
                ssim_mean.item(),
                lpips_mean.item(),
                evaluation_time,
                self.model.num_gaussians,
            )

        # Upate the viewer with evaluation results
        if self._viewer is not None:
            self._viewer.log_evaluation_iteration(
                self._global_step,
                psnr_mean.item(),
                ssim_mean.item(),
                lpips_mean.item(),
                evaluation_time,
                self.model.num_gaussians,
            )
