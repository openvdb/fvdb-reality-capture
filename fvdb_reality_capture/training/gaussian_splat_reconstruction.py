# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
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
from fvdb.viz import Viewer
from scipy.spatial import cKDTree  # type: ignore
from torch.utils.tensorboard import SummaryWriter

from ..sfm_scene import SfmScene
from .camera_pose_adjust import CameraPoseAdjustment
from .gaussian_splat_optimizer import (
    GaussianSplatOptimizer,
    GaussianSplatOptimizerConfig,
)
from .lpips import LPIPSLoss
from .sfm_dataset import SfmDataset
from .utils import crop_image_batch, make_unique_name_directory_based_on_time


@dataclass
class SceneOptimizationConfig:
    """
    Parameters for the radiance field optimization process.
    See the comments for each parameter for details.
    """

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
    save_at_percent: List[int] = field(default_factory=lambda: [20, 100])

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
    # When to start refining Gaussians during optimization
    refine_start_epoch: int = 3
    # When to stop refining Gaussians during optimization
    refine_stop_epoch: int = 100
    # How often to refine Gaussians during optimization
    refine_every_epoch: float = 0.75
    # How often to reset the opacities of the Gaussians during optimization
    reset_opacities_every_epoch: int = 16
    # When to stop using the 2d projected scale for refinement (default of 0 is to never use it)
    refine_using_scale2d_stop_epoch: int = 0
    # Whether to ignore masks during training
    ignore_masks: bool = False
    # Whether to remove Gaussians that fall outside the scene bounding box
    remove_gaussians_outside_scene_bbox: bool = False
    # What units to use for the 3D scale thresholds in the optimizer (i.e. insertion_scale_3d_threshold, deletion_scale_3d_threshold)
    # If set to "scene_scale", the thresholds are multiplied by the scene scale
    optimizer_scale_3d_threshold_units: Literal["absolute", "scene_scale"] = "scene_scale"

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
    # At which epoch to start optimizing camera postions. Default matches when we stop refining Gaussians.
    pose_opt_start_epoch: int = 0
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


class GaussianSplatReconstruction:
    """Engine for training and testing."""

    version = "0.1.0"

    _magic = "GaussianSplattingCheckpoint"

    __PRIVATE__ = object()

    @classmethod
    def from_sfm_scene(
        cls,
        sfm_scene: SfmScene,
        config: SceneOptimizationConfig = SceneOptimizationConfig(),
        optimizer_config: GaussianSplatOptimizerConfig = GaussianSplatOptimizerConfig(),
        use_every_n_as_val: int = -1,
        run_name: str | None = None,
        results_path: str | pathlib.Path | None = None,
        viewer: Viewer | None = None,
        tensorboard_path: pathlib.Path | None = None,
        viewer_update_interval_epochs: int = 10,
        tensorboard_log_interval_steps: int = 10,
        device: str | torch.device = "cuda",
        save_eval_images: bool = False,
    ):
        if isinstance(results_path, str):
            results_path = pathlib.Path(results_path)

        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        train_indices, val_indices = cls._make_index_splits(sfm_scene, use_every_n_as_val)
        train_dataset = SfmDataset(sfm_scene, train_indices)
        val_dataset = SfmDataset(sfm_scene, val_indices)

        logger.info(
            f"Created dataset training and test datasets with {len(train_dataset)} training images and {len(val_dataset)} test images."
        )

        # Initialize model
        model = GaussianSplatReconstruction._init_model(config, device, train_dataset)
        logger.info(f"Model initialized with {model.num_gaussians:,} Gaussians")

        # Initialize optimizer
        scene_scale = GaussianSplatReconstruction._compute_scene_scale(train_dataset.sfm_scene)
        max_steps = config.max_epochs * len(train_dataset)
        mean_lr_decay_exponent = 0.01 ** (1.0 / max_steps)
        # Copy the optimizer config so we don't modify the input config
        opt_config = GaussianSplatOptimizerConfig(**vars(optimizer_config))
        if config.optimizer_scale_3d_threshold_units == "scene_scale":
            opt_config.deletion_scale_3d_threshold *= scene_scale * 1.1
            opt_config.insertion_scale_3d_threshold *= scene_scale * 1.1
        opt_config.means_lr *= scene_scale * 1.1  # Scale the means learning rate by the scene scale
        optimizer = GaussianSplatOptimizer.from_model_and_config(
            model=model,
            config=opt_config,
            means_lr_decay_exponent=mean_lr_decay_exponent,
            batch_size=config.batch_size,
        )

        # Initialize pose optimizer
        pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler = None, None, None
        if config.optimize_camera_poses:
            pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler = cls._make_pose_optimizer(
                config, device, len(train_dataset)
            )

        # Setup output directories.
        run_name, run_results_path = cls._make_run_directory(
            run_name=run_name, results_base_path=results_path, exists_ok=False
        )

        return GaussianSplatReconstruction(
            model=model,
            sfm_scene=sfm_scene,
            optimizer=optimizer,
            config=config,
            optimizer_config=optimizer_config,
            train_indices=train_indices,
            val_indices=val_indices,
            pose_adjust_model=pose_adjust_model,
            pose_adjust_optimizer=pose_adjust_optimizer,
            pose_adjust_scheduler=pose_adjust_scheduler,
            start_step=0,
            run_name=run_name,
            run_results_path=run_results_path,
            viewer=viewer,
            tensorboard_path=tensorboard_path,
            eval_logs_images=save_eval_images,
            tensorboard_log_interval=tensorboard_log_interval_steps,
            viewer_log_interval=viewer_update_interval_epochs,
            _private=GaussianSplatReconstruction.__PRIVATE__,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | pathlib.Path,
        override_sfm_scene: SfmScene | None = None,
        override_use_every_n_as_val: int | None = None,
        override_results_path: pathlib.Path | str | None = None,
        viewer: Viewer | None = None,
        tensorboard_path: str | pathlib.Path | None = None,
        viewer_update_interval_epochs: int = 10,
        tensorboard_log_interval_steps: int = 10,
        save_eval_images: bool = False,
        device: str | torch.device = "cuda",
    ) -> "GaussianSplatReconstruction":
        logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        logger.info(f"Loading checkpoint from {checkpoint_path} on device {device}")
        if isinstance(checkpoint_path, str):
            checkpoint_path = pathlib.Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if checkpoint.get("magic", "") != cls._magic:
            raise ValueError(f"Checkpoint at {checkpoint_path} is not a valid checkpoint file.")
        if checkpoint.get("version", "") != cls.version:
            raise ValueError(
                f"Checkpoint version {checkpoint.get('version', '')} does not match current version {cls.version}."
            )
        if not isinstance(checkpoint.get("step", None), int):
            raise ValueError("Checkpoint step is missing or invalid.")
        if not isinstance(checkpoint.get("config", None), dict):
            raise ValueError("Checkpoint config is missing or invalid.")
        if not isinstance(checkpoint.get("sfm_scene", None), dict):
            raise ValueError("Checkpoint SfM scene is missing or invalid.")
        if not isinstance(checkpoint.get("splats", None), dict):
            raise ValueError("Checkpoint model state is missing or invalid.")
        if not isinstance(checkpoint.get("optimizer", None), dict):
            raise ValueError("Checkpoint optimizer state is missing or invalid.")
        if not isinstance(checkpoint.get("train_indices", None), (list, np.ndarray, torch.Tensor)):
            raise ValueError("Checkpoint train indices are missing or invalid.")
        if not isinstance(checkpoint.get("val_indices", None), (list, np.ndarray, torch.Tensor)):
            raise ValueError("Checkpoint val indices are missing or invalid.")
        if not isinstance(checkpoint.get("optimizer_config", None), dict):
            raise ValueError("Checkpoint optimizer_config is missing or invalid.")
        if not isinstance(checkpoint.get("run_name", None), str):
            raise ValueError("Checkpoint run name is missing or invalid.")
        if not isinstance(checkpoint.get("results_path", None), (str)):
            raise ValueError("Checkpoint results_path is missing or invalid.")

        results_path = override_results_path if override_results_path is not None else checkpoint["results_path"]
        if isinstance(results_path, str):
            results_path = pathlib.Path(results_path)

        ckpt_results_path = checkpoint["results_path"]
        if ckpt_results_path is not None and not isinstance(ckpt_results_path, str):
            raise ValueError("Checkpoint results_path is invalid.")

        # If you didn't specif a results path, use the one from the checkpoint
        if results_path is None:
            results_path = ckpt_results_path
        if not isinstance(checkpoint.get("results_path", None), (str, type(None))):
            raise ValueError("Checkpoint results_path is missing or invalid.")

        if results_path is None:
            raise ValueError("No results path specified and no results path found in checkpoint.")

        global_step = checkpoint["step"]
        run_name = checkpoint["run_name"] + "_resumed"
        optimization_config = SceneOptimizationConfig(**checkpoint["config"])
        optimizer_config = GaussianSplatOptimizerConfig(**checkpoint["optimizer_config"])
        sfm_scene: SfmScene = SfmScene.from_state_dict(checkpoint["sfm_scene"])
        train_indices = np.array(checkpoint["train_indices"], dtype=int)
        val_indices = np.array(checkpoint["val_indices"], dtype=int)

        model = GaussianSplat3d.from_state_dict(checkpoint["splats"])
        optimizer = GaussianSplatOptimizer.from_state_dict(model, checkpoint["optimizer"])

        logger.info(f"Loaded checkpoint with {model.num_gaussians:,} Gaussians.")

        if override_sfm_scene is not None:
            sfm_scene = override_sfm_scene
            logger.info("Using override SfM scene instead of the one from the checkpoint.")
            sfm_scene = checkpoint.dataset_transform(sfm_scene)

        if override_use_every_n_as_val is not None:
            indices = np.arange(sfm_scene.num_images)
            if override_use_every_n_as_val > 0:
                mask = np.ones(len(indices), dtype=bool)
                mask[::override_use_every_n_as_val] = False
                train_indices = indices[mask]
                val_indices = indices[~mask]
            else:
                train_indices = indices
                val_indices = np.array([], dtype=int)

        if override_sfm_scene is not None:
            if train_indices.min() < 0 or train_indices.max() >= sfm_scene.num_images:
                raise ValueError(
                    "Loaded training indices are out of bounds for the overridden SfM scene. If you are changing the SfM scene, you may also need to set override_use_every_n_as_val to change the train/val split."
                )
            if val_indices.size > 0 and (val_indices.min() < 0 or val_indices.max() >= sfm_scene.num_images):
                raise ValueError(
                    "Loaded validation indices are out of bounds for the overridden SfM scene. If you are changing the SfM scene, you may also need to set override_use_every_n_as_val to change the train/val split."
                )

        num_training_poses = len(train_indices)
        if num_training_poses == 0:
            raise ValueError("Checkpoint has no training poses.")

        pose_adjust_model = None
        pose_adjust_optimizer = None
        pose_adjust_scheduler = None
        if checkpoint.get("pose_adjust_model", None) is not None:
            if not isinstance(checkpoint.get("pose_adjust_model", None), dict):
                raise ValueError("Checkpoint pose adjustment model state is invalid.")
            if not isinstance(checkpoint.get("pose_adjust_optimizer", None), dict):
                raise ValueError("Checkpoint pose adjustment optimizer state is invalid.")
            if not isinstance(checkpoint.get("pose_adjust_scheduler", None), dict):
                raise ValueError("Checkpoint pose adjustment scheduler state is invalid.")
            (
                pose_adjust_model,
                pose_adjust_optimizer,
                pose_adjust_scheduler,
            ) = cls._make_pose_optimizer(optimization_config, device, num_training_poses)
            pose_adjust_model.load_state_dict(checkpoint["pose_adjust_model"])
            pose_adjust_optimizer.load_state_dict(checkpoint["pose_adjust_optimizer"])
            pose_adjust_scheduler.load_state_dict(checkpoint["pose_adjust_scheduler"])

        if isinstance(results_path, str):
            results_path = pathlib.Path(results_path)

        # Setup output directories.
        run_name, run_results_path = cls._make_run_directory(
            run_name=run_name, results_base_path=results_path, exists_ok=False
        )

        np.random.seed(optimization_config.seed)
        random.seed(optimization_config.seed)
        torch.manual_seed(optimization_config.seed)

        if isinstance(tensorboard_path, str):
            tensorboard_path = pathlib.Path(tensorboard_path)

        return GaussianSplatReconstruction(
            model=model,
            sfm_scene=sfm_scene,
            optimizer=optimizer,
            config=optimization_config,
            optimizer_config=optimizer_config,
            train_indices=train_indices,
            val_indices=val_indices,
            pose_adjust_model=pose_adjust_model,
            pose_adjust_optimizer=pose_adjust_optimizer,
            pose_adjust_scheduler=pose_adjust_scheduler,
            start_step=global_step,
            run_name=run_name,
            run_results_path=run_results_path,
            viewer=viewer,
            tensorboard_path=tensorboard_path,
            eval_logs_images=save_eval_images,
            tensorboard_log_interval=tensorboard_log_interval_steps,
            viewer_log_interval=viewer_update_interval_epochs,
            _private=GaussianSplatReconstruction.__PRIVATE__,
        )

    def __init__(
        self,
        model: GaussianSplat3d,
        sfm_scene: SfmScene,
        optimizer: GaussianSplatOptimizer,
        config: SceneOptimizationConfig,
        optimizer_config: GaussianSplatOptimizerConfig,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        pose_adjust_model: CameraPoseAdjustment | None,
        pose_adjust_optimizer: torch.optim.Adam | None,
        pose_adjust_scheduler: torch.optim.lr_scheduler.ExponentialLR | None,
        start_step: int,
        run_name: str,
        run_results_path: pathlib.Path | None,
        viewer: Viewer | None,
        tensorboard_path: pathlib.Path | None,
        eval_logs_images: bool,
        tensorboard_log_interval: int,
        viewer_log_interval: int,
        _private: object | None = None,
    ) -> None:
        """
        Initialize the Runner with the provided configuration, model, optimizer, datasets, and paths.

        Note: This constructor should only be called by the `new_run` or `resume_from_checkpoint` methods.

        Args:
            model (GaussianSplat3d): The Gaussian Splatting model to train.
            sfm_scene (SfmScene): The Structure-from-Motion scene.
            optimizer (GaussianSplatOptimizer | None): The optimizer for the model.
            config (Config): Configuration object containing model parameters.
            optimizer_config (GaussianSplatOptimizerConfig): Configuration for the optimizer.
            train_indices (np.ndarray): The indices for the training set.
            val_indices (np.ndarray): The indices for the validation set.
            pose_adjust_model (CameraPoseAdjustment | None): The camera pose adjustment model, if used
            pose_adjust_optimizer (torch.optim.Adam | None): The optimizer for camera pose adjustment, if used.
            pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None): The learning rate scheduler
                for camera pose adjustment, if used.
            start_step (int): The step to start training from (useful for resuming training
                from a checkpoint).
            run_name (str | None): The name of the training run or None for an un-named run.
            run_results_path (pathlib.Path | None): The base path to save results for this run.
            viewer (Viewer | None): The viewer instance to use for this run.
            tensorboard_path (pathlib.Path | None): The path to save TensorBoard logs or None to disable.
            tensorboard_log_interval (int): Interval (in steps) at which to log to TensorBoard if a path is specified.
            viewer_log_interval (int): Interval (in epochs) at which to update the viewer with new results if a viewer is specified.
            _private (object | None): Private object to ensure this class is only initialized through `new_run` or `resume_from_checkpoint`.
        """
        if _private is not GaussianSplatReconstruction.__PRIVATE__:
            raise ValueError("Runner should only be initialized through `new_run` or `resume_from_checkpoint`.")

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._cfg = config
        self._optimizer_config = optimizer_config
        self._model = model
        self._optimizer = optimizer
        self._pose_adjust_model = pose_adjust_model
        self._pose_adjust_optimizer = pose_adjust_optimizer
        self._pose_adjust_scheduler = pose_adjust_scheduler
        self._start_step = start_step
        self._update_viewer_every = viewer_log_interval

        self._sfm_scene = sfm_scene
        self._training_dataset = SfmDataset(sfm_scene=sfm_scene, dataset_indices=train_indices)
        self._validation_dataset = SfmDataset(sfm_scene=sfm_scene, dataset_indices=val_indices)

        self.device = model.device

        # Set up directories for saving results if a results path is provided
        self._run_name = run_name
        self._results_path = run_results_path
        if self._results_path is not None:
            self._image_render_path = self._results_path / "eval_renders" if eval_logs_images else None
            self._stats_path = self._results_path / "stats"
            self._checkpoints_path = self._results_path / "checkpoints"

            self._stats_path.mkdir(parents=True, exist_ok=True)
            self._checkpoints_path.mkdir(parents=True, exist_ok=True)
            if self._image_render_path is not None:
                self._image_render_path.mkdir(parents=True, exist_ok=True)
        else:
            self._image_render_path = None
            self._stats_path = None
            self._checkpoints_path = None

        self._global_step: int = 0

        # Create an optional TensorBoard summary writer.
        if tensorboard_path is not None and tensorboard_log_interval > 0:
            tensorboard_run_path = tensorboard_path / self._run_name
            tensorboard_run_path.mkdir(parents=True, exist_ok=True)
            self._summary_writer = SummaryWriter(log_dir=str(tensorboard_run_path))
        else:
            self._summary_writer = None
        self._tensorboard_log_interval: int = tensorboard_log_interval

        # Setup viewer for visualizing training progress if a Viewer is provided.
        self._viewer = viewer
        if self._viewer is not None:
            with torch.no_grad():
                self._viewer.add_gaussian_splat_3d(f"Gaussian Scene for run {self._run_name}", self.model)
                train_cam_poses = torch.from_numpy(self._training_dataset.camera_to_world_matrices).to(
                    dtype=torch.float32
                )
                train_projection_matrices = torch.from_numpy(
                    self._training_dataset.projection_matrices.astype(np.float32)
                )
                first_cam_pos = self._training_dataset.camera_to_world_matrices[0, :3, 3]
                self._viewer.set_camera_lookat(
                    eye=first_cam_pos,
                    center=self.model.means.mean(dim=0).cpu().numpy(),
                    up=[0, 0, -1],
                )
                # image_sizes = torch.from_numpy(self._training_dataset.image_sizes.astype(np.int32))
                # self._viewer.add_camera_view(
                #     name=f"Training Cameras for run {self._run_name}",
                #     cam_to_world_matrices=train_cam_poses,
                #     projection_matrices=train_projection_matrices,
                #     image_sizes=image_sizes,
                # )

        # Losses & Metrics.
        if self.config.lpips_net == "alex":
            self._lpips = LPIPSLoss(backbone="alex").to(model.device)
        elif self.config.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self._lpips = LPIPSLoss(backbone="vgg").to(model.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {self.config.lpips_net}")

    def _tensorboard_log_train(
        self, loss: float, l1loss: float, ssimloss: float, sh_degree: int, pose_loss: float | None = None
    ):
        if self._summary_writer is None:
            return

        mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
        mem_reserved = torch.cuda.max_memory_reserved() / 1024**3

        self._summary_writer.add_scalar("train/loss", loss, self._global_step)
        self._summary_writer.add_scalar("train/l1loss", l1loss, self._global_step)
        self._summary_writer.add_scalar("train/ssimloss", ssimloss, self._global_step)
        if pose_loss is not None:
            self._summary_writer.add_scalar("train/pose_loss", pose_loss, self._global_step)
        self._summary_writer.add_scalar("train/num_gaussians", self._model.num_gaussians, self._global_step)
        self._summary_writer.add_scalar("train/sh_degree", sh_degree, self._global_step)
        self._summary_writer.add_scalar("train/mem_allocated", mem_allocated, self._global_step)
        self._summary_writer.add_scalar("train/mem_reserved", mem_reserved, self._global_step)

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

    @torch.no_grad()
    def _save_checkpoint(self, checkpoint_path: pathlib.Path) -> None:
        """
        Save the current state of the training run to a checkpoint file.

        Args:
            checkpoint_path (pathlib.Path): The path to save the checkpoint file to.
        """
        if self._checkpoints_path is None:
            raise RuntimeError("No checkpoints path specified, cannot save checkpoint.")
        self._logger.info(f"Saving checkpoint at step {self._global_step} to path {checkpoint_path}")
        checkpoint = {
            "magic": "GaussianSplattingCheckpoint",
            "version": self.version,
            "step": self._global_step,
            "run_name": self._run_name,
            "results_path": str(self._stats_path.parent.parent) if self._stats_path is not None else None,
            "splats": self._model.state_dict(),
            "sfm_scene": self._sfm_scene.state_dict(),
            "config": vars(self.config),
            "optimizer_config": vars(self._optimizer_config),
            "train_indices": self._training_dataset.indices,
            "val_indices": self._validation_dataset.indices,
            "optimizer": self._optimizer.state_dict(),
            "num_training_poses": self._pose_adjust_model.num_poses if self._pose_adjust_model else None,
            "pose_adjust_model": self._pose_adjust_model.state_dict() if self._pose_adjust_model else None,
            "pose_adjust_optimizer": self._pose_adjust_optimizer.state_dict() if self._pose_adjust_optimizer else None,
            "pose_adjust_scheduler": self._pose_adjust_scheduler.state_dict() if self._pose_adjust_scheduler else None,
        }
        torch.save(checkpoint, checkpoint_path)

    @torch.no_grad()
    def _save_checkpoint_and_ply(self, ckpt_path: pathlib.Path, ply_path: pathlib.Path):
        """
        Saves a checkpoint and a PLY file to disk
        """
        if self._checkpoints_path is None:
            return

        self._save_checkpoint(ckpt_path)

        self.model.save_ply(
            ply_path,
            metadata=self._splat_ply_metadata(),
        )

    @classmethod
    def _make_run_directory(
        cls,
        run_name: str | None,
        results_base_path: pathlib.Path | None,
        exists_ok: bool = False,
    ):
        """
        Create or get the path to the run directory for the training run.

        Args:
            run_name (str | None): The name of the run. If None, a unique name will be generated.
            results_base_path (pathlib.Path | None): The base path where results will be saved. If None, no results will be saved.
                and the function will return None for the results path.
            exists_ok (bool): If True, will not raise an error if the run name already exists.
        Returns:
            run_name (str): The name of the run.
            results_path (pathlib.Path | None): The path to the run directory, or None if no results path is provided.
        """
        logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if results_base_path is None:
            logger.info("No results will be saved. You can set `results_path` to save the training run.")
            # If no results are saved and you didn't pass a run name, we'll generate a unique one
            if run_name is None:
                run_name = str(uuid.uuid4())
                logger.info(f"Generated a unique run name '{run_name}' for this run.")
            return run_name, None
        else:
            results_base_path.mkdir(exist_ok=True)
            if run_name is None:
                run_name, results_path = make_unique_name_directory_based_on_time(results_base_path, prefix="run")
            else:
                results_path = results_base_path / pathlib.Path(run_name)
                if not exists_ok and results_path.exists():
                    raise FileExistsError(
                        f"Run name {run_name} already exists in results path {results_base_path}. "
                        "Please provide a different run name or set exists_ok=True."
                    )
                results_path.mkdir(exist_ok=True)
            logger.info(f"Created run named {run_name}. Results will be saved to {results_path.absolute()}.")
            return run_name, results_path

    @torch.no_grad()
    def _splat_ply_metadata(self) -> dict[str, torch.Tensor | float | int | str]:
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
            "scene_scale": GaussianSplatReconstruction._compute_scene_scale(self.training_dataset.sfm_scene),
            "eps2d": self.config.eps_2d,
            "near_plane": self.config.near_plane,
            "far_plane": self.config.far_plane,
            "min_radius_2d": self.config.min_radius_2d,
            "antialias": int(self.config.antialias),
            "tile_size": self.config.tile_size,
        }

    @property
    def optimization_metadata(self) -> dict[str, torch.Tensor | float | int | str]:
        """
        Get metadata about the current optimization state, including camera parameters and scene scale.

        Returns:
            dict: A dictionary containing metadata about the optimization state. It's keys include:
                - normalization_transform: The transformation matrix used to normalize the scene.
                - camera_to_world_matrices: The optimized camera-to-world matrices for the training images.
                - projection_matrices: The projection matrices for the training images.
                - image_sizes: The sizes of the training images.
                - scene_scale: The computed scale of the scene.
                - eps2d: The 2D epsilon value used in rendering.
                - near_plane: The near plane distance used in rendering.
                - far_plane: The far plane distance used in rendering.
                - min_radius_2d: The minimum 2D radius used in rendering.
                - antialias: Whether anti-aliasing is enabled (1) or not (0).
                - tile_size: The tile size used in rendering.
        """
        return self._splat_ply_metadata()

    @property
    def config(self) -> SceneOptimizationConfig:
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

    @property
    def results_path(self) -> pathlib.Path | None:
        """
        Get the base path where all results (stats, renders, checkpoints, tensorboard logs) are saved.

        Returns:
            pathlib.Path | None: The base results path, or None if not set.
        """
        if self._stats_path is not None:
            return self._stats_path.parent
        return None

    @staticmethod
    def _init_model(
        config: SceneOptimizationConfig,
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
            kd_tree = cKDTree(x_np)  # type: ignore
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
        if use_sfm_depths and sfm_scene.has_visible_point_indices:
            # Estimate the scene scale as the median across the median distances from cameras to the
            # sfm points they see. If there is not too much variance in how far the cameras are from the scene
            # this gives a rough estimate of the scene scale.
            median_depth_per_camera = []
            for image_meta in sfm_scene.images:
                # Don't use cameras that don't see any points in the estimate
                assert (
                    image_meta.point_indices is not None
                ), "SfmScene.has_visible_point_indices is True but image has no point indices"
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
    def _make_index_splits(sfm_scene: SfmScene, use_every_n_as_val: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Create training and validation splits from the images in the SfmScene.

        Args:
            sfm_scene (SfmScene): The scene loaded from an structure-from-motion (SfM) pipeline.
            use_every_n_as_val (int): How often to use a training image as a validation image

        Returns:
            train_indices (np.ndarray): Indices of images to use for training.
            val_indices (np.ndarray): Indices of images to use for validation.
        """
        indices = np.arange(sfm_scene.num_images)
        if use_every_n_as_val > 0:
            mask = np.ones(len(indices), dtype=bool)
            mask[::use_every_n_as_val] = False
            train_indices = indices[mask]
            val_indices = indices[~mask]
        else:
            train_indices = indices
            val_indices = np.array([], dtype=np.int64)
        return train_indices, val_indices

    @classmethod
    def _make_pose_optimizer(
        cls, optimization_config: SceneOptimizationConfig, device: torch.device | str, num_images: int
    ) -> tuple[CameraPoseAdjustment, torch.optim.Adam, torch.optim.lr_scheduler.ExponentialLR]:
        """
        Create a camera pose adjustment model, optimizer, and scheduler if camera pose optimization is enabled in the config.

        Args:
            optimization_config (Config): Configuration object containing optimization parameters.
            device (torch.device | str): The device to run the model on (e.g., "cuda" or "cpu").
            num_images (int): The number of images in the dataset.

        Returns:
            pose_adjust_model (CameraPoseAdjustment | None):
                The camera pose adjustment model, or None if not used.
            pose_adjust_optimizer (torch.optim.Adam | None):
                The optimizer for the pose adjustment model, or None if not used.
            pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None):
                The learning rate scheduler for the pose adjustment optimizer, or None if not used.
        """
        if not optimization_config.optimize_camera_poses:
            raise ValueError("Camera pose optimization is not enabled in the config.")

        # Module to adjust camera poses during training
        pose_adjust_model = CameraPoseAdjustment(num_images, init_std=optimization_config.pose_opt_init_std).to(device)

        # Increase learning rate for pose optimization and add gradient clipping
        pose_adjust_optimizer = torch.optim.Adam(
            pose_adjust_model.parameters(),
            lr=optimization_config.pose_opt_lr * 100.0,
            weight_decay=optimization_config.pose_opt_reg,
        )

        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(pose_adjust_model.parameters(), max_norm=1.0)

        # Add learning rate scheduler for pose optimization
        pose_opt_start_step = int(optimization_config.pose_opt_start_epoch * num_images)
        pose_opt_stop_step = int(optimization_config.pose_opt_stop_epoch * num_images)
        num_pose_opt_steps = max(1, pose_opt_stop_step - pose_opt_start_step)
        pose_adjust_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            pose_adjust_optimizer, gamma=optimization_config.pose_opt_lr_decay ** (1.0 / num_pose_opt_steps)
        )
        return pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler

    def train(self):
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
        pose_opt_start_step: int = int(self.config.pose_opt_start_epoch * len(self.training_dataset))
        pose_opt_stop_step: int = int(self.config.pose_opt_stop_epoch * len(self.training_dataset))

        update_viewer_every_step = int(self._update_viewer_every * len(self.training_dataset))

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

                cam_to_world_mats: torch.Tensor = minibatch["camera_to_world"].to(self.device)  # [B, 4, 4]
                world_to_cam_mats: torch.Tensor = minibatch["world_to_camera"].to(self.device)  # [B, 4, 4]

                # Camera pose optimization
                image_ids = minibatch["image_id"].to(self.device)  # [B]
                if self.pose_adjust_model is not None:
                    if self._global_step == pose_opt_start_step:
                        self._logger.info(
                            f"Starting to optimize camera poses at step {self._global_step:,} (epoch {epoch})"
                        )
                    if pose_opt_start_step <= self._global_step < pose_opt_stop_step:
                        cam_to_world_mats = self.pose_adjust_model(cam_to_world_mats, image_ids)
                    elif self._global_step >= pose_opt_stop_step:
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
                    GaussianSplat3d.ProjectionType.PERSPECTIVE,
                    sh_degree_to_use,
                    self.config.min_radius_2d,
                    self.config.eps_2d,
                    self.config.antialias,
                )

                # If you have very large images, you can iterate over disjoint crops and accumulate gradients
                # If self.optimization_config.crops_per_image is 1, then this just returns the image
                for pixels, mask_pixels, crop, is_last in crop_image_batch(image, mask, self.config.crops_per_image):
                    # Actual pixels to compute the loss on, normalized to [0, 1]
                    pixels = pixels.to(self.device) / 255.0  # [1, H, W, 3]

                    # Render an image from the gaussian splats
                    # possibly using a crop of the full image
                    crop_origin_w, crop_origin_h, crop_w, crop_h = crop
                    colors, alphas = self.model.render_from_projected_gaussians(
                        projected_gaussians,
                        crop_w,
                        crop_h,
                        crop_origin_w,
                        crop_origin_h,
                        self.config.tile_size,
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
                    loss = torch.lerp(l1loss, ssimloss, self.config.ssim_lambda)

                    # Rgularize opacity to ensure Gaussian's don't become too opaque
                    if self.config.opacity_reg > 0.0:
                        loss = loss + self.config.opacity_reg * torch.abs(self.model.opacities).mean()

                    # Regularize scales to ensure Gaussians don't become too large
                    if self.config.scale_reg > 0.0:
                        loss = loss + self.config.scale_reg * torch.abs(self.model.scales).mean()

                    # If you're optimizing poses, regularize the pose parameters so the poses
                    # don't drift too far from the initial values
                    if (
                        self.pose_adjust_model is not None
                        and pose_opt_start_step <= self._global_step < pose_opt_stop_step
                    ):
                        pose_params = self.pose_adjust_model.pose_embeddings(image_ids)
                        pose_reg = torch.mean(torch.abs(pose_params))
                        loss = loss + self.config.pose_opt_reg * pose_reg
                    else:
                        pose_reg = None

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
                    num_dup, num_split, num_prune = self.optimizer.refine(
                        use_scales_for_deletion=use_scales_for_refinement,
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

                        self.optimizer.filter_gaussians(~outside_mask)
                        ng_post = self.model.num_gaussians
                        nclip = ng_prior - ng_post
                        self._logger.debug(
                            f"Clipped {nclip:,} Gaussians outside the crop bounding box min={bbox_min}, max={bbox_max}."
                        )

                # Reset the opacity parameters every so often
                if (
                    self.config.reset_opacities_every_epoch > 0
                    and self._global_step % reset_opacities_every_step == 0
                    and self._global_step > 0
                ):
                    self.optimizer.reset_opacities()

                # Step the Gaussian optimizer
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # If you enabled pose optimization, step the pose optimizer if we performed a
                # pose update this iteration
                if self.config.optimize_camera_poses and pose_opt_start_step <= self._global_step < pose_opt_stop_step:
                    assert (
                        self.pose_adjust_optimizer is not None
                    ), "Pose optimizer should be initialized if pose optimization is enabled."
                    assert (
                        self.pose_adjust_scheduler is not None
                    ), "Pose scheduler should be initialized if pose optimization is enabled."
                    self.pose_adjust_optimizer.step()
                    self.pose_adjust_scheduler.step()
                    self.pose_adjust_optimizer.zero_grad(set_to_none=True)

                # Log to tensorboard if you requested it
                if self._summary_writer is not None and self._global_step % self._tensorboard_log_interval == 0:
                    self._tensorboard_log_train(
                        loss=loss.item(),
                        l1loss=l1loss.item(),
                        ssimloss=ssimloss.item(),
                        sh_degree=sh_degree_to_use,
                        pose_loss=pose_reg.item() if pose_reg is not None else None,
                    )

                # Update the viewer
                if self._viewer is not None and self._global_step % update_viewer_every_step == 0:
                    with torch.no_grad():
                        self._logger.debug(f"Updating viewer at step {self._global_step:,}")
                        self._viewer.add_gaussian_splat_3d("Gaussian Scene", self.model)

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
                self.eval()

        if self._checkpoints_path is not None and 100 in self.config.save_at_percent:
            # If we already saved the final checkpoint at 100%, create a symlink to it so there is always a ckpt_final.pt
            final_ckpt_path = self._checkpoints_path / pathlib.Path(f"ckpt_{self._global_step:04d}.pt")
            final_ckpt_symlink_path = self._checkpoints_path / pathlib.Path("ckpt_final.pt")
            final_ply_path = self._checkpoints_path / pathlib.Path(f"ckpt_{self._global_step:04d}.ply")
            final_ply_symlink_path = self._checkpoints_path / pathlib.Path("ckpt_final.ply")
            self._logger.info(
                f"Training completed. Creating symlink {final_ckpt_symlink_path} pointing to final checkpoint at {final_ckpt_path}."
            )
            # Use relative paths for symlink so it works if you move the results directory
            final_ckpt_symlink_path.absolute().symlink_to(
                final_ckpt_path.absolute().relative_to(final_ckpt_symlink_path.absolute().parent)
            )
            final_ply_symlink_path.absolute().symlink_to(
                final_ply_path.absolute().relative_to(final_ply_symlink_path.absolute().parent)
            )
        elif self._checkpoints_path is not None and 100 not in self.config.save_at_percent:
            ckpt_path = self._checkpoints_path / pathlib.Path(f"ckpt_final.pt")
            self._logger.info(f"Saving checkpoint at epoch {epoch + 1} to {ckpt_path}.")
            ply_path = self._checkpoints_path / pathlib.Path(f"ckpt_final.ply")
            self._logger.info(f"Saving PLY file at epoch {epoch + 1} to {ply_path}.")
            self._save_checkpoint_and_ply(ckpt_path, ply_path)
        else:
            self._logger.info("Training completed. No checkpoints path specified, not saving final checkpoint.")

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
                GaussianSplat3d.ProjectionType.PERSPECTIVE,
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
        if self._summary_writer is not None:
            self._summary_writer.add_scalar("PSNR", psnr_mean.item(), self._global_step)
            self._summary_writer.add_scalar("SSIM", ssim_mean.item(), self._global_step)
            self._summary_writer.add_scalar("LPIPS", lpips_mean.item(), self._global_step)
            self._summary_writer.add_scalar("Evaluation Time", evaluation_time, self._global_step)
            self._summary_writer.add_scalar("Num Gaussians", self.model.num_gaussians, self._global_step)

        # Update the viewer with evaluation results
        if self._viewer is not None:
            self._logger.debug(f"Updating viewer after evaluation at step {self._global_step:,}")
            self._viewer.add_gaussian_splat_3d("Gaussian Scene", self.model)
