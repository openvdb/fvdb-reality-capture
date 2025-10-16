# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, List, Literal

import numpy as np
import torch
import torch.nn.functional as nnf
import torch.utils.data
import tqdm
from fvdb import GaussianSplat3d
from fvdb.utils.metrics import psnr, ssim
from fvdb.viz import Viewer
from scipy.spatial import cKDTree  # type: ignore

from ..sfm_scene import SfmScene
from .camera_pose_adjust import CameraPoseAdjustment
from .gaussian_splat_optimizer import (
    BaseGaussianSplatOptimizer,
    GaussianSplatOptimizer,
    GaussianSplatOptimizerConfig,
)
from .gaussian_splat_reconstruction_writer import (
    GaussianSplatReconstructionBaseWriter,
    GaussianSplatReconstructionWriter,
)
from .lpips import LPIPSLoss
from .sfm_dataset import SfmDataset
from .utils import crop_image_batch


@dataclass
class GaussianSplatReconstructionConfig:
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
    refine_every_epoch: float = 0.65
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
        writer: GaussianSplatReconstructionBaseWriter = GaussianSplatReconstructionWriter(
            run_name=None, save_path=None
        ),
        viewer: Viewer | None = None,
        config: GaussianSplatReconstructionConfig = GaussianSplatReconstructionConfig(),
        optimizer_config: GaussianSplatOptimizerConfig = GaussianSplatOptimizerConfig(),
        use_every_n_as_val: int = -1,
        viewer_update_interval_epochs: int = 10,
        log_interval_steps: int = 10,
        device: str | torch.device = "cuda",
    ):
        """
        Create a `GaussianSplatReconstruction` instance from an `SfmScene`, used to reconstruct
        a 3D Gaussian Splat radiance field from posed images. The optimization process can be
        configured using the `config` and `optimizer_config` parameters, though the defaults
        should produce acceptable results.

        There are also several parameters to configure logging and visualization of the training
        process, as well as saving results.

        Args:
            sfm_scene (SfmScene): The Structure-from-Motion scene containing images and camera poses.
            config (GaussianSplatReconstructionConfig): Configuration for the reconstruction process.
            optimizer_config (GaussianSplatOptimizerConfig): Configuration for the optimizer.
            writer (GaussianReconstrutionBaseWriter): Writer instance to handle saving images, ply files,
                and other results.
            viewer (Viewer | None): Optional Viewer instance for visualizing training progress. If None,
                no visualization is performed.
            use_every_n_as_val (int): Use every n-th image as a validation image. Default of -1
                means no validation images are used.
            viewer_update_interval_epochs (int): Interval in epochs at which to update the viewer.
                An epoch is one full pass through the training dataset.
            log_interval_steps (int): Interval in steps to log to TensorBoard.
            device (str | torch.device): Device to run the reconstruction on.
        Returns:
            GaussianSplatReconstruction: An instance ready to reconstruct the scene.
        """

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
        max_steps = config.max_epochs * len(train_dataset)
        optimizer = GaussianSplatOptimizer.from_model_and_scene(
            model=model,
            sfm_scene=train_dataset.sfm_scene,
            config=optimizer_config,
        )
        optimizer.reset_learning_rates_and_decay(batch_size=config.batch_size, expected_steps=max_steps)

        # Initialize pose optimizer
        pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler = None, None, None
        if config.optimize_camera_poses:
            pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler = cls._make_pose_optimizer(
                config, device, len(train_dataset)
            )

        return GaussianSplatReconstruction(
            model=model,
            sfm_scene=sfm_scene,
            optimizer=optimizer,
            config=config,
            train_indices=train_indices,
            val_indices=val_indices,
            pose_adjust_model=pose_adjust_model,
            pose_adjust_optimizer=pose_adjust_optimizer,
            pose_adjust_scheduler=pose_adjust_scheduler,
            writer=writer,
            start_step=0,
            viewer=viewer,
            log_interval_steps=log_interval_steps,
            viewer_update_interval_epochs=viewer_update_interval_epochs,
            _private=GaussianSplatReconstruction.__PRIVATE__,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        override_sfm_scene: SfmScene | None = None,
        override_use_every_n_as_val: int | None = None,
        writer: GaussianSplatReconstructionBaseWriter = GaussianSplatReconstructionWriter(
            run_name=None, save_path=None
        ),
        viewer: Viewer | None = None,
        viewer_update_interval_epochs: int = 1,
        log_interval_steps: int = 10,
        device: str | torch.device = "cuda",
    ):
        """
        Load a `GaussianSplatReconstruction` instance from a state dictionary (extracted with the `state_dict()` method).
        This will restore the model, optimizer, and training state.
        You can optionally override the SfM scene and the train/validation split
        This is useful for resuming training on a different dataset or with a different train/val split.

        Args:
            state_dict (dict): State dictionary containing the model, optimizer, and training state. Generated by
                the `state_dict()` method.
            override_sfm_scene (SfmScene | None): Optional SfM scene to use instead of the one in the state_dict.
            override_use_every_n_as_val (int | None): If specified, will override the train/val split using this value.
                Default of None means to use the train/val split from the state_dict.
            writer (GaussianReconstructionBaseWriter): Writer instance to handle saving images, ply files,
                and other results.
            viewer (Viewer | None): Optional Viewer instance for visualizing training progress. If None, no
                visualization is performed.
            viewer_update_interval_epochs (int): Interval in epochs at which to update the viewer. An epoch is one
                full pass through the training dataset.
            log_interval_steps (int): Interval in steps to log to TensorBoard.
            device (str | torch.device): Device to run the reconstruction on.
        """
        logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        # Ensure this is a valid state dict
        if state_dict.get("magic", "") != cls._magic:
            raise ValueError(f"State dict has invalid magic value.")

        # Ensure the state_dict version matches the current version of this class
        if state_dict.get("version", "") != cls.version:
            raise ValueError(
                f"Checkpoint version {state_dict.get('version', '')} does not match current version {cls.version}."
            )

        # Check that all required keys in the state dict are present and their values have the correct types
        if not isinstance(state_dict.get("step", None), int):
            raise ValueError("Checkpoint step is missing or invalid.")
        if not isinstance(state_dict.get("config", None), dict):
            raise ValueError("Checkpoint config is missing or invalid.")
        if not isinstance(state_dict.get("sfm_scene", None), dict):
            raise ValueError("Checkpoint SfM scene is missing or invalid.")
        if not isinstance(state_dict.get("model", None), dict):
            raise ValueError("Checkpoint model state is missing or invalid.")
        if not isinstance(state_dict.get("optimizer", None), dict):
            raise ValueError("Checkpoint optimizer state is missing or invalid.")
        if not isinstance(state_dict.get("train_indices", None), (list, np.ndarray, torch.Tensor)):
            raise ValueError("Checkpoint train indices are missing or invalid.")
        if not isinstance(state_dict.get("val_indices", None), (list, np.ndarray, torch.Tensor)):
            raise ValueError("Checkpoint val indices are missing or invalid.")
        if "num_training_poses" not in state_dict:
            raise ValueError("Checkpoint is missing num_training_poses key.")
        if "pose_adjust_model" not in state_dict:
            raise ValueError("Checkpoint is missing pose_adjust_model key.")
        if "pose_adjust_optimizer" not in state_dict:
            raise ValueError("Checkpoint is missing pose_adjust_optimizer key.")
        if "pose_adjust_scheduler" not in state_dict:
            raise ValueError("Checkpoint is missing pose_adjust_scheduler key.")

        global_step = state_dict["step"]
        config = GaussianSplatReconstructionConfig(**state_dict["config"])
        if override_sfm_scene is not None:
            sfm_scene: SfmScene = override_sfm_scene
            logger.info("Using override SfM scene instead of the one from the checkpoint.")
        else:
            sfm_scene: SfmScene = SfmScene.from_state_dict(state_dict["sfm_scene"])
        if override_use_every_n_as_val is not None:
            train_indices, val_indices = cls._make_index_splits(sfm_scene, override_use_every_n_as_val)
        else:
            train_indices = np.array(state_dict["train_indices"], dtype=int)
            val_indices = np.array(state_dict["val_indices"], dtype=int)
        model = GaussianSplat3d.from_state_dict(state_dict["model"])
        optimizer = GaussianSplatOptimizer.from_state_dict(model, state_dict["optimizer"])
        num_training_poses = state_dict["num_training_poses"]
        pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler = None, None, None

        if state_dict["pose_adjust_model"] is not None:
            if not isinstance(state_dict.get("pose_adjust_model", None), dict):
                raise ValueError("Checkpoint pose adjustment model state is invalid.")
            if not isinstance(state_dict.get("pose_adjust_optimizer", None), dict):
                raise ValueError("Checkpoint pose adjustment optimizer state is invalid.")
            if not isinstance(state_dict.get("pose_adjust_scheduler", None), dict):
                raise ValueError("Checkpoint pose adjustment scheduler state is invalid.")
            pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler = cls._make_pose_optimizer(
                config, device, num_training_poses
            )
            pose_adjust_model.load_state_dict(state_dict["pose_adjust_model"])
            pose_adjust_optimizer.load_state_dict(state_dict["pose_adjust_optimizer"])
            pose_adjust_scheduler.load_state_dict(state_dict["pose_adjust_scheduler"])

        return GaussianSplatReconstruction(
            model=model,
            sfm_scene=sfm_scene,
            optimizer=optimizer,
            config=config,
            train_indices=train_indices,
            val_indices=val_indices,
            pose_adjust_model=pose_adjust_model,
            pose_adjust_optimizer=pose_adjust_optimizer,
            pose_adjust_scheduler=pose_adjust_scheduler,
            writer=writer,
            start_step=global_step,
            viewer=viewer,
            log_interval_steps=log_interval_steps,
            viewer_update_interval_epochs=viewer_update_interval_epochs,
            _private=GaussianSplatReconstruction.__PRIVATE__,
        )

    def __init__(
        self,
        model: GaussianSplat3d,
        sfm_scene: SfmScene,
        optimizer: BaseGaussianSplatOptimizer,
        config: GaussianSplatReconstructionConfig,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        pose_adjust_model: CameraPoseAdjustment | None,
        pose_adjust_optimizer: torch.optim.Adam | None,
        pose_adjust_scheduler: torch.optim.lr_scheduler.ExponentialLR | None,
        writer: GaussianSplatReconstructionBaseWriter,
        start_step: int,
        viewer: Viewer | None,
        log_interval_steps: int,
        viewer_update_interval_epochs: int,
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
            train_indices (np.ndarray): The indices for the training set.
            val_indices (np.ndarray): The indices for the validation set.
            pose_adjust_model (CameraPoseAdjustment | None): The camera pose adjustment model, if used
            pose_adjust_optimizer (torch.optim.Adam | None): The optimizer for camera pose adjustment, if used.
            pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None): The learning rate scheduler
                for camera pose adjustment, if used.
            writer (GaussianSplatReconstructionBaseWriter): Writer instance to handle saving images, ply files,
                and other results.
            start_step (int): The step to start training from (useful for resuming training
                from a checkpoint).
            viewer (Viewer | None): The viewer instance to use for this run.
            log_interval_steps (int): Interval (in steps) at which to log metrics during training.
            viewer_update_interval_epochs (int): Interval (in epochs) at which to update the viewer with new results if a viewer is specified.
            _private (object | None): Private object to ensure this class is only initialized through `new_run` or `resume_from_checkpoint`.
        """
        if _private is not GaussianSplatReconstruction.__PRIVATE__:
            raise ValueError("Runner should only be initialized through `new_run` or `resume_from_checkpoint`.")

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._cfg = config
        self._model = model
        self._optimizer = optimizer
        self._pose_adjust_model = pose_adjust_model
        self._pose_adjust_optimizer = pose_adjust_optimizer
        self._pose_adjust_scheduler = pose_adjust_scheduler
        self._start_step = start_step
        self._viewer_update_interval_epochs = viewer_update_interval_epochs

        self._sfm_scene = sfm_scene
        self._training_dataset = SfmDataset(sfm_scene=sfm_scene, dataset_indices=train_indices)
        self._validation_dataset = SfmDataset(sfm_scene=sfm_scene, dataset_indices=val_indices)

        self.device: torch.device = model.device

        self._global_step: int = 0

        self._log_interval_steps: int = log_interval_steps

        self._writer = writer

        # Setup viewer for visualizing training progress if a Viewer is provided.
        self._viewer = viewer
        if self._viewer is not None:
            with torch.no_grad():
                self._viewer.add_gaussian_splat_3d(f"Gaussian Scene", self.model)

        # Losses & Metrics.
        if self.config.lpips_net == "alex":
            self._lpips = LPIPSLoss(backbone="alex").to(model.device)
        elif self.config.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self._lpips = LPIPSLoss(backbone="vgg").to(model.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {self.config.lpips_net}")

    @torch.no_grad()
    def state_dict(self) -> dict[str, Any]:
        """
        Get the state dictionary of the current training state, including model, optimizer, and training parameters.

        Returns:
            dict: A dictionary containing the state of the training process. Its keys include:
                - magic: A magic string to identify the checkpoint type.
                - version: The version of the checkpoint format.
                - step: The current global training step.
                - config: The configuration parameters used for training.
                - sfm_scene: The state dictionary of the SfM scene.
                - model: The state dictionary of the Gaussian Splatting model.
                - optimizer: The state dictionary of the optimizer.
                - train_indices: The indices of the training dataset.
                - val_indices: The indices of the validation dataset.
                - num_training_poses: The number of training poses if pose adjustment is used, otherwise None.
                - pose_adjust_model: The state dictionary of the camera pose adjustment model if used, otherwise None.
                - pose_adjust_optimizer: The state dictionary of the pose adjustment optimizer if used, otherwise None.
                - pose_adjust_scheduler: The state dictionary of the pose adjustment scheduler if used, otherwise None.
        """
        return {
            "magic": "GaussianSplattingCheckpoint",
            "version": self.version,
            "step": self._global_step,
            "config": vars(self.config),
            "sfm_scene": self._sfm_scene.state_dict(),
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "train_indices": self._training_dataset.indices,
            "val_indices": self._validation_dataset.indices,
            "num_training_poses": self._pose_adjust_model.num_poses if self._pose_adjust_model else None,
            "pose_adjust_model": self._pose_adjust_model.state_dict() if self._pose_adjust_model else None,
            "pose_adjust_optimizer": self._pose_adjust_optimizer.state_dict() if self._pose_adjust_optimizer else None,
            "pose_adjust_scheduler": self._pose_adjust_scheduler.state_dict() if self._pose_adjust_scheduler else None,
        }

    @property
    def optimization_metadata(self) -> dict[str, torch.Tensor | float | int | str]:
        """
        Get metadata about the current optimization state, including camera parameters and scene scale.

        Returns:
            dict: A dictionary containing metadata about the optimization state. It's keys include:
                - normalization_transform: The transformation matrix used to normalize the scene.
                - camera_to_world_matrices: The optimized camera-to-world matrices for the images used during
                    reconstruction.
                - projection_matrices: The projection matrices for the images used during reconstruction.
                - image_sizes: The sizes of the images used during reconstruction.
                - scene_scale: The computed scale of the scene.
                - eps2d: The 2D epsilon value used in rendering.
                - near_plane: The near plane distance used in rendering.
                - far_plane: The far plane distance used in rendering.
                - min_radius_2d: The minimum 2D radius used in rendering.
                - antialias: Whether anti-aliasing is enabled (1) or not (0).
                - tile_size: The tile size used in rendering.
        """
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
            "eps2d": self.config.eps_2d,
            "near_plane": self.config.near_plane,
            "far_plane": self.config.far_plane,
            "min_radius_2d": self.config.min_radius_2d,
            "antialias": int(self.config.antialias),
            "tile_size": self.config.tile_size,
        }

    @property
    def config(self) -> GaussianSplatReconstructionConfig:
        """
        Get the configuration object for the current training run.

        Returns:
            GaussianSplatReconstructionConfig: The configuration object containing all parameters for the training run.
        """
        return self._cfg

    @property
    def model(self) -> GaussianSplat3d:
        """
        Get the Gaussian Splatting model being trained.

        Returns:
            GaussianSplat3d: The model instance.
        """
        return self._model

    @property
    def optimizer(self) -> BaseGaussianSplatOptimizer:
        """
        Get the optimizer used for training the Gaussian Splatting model.

        Returns:
            optimizer (BaseGaussianSplatOptimizer): The optimizer instance.
        """
        return self._optimizer

    @property
    def pose_adjust_model(self) -> CameraPoseAdjustment | None:
        """
        Get the camera pose adjustment model used for optimizing camera poses during training.

        Returns:
            pose_adjust_model (CameraPoseAdjustment | None): The pose adjustment model instance, or None if not used.
        """
        return self._pose_adjust_model

    @property
    def pose_adjust_optimizer(self) -> torch.optim.Adam | None:
        """
        Get the optimizer used for adjusting camera poses during training.

        Returns:
            pose_adjust_optimizer (torch.optim.Optimizer | None): The pose adjustment optimizer instance, or None if not used.
        """
        return self._pose_adjust_optimizer

    @property
    def pose_adjust_scheduler(self) -> torch.optim.lr_scheduler.ExponentialLR | None:
        """
        Get the learning rate scheduler used for adjusting camera poses during training.

        Returns:
            pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None): The pose adjustment scheduler instance, or None if not used.
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

    @staticmethod
    def _init_model(
        config: GaussianSplatReconstructionConfig,
        device: torch.device | str,
        training_dataset: SfmDataset,
    ):
        """
        Initialize the Gaussian Splatting model with random parameters based on the training dataset.

        Args:
            config (GaussianSplatReconstructionConfig): Configuration object containing model parameters.
            device (torch.device | str): The device to run the model on (e.g., "cuda" or "cpu").
            training_dataset (SfmDataset): The dataset used for training, which provides the initial points and RGB values
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

        return model

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
        cls, optimization_config: GaussianSplatReconstructionConfig, device: torch.device | str, num_images: int
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

    def _clip_gaussians_to_scene_bbox(self) -> None:
        """
        Remove all Gaussians whose means lie outside the scene bounding box defined in the training dataset.
        """
        bbox_min, bbox_max = self.training_dataset.scene_bbox
        if (
            np.any(np.isinf(bbox_min))
            or np.any(np.isinf(bbox_max))
            or np.any(np.isnan(bbox_min))
            or np.any(np.isnan(bbox_max))
        ):
            self._logger.warning("Scene bounding box is infinite or NaN. Skipping Gaussian clipping.")
            return

        num_gaussians_before_clipping = self.model.num_gaussians
        with torch.no_grad():
            points = self.model.means
            outside_mask = torch.logical_or(points[:, 0] < bbox_min[0], points[:, 0] > bbox_max[0])
            outside_mask.logical_or_(points[:, 1] < bbox_min[1])
            outside_mask.logical_or_(points[:, 1] > bbox_max[1])
            outside_mask.logical_or_(points[:, 2] < bbox_min[2])
            outside_mask.logical_or_(points[:, 2] > bbox_max[2])

        self.optimizer.filter_gaussians(~outside_mask)
        num_gaussians_after_clipping = self.model.num_gaussians
        num_clipped_gaussians = num_gaussians_before_clipping - num_gaussians_after_clipping
        self._logger.debug(
            f"Clipped {num_clipped_gaussians:,} Gaussians outside the crop bounding box min={bbox_min}, max={bbox_max}."
        )

    def train(self, show_progress: bool = True, log_tag: str = "train") -> None:
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
        - Saving checkpoints and evaluation renders at specified intervals.

        Args:
            show_progress (bool): Whether to display a progress bar during training.
            log_tag (str): Tag to use for logging metrics (e.g., "train). Data logged will use this tag as a prefix.
                For metrics, this will be "{log_tag}/metric_name".
                For checkpoints, this will be "{log_tag}_ckpt.pt".
                For PLY files, this will be "{log_tag}_ckpt.ply".
                Note: When calling evaluation from the training loop, the log_tag for evaluation will be log_tag+"_eval".
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
        increase_sh_degree_every_step: int = int(
            self.config.increase_sh_degree_every_epoch * len(self.training_dataset)
        )
        pose_opt_start_step: int = int(self.config.pose_opt_start_epoch * len(self.training_dataset))
        pose_opt_stop_step: int = int(self.config.pose_opt_stop_epoch * len(self.training_dataset))

        update_viewer_every_step = int(self._viewer_update_interval_epochs * len(self.training_dataset))

        # Progress bar to track training progress
        if self.config.max_steps is not None:
            self._logger.info(
                f"Using max_steps={self.config.max_steps} (overriding computed {computed_total_steps} steps)"
            )
        if show_progress:
            pbar = tqdm.tqdm(range(0, total_steps), unit="imgs", desc="Training")
        else:
            pbar = None

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
                    if pbar is not None:
                        pbar.set_description(
                            f"Skipping step {self._global_step:,} (before start step {self._start_step:,})"
                        )
                        pbar.update(batch_size)
                        self._global_step = pbar.n
                    else:
                        self._global_step += batch_size
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
                    pixels: torch.Tensor = pixels.to(device=self.device) / 255.0  # [1, H, W, 3]

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
                    l1loss = nnf.l1_loss(colors, pixels)
                    ssimloss = 1.0 - ssim(
                        colors.permute(0, 3, 1, 2).contiguous(),
                        pixels.permute(0, 3, 1, 2).contiguous(),
                    )
                    loss = torch.lerp(l1loss, ssimloss, torch.Tensor([self.config.ssim_lambda]).to(ssimloss))

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
                if pbar is not None:
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
                    self.optimizer.refine()

                    # If you specified a crop bounding box, clip the Gaussians that are outside the crop
                    # bounding box. This is useful if you want to train on a subset of the scene
                    # and don't want to waste resources on Gaussians that are outside the crop.
                    if self.config.remove_gaussians_outside_scene_bbox:
                        self._clip_gaussians_to_scene_bbox()

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

                # Log metrics
                if self._global_step % self._log_interval_steps == 0:
                    mem_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                    self._writer.log_metric(self._global_step, f"{log_tag}/loss", loss.item())
                    self._writer.log_metric(self._global_step, f"{log_tag}/l1loss", l1loss.item())
                    self._writer.log_metric(self._global_step, f"{log_tag}/ssimloss", ssimloss.item())
                    self._writer.log_metric(self._global_step, f"{log_tag}/num_gaussians", self.model.num_gaussians)
                    self._writer.log_metric(self._global_step, f"{log_tag}/sh_degree", sh_degree_to_use)
                    self._writer.log_metric(self._global_step, f"{log_tag}/mem_allocated", mem_allocated)
                    self._writer.log_metric(self._global_step, f"{log_tag}/mem_reserved", mem_reserved)
                    if pose_reg is not None:
                        self._writer.log_metric(self._global_step, f"{log_tag}/pose_reg_loss", pose_reg.item())

                # Update the viewer
                if self._viewer is not None and self._global_step % update_viewer_every_step == 0:
                    with torch.no_grad():
                        self._logger.info(f"Updating viewer at step {self._global_step:,}")
                        self._viewer.add_gaussian_splat_3d("Gaussian Scene", self.model)

                # Update the progress bar and global step
                if pbar is not None:
                    pbar.update(batch_size)
                    self._global_step = pbar.n
                else:
                    self._global_step += batch_size

                # Check if we've reached max_steps and break out of training
                if self.config.max_steps is not None and self._global_step >= self.config.max_steps:
                    reached_max_steps = True
                    break

            # Check if we've reached max_steps and break out of outer epoch loop
            if reached_max_steps:
                break

            # Save the model if we've reached a percentage of the total epochs specified in save_at_percent
            if epoch in [(pct * self.config.max_epochs // 100) - 1 for pct in self.config.save_at_percent]:
                if self._global_step <= self._start_step:
                    self._logger.info(
                        f"Skipping checkpoint save at epoch {epoch + 1} (before start step {self._start_step})."
                    )
                    continue
                self._logger.info(f"Saving checkpoint at global step {self._global_step}.")
                self._writer.save_checkpoint(self._global_step, f"{log_tag}_ckpt.pt", self.state_dict())
                self._writer.save_ply(self._global_step, f"{log_tag}_ckpt.ply", self.model, self.optimization_metadata)

            # Run evaluation if we've reached a percentage of the total epochs specified in eval_at_percent
            if epoch in [(pct * self.config.max_epochs // 100) - 1 for pct in self.config.eval_at_percent]:
                if len(self.validation_dataset) == 0:
                    continue
                if self._global_step <= self._start_step:
                    self._logger.info(
                        f"Skipping evaluation at epoch {epoch + 1} (before start step {self._start_step})."
                    )
                    continue
                self.eval(log_tag=log_tag + "_eval")

        self._logger.info("Training completed.")

    @torch.no_grad()
    def eval(self, log_tag: str = "eval") -> None:
        """
        Run evaluation of the Gaussian Splatting model on the validation dataset.

        This method evaluates the model by rendering images from the projected Gaussians and computing
        various image quality metrics.

        Args:
            log_tag (str): Tag to use for logging metrics and images. Data logged will use this tag as a prefix.
                For metrics, this will be "{log_tag}/metric_name".
                For images, this will be "{log_tag}/predicted_imageXXXX.jpg" and "{log_tag}/ground_truth_imageXXXX.jpg".
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

            # Save images
            self._writer.save_image(self._global_step, f"{log_tag}/predicted_image{i:04d}.jpg", predicted_image)
            self._writer.save_image(self._global_step, f"{log_tag}/ground_truth_image{i:04d}.jpg", ground_truth_image)

            ground_truth_image = ground_truth_image.permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
            predicted_image = predicted_image.permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
            metrics["psnr"].append(psnr(predicted_image, ground_truth_image))
            metrics["ssim"].append(ssim(predicted_image, ground_truth_image))
            metrics["lpips"].append(self._lpips(predicted_image, ground_truth_image))

        evaluation_time /= len(valloader)

        psnr_mean = torch.stack(metrics["psnr"]).mean()
        ssim_mean = torch.stack(metrics["ssim"]).mean()
        lpips_mean = torch.stack(metrics["lpips"]).mean()
        self._logger.info(f"Evaluation for stage {log_tag} completed. Average time per image: {evaluation_time:.3f}s")
        self._logger.info(f"PSNR: {psnr_mean.item():.3f}, SSIM: {ssim_mean.item():.4f}, LPIPS: {lpips_mean.item():.3f}")

        self._writer.log_metric(self._global_step, f"{log_tag}/psnr", psnr_mean.item())
        self._writer.log_metric(self._global_step, f"{log_tag}/ssim", ssim_mean.item())
        self._writer.log_metric(self._global_step, f"{log_tag}/lpips", lpips_mean.item())
        self._writer.log_metric(self._global_step, f"{log_tag}/evaluation_time", evaluation_time)
        self._writer.log_metric(self._global_step, f"{log_tag}/num_gaussians", self.model.num_gaussians)

        # Update the viewer with evaluation results
        if self._viewer is not None:
            self._logger.info(f"Updating viewer after evaluation at step {self._global_step:,}")
            self._viewer.add_gaussian_splat_3d("Gaussian Scene", self.model)
