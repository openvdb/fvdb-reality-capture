# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import hashlib
import logging
import math
import os
import pathlib
import random
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torchvision
import torchvision.transforms.functional as tvF
import tqdm
from fvdb_reality_capture.radiance_fields.gaussian_splatting import GaussianSplat3d
from fvdb_reality_capture.checkpoints import load_training_checkpoint
from fvdb_reality_capture.sfm_scene import SfmScene
from fvdb_reality_capture.tools import filter_splats_above_scale
from fvdb_reality_capture.instance_segmentation.checkpoint import (
    GARFVDB_TRAINING_METHOD,
    GARFVDB_TRAINING_METHOD_VERSION,
)
from fvdb_reality_capture.instance_segmentation.config import GARfVDBConfig, GARfVDBTrainingConfig
from fvdb_reality_capture.instance_segmentation.loss import calculate_loss
from fvdb_reality_capture.instance_segmentation.model import GARfVDBModel
from fvdb_reality_capture.instance_segmentation.optim import ExponentialLRWithRampUpScheduler
from fvdb_reality_capture.instance_segmentation.training.dataset import (
    GARfVDBInputCollateFn,
    InfiniteSampler,
    SegmentationDataset,
)
from fvdb_reality_capture.instance_segmentation.training.dataset_transforms import (
    GPURandomSelectMaskIDAndScale,
    RandomSamplePixels,
    RandomSelectMaskIDAndScale,
    TransformedSegmentationDataset,
)
from fvdb_reality_capture.instance_segmentation.training.segmentation_writer import GARfVDBWriter
from fvdb_reality_capture.instance_segmentation.util import pca_projection_fast
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from fvdb_reality_capture.instance_segmentation.garfvdb import GARfVDB


class TensorboardLogger:
    """Utility class for logging training metrics to TensorBoard."""

    def __init__(
        self,
        log_dir: pathlib.Path,
        log_every_step: int = 100,
        log_images_to_tensorboard: bool = False,
    ) -> None:
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard event files.
            log_every_step: Logging frequency in training steps.
            log_images_to_tensorboard: Whether to log rendered images.
        """
        self._log_every_step = log_every_step
        self._log_dir = log_dir
        self._log_images_to_tensorboard = log_images_to_tensorboard
        self._tb_writer = SummaryWriter(log_dir=log_dir)

    def log_training_iteration(
        self,
        step: int,
        metrics: dict[str, torch.Tensor],
        mem: float,
        gt_img: torch.Tensor | None,
        pred_img: torch.Tensor | None,
    ):
        """
        Log training metrics to TensorBoard.

        Args:
            step: Current training step.
            metrics: Dictionary of metrics to log.
            mem: Maximum GPU memory allocated in GB.
            gt_img: Ground truth image for visualization.
            pred_img: Predicted image for visualization.
        """
        if self._log_every_step > 0 and step % self._log_every_step == 0 and self._tb_writer is not None:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            # Log loss components to tensorboard
            for key, value in metrics.items():
                self._tb_writer.add_scalar(f"train/{key}", value.item(), step)
            self._tb_writer.add_scalar("train/mem", mem, step)
            if self._log_images_to_tensorboard and gt_img is not None and pred_img is not None:
                canvas = torch.cat([gt_img, pred_img], dim=2).detach().cpu().numpy()
                canvas = canvas.reshape(-1, *canvas.shape[2:])
                self._tb_writer.add_image("train/render", canvas, step)
            self._tb_writer.flush()

    def log_evaluation_iteration(
        self,
        step: int,
        loss: float,
        beauty_output: torch.Tensor,
        beauty_gt: torch.Tensor,
        sample_mask: torch.Tensor,
        sample_image: torch.Tensor,
    ):
        """
        Log evaluation metrics to TensorBoard.

        Args:
            step: The training step after which the evaluation was performed.
            loss: Loss value for the evaluation (averaged over all images in the validation set).
            beauty_output: Gaussian splat rendered beauty output for the evaluation
            beauty_gt: Ground truth beauty image from validation set for the evaluation
            sample_mask: Mask for the evaluation
            sample_image: Sample blended image with beauty output and mask for the evaluation
        """

        self._tb_writer.add_scalar("eval/loss", loss, step)
        self._tb_writer.add_image("eval/beauty_output", beauty_output, step)
        self._tb_writer.add_image("eval/beauty_gt", beauty_gt, step)
        self._tb_writer.add_image("eval/sample_mask", sample_mask, step)
        self._tb_writer.add_image("eval/sample_image", sample_image, step)


class GARfVDBTrainer:
    """Training and evaluation engine for scale-conditioned Gaussian splat segmentation.

    This class manages the complete training pipeline for GARfVDB segmentation models,
    including dataset loading, optimization, checkpointing, and evaluation. It supports
    both training from scratch and resuming from checkpoints.

    Attributes:
        version: Checkpoint format version string.
        model: The GARfVDB segmentation model.
        gs_model: The underlying GaussianSplat3d radiance field.
        sfm_scene: The transformed SfmScene with segmentation masks.
        config: Training configuration parameters.
    """

    version = GARFVDB_TRAINING_METHOD_VERSION

    _magic = "GARfVDBInstanceSegmentationCheckpoint"

    __PRIVATE__ = object()

    def __init__(
        self,
        model: GARfVDBModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        config: GARfVDBTrainingConfig,
        gs_model: GaussianSplat3d,
        gs_model_path: pathlib.Path,
        sfm_scene: SfmScene,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        train_transform: torchvision.transforms.Compose,
        val_transform: torchvision.transforms.Compose,
        writer: GARfVDBWriter,
        start_step: int,
        log_interval_steps: int,
        viewer_update_interval_epochs: int,
        grouping_scale_stats: torch.Tensor | None = None,
        viz_callback: Callable[["GARfVDBTrainer", int], None] | None = None,
        cache_dataset: bool = True,
        reconstruction_metadata: dict[str, Any] | None = None,
        enable_viewer: bool = False,
        viewer_scene_name: str = "GARfVDB Training",
        viewer_scale_fraction: float = 0.1,
        viewer_mask_blend: float = 0.5,
        viewer_lock_pca_colors: bool = False,
        viewer_overlay_width: int = 1440,
        viewer_overlay_height: int = 720,
        viewer_overlay_downsample: int = 2,
        viewer_update_interval_seconds: float = 0.5,
        _private: object | None = None,
    ) -> None:
        """
        Initialize the Runner with the provided configuration, model, optimizer, datasets, and paths.

        Note: This constructor should only be called by the `new` or `resume_from_checkpoint` methods.

        Args:
            model (GARfVDBModel): The GARfVDB model to train.
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
            config (GARfVDBTrainingConfig): Configuration object containing model parameters.
            gs_model (GaussianSplat3d): The optimized GaussianSplat3d model.
            gs_model_path (pathlib.Path): Path to the GaussianSplat3d model file.
            sfm_scene (SfmScene): The Structure-from-Motion scene.
            train_indices (np.ndarray): The indices for the training set.
            val_indices (np.ndarray): The indices for the validation set.
            train_transform (torchvision.transforms.Compose): The transform for training data.
            val_transform (torchvision.transforms.Compose): The transform for validation data.
            writer (GARfVDBWriter): The writer to use for logging and saving results.
            start_step (int): The step to start training from (useful for resuming from a checkpoint).
            log_interval_steps (int): How often to log metrics to TensorBoard.
            viewer_update_interval_epochs (int): How often to update the viewer.
            grouping_scale_stats (torch.Tensor | None): The scale statistics of the GaussianSplat3d model.
            viz_callback (Callable | None): Optional callback function called at epoch boundaries for visualization.
                The callback receives the runner instance and the current epoch number.
            cache_dataset (bool): If True, cache images and masks in memory to speed up data loading.
                Set to False to reduce memory usage for large datasets. Default is True.
            _private (object | None): Private object to ensure this class is only initialized
                through `new` or `resume_from_checkpoint`.
        """
        if _private is not GARfVDBTrainer.__PRIVATE__:
            raise ValueError("Runner should only be initialized through `new_run` or `resume_from_checkpoint`.")

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._cfg = config

        self._start_step = start_step

        self._sfm_scene = sfm_scene

        self._gs_model = gs_model
        self._gs_model_path = gs_model_path
        self._reconstruction_metadata = dict(reconstruction_metadata or {})
        self._carrier_means_sha256 = hashlib.sha256(
            gs_model.means.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()

        # Store scale statistics for checkpoint restoration
        # If provided (from checkpoint), use them directly; otherwise compute from the dataset
        if grouping_scale_stats is not None:
            self._grouping_scale_stats = grouping_scale_stats
        else:
            _full_dataset_for_scales = SegmentationDataset(
                sfm_scene=sfm_scene,
                cache_loaded_masks=cache_dataset,
                cache_images=cache_dataset,
            )
            self._grouping_scale_stats = _full_dataset_for_scales.scales

        self._training_dataset = SegmentationDataset(
            sfm_scene=sfm_scene,
            dataset_indices=train_indices,
            cache_loaded_masks=cache_dataset,
            cache_images=cache_dataset,
        )
        self._validation_dataset = SegmentationDataset(
            sfm_scene=sfm_scene,
            dataset_indices=val_indices,
            cache_loaded_masks=cache_dataset,
            cache_images=cache_dataset,
        )
        self._train_transforms = train_transform
        self._val_transforms = val_transform
        self._training_dataset = TransformedSegmentationDataset(
            self._training_dataset,
            train_transform,
        )
        self._validation_dataset = TransformedSegmentationDataset(
            self._validation_dataset,
            val_transform,
        )
        self._logger.info(
            f"Created dataset training and test datasets with {len(self._training_dataset)} training images and {len(self._validation_dataset)} test images."
        )

        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        if isinstance(self._scheduler, ExponentialLRWithRampUpScheduler):
            self._scheduler.max_steps = self.total_steps

        self._global_step: int = 0

        self._writer = writer
        self._log_interval_steps = log_interval_steps
        self._viewer_update_interval_epochs = viewer_update_interval_epochs
        self._viz_callback = viz_callback

        # Live training viewer (fvdb Viewer with interactive scale/opacity/show-overlay controls).
        # fviz.init(...) must already have been called by the caller before the model was moved to CUDA.
        self._enable_viewer = enable_viewer
        self._viewer_scene_name = viewer_scene_name
        self._viewer_scale_fraction = viewer_scale_fraction
        self._viewer_mask_blend = viewer_mask_blend
        self._viewer_lock_pca_colors = viewer_lock_pca_colors
        self._viewer_overlay_width = viewer_overlay_width
        self._viewer_overlay_height = viewer_overlay_height
        self._viewer_overlay_downsample = viewer_overlay_downsample
        self._viewer_update_interval_seconds = viewer_update_interval_seconds
        self._overlay_viewer = None
        self._last_viewer_render_time = 0.0

    @property
    def total_steps(self) -> int:
        """Get the total number of steps for training."""
        computed_total_steps: int = int(self._cfg.max_epochs * len(self._training_dataset))
        total_steps: int = self._cfg.max_steps if self._cfg.max_steps is not None else computed_total_steps
        return total_steps

    @property
    def model(self) -> GARfVDBModel:
        """Get the GARfVDB segmentation model."""
        return self._model

    @property
    def gs_model(self) -> GaussianSplat3d:
        """Get the underlying GaussianSplat3d model."""
        return self._gs_model

    @property
    def sfm_scene(self) -> SfmScene:
        """Get the transformed SfmScene used for training."""
        return self._sfm_scene

    @property
    def config(self) -> GARfVDBTrainingConfig:
        """Get the configuration used for training."""
        return self._cfg

    @property
    def grouping_scale_stats(self) -> torch.Tensor:
        """Get the grouping scale statistics used for the model."""
        return self._grouping_scale_stats

    @property
    def device(self) -> torch.device:
        """Get the device the model is running on."""
        dev = self._model.device
        if isinstance(dev, str):
            return torch.device(dev)
        return dev

    @property
    def viz_callback(self) -> Callable[["GARfVDBTrainer", int], None] | None:
        """Get or set the visualization callback invoked at epoch boundaries."""
        return self._viz_callback

    @viz_callback.setter
    def viz_callback(self, callback: Callable[["GARfVDBTrainer", int], None] | None) -> None:
        self._viz_callback = callback

    @staticmethod
    def _init_model(
        model_config: GARfVDBConfig,
        gs_model: GaussianSplat3d,
        grouping_scale_stats: torch.Tensor,
        device: torch.device | str,
    ) -> GARfVDBModel:
        """
        Initialize the GARfVDB model.

        Args:
            model_config: Configuration object containing model parameters.
            gs_model: The optimized GaussianSplat3d model.
            grouping_scale_stats: The scale statistics of the GaussianSplat3d model.
            device: The device to run the model on (e.g., "cuda" or "cpu").
        """

        return GARfVDBModel(
            gs_model,
            grouping_scale_stats,
            model_config=model_config,
            device=device,
        )

    @classmethod
    def new(
        cls,
        sfm_scene: SfmScene,
        gs_model: GaussianSplat3d,
        gs_model_path: pathlib.Path,
        writer: GARfVDBWriter,
        config: GARfVDBTrainingConfig = GARfVDBTrainingConfig(),
        device: str | torch.device = "cuda:0",
        use_every_n_as_val: int = 100,
        exclude_indices: Sequence[int] | None = None,
        viewer_update_interval_epochs: int = 10,
        log_interval_steps: int = 10,
        viz_callback: Callable[["GARfVDBTrainer", int], None] | None = None,
        cache_dataset: bool = True,
        reconstruction_metadata: dict[str, Any] | None = None,
        enable_viewer: bool = False,
        viewer_scale_fraction: float = 0.1,
        viewer_mask_blend: float = 0.5,
        viewer_lock_pca_colors: bool = False,
        viewer_overlay_width: int = 1440,
        viewer_overlay_height: int = 720,
        viewer_overlay_downsample: int = 2,
    ) -> "GARfVDBTrainer":
        """
        Create a `GARfVDBTrainer` instance for a new training run.

        Args:
            sfm_scene (SfmScene): The SfmScene to train on.
            gs_model (GaussianSplat3d): The optimized GaussianSplat3d model.
            gs_model_path (pathlib.Path): Path to the GaussianSplat3d model file.
            writer (GARfVDBWriter): The writer to use for logging and saving results.
            config (GARfVDBTrainingConfig): Configuration object containing model parameters.
            device (str | torch.device): The device to run the model on (e.g., "cuda" or "cpu").
            use_every_n_as_val (int): Use every nth image as validation data.
            exclude_indices (Sequence[int] | None): Indices to exclude from both training and validation.
                These images will still be used for computing scale statistics but won't be trained on.
                This is useful for excluding test/evaluation images from training (e.g., NVOS benchmark).
            viewer_update_interval_epochs (int): How often to update the viewer.
            log_interval_steps (int): How often to log metrics to TensorBoard.
            viz_callback (Callable | None): Optional callback function called at epoch boundaries for visualization.
                The callback receives the runner instance and the current epoch number.
            cache_dataset (bool): If True, cache images and masks in memory to speed up data loading.
                Set to False to reduce memory usage for large datasets. Default is True.

        Returns:
            GARfVDBTrainer: A `GARfVDBTrainer` instance initialized with the specified configuration and datasets.
        """

        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if not config.model.use_grid:
            raise ValueError("First-class GARfVDB training requires GARfVDBConfig.use_grid=True")

        logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        ## SfmScene
        ## Split into train and validation sets
        indices = np.arange(sfm_scene.num_images)

        # Exclude specified indices (e.g., test images for evaluation benchmarks)
        if exclude_indices is not None and len(exclude_indices) > 0:
            exclude_set = set(exclude_indices)
            indices = np.array([i for i in indices if i not in exclude_set], dtype=int)
            logger.info(f"Excluding {len(exclude_indices)} images from training: {list(exclude_indices)}")

        if use_every_n_as_val > 0:
            mask = np.ones(len(indices), dtype=bool)
            mask[::use_every_n_as_val] = False
            train_indices = indices[mask]
            val_indices = indices[~mask]
        else:
            train_indices = indices
            val_indices = np.array([], dtype=int)

        ## SegmentationDataset transforms
        # For training, randomly sample pixels from each image
        # Note: RandomSelectMaskIDAndScale is applied on GPU after data transfer for better performance
        train_transforms = torchvision.transforms.Compose(
            [
                RandomSamplePixels(config.sample_pixels_per_image, scale_bias_strength=0.0),
            ]
        )
        val_transforms = torchvision.transforms.Compose(
            [
                RandomSamplePixels(config.sample_pixels_per_image),
            ]
        )
        # For testing, use the full image, scaled down for memory reasons
        # test_dataset = TransformedSegmentationDataset(test_dataset, Compose([Resize(1 / 6), RandomSelectMaskIDAndScale()]))

        ## Initialize Model
        # Scale grouping stats
        full_dataset = SegmentationDataset(
            sfm_scene,
            cache_loaded_masks=cache_dataset,
            cache_images=cache_dataset,
        )
        grouping_scale_stats = full_dataset.scales

        gs_model = filter_splats_above_scale(gs_model, 0.1)
        # gs_model = filter_splat_means(gs_model, [0.95, 0.95, 0.95, 0.95, 0.95, 0.999])

        model = GARfVDBTrainer._init_model(config.model, gs_model, grouping_scale_stats, device)
        logger.info(f"Model initialized with {gs_model.num_gaussians:,} Gaussians")

        ## Initialize Optimizer
        base_lr = 1e-5
        lr = base_lr
        if config.scale_lr_with_batch_size and config.batch_size > 1:
            lr = base_lr * math.sqrt(config.batch_size)
            logger.info(f"Scaled LR from {base_lr:.2e} to {lr:.2e} (sqrt({config.batch_size}) scaling)")
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            eps=1e-15,
            weight_decay=1e-6,
        )

        scheduler = ExponentialLRWithRampUpScheduler(
            optimizer=optimizer,
            lr_init=lr,
            lr_final=1e-6,
        )

        return GARfVDBTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            gs_model=gs_model,
            gs_model_path=gs_model_path,
            sfm_scene=sfm_scene,
            train_indices=train_indices,
            val_indices=val_indices,
            train_transform=train_transforms,
            val_transform=val_transforms,
            writer=writer,
            log_interval_steps=log_interval_steps,
            viewer_update_interval_epochs=viewer_update_interval_epochs,
            start_step=0,
            viz_callback=viz_callback,
            cache_dataset=cache_dataset,
            reconstruction_metadata=reconstruction_metadata,
            enable_viewer=enable_viewer,
            viewer_scale_fraction=viewer_scale_fraction,
            viewer_mask_blend=viewer_mask_blend,
            viewer_lock_pca_colors=viewer_lock_pca_colors,
            viewer_overlay_width=viewer_overlay_width,
            viewer_overlay_height=viewer_overlay_height,
            viewer_overlay_downsample=viewer_overlay_downsample,
            _private=GARfVDBTrainer.__PRIVATE__,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        gs_model: GaussianSplat3d,
        gs_model_path: pathlib.Path,
        writer: GARfVDBWriter | None = None,
        device: str | torch.device = "cuda:0",
        eval_only: bool = False,
    ) -> "GARfVDBTrainer":
        """
        Load a :class:`GARfVDBTrainer` instance from a state dictionary.

        This method restores the model, optimizer, SfmScene (with correct transforms/scales), and configuration
        from a checkpoint. The restored SfmScene contains the correct scale statistics that were used during training.

        Args:
            state_dict (dict): State dictionary containing the model, optimizer, and configuration state.
                Generated by the :meth:`state_dict` method.
            gs_model (GaussianSplat3d): GaussianSplat3d model.
            gs_model_path (pathlib.Path): Path to the GaussianSplat3d model.
            writer (GARfVDBWriter | None): Optional writer for logging. If None, a default
                writer with no output will be used.
            device (str | torch.device): Device to load the model onto.
            eval_only (bool): If True, disables gradients on all model parameters for evaluation only.
                This is useful when loading for visualization where training is not needed.

        Returns:
            GARfVDBTrainer: A restored instance ready for evaluation or continued training.
        """
        logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        # Validate checkpoint
        magic = state_dict.get("magic")
        if magic != cls._magic:
            raise ValueError(f"State dict has invalid GARfVDB magic value: {magic!r}")
        if state_dict.get("version") != cls.version:
            raise ValueError(
                f"Checkpoint version {state_dict.get('version')!r} does not match supported version {cls.version!r}."
            )
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

        global_step = state_dict["step"]
        config_state = dict(state_dict["config"])
        model_config_state = config_state.get("model", {})
        if isinstance(model_config_state, dict):
            config_state["model"] = GARfVDBConfig(**model_config_state)
        config = GARfVDBTrainingConfig(**config_state)

        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Restore SfmScene (this is the transformed scene with correct scales)
        sfm_scene: SfmScene = SfmScene.from_state_dict(state_dict["sfm_scene"])
        logger.info(f"Restored SfmScene with {sfm_scene.num_images} images")

        # Disable gradients on GS model tensors if eval_only
        if eval_only:
            gs_model.means.detach_()
            gs_model.quats.detach_()
            gs_model.log_scales.detach_()
            gs_model.logit_opacities.detach_()
            gs_model.sh0.detach_()
            gs_model.shN.detach_()

        # Get train/val indices
        train_indices = np.array(state_dict["train_indices"], dtype=int)
        val_indices = np.array(state_dict["val_indices"], dtype=int)

        # Get scale statistics from the checkpoint (saved during training)
        if "grouping_scale_stats" not in state_dict:
            raise ValueError(
                "Checkpoint is missing 'grouping_scale_stats'. Please retrain to generate a new checkpoint."
            )
        grouping_scale_stats = state_dict["grouping_scale_stats"]
        if isinstance(grouping_scale_stats, np.ndarray):
            grouping_scale_stats = torch.from_numpy(grouping_scale_stats)
        grouping_scale_stats = grouping_scale_stats.to(device)
        logger.info(f"Restored grouping_scale_stats with {len(grouping_scale_stats)} entries")

        # Initialize model from state dict
        model = GARfVDBModel.create_from_state_dict(state_dict["model"], config.model, gs_model, grouping_scale_stats)
        model.to(device)
        model.eval()

        # Disable gradients on all model parameters if eval_only
        if eval_only:
            for param in model.parameters():
                param.requires_grad_(False)
            logger.info("Restored GARfVDB segmentation model (eval mode, gradients disabled)")
        else:
            logger.info("Restored GARfVDB segmentation model")

        # Create transforms (same as in new())
        train_transforms = torchvision.transforms.Compose(
            [
                RandomSamplePixels(config.sample_pixels_per_image, scale_bias_strength=0.0),
                RandomSelectMaskIDAndScale(),
            ]
        )
        val_transforms = torchvision.transforms.Compose(
            [
                RandomSamplePixels(config.sample_pixels_per_image),
                RandomSelectMaskIDAndScale(),
            ]
        )

        # Initialize optimizer
        base_lr = 1e-5
        lr = base_lr
        if config.scale_lr_with_batch_size and config.batch_size > 1:
            lr = base_lr * math.sqrt(config.batch_size)
            logger.info(f"Scaled LR from {base_lr:.2e} to {lr:.2e} (sqrt({config.batch_size}) scaling)")
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            eps=1e-15,
            weight_decay=1e-6,
        )
        optimizer.load_state_dict(state_dict["optimizer"])

        # Create scheduler using the LR restored from the checkpoint
        restored_lr = optimizer.param_groups[0]["lr"]
        scheduler = ExponentialLRWithRampUpScheduler(
            optimizer=optimizer,
            lr_init=restored_lr,
            lr_final=1e-6,
        )
        scheduler_state = state_dict.get("scheduler")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        # Create default writer if not provided
        if writer is None:
            writer = GARfVDBWriter(run_name=None, save_path=None)

        logger.info(f"Restored from checkpoint at step {global_step}")

        return GARfVDBTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            gs_model=gs_model,
            gs_model_path=gs_model_path,
            sfm_scene=sfm_scene,
            train_indices=train_indices,
            val_indices=val_indices,
            train_transform=train_transforms,
            val_transform=val_transforms,
            writer=writer,
            log_interval_steps=10,
            viewer_update_interval_epochs=-1,
            start_step=global_step,
            grouping_scale_stats=grouping_scale_stats,
            reconstruction_metadata=state_dict.get("reconstruction_metadata", {}),
            _private=GARfVDBTrainer.__PRIVATE__,
        )

    @classmethod
    def load_checkpoint_state(
        cls, checkpoint_path: pathlib.Path, map_location: str | torch.device = "cpu"
    ) -> dict[str, Any]:
        """Load a first-class GARfVDB training checkpoint dictionary."""
        return load_training_checkpoint(
            checkpoint_path,
            map_location=map_location,
            expected_method=GARFVDB_TRAINING_METHOD,
        ).state

    @classmethod
    def from_checkpoint_file(
        cls,
        checkpoint_path: pathlib.Path,
        *,
        writer: GARfVDBWriter | None = None,
        device: str | torch.device = "cuda:0",
        reconstruction_path: pathlib.Path | None = None,
    ) -> "GARfVDBTrainer":
        """Restore a trainer and its carrier from a checkpoint path."""
        checkpoint = cls.load_checkpoint_state(checkpoint_path, map_location=device)
        return cls.from_checkpoint_state(
            checkpoint,
            writer=writer,
            device=device,
            reconstruction_path=reconstruction_path,
        )

    @classmethod
    def from_checkpoint_state(
        cls,
        checkpoint: dict[str, Any],
        *,
        writer: GARfVDBWriter | None = None,
        device: str | torch.device = "cuda:0",
        reconstruction_path: pathlib.Path | None = None,
    ) -> "GARfVDBTrainer":
        """Restore a trainer and carrier from already loaded method state."""
        from fvdb_reality_capture.radiance_fields import load_splats_from_file

        stored_path = checkpoint.get("gs_model_path")
        carrier_path = reconstruction_path or (pathlib.Path(stored_path) if stored_path is not None else None)
        if carrier_path is None:
            raise ValueError("Checkpoint has no reconstruction path; pass reconstruction_path explicitly.")
        if not carrier_path.exists():
            raise FileNotFoundError(
                f"Reconstruction referenced by GARfVDB checkpoint does not exist: {carrier_path}. "
                "Pass --reconstruction-path if it moved."
            )
        gs_model, metadata = load_splats_from_file(carrier_path, device)
        gs_model = filter_splats_above_scale(gs_model, checkpoint.get("carrier_filter_threshold", 0.1))
        expected_count = checkpoint.get("carrier_num_gaussians")
        if expected_count is not None and gs_model.num_gaussians != expected_count:
            raise ValueError(
                f"Reconstructed carrier has {gs_model.num_gaussians} Gaussians; checkpoint expects {expected_count}."
            )
        expected_hash = checkpoint.get("carrier_means_sha256")
        actual_hash = hashlib.sha256(gs_model.means.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError("Reconstructed carrier ordering/content does not match the GARfVDB checkpoint")
        checkpoint["reconstruction_metadata"] = checkpoint.get("reconstruction_metadata", metadata)
        return cls.from_state_dict(
            checkpoint,
            gs_model=gs_model,
            gs_model_path=carrier_path,
            writer=writer,
            device=device,
        )

    def to_product(self) -> "GARfVDB":
        """Create the portable inference product represented by the trained state."""
        from fvdb_reality_capture.instance_segmentation.garfvdb import GARfVDB

        return GARfVDB(model=self._model, reconstruction_metadata=self._reconstruction_metadata)

    def _setup_viewer(self) -> None:
        """Start the live training viewer if enabled. fviz.init(...) must already have been called."""
        if not self._enable_viewer:
            return
        try:
            import fvdb.viz as fviz

            from fvdb_reality_capture.instance_segmentation.viewer import GARfVDBOverlayViewer

            scene = fviz.get_scene(self._viewer_scene_name)
            # to_product() wraps the live model by reference, so overlay frames reflect current weights.
            self._overlay_viewer = GARfVDBOverlayViewer(
                scene,
                self.to_product(),
                device=self._model.device,
                overlay_width=self._viewer_overlay_width,
                overlay_height=self._viewer_overlay_height,
                overlay_downsample=self._viewer_overlay_downsample,
                initial_scale_fraction=self._viewer_scale_fraction,
                initial_mask_blend=self._viewer_mask_blend,
                lock_pca_colors=self._viewer_lock_pca_colors,
            )
            fviz.show()
            self._logger.info("GARfVDB training viewer running.")
        except Exception as e:  # never let viewer setup take down training
            self._logger.warning(f"Failed to start GARfVDB training viewer: {e}")
            self._overlay_viewer = None

    def _pump_viewer(self) -> None:
        """Render one overlay frame from the current model, throttled by wall-clock. Never raises."""
        if self._overlay_viewer is None:
            return
        now = time.monotonic()
        if now - self._last_viewer_render_time < self._viewer_update_interval_seconds:
            return
        self._last_viewer_render_time = now
        was_training = self._model.training
        try:
            self._model.eval()
            with torch.no_grad():
                self._overlay_viewer.render_once()
        except Exception as e:  # a viewer hiccup must never interrupt training
            self._logger.warning(f"GARfVDB training viewer update failed: {e}")
        finally:
            if was_training:
                self._model.train()

    def train(self, show_progress: bool = True, log_tag: str = "train"):
        if self._optimizer is None:
            raise ValueError("This runner was not created with an optimizer. Cannot run training.")
        logging.debug(f"Training with batch size {self._cfg.batch_size}")

        # Pre-warm the cache BEFORE creating DataLoader workers
        # Workers inherit the populated cache via fork, eliminating disk I/O during training
        with nvtx.range("dataset_warmup_cache"):
            self._training_dataset.warmup_cache()

        # Scale num_workers based on dataset size to avoid IPC overhead dominating
        # Even for small datasets, use a few workers to overlap CPU transforms with GPU training
        dataset_size = len(self._training_dataset)
        if dataset_size <= self._cfg.batch_size * 2:
            num_workers = 2
            self._logger.info(f"Small dataset ({dataset_size} images): using num_workers=2")
        elif dataset_size <= 64:
            # Small dataset: limit workers to avoid contention
            num_workers = min(4, os.cpu_count() or 4)
        else:
            num_workers = min(8, os.cpu_count() or 4)

        # Use InfiniteSampler to avoid pauses at epoch boundaries
        # The sampler never stops, so we track epochs by counting batches
        infinite_sampler = InfiniteSampler(self._training_dataset, shuffle=True, seed=self._cfg.seed)

        # Use collate function with mask_cdf included for GPU-based mask selection
        from functools import partial

        collate_fn = partial(GARfVDBInputCollateFn, include_mask_cdf=True)

        # For small datasets, use smaller prefetch to avoid wraparound thrashing
        prefetch = 2 if num_workers > 0 else None
        if num_workers > 0 and dataset_size < self._cfg.batch_size * 4:
            prefetch = 1  # Reduce prefetch for tiny datasets

        trainloader = torch.utils.data.DataLoader(
            self._training_dataset,
            batch_size=self._cfg.batch_size,
            sampler=infinite_sampler,  # Use infinite sampler instead of shuffle
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=prefetch,
            collate_fn=collate_fn,
        )

        # GPU transform for mask selection (moved from CPU for better performance)
        gpu_mask_transform = GPURandomSelectMaskIDAndScale()

        self._model.train()
        self._optimizer.zero_grad()

        # Start the live training viewer (no-op unless enabled).
        self._setup_viewer()

        pbar = tqdm.tqdm(range(self.total_steps), desc="Training")

        # Gradient accumulation settings
        gradient_accumulation_steps = self._cfg.accumulate_grad_steps
        # Track accumulated loss as a tensor to avoid .item() sync on every step
        accumulated_loss: torch.Tensor | float = 0.0

        # Zero out gradients before training in case we resume training
        self._optimizer.zero_grad()

        # Track epochs by samples processed (global_step), not batches
        # This correctly aligns with how InfiniteSampler yields exactly dataset_size indices per epoch
        samples_per_epoch = len(self._training_dataset)
        prev_epoch = -1

        trainloader_iter = iter(trainloader)
        step = 0
        while True:
            # Refresh the live viewer between steps (throttled internally; no-op unless enabled).
            self._pump_viewer()

            with nvtx.range("dataloader_fetch_batch"):
                try:
                    minibatch = next(trainloader_iter)
                except StopIteration:
                    break

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

            # Move to device
            with nvtx.range("data_to_device"):
                minibatch = minibatch.to(self.device)

            # Apply GPU-accelerated mask transform
            with nvtx.range("gpu_mask_transform"):
                minibatch = gpu_mask_transform(minibatch)

            # Debug prints for first iteration
            if step == 0:
                logging.debug(f"Training with sample_pixels_per_image={self._cfg.sample_pixels_per_image}")
                logging.debug(f"Batch size: {minibatch['image'].shape[0]}")
                logging.debug(f"scales shape: {minibatch['scales'].shape}")
                logging.debug(f"mask_ids shape: {minibatch['mask_ids'].shape}")
                if "pixel_coords" in minibatch and minibatch["pixel_coords"] is not None:
                    logging.debug(f"pixel_coords shape: {minibatch['pixel_coords'].shape}")
                logging.debug(f"mean2d shape: {self._model.gs_model.means.shape}")
                logging.debug(f"Using gradient accumulation over {gradient_accumulation_steps} steps")

            ### Forward pass ###
            with nvtx.range("forward_pass"):
                cam_enc_feats = self._model.get_encoded_features(minibatch)
                logging.debug(f"Cam enc feats shape: {cam_enc_feats.shape}")

            with nvtx.range("loss_calculation"):
                loss_dict = calculate_loss(self._model, cam_enc_feats, minibatch)
                loss = loss_dict["total_loss"]
                logging.debug(f"Loss: {loss.item()}")

            # Scale loss by accumulation steps to maintain same effective learning rate
            loss = loss / gradient_accumulation_steps
            # Accumulate as tensor to avoid .item() sync on every step
            accumulated_loss = accumulated_loss + loss.detach()

            # Zero gradients only at the beginning of accumulation cycle
            if step % gradient_accumulation_steps == 0:
                self._optimizer.zero_grad()
            logging.debug(f"Backward pass")
            with nvtx.range("backward_pass"):
                loss.backward()

            # Perform optimizer step and gradient clipping only after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0:
                logging.debug(f"Optimizer step")
                with nvtx.range("optimizer_step"):
                    if self._cfg.grad_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._cfg.grad_clip_max_norm)
                    self._optimizer.step()
                    self._scheduler.step()

                # Reset accumulated loss
                accumulated_loss = 0.0

            ## Logging and checkpointing
            # Log train metrics
            if self._global_step % self._log_interval_steps == 0:
                display_loss = loss.item() * gradient_accumulation_steps
                # For display purposes, show the scaled loss
                pbar.set_postfix(loss=f"{display_loss:.4g}")

                mem_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                self._writer.log_metric(self._global_step, f"{log_tag}/loss", display_loss)
                self._writer.log_metric(self._global_step, f"{log_tag}/mem_allocated", mem_allocated)
                self._writer.log_metric(self._global_step, f"{log_tag}/mem_reserved", mem_reserved)

            # Update the progress bar and global step
            if pbar is not None:
                pbar.update(batch_size)
                self._global_step = pbar.n
            else:
                self._global_step += batch_size

            # Calculate current epoch from samples processed
            epoch = self._global_step // samples_per_epoch

            # Check for epoch boundary - run save/eval only once per epoch transition
            if epoch > prev_epoch:
                prev_epoch = epoch

                # Update visualization if enabled
                if (
                    self._viz_callback is not None
                    and self._viewer_update_interval_epochs > 0
                    and epoch % self._viewer_update_interval_epochs == 0
                ):
                    try:
                        self._viz_callback(self, epoch)
                    except Exception as e:
                        self._logger.warning(f"Visualization callback failed: {e}")

                # Save the model if we've reached a percentage of the total epochs specified in save_at_percent
                if epoch in [pct * self._cfg.max_epochs // 100 for pct in self._cfg.save_at_percent]:
                    logging.info(f"Saving checkpoint at epoch {epoch}")
                    logging.info(f"Global step: {self._global_step}")
                    if self._global_step > self._start_step:
                        self._logger.info(f"Saving checkpoint at global step {self._global_step}.")
                        self._writer.save_checkpoint(self._global_step, f"{log_tag}_ckpt.pt", self.state_dict())

                # Run evaluation if we've reached a percentage of the total epochs specified in eval_at_percent
                if epoch in [pct * self._cfg.max_epochs // 100 for pct in self._cfg.eval_at_percent]:
                    logging.info(f"Running evaluation at epoch {epoch}")
                    if len(self._validation_dataset) > 0 and self._global_step > self._start_step:
                        self.eval(log_tag=log_tag + "_eval")

            # Check if we've reached max_steps or max_epochs
            if self._cfg.max_steps is not None and self._global_step >= self._cfg.max_steps:
                logging.debug(f"Reached max steps: {self._cfg.max_steps}")
                break
            if epoch >= self._cfg.max_epochs:
                logging.debug(f"Reached max epochs: {self._cfg.max_epochs}")
                break

            step += 1

        self._logger.info("Training completed.")

    @torch.inference_mode()
    def eval(self, log_tag: str = "eval") -> None:
        """
        Run evaluation of the Gaussian Splatting model on the validation dataset.

        Args:
            log_tag (str): The tag to use for logging the evaluation results.
        """
        self._logger.info("Running evaluation...")

        valloader = torch.utils.data.DataLoader(
            self._validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=lambda batch: GARfVDBInputCollateFn(batch, collate_full_image=True, include_mask_cdf=True),
        )

        # GPU transform for mask selection
        gpu_mask_transform = GPURandomSelectMaskIDAndScale()

        ## Log metrics
        metrics = {"psnr": [], "loss": []}
        for val_batch in valloader:
            val_batch = val_batch.to(self.device)
            val_batch = gpu_mask_transform(val_batch)
            with nvtx.range("eval_forward_pass"):
                val_enc_feats = self._model.get_encoded_features(val_batch)
            with nvtx.range("eval_loss_calculation"):
                val_loss_dict = calculate_loss(self._model, val_enc_feats, val_batch)
            metrics["loss"].append(val_loss_dict["total_loss"])

        loss_mean = torch.stack(metrics["loss"]).mean().cpu()

        self._logger.info(f"Evaluation for stage {log_tag} completed. Average loss: {loss_mean.item():.3f}")

        self._writer.log_metric(self._global_step, f"{log_tag}/loss", loss_mean.item())

        # Clean up memory because no-grid rendering is memory intensive
        del loss_mean, metrics
        torch.cuda.empty_cache()

        ## Validation image
        val_batch = next(iter(valloader)).to(self.device)
        val_batch = gpu_mask_transform(val_batch)

        world_to_cam_matrix = val_batch["world_to_camera"]
        projection_matrix = val_batch["projection"]
        beauty_gt = val_batch["image_full"]
        img_w, img_h = val_batch["image_w"][0], val_batch["image_h"][0]
        with nvtx.range("eval_render_beauty"):
            beauty_output, _ = self.gs_model.render_images(
                world_to_cam_matrix,
                projection_matrix,
                img_w,
                img_h,
                0.01,
                1e10,
            )
        # Ground truth and rendered image for reference
        beauty_output = torch.clamp(beauty_output, 0.0, 1.0)

        # scale image to 1/2 size
        def downscale(image: torch.Tensor) -> torch.Tensor:
            return tvF.resize(image.permute(0, 3, 1, 2), size=[img_h // 2, img_w // 2], antialias=True).permute(
                0, 2, 3, 1
            )

        # Save reference images to step 0 (they don't change during training)
        self._writer.save_image(0, f"{log_tag}/beauty_image.jpg", downscale(beauty_output).cpu())
        self._writer.save_image(0, f"{log_tag}/ground_truth_image.jpg", downscale(beauty_gt).cpu())

        # Mask outputs at 10% of the maximum scale
        desired_scale = torch.max(val_batch["scales"]) * 0.1

        with nvtx.range("eval_get_mask_output"):
            val_mask_output, mask_alpha = self._model.get_mask_output(val_batch, desired_scale.item())
        logging.debug(f"mask_alpha zeros: {mask_alpha.sum()}")

        with nvtx.range("eval_pca_projection"):
            pca_output = pca_projection_fast(val_mask_output, 3, mask=mask_alpha.squeeze(-1) > 0)
        pca_output = pca_output.clamp(0.0, 1.0)  # [B, H, W, 3]
        self._writer.save_image(self._global_step, f"{log_tag}/mask.jpg", downscale(pca_output).cpu())
        alpha = 0.7
        blended = beauty_output * (1 - alpha) + pca_output * alpha

        self._writer.save_image(self._global_step, f"{log_tag}/mask_beauty_blended.jpg", downscale(blended).cpu())

        # # Log test loss images
        # if self.config.log_test_images:
        #     with torch.no_grad():
        #         for test_batch_idx, test_batch in enumerate(testloader):
        #             if test_batch_idx != 0:
        #                 break

        #             # Log the image
        #             # Permute from [H, W, C] to [C, H, W] for TensorBoard
        #             test_image = test_batch["image"][0].cpu().permute(2, 0, 1)
        #             self.writer.add_image("test/sample_image", test_image, step)
        #             # Move input to device
        #             for k, v in test_batch.items():
        #                 test_batch[k] = v.to(self.device)
        #             # Set scales to 0.1
        #             test_batch["scales"] = torch.full_like(test_batch["scales"], 0.05)
        #             test_enc_feats = self.model.get_encoded_features(test_batch)

        #             # NOTE: If this starts using too much memory, we can move the image loss calculation to the CPU
        #             test_loss_dict = calculate_loss(self.model, test_enc_feats, test_batch, return_loss_images=True)
        #             for key in ("instance_loss_1", "instance_loss_2", "instance_loss_4"):
        #                 self.writer.add_image(f"test/{key}_img", test_loss_dict[f"{key}_img"].detach().cpu(), step)
        #             self.model.to(self.device)

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
                - gs_model_path: Path to the GaussianSplat3d model file.
                - grouping_scale_stats: The scale statistics of the GaussianSplat3d model.
                - model: The state dictionary of the Gaussian Splatting model.
                - optimizer: The state dictionary of the optimizer.
                - train_indices: The indices of the training dataset.
                - val_indices: The indices of the validation dataset.
        """
        return {
            "magic": self._magic,
            "version": self.version,
            "step": self._global_step,
            "config": asdict(self._cfg),
            "sfm_scene": self._sfm_scene.state_dict(),
            "gs_model_path": self._gs_model_path,
            "reconstruction_metadata": self._reconstruction_metadata,
            "carrier_filter_threshold": 0.1,
            "carrier_num_gaussians": self._gs_model.num_gaussians,
            "carrier_means_sha256": self._carrier_means_sha256,
            "grouping_scale_stats": self._grouping_scale_stats.cpu(),
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "train_indices": self._training_dataset.indices,
            "val_indices": self._validation_dataset.indices,
        }


__all__ = ["GARfVDBTrainer"]
