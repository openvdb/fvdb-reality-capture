# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import dataclass, field

import torch
from fvdb_reality_capture.radiance_fields.gaussian_splatting import GaussianSplat3d

from fvdb_reality_capture.instance_segmentation.scene_transforms import (
    GenerateGARfVDBMasks,
)
from fvdb_reality_capture.transforms import (
    Identity,
    SceneTransformConfig,
    TransformScene,
    UndistortImages,
)


@dataclass
class GARfVDBConfig:
    """Configuration parameters for the GARfVDB segmentation model."""

    depth_samples: int = 24
    """Number of depth samples per ray for feature computation."""

    use_grid: bool = True
    """Use 3D feature grids (GARField-style). First-class training and products require ``True``."""

    use_grid_conv: bool = False
    """If True, apply sparse convolutions to grid features."""

    enc_feats_one_idx_per_ray: bool = False
    """If True, stochastically sample one feature per ray instead of weighted averaging."""

    num_grids: int = 24
    """Number of feature grids at different resolutions."""

    grid_feature_dim: int = 8
    """Feature dimension per grid."""

    mlp_hidden_dim: int = 256
    """Hidden layer dimension in the MLP."""

    mlp_num_layers: int = 4
    """Number of hidden layers in the MLP."""

    mlp_output_dim: int = 256
    """Output dimension of the MLP (feature embedding size)."""


@dataclass
class GARfVDBTrainingConfig:
    """Configuration parameters for the segmentation training process."""

    seed: int = 42
    """Random seed for reproducibility."""

    max_steps: int | None = None
    """Maximum number of training steps. If None, uses max_epochs."""

    max_epochs: int = 100
    """Maximum number of training epochs."""

    sample_pixels_per_image: int = 256
    """Number of pixels to sample per image for training."""

    batch_size: int = 1
    """Number of images per training batch."""

    accumulate_grad_steps: int = 1
    """Number of gradient accumulation steps."""

    scale_lr_with_batch_size: bool = True
    """Scale learning rate by sqrt(batch_size) to compensate for fewer optimizer steps per epoch
    when using larger batches. Disabled when batch_size <= 1."""

    grad_clip_max_norm: float | None = 1.0
    """Maximum gradient norm for clipping. Set to None to disable gradient clipping."""

    model: GARfVDBConfig = field(default_factory=GARfVDBConfig)
    """Model architecture configuration."""

    log_test_images: bool = False
    """Whether to log test images during training."""

    eval_at_percent: list[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 75, 100])
    """Percentages of total epochs at which to run evaluation."""

    save_at_percent: list[int] = field(default_factory=lambda: [10, 20, 100])
    """Percentages of total epochs at which to save checkpoints."""


@dataclass
class GARfVDBTransformConfig(SceneTransformConfig):
    """Configuration for SfmScene transforms applied before segmentation training."""

    compute_segmentation_masks: bool = True
    """Whether to compute SAM2 segmentation masks."""

    sam2_points_per_side: int = 40
    """SAM2 grid density for automatic mask generation."""

    sam2_points_per_batch: int = 128
    """Number of SAM2 point prompts run through the mask decoder per forward pass. Higher values
    speed up mask generation at the cost of more GPU memory; does not change the generated masks."""

    sam2_pred_iou_thresh: float = 0.80
    """SAM2 predicted IoU threshold for mask filtering."""

    sam2_stability_score_thresh: float = 0.80
    """SAM2 stability score threshold for mask filtering."""

    device: torch.device | str = "cuda:0"
    """Device for SAM2 model inference."""

    def build_scene_transforms(self, gs3d: GaussianSplat3d, normalization_transform: torch.Tensor | None):
        alignment_transform = (
            TransformScene(normalization_transform.cpu().numpy()) if normalization_transform is not None else Identity()
        )
        terminal_transforms = (
            [
                UndistortImages(),
                GenerateGARfVDBMasks(
                    gs3d=gs3d,
                    checkpoint="large",
                    points_per_side=self.sam2_points_per_side,
                    points_per_batch=self.sam2_points_per_batch,
                    pred_iou_thresh=self.sam2_pred_iou_thresh,
                    stability_score_thresh=self.sam2_stability_score_thresh,
                    device=self.device,
                ),
            ]
            if self.compute_segmentation_masks
            else []
        )
        return self.build_scene_transform(
            alignment_transform=alignment_transform,
            terminal_transforms=terminal_transforms,
        )


__all__ = [
    "GARfVDBConfig",
    "GARfVDBTrainingConfig",
    "GARfVDBTransformConfig",
]
