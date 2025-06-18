# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration parameters specific to the model."""

    depth_samples: int = 24
    use_grid: bool = True
    use_grid_conv: bool = False
    grid_feature_dim: int = 8
    gs_features: int = 192
    max_grouping_scale: float = 2.0
    mlp_hidden_dim: int = 256
    mlp_num_layers: int = 4
    mlp_output_dim: int = 256
    log_test_images: bool = False


@dataclass
class Config:
    """Configuration parameters for the training process."""

    num_train_iters: int = 10000
    sample_pixels_per_image: int = 256
    batch_size: int = 8
    model: ModelConfig = field(default_factory=ModelConfig)
