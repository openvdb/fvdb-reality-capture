# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Reusable configuration for standard Reality Capture scene transforms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .base_transform import BaseTransform
from .compose import Compose
from .crop_scene import CropScene, CropSceneToPoints
from .downsample_images import DownsampleImages
from .filter_images_with_low_points import FilterImagesWithLowPoints
from .percentile_filter_points import PercentileFilterPoints


@dataclass
class SceneTransformConfig:
    """Configure the standard filtering, image, and crop portion of a scene pipeline.

    Methods supply their own alignment transform and optional terminal transforms.
    This keeps common Reality Capture preprocessing ordered consistently without
    constraining method-specific transforms or scene attributes.
    """

    image_downsample_factor: int = 1
    """Factor by which to downsample images."""

    rescale_jpeg_quality: int = 95
    """JPEG quality used when writing downsampled images."""

    points_percentile_filter: float = 0.0
    """Percentile of point outliers removed independently from each bound."""

    crop_bbox: tuple[float, float, float, float, float, float] | None = None
    """Optional ``(xmin, ymin, zmin, xmax, ymax, zmax)`` scene-space crop."""

    crop_to_points: bool = False
    """Crop the scene bounds to its filtered point-cloud extent."""

    min_points_per_image: int = 5
    """Minimum visible 3D points required to retain an image."""

    def build_scene_transform(
        self,
        *,
        alignment_transform: BaseTransform,
        terminal_transforms: Sequence[BaseTransform] = (),
    ) -> Compose:
        """Build a pipeline with application-specific transforms appended last."""
        transforms: list[BaseTransform] = [
            alignment_transform,
            PercentileFilterPoints(
                percentile_min=np.full((3,), self.points_percentile_filter),
                percentile_max=np.full((3,), 100.0 - self.points_percentile_filter),
            ),
            DownsampleImages(
                image_downsample_factor=self.image_downsample_factor,
                rescaled_jpeg_quality=self.rescale_jpeg_quality,
            ),
            FilterImagesWithLowPoints(min_num_points=self.min_points_per_image),
        ]
        if self.crop_bbox is not None:
            transforms.append(CropScene(self.crop_bbox))
        if self.crop_to_points:
            transforms.append(CropSceneToPoints(margin=0.0))
        transforms.extend(terminal_transforms)
        return Compose(*transforms)


__all__ = ["SceneTransformConfig"]
