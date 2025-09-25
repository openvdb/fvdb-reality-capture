# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any, Sequence

import numpy as np

from ..sfm_scene import SfmCache, SfmScene
from .base_transform import BaseTransform, transform


@transform
class PercentileFilterPoints(BaseTransform):
    """
    A transform that filters points in an SfmScene based on percentile bounds for x, y, and z coordinates.

    This transform creates a new SfmScene with points that fall within the specified percentile bounds
    along each axis.

    _e.g._ if percentile_min is (0, 0, 0) and percentile_max is (100, 100, 100),
        all points will be included in the output scene.

    _e.g._ if percentile_min is (10, 20, 30) and percentile_max is (90, 80, 70),
        only points with x-coordinates in the 10th to 90th percentile,
        y-coordinates in the 20th to 80th percentile, and z-coordinates
        in the 30th to 70th percentile will be included in the output scene.
    """

    version = "1.0.0"

    def __init__(
        self, percentile_min: Sequence[float | int] | np.ndarray, percentile_max: Sequence[float | int] | np.ndarray
    ):
        """
        Initialize the PercentileFilterPoints transform.

        Args:
            percentile_min (Sequence[float | int] | np.ndarray): Tuple of minimum percentiles (from 0 to 100) for x, y, z coordinates
                or None to use (0, 0, 0) (default: None)
            percentile_max (Sequence[float | int] | np.ndarray): Tuple of maximum percentiles (from 0 to 100) for x, y, z coordinates
                or None to use (100, 100, 100) (default: None)
        """
        super().__init__()
        if len(percentile_min) != 3:
            raise ValueError(f"percentile_min must be a sequence of length 3. Got {percentile_min} instead.")
        if len(percentile_max) != 3:
            raise ValueError(f"percentile_max must be a sequence of length 3. Got {percentile_max} instead.")
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._percentile_min = np.asarray(percentile_min).astype(np.float32)
        self._percentile_max = np.asarray(percentile_max).astype(np.float32)

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Apply the percentile filtering transform to the input scene and cache.

        Args:
            input_scene (SfmScene): The input scene containing points to be filtered.

        Returns:
            output_scene (SfmScene): A new SfmScene with points filtered based on the specified percentile bounds.
        """
        self._logger.info(
            f"Filtering points based on percentiles: min={self._percentile_min}, max={self._percentile_max}"
        )
        percentile_min = np.clip(self._percentile_min, 0.0, 100.0)
        percentile_max = np.clip(self._percentile_max, 0.0, 100.0)

        if np.all(percentile_min <= 0) and np.any(percentile_max >= 100):
            self._logger.info("No points will be filtered out, returning the input scene unchanged.")
            return input_scene

        points = input_scene.points
        lower_boundx = np.percentile(points[:, 0], percentile_min[0])
        upper_boundx = np.percentile(points[:, 0], percentile_max[0])

        lower_boundy = np.percentile(points[:, 1], percentile_min[1])
        upper_boundy = np.percentile(points[:, 1], percentile_max[1])

        lower_boundz = np.percentile(points[:, 2], percentile_min[2])
        upper_boundz = np.percentile(points[:, 2], percentile_max[2])

        good_map = np.logical_and.reduce(
            [
                points[:, 0] > lower_boundx,
                points[:, 0] < upper_boundx,
                points[:, 1] > lower_boundy,
                points[:, 1] < upper_boundy,
                points[:, 2] > lower_boundz,
                points[:, 2] < upper_boundz,
            ]
        )

        if np.sum(good_map) == 0:
            raise ValueError(
                f"No points found in the specified percentile range: "
                f"min={percentile_min}, max={percentile_max}. "
                "Please adjust the percentile values."
            )

        output_scene = input_scene.filter_points(good_map)

        # Note: The input_cache is returned unchanged as this transform does not modify the cache.
        return output_scene

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the PercentileFilterPoints transform for serialization.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {
            "name": self.name(),
            "version": self.version,
            "percentile_min": self._percentile_min.tolist(),
            "percentile_max": self._percentile_max.tolist(),
        }

    @staticmethod
    def name() -> str:
        """
        Return the name of the PercentileFilterPoints transform.

        Returns:
            str: The name of the PercentileFilterPoints transform.
        """
        return "PercentileFilterPoints"

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "PercentileFilterPoints":
        """
        Create a PercentileFilterPoints transform from a state dictionary.

        Args:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.

        Returns:
            PercentileFilterPoints: An instance of the PercentileFilterPoints transform.
        """
        if state_dict["name"] != "PercentileFilterPoints":
            raise ValueError(
                f"Expected state_dict with name 'PercentileFilterPoints', got {state_dict['name']} instead."
            )
        if "percentile_min" not in state_dict or "percentile_max" not in state_dict:
            raise ValueError("State dictionary must contain 'percentile_min' and 'percentile_max' keys.")

        return PercentileFilterPoints(
            percentile_min=np.asarray(state_dict["percentile_min"]),
            percentile_max=np.asarray(state_dict["percentile_max"]),
        )
