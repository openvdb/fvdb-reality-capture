# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any

import numpy as np
from fvdb.types import NumericMaxRank1, to_Vec3f

from fvdb_reality_capture.sfm_scene import SfmScene

from .base_transform import BaseTransform, transform


@transform
class PercentileFilterPoints(BaseTransform):
    """
    A :class:`~base_transform.BaseTransform` that filters points in an
    :class:`~fvdb_reality_capture.sfm_scene.SfmScene` based on percentile bounds for x, y, and z coordinates.

    When applied to an input scene, this transform returns a new :class:`~fvdb_reality_capture.sfm_scene.SfmScene`
    with points that fall within the specified percentile bounds of the input scene's points along each axis.

    *e.g.* If percentile_min is ``(0, 0, 0)`` and percentile_max is ``(100, 100, 100)``,
    all points will be included in the output scene.

    *e.g.* If percentile_min is ``(10, 20, 30)`` and percentile_max is ``(90, 80, 70)``,
    only points with x-coordinates in the 10th to 90th percentile,
    y-coordinates in the 20th to 80th percentile, and z-coordinates
    in the 30th to 70th percentile will be included in the output scene.

    Example usage:

    .. code-block:: python

        from fvdb_reality_capture.transforms import PercentileFilterPoints
        from fvdb_reality_capture.sfm_scene import SfmScene

        # Create a PercentileFilterPoints transform to filter points between the 10th and 90th percentiles
        transform = PercentileFilterPoints(percentile_min=(10, 10, 10), percentile_max=(90, 90, 90))

        # Apply the transform to an SfmScene
        input_scene: SfmScene = ...
        output_scene: SfmScene = transform(input_scene)

    """

    version = "1.0.0"

    def __init__(self, percentile_min: NumericMaxRank1, percentile_max: NumericMaxRank1):
        """
        Create a new :class:`PercentileFilterPoints` transform which filters points in an
        :class:`~fvdb_reality_capture.sfm_scene.SfmScene` based on percentile bounds for x, y, and z coordinates.

        Args:
            percentile_min (NumericMaxRank1): Tuple of minimum percentiles (from 0 to 100) for x, y, z coordinates
                or None to use (0, 0, 0) (default: None)
            percentile_max (NumericMaxRank1): Tuple of maximum percentiles (from 0 to 100) for x, y, z coordinates
                or None to use (100, 100, 100) (default: None)
        """
        super().__init__()
        percentile_min = to_Vec3f(percentile_min).numpy()
        percentile_max = to_Vec3f(percentile_max).numpy()

        if np.any(percentile_min < 0) or np.any(percentile_min > 100):
            raise ValueError(f"percentile_min must be between 0 and 100. Got {percentile_min} instead.")
        if np.any(percentile_max < 0) or np.any(percentile_max > 100):
            raise ValueError(f"percentile_max must be between 0 and 100. Got {percentile_max} instead.")

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._percentile_min = np.asarray(percentile_min).astype(np.float32)
        self._percentile_max = np.asarray(percentile_max).astype(np.float32)

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Return a new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` with points filtered based on the specified
        percentile bounds.

        Args:
            input_scene (SfmScene): The input :class:`~fvdb_reality_capture.sfm_scene.SfmScene` containing points
                to be filtered.

        Returns:
            output_scene (SfmScene): A new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` with points filtered
                based on the specified percentile bounds.
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
        Return the state of the :class:`PercentileFilterPoints` transform for serialization.

        You can use this state dictionary to recreate the transform using :meth:`from_state_dict`.

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
        Return the name of the :class:`PercentileFilterPoints` transform. **i.e.** ``"PercentileFilterPoints"``.

        Returns:
            str: The name of the :class:`PercentileFilterPoints` transform. **i.e.** ``"PercentileFilterPoints"``.
        """
        return "PercentileFilterPoints"

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "PercentileFilterPoints":
        """
        Create a :class:`PercentileFilterPoints` transform from a state dictionary generated with :meth:`state_dict`.

        Args:
            state_dict (dict): The state dictionary for the transform.

        Returns:
            transform (PercentileFilterPoints): An instance of the :class:`PercentileFilterPoints` transform.
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
