# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any

import numpy as np

from ..sfm_scene import SfmScene
from .base_transform import BaseTransform, transform


@transform
class FilterImagesWithLowPoints(BaseTransform):
    """
    A transform that filters out images from a SfM scene that have fewer than a specified minimum number of visible points.

    Any images that have a number of visible points less than or equal to `min_num_points`  will be removed from the scene.
    """

    version = "1.0.0"

    def __init__(
        self,
        min_num_points: int = 0,
    ):
        """
        Create a FilterImagesWithLowPoints transform which removes images from the scene which have fewer than or equal to `min_num_points` visible points.

        Args:
            min_num_points (int): The minimum number of visible points required to keep an image in the scene.
                Images with fewer or equal points will be removed.
        """
        super().__init__()
        self._min_num_points = min_num_points
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Perform the filtering on the input scene.

        Args:
            input_scene (SfmScene): The input scene.

        Returns:
            output_scene (SfmScene): A new SfmScene containing only images which have more than `min_num_points` visible points.
        """
        image_mask = np.array(
            [img_meta.point_indices.shape[0] > self._min_num_points for img_meta in input_scene.images], dtype=bool
        )

        return input_scene.filter_images(image_mask)

    @property
    def min_num_points(self) -> int:
        """
        Get the minimum number of points required to keep an image in the scene.

        Returns:
            int: The minimum number of points required to keep an image in the scene.
        """
        return self._min_num_points

    @staticmethod
    def name() -> str:
        """
        Return the name of the FilterImagesWithLowPoints transform.

        Returns:
            str: The name of the FilterImagesWithLowPoints transform.
        """
        return "FilterImagesWithLowPoints"

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the FilterImagesWithLowPoints transform for serialization.
        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {
            "name": self.name(),
            "version": self.version,
            "min_num_points": self._min_num_points,
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "FilterImagesWithLowPoints":
        """
        Create a FilterImagesWithLowPoints transform from a state dictionary.

        Args:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.

        Returns:
            FilterImagesWithLowPoints: An instance of the FilterImagesWithLowPoints transform.
        """
        if state_dict["name"] != "FilterImagesWithLowPoints":
            raise ValueError(
                f"Expected state_dict with name 'FilterImagesWithLowPoints', got {state_dict['name']} instead."
            )

        return FilterImagesWithLowPoints(
            min_num_points=state_dict["min_num_points"],
        )
