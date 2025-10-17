# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any

import numpy as np

from fvdb_reality_capture.sfm_scene import SfmScene

from .base_transform import BaseTransform, transform


@transform
class FilterImagesWithLowPoints(BaseTransform):
    """
    A :class:`~base_transform.BaseTransform` which filters out posed images from
    an :class:`~fvdb_reality_capture.sfm_scene.SfmScene` that have fewer than a specified minimum number
    of visible points.

    Any images that have a number of visible points less than or equal to ``min_num_points``  will
    be removed from the scene.

    .. note::
        If the input :class:`~fvdb_reality_capture.sfm_scene.SfmScene` does not have point indices for its posed images
        (i.e. it has :obj:`~fvdb_reality_capture.sfm_scene.SfmScene.has_visible_point_indices` set to ``False``), then
        this transform is a no-op.

    Example usage:

    .. code-block:: python

        # Example usage:
        from fvdb_reality_capture import transforms
        from fvdb_reality_capture.sfm_scene import SfmScene

        # Create a transform to filter out images with 50 or fewer visible points.
        scene_transform = transforms.FilterImagesWithLowPoints(min_num_points=50)

        input_scene: SfmScene = ...  # Load or create an SfmScene

        # The transformed scene will only contain posed images with more than 50 visible points.
        transformed_scene: SfmScene = scene_transform(input_scene)

    """

    version = "1.0.0"

    def __init__(
        self,
        min_num_points: int = 0,
    ):
        """
        Create a new :class:`FilterImagesWithLowPoints` transform which removes posed images from the scene which have
        fewer than or equal to ``min_num_points`` visible points.

        Args:
            min_num_points (int): The minimum number of visible points required to keep a posed image in the scene.
                Posed images with fewer or equal visible points will be removed.
        """
        super().__init__()
        self._min_num_points = min_num_points
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Return a new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` containing only posed images which have more
        than ``min_num_points`` visible points.

        .. note::
            If the input :class:`~fvdb_reality_capture.sfm_scene.SfmScene` does not have point indices for its
            posed images (i.e. :obj:`fvdb_reality_capture.sfm_scene.SfmScene.has_visible_point_indices` is ``False``),
            then this transform is a no-op.


        Args:
            input_scene (SfmScene): The input scene.

        Returns:
            output_scene (SfmScene): A new SfmScene containing only posed images which have more than ``min_num_points``
                visible points. If the input scene does not have point indices for its posed images,
                the input scene is returned unmodified.
        """
        if not input_scene.has_visible_point_indices:
            self._logger.info(
                "Input scene does not have point indices for its posed images. Returning the input scene unmodified."
            )
            return input_scene
        image_mask = np.array(
            [
                img_meta.point_indices.shape[0] > self._min_num_points
                for img_meta in input_scene.images
                if img_meta.point_indices is not None
            ],
            dtype=bool,
        )

        return input_scene.filter_images(image_mask)

    @property
    def min_num_points(self) -> int:
        """
        Get the minimum number of points required to keep a posed image in the scene when applying this transform.

        Returns:
            min_num_points (int): The minimum number of points required to keep a posed image in the scene when applying
                this transform.
        """
        return self._min_num_points

    @staticmethod
    def name() -> str:
        """
        Return the name of the :class:`FilterImagesWithLowPoints` transform. **i.e.** ``"FilterImagesWithLowPoints"``.

        Returns:
            str: The name of the :class:`FilterImagesWithLowPoints` transform. **i.e.** ``"FilterImagesWithLowPoints"``.
        """
        return "FilterImagesWithLowPoints"

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the :class:`FilterImagesWithLowPoints` transform for serialization.

        You can use this state dictionary to recreate the transform using :meth:`from_state_dict`.

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
        Create a :class:`FilterImagesWithLowPoints` transform from a state dictionary generated with :meth:`state_dict`.

        Args:
            state_dict (dict): The state dictionary for the transform.

        Returns:
            transform (FilterImagesWithLowPoints): An instance of the :class:`FilterImagesWithLowPoints` transform.
        """
        if state_dict["name"] != "FilterImagesWithLowPoints":
            raise ValueError(
                f"Expected state_dict with name 'FilterImagesWithLowPoints', got {state_dict['name']} instead."
            )

        return FilterImagesWithLowPoints(
            min_num_points=state_dict["min_num_points"],
        )
