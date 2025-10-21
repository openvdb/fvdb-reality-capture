# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any

import numpy as np

from fvdb_reality_capture.sfm_scene import SfmScene

from .base_transform import BaseTransform, transform


@transform
class TransformScene(BaseTransform):
    """
    A :class:`~base_transform.BaseTransform` which transforms an :class:`~fvdb_reality_capture.sfm_scene.SfmScene`
    using a transformation matrix.

    The transformation matrix is applied to the entire scene, including both points and camera poses.

    Example usage:

    .. code-block:: python

        from fvdb_reality_capture import transforms
        from fvdb_reality_capture.sfm_scene import SfmScene

        # Create a TransformScene to transform the scene using a transformation matrix
        transform = transforms.TransformScene(transformation_matrix=np.eye(4))

        # Apply the transform to an SfmScene
        input_scene: SfmScene = ...
        output_scene: SfmScene = transform(input_scene)

    """

    version = "1.0.0"

    def __init__(self, transformation_matrix: np.ndarray):
        """
        Create a new :class:`TransformScene` transform which transforms an
        :class:`~fvdb_reality_capture.sfm_scene.SfmScene` using the specified transformation matrix.

        Args:
            transformation_matrix (np.ndarray): The 4x4 transformation matrix to apply to the scene.
        """
        super().__init__()
        self._transformation_matrix = transformation_matrix
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Return a new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` which is the result of applying the
        transformation matrix to the input scene.

        The transformation matrix is applied to the entire scene, including both points and camera poses.

        Args:
            input_scene (SfmScene): Input :class:`~fvdb_reality_capture.sfm_scene.SfmScene` object containing
                camera and point data

        Returns:
            output_scene (SfmScene): A new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` after applying
                the transformation matrix.
        """
        self._logger.info(f"Transforming SfmScene with transformation matrix: {self._transformation_matrix}")

        return input_scene.apply_transformation_matrix(self._transformation_matrix)

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the :class:`TransformScene` transform for serialization.

        You can use this state dictionary to recreate the transform using :meth:`from_state_dict`.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {"name": self.name(), "version": self.version, "transformation_matrix": self._transformation_matrix}

    @staticmethod
    def name() -> str:
        """
        Return the name of the :class:`TransformScene` transform. **i.e.** ``"TransformScene"``.

        Returns:
            str: The name of the :class:`TransformScene` transform. **i.e.** ``"TransformScene"``.
        """
        return "TransformScene"

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "TransformScene":
        """
        Create a :class:`TransformScene` transform from a state dictionary generated with :meth:`state_dict`.

        Args:
            state_dict (dict): The state dictionary for the transform.

        Returns:
            transform (TransformScene): An instance of the :class:`TransformScene` transform.
        """
        if state_dict["name"] != "TransformScene":
            raise ValueError(f"Expected state_dict with name 'TransformScene', got {state_dict['name']} instead.")
        if "transformation_matrix" not in state_dict:
            raise ValueError("State dictionary must contain 'transformation_matrix' key.")

        transformation_matrix = state_dict["transformation_matrix"]
        return TransformScene(transformation_matrix)
