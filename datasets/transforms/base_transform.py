# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from abc import ABC, abstractmethod
from typing import Any

from ..image_dataset_cache import ImageDatasetCache
from ..sfm_scene import SfmScene


class BaseTransform(ABC):
    """Base class for all transforms."""

    def __init__(self, *args: Any, **kwds: Any):
        pass

    @abstractmethod
    def __call__(self, input_scene: SfmScene, input_cache: ImageDatasetCache) -> tuple[SfmScene, ImageDatasetCache]:
        """
        Apply the transform to the data.

        Args:
            input_scene: The input scene to transform.
            input_cache: The input cache to use to save intermediate results.

        Returns:
            output_scene: The transformed scene.
            output_cache: The cache to use for further transforms.
        """
        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        """
        Return the name of the transform.

        Returns:
            str: The name of the transform.
        """
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the transform for serialization.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "BaseTransform":
        """
        Create a transform from a state dictionary.

        Args:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.

        Returns:
            BaseTransform: An instance of the transform.
        """
        pass

    def __repr__(self):
        return self.__class__.__name__
