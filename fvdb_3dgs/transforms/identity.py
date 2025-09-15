# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Any

from ..sfm_scene import SfmScene
from .base_transform import BaseTransform, transform


@transform
class Identity(BaseTransform):
    """
    An identity transform that does not modify the input scene or cache.
    """

    version = "1.0.0"

    def __init__(
        self,
    ):
        """
        Create a new Identity transform instance.
        """
        super().__init__()

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Perform the identity transform on the input scene and cache.

        Args:
            input_scene (SfmScene): The input scene containing images to be downsampled.

        Returns:
            output_scene (SfmScene): The input scene, unchanged.
        """

        return input_scene

    @staticmethod
    def name() -> str:
        """
        Return the name of the Identity transform.

        Returns:
            str: The name of the Identity transform.
        """
        return "Identity"

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the Identity transform for serialization.
        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {
            "name": self.name(),
            "version": self.version,
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "Identity":
        """
        Create a Identity transform from a state dictionary.

        Args:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.

        Returns:
            Identity: An instance of the Identity transform.
        """
        if state_dict["name"] != "Identity":
            raise ValueError(f"Expected state_dict with name 'Identity', got {state_dict['name']} instead.")

        return Identity()
