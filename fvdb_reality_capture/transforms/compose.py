# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Any

from ..sfm_scene import SfmScene
from .base_transform import REGISTERED_TRANSFORMS, BaseTransform, transform


@transform
class Compose(BaseTransform):
    """
    A transform that composes multiple transforms together.
    This is useful for applying a sequence of transformations to the data.

    The transforms are applied in the order they are provided, allowing for complex data processing pipelines.

    Attributes:
        transforms (list[BaseTransform]): A list of transform instances to compose.
    """

    version = "1.0.0"

    def __init__(self, *transforms):
        """
        Initialize the Compose transform with a list of transforms.

        Args:
            *transforms: A variable number of transform instances to compose.
        """
        super().__init__()
        self.transforms = transforms
        for transform in self.transforms:
            if not isinstance(transform, BaseTransform):
                raise TypeError(f"Expected a BaseTransform instance, got {type(transform)} instead.")

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        for transform in self.transforms:
            input_scene = transform(input_scene)
        return input_scene

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the Compose transform for serialization.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {
            "name": self.name(),
            "version": self.version,
            "transforms": [
                {"name": transform.name(), "state": transform.state_dict()} for transform in self.transforms
            ],
        }

    @staticmethod
    def name() -> str:
        """
        Return the name of the Compose transform.

        Returns:
            str: The name of the Compose transform.
        """
        return "Compose"

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "Compose":
        """
        Create a Compose transform from a state dictionary.

        Args:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.

        Returns:
            Compose: An instance of the Compose transform.
        """
        if state_dict["name"] != "Compose":
            raise ValueError(f"Expected state_dict with name 'Compose', got {state_dict['name']} instead.")
        if "transforms" not in state_dict:
            raise ValueError("State dictionary must contain 'transforms' key.")

        if not isinstance(state_dict["transforms"], list):
            raise TypeError(f"Expected 'transforms' to be a list, got {type(state_dict['transforms'])} instead.")

        transforms = []
        for transform_state in state_dict["transforms"]:
            if not isinstance(transform_state, dict):
                raise TypeError(f"Expected each transform state to be a dict, got {type(transform_state)} instead.")
            if "name" not in transform_state:
                raise ValueError("Each transform state must contain a 'name' key.")

            if "state" not in transform_state:
                raise ValueError("Each transform state must contain a 'state' key.")

            StateDictType = REGISTERED_TRANSFORMS.get(transform_state["name"], None)
            if StateDictType is None:
                raise ValueError(
                    f"Transform '{transform_state['name']}' is not registered. Transform classes must be registered "
                    f"with the `transform` decorator which will be called when the transform is defined. "
                    f"Ensure the transform class uses the `transform` decorator and was imported before calling from_state_dict."
                )
            transforms.append(StateDictType.from_state_dict(transform_state["state"]))
        return Compose(*transforms)
