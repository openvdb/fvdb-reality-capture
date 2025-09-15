# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from ..sfm_scene import SfmScene

# Keeps track of names of registered transforms and their classes.
REGISTERED_TRANSFORMS = {}


DerivedTransform = TypeVar("DerivedTransform", bound=type)


def transform(cls: DerivedTransform) -> DerivedTransform:
    """
    Decorator to register a transform class.

    Args:
        cls: The transform class to register.

    Returns:
        cls: The registered transform class.
    """
    if not issubclass(cls, BaseTransform):
        raise TypeError(f"Transform {cls} must inherit from BaseTransform.")

    if cls.name() in REGISTERED_TRANSFORMS:
        del REGISTERED_TRANSFORMS[cls.name()]

    REGISTERED_TRANSFORMS[cls.name()] = cls

    return cls


class BaseTransform(ABC):
    """Base class for all transforms."""

    def __init__(self, *args: Any, **kwds: Any):
        pass

    @abstractmethod
    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Apply the transform to the data.

        Args:
            input_scene (SfmScene): The input scene to transform.

        Returns:
            output_scene (SfmScene): The transformed scene.
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
        StateDictType = REGISTERED_TRANSFORMS.get(state_dict["name"], None)
        if StateDictType is None:
            raise ValueError(
                f"Transform '{state_dict['name']}' is not registered. Transform classes must be registered "
                f"with the `transform` decorator which will be called when the transform is defined. "
                f"Ensure the transform class uses the `transform` decorator and was imported before calling from_state_dict."
            )
        return StateDictType.from_state_dict(state_dict)

    def __repr__(self):
        return self.__class__.__name__
