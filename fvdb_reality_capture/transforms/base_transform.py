# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from fvdb_reality_capture.sfm_scene import SfmScene

# Keeps track of names of registered transforms and their classes.
REGISTERED_TRANSFORMS = {}


DerivedTransform = TypeVar("DerivedTransform", bound=type)


def transform(cls: DerivedTransform) -> DerivedTransform:
    """
    Decorator to register a transform class which inherits from :class:`BaseTransform`.

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
    """
    Base class for all transforms.

    Transforms are used to modify an :class:`~fvdb_reality_capture.sfm_scene.SfmScene` before it is used for
    reconstruction or other processing. They can be used to filter images, adjust camera parameters, or perform other
    modifications to the scene.

    Subclasses of :class:`BaseTransform` must implement the following methods:
    """

    @abstractmethod
    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Abstract method to apply the transform to the input scene and return the transformed scene.

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
        Abstract method to return the name of the transform.

        Returns:
            str: The name of the transform.
        """
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """
        Abstract method to return a dictionary containing information to serialize/deserialize the transform.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "BaseTransform":
        """
        Abstract method to create a transform from a state dictionary generated with :meth:`state_dict`.

        Args:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.

        Returns:
            transform (BaseTransform): An instance of the transform.
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
