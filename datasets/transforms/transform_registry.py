# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from typing import TypeVar

from .base_transform import BaseTransform

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
        raise ValueError(
            f"Transform name '{cls.name()}' is already registered. You must use unique names for each transform."
        )

    REGISTERED_TRANSFORMS[cls.name()] = cls

    return cls
