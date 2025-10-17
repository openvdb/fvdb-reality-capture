# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Any

from fvdb_reality_capture.sfm_scene import SfmScene

from .base_transform import BaseTransform, transform


@transform
class Identity(BaseTransform):
    """
    A :class:`~base_transform.BaseTransform` that performs the identity transform on an
    :class:`~fvdb_reality_capture.sfm_scene.SfmScene`. This transform returns the input scene unchanged.
    It can be useful as a placeholder or default transform in a processing pipeline.

    Example usage:

    .. code-block:: python

        # Example usage:
        from fvdb_reality_capture import transforms
        from fvdb_reality_capture.sfm_scene import SfmScene

        # Use Identity as a default parameter value
        def append_normalize(transform: transforms.BaseTransform = transforms.Identity()):
            return transforms.Compose(
                transform,
                transforms.NormalizeScene("pca"),
            )

        # Use Identity to return a no-op for later use
        def get_transform(condition: bool) -> transforms.BaseTransform:
            if condition:
                return transforms.DownsampleImages(2)
            else:
                # Still return a valid transform that is a no-op
                return transforms.Identity()

        get_transform(False)  # Returns an Identity transform
        get_transform(True)   # Returns a DownsampleImages transform
    """

    version = "1.0.0"

    def __init__(
        self,
    ):
        """
        Create a new :class:`Identity` transform representing a No-Op.
        """
        super().__init__()

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Return the input  :class:`~fvdb_reality_capture.sfm_scene.SfmScene` unchanged.

        Args:
            input_scene (SfmScene): The input scene.

        Returns:
            output_scene (SfmScene): The input scene, unchanged.
        """

        return input_scene

    @staticmethod
    def name() -> str:
        """
        Return the name of the :class:`Identity` transform. **i.e.** ``"Identity"``.

        Returns:
            str: The name of the :class:`Identity` transform. **i.e.** ``"Identity"``.
        """
        return "Identity"

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the :class:`Identity` transform for serialization.

        You can use this state dictionary to recreate the transform using :meth:`from_state_dict`.

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
        Create a :class:`Identity` transform from a state dictionary created with :meth:`state_dict`.

        Args:
            state_dict (dict): The state dictionary for the transform.

        Returns:
            transform (Identity): An instance of the :class:`Identity` transform.
        """
        if state_dict["name"] != "Identity":
            raise ValueError(f"Expected state_dict with name 'Identity', got {state_dict['name']} instead.")

        return Identity()
