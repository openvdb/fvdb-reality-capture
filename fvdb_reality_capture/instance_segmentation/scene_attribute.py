# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Scene attribute containing GARfVDB mask-supervision artifacts."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import torch

from fvdb_reality_capture.sfm_scene import SceneAttribute, scene_attribute

GARFVDB_MASK_ATTRIBUTE_NAME = "garfvdb_masks"
GARFVDB_MASK_ATTRIBUTE_SCHEMA_VERSION = 1
GARFVDB_MASK_DATA_SCHEMA_VERSION = 1


@scene_attribute
class GARfVDBMaskAttribute(SceneAttribute):
    """Per-image GARfVDB supervision produced by a mask-generation transform.

    The attribute owns one path per image. Each file stores the mask IDs, mask
    sampling CDF, and scene-unit grouping scales needed by GARfVDB training.
    Other segmentation methods can attach their own attribute types and data
    contracts without depending on this GARfVDB-specific representation.
    """

    def __init__(self, paths: list[str | pathlib.Path], provenance: dict[str, Any] | None = None) -> None:
        self._paths = [str(pathlib.Path(path).absolute()) for path in paths]
        self._provenance = dict(provenance or {})

    @property
    def paths(self) -> list[str]:
        """Return a copy of the ordered per-image artifact paths."""
        return list(self._paths)

    @property
    def provenance(self) -> dict[str, Any]:
        """Return generation metadata such as the Gaussian means hash and SAM2 settings."""
        return dict(self._provenance)

    @staticmethod
    def type_name() -> str:
        return "GARfVDBMaskAttribute"

    def validate(self, attr_name: str, num_points: int, num_images: int, camera_ids: set[int]) -> None:
        if len(self._paths) != num_images:
            raise ValueError(
                f"Attribute {attr_name!r} has {len(self._paths)} mask artifacts but the scene has {num_images} images."
            )

    def on_filter_images(self, mask: np.ndarray) -> "GARfVDBMaskAttribute":
        return GARfVDBMaskAttribute(
            [path for path, keep in zip(self._paths, mask) if keep],
            provenance=self._provenance,
        )

    def on_select_images(self, indices: np.ndarray) -> "GARfVDBMaskAttribute":
        return GARfVDBMaskAttribute(
            [self._paths[int(index)] for index in indices],
            provenance=self._provenance,
        )

    def on_downsample_images(
        self,
        attr_name: str,
        downsample_factor: int,
        output_cache: Any,
    ) -> "GARfVDBMaskAttribute":
        raise ValueError(
            f"Cannot downsample images after generating {attr_name!r}. "
            "Place GARfVDB mask generation after all image transforms."
        )

    def on_crop_scene(self, attr_name: str, bbox: np.ndarray, output_cache: Any) -> "GARfVDBMaskAttribute":
        raise ValueError(
            f"Cannot crop the scene after generating {attr_name!r}. "
            "Place GARfVDB mask generation after all crop transforms."
        )

    def on_spatial_transform(self, matrix: np.ndarray) -> "GARfVDBMaskAttribute":
        raise ValueError(
            "Cannot spatially transform a scene after generating GARfVDB masks because their scales are in scene "
            "units. Place GARfVDB mask generation after scene alignment and normalization."
        )

    def load(self, index: int) -> dict[str, torch.Tensor]:
        """Load and validate the GARfVDB mask data for one scene image."""
        try:
            path = pathlib.Path(self._paths[index])
        except IndexError as exc:
            raise IndexError(f"GARfVDB mask index {index} is out of range for {len(self._paths)} images") from exc
        if not path.is_file():
            raise FileNotFoundError(f"GARfVDB mask artifact does not exist: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict):
            raise ValueError(f"GARfVDB mask artifact must contain a dictionary: {path}")
        if data.get("schema_version") != GARFVDB_MASK_DATA_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported GARfVDB mask schema_version {data.get('schema_version')!r} in {path}; "
                f"expected {GARFVDB_MASK_DATA_SCHEMA_VERSION}."
            )
        # mask_cdf is not stored on disk; it is recomputed from pixel_to_mask_id at load time by the dataset.
        required = {"scales", "pixel_to_mask_id"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"GARfVDB mask artifact {path} is missing fields: {sorted(missing)}")
        if any(not isinstance(data[key], torch.Tensor) for key in required):
            raise TypeError(f"GARfVDB mask artifact fields must be tensors: {path}")
        return {
            "scales": data["scales"],
            "pixel_to_mask_id": data["pixel_to_mask_id"],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GARFVDB_MASK_ATTRIBUTE_SCHEMA_VERSION,
            "paths": self._paths,
            "provenance": self._provenance,
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "GARfVDBMaskAttribute":
        if state_dict.get("schema_version") != GARFVDB_MASK_ATTRIBUTE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported GARfVDBMaskAttribute schema_version {state_dict.get('schema_version')!r}; "
                f"expected {GARFVDB_MASK_ATTRIBUTE_SCHEMA_VERSION}."
            )
        return GARfVDBMaskAttribute(
            paths=state_dict["paths"],
            provenance=state_dict.get("provenance", {}),
        )


__all__ = [
    "GARFVDB_MASK_ATTRIBUTE_NAME",
    "GARFVDB_MASK_ATTRIBUTE_SCHEMA_VERSION",
    "GARFVDB_MASK_DATA_SCHEMA_VERSION",
    "GARfVDBMaskAttribute",
]
