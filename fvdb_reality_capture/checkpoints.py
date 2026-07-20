# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Method-neutral, versioned training-checkpoint container I/O."""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

TRAINING_CHECKPOINT_SCHEMA = "fvdb_reality_capture.training_checkpoint"
TRAINING_CHECKPOINT_SCHEMA_VERSION = 1


class TrainingCheckpointError(ValueError):
    """Raised when a training checkpoint is invalid or unsupported."""


class TrainingCheckpointVersionError(TrainingCheckpointError):
    """Raised when no reader exists for a checkpoint schema version."""


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Validated method-neutral checkpoint metadata and method-owned state."""

    method: str
    method_version: str
    state: dict[str, Any]
    schema_version: int = TRAINING_CHECKPOINT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized checkpoint container."""
        return {
            "schema": TRAINING_CHECKPOINT_SCHEMA,
            "schema_version": self.schema_version,
            "method": self.method,
            "method_version": self.method_version,
            "state": self.state,
        }


def create_training_checkpoint(
    method: str,
    state: dict[str, Any],
    *,
    method_version: str,
) -> TrainingCheckpoint:
    """Create a current-version checkpoint container around method-owned state."""
    if not isinstance(method, str) or not method:
        raise TrainingCheckpointError("Checkpoint method must be a non-empty string")
    if not isinstance(state, dict):
        raise TrainingCheckpointError(f"Checkpoint state must be a dictionary, got {type(state).__name__}")
    if not isinstance(method_version, str) or not method_version:
        raise TrainingCheckpointError("Checkpoint method_version must be a non-empty string")
    return TrainingCheckpoint(method=method, method_version=method_version, state=state)


def _read_training_checkpoint_v1(root: Mapping[str, Any]) -> TrainingCheckpoint:
    method = root.get("method")
    method_version = root.get("method_version")
    state = root.get("state")
    if not isinstance(method, str) or not method:
        raise TrainingCheckpointError("Checkpoint method must be a non-empty string")
    if not isinstance(method_version, str) or not method_version:
        raise TrainingCheckpointError("Checkpoint method_version must be a non-empty string")
    if not isinstance(state, dict):
        raise TrainingCheckpointError(f"Checkpoint state must be a dictionary, got {type(state).__name__}")
    return TrainingCheckpoint(method=method, method_version=method_version, state=state)


_CHECKPOINT_READERS: dict[int, Callable[[Mapping[str, Any]], TrainingCheckpoint]] = {
    1: _read_training_checkpoint_v1,
}
SUPPORTED_TRAINING_CHECKPOINT_SCHEMA_VERSIONS = tuple(sorted(_CHECKPOINT_READERS))

LegacyCheckpointAdapter: TypeAlias = Callable[[Mapping[str, Any]], TrainingCheckpoint | None]
_LEGACY_CHECKPOINT_ADAPTERS: list[LegacyCheckpointAdapter] = []


def register_legacy_checkpoint_adapter(adapter: LegacyCheckpointAdapter) -> None:
    """Register a method-owned reader for a pre-container checkpoint format."""
    if adapter in _LEGACY_CHECKPOINT_ADAPTERS:
        raise ValueError("Legacy checkpoint adapter is already registered")
    _LEGACY_CHECKPOINT_ADAPTERS.append(adapter)


def parse_training_checkpoint(root: Any) -> TrainingCheckpoint:
    """Validate an container or ask registered method-owned legacy adapters."""
    if not isinstance(root, dict):
        raise TrainingCheckpointError(f"Checkpoint root must be a dictionary, got {type(root).__name__}")

    if root.get("schema") == TRAINING_CHECKPOINT_SCHEMA:
        schema_version = root.get("schema_version")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
            raise TrainingCheckpointVersionError(
                f"Training checkpoint has an invalid or missing schema_version: {schema_version!r}"
            )
        reader = _CHECKPOINT_READERS.get(schema_version)
        if reader is None:
            supported = ", ".join(str(version) for version in SUPPORTED_TRAINING_CHECKPOINT_SCHEMA_VERSIONS)
            detail = (
                "The checkpoint was written by a newer fvdb-reality-capture release."
                if schema_version > TRAINING_CHECKPOINT_SCHEMA_VERSION
                else "This release has no reader for that older checkpoint version."
            )
            raise TrainingCheckpointVersionError(
                f"Unsupported training checkpoint schema_version {schema_version}; supported versions: "
                f"{supported}. {detail}"
            )
        return reader(root)

    if "schema" in root:
        raise TrainingCheckpointError(
            f"Unsupported checkpoint schema {root.get('schema')!r}; expected {TRAINING_CHECKPOINT_SCHEMA!r}"
        )

    for adapter in _LEGACY_CHECKPOINT_ADAPTERS:
        checkpoint = adapter(root)
        if checkpoint is not None:
            return checkpoint

    raise TrainingCheckpointError(
        "Checkpoint is not a versioned fvdb-reality-capture training checkpoint and has no supported legacy format"
    )


def load_training_checkpoint(
    path: str | pathlib.Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_method: str | None = None,
) -> TrainingCheckpoint:
    """Load and validate a training checkpoint from disk."""
    checkpoint_path = pathlib.Path(path)
    if checkpoint_path.is_dir():
        raise TrainingCheckpointError(f"Training checkpoint path is a directory, not a checkpoint file: {path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Training checkpoint does not exist: {checkpoint_path}")
    root = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    checkpoint = parse_training_checkpoint(root)
    if expected_method is not None and checkpoint.method != expected_method:
        raise TrainingCheckpointError(f"Checkpoint method is {checkpoint.method!r}; expected {expected_method!r}")
    return checkpoint


__all__ = [
    "LegacyCheckpointAdapter",
    "SUPPORTED_TRAINING_CHECKPOINT_SCHEMA_VERSIONS",
    "TRAINING_CHECKPOINT_SCHEMA",
    "TRAINING_CHECKPOINT_SCHEMA_VERSION",
    "TrainingCheckpoint",
    "TrainingCheckpointError",
    "TrainingCheckpointVersionError",
    "create_training_checkpoint",
    "load_training_checkpoint",
    "parse_training_checkpoint",
    "register_legacy_checkpoint_adapter",
]
