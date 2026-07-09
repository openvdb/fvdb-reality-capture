# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Gaussian reconstruction checkpoint identity and compatibility adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fvdb_reality_capture.checkpoints import (
    TrainingCheckpoint,
    TrainingCheckpointError,
    register_legacy_checkpoint_adapter,
)

GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD = "radiance_fields.gaussian_splat"

# Single source of truth for the Gaussian reconstruction checkpoint format version. The runner's
# ``state_dict``/``from_state_dict`` and the disk writer's envelope ``method_version`` both derive from
# this constant so the two can never drift.
GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD_VERSION = "0.1.0"


def _adapt_flat_gaussian_checkpoint(
    root: Mapping[str, Any],
) -> TrainingCheckpoint | None:
    """Recognize the released pre-envelope Gaussian checkpoint format."""
    if root.get("magic") != "GaussianSplattingCheckpoint":
        return None
    method_version = root.get("version")
    if not isinstance(method_version, str) or not method_version:
        raise TrainingCheckpointError("Legacy Gaussian checkpoint has no valid version")
    return TrainingCheckpoint(
        method=GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD,
        method_version=method_version,
        state=dict(root),
    )


register_legacy_checkpoint_adapter(_adapt_flat_gaussian_checkpoint)


__all__ = [
    "GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD",
    "GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD_VERSION",
]
