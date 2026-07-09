# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Shared I/O helpers for loading Gaussian splat reconstructions."""

from __future__ import annotations

import pathlib

from fvdb import GaussianSplat3d
from fvdb.types import DeviceIdentifier

from fvdb_reality_capture.checkpoints import load_training_checkpoint

from .checkpoint import GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD
from .gaussian_splat_reconstruction import GaussianSplatReconstruction


def load_splats_from_file(path: pathlib.Path, device: DeviceIdentifier) -> tuple[GaussianSplat3d, dict]:
    """Load a Gaussian splat model and its metadata from a PLY or reconstruction checkpoint.

    The metadata may contain camera information (if it was a PLY saved during training).

    Args:
        path (pathlib.Path): Path to the PLY or checkpoint file.
        device (DeviceIdentifier): Device to load the model onto.

    Returns:
        model (GaussianSplat3d): The loaded Gaussian Splat model.
        metadata (dict): The metadata associated with the model.
    """
    if path.suffix.lower() == ".ply":
        model, metadata = GaussianSplat3d.from_ply(path, device)
    elif path.suffix.lower() in (".pt", ".pth"):
        checkpoint = load_training_checkpoint(
            path,
            map_location=device,
            expected_method=GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD,
        )
        runner = GaussianSplatReconstruction.from_state_dict(checkpoint.state, device=device)
        model = runner.model
        metadata = runner.reconstruction_metadata
    else:
        raise ValueError("Input path must end in .ply, .pt, or .pth")

    return model, metadata


__all__ = ["load_splats_from_file"]
