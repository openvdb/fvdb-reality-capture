# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
from typing import Literal

import torch
from fvdb import GaussianSplat3d
from fvdb.types import DeviceIdentifier

from fvdb_reality_capture.sfm_scene import SfmScene
from fvdb_reality_capture.training import GaussianSplatReconstruction

DatasetType = Literal["colmap", "simple_directory", "e57"]


def load_splats_from_file(path: pathlib.Path, device: DeviceIdentifier) -> tuple[GaussianSplat3d, dict]:
    """
    Load a PLY or a checkpoint file and metadata.
    The metadata may contain camera information (if it was a PLY saved during training).
    If so, we will add the camera views to the viewer.

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
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        runner = GaussianSplatReconstruction.from_state_dict(checkpoint, device=device)
        model = runner.model
        metadata = runner.optimization_metadata
    else:
        raise ValueError("Input path must end in .ply, .pt, or .pth")

    return model, metadata


def load_sfm_scene(path: pathlib.Path, dataset_type: DatasetType) -> SfmScene:
    """
    Load an SfM scene from the specified dataset path and type.

    Args:
        path (pathlib.Path): Path to the dataset folder.
        dataset_type (DatasetType): Type of the dataset. One of "colmap", "simple_directory", or "e57".

    Returns:
        SfmScene: The loaded SfM scene.
    """
    if dataset_type == "colmap":
        sfm_scene = SfmScene.from_colmap(path)
    elif dataset_type == "simple_directory":
        sfm_scene = SfmScene.from_simple_directory(path)
    elif dataset_type == "e57":
        sfm_scene = SfmScene.from_e57(path)
    else:
        raise ValueError(f"Unsupported dataset_type {dataset_type}")

    return sfm_scene
