# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
from typing import Literal

import torch
from fvdb import GaussianSplat3d
from fvdb.types import DeviceIdentifier, to_Mat33fBatch, to_Mat44fBatch, to_Vec2iBatch

from fvdb_reality_capture.radiance_fields import GaussianSplatReconstruction
from fvdb_reality_capture.sfm_scene import SfmScene
from fvdb_reality_capture.tools import export_splats_to_usdz

DatasetType = Literal["colmap", "simple_directory", "e57"]
NearFarUnits = Literal["absolute", "camera_extent", "median_depth"]


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
        metadata = runner.reconstruction_metadata
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


def save_model_from_runner(out_path: pathlib.Path, runner: GaussianSplatReconstruction) -> None:
    """
    Save the model from the runner to the specified output path in either PLY or USDZ format
    depending on the file extension.

    Args:
        out_path (pathlib.Path): Path to save the output file. Must end in .ply or .usdz.
        runner (GaussianSplatReconstruction): The runner containing the model to be saved.
    """
    if out_path.suffix.lower() == ".ply":
        runner.save_ply(out_path)
    elif out_path.suffix.lower() == ".usdz":
        runner.save_usdz(out_path)
    else:
        raise ValueError("Output path must end in .ply or .usdz")


def save_model_from_splats(out_path: pathlib.Path, model: GaussianSplat3d, metadata: dict) -> None:
    """
    Save the given Gaussian Splat model to the specified output path in either PLY or USDZ format
    depending on the file extension.

    Args:
        out_path (pathlib.Path): Path to save the output file. Must end in .ply or .usdz.
        model (GaussianSplat3d): The Gaussian Splat model to be saved.
        metadata (dict): Metadata to be saved with the model.
    """
    if out_path.suffix.lower() == ".ply":
        model.save_ply(out_path, metadata)
    elif out_path.suffix.lower() == ".usdz":
        export_splats_to_usdz(model, str(out_path))
    else:
        raise ValueError("Output path must end in .ply or .usdz")


def near_far_for_units(
    near_far_units: NearFarUnits,
    near: float,
    far: float,
    median_depths: torch.Tensor | None,
    camera_to_world_matrices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute near and far plane distances based on the specified units.

    Args:
        near_far_units (NearFarUnits): The units to use for near and far plane distances.
        near (float): The base near plane distance.
        far (float): The base far plane distance.
        median_depths (torch.Tensor | None): Tensor of median depths for each camera, required if
            using "median_depth" units.
        camera_to_world_matrices (torch.Tensor | None): Tensor of camera-to-world matrices, required if
            using "camera_extent" or "median_depth" units.

    Returns:
        near_scaled (torch.Tensor): The computed near plane distances (either a tensor of per-image values or a single value).
        far_scaled (torch.Tensor): The computed far plane distances (either a tensor of per-image values or a single value).
    """
    if near_far_units == "median_depth":
        if median_depths is None:
            raise ValueError("'median_depths' is required to use 'median_depth' near/far units")
        if camera_to_world_matrices is None:
            raise ValueError("'camera_to_world_matrices' is required to use 'median_depth' near/far units")
        if torch.any(median_depths.isnan()) or torch.any(median_depths <= 0.0):
            raise ValueError("median_depths in metadata must be positive and non-NaN")
        return near * median_depths, far * median_depths
    elif near_far_units == "camera_extent":
        if camera_to_world_matrices is None:
            raise ValueError("'camera_to_world_matrices' is required to use 'camera_extent' near/far units")
        scene_centroid = camera_to_world_matrices[:, :3, 3].mean(dim=0)
        max_camera_distance = torch.linalg.norm(
            camera_to_world_matrices[:, :3, 3] - scene_centroid[None, :], dim=1
        ).max()
        return near * max_camera_distance, far * max_camera_distance
    elif near_far_units == "absolute":
        return torch.tensor([near]), torch.tensor([far])
    else:
        raise ValueError(f"Invalid near_far_units: {near_far_units}")


def load_camera_metadata(metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load camera metadata from the given dictionary.

    Args:
        metadata (dict): Metadata dictionary containing camera information. Must contain the keys
            'camera_to_world_matrices', 'projection_matrices', and 'image_sizes'.

    Returns:
        camera_to_world_matrices (torch.Tensor): Tensor of shape (N, 4, 4) containing camera-to-world matrices.
        projection_matrices (torch.Tensor): Tensor of shape (N, 3, 3) containing projection matrices.
        image_sizes (torch.Tensor): Tensor of shape (N, 2) containing image sizes.
    """

    if "camera_to_world_matrices" not in metadata:
        raise ValueError("Gaussian splats file must contain 'camera_to_world_matrices'")

    if "projection_matrices" not in metadata:
        raise ValueError("Gaussian splats file must contain 'projection_matrices'")

    if "image_sizes" not in metadata:
        raise ValueError("Gaussian splats file must contain 'image_sizes'")

    camera_to_world_matrices = to_Mat44fBatch(metadata["camera_to_world_matrices"])
    projection_matrices = to_Mat33fBatch(metadata["projection_matrices"])
    image_sizes = to_Vec2iBatch(metadata["image_sizes"])

    return camera_to_world_matrices, projection_matrices, image_sizes
