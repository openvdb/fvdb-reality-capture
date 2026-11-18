# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
from typing import Literal

import numpy as np
import torch
import trimesh
from fvdb.types import to_Mat33fBatch, to_Mat44fBatch, to_Vec2iBatch

from fvdb_reality_capture import GaussianSplat3d
from fvdb_reality_capture.radiance_fields import (
    GaussianSplatReconstruction,
    load_splats_from_file,
)
from fvdb_reality_capture.sfm_scene import SfmScene
from fvdb_reality_capture.tools import export_splats_to_usd

from ...enums import CameraModel

DatasetType = Literal["colmap", "simple_directory", "e57"]
NearFarUnits = Literal["absolute", "camera_extent", "median_depth"]

# ``load_splats_from_file`` is implemented in the library layer (``radiance_fields``) because the
# GARfVDB trainer's public resume API also needs it and must not depend on CLI internals. It is
# re-exported here so all of the CLI load/save helpers remain importable from one place. Listing it in
# ``__all__`` marks the re-export as intentional so it is not stripped by an unused-import pass.
__all__ = [
    "DatasetType",
    "NearFarUnits",
    "colors_to_uint8",
    "load_camera_metadata",
    "load_mesh",
    "load_sfm_scene",
    "load_splats_from_file",
    "near_far_for_units",
    "save_colored_mesh",
    "save_colored_points",
    "save_model_from_runner",
    "save_model_from_splats",
]


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
    Save the model from the runner to the specified output path in either PLY or USD format
    depending on the file extension.

    Args:
        out_path (pathlib.Path): Path to save the output file. Must end in .ply, .usdc, or .usdz.
        runner (GaussianSplatReconstruction): The runner containing the model to be saved.
    """
    if out_path.suffix.lower() == ".ply":
        runner.save_ply(out_path)
    elif out_path.suffix.lower() in (".usdc", ".usdz"):
        runner.save_usd(out_path, usdz=out_path.suffix.lower() == ".usdz")
    else:
        raise ValueError("Output path must end in .ply, .usdc, or .usdz")


def save_model_from_splats(out_path: pathlib.Path, model: GaussianSplat3d, metadata: dict) -> None:
    """
    Save the given Gaussian Splat model to the specified output path in either PLY or USD format
    depending on the file extension.

    Args:
        out_path (pathlib.Path): Path to save the output file. Must end in .ply, .usdc, or .usdz.
        model (GaussianSplat3d): The Gaussian Splat model to be saved.
        metadata (dict): Metadata to be saved with the model.
    """
    if out_path.suffix.lower() == ".ply":
        model.save_ply(out_path, metadata)
    elif out_path.suffix.lower() in (".usdc", ".usdz"):
        export_splats_to_usd(model, str(out_path), usdz=out_path.suffix.lower() == ".usdz")
    else:
        raise ValueError("Output path must end in .ply, .usdc, or .usdz")


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


def load_camera_metadata(metadata: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load camera metadata from the given dictionary.

    Args:
        metadata (dict): Metadata dictionary containing camera information. Must contain the keys
            'camera_to_world_matrices', 'projection_matrices', and 'image_sizes'. It may also
            contain 'camera_models' and 'distortion_coeffs'. If those are absent, pinhole defaults
            are returned for backwards compatibility.

    Returns:
        camera_to_world_matrices (torch.Tensor): Tensor of shape (N, 4, 4) containing camera-to-world matrices.
        projection_matrices (torch.Tensor): Tensor of shape (N, 3, 3) containing projection matrices.
        image_sizes (torch.Tensor): Tensor of shape (N, 2) containing image sizes.
        camera_models (torch.Tensor): Tensor of shape (N,) containing integer-encoded camera models.
        distortion_coeffs (torch.Tensor): Tensor of shape (N, 12) containing packed distortion coeffs.
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
    if "camera_models" in metadata:
        camera_models = torch.as_tensor(metadata["camera_models"], dtype=torch.int32)
    else:
        camera_models = torch.full((camera_to_world_matrices.shape[0],), int(CameraModel.PINHOLE), dtype=torch.int32)
    if "distortion_coeffs" in metadata:
        distortion_coeffs = torch.as_tensor(metadata["distortion_coeffs"], dtype=torch.float32)
    else:
        distortion_coeffs = torch.zeros((camera_to_world_matrices.shape[0], 12), dtype=torch.float32)

    return camera_to_world_matrices, projection_matrices, image_sizes, camera_models, distortion_coeffs


def colors_to_uint8(colors: torch.Tensor) -> torch.Tensor:
    """
    Convert per-vertex colors to ``uint8`` RGB for writing to mesh/point cloud files.

    ``uint8`` colors are returned unchanged. Floating point colors are expected in ``[0, 1]``: they are clamped to
    that range, scaled by 255, and truncated (not rounded), matching the conversion point_cloud_utils used.

    Args:
        colors (torch.Tensor): An ``(N, 3)``-shaped tensor of ``uint8`` or floating point colors.

    Returns:
        torch.Tensor: An ``(N, 3)``-shaped ``uint8`` tensor of colors.
    """
    if colors.dtype == torch.uint8:
        return colors
    return (colors.to(torch.float32).clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def save_colored_points(path: pathlib.Path, positions: torch.Tensor, colors: torch.Tensor) -> None:
    """
    Save a colored point cloud. The file format is inferred from the extension of ``path`` (e.g. ``.ply``).

    Args:
        path (pathlib.Path): Output file path.
        positions (torch.Tensor): An ``(N, 3)``-shaped tensor of point positions (written as ``float32``).
        colors (torch.Tensor): An ``(N, 3)``-shaped tensor of colors; see :func:`colors_to_uint8`.
    """
    positions_np = positions.to(torch.float32).cpu().numpy()
    colors_np = colors_to_uint8(colors).cpu().numpy()
    trimesh.PointCloud(positions_np, colors=colors_np).export(str(path))


def save_colored_mesh(path: pathlib.Path, vertices: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor) -> None:
    """
    Save a triangle mesh with per-vertex colors. The file format is inferred from the extension of ``path``.

    Vertices and faces are written as given: no vertices are merged or reordered.

    Args:
        path (pathlib.Path): Output file path.
        vertices (torch.Tensor): A ``(V, 3)``-shaped tensor of vertex positions (written as ``float32``).
        faces (torch.Tensor): An ``(F, 3)``-shaped integer tensor of vertex indices.
        colors (torch.Tensor): A ``(V, 3)``-shaped tensor of per-vertex colors; see :func:`colors_to_uint8`.
    """
    mesh = trimesh.Trimesh(
        vertices=vertices.to(torch.float32).cpu().numpy(),
        faces=faces.cpu().numpy(),
        vertex_colors=colors_to_uint8(colors).cpu().numpy(),
        process=False,
    )
    mesh.export(str(path))


def load_mesh(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a triangle mesh from any format trimesh supports (e.g. ``.ply``, ``.obj``, ``.stl``).

    Vertices and faces are returned as stored: no vertices are merged, reordered, or dropped (including vertices
    that no face references).

    Args:
        path (pathlib.Path): Path to the mesh file.

    Returns:
        vertices (np.ndarray): A ``(V, 3)``-shaped ``float32`` array of vertex positions.
        faces (np.ndarray): An ``(F, 3)``-shaped ``int32`` array of vertex indices.
    """
    # maintain_order stops the OBJ loader from dropping and reordering vertices; other formats ignore it.
    mesh = trimesh.load_mesh(path, process=False, maintain_order=True)
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)
