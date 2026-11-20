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


def _load_obj(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read vertex positions and faces from a Wavefront OBJ file, exactly as stored.

    trimesh's OBJ loader splits a mesh by material (``usemtl``) and joins the pieces back together, which
    duplicates and renumbers the vertices on material boundaries. Reading the ``v`` and ``f`` lines directly keeps
    the vertex list and indices as they are in the file. Texture coordinates, normals, materials, groups, and
    per-vertex colors are ignored. Polygons with more than three vertices are split into a triangle fan.

    Args:
        path (pathlib.Path): Path to the ``.obj`` file.

    Returns:
        vertices (np.ndarray): A ``(V, 3)``-shaped ``float64`` array of vertex positions.
        faces (np.ndarray): An ``(F, 3)``-shaped ``int64`` array of zero-based vertex indices.
    """
    vertices: list[list[str]] = []  # coordinates are parsed as strings and converted by numpy at the end
    triangles: list[int] = []  # flat list of 1-based vertex indices, three per triangle
    pending = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            if "#" in line:
                line = line.split("#", 1)[0]
            if pending or line.rstrip().endswith("\\"):  # line continuation
                line = pending + line.rstrip()
                if line.endswith("\\"):
                    pending = line[:-1] + " "
                    continue
                pending = ""
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "v":
                if len(tokens) < 4:
                    raise ValueError(f"{path}:{line_number}: vertex has fewer than 3 coordinates")
                vertices.append(tokens[1:4])
            elif tokens[0] == "f":
                num_corners = len(tokens) - 1
                if num_corners < 3:
                    raise ValueError(f"{path}:{line_number}: face has fewer than 3 vertices")
                # Each face vertex is v, v/vt, v//vn, or v/vt/vn. Indices are 1-based, or negative to count back
                # from the most recently defined vertex (-1 is the last one).
                indices = [int(token.split("/", 1)[0]) for token in tokens[1:]]
                if min(indices) < 0:
                    num_vertices = len(vertices)
                    indices = [i if i >= 0 else num_vertices + i + 1 for i in indices]
                if num_corners == 3:
                    triangles.extend(indices)
                else:
                    for k in range(1, num_corners - 1):
                        triangles += (indices[0], indices[k], indices[k + 1])

    vertices_np = np.array(vertices, dtype=np.float64).reshape(-1, 3)
    faces_np = np.array(triangles, dtype=np.int64).reshape(-1, 3) - 1
    if faces_np.size > 0 and (faces_np.min() < 0 or faces_np.max() >= vertices_np.shape[0]):
        raise ValueError(f"{path} has face indices outside the range of its {vertices_np.shape[0]} vertices")
    return vertices_np, faces_np


def load_mesh(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a triangle mesh from a ``.obj`` file or any format trimesh supports (e.g. ``.ply``, ``.stl``, ``.glb``).

    For OBJ and PLY files, vertices and faces are returned as stored: no vertices are merged, reordered, split, or
    dropped (including vertices that no face references). Polygons with more than three vertices are triangulated.

    Args:
        path (pathlib.Path): Path to the mesh file.

    Returns:
        vertices (np.ndarray): A ``(V, 3)``-shaped ``float32`` array of vertex positions.
        faces (np.ndarray): An ``(F, 3)``-shaped ``int32`` array of vertex indices.

    Raises:
        ValueError: If the file does not contain a mesh with at least one face (e.g. it is a point cloud).
    """
    path = pathlib.Path(path)
    if path.suffix.lower() == ".obj":
        vertices, faces = _load_obj(path)
    else:
        # process=False stops trimesh from merging duplicate vertices and dropping unreferenced ones.
        mesh = trimesh.load_mesh(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"{path} does not contain a triangle mesh (loaded as {type(mesh).__name__})")
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    # trimesh loads a PLY with vertices but no faces (a point cloud) as an empty mesh, so check for faces explicitly.
    if faces.shape[0] == 0:
        raise ValueError(f"{path} contains no faces; expected a triangle mesh, not a point cloud")
    return vertices.astype(np.float32), faces.astype(np.int32)
