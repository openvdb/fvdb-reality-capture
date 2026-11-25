# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Mesh and point cloud file I/O. Files are read and written with trimesh, except OBJ files, which are read directly
(see :func:`_load_obj`).
"""

import pathlib

import numpy as np
import torch
import trimesh


def colors_to_uint8(colors: torch.Tensor) -> torch.Tensor:
    """
    Convert per-vertex colors to ``uint8`` RGB for writing to mesh/point cloud files.

    ``uint8`` colors are returned unchanged. Floating point colors are expected in ``[0, 1]``: they are clamped to
    that range, scaled by 255, and truncated (not rounded) as point_cloud_utils did. Unlike point_cloud_utils,
    which rejected out-of-range values, this clamps them.

    Args:
        colors (torch.Tensor): An ``(N, 3)``-shaped tensor of ``uint8`` or floating point colors.

    Returns:
        torch.Tensor: An ``(N, 3)``-shaped ``uint8`` tensor of colors.

    Raises:
        TypeError: If ``colors`` is neither ``uint8`` nor floating point (e.g. ``int64`` colors in ``[0, 255]``,
            which would otherwise be clamped to 0 or 255).
        ValueError: If floating point ``colors`` contain NaN.
    """
    if colors.dtype == torch.uint8:
        return colors
    if not colors.is_floating_point():
        raise TypeError(f"colors must be uint8 or floating point in [0, 1], got {colors.dtype}")
    if torch.isnan(colors).any():
        raise ValueError("colors contain NaN")
    return (colors.to(torch.float32).clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def _vertices_to_numpy(vertices: torch.Tensor, colors: torch.Tensor, name: str) -> np.ndarray:
    """Check ``vertices`` and ``colors`` against each other and return ``vertices`` as a ``float32`` numpy array."""
    if vertices.dim() != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(vertices.shape)}")
    if colors.shape != vertices.shape:
        raise ValueError(f"colors must have shape {tuple(vertices.shape)} to match {name}, got {tuple(colors.shape)}")
    if vertices.is_complex() or vertices.dtype == torch.bool:
        raise TypeError(f"{name} must be a real-valued tensor, got {vertices.dtype}")
    vertices_np = vertices.to(torch.float32).cpu().numpy()
    _check_finite(vertices_np, name)
    return vertices_np


def _check_finite(vertices: np.ndarray, source: str) -> None:
    if not np.isfinite(vertices).all():
        raise ValueError(f"{source} contains NaN or infinite vertex coordinates (after conversion to float32)")


def _check_face_indices(faces: np.ndarray, num_vertices: int, source: str) -> None:
    if faces.shape[0] > 0 and (int(faces.min()) < 0 or int(faces.max()) >= num_vertices):
        raise ValueError(f"{source} has face indices outside the range of its {num_vertices} vertices")


def save_colored_points(path: pathlib.Path, positions: torch.Tensor, colors: torch.Tensor) -> None:
    """
    Save a colored point cloud. The file format is inferred from the extension of ``path`` (e.g. ``.ply``).

    Args:
        path (pathlib.Path): Output file path.
        positions (torch.Tensor): An ``(N, 3)``-shaped tensor of point positions (written as ``float32``).
        colors (torch.Tensor): An ``(N, 3)``-shaped tensor of colors; see :func:`colors_to_uint8`.

    Raises:
        ValueError: If ``positions`` or ``colors`` do not have shape ``(N, 3)``, or ``positions`` contains values that
            are not finite in ``float32``.
        TypeError: If ``positions`` is complex or boolean.
    """
    positions_np = _vertices_to_numpy(positions, colors, "positions")
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

    Raises:
        ValueError: If the shapes do not match the ones above, ``faces`` is empty (use :func:`save_colored_points`
            for a point cloud; :func:`load_mesh` rejects a file without faces), ``faces`` index outside ``[0, V)``,
            or ``vertices`` contains values that are not finite in ``float32``.
        TypeError: If ``faces`` is not an integer tensor, or ``vertices`` is complex or boolean.
    """
    vertices_np = _vertices_to_numpy(vertices, colors, "vertices")
    if faces.dim() != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {tuple(faces.shape)}")
    faces_np = faces.cpu().numpy()
    if not np.issubdtype(faces_np.dtype, np.integer):
        raise TypeError(f"faces must be an integer tensor, got {faces.dtype}")
    if faces_np.shape[0] == 0:
        raise ValueError("faces is empty; use save_colored_points to save a point cloud")
    _check_face_indices(faces_np, vertices_np.shape[0], "faces")
    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
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
        faces (np.ndarray): An ``(F, 3)``-shaped ``int64`` array of zero-based vertex indices. They are not
            range-checked here.

    Raises:
        ValueError: If a ``v`` or ``f`` line is malformed, or the file ends with a ``\\`` line continuation.
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
    if pending:
        raise ValueError(f"{path}: file ends with an unterminated line continuation")

    vertices_np = np.array(vertices, dtype=np.float64).reshape(-1, 3)
    faces_np = np.array(triangles, dtype=np.int64).reshape(-1, 3) - 1  # indices are checked in load_mesh
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
        ValueError: If the file does not contain a mesh with at least one face (e.g. it is a point cloud), a face
            refers to a vertex that does not exist, a vertex coordinate is not finite in ``float32``, or there are
            too many vertices to index with ``int32``.
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
    if vertices.shape[0] > np.iinfo(np.int32).max:
        raise ValueError(f"{path} has {vertices.shape[0]} vertices, too many to index with int32 faces")
    # trimesh does not check face indices on load, so check them for every format.
    _check_face_indices(faces, vertices.shape[0], str(path))
    with np.errstate(over="ignore"):  # values too large for float32 become inf and are rejected below
        vertices = vertices.astype(np.float32)
    _check_finite(vertices, str(path))
    return vertices, faces.astype(np.int32)
