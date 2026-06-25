#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Prepare mesh and/or Gaussian splat assets for Isaac Sim.
- assumes ecef2enu normalization is applied to the scene
    - turn off upright rotation with --no-ecef2enu-rotation
- Exports mesh and splat as a single aligned usdz
- mesh is water tight so robots can walk on it and objects dont fall through
    - turn off with --no-watertight
- can crop scene with --bbox
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from pathlib import Path
from typing import Optional

import numpy as np
import point_cloud_utils as pcu
import torch
from fvdb import GaussianSplat3d

from fvdb_reality_capture.tools import export_splats_to_usdz


def _crop_splat_model(
    model: GaussianSplat3d,
    bbox: list[float] | None,
    logger: logging.Logger,
) -> GaussianSplat3d:
    if bbox is None:
        return model

    xyz = model.means.cpu().numpy()
    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    mask = (
        (xyz[:, 0] >= min_x)
        & (xyz[:, 0] <= max_x)
        & (xyz[:, 1] >= min_y)
        & (xyz[:, 1] <= max_y)
        & (xyz[:, 2] >= min_z)
        & (xyz[:, 2] <= max_z)
    )
    mask_tensor = torch.from_numpy(mask).to(model.device)
    cropped = model[mask_tensor]
    logger.info("Cropped splats from %d to %d points", len(xyz), len(cropped.means))
    return cropped


def _prepare_mesh(
    input_path: pathlib.Path,
    bbox: list[float] | None,
    resolution: int,
    logger: logging.Logger,
    *,
    watertight: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and optionally crop a mesh; vertices stay in the training frame."""
    vertices, faces = pcu.load_mesh_vf(str(input_path))
    logger.info("Preparing mesh from %s", input_path)

    if bbox is not None:
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        mask = (
            (vertices[:, 0] >= min_x)
            & (vertices[:, 0] <= max_x)
            & (vertices[:, 1] >= min_y)
            & (vertices[:, 1] <= max_y)
            & (vertices[:, 2] >= min_z)
            & (vertices[:, 2] <= max_z)
        )
        keep_indices = np.where(mask)[0]
        old_to_new = np.full(vertices.shape[0], -1)
        old_to_new[keep_indices] = np.arange(len(keep_indices))
        vertices = vertices[keep_indices]

        valid_faces = []
        for face in faces:
            if all(old_to_new[idx] != -1 for idx in face):
                valid_faces.append([old_to_new[idx] for idx in face])
        faces = np.array(valid_faces, dtype=np.int32)
        logger.info(
            "Cropped mesh bounds min=%s max=%s",
            vertices.min(axis=0),
            vertices.max(axis=0),
        )

    if watertight:
        vertices, faces = pcu.make_mesh_watertight(vertices, faces, resolution=resolution)
        logger.info(
            "Watertight mesh: %d vertices, %d faces",
            vertices.shape[0],
            faces.shape[0],
        )
    else:
        logger.info(
            "Using mesh as-is (watertight skipped): %d vertices, %d faces",
            vertices.shape[0],
            faces.shape[0],
        )
    return vertices.astype(np.float32), faces.astype(np.int32)


def _write_mesh_obj(vertices: np.ndarray, faces: np.ndarray, output_path: pathlib.Path) -> None:
    """Write a plain OBJ file (training-frame coordinates)."""
    with open(output_path, "w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def crop_and_convert_splat_to_usdz(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    bbox: list[float] | None = None,
    apply_ecef2enu_rotation: bool = True,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """Convert a Gaussian splat PLY to USDZ with optional scene-level Isaac alignment."""
    model, _metadata = GaussianSplat3d.from_ply(str(input_path))
    model = _crop_splat_model(model, bbox, logger)
    export_splats_to_usdz(
        model,
        output_path,
        apply_ecef2enu_rotation=apply_ecef2enu_rotation,
    )


def crop_and_convert_mesh_to_obj(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    bbox: list[float] | None = None,
    resolution: int = 100_000,
    *,
    watertight: bool = True,
    logger: logging.Logger = logging.getLogger(__name__),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a mesh to OBJ in the training coordinate frame."""
    vertices, faces = _prepare_mesh(input_path, bbox, resolution, logger, watertight=watertight)
    _write_mesh_obj(vertices, faces, output_path)
    logger.info("Saved mesh OBJ to %s", output_path)
    return vertices, faces


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Crop mesh and/or splat assets and export an Isaac-ready combined USDZ",
    )
    parser.add_argument("--input-splat", type=Path, help="Input splat file (PLY format)")
    parser.add_argument("--input-mesh", type=Path, help="Input mesh file (PLY/OBJ format)")
    parser.add_argument("--output-path", type=Path, required=True, help="Output path without extension")
    parser.add_argument("--resolution", type=int, default=100_000, help="Watertight mesh resolution")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=6,
        metavar=("MIN_X", "MIN_Y", "MIN_Z", "MAX_X", "MAX_Y", "MAX_Z"),
        help="Optional crop bounds: min_x min_y min_z max_x max_y max_z",
    )
    parser.add_argument(
        "--write-obj",
        action="store_true",
        help="Also write a training-frame OBJ alongside the USDZ when --input-mesh is set",
    )
    parser.add_argument(
        "--no-ecef2enu-rotation",
        default=False,
        action="store_true",
        help="Skip -90° X upright rotation for ecef2enu-normalized scenes",
    )
    parser.add_argument(
        "--no-watertight",
        action="store_true",
        help="Skip making the mesh watertight, might cause collision issues if used in Isaac Sim",
    )
    args = parser.parse_args()
    if not args.input_splat and not args.input_mesh:
        parser.error("At least one of --input-splat or --input-mesh must be provided")

    apply_ecef2enu_rotation = not args.no_ecef2enu_rotation
    usdz_output_path = args.output_path.with_suffix(".usdz")
    mesh_output_path = args.output_path.with_suffix(".obj")

    model: Optional[GaussianSplat3d] = None
    mesh_vertices: Optional[np.ndarray] = None
    mesh_faces: Optional[np.ndarray] = None

    if args.input_splat:
        model, _ = GaussianSplat3d.from_ply(str(args.input_splat))
        model = _crop_splat_model(model, args.bbox, logger)

    if args.input_mesh:
        mesh_vertices, mesh_faces = _prepare_mesh(
            args.input_mesh,
            args.bbox,
            args.resolution,
            logger,
            watertight=not args.no_watertight,
        )
        if args.write_obj:
            _write_mesh_obj(mesh_vertices, mesh_faces, mesh_output_path)

    if model is None and mesh_vertices is None:
        parser.error("No assets left after cropping")

    export_splats_to_usdz(
        model,
        usdz_output_path,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        apply_ecef2enu_rotation=apply_ecef2enu_rotation,
    )


if __name__ == "__main__":
    main()
