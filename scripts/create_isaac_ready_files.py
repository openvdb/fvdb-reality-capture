#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Prepare mesh and/or Gaussian splat assets for Isaac Sim.
- assumes ecef2enu normalization is applied to the scene
    - turn off with --no-usd-transform
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
from pxr import Gf, Usd, UsdGeom, UsdUtils, Vt

from fvdb_reality_capture.tools._export_splats_to_usdz import (
    NamedUSDStage,
    USD_GAUSSIANS_ROOT_PATH,
    _initialize_particlefield_usd_stage,
    _write_particlefield3d_gaussian_splat,
    _write_particlefield_usdz,
)

USD_SCENE_ROOT_PATH = "/World/Scene"
USD_MESH_PAYLOAD_PATH = "/World/mesh"
USD_MESH_SCENE_PATH = f"{USD_SCENE_ROOT_PATH}/mesh"


def create_rotation_matrix_x(degrees: float) -> np.ndarray:
    """Rotation matrix for +degrees about the X axis (column-vector convention)."""
    rad = np.radians(degrees)
    cos, sin = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float64)


def rotation_matrix_to_gf_matrix4d(rotation: np.ndarray) -> Gf.Matrix4d:
    """Convert a column-vector rotation matrix to USD's Gf.Matrix4d."""
    # NumPy uses column vectors (p' = R @ p). USD xforms use row-vector layout, so
    # pass R.T via SetTransform — same convention as 3dgrut export.
    r = rotation[:3, :3].astype(np.float64)
    matrix = Gf.Matrix4d()
    matrix.SetTransform(Gf.Matrix3d(*r.T.flatten()), Gf.Vec3d(0.0, 0.0, 0.0))
    return matrix


def get_isaac_scene_alignment_matrix() -> Gf.Matrix4d:
    """
    For ecef2enu normalized scenes (Z-up), rotate the whole USDZ -90° about X
    so content is upright in Isaac Sim's Y-up USD stage.
    """
    rotation = create_rotation_matrix_x(-90)
    return rotation_matrix_to_gf_matrix4d(rotation)


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


def build_mesh_payload_stage(vertices: np.ndarray, faces: np.ndarray) -> Usd.Stage:
    """Create a USD stage containing a single triangle mesh at /World/mesh."""
    stage = _initialize_particlefield_usd_stage()
    mesh = UsdGeom.Mesh.Define(stage, USD_MESH_PAYLOAD_PATH)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    return stage


def build_gaussians_payload_stage(model: GaussianSplat3d) -> Usd.Stage:
    """Create a USD stage with ParticleField gaussians in the training frame."""
    stage = _initialize_particlefield_usd_stage()
    UsdGeom.Xform.Define(stage, USD_GAUSSIANS_ROOT_PATH)
    _write_particlefield3d_gaussian_splat(stage, model)
    return stage


def _add_scene_xform(stage: Usd.Stage, matrix: Optional[Gf.Matrix4d]) -> UsdGeom.Xform:
    """Create /World/Scene and optionally set its transform op."""
    scene_xform = UsdGeom.Xform.Define(stage, USD_SCENE_ROOT_PATH)
    if matrix is not None:
        scene_xform.AddTransformOp().Set(matrix)
    return scene_xform


def compose_isaac_scene_usdz(
    output_path: pathlib.Path,
    model: Optional[GaussianSplat3d] = None,
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_faces: Optional[np.ndarray] = None,
    apply_scene_transform: bool = True,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """
    Package mesh and/or splats into one USDZ with scene-level transforms.

    Hierarchy:
        /World/Scene                 (Isaac alignment xform)
          /Gaussians                 (reference -> gaussians.usdc)
          /mesh                      (grouping xform, no extra rotation)
            /geometry                (reference -> mesh.usdc)
    """
    if model is None and mesh_vertices is None:
        raise ValueError("At least one of model or mesh_vertices must be provided")

    stages: list[NamedUSDStage] = []
    root_stage = _initialize_particlefield_usd_stage()

    # Payload .usdc files are packed into the USDZ after references are authored;
    # suppress expected "could not open asset" warnings during in-memory composition.
    _ = UsdUtils.CoalescingDiagnosticDelegate()

    scene_matrix = get_isaac_scene_alignment_matrix() if apply_scene_transform else None
    _add_scene_xform(root_stage, scene_matrix)
    if scene_matrix is not None:
        logger.info("Applied Isaac scene alignment (-90° X) on %s", USD_SCENE_ROOT_PATH)

    if model is not None:
        gaussians_stage = NamedUSDStage(filename="gaussians.usdc", stage=build_gaussians_payload_stage(model))
        stages.append(gaussians_stage)
        gaussians_ref = root_stage.OverridePrim(f"{USD_SCENE_ROOT_PATH}/Gaussians")
        gaussians_ref.GetReferences().AddReference(gaussians_stage.filename, USD_GAUSSIANS_ROOT_PATH)
        logger.info("Referenced gaussians payload at %s/Gaussians", USD_SCENE_ROOT_PATH)

    if mesh_vertices is not None and mesh_faces is not None:
        mesh_stage = NamedUSDStage(
            filename="mesh.usdc",
            stage=build_mesh_payload_stage(mesh_vertices, mesh_faces),
        )
        stages.append(mesh_stage)

        UsdGeom.Xform.Define(root_stage, USD_MESH_SCENE_PATH)
        mesh_ref = root_stage.OverridePrim(f"{USD_MESH_SCENE_PATH}/geometry")
        mesh_ref.GetReferences().AddReference(mesh_stage.filename, USD_MESH_PAYLOAD_PATH)
        logger.info("Referenced mesh payload at %s/geometry", USD_MESH_SCENE_PATH)

    default_stage = NamedUSDStage(filename="default.usda", stage=root_stage)
    _write_particlefield_usdz(output_path, [default_stage, *stages])
    logger.info("Wrote Isaac scene USDZ to %s", output_path)


def crop_and_convert_splat_to_usdz(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    bbox: list[float] | None = None,
    apply_scene_transform: bool = True,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """Convert a Gaussian splat PLY to USDZ with optional scene-level Isaac alignment."""
    model, _metadata = GaussianSplat3d.from_ply(str(input_path))
    model = _crop_splat_model(model, bbox, logger)
    compose_isaac_scene_usdz(
        output_path,
        model=model,
        apply_scene_transform=apply_scene_transform,
        logger=logger,
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
        "--no-usd-transform",
        default=False,
        action="store_true",
        help="Skip rotating USDZ upright under ecef2enu convention, use if scene is not ecef2enu normalized",
    )
    parser.add_argument(
        "--no-watertight",
        action="store_true",
        help="Skip making the mesh watertight, might cause collision issues if used in Isaac Sim",
    )
    args = parser.parse_args()
    if not args.input_splat and not args.input_mesh:
        parser.error("At least one of --input-splat or --input-mesh must be provided")

    apply_scene_transform = not (args.no_usd_transform)
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

    compose_isaac_scene_usdz(
        usdz_output_path,
        model=model,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        apply_scene_transform=apply_scene_transform,
        logger=logger,
    )


if __name__ == "__main__":
    main()
