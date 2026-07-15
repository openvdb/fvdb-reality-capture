# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

# pip install msgpack numpy usd-core types-usd

"""
Isaac Sim version prior to 6.0 use a format created before the current Splat USD standard
https://openusd.org/dev/user_guides/schemas/usdVol/ParticleField3DGaussianSplat.html
If using a version of Isaac sim prior to 6.0, use the legacy format with --legacy flag. Newer versions should use the default particle field format.
"""
import gzip
import io
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import msgpack
import numpy as np
import torch
from fvdb import GaussianSplat3d
from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, UsdVol, Vt

logger = logging.getLogger(__name__)

DEFAULT_PROJECTION_MODE_HINT = "perspective"
DEFAULT_SORTING_MODE_HINT = "cameraDistance"
GAUSSIANS_PAYLOAD_PRIM = "/gaussians"
MESH_PAYLOAD_PRIM = "/mesh"


def _usd_safe_name(name: str) -> str:
    """Sanitize an arbitrary string into a valid USD prim name."""
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not safe_name:
        return "gaussians"
    if safe_name[0].isdigit():
        safe_name = f"asset_{safe_name}"
    return safe_name


def _usd_asset_name_from_path(out_path: Path, asset_name: Optional[str] = None) -> str:
    """Derive a valid USD prim name, from ``asset_name`` if given, else the output ``.usdz`` file stem."""
    return _usd_safe_name(asset_name if asset_name is not None else out_path.stem)


def _asset_scene_paths(asset_name: str) -> tuple[str, str, str]:
    """Return ``(asset_xform, gaussians, mesh)`` prim paths under ``/World``."""
    asset_path = f"/World/{asset_name}"
    return asset_path, f"{asset_path}/gaussians", f"{asset_path}/mesh"


def _initialize_asset_root_stage(_asset_name: str) -> Usd.Stage:
    """Y-up root stage with ``/World`` as ``defaultPrim`` (Isaac scene convention)."""
    return _initialize_particlefield3d_usd_stage()


def _initialize_payload_stage(default_prim_name: str) -> Usd.Stage:
    """In-memory Y-up stage for a single payload prim (no ``/World`` wrapper)."""
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Y")
    stage.SetMetadata("defaultPrim", default_prim_name)
    return stage


@dataclass(kw_only=True)
class NamedUSDStage:
    filename: str
    stage: Usd.Stage

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.stage.Export(str(out_dir / self.filename))

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=self.filename, delete=False) as temp_file:
            temp_file_path = temp_file.name
        self.stage.GetRootLayer().Export(temp_file_path)
        with open(temp_file_path, "rb") as file:
            usd_data = file.read()
        zip_file.writestr(self.filename, usd_data)
        os.unlink(temp_file_path)


def _initialize_legacy_nurec_usd_stage() -> Usd.Stage:
    """
    Initialize a Z-up USD stage for the legacy Omniverse NuRec export.

    This format uses ``UsdVol.Volume`` with embedded ``.nurec`` field assets and was
    the Isaac Sim / Omniverse path before OpenUSD's ``ParticleField3DGaussianSplat``
    schema (Isaac Sim 6.0+). Retained for ``export_splats_to_usd(..., legacy=True, usdz=True)``.

    Returns:
        Usd.Stage: In-memory stage with ``/World`` as default prim.
    """
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1)
    stage.SetMetadata("upAxis", "Z")

    # Define xform containing everything.
    world_path = "/World"
    UsdGeom.Xform.Define(stage, world_path)
    stage.SetMetadata("defaultPrim", world_path[1:])

    return stage


def _initialize_particlefield3d_usd_stage() -> Usd.Stage:
    """
    Initialize an in-memory Y-up USD stage for ParticleField3DGaussianSplat export.

    Returns:
        Usd.Stage: In-memory stage with ``/World`` as default prim.
    """
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Y")

    world_path = "/World"
    UsdGeom.Xform.Define(stage, world_path)
    stage.SetMetadata("defaultPrim", world_path[1:])

    return stage


@dataclass
class _PostActivationGaussianArrays:
    """Gaussian splat arrays after applying scale, opacity, and rotation activations."""

    positions: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    densities: np.ndarray
    albedo: np.ndarray
    specular: np.ndarray
    sh_degree: int

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]


def _extract_postactivation_gaussian_arrays(
    model: GaussianSplat3d,
) -> _PostActivationGaussianArrays:
    """
    Convert fvdb model tensors to post-activation arrays for ParticleField3DGaussianSplat export.

    Args:
        model (GaussianSplat3d): Gaussian splat model in fvdb training parameterization.

    Returns:
        Arrays with activations applied (exp scale, sigmoid opacity, normalized quats).
    """
    positions = model.means.detach().cpu().numpy().astype(np.float32)
    rotations = model.quats.detach().cpu().numpy().astype(np.float32)
    scales = torch.exp(model.log_scales).detach().cpu().numpy().astype(np.float32)
    densities = torch.sigmoid(model.logit_opacities).detach().cpu().numpy().astype(np.float32)
    sh0 = model.sh0.detach().cpu().numpy().astype(np.float32)
    shN = model.shN.detach().cpu().numpy().astype(np.float32)
    sh_degree = int(model.sh_degree)

    num_gaussians = positions.shape[0]
    num_rest_coeffs = (sh_degree + 1) ** 2 - 1

    quat_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / np.clip(quat_norms, 1e-8, None)

    albedo = sh0[:, 0, :].reshape(num_gaussians, 3)
    specular = shN.reshape(num_gaussians, -1)
    expected_specular_cols = num_rest_coeffs * 3
    if specular.shape[1] != expected_specular_cols:
        logger.warning(
            "shN has %d coefficient columns but SH degree %d expects %d; padding/truncating to match. "
            "This usually indicates a mismatch between the model's SH degree and its shN tensor.",
            specular.shape[1],
            sh_degree,
            expected_specular_cols,
        )
        padded = np.zeros((num_gaussians, expected_specular_cols), dtype=np.float32)
        if specular.shape[1] > 0:
            padded[:, : min(specular.shape[1], expected_specular_cols)] = specular[:, :expected_specular_cols]
        specular = padded

    if densities.ndim == 1:
        densities = densities[:, np.newaxis]

    return _PostActivationGaussianArrays(
        positions=positions,
        rotations=rotations,
        scales=scales,
        densities=densities,
        albedo=albedo,
        specular=specular,
        sh_degree=sh_degree,
    )


def _pack_particlefield3d_sh_coefficients(
    albedo: np.ndarray,
    specular: np.ndarray,
    num_gaussians: int,
    sh_degree: int,
) -> tuple[np.ndarray, int]:
    """
    Pack DC and higher-order SH into a flat Vec3f array for ParticleField3DGaussianSplat USD.

    Layout per gaussian: (degree+1)^2 RGB triplets in basis order.

    Args:
        albedo: DC (SH0) RGB per gaussian, shape (N, 3).
        specular: Higher-order SH coefficients, shape (N, num_rest_coeffs * 3).
        num_gaussians: Number of gaussians.
        sh_degree: Spherical harmonics degree.

    Returns:
        Flat (N * num_sh_coeffs, 3) array and number of SH coefficients per gaussian.
    """
    if sh_degree == 0:
        return albedo.reshape(-1, 3), 1

    num_sh_coeffs = (sh_degree + 1) ** 2
    num_rest_coeffs = num_sh_coeffs - 1
    specular_reshaped = specular.reshape((num_gaussians, num_rest_coeffs, 3))
    albedo_expanded = albedo.reshape((num_gaussians, 1, 3))
    all_coeffs = np.concatenate([albedo_expanded, specular_reshaped], axis=1)
    return all_coeffs.reshape(-1, 3), num_sh_coeffs


def _compute_gaussian_bounding_extent(positions: np.ndarray) -> Vt.Vec3fArray:
    """
    Compute axis-aligned bounding box [min, max] from gaussian centers.

    Args:
        positions: Gaussian center positions, shape (N, 3).

    Returns:
        Two-element ``Vt.Vec3fArray`` with min and max corners.
    """
    if positions.shape[0] == 0:
        raise ValueError("Cannot compute a bounding extent: the Gaussian splat model has no gaussians")
    min_bounds = np.min(positions, axis=0)
    max_bounds = np.max(positions, axis=0)
    return Vt.Vec3fArray(
        [
            Gf.Vec3f(float(min_bounds[0]), float(min_bounds[1]), float(min_bounds[2])),
            Gf.Vec3f(float(max_bounds[0]), float(max_bounds[1]), float(max_bounds[2])),
        ]
    )


def _apply_particlefield3d_color_space(prim: Usd.Prim, linear_srgb: bool) -> None:
    """
    Apply ColorSpaceAPI on the ParticleField3DGaussianSplat prim (matches 3dgrut export).

    Per 3dgrut/USD color space conventions:
    - lin_rec709_scene: linear Rec.709 (post-processed/linear RGB data)
    - srgb_rec709_display: sRGB Rec.709 (gamma-encoded data, fvdb default training)

    Args:
        prim: ParticleField3DGaussianSplat prim to tag.
        linear_srgb: If True, use ``lin_rec709_scene``; else ``srgb_rec709_display``.
    """
    color_space = "lin_rec709_scene" if linear_srgb else "srgb_rec709_display"
    color_space_api = Usd.ColorSpaceAPI.Apply(prim)
    color_space_api.CreateColorSpaceNameAttr().Set(color_space)


def _write_particlefield3d_gaussian_splat(
    stage: Usd.Stage,
    model: GaussianSplat3d,
    prim_path: str,
    linear_srgb: bool = False,
    projection_mode_hint: str = DEFAULT_PROJECTION_MODE_HINT,
    sorting_mode_hint: str = DEFAULT_SORTING_MODE_HINT,
) -> Usd.Prim:
    """
    Write post-activation gaussian data to a ParticleField3DGaussianSplat prim.

    Args:
        stage (Usd.Stage): USD stage to author the prim on.
        model (GaussianSplat3d): Gaussian splat model to export.
        prim_path (str): Absolute prim path (e.g. ``/World/ambulance/gaussians``).
        linear_srgb: Color space flag passed to ColorSpaceAPI (see 3dgrut convention).
        projection_mode_hint: ParticleField3DGaussianSplat projection hint.
        sorting_mode_hint: ParticleField3DGaussianSplat sorting hint.

    Returns:
        The authored ParticleField3DGaussianSplat prim.
    """
    attrs = _extract_postactivation_gaussian_arrays(model)
    num_gaussians = attrs.num_gaussians
    sh_degree = attrs.sh_degree
    num_sh_coeffs = (sh_degree + 1) ** 2

    logger.info(f"Creating ParticleField3DGaussianSplat at {prim_path}")
    logger.info(f"  Gaussians: {num_gaussians:,}")
    logger.info(f"  SH degree: {sh_degree} ({num_sh_coeffs} coeffs per gaussian)")

    if sh_degree > 0:
        shN_max = float(np.max(np.abs(attrs.specular)))
        shN_mean = float(np.mean(np.abs(attrs.specular)))
        logger.info(f"  shN magnitude: max={shN_max:.6f}, mean={shN_mean:.6f}")
        if shN_max < 1e-8:
            logger.warning("shN coefficients are all near zero — scene will look like SH degree 0")

    gauss_schema = UsdVol.ParticleField3DGaussianSplat.Define(stage, prim_path)
    prim = gauss_schema.GetPrim()

    gauss_schema.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(attrs.positions))
    quats_list = [Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in attrs.rotations]
    gauss_schema.CreateOrientationsAttr().Set(Vt.QuatfArray(quats_list))
    gauss_schema.CreateScalesAttr().Set(Vt.Vec3fArray.FromNumpy(attrs.scales))

    densities_clamped = np.clip(attrs.densities.flatten(), 0.0, 1.0)
    gauss_schema.CreateOpacitiesAttr().Set(Vt.FloatArray.FromNumpy(densities_clamped.astype(np.float32)))

    gauss_schema.CreateRadianceSphericalHarmonicsDegreeAttr().Set(sh_degree)
    sh_coeffs_attr = gauss_schema.CreateRadianceSphericalHarmonicsCoefficientsAttr()
    all_sh_flat, num_sh_coeffs = _pack_particlefield3d_sh_coefficients(
        attrs.albedo, attrs.specular, num_gaussians, sh_degree
    )
    sh_coeffs_attr.Set(Vt.Vec3fArray.FromNumpy(all_sh_flat.astype(np.float32)))
    sh_coeffs_attr.SetMetadata("elementSize", num_sh_coeffs)

    gauss_schema.CreateProjectionModeHintAttr().Set(projection_mode_hint)
    gauss_schema.CreateSortingModeHintAttr().Set(sorting_mode_hint)

    _apply_particlefield3d_color_space(prim, linear_srgb)
    gauss_schema.CreateExtentAttr().Set(_compute_gaussian_bounding_extent(attrs.positions))

    logger.info(f"Created ParticleField3DGaussianSplat with {num_gaussians:,} Gaussians")
    return prim


def _build_particlefield3d_gaussians_payload(
    model: GaussianSplat3d,
    *,
    linear_srgb: bool = False,
    sorting_mode_hint: str = DEFAULT_SORTING_MODE_HINT,
    projection_mode_hint: str = DEFAULT_PROJECTION_MODE_HINT,
) -> NamedUSDStage:
    """
    Build a ``gaussians.usdc`` payload with a single generic ``/gaussians`` prim.

    Asset naming (e.g. ``/World/ambulance``) is applied only in ``default.usda``.

    Args:
        model (GaussianSplat3d): Gaussian splat model to export.
        linear_srgb (bool): Color space flag for radiance SH coefficients.
        sorting_mode_hint: ParticleField3DGaussianSplat sorting hint.
        projection_mode_hint: ParticleField3DGaussianSplat projection hint.

    Returns:
        NamedUSDStage with filename ``gaussians.usdc`` and an in-memory stage.
    """
    stage = _initialize_payload_stage("gaussians")
    _write_particlefield3d_gaussian_splat(
        stage,
        model,
        GAUSSIANS_PAYLOAD_PRIM,
        linear_srgb=linear_srgb,
        sorting_mode_hint=sorting_mode_hint,
        projection_mode_hint=projection_mode_hint,
    )
    return NamedUSDStage(filename="gaussians.usdc", stage=stage)


def _create_rotation_matrix_x(degrees: float) -> np.ndarray:
    """Rotation matrix for +degrees about the X axis (column-vector convention)."""
    rad = np.radians(degrees)
    cos, sin = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float64)


def _rotation_matrix_to_gf_matrix4d(rotation: np.ndarray) -> Gf.Matrix4d:
    """Convert a column-vector rotation matrix to USD's Gf.Matrix4d."""
    r = rotation[:3, :3].astype(np.float64)
    matrix = Gf.Matrix4d()
    matrix.SetTransform(Gf.Matrix3d(*r.T.flatten()), Gf.Vec3d(0.0, 0.0, 0.0))
    return matrix


def _get_isaac_scene_alignment_matrix() -> Gf.Matrix4d:
    """Rotate ecef2enu-normalized Z-up content -90° about X for Isaac Sim's Y-up stage."""
    return _rotation_matrix_to_gf_matrix4d(_create_rotation_matrix_x(-90))


def _write_mesh_prim(
    stage: Usd.Stage,
    vertices: np.ndarray,
    faces: np.ndarray,
    prim_path: str,
) -> UsdGeom.Mesh:
    """
    Write a triangle mesh directly onto ``stage`` at ``prim_path``.

    Args:
        stage: USD stage to author the mesh on.
        vertices: Mesh vertex positions, shape (V, 3).
        faces: Triangle face indices, shape (F, 3).
        prim_path: Absolute prim path for the mesh (e.g. ``/World/ambulance/mesh``).

    Returns:
        The authored ``UsdGeom.Mesh``.
    """
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    return mesh


def _build_mesh_payload_stage(
    vertices: np.ndarray,
    faces: np.ndarray,
    mesh_prim_path: str,
) -> Usd.Stage:
    """
    Create a USD stage containing a single triangle mesh payload.

    Args:
        vertices: Mesh vertex positions, shape (V, 3).
        faces: Triangle face indices, shape (F, 3).
        mesh_prim_path: Absolute prim path for the mesh (e.g. ``/mesh`` in a payload).

    Returns:
        In-memory stage with one ``UsdGeom.Mesh`` at ``mesh_prim_path``.
    """
    stage = _initialize_payload_stage(Path(mesh_prim_path).name)
    _write_mesh_prim(stage, vertices, faces, mesh_prim_path)
    return stage


def _define_asset_xform(
    stage: Usd.Stage,
    asset_path: str,
    matrix: Optional[Gf.Matrix4d],
) -> UsdGeom.Xform:
    """
    Create an asset root xform and optionally set its transform op.

    Args:
        stage: Root ``default.usda`` stage.
        asset_path: Path for the asset grouping xform (e.g. ``/World/ambulance``).
        matrix: Optional alignment transform (ecef2enu upright rotation).

    Returns:
        The asset root ``UsdGeom.Xform``.
    """
    asset_xform = UsdGeom.Xform.Define(stage, asset_path)
    if matrix is not None:
        asset_xform.AddTransformOp().Set(matrix)
    return asset_xform


def _compose_isaac_scene_usdz(
    out_path: Path,
    model: Optional[GaussianSplat3d],
    mesh_vertices: Optional[np.ndarray],
    mesh_faces: Optional[np.ndarray],
    *,
    apply_ecef2enu_rotation: bool,
    linear_srgb: bool,
    sorting_mode_hint: str,
    projection_mode_hint: str,
    asset_name: Optional[str] = None,
) -> None:
    """
    Package mesh and/or splats into one Isaac-ready USDZ under a single asset xform.

    Hierarchy (e.g. output ``ambulance.usdz``)::

        /World/ambulance          (asset xform; optional ecef2enu rotation)
            gaussians               (ParticleField3DGaussianSplat)
            mesh                    (collision mesh, when provided)

    Args:
        out_path (Path): Output ``.usdz`` path.
        model (GaussianSplat3d | None): Optional Gaussian splat model.
        mesh_vertices (np.ndarray | None): Optional mesh vertex positions.
        mesh_faces (np.ndarray | None): Optional mesh face indices; required when ``mesh_vertices`` is set.
        apply_ecef2enu_rotation (bool): Apply -90° X upright rotation on the asset xform.
        linear_srgb (bool): Color space flag for ParticleField3DGaussianSplat export.
        sorting_mode_hint: ParticleField3DGaussianSplat sorting hint.
        projection_mode_hint: ParticleField3DGaussianSplat projection hint.
        asset_name (str | None): Optional override for the asset name under ``/World``.
            Defaults to the output file's stem.

    Returns:
        None
    """
    if model is None and mesh_vertices is None:
        raise ValueError("At least one of model or mesh_vertices must be provided")
    if mesh_vertices is not None and mesh_faces is None:
        raise ValueError("mesh_faces is required when mesh_vertices is provided")

    mesh_payload_path = MESH_PAYLOAD_PRIM
    asset_name = _usd_asset_name_from_path(out_path, asset_name)
    asset_path, gaussians_scene_path, mesh_scene_path = _asset_scene_paths(asset_name)

    payload_stages: list[NamedUSDStage] = []
    root_stage = _initialize_asset_root_stage(asset_name)

    # Payload .usdc files are packed into the USDZ after references are authored;
    # suppress expected "could not open asset" warnings during in-memory composition.
    _ = UsdUtils.CoalescingDiagnosticDelegate()

    scene_matrix = _get_isaac_scene_alignment_matrix() if apply_ecef2enu_rotation else None
    _define_asset_xform(root_stage, asset_path, scene_matrix)
    if scene_matrix is not None:
        logger.info("Applied Isaac asset alignment (-90° X) on %s", asset_path)

    if model is not None:
        gaussians_stage = _build_particlefield3d_gaussians_payload(
            model,
            linear_srgb=linear_srgb,
            sorting_mode_hint=sorting_mode_hint,
            projection_mode_hint=projection_mode_hint,
        )
        payload_stages.append(gaussians_stage)
        gaussians_ref = root_stage.OverridePrim(gaussians_scene_path)
        gaussians_ref.GetReferences().AddReference(gaussians_stage.filename, GAUSSIANS_PAYLOAD_PRIM)
        logger.info("Referenced gaussians payload at %s", gaussians_scene_path)

    has_mesh = mesh_vertices is not None and mesh_faces is not None
    if has_mesh:
        mesh_stage = NamedUSDStage(
            filename="mesh.usdc",
            stage=_build_mesh_payload_stage(mesh_vertices, mesh_faces, mesh_payload_path),
        )
        payload_stages.append(mesh_stage)

        mesh_ref = root_stage.OverridePrim(mesh_scene_path)
        mesh_ref.GetReferences().AddReference(mesh_stage.filename, mesh_payload_path)
        logger.info("Referenced mesh payload at %s", mesh_scene_path)

    default_stage = NamedUSDStage(filename="default.usda", stage=root_stage)
    _write_particlefield3d_usdz(out_path, [default_stage, *payload_stages])
    logger.info("Wrote Isaac scene USDZ to %s", out_path)


def _build_particlefield3d_scene_stage(
    model: Optional[GaussianSplat3d],
    mesh_vertices: Optional[np.ndarray],
    mesh_faces: Optional[np.ndarray],
    *,
    apply_ecef2enu_rotation: bool,
    linear_srgb: bool,
    sorting_mode_hint: str,
    projection_mode_hint: str,
    asset_name: str,
) -> Usd.Stage:
    """
    Build one self-contained stage with gaussians/mesh authored directly (no references), for
    single-file ``.usdc`` export.

    Hierarchy (e.g. asset name ``ambulance``)::

        /World/ambulance          (asset xform; optional ecef2enu rotation)
            gaussians               (ParticleField3DGaussianSplat)
            mesh                    (collision mesh, when provided)

    Args:
        model (GaussianSplat3d | None): Optional Gaussian splat model.
        mesh_vertices (np.ndarray | None): Optional mesh vertex positions.
        mesh_faces (np.ndarray | None): Optional mesh face indices; required when ``mesh_vertices`` is set.
        apply_ecef2enu_rotation (bool): Apply -90° X upright rotation on the asset xform.
        linear_srgb (bool): Color space flag for ParticleField3DGaussianSplat export.
        sorting_mode_hint: ParticleField3DGaussianSplat sorting hint.
        projection_mode_hint: ParticleField3DGaussianSplat projection hint.
        asset_name (str): USD-safe asset name placed under ``/World``.

    Returns:
        Usd.Stage: Self-contained stage with ``/World`` as ``defaultPrim``.
    """
    asset_path, gaussians_scene_path, mesh_scene_path = _asset_scene_paths(asset_name)
    stage = _initialize_asset_root_stage(asset_name)

    scene_matrix = _get_isaac_scene_alignment_matrix() if apply_ecef2enu_rotation else None
    _define_asset_xform(stage, asset_path, scene_matrix)
    if scene_matrix is not None:
        logger.info("Applied Isaac asset alignment (-90° X) on %s", asset_path)

    if model is not None:
        _write_particlefield3d_gaussian_splat(
            stage,
            model,
            gaussians_scene_path,
            linear_srgb=linear_srgb,
            sorting_mode_hint=sorting_mode_hint,
            projection_mode_hint=projection_mode_hint,
        )

    if mesh_vertices is not None and mesh_faces is not None:
        _write_mesh_prim(stage, mesh_vertices, mesh_faces, mesh_scene_path)
        logger.info("Wrote mesh at %s", mesh_scene_path)

    return stage


def _serialize_nurec_usd(
    model_file, positions: np.ndarray, normalizing_transform: np.ndarray = np.eye(4)
) -> NamedUSDStage:
    """
    Create a USD file for the 3DGS model.

    Args:
        model_file: NamedSerialized object containing the compressed msgpack data
        positions: Positions extracted from PLY file for AABB calculation
        normalizing_transform: 4x4 transformation matrix to normalize the scene (defaults to identity)

    Returns:
        NamedUSDStage object containing the USD stage
    """
    logger.info("Creating USD file containing NuRec model")

    # Calculate AABB from positions
    min_coord = np.min(positions, axis=0)
    max_coord = np.max(positions, axis=0)
    logger.info(f"Model bounding box: min={min_coord}, max={max_coord}")

    # Convert numpy values to Python floats
    min_x, min_y, min_z = float(min_coord[0]), float(min_coord[1]), float(min_coord[2])
    max_x, max_y, max_z = float(max_coord[0]), float(max_coord[1]), float(max_coord[2])

    min_list = [min_x, min_y, min_z]
    max_list = [max_x, max_y, max_z]

    # Initialize the USD stage with standard settings
    stage = _initialize_legacy_nurec_usd_stage()

    # Set up render settings
    render_settings = {
        "rtx:rendermode": "RaytracedLighting",
        "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
        "rtx:post:histogram:enabled": False,
        "rtx:post:registeredCompositing:invertToneMap": True,
        "rtx:post:registeredCompositing:invertColorCorrection": True,
        "rtx:material:enableRefraction": False,
        "rtx:post:tonemap:op": 2,
        "rtx:raytracing:fractionalCutoutOpacity": False,
        "rtx:matteObject:visibility:secondaryRays": True,
    }
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", render_settings)

    # Define UsdVol::Volume
    gauss_path = "/World/gauss"
    gauss_volume = UsdVol.Volume.Define(stage, gauss_path)
    gauss_prim = gauss_volume.GetPrim()

    # Apply normalizing transform (identity by default)
    # Default conversion matrix from 3DGRUT to USDZ
    default_conv_tf = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    normalizing_inverse = np.linalg.inv(normalizing_transform)
    corrected_matrix = normalizing_inverse @ default_conv_tf

    # Apply transform directly to the gauss volume
    matrix_op = gauss_volume.AddTransformOp()
    matrix_op.Set(Gf.Matrix4d(*corrected_matrix.flatten()))

    # Define nurec volume properties
    gauss_prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)

    # Enable transform of UsdVol::Volume to take effect
    gauss_prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    # Define field assets and link to volumetric Gaussians prim
    density_field_path = gauss_path + "/density_field"
    density_field = stage.DefinePrim(density_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("density", density_field_path)

    emissive_color_field_path = gauss_path + "/emissive_color_field"
    emissive_color_field = stage.DefinePrim(emissive_color_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("emissiveColor", emissive_color_field_path)

    # Set file paths for field assets
    nurec_relative_path = "./" + model_file.filename
    density_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    density_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("density")
    density_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float")
    density_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("density")

    emissive_color_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    emissive_color_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    emissive_color_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float3")
    emissive_color_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("emissiveColor")

    # Set identity color correction matrix
    emissive_color_field.CreateAttribute("omni:nurec:ccmR", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([1.0, 0.0, 0.0, 0.0])
    )
    emissive_color_field.CreateAttribute("omni:nurec:ccmG", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([0.0, 1.0, 0.0, 0.0])
    )
    emissive_color_field.CreateAttribute("omni:nurec:ccmB", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([0.0, 0.0, 1.0, 0.0])
    )

    # Set extent and crop boundaries
    gauss_prim.GetAttribute("extent").Set([min_list, max_list])

    # Set zero offset
    gauss_offset = [0.0, 0.0, 0.0]
    gauss_prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(gauss_offset))

    # Set crop bounds
    min_vec = Gf.Vec3d(min_x, min_y, min_z)
    max_vec = Gf.Vec3d(max_x, max_y, max_z)
    gauss_prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(min_vec)
    gauss_prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(max_vec)

    # Create empty proxy mesh relationship for forward compatibility
    gauss_prim.CreateRelationship("proxy")

    return NamedUSDStage(filename="gauss.usda", stage=stage)


def update_render_settings(stage: Usd.Stage, referenced_layer: Sdf.Layer) -> None:
    """
    Update render settings from a referenced layer.

    Args:
        stage: The stage to update
        referenced_layer: The layer containing render settings to copy
    """
    if "renderSettings" not in referenced_layer.customLayerData:
        return  # Do nothing if render settings are not present in the referenced layer

    new_render_settings = referenced_layer.customLayerData["renderSettings"]
    current_render_settings = stage.GetRootLayer().customLayerData.get("renderSettings", {})
    if current_render_settings is None:
        current_render_settings = {}

    current_render_settings.update(new_render_settings)
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", current_render_settings)


def serialize_usd_default_layer(gauss_stage: NamedUSDStage) -> NamedUSDStage:
    """
    Create a default USD layer that references the gauss stage.

    Args:
        gauss_stage: The NamedUSDStage object containing the gauss USD stage

    Returns:
        NamedUSDStage: The default USD stage with the gauss reference
    """
    stage = _initialize_legacy_nurec_usd_stage()

    # The delegate captures all errors about dangling references, effectively silencing them.
    _ = UsdUtils.CoalescingDiagnosticDelegate()

    # Create a reference to the gauss stage
    prim = stage.OverridePrim(f"/World/{Path(gauss_stage.filename).stem}")
    # Assume that all reference paths are in the same directory, so that they are also valid relative file paths.
    prim.GetReferences().AddReference(gauss_stage.filename)

    # Copy render settings from the gauss stage's layer
    gauss_layer = gauss_stage.stage.GetRootLayer()
    if "renderSettings" in gauss_layer.customLayerData:
        update_render_settings(stage, gauss_layer)

    # Return as NamedUSDStage
    return NamedUSDStage(filename="default.usda", stage=stage)


def write_to_usdz(file_path: Path, model_file, gauss_usd: NamedUSDStage, default_usd: NamedUSDStage) -> None:
    """
    Write the USDZ file containing the model data and USD stages.

    Args:
        file_path: Path to write the USDZ file to
        model_file: The compressed model data
        gauss_usd: The gauss USD stage
        default_usd: The default USD stage
    """
    # Make sure path to usdz-file exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        # Save default.usda first (required by USDZ spec)
        default_usd.save_to_zip(zip_file)

        # Save the model file and gauss USD stage
        model_file.save_to_zip(zip_file)
        gauss_usd.save_to_zip(zip_file)

    logger.info(f"USDZ file created successfully at {file_path}")


def _serialize_particlefield3d_default_layer(
    gaussians_stage: NamedUSDStage,
    asset_name: str,
) -> NamedUSDStage:
    """
    Create ``default.usda`` that references a gaussians payload layer.

    Args:
        gaussians_stage: Payload stage (typically ``gaussians.usdc``).
        asset_name: USD-safe asset name from the output file stem.

    Returns:
        NamedUSDStage with filename ``default.usda`` and an in-memory stage.
    """
    stage = _initialize_asset_root_stage(asset_name)

    # gaussians.usdc is written to the USDZ after references are authored.
    _ = UsdUtils.CoalescingDiagnosticDelegate()

    asset_path, gaussians_scene_path, _ = _asset_scene_paths(asset_name)
    UsdGeom.Xform.Define(stage, asset_path)
    prim = stage.OverridePrim(gaussians_scene_path)
    prim.GetReferences().AddReference(gaussians_stage.filename, GAUSSIANS_PAYLOAD_PRIM)

    return NamedUSDStage(filename="default.usda", stage=stage)


def _write_particlefield3d_usdz(
    file_path: Path,
    stages: list[NamedUSDStage],
    extra_files: Optional[list["NamedSerialized"]] = None,
) -> None:
    """
    Write a USDZ archive from in-memory USD stages (``default.usda`` first).

    Args:
        file_path: Output ``.usdz`` path.
        stages: Ordered list of stages to pack (root layer first).
        extra_files: Optional sidecar files (e.g. legacy ``.nurec`` payloads).
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        for stage in stages:
            stage.save_to_zip(zip_file)
        for extra in extra_files or []:
            extra.save_to_zip(zip_file)

    logger.info(f"USDZ file created successfully at {file_path}")


def _resize_sh_coefficients(shN: np.ndarray, target_sh_degree: int) -> np.ndarray:
    """
    Pad (with zeros) or truncate the directional SH coefficients so the exported model has
    exactly ``(target_sh_degree + 1) ** 2 - 1`` coefficients per channel.

    Args:
        shN (np.ndarray): Directional SH coefficients of shape ``(N, K - 1, C)`` where ``K`` is the
            number of SH bases in the source model and ``C`` is the number of channels.
        target_sh_degree (int): Desired SH degree for the exported model.

    Returns:
        np.ndarray: SH coefficients of shape ``(N, (target_sh_degree + 1) ** 2 - 1, C)``. Padded
            coefficients are zero (which leaves the rendered radiance unchanged).

    Raises:
        ValueError: If ``target_sh_degree`` is negative or ``shN`` is not a 3D ``(N, K-1, C)`` array.
    """
    if target_sh_degree < 0:
        raise ValueError(f"target_sh_degree must be non-negative, but got {target_sh_degree}")
    if shN.ndim != 3:
        raise ValueError(f"shN must be a 3D array of shape (N, K-1, C), but got shape {shN.shape}")
    target_coeffs = (target_sh_degree + 1) ** 2 - 1
    num_gaussians, current_coeffs, num_channels = shN.shape
    if current_coeffs == target_coeffs:
        return shN
    if current_coeffs > target_coeffs:
        return shN[:, :target_coeffs, :]
    padding = np.zeros((num_gaussians, target_coeffs - current_coeffs, num_channels), dtype=shN.dtype)
    return np.concatenate([shN, padding], axis=1)


def build_legacy_gaussians_payload(
    model: GaussianSplat3d,
    archive_stem: str,
    target_sh_degree: Optional[int] = None,
) -> tuple[NamedUSDStage, "NamedSerialized"]:
    """
    Build ``gauss.usda`` and ``.nurec`` payload layers for legacy NuRec USDZ export.

    Args:
        model (GaussianSplat3d): Gaussian splat model to serialize.
        archive_stem (str): Base filename stem for ``{stem}.nurec`` and referenced layers.
        target_sh_degree (int | None): SH degree to normalize the exported model to. This sets the
            number of directional SH coefficients written to ``features_specular`` and the reported
            ``n_active_features`` (coefficients are zero-padded or truncated to match); the NuRec
            ``radiance_sph_degree`` buffer capacity is left at 3 regardless. The legacy NuRec importer
            (Isaac Sim prior to 6.0) only supports SH degree 0 or 3 and silently fails to import
            models with an intermediate number of coefficients, so an explicit value must be ``0`` or
            ``3``. If ``None`` (the default), degree 0 and 3 models are exported unchanged while any
            other degree is normalized to degree 3 (degrees 1-2 are zero-padded up, and any degree
            above 3 is truncated). See issue #124.

    Returns:
        Tuple of (gauss USD stage, compressed NuRec model file).

    Raises:
        ValueError: If ``target_sh_degree`` is not one of ``None``, ``0``, or ``3``.
    """
    means = model.means.cpu().numpy()
    quats = model.quats.cpu().numpy()
    log_scales = model.log_scales.cpu().numpy()
    logit_opacities = model.logit_opacities.cpu().numpy()
    sh0 = model.sh0.cpu().numpy()
    shN = model.shN.cpu().numpy()

    # Normalize the directional SH to a degree the legacy NuRec importer supports. By default any
    # degree other than 0 or 3 is normalized to degree 3 (degrees 1-2, which silently fail to import
    # in Isaac Sim < 6.0, are zero-padded up; any degree above 3 is truncated), leaving the
    # already-supported degree 0 and 3 exports untouched. See issue #124.
    if target_sh_degree is None:
        target_sh_degree = model.sh_degree if model.sh_degree in (0, 3) else 3
    elif target_sh_degree not in (0, 3):
        raise ValueError(
            "target_sh_degree must be 0 or 3 (the SH degrees the legacy NuRec importer supports), "
            f"but got {target_sh_degree}"
        )
    shN = _resize_sh_coefficients(shN, target_sh_degree)
    n_sh_coeffs = (target_sh_degree + 1) ** 2

    usdz_params = {
        "positions": means,
        "rotations": quats,
        "scales": log_scales,
        "densities": logit_opacities,
        "features_albedo": sh0,
        "features_specular": shN,
        "n_active_features": n_sh_coeffs,
        "density_kernel_degree": 2,
        "density_activation": "sigmoid",
        "scale_activation": "exp",
        "rotation_activation": "normalize",
        "density_kernel_density_clamping": True,
        "density_kernel_min_response": 0.0113,
        # NuRec is only validated with a degree-3 radiance buffer, and degree-0 exports (which carry
        # no directional coefficients) reuse that same buffer, so this is always 3.
        "radiance_sph_degree": 3,
        "transmittance_threshold": 0.0001,
        "global_z_order": True,
        "n_rolling_shutter_iterations": 5,
        "ut_alpha": 1.0,
        "ut_beta": 2.0,
        "ut_kappa": 0.0,
        "ut_require_all_sigma_points": False,
        "image_margin_factor": 0.1,
        "rect_bounding": True,
        "tight_opacity_bounding": True,
        "tile_based_culling": True,
        "k_buffer_size": 0,
    }

    template = fill_3dgut_template(**usdz_params)

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
        packed = msgpack.packb(template)
        f.write(packed)  # type: ignore

    model_file = NamedSerialized(filename=f"{archive_stem}.nurec", serialized=buffer.getvalue())
    gauss_stage = _serialize_nurec_usd(model_file, means, np.eye(4))
    return gauss_stage, model_file


@dataclass(kw_only=True)
class NamedSerialized:
    """
    Class to store serialized data with a filename.
    """

    filename: str
    serialized: Union[str, bytes]

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        """
        Save the serialized data to a zip file.

        Args:
            zip_file: Zip file to save the data to
        """
        zip_file.writestr(self.filename, self.serialized)


def _fill_state_dict_tensors(
    template: dict[str, Any],
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    features_albedo: np.ndarray,
    features_specular: np.ndarray,
    n_active_features: int,
    dtype=np.float16,
) -> None:
    """
    Helper function to fill the state dict tensors in a template.

    Args:
        template: Template dictionary to fill
        positions: Gaussian positions (N, 3)
        rotations: Gaussian rotations (N, 4)
        scales: Gaussian scales (N, 3)
        densities: Gaussian densities (N, 1)
        features_albedo: Gaussian albedo features (N, 3)
        features_specular: Gaussian specular features (N, M)
        n_active_features: Active SH degree
        dtype: Data type to convert to (default: np.float16)
    """
    # Convert data to specified format for efficiency
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.positions"] = positions.astype(dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.rotations"] = rotations.astype(dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.scales"] = scales.astype(dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.densities"] = densities.astype(dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_albedo"] = features_albedo.astype(
        dtype
    ).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_specular"] = features_specular.astype(
        dtype
    ).tobytes()

    # Create empty extra_signal tensor
    extra_signal = np.zeros((positions.shape[0], 0), dtype=dtype)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.extra_signal"] = extra_signal.tobytes()

    # Store n_active_features as binary data (64-bit integer)
    n_active_features_binary = np.array([n_active_features], dtype=np.int64).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.n_active_features"] = n_active_features_binary

    # Store shapes
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.positions.shape"] = list(positions.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.rotations.shape"] = list(rotations.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.scales.shape"] = list(scales.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.densities.shape"] = list(densities.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_albedo.shape"] = list(features_albedo.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_specular.shape"] = list(
        features_specular.shape
    )
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.extra_signal.shape"] = list(extra_signal.shape)
    # Empty array for scalar value
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.n_active_features.shape"] = []


def fill_3dgut_template(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    features_albedo: np.ndarray,
    features_specular: np.ndarray,
    n_active_features: int,
    density_activation: str = "sigmoid",
    scale_activation: str = "exp",
    rotation_activation: str = "normalize",
    density_kernel_degree: int = 2,
    density_kernel_density_clamping: bool = False,
    density_kernel_min_response: float = 0.0113,
    radiance_sph_degree: int = 3,
    transmittance_threshold: float = 0.001,
    global_z_order: bool = False,
    n_rolling_shutter_iterations: int = 5,
    ut_alpha: float = 1.0,
    ut_beta: float = 2.0,
    ut_kappa: float = 0.0,
    ut_require_all_sigma_points: bool = False,
    image_margin_factor: float = 0.1,
    rect_bounding: bool = True,
    tight_opacity_bounding: bool = True,
    tile_based_culling: bool = True,
    k_buffer_size: int = 0,
) -> dict[str, Any]:
    """
    Create and fill the 3DGUT JSON template with gaussian data.

    Args:
        positions: Gaussian positions (N, 3)
        rotations: Gaussian rotations (N, 4)
        scales: Gaussian scales (N, 3)
        densities: Gaussian densities (N, 1)
        features_albedo: Gaussian albedo features (N, 3)
        features_specular: Gaussian specular features (N, M)
        n_active_features: Active SH degree

        Render parameters interfaced between 3DGRUT and NuRec:

        density_kernel_degree: Kernel degree for density computation
        density_activation: Activation function for density
        scale_activation: Activation function for scale
        rotation_activation: Activation function for rotation
        density_kernel_density_clamping: Whether to clamp density kernel
        density_kernel_min_response: Minimum response for density kernel
        radiance_sph_degree: SH degree for radiance
        transmittance_threshold: Threshold for transmittance (min_transmittance in 3DGRUT)

        3DGUT-specific splatting parameters:

        global_z_order: Whether to use global z-order
        n_rolling_shutter_iterations: Number of rolling shutter iterations
        ut_alpha: Alpha parameter for unscented transform
        ut_beta: Beta parameter for unscented transform
        ut_kappa: Kappa parameter for unscented transform
        ut_require_all_sigma_points: Whether to require all sigma points
        image_margin_factor: Image margin factor (ut_in_image_margin_factor in 3DGRUT)
        rect_bounding: Whether to use rectangular bounding
        tight_opacity_bounding: Whether to use tight opacity bounding
        tile_based_culling: Whether to use tile-based culling
        k_buffer_size: Size of the k-buffer

    Returns:
        Dictionary with the filled 3DGUT template
    """
    template = {
        "nre_data": {
            "version": "0.2.576",
            "model": "nre",
            "config": {
                "layers": {
                    "gaussians": {
                        "name": "sh-gaussians",
                        "device": "cuda",
                        "density_activation": density_activation,
                        "scale_activation": scale_activation,
                        "rotation_activation": rotation_activation,
                        "precision": 16,
                        "particle": {
                            "density_kernel_planar": False,  # TODO: Does this have an equivalent in 3DGRUT?
                            "density_kernel_degree": density_kernel_degree,
                            "density_kernel_density_clamping": density_kernel_density_clamping,
                            "density_kernel_min_response": density_kernel_min_response,
                            "radiance_sph_degree": radiance_sph_degree,
                        },
                        "transmittance_threshold": transmittance_threshold,
                    }
                },
                "renderer": {
                    "name": "3dgut-nrend",
                    "log_level": 3,
                    "force_update": False,
                    "update_step_train_batch_end": False,
                    "per_ray_features": False,
                    "global_z_order": global_z_order,
                    "projection": {
                        "n_rolling_shutter_iterations": n_rolling_shutter_iterations,
                        "ut_dim": 3,  # TODO: Does this have an equivalent in 3DGRUT?
                        "ut_alpha": ut_alpha,
                        "ut_beta": ut_beta,
                        "ut_kappa": ut_kappa,
                        "ut_require_all_sigma_points": ut_require_all_sigma_points,
                        "image_margin_factor": image_margin_factor,
                        "min_projected_ray_radius": 0.5477225575051661,
                    },
                    "culling": {
                        "rect_bounding": rect_bounding,
                        "tight_opacity_bounding": tight_opacity_bounding,
                        "tile_based": tile_based_culling,
                        "near_clip_distance": 0.2,  # TODO: Does this have an equivalent in 3DGRUT?
                        # TODO: Does this have an equivalent in 3DGRUT?
                        "far_clip_distance": 3.402823466e38,
                    },
                    "render": {"mode": "kbuffer", "k_buffer_size": k_buffer_size},
                },
                "name": "gaussians_primitive",
                "appearance_embedding": {
                    "name": "skip-appearance",
                    "embedding_dim": 0,
                    "device": "cuda",
                },
                "background": {
                    "name": "skip-background",
                    "device": "cuda",
                    "composite_in_linear_space": False,
                },
            },
            "state_dict": {
                "._extra_state": {"obj_track_ids": {"gaussians": []}},
                ".gaussians_nodes.gaussians.positions": None,
                ".gaussians_nodes.gaussians.rotations": None,
                ".gaussians_nodes.gaussians.scales": None,
                ".gaussians_nodes.gaussians.densities": None,
                ".gaussians_nodes.gaussians.extra_signal": None,
                ".gaussians_nodes.gaussians.features_albedo": None,
                ".gaussians_nodes.gaussians.features_specular": None,
                ".gaussians_nodes.gaussians.n_active_features": None,
                # Shapes
                ".gaussians_nodes.gaussians.positions.shape": None,
                ".gaussians_nodes.gaussians.rotations.shape": None,
                ".gaussians_nodes.gaussians.scales.shape": None,
                ".gaussians_nodes.gaussians.densities.shape": None,
                ".gaussians_nodes.gaussians.extra_signal.shape": None,
                ".gaussians_nodes.gaussians.features_albedo.shape": None,
                ".gaussians_nodes.gaussians.features_specular.shape": None,
                ".gaussians_nodes.gaussians.n_active_features.shape": None,
            },
        }
    }

    # Fill in the state dict tensors
    _fill_state_dict_tensors(
        template,
        positions,
        rotations,
        scales,
        densities,
        features_albedo,
        features_specular,
        n_active_features,
    )

    return template


def _export_splats_to_usdz_legacy(
    model: GaussianSplat3d, out_path: Union[str, Path], target_sh_degree: Optional[int] = None
) -> None:
    """
    Export an :class:`fvdb.GaussianSplat3d` model to a USDZ file using the legacy NuRec format (UsdVol.Volume + .nurec msgpack).

    Args:
        model (fvdb.GaussianSplat3d): The Gaussian Splat model to save to a usdz file
        out_path (str | Path): The output path for the usdz file. If the file extension is not ``.usdz``,
            it will be added. *e.g.*, ``./scene`` will save to ``./scene.usdz``.
        target_sh_degree (int | None): SH degree to write into the exported model (see
            :func:`build_legacy_gaussians_payload`). ``None`` (default) normalizes any non-0/3 degree to 3.
    """
    if isinstance(out_path, str):
        out_path = Path(out_path)
    out_path = out_path.with_suffix(".usdz")

    gauss_usd, model_file = build_legacy_gaussians_payload(model, out_path.stem, target_sh_degree=target_sh_degree)
    default_usd = serialize_usd_default_layer(gauss_usd)
    write_to_usdz(out_path, model_file, gauss_usd, default_usd)


def _export_splats_to_usdz_particlefield3d(
    model: GaussianSplat3d,
    out_path: Path,
    linear_srgb: bool = False,
    sorting_mode_hint: str = DEFAULT_SORTING_MODE_HINT,
    projection_mode_hint: str = DEFAULT_PROJECTION_MODE_HINT,
    asset_name: Optional[str] = None,
) -> None:
    """
    Export a :class:`fvdb.GaussianSplat3d` to USDZ using the ParticleField3DGaussianSplat schema.

    Args:
        model (GaussianSplat3d): Gaussian splat model to export.
        out_path (Path): Output ``.usdz`` path.
        linear_srgb (bool): Color space flag for radiance SH coefficients.
        sorting_mode_hint: ParticleField3DGaussianSplat sorting hint.
        projection_mode_hint: ParticleField3DGaussianSplat projection hint.
        asset_name (str | None): Optional override for the asset name under ``/World``.
            Defaults to the output file's stem.
    """
    logger.info("Creating USD file with ParticleField3DGaussianSplat schema")
    logger.info("Using post-activation gaussian attributes")

    asset_name = _usd_asset_name_from_path(out_path, asset_name)
    gaussians_stage = _build_particlefield3d_gaussians_payload(
        model,
        linear_srgb=linear_srgb,
        sorting_mode_hint=sorting_mode_hint,
        projection_mode_hint=projection_mode_hint,
    )
    default_stage = _serialize_particlefield3d_default_layer(gaussians_stage, asset_name)
    _write_particlefield3d_usdz(out_path, [default_stage, gaussians_stage])


@torch.no_grad()
def export_splats_to_usd(
    model: Optional[GaussianSplat3d],
    out_path: Union[str, Path],
    *,
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_faces: Optional[np.ndarray] = None,
    apply_ecef2enu_rotation: bool = False,
    legacy: bool = False,
    usdz: bool = False,
    linear_srgb: bool = False,
    sorting_mode_hint: str = DEFAULT_SORTING_MODE_HINT,
    projection_mode_hint: str = DEFAULT_PROJECTION_MODE_HINT,
    asset_name: Optional[str] = None,
    target_sh_degree: Optional[int] = None,
) -> Path:
    """
    Export a :class:`fvdb.GaussianSplat3d` (and optional collision mesh) to a USD file.

    By default, exports a single self-contained ``.usdc`` file. Pass ``usdz=True`` to instead
    package the export as a ``.usdz`` archive (required for the legacy NuRec format, which
    references an external ``.nurec`` sidecar file).

    When ``mesh_vertices`` / ``mesh_faces`` or ``apply_ecef2enu_rotation`` are set, the export packages
    splats and mesh under ``/World/<asset>/`` for Isaac Sim (ParticleField3DGaussianSplat + ``UsdGeom.Mesh``).
    Otherwise exports splats only (ParticleField3DGaussianSplat by default, or legacy NuRec with ``legacy=True``).

    Args:
        model (GaussianSplat3d | None): The Gaussian splat model to export. Required unless exporting mesh-only.
        out_path (str | Path): The output path for the USD file. Its extension is replaced with
            ``.usdz`` (if ``usdz=True``) or ``.usdc`` (default). *e.g.*, ``./scene`` will save to
            ``./scene.usdc``.
        mesh_vertices (np.ndarray | None): Optional mesh vertex positions; packages under ``/World/<asset_name>/mesh``
            (same asset xform as splats).
        mesh_faces (np.ndarray | None): Optional mesh face indices. Required when ``mesh_vertices`` is set.
        apply_ecef2enu_rotation (bool): When True, apply the -90° X upright rotation on
            ``/World/<asset_name>``. Splats-only or with mesh.
        legacy (bool): If True, export using the legacy NuRec format (isaac sim versions prior to 6.0)
            (UsdVol.Volume + .nurec msgpack). If False (default), export using the
            OpenUSD ParticleField3DGaussianSplat schema. Incompatible with mesh export and
            ``apply_ecef2enu_rotation``. Requires ``usdz=True``.
        usdz (bool): If True, package the export as a ``.usdz`` archive instead of a single ``.usdc`` file.
        linear_srgb (bool): ParticleField3DGaussianSplat export only. Sets ``ColorSpaceAPI`` to
            ``lin_rec709_scene`` when True, else ``srgb_rec709_display`` (matches 3dgrut).
            fvdb trains against ``image / 255`` (gamma-encoded sRGB), so ``False`` (default)
            matches training; use ``True`` only if your training pipeline optimizes in linear space.
        sorting_mode_hint (str): ParticleField3DGaussianSplat sorting hint (default: ``cameraDistance``).
        projection_mode_hint (str): ParticleField3DGaussianSplat projection hint (default: ``perspective``).
        asset_name (str | None): Optional override for the asset name placed under ``/World``
            (e.g. ``/World/<asset_name>``). Defaults to the output file's stem. Ignored when
            ``legacy=True``, which uses a fixed prim name.
        target_sh_degree (int | None): Legacy export only. SH degree to write into the exported
            model; an explicit value must be ``0`` or ``3`` (the degrees the legacy NuRec importer
            in Isaac Sim prior to 6.0 supports). Directional SH coefficients are zero-padded or
            truncated to match. ``None`` (the default) normalizes any non-0/3 degree to 3. Ignored for
            the ParticleField3DGaussianSplat export, which always writes the model's native SH
            degree. See issue #124.

    Returns:
        Path: The final output path (with the resolved ``.usdc``/``.usdz`` extension).
    """
    if isinstance(out_path, str):
        out_path = Path(out_path)
    out_path = out_path.with_suffix(".usdz" if usdz else ".usdc")

    if legacy and not usdz:
        raise ValueError("legacy export requires usdz=True (the legacy NuRec format is only packaged as .usdz)")
    if legacy and mesh_vertices is not None:
        raise ValueError("legacy export does not support mesh export")
    if legacy and apply_ecef2enu_rotation:
        raise ValueError("legacy export does not support apply_ecef2enu_rotation")
    if model is None and mesh_vertices is None:
        raise ValueError("A Gaussian Splat model, mesh (vertices and faces), or both must be provided")
    if mesh_vertices is not None and mesh_faces is None:
        raise ValueError("mesh_faces is required when mesh_vertices is provided")

    if not usdz:
        resolved_asset_name = _usd_asset_name_from_path(out_path, asset_name)
        stage = _build_particlefield3d_scene_stage(
            model,
            mesh_vertices,
            mesh_faces,
            apply_ecef2enu_rotation=apply_ecef2enu_rotation,
            linear_srgb=linear_srgb,
            sorting_mode_hint=sorting_mode_hint,
            projection_mode_hint=projection_mode_hint,
            asset_name=resolved_asset_name,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(out_path))
        logger.info(f"Wrote USD scene to {out_path}")
        return out_path

    if mesh_vertices is not None or apply_ecef2enu_rotation:
        _compose_isaac_scene_usdz(
            out_path,
            model,
            mesh_vertices,
            mesh_faces,
            apply_ecef2enu_rotation=apply_ecef2enu_rotation,
            linear_srgb=linear_srgb,
            sorting_mode_hint=sorting_mode_hint,
            projection_mode_hint=projection_mode_hint,
            asset_name=asset_name,
        )
        return out_path

    if model is None:
        raise ValueError("model is required for splats-only export")

    if legacy:
        _export_splats_to_usdz_legacy(model, out_path, target_sh_degree=target_sh_degree)
    else:
        _export_splats_to_usdz_particlefield3d(
            model,
            out_path,
            linear_srgb=linear_srgb,
            sorting_mode_hint=sorting_mode_hint,
            projection_mode_hint=projection_mode_hint,
            asset_name=asset_name,
        )
    return out_path
