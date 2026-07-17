# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Portable GARfVDB artifact I/O."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import tempfile
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import fvdb
import torch
from safetensors.torch import load_file, save_file

from fvdb_reality_capture.radiance_fields.gaussian_splatting import GaussianSplat3d

from .config import GARfVDBConfig
from .model import GARfVDBModel

if TYPE_CHECKING:
    from .garfvdb import GARfVDB


ARTIFACT_SCHEMA = "fvdb_reality_capture.garfvdb"
ARTIFACT_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
ENCODER_NAME = "encoder.nvdb"
NETWORK_NAME = "network.safetensors"
CARRIER_NAME = "carrier.ply"


class GARfVDBArtifactError(ValueError):
    """Raised when a GARfVDB artifact is missing, corrupt, or incompatible."""


class GARfVDBArtifactVersionError(GARfVDBArtifactError):
    """Raised when no reader is available for a bundle's schema version."""


def is_garfvdb_bundle(path: str | pathlib.Path) -> bool:
    """Return whether *path* looks like a GARfVDB bundle without loading payloads."""
    bundle_path = pathlib.Path(path)
    manifest_path = bundle_path / MANIFEST_NAME
    if not bundle_path.is_dir() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("schema") == ARTIFACT_SCHEMA


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _grid_names(grid_count: int) -> list[str]:
    return [f"encoder_{index:02d}" for index in range(grid_count)]


def _require_bundle_suffix(path: pathlib.Path) -> None:
    if path.suffix != ".garfvdb":
        raise ValueError(f"GARfVDB bundle path must end in '.garfvdb': {path}")


def save_garfvdb_bundle(product: GARfVDB, path: str | pathlib.Path) -> pathlib.Path:
    """Save a portable, pickle-free GARfVDB inference bundle."""
    output_path = pathlib.Path(path)
    _require_bundle_suffix(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output GARfVDB bundle already exists: {output_path}")
    if not product.model.model_config.use_grid:
        raise ValueError("Portable GARfVDB artifacts require GARfVDBConfig.use_grid=True")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = pathlib.Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    try:
        encoder_path = temporary_path / ENCODER_NAME
        network_path = temporary_path / NETWORK_NAME
        carrier_path = temporary_path / CARRIER_NAME

        grid = product.model.encoder_gridbatch
        features = grid.jagged_like(product.model.encoder_gridbatch_features_data.detach())
        names = _grid_names(grid.grid_count)
        grid.save_nanovdb(
            str(encoder_path),
            data=features,
            names=names,
            compressed=True,
        )

        save_file(product.model.network_state_dict(), network_path)
        product.carrier.save_ply(str(carrier_path), product.reconstruction_metadata)

        voxel_counts = grid.num_voxels.detach().cpu().tolist()
        manifest: dict[str, Any] = {
            "schema": ARTIFACT_SCHEMA,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "method": "garfvdb",
            "model_config": asdict(product.model.model_config),
            "encoder": {
                "file": ENCODER_NAME,
                "grid_names": names,
                "grid_count": grid.grid_count,
                "total_voxels": grid.total_voxels,
                "voxel_counts": voxel_counts,
                "voxel_sizes": grid.voxel_sizes.detach().cpu().tolist(),
                "origins": grid.origins.detach().cpu().tolist(),
                "feature_dim": product.model.model_config.grid_feature_dim,
                "feature_dtype": str(product.model.encoder_gridbatch_features_data.dtype),
            },
            "network": {"file": NETWORK_NAME},
            "carrier": {
                "file": CARRIER_NAME,
                "num_gaussians": product.carrier.num_gaussians,
            },
            "versions": {
                "fvdb_reality_capture": _package_version("fvdb_reality_capture"),
                "fvdb_core": _package_version("fvdb-core"),
                "safetensors": _package_version("safetensors"),
            },
            "checksums": {
                ENCODER_NAME: _sha256(encoder_path),
                NETWORK_NAME: _sha256(network_path),
                CARRIER_NAME: _sha256(carrier_path),
            },
        }
        (temporary_path / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    return output_path


def _load_manifest(path: pathlib.Path) -> dict[str, Any]:
    manifest_path = path / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GARfVDBArtifactError(f"GARfVDB manifest is missing: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise GARfVDBArtifactError(f"GARfVDB manifest is not valid JSON: {manifest_path}") from exc

    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise GARfVDBArtifactError(
            f"Unsupported artifact schema {manifest.get('schema')!r}; expected {ARTIFACT_SCHEMA!r}."
        )
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
        raise GARfVDBArtifactVersionError(
            f"GARfVDB manifest has an invalid or missing schema_version: {schema_version!r}."
        )
    return manifest


def _payload_path(bundle: pathlib.Path, manifest: dict[str, Any], section: str) -> pathlib.Path:
    try:
        filename = manifest[section]["file"]
    except (KeyError, TypeError) as exc:
        raise GARfVDBArtifactError(f"Manifest section {section!r} does not define a payload file.") from exc
    if not isinstance(filename, str) or pathlib.Path(filename).name != filename:
        raise GARfVDBArtifactError(f"Unsafe payload filename in manifest: {filename!r}")
    path = bundle / filename
    if not path.is_file():
        raise GARfVDBArtifactError(f"GARfVDB payload is missing: {path}")
    expected_checksum = manifest.get("checksums", {}).get(filename)
    if not isinstance(expected_checksum, str) or _sha256(path) != expected_checksum:
        raise GARfVDBArtifactError(f"GARfVDB payload checksum does not match: {path}")
    return path


def _load_garfvdb_bundle_v1(
    bundle_path: pathlib.Path,
    manifest: dict[str, Any],
    device: str | torch.device,
) -> GARfVDB:
    """Read and validate schema version 1 payloads."""
    from .garfvdb import GARfVDB

    encoder_path = _payload_path(bundle_path, manifest, "encoder")
    network_path = _payload_path(bundle_path, manifest, "network")
    carrier_path = _payload_path(bundle_path, manifest, "carrier")

    model_config = GARfVDBConfig(**manifest.get("model_config", {}))
    if not model_config.use_grid:
        raise GARfVDBArtifactError("Portable GARfVDB artifacts require model_config.use_grid=True")

    grid, features, names = fvdb.functional.load_nanovdb(str(encoder_path), device=device)
    encoder_manifest = manifest.get("encoder", {})
    expected_names = encoder_manifest.get("grid_names")
    canonical_names = _grid_names(model_config.num_grids)
    if expected_names != canonical_names:
        raise GARfVDBArtifactError(f"Manifest grid names are not canonical: {expected_names!r} != {canonical_names!r}")
    if names != canonical_names:
        raise GARfVDBArtifactError(f"NanoVDB grid names are not canonical: {names!r} != {canonical_names!r}")
    if grid.grid_count != encoder_manifest.get("grid_count") or grid.grid_count != model_config.num_grids:
        raise GARfVDBArtifactError("NanoVDB grid count does not match the manifest/model configuration")
    if grid.total_voxels != encoder_manifest.get("total_voxels"):
        raise GARfVDBArtifactError("NanoVDB total voxel count does not match the manifest")
    if grid.num_voxels.detach().cpu().tolist() != encoder_manifest.get("voxel_counts"):
        raise GARfVDBArtifactError("NanoVDB per-grid voxel counts do not match the manifest")
    if encoder_manifest.get("feature_dim") != model_config.grid_feature_dim:
        raise GARfVDBArtifactError("NanoVDB feature dimension does not match the model configuration")
    if features.jdata.shape != (grid.total_voxels, model_config.grid_feature_dim):
        raise GARfVDBArtifactError(
            "NanoVDB feature data has the wrong shape: "
            f"expected {(grid.total_voxels, model_config.grid_feature_dim)}, got {tuple(features.jdata.shape)}"
        )
    if str(features.dtype) != encoder_manifest.get("feature_dtype"):
        raise GARfVDBArtifactError("NanoVDB feature dtype does not match the manifest")
    expected_voxel_sizes = torch.as_tensor(encoder_manifest.get("voxel_sizes"), dtype=grid.voxel_sizes.dtype)
    expected_origins = torch.as_tensor(encoder_manifest.get("origins"), dtype=grid.origins.dtype)
    if not torch.allclose(grid.voxel_sizes.cpu(), expected_voxel_sizes) or not torch.allclose(
        grid.origins.cpu(), expected_origins
    ):
        raise GARfVDBArtifactError("NanoVDB transforms do not match the manifest")

    carrier, reconstruction_metadata = GaussianSplat3d.from_ply(carrier_path, device)
    if carrier.num_gaussians != manifest.get("carrier", {}).get("num_gaussians"):
        raise GARfVDBArtifactError("Carrier Gaussian count does not match the manifest")

    network_state = load_file(network_path, device=str(torch.device(device)))
    model = GARfVDBModel.from_artifact_components(
        gs_model=carrier,
        model_config=model_config,
        encoder_gridbatch=grid,
        encoder_features=features.jdata,
        network_state=network_state,
        device=device,
    )
    return GARfVDB(model=model, reconstruction_metadata=reconstruction_metadata)


_ARTIFACT_READERS = {
    1: _load_garfvdb_bundle_v1,
}
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = tuple(sorted(_ARTIFACT_READERS))
if ARTIFACT_SCHEMA_VERSION not in _ARTIFACT_READERS:
    raise RuntimeError("The current GARfVDB artifact schema has no registered reader")


def load_garfvdb_bundle(path: str | pathlib.Path, device: str | torch.device = "cuda:0") -> GARfVDB:
    """Load a portable GARfVDB bundle using its version-specific reader."""
    bundle_path = pathlib.Path(path)
    _require_bundle_suffix(bundle_path)
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"GARfVDB bundle does not exist: {bundle_path}")
    manifest = _load_manifest(bundle_path)
    schema_version = manifest["schema_version"]
    reader = _ARTIFACT_READERS.get(schema_version)
    if reader is None:
        supported = ", ".join(str(value) for value in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS)
        if schema_version > ARTIFACT_SCHEMA_VERSION:
            detail = "The bundle was written by a newer version of fvdb-reality-capture."
        else:
            detail = "This release no longer has a reader for that older bundle version."
        raise GARfVDBArtifactVersionError(
            f"Unsupported GARfVDB schema_version {schema_version}; supported versions: {supported}. {detail}"
        )
    return reader(bundle_path, manifest, device)


__all__ = [
    "ARTIFACT_SCHEMA",
    "ARTIFACT_SCHEMA_VERSION",
    "GARfVDBArtifactError",
    "GARfVDBArtifactVersionError",
    "SUPPORTED_ARTIFACT_SCHEMA_VERSIONS",
    "is_garfvdb_bundle",
    "load_garfvdb_bundle",
    "save_garfvdb_bundle",
]
