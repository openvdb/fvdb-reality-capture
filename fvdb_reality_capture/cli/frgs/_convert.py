# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib
from dataclasses import dataclass

import point_cloud_utils as pcu
import torch
import tyro
from fvdb import GaussianSplat3d

from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.radiance_fields import GaussianSplatReconstruction
from fvdb_reality_capture.tools import export_splats_to_usd


@dataclass
class Convert(BaseCommand):
    """
    Convert a Gaussian Splat in one format to another. Currently the following conversions are supported:
        - PLY to USD (.usdc single file, or .usdz archive)
        - Checkpoint to USD (.usdc single file, or .usdz archive)
        - PLY to PLY (copy)
        - Checkpoint to PLY (export)
        - PLY or checkpoint to USD with optional mesh and ecef2enu upright rotation


    Example usage:

        # Convert a PLY file to a single-file USDC
        frgs convert input.ply output.usdc

        # Convert a Checkpoint file to a single-file USDC
        frgs convert input.pt output.usdc

        # Convert a PLY file to a USDZ archive instead
        frgs convert input.ply output.usdc --usdz

        # Splats only, rotated for ecef2enu-normalized scenes
        frgs convert input.ply output.usdc --ecef2enu-rotation

        # Splats + mesh for Isaac Sim, rotated for ecef2enu-normalized scenes
        frgs convert input.ply output.usdc --mesh-path mesh.ply --ecef2enu-rotation

        # Legacy NuRec USDZ for Isaac Sim 5.x
        frgs convert input.ply output.usdc --legacy --usdz

        # Custom asset prim name (/World/my_asset) instead of the output file name
        frgs convert input.ply output.usdc --prim-path my_asset

    """

    # Path to the input file. Must be a .ply file or Checkpoint (.pt or .pth) file.
    in_path: tyro.conf.Positional[pathlib.Path]

    # Path to the output file. Must be a .ply, .usdc, or .usdz file.
    out_path: tyro.conf.Positional[pathlib.Path]

    # USD export only. Optional mesh (PLY/OBJ) under /World/<output_file_name>/mesh (shared asset xform).
    mesh_path: pathlib.Path | None = None

    # USD export only. Apply -90° X upright rotation on /World/<output_file_name> for ecef2enu-normalized scenes.
    ecef2enu_rotation: bool = False

    # USD export only. Export legacy NuRec format (UsdVol.Volume + .nurec) for Isaac Sim 5.x. Requires --usdz.
    legacy: bool = False

    # USD export only. Name of the asset prim placed under /World (i.e. /World/<prim-path>). Defaults to the output file name.
    prim_path: str | None = None

    # USD export only. Package the export as a .usdz archive instead of a single .usdc file.
    # Implied if the output path already ends in .usdz.
    usdz: bool = False

    # Legacy export only (--legacy). SH degree to write into the exported model; must be 0 or 3
    # (the degrees the Isaac Sim < 6.0 NuRec importer supports). Directional SH coefficients are
    # zero-padded or truncated to match. Leaving this unset normalizes any non-0/3 degree to 3.
    # Ignored by the default ParticleField3DGaussianSplat export, which writes the native SH degree.
    target_sh_degree: int | None = None

    @torch.no_grad()
    def execute(self) -> None:
        valid_input_types = (".ply", ".pt", ".pth")
        valid_output_types = (".usdc", ".usdz", ".ply")
        valid_conversions = {
            ".ply": [".usdc", ".usdz", ".ply"],
            ".pt": [".usdc", ".usdz", ".ply"],
            ".pth": [".usdc", ".usdz", ".ply"],
        }
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        in_file_type = self.in_path.suffix.lower()
        out_file_type = self.out_path.suffix.lower()
        usdz = self.usdz or out_file_type == ".usdz"

        if in_file_type not in valid_input_types:
            raise ValueError(f"Input file type {in_file_type} is not supported. Must be one of {valid_input_types}")
        if out_file_type not in valid_output_types:
            raise ValueError(f"Output file type {out_file_type} is not supported. Must be one of {valid_output_types}")

        if out_file_type not in valid_conversions[in_file_type]:
            raise ValueError(
                f"Conversion from {in_file_type} to {out_file_type} is not supported. "
                f"Supported output types for {in_file_type} are: {valid_conversions[in_file_type]}"
            )
        if self.mesh_path is not None and out_file_type == ".ply":
            raise ValueError("--mesh-path is only supported for USD export (output file must end in .usdc or .usdz)")
        if self.ecef2enu_rotation and out_file_type == ".ply":
            raise ValueError(
                "--ecef2enu-rotation is only supported for USD export (output file must end in .usdc or .usdz)"
            )
        if self.legacy and out_file_type == ".ply":
            raise ValueError("--legacy is only supported for USD export (output file must end in .usdc or .usdz)")
        if self.legacy and self.mesh_path is not None:
            raise ValueError("--legacy cannot be used with --mesh-path")
        if self.legacy and self.ecef2enu_rotation:
            raise ValueError("--legacy cannot be used with --ecef2enu-rotation")
        if self.legacy and not usdz:
            raise ValueError("--legacy requires --usdz (the legacy NuRec format is only packaged as .usdz)")
        if self.prim_path is not None and out_file_type == ".ply":
            raise ValueError("--prim-path is only supported for USD export (output file must end in .usdc or .usdz)")
        if self.legacy and self.prim_path is not None:
            raise ValueError("--legacy cannot be used with --prim-path")
        if self.target_sh_degree is not None and not self.legacy:
            raise ValueError("--target-sh-degree only applies to legacy export (use with --legacy)")
        if self.target_sh_degree is not None and self.target_sh_degree not in (0, 3):
            raise ValueError("--target-sh-degree must be 0 or 3 (the SH degrees the legacy NuRec importer supports)")

        if in_file_type == ".ply":
            model, metadata = GaussianSplat3d.from_ply(self.in_path)
            logger.info(f"Loaded Gaussian Splat model with {model.num_gaussians} splats from {self.in_path}")
        elif in_file_type in (".pt", ".pth"):
            checkpoint = torch.load(self.in_path, map_location="cpu", weights_only=False)
            runner = GaussianSplatReconstruction.from_state_dict(checkpoint)
            model = runner.model
            metadata = runner.reconstruction_metadata
            logger.info(f"Loaded Gaussian Splat model with {model.num_gaussians} splats from {self.in_path}")

        mesh_vertices = None
        mesh_faces = None
        if self.mesh_path is not None:
            if not self.mesh_path.is_file():
                raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")
            vertices, faces = pcu.load_mesh_vf(str(self.mesh_path))
            mesh_vertices = vertices.astype("float32")
            mesh_faces = faces.astype("int32")
            logger.info(
                "Loaded mesh with %d vertices and %d faces from %s",
                mesh_vertices.shape[0],
                mesh_faces.shape[0],
                self.mesh_path,
            )

        if out_file_type == ".ply":
            model.save_ply(self.out_path, metadata=metadata)
            logger.info(f"Saved Gaussian Splat model with {model.num_gaussians} splats to {self.out_path}")
        else:
            written_path = export_splats_to_usd(
                model,
                self.out_path,
                mesh_vertices=mesh_vertices,
                mesh_faces=mesh_faces,
                apply_ecef2enu_rotation=self.ecef2enu_rotation,
                legacy=self.legacy,
                usdz=usdz,
                asset_name=self.prim_path,
                target_sh_degree=self.target_sh_degree,
            )
            if self.legacy:
                logger.info(
                    "Exported legacy NuRec Gaussian Splat model with %d splats to %s",
                    model.num_gaussians,
                    written_path,
                )
            elif mesh_vertices is not None:
                logger.info(
                    "Exported Gaussian Splat model with %d splats and mesh to %s",
                    model.num_gaussians,
                    written_path,
                )
            elif self.ecef2enu_rotation:
                logger.info(
                    "Exported Gaussian Splat model with %d splats and ecef2enu rotation to %s",
                    model.num_gaussians,
                    written_path,
                )
            else:
                logger.info(f"Exported Gaussian Splat model with {model.num_gaussians} splats to {written_path}")
