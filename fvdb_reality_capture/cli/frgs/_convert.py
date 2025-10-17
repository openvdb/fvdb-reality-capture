# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib
from dataclasses import dataclass

import torch
import tyro
from fvdb import GaussianSplat3d

from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.radiance_fields import GaussianSplatReconstruction
from fvdb_reality_capture.tools import export_splats_to_usdz


@dataclass
class Convert(BaseCommand):
    """
    Convert a Gaussian Splat in one format to another. Currently the following conversions are supported:
        - PLY to USDZ
        - Checkpoint to USDZ
        - PLY to PLY (copy)
        - Checkpoint to PLY (export)


    Example usage:

        # Convert a PLY file to a USDZ file
        frgs frgs convert input.ply output.usdz

        # Convert a Checkpoint file to a USDZ file
        frgs frgs convert input.pt output.usdz

    """

    # Path to the input file. Must be a .ply file or Checkpoint (.pt or .pth) file.
    in_path: tyro.conf.Positional[pathlib.Path]

    # Path to the output file. Must be a .ply file, Checkpoint (.pt or .pth) file, or .usdz file.
    out_path: tyro.conf.Positional[pathlib.Path]

    @torch.no_grad()
    def execute(self) -> None:
        valid_input_types = (".ply", ".pt", ".pth")
        valid_output_types = (".usdz", ".ply")
        valid_conversions = {
            ".ply": [".usdz", ".ply"],
            ".pt": [".usdz", ".ply"],
            ".pth": [".usdz", ".ply"],
        }
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        in_file_type = self.in_path.suffix.lower()
        out_file_type = self.out_path.suffix.lower()

        if in_file_type not in valid_input_types:
            raise ValueError(f"Input file type {in_file_type} is not supported. Must be one of {valid_input_types}")
        if out_file_type not in valid_output_types:
            raise ValueError(f"Output file type {out_file_type} is not supported. Must be one of {valid_output_types}")

        if out_file_type not in valid_conversions[in_file_type]:
            raise ValueError(
                f"Conversion from {in_file_type} to {out_file_type} is not supported. "
                f"Supported output types for {in_file_type} are: {valid_conversions[in_file_type]}"
            )
        if in_file_type == ".ply":
            model, metadata = GaussianSplat3d.from_ply(self.in_path)
            logger.info(f"Loaded Gaussian Splat model with {model.num_gaussians} splats from {self.in_path}")
        elif in_file_type in (".pt", ".pth"):
            checkpoint = torch.load(self.in_path, map_location="cpu", weights_only=False)
            runner = GaussianSplatReconstruction.from_state_dict(checkpoint)
            model = runner.model
            metadata = runner.reconstruction_metadata
            logger.info(f"Loaded Gaussian Splat model with {model.num_gaussians} splats from {self.in_path}")

        if out_file_type == ".ply":
            model.save_ply(self.out_path, metadata=metadata)
            logger.info(f"Saved Gaussian Splat model with {model.num_gaussians} splats to {self.out_path}")
        elif out_file_type == ".usdz":
            export_splats_to_usdz(model, self.out_path)
            logger.info(f"Exported Gaussian Splat model with {model.num_gaussians} splats to {self.out_path}")
