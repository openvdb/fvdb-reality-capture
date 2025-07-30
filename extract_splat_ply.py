# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib

import point_cloud_utils as pcu
import torch
import tyro
from training import Checkpoint, extract_mesh_from_checkpoint


def main(
    checkpoint_path: pathlib.Path,
    output_path: pathlib.Path = pathlib.Path("splats.ply"),
    device: str = "cuda",
):
    """
    Extract a mesh from a saved checkpoint file.

    Args:
        checkpoint_path (pathlib.Path): Path to the checkpoint file containing the Gaussian splat model.
        truncation_margin (float): Margin for truncating the mesh, in world units.
        near (float): Near plane distance below which to ignore depth samples (default is 0.1).
        far (float): Far plane distance above which to ignore depth samples.
            Units are in multiples of the scene scale (variance in distance from camera positions around their mean).
        output_path (pathlib.Path): Path to save the extracted mesh (default is "mesh.ply").
        device (str): Device to use for computation (default is "cuda").
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    logger = logging.getLogger("extract_splat_ply")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = Checkpoint.load(checkpoint_path)

    logger.info(f"Savling splat PLY from checkpoint")

    checkpoint.splats.to(device).save_ply(
        output_path,
    )
    logger.info("Splat PLY saved successfully.")


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
