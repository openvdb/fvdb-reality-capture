# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib

import point_cloud_utils as pcu
import torch
import tyro
from training import (
    Checkpoint,
    extract_mesh_from_checkpoint_dlnr,
    extract_tsdf_from_checkpoint_dlnr,
)


def main(
    checkpoint_path: pathlib.Path,
    truncation_margin: float,
    baseline: float = 0.07,
    near: float = 4.0,
    far: float = 20.0,
    dlnr_backbone: str = "middleburry",
    output_path: pathlib.Path = pathlib.Path("mesh.ply"),
    device: str = "cuda",
):
    """
    Extract a mesh from a saved checkpoint file using TSDF fusion and depth maps estimated using the DLNR model.

    The algorithm renders a stereo pair of images from the Gaussian splat model by adding a small baseline to each camera position.
    (specified by the baseline parameter). It then uses DLNR to estimate depth maps for these stereo pairs which are fed to TSDF fusion.

    The mesh extraction algorithm is based on the paper:
    "GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views"
    (https://arxiv.org/abs/2404.01810)

    Args:
        checkpoint_path (pathlib.Path): Path to the checkpoint file containing the Gaussian splat model.
        truncation_margin (float): Margin for truncating the mesh, in world units.
        baseline (float): Baseline distance (in fraction of the scene scale) for generating stereo pairs as input to (default is 0.07).
            The scene scale is defined as the variance in distance from camera positions around their mean.
        near (float): Near plane distance (as a multiple of the baseline) below which we'll ignore depth samples (default is 4.0).
        far (float): Far plane distance (as a multiple of the baseline) above which we'll ignore depth samples (default is 20.0).
        dlnr_backbone (str): Backbone to use for the DLNR model, either "middleburry" or "sceneflow" (default is "middleburry").
        output_path (pathlib.Path): Path to save the extracted mesh (default is "mesh.ply").
        device (str): Device to use for computation (default is "cuda").
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    logger = logging.getLogger("extract_mesh_dlnr")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = Checkpoint.load(checkpoint_path)

    v, f, c = extract_mesh_from_checkpoint_dlnr(
        checkpoint=checkpoint,
        truncation_margin=truncation_margin,
        baseline=baseline,
        near=near,
        far=far,
        device=device,
        dlnr_backbone=dlnr_backbone,
        show_progress=True,
    )

    v, f, c = v.to(torch.float32).cpu().numpy(), f.cpu().numpy(), c.to(torch.float32).cpu().numpy()
    logger.info(f"Extracted mesh with {v.shape[0]} vertices and {f.shape[0]} faces.")

    logger.info(f"Saving mesh to {output_path}")
    pcu.save_mesh_vfc(str(output_path), v, f, c)
    logger.info("Mesh saved successfully.")


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
