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
    truncation_margin: float,
    near: float = 0.1,
    far: float = 4.0,
    output_path: pathlib.Path = pathlib.Path("mesh.ply"),
    device: str = "cuda",
):
    """
    Extract a mesh from a saved checkpoint file.

    Args:
        checkpoint_path (pathlib.Path): Path to the checkpoint file containing the Gaussian splat model.
        truncation_margin (float): Margin for truncating the mesh, in world units.
        near (float): Near plane distance below which we'll ignore depth samples (default is 0.1).
        far (float): Far plane distance above which we'll ignore depth samples.
            Units are in multiples of the scene scale (variance in distance from camera positions around their mean).
        output_path (pathlib.Path): Path to save the extracted mesh (default is "mesh.ply").
        device (str): Device to use for computation (default is "cuda").
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    logger = logging.getLogger("extract_mesh")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = Checkpoint.load(checkpoint_path)

    if checkpoint.train_dataset is None:
        raise ValueError(
            "Checkpoint does not contain a training dataset. Cannot extract point cloud. "
            "Please provide a checkpoint that includes training dataset information."
        )
    far = far * checkpoint.train_dataset.scene_scale

    logger.info(
        f"Extracting mesh from checkpoint using near={near:0.3f}, far={far:0.3f}, and truncation margin={truncation_margin:0.3f}"
    )

    v, f, c = extract_mesh_from_checkpoint(
        checkpoint=checkpoint,
        truncation_margin=truncation_margin,
        near=near,
        far=far,
        device=device,
        show_progress=True,
    )

    logger.info(f"Extracted mesh with {v.shape[0]} vertices and {f.shape[0]} faces.")

    v, f, c = v.to(torch.float32).cpu().numpy(), f.cpu().numpy(), c.to(torch.float32).cpu().numpy()

    logger.info(f"Saving mesh to {output_path}")
    pcu.save_mesh_vfc(str(output_path), v, f, c)
    logger.info("Mesh saved successfully.")


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
