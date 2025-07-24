# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib

import point_cloud_utils as pcu
import torch
import tyro
from checkpoint import Checkpoint, extract_point_cloud_from_checkpoint


def main(
    checkpoint_path: pathlib.Path,
    near: float = 0.1,
    far: float = 4.0,
    depth_image_downsample_factor: int = 8,
    output_path: pathlib.Path = pathlib.Path("point_cloud.ply"),
    device: str = "cuda",
):
    """
    Extract a mesh from a saved checkpoint file.

    Args:
        checkpoint_path (pathlib.Path): Path to the checkpoint file containing the Gaussian splat model.
        near (float): Near plane distance below which we'll ignore depth samples (default is 0.1).
        far (float): Far plane distance above which we'll ignore depth samples.
            Units are in multiples of the scene scale (variance in distance from camera positions around their mean).
        output_path (pathlib.Path): Path to save the extracted mesh (default is "mesh.ply").
        device (str): Device to use for computation (default is "cuda").
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    logger = logging.getLogger("extract_point_cloud")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = Checkpoint.load(checkpoint_path)

    if checkpoint.train_dataset is None:
        raise ValueError(
            "Checkpoint does not contain a training dataset. Cannot extract point cloud. "
            "Please provide a checkpoint that includes training dataset information."
        )
    far = far * checkpoint.train_dataset.scene_scale

    logger.info(
        f"Extracting point cloud from checkpoint using near={near:0.3f}, far={far:0.3f}, downsample factor={depth_image_downsample_factor}"
    )
    positions, colors = extract_point_cloud_from_checkpoint(
        checkpoint=checkpoint,
        near=near,
        far=far,  # Use the scene scale from the training dataset
        depth_image_downsample_factor=depth_image_downsample_factor,
        device=device,
        show_progress=True,
    )

    logger.info(f"Extracted {positions.shape[0]} points with colors.")
    positions, colors = positions.to(torch.float32).cpu().numpy(), colors.to(torch.float32).cpu().numpy()

    logger.info(f"Saving point cloud to {output_path}")
    pcu.save_mesh_vc(str(output_path), positions, colors)
    logger.info("Point cloud saved successfully.")


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
