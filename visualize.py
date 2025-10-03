# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time

import numpy as np
import torch
import tyro
from fvdb import GaussianSplat3d
from fvdb.viz import Viewer


def main(
    ply_path: pathlib.Path,
    viewer_port: int = 8888,
    verbose: bool = False,
    device: str | torch.device = "cuda",
):
    """
    Visualize a scene in a saved checkpoint file.

    Args:
        ply_path (pathlib.Path): Path to a PLY file containing the Gaussian splat model.
        viewer_port (int): The port to expose the viewer server on
        verbose (bool): If True, then the viewer will log verbosely.
        device (str | torch.device): Device to use for computation (default is "cuda").
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Starting viewer server on port {viewer_port}")
    viewer = Viewer(port=viewer_port, verbose=verbose)

    logger.info(f"Loading Gaussian Splats from {ply_path}")
    model, metadata = GaussianSplat3d.from_ply(ply_path, device)

    logger.info(f"Adding Gaussian Splat model with {len(model.means)} Gaussians to viewer")
    viewer.add_gaussian_splat_3d(name="model", gaussian_splat_3d=model)

    has_camera_to_world_matrices = "camera_to_world_matrices" in metadata and isinstance(
        metadata["camera_to_world_matrices"], torch.Tensor
    )
    has_projection_matrices = "projection_matrices" in metadata and isinstance(
        metadata["projection_matrices"], torch.Tensor
    )

    scene_centroid = model.means.mean(dim=0).cpu().numpy()
    if not has_camera_to_world_matrices:
        scene_radius = (model.means.max(dim=0).values - model.means.min(dim=0).values).max().item() / 2.0
        initial_camera_position = scene_centroid + np.ones(3) * scene_radius * 0.5
    else:
        assert isinstance(metadata["camera_to_world_matrices"], torch.Tensor)
        initial_camera_position = metadata["camera_to_world_matrices"][0, :3, 3]

    logger.info(f"Setting viewer camera to {initial_camera_position} looking at {scene_centroid}")
    viewer.set_camera_lookat(
        camera_origin=initial_camera_position,
        lookat_point=scene_centroid,
        up_direction=[0, 0, 1],
    )

    if has_camera_to_world_matrices and has_projection_matrices:
        assert isinstance(metadata["camera_to_world_matrices"], torch.Tensor)
        assert isinstance(metadata["projection_matrices"], torch.Tensor)
        viewer.add_camera_view(
            "training cameras", metadata["camera_to_world_matrices"].cpu(), metadata["projection_matrices"].cpu()
        )
    else:
        logger.info("No camera information  found in PLY metadata, not adding camera views to viewer")
    logger.info("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
