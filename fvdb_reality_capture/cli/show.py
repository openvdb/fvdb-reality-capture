# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from fvdb.viz import Viewer

from .base_command import BaseCommand
from .common import load_splats_from_file


@dataclass
class Show(BaseCommand):
    """
    Visualize a scene in a saved PLY or checkpoint file.
    """

    # Path to the input PLY or checkpoint file. Must end in .ply, .pt, or .pth.
    input_path: tyro.conf.Positional[pathlib.Path]

    # The port to expose the viewer server on.
    viewer_port: int = 8888

    # If True, then the viewer will log verbosely.
    verbose: bool = False

    # Device to use for computation (default is "cuda").
    device: str | torch.device = "cuda"

    @torch.no_grad()
    def execute(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
        logger = logging.getLogger(__name__)

        logger.info(f"Starting viewer server on port {self.viewer_port}")
        viewer = Viewer(port=self.viewer_port, verbose=self.verbose)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file {self.input_path} does not exist.")

        logger.info(f"Loading Gaussian Splats from {self.input_path}")

        # Load a PLY or a checkpoint file and metadata.
        # The metadata may contain camera information (if it was a PLY saved during training).
        # If so, we will add the camera views to the viewer.
        model, metadata = load_splats_from_file(self.input_path, self.device)

        logger.info(f"Loaded {model.num_gaussians} Gaussians.")

        # Check if the loaded metadata has camera information.
        # If so, we will use it to set the initial camera position and add camera views.
        has_camera_to_world_matrices = "camera_to_world_matrices" in metadata and isinstance(
            metadata["camera_to_world_matrices"], torch.Tensor
        )
        has_projection_matrices = "projection_matrices" in metadata and isinstance(
            metadata["projection_matrices"], torch.Tensor
        )

        # If we have camera information, use it to set the initial camera position, looking at the scene centroid
        # and positioned at the position of first camera. # Otherwise, just position at half the scene radius
        # away from the centroid along the (1, 1, 1) diagonal.
        scene_centroid = model.means.mean(dim=0).cpu().numpy()
        if not has_camera_to_world_matrices:
            scene_radius = (model.means.max(dim=0).values - model.means.min(dim=0).values).max().item() / 2.0
            initial_camera_position = scene_centroid + np.ones(3) * scene_radius * 0.5
        else:
            if not isinstance(metadata["camera_to_world_matrices"], (torch.Tensor, np.ndarray)):
                raise ValueError("camera_to_world_matrices in metadata must be a torch.Tensor or numpy.ndarray")
            initial_camera_position = metadata["camera_to_world_matrices"][0, :3, 3]

        logger.info(f"Setting viewer camera to {initial_camera_position} looking at {scene_centroid}")
        viewer.set_camera_lookat(
            eye=initial_camera_position,
            center=scene_centroid,
            up=[0, 0, -1],
        )

        if has_camera_to_world_matrices and has_projection_matrices:
            assert isinstance(metadata["camera_to_world_matrices"], (torch.Tensor, np.ndarray))
            assert isinstance(metadata["projection_matrices"], (torch.Tensor, np.ndarray))
            image_sizes = metadata.get("image_sizes", None)
            assert isinstance(image_sizes, (torch.Tensor, np.ndarray))
            if isinstance(image_sizes, np.ndarray):
                image_sizes = torch.from_numpy(image_sizes)
            if isinstance(metadata["camera_to_world_matrices"], np.ndarray):
                metadata["camera_to_world_matrices"] = torch.from_numpy(metadata["camera_to_world_matrices"])
            if isinstance(metadata["projection_matrices"], np.ndarray):
                metadata["projection_matrices"] = torch.from_numpy(metadata["projection_matrices"])

            viewer.add_camera_view(
                "training cameras",
                metadata["camera_to_world_matrices"].cpu(),
                metadata["projection_matrices"].cpu(),
                image_sizes,
            )
        else:
            logger.info("No camera information found in metadata, not adding camera views to viewer")

        viewer.add_gaussian_splat_3d(
            "gaussian splats",
            model,
        )
        logger.info("Viewer running... Ctrl+C to exit.")
        viewer.show()
        time.sleep(1000000)
