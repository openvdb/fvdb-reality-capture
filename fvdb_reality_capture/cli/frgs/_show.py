# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Annotated

import fvdb.viz as fviz
import numpy as np
import torch
import tyro
from fvdb.types import to_Mat33fBatch, to_Mat44fBatch, to_Vec2fBatch
from tyro.conf import arg

from fvdb_reality_capture.cli import BaseCommand

from ._common import load_splats_from_file


@dataclass
class Show(BaseCommand):
    """
    Visualize a Gaussian splat radiance field in a saved PLY or checkpoint file. This will plot the splats in an interactive viewer
    shown in a browser window.

    # Example usage:

        # Visualize a Gaussian splat model saved in `model.ply`
        frgs show model.ply --viewer-port 8888

        # Visualize a Gaussian splat model saved in `model.pt`
        frgs show model.pt --viewer-port 8888

    """

    # Path to the input PLY or checkpoint file. Must end in .ply, .pt, or .pth.
    input_path: tyro.conf.Positional[pathlib.Path]

    # The port to expose the viewer server on.
    viewer_port: Annotated[int, arg(aliases=["-p"])] = 8080

    # The port to expose the viewer server on.
    viewer_ip_address: Annotated[str, arg(aliases=["-ip"])] = "127.0.0.1"

    # If True, then the viewer will log verbosely.
    verbose: Annotated[bool, arg(aliases=["-v"])] = False

    # Device to use for computation (default is "cuda").
    device: str | torch.device = "cuda"

    @torch.no_grad()
    def execute(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
        logger = logging.getLogger(__name__)

        logger.info(f"Starting viewer server on {self.viewer_ip_address}:{self.viewer_port}")
        fviz.init(ip_address=self.viewer_ip_address, port=self.viewer_port, verbose=self.verbose)
        viz_scene = fviz.get_scene("Gaussian Splat Model Visualization")

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file {self.input_path} does not exist.")

        logger.info(f"Loading Gaussian Splats from {self.input_path}")

        # Load a PLY or a checkpoint file and metadata.
        # The metadata may contain camera information (if it was a PLY saved during optimization with fvdb-reality-capture).
        # If so, we will add the camera views to the viewer.
        model, metadata = load_splats_from_file(self.input_path, self.device)

        logger.info(f"Loaded {model.num_gaussians} Gaussians.")

        # Check if the loaded metadata has camera information.
        # If so, we will use it to set the initial camera position and add camera views.

        cam_to_world_matrices = metadata.get("camera_to_world_matrices", None)
        if cam_to_world_matrices is not None:
            cam_to_world_matrices = to_Mat44fBatch(cam_to_world_matrices).cpu()

        projection_matrices = metadata.get("projection_matrices", None)
        if projection_matrices is not None:
            projection_matrices = to_Mat33fBatch(projection_matrices).cpu()

        image_sizes = metadata.get("image_sizes", None)
        if image_sizes is not None:
            image_sizes = to_Vec2fBatch(image_sizes).cpu()

        # If we have camera information, use it to set the initial camera position, looking at the scene centroid
        # and positioned at the position of first camera. # Otherwise, just position at half the scene radius
        # away from the centroid along the (1, 1, 1) diagonal.
        scene_centroid = model.means.mean(dim=0).cpu().numpy()
        if cam_to_world_matrices is None:
            scene_radius = (model.means.max(dim=0).values - model.means.min(dim=0).values).max().item() / 2.0
            initial_camera_position = scene_centroid + np.ones(3) * scene_radius
        else:
            initial_camera_position = cam_to_world_matrices[0, :3, 3].cpu().numpy()

        logger.info(f"Setting scene camera to {initial_camera_position} looking at {scene_centroid}")
        viz_scene.set_camera_lookat(
            eye=initial_camera_position,
            center=scene_centroid,
            up=[0, 0, -1],
        )

        if cam_to_world_matrices is not None and projection_matrices is not None:
            viz_scene.add_cameras(
                name="Cameras",
                camera_to_world_matrices=cam_to_world_matrices,
                projection_matrices=projection_matrices,
                image_sizes=image_sizes,
            )
        else:
            logger.info("No camera information found in metadata, not adding cameras to the scene")

        viz_scene.add_gaussian_splat_3d(
            "Gaussian Splats",
            model,
        )
        logger.info("Viewer running... Ctrl+C to exit.")
        fviz.show()
        time.sleep(1000000)
