# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
from dataclasses import dataclass
from typing import Annotated

import point_cloud_utils as pcu
import torch
import tyro
from tyro.conf import arg

from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.tools import point_cloud_from_splats

from ._common import (
    NearFarUnits,
    load_camera_metadata,
    load_splats_from_file,
    near_far_for_units,
)


@dataclass
class Points(BaseCommand):
    """
    Extract a point cloud with colors/features from a Gaussian splat file by unprojecting depth images rendered from it.

    This algorithm can optionally filter out points near depth discontinuities using the following heurstic:
        1. Apply a small Gaussian filter to the depth images to reduce noise.
        2. Run a Canny edge detector on the depth immage to find
        depth discontinuities. The result is an image mask where pixels near depth edges are marked.
        3. Dilate the edge mask to remove depth samples near edges.
        4. Remove points from the point cloud where the corresponding depth pixel is marked in the dilated edge mask.


    Example usage:

        # Extract a point cloud from a Gaussian splat model saved in `model.pt`
        frgs points model.pt --output-path points.ply

        # Extract a point cloud from a Gaussian splat model saved in `model.ply`
        frgs points model.ply --output-path points.ply
    """

    input_path: tyro.conf.Positional[pathlib.Path]

    # Near plane distance for which depth values are considered valid.
    # The units depend on the `near_far_units` parameter.
    # By default, this is a multiple of the median depth of each image.
    near: Annotated[float, arg(aliases=["-n"])] = 0.2

    # Far plane distance for which depth values are considered valid.
    # The units depend on the `near_far_units` parameter.
    # By default, this is a multiple of the median depth of each image.
    far: Annotated[float, arg(aliases=["-f"])] = 1.5

    # Alpha threshold to mask pixels where the Gaussian splat model is transparent,
    # usually indicating the background. (default is 0.1).
    alpha_threshold: Annotated[float, arg(aliases=["-at"])] = 0.1

    # Standard deviation for the Gaussian filter applied to the depth image
    # before Canny edge detection (default is 1.0). Set to 0.0 to disable canny edge filtering.
    canny_edge_std: Annotated[float, arg(aliases=["-ces"])] = 1.0

    # Dilation size for the Canny edge mask (default is 5).
    canny_mask_dilation: Annotated[int, arg(aliases=["-cmd"])] = 5

    # Factor by which to downsample the rendered images for depth estimation (default is 1, _i.e._ no downsampling).
    image_downsample_factor: Annotated[int, arg(aliases=["-idf"])] = 1

    # Which units to use for near and far clipping.
    # - "absolute" means the near and far values are in world units.
    # - "camera_extent" means the near and far values are fractions of the maximum distance from any camera to
    #   the centroid of all cameras (this is good for orbit captures).
    # - "median_depth" means the near and far values are fractions of the median depth of each image. This is good for
    #   captures where the cameras are not evenly distributed around the scene.
    # (default is "median_depth").
    near_far_units: Annotated[NearFarUnits, arg(aliases=["-nfu"])] = "median_depth"

    # Path to save the extracted mesh (default is "mesh.ply").
    output_path: Annotated[pathlib.Path, arg(aliases=["-o"])] = pathlib.Path("points.ply")

    # Device to use for computation (default is "cuda").
    device: Annotated[str, arg(aliases=["-d"])] = "cuda"

    def execute(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

        logger = logging.getLogger(__name__)

        logger.info(f"Loading Gaussian splats from from {self.input_path}")

        model, metadata = load_splats_from_file(self.input_path, self.device)

        camera_to_world_matrices, projection_matrices, image_sizes = load_camera_metadata(metadata)

        near, far = near_far_for_units(
            self.near_far_units,
            self.near,
            self.far,
            metadata.get("median_depths", None),
            camera_to_world_matrices,
        )

        model = model.to(self.device)

        logger.info(
            f"Extracting point cloud from splats using near/far units {self.near_far_units} and image downsample factor {self.image_downsample_factor}..."
        )
        positions, colors = point_cloud_from_splats(
            model=model,
            camera_to_world_matrices=camera_to_world_matrices,
            projection_matrices=projection_matrices,
            image_sizes=image_sizes,
            near=near,
            far=far,
            alpha_threshold=self.alpha_threshold,
            canny_edge_std=self.canny_edge_std,
            canny_mask_dilation=self.canny_mask_dilation,
            image_downsample_factor=self.image_downsample_factor,
            show_progress=True,
        )

        logger.info(f"Extracted {positions.shape[0]:,} points with colors.")
        colors = colors.to(torch.float32) / 255.0
        positions, colors = positions.to(torch.float32).cpu().numpy(), colors.cpu().numpy()

        logger.info(f"Saving point cloud to {self.output_path}")
        pcu.save_mesh_vc(str(self.output_path), positions, colors)
        logger.info("Point cloud saved successfully.")
