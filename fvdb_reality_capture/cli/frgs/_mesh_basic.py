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
from fvdb_reality_capture.tools import mesh_from_splats

from ._common import (
    NearFarUnits,
    load_camera_metadata,
    load_splats_from_file,
    near_far_for_units,
)


@dataclass
class MeshBasic(BaseCommand):
    """
    Extract a triangle mesh from a saved Gaussian splat file with TSDF fusion using depth maps rendered from the Gaussian splat model.

    The algorithm proceeds in three steps:

    1. First, it renders depth and color/feature images from the Gaussian splat radiance field at each of the specified
       camera views.

    2. Second, it integrates the depths and colors/features into a sparse fvdb.Grid in a narrow band
       around the surface using sparse truncated signed distance field (TSDF) fusion.
       The result is a sparse voxel grid representation of the scene where each voxel stores a signed distance
       value and color (or other features).

    3. Third, it extracts a mesh using the sparse marching cubes algorithm implemented in fvdb.Grid.marching_cubes
       over the Grid and TSDF values. This step produces a triangle mesh with vertex colors sampled from the
       colors/features stored in the Grid.


    Example usage:

        # Extract a mesh from a Gaussian splat model saved in `model.pt` with a truncation margin of 0.05
        frgs mesh-basic model.pt 0.05 --output-path mesh.ply

        # Extract a mesh from a Gaussian splat model saved in `model.ply` with a truncation margin of 0.1
        # with a grid shell thickness of 5 voxels, near plane at 0.1x median depth, far plane at 2.0x median depth
        # of each images.
        frgs mesh-basic model.ply 0.1 --output-path mesh.ply --grid-shell-thickness 5.0 --near 0.1 --far 2.0
    """

    # Path to the input PLY or checkpoint file. Must end in .ply, .pt, or .pth.
    input_path: tyro.conf.Positional[pathlib.Path]

    # Truncation margin for TSDF volume. This is the distance (in world units)
    # that the TSDF values are truncated to.
    truncation_margin: tyro.conf.Positional[float]

    # The number of voxels along each axis to include in the TSDF volume.
    # This defines the resolution of the narrow band around the surface.
    grid_shell_thickness: Annotated[float, arg(aliases=["-g"])] = 3.0

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
    output_path: Annotated[pathlib.Path, arg(aliases=["-o"])] = pathlib.Path("mesh.ply")

    # Device to use for computation (default is "cuda").
    device: Annotated[str, arg(aliases=["-d"])] = "cuda"

    """
    Extract a mesh from a Gaussian Splat reconstruction.

    """

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
            f"Extracting mesh from splats using near/far units {self.near_far_units} and image downsample factor {self.image_downsample_factor}..."
        )
        v, f, c = mesh_from_splats(
            model=model,
            camera_to_world_matrices=camera_to_world_matrices,
            projection_matrices=projection_matrices,
            image_sizes=image_sizes,
            truncation_margin=self.truncation_margin,
            grid_shell_thickness=self.grid_shell_thickness,
            near=near,
            far=far,
            alpha_threshold=self.alpha_threshold,
            image_downsample_factor=self.image_downsample_factor,
            show_progress=True,
        )

        v, f, c = v.to(torch.float32).cpu().numpy(), f.cpu().numpy(), c.to(torch.float32).cpu().numpy()
        logger.info(f"Extracted mesh with {v.shape[0]} vertices and {f.shape[0]} faces.")

        logger.info(f"Saving mesh to {self.output_path}")
        pcu.save_mesh_vfc(str(self.output_path), v, f, c)
        logger.info("Mesh saved successfully.")
