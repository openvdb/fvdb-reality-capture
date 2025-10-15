# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib
from dataclasses import dataclass
from typing import Annotated, Literal

import point_cloud_utils as pcu
import torch
import tyro
from fvdb.types import to_Mat33fBatch, to_Mat44fBatch, to_Vec2iBatch, to_VecNf
from tyro.conf import arg

from fvdb_reality_capture.tools import mesh_from_splats

from ._common import BaseCommand, load_splats_from_file

NearFarUnits = Literal["absolute", "camera_extent", "median_depth"]


@dataclass
class MeshBasic(BaseCommand):
    """
    Extract a triangle mesh from a saved Gaussian splat file with TSDF fusion using depth maps rendered from the Gaussian splat model.

    In short, this algorithm works by rendering images and depth maps from multiple views of the Gaussian splat model,
    and then integrating these depth maps and images into a sparse `fvdb.Grid` in a narrow band around the surface using a weighted averaging scheme.
    The algorithm returns this grid along with signed distance values and colors (or other features) at each voxel.

    The algorithm then extracts a mesh using the marching cubes algorithm implemented in `fvdb.marching_cubes.marching_cubes`
    over the Grid and TSDF values.

    The TSDF fusion algorithm is a method for integrating multiple depth maps into a single volumetric representation of a scene encoding a
    truncated signed distance field (_i.e._ a signed distance field in a narrow band around the surface). TSDF fusion was first described in the paper
    "KinectFusion: Real-Time Dense Surface Mapping and Tracking"
    (https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf).
    We use a modified version of this algorithm which only allocates voxels in a narrow band around the surface of the model
    to reduce memory usage and speed up computation.
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

        if "camera_to_world_matrices" not in metadata:
            raise ValueError("Gaussian splats file must contain 'camera_to_world_matrices'")

        if "projection_matrices" not in metadata:
            raise ValueError("Gaussian splats file must contain 'projection_matrices'")

        if "image_sizes" not in metadata:
            raise ValueError("Gaussian splats file must contain 'image_sizes'")

        camera_to_world_matrices = to_Mat44fBatch(metadata["camera_to_world_matrices"]).to(self.device)
        projection_matrices = to_Mat33fBatch(metadata["projection_matrices"]).to(self.device)
        image_sizes = to_Vec2iBatch(metadata["image_sizes"])

        if self.near_far_units == "median_depth":
            if "median_depths" not in metadata:
                raise ValueError(
                    "Gaussian splats file must contain 'median_depths' to use 'median_depth' near/far units"
                )
            median_depths = to_VecNf(metadata["median_depths"], camera_to_world_matrices.shape[0])
            if torch.any(median_depths.isnan()) or torch.any(median_depths <= 0.0):
                raise ValueError("median_depths in metadata must be positive and non-NaN")
            near = self.near * median_depths
            far = self.far * median_depths
        elif self.near_far_units == "camera_extent":
            scene_centroid = camera_to_world_matrices[:, :3, 3].mean(dim=0)
            max_camera_distance = torch.linalg.norm(
                camera_to_world_matrices[:, :3, 3] - scene_centroid[None, :], dim=1
            ).max()
            near = self.near * max_camera_distance
            far = self.far * max_camera_distance
        elif self.near_far_units == "absolute":
            near = self.near
            far = self.far
        else:
            raise ValueError(f"Invalid near_far_units: {self.near_far_units}")

        model = model.to(self.device)

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
