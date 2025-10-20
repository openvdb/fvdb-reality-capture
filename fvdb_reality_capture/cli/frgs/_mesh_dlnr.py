# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import point_cloud_utils as pcu
import torch
import tyro
from fvdb.types import to_Mat33fBatch, to_Mat44fBatch, to_Vec2iBatch, to_VecNf
from tyro.conf import arg

from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.tools import mesh_from_splats_dlnr

from ._common import load_splats_from_file


@dataclass
class MeshDLNR(BaseCommand):
    """
    Extract a mesh from a saved Gaussian splat file using TSDF fusion and depth maps estimated using the DLNR model.

    1. First, it renders stereo pairs of images from the Gaussian splat radiance field, and uses
       DLNR to compute depth maps from these stereo pairs in the frame of the first image in the pair.
       The result is a set of depth maps aligned with the rendered images.

    2. Second, it integrates the depths and colors/features into a sparse fvdb.Grid in a narrow band
       around the surface using sparse truncated signed distance field (TSDF) fusion.
       The result is a sparse voxel grid representation of the scene where each voxel stores a signed distance
       value and color (or other features).

    3. Third, it extracts a mesh using the sparse marching cubes algorithm implemented in fvdb.Grid.marching_cubes
       over the Grid and TSDF values. This step produces a triangle mesh with vertex colors sampled from the
       colors/features stored in the Grid.


    Example usage:

        # Extract a mesh from a Gaussian splat model saved in `model.pt` with a truncation margin of 0.05
        frgs mesh-dlnr model.pt 0.05 --output-path mesh.ply

        # Extract a mesh from a Gaussian splat model saved in `model.ply` with a truncation margin of 0.1
        # with a grid shell thickness of 5 voxels
        frgs mesh-dlnr model.ply 0.1 --output-path mesh.ply --grid-shell-thickness 5.0
    """

    # Path to the input PLY or checkpoint file. Must end in .ply, .pt, or .pth.
    input_path: tyro.conf.Positional[pathlib.Path]

    # Truncation margin for TSDF volume. This is the distance (in world units)
    # that the TSDF values are truncated to.
    truncation_margin: tyro.conf.Positional[float]

    # The number of voxels along each axis to include in the TSDF volume.
    # This defines the resolution of the narrow band around the surface.
    grid_shell_thickness: Annotated[float, arg(aliases=["-g"])] = 3.0

    # Baseline distance (as a fraction of the mean depth of each image) used
    # for generating stereo pairs as input to the DLNR model (default is 0.07).
    baseline: Annotated[float, arg(aliases=["-b"])] = 0.07

    # Near plane distance (as a multiple of the baseline) for which depth values are considered valid.
    near: Annotated[float, arg(aliases=["-n"])] = 4.0

    # Far plane distance (as a multiple of the baseline) for which depth values are considered valid.
    far: Annotated[float, arg(aliases=["-f"])] = 20.0

    # Alpha threshold to mask pixels where the Gaussian splat model is transparent,
    # usually indicating the background. (default is 0.1).
    alpha_threshold: Annotated[float, arg(aliases=["-at"])] = 0.1

    # Reprojection error threshold for occlusion masking in pixels (default is 3.0).
    disparity_reprojection_threshold: Annotated[float, arg(aliases=["-dt"])] = 3.0

    # Factor by which to downsample the rendered images for depth estimation (default is 1, _i.e._ no downsampling).
    image_downsample_factor: Annotated[int, arg(aliases=["-idf"])] = 1

    # Backbone to use for the DLNR model, either "middleburry" or "sceneflow" (default is "middleburry").
    dlnr_backbone: Annotated[str, arg(aliases=["-db"])] = "middleburry"

    # If True, use the provided baseline as an absolute distance in world units (default is False).
    use_absolute_baseline: Annotated[bool, arg(aliases=["-ab"])] = False

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

        camera_to_world_matrices = to_Mat44fBatch(metadata["camera_to_world_matrices"])
        projection_matrices = to_Mat33fBatch(metadata["projection_matrices"])
        image_sizes = to_Vec2iBatch(metadata["image_sizes"])

        model = model.to(self.device)

        if self.use_absolute_baseline:
            logger.info(f"Extracting mesh using DLNR stereo matching with a baseline of {self.baseline} world units")
        else:
            logger.info(
                f"Extracting mesh using DLNR stereo matching with a baseline of {self.baseline * 100:.1f}% of the mean depth of each image"
            )
        v, f, c = mesh_from_splats_dlnr(
            model=model,
            camera_to_world_matrices=camera_to_world_matrices,
            projection_matrices=projection_matrices,
            image_sizes=image_sizes,
            truncation_margin=self.truncation_margin,
            grid_shell_thickness=self.grid_shell_thickness,
            baseline=self.baseline,
            near=self.near,
            far=self.far,
            disparity_reprojection_threshold=self.disparity_reprojection_threshold,
            alpha_threshold=self.alpha_threshold,
            image_downsample_factor=self.image_downsample_factor,
            dlnr_backbone=self.dlnr_backbone,
            use_absolute_baseline=self.use_absolute_baseline,
            show_progress=True,
        )

        v, f, c = v.to(torch.float32).cpu().numpy(), f.cpu().numpy(), c.to(torch.float32).cpu().numpy()
        logger.info(f"Extracted mesh with {v.shape[0]} vertices and {f.shape[0]} faces.")

        logger.info(f"Saving mesh to {self.output_path}")
        pcu.save_mesh_vfc(str(self.output_path), v, f, c)
        logger.info("Mesh saved successfully.")
