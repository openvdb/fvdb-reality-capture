# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import torch
import tyro
from fvdb.viz import Viewer
from tyro.conf import arg

import fvdb_reality_capture
from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.transforms import (
    Compose,
    FilterImagesWithLowPoints,
    NormalizeScene,
    PercentileFilterPoints,
)

from ._common import DatasetType, load_sfm_scene


def center_and_scale_scene(
    sfm_scene: fvdb_reality_capture.sfm_scene.SfmScene, scale: float
) -> fvdb_reality_capture.sfm_scene.SfmScene:
    """
    Center and scale the scene by the given scale factor so that the median of the points is at the origin.

    Args:
        sfm_scene (SfmScene): The input SfM scene
        scale (float): The scale factor to apply to the scene
    """
    centroid = np.median(sfm_scene.points, axis=0)
    transform = np.diag([scale, scale, scale, 1.0])
    transform[0:3, 3] = -centroid * scale
    return sfm_scene.apply_transformation_matrix(transform)


@dataclass
class ShowData(BaseCommand):
    """
    Visualize a scene in a dataset folder. This will plot the scene's cameras and point cloud in an interactive viewer
    shown in a browser window.

    The dataset folder should either contain a colmap dataset, a set of e57 files, a simple_directory dataset:

    COLMAP Data format: A folder should containining:
        - cameras.txt
        - images.txt
        - points3D.txt
        - A folder named "images" containing the image files.
        - An optional "masks" folder with the same layout as images containing masks of which pixels are valid.

    E57 format: A folder containing one or more .e57 files.

    Simple Directory format: A folder containing:
        - images/ A directory of images (jpg, png, etc).
        - An optional "masks/" folder with the same layout as images containing masks of which pixels are valid.
        - A cameras.json file containing camera intrinsics and extrinsics for each image. It should be a list of objects
            with the following format:{
                "camera_name": "camera_0000",
                "width": 2048,
                "height": 2048,
                "camera_intrinsics": [], # 3x3 matrix in row-major order
                "world_to_camera": [], # 4x4 matrix in row-major order
                "image_path": "name_of_image_file_relative_to_images_folder"

    Example usage:

        # Visualize a Colmap dataset in the folder ./colmap_dataset
        frgs show-data ./colmap_dataset

        # Visualize an e57 dataset in the folder ./e57_dataset
        frgs show-data ./e57_dataset --dataset-type e57

        # Visualize a simple directory dataset in the folder ./simple_directory_dataset
        frgs show-data ./simple_directory_dataset --dataset-type simple_directory
    """

    # Path to the dataset folder.
    dataset_path: tyro.conf.Positional[pathlib.Path]

    # The port to expose the viewer server on.
    viewer_port: Annotated[int, arg(aliases=["-p"])] = 8888

    # If True, then the viewer will log verbosely.
    verbose: Annotated[bool, arg(aliases=["-v"])] = False

    # Percentile filter for points. Points with any coordinate below this percentile or above (100 - this percentile)
    # will be removed from the point cloud. This can help remove outliers. Set to 0.0 to disable.
    points_percentile_filter: Annotated[float, arg(aliases=["-ppf"])] = 0.0

    # Minimum number of points a camera must observe to be included in the viewer.
    min_points_per_image: Annotated[int, arg(aliases=["-mpi"])] = 5

    # Type of dataset to load.
    dataset_type: Annotated[DatasetType, arg(aliases=["-dt"])] = "colmap"

    @torch.no_grad()
    def execute(self) -> None:

        logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
        logger = logging.getLogger(__name__)

        viewer = Viewer(port=self.viewer_port, verbose=self.verbose)

        sfm_scene = load_sfm_scene(self.dataset_path, self.dataset_type)
        sfm_scene = Compose(
            NormalizeScene("pca"),
            PercentileFilterPoints([self.points_percentile_filter] * 3, [100.0 - self.points_percentile_filter] * 3),
            FilterImagesWithLowPoints(min_num_points=self.min_points_per_image),
        )(sfm_scene)

        cam_positions = sfm_scene.camera_to_world_matrices[:, 0:3, 3]
        cam_extent = cam_positions.max(0) - cam_positions.min(0)
        cam_diagonal = float(np.linalg.norm(cam_extent))
        points_extent = sfm_scene.points.max(0) - sfm_scene.points.min(0)
        points_diagnonal = float(np.linalg.norm(points_extent))

        axis_scale = 0.01 * min(cam_diagonal, points_diagnonal)

        # Find a camera whose position is far from the scene centroid and
        # whose up vector is not aligned with the view direction.
        cam_eye = cam_positions[0]
        cam_lookat = cam_positions.mean(0)
        cam_view_direction = cam_lookat - cam_eye
        cam_eye += cam_view_direction * 0.5
        cam_up = np.array([0.0, 0.0, 1.0])
        if np.allclose(cam_eye - cam_lookat, cam_up):
            cam_up = np.array([0.0, 1.0, 0.0])

        viewer.set_camera_lookat(eye=cam_eye, center=cam_lookat, up=cam_up)

        viewer.add_camera_view(
            name="cameras",
            camera_to_world_matrices=sfm_scene.camera_to_world_matrices,
            projection_matrices=torch.from_numpy(sfm_scene.projection_matrices),
            image_sizes=torch.from_numpy(sfm_scene.image_sizes),
            frustum_line_width=2.0,
            frustum_scale=3.0 * axis_scale,
            axis_length=2.0 * axis_scale,
            axis_thickness=0.1 * axis_scale,
        )

        viewer.add_point_cloud(
            name="points",
            points=sfm_scene.points,
            colors=sfm_scene.points_rgb.astype(np.float32) / 255.0,
            point_sizes=1.0,
        )

        viewer.show()

        logger.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
