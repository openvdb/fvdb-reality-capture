# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import json
import logging
import pathlib

import numpy as np
import point_cloud_utils as pcu
import tqdm

from .sfm_cache import SfmCache
from .sfm_metadata import SfmCameraMetadata, SfmCameraType, SfmImageMetadata


def load_simple_scene(data_path: pathlib.Path):
    """
    Load cameras, posed-images, and points from a directory of images, camera parameters (stored as JSON),
    and 3d points (stored as a PLY).

    The directory should contain:
        - images/: A directory of images.
        - cameras.json: A JSON file containing camera parameters.
            The cameras.json file is a list of dictionaries, each containing:
                - camera_name: The name of the image file.
                - width: The width of the image.
                - height: The height of the image.
                - camera_intrinsics: The perspective projection matrix
                - world_to_camera: The world-to-camera transformation matrix.
                - image_path: The path to the image file relative to the images directory.
        - points.ply: A PLY file containing 3D points.
    Args:
        data_ (pathlib.Path): The path to the data

    Returns:
        sfm_scene (SfmScene): An in-memory representation of the SfmScene for the output of the COLMAP run.
    """

    cameras_json_path = data_path / "cameras.json"
    if not cameras_json_path.exists():
        raise FileNotFoundError(f"cameras.json not found in {data_path}")
    images_path = data_path / "images"
    if not images_path.exists():
        raise FileNotFoundError(f"images/ directory not found in {data_path}")
    if not images_path.is_dir():
        raise NotADirectoryError(f"images/ is not a directory in {data_path}")
    points_path = data_path / "pointcloud.ply"
    if not points_path.exists():
        raise FileNotFoundError(f"pointcloud.ply not found in {data_path}")

    with open(data_path / "cameras.json", "r") as f:
        cameras = json.load(f)

    camera_metadata = {}
    image_metadata = []

    for i, camera in enumerate(cameras):
        im_path = images_path / camera["image_path"]
        im_path = im_path.absolute()
        if not im_path.exists():
            raise FileNotFoundError(f"Image {im_path} not found in images/ directory in {data_path}")
        if not im_path.is_file():
            raise FileNotFoundError(f"Image {im_path} is not a file in images/ directory in {data_path}")
        width = camera["width"]
        height = camera["height"]
        projection_matrix = np.array(camera["camera_intrinsics"], dtype=np.float32).reshape(3, 3)
        fx, fy, cx, cy = (
            projection_matrix[0, 0],
            projection_matrix[1, 1],
            projection_matrix[0, 2],
            projection_matrix[1, 2],
        )
        camera_type = SfmCameraType.PINHOLE
        camera_id = i + 1
        camera_metadata[camera_id] = SfmCameraMetadata(
            img_width=width,
            img_height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_type=camera_type,
            distortion_parameters=np.array([], dtype=np.float32),
        )

        world_to_camera_matrix = np.array(camera["world_to_camera"]).reshape(4, 4)
        camera_to_world_matrix = np.linalg.inv(world_to_camera_matrix)
        image_path = str(im_path)
        image_metadata.append(
            SfmImageMetadata(
                world_to_camera_matrix=world_to_camera_matrix,
                camera_to_world_matrix=camera_to_world_matrix,
                camera_id=camera_id,
                camera_metadata=camera_metadata[camera_id],
                image_path=image_path,
                mask_path="",
                point_indices=np.array([], dtype=np.int32),
                image_id=i,
            )
        )

    points, colors = pcu.load_mesh_vc(points_path)
    if points is None:
        raise ValueError(f"Failed to load points from {points_path}")
    if points.shape[0] == 0:
        raise ValueError(f"No points found in {points_path}")
    if colors is None:
        raise ValueError(f"No colors found in {points_path}")
    if colors.shape[0] != points.shape[0]:
        raise ValueError(f"Number of colors does not match number of points in {points_path}")
    errors = np.zeros((points.shape[0],), dtype=np.float32)

    cache = SfmCache.get_cache(data_path / "_cache", "sfm_dataset_cache", "Cache for SFM dataset")

    return camera_metadata, image_metadata, points, colors, errors, cache
