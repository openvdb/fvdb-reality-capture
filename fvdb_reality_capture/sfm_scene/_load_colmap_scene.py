# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib

import numpy as np
import tqdm
from fvdb import CameraModel

from ._colmap_utils import Camera as ColmapCamera
from ._colmap_utils import Image as ColmapImage
from ._colmap_utils import SceneManager
from .sfm_cache import SfmCache
from .sfm_metadata import SfmCameraMetadata, SfmPosedImageMetadata


def _camera_model_and_distortion_coeffs_from_colmap_camera(cam: ColmapCamera) -> tuple[CameraModel, np.ndarray]:
    """
    Convert a COLMAP camera into the canonical FVDB camera model and packed distortion coefficients.

    Args:
        cam (ColmapCamera): The COLMAP camera object.

    Returns:
        tuple[CameraModel, np.ndarray]: The canonical camera model and distortion coefficients in
            FVDB packed layout ``[k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]``.
    """
    coeffs = np.zeros((12,), dtype=np.float32)
    if cam.camera_type == 0 or cam.camera_type == "SIMPLE_PINHOLE":
        return CameraModel.PINHOLE, np.empty((0,), dtype=np.float32)
    if cam.camera_type == 1 or cam.camera_type == "PINHOLE":
        return CameraModel.PINHOLE, np.empty((0,), dtype=np.float32)
    if cam.camera_type == 2 or cam.camera_type == "SIMPLE_RADIAL":
        coeffs[0] = cam.k1
        return CameraModel.OPENCV_RADTAN_5, coeffs
    if cam.camera_type == 3 or cam.camera_type == "RADIAL":
        coeffs[0] = cam.k1
        coeffs[1] = cam.k2
        return CameraModel.OPENCV_RADTAN_5, coeffs
    if cam.camera_type == 4 or cam.camera_type == "OPENCV":
        coeffs[0] = cam.k1
        coeffs[1] = cam.k2
        coeffs[6] = cam.p1
        coeffs[7] = cam.p2
        return CameraModel.OPENCV_RADTAN_5, coeffs
    if cam.camera_type == 5 or cam.camera_type == "OPENCV_FISHEYE":
        raise ValueError("COLMAP OPENCV_FISHEYE cameras are not supported by fvdb.CameraModel")
    raise ValueError(f"Unknown camera type {cam.camera_type}")


def _load_colmap_internal(colmap_path: pathlib.Path) -> SceneManager:
    """
    Internal call to load colmap data into a `SceneManager` which encodes the raw colmap information
    before we extract an `SfmScene` from it.

    Args:
        colmap_path (pathlib.Path): The path to the COLMAP dataset directory.

    Returns:
        scene_manager (SceneManager): An internal object holding metadata about a COLMAP run.
    """

    if not colmap_path.exists():
        raise FileNotFoundError(f"COLMAP directory {colmap_path} does not exist.")

    colmap_sparse_path = colmap_path / "sparse" / "0"
    if not colmap_sparse_path.exists():
        colmap_sparse_path = colmap_path / "sparse"
    if not colmap_sparse_path.exists():
        raise FileNotFoundError(f"COLMAP directory {colmap_sparse_path} does not exist.")

    scene_manager = SceneManager(f"{colmap_sparse_path}/")  # Need the trailing slash for the SceneManager
    scene_manager.load_cameras()
    scene_manager.load_images()
    scene_manager.load_points3D()

    return scene_manager


def load_colmap_scene(colmap_path: pathlib.Path):
    """
    Load cameras, posed-images, and points (with a cache to store derived quantities) from the output
    of a COLMAP structure-from-motion (SfM) pipeline. COLMAP produces a directory of images, a set of
    correspondence points, as well as a lightweight SqLite database containing image poses
    (camera to world matrices), camera intrinsics (projection matrices, camera type, etc.), and
    indices of which points are seen from which images.

    Args:
        colmap_path (pathlib.Path): The path to the output of a COLMAP run.

    Returns:
        sfm_scene (SfmScene): An in-memory representation of the SfmScene for the output of the COLMAP run.
    """
    scene_manager = _load_colmap_internal(colmap_path)
    num_images = len(scene_manager.images)

    cache = SfmCache.get_cache(colmap_path / "_cache", "sfm_dataset_cache", "Cache for SFM dataset")

    logger = logging.getLogger(f"{__name__}.load_colmap_scene")

    image_world_to_cam_mats = []
    image_camera_ids = []
    image_colmap_ids = []
    image_file_names = []
    image_absolute_paths = []
    image_mask_absolute_paths = []
    loaded_cameras = dict()
    colmap_images_path = colmap_path / "images"
    colmap_masks_path = colmap_path / "masks"
    for colmap_image_id in scene_manager.images:
        colmap_image: ColmapImage = scene_manager.images[colmap_image_id]
        image_world_to_cam_mats.append(colmap_image.world_to_cam_matrix())
        image_camera_ids.append(colmap_image.camera_id)
        image_colmap_ids.append(colmap_image_id)
        image_file_names.append(colmap_image.name)
        image_absolute_paths.append(colmap_images_path / colmap_image.name)

        if colmap_masks_path.exists():
            image_mask_path = colmap_masks_path / colmap_image.name
            if image_mask_path.exists():
                image_mask_absolute_paths.append(image_mask_path)
            elif image_mask_path.with_suffix(".png").exists():
                image_mask_absolute_paths.append(image_mask_path.with_suffix(".png"))
            else:
                image_mask_absolute_paths.append("")
        else:
            image_mask_absolute_paths.append("")

        if colmap_image.camera_id not in loaded_cameras:
            colmap_camera: ColmapCamera = scene_manager.cameras[colmap_image.camera_id]
            camera_model, distortion_coeffs = _camera_model_and_distortion_coeffs_from_colmap_camera(colmap_camera)
            fx, fy, cx, cy = colmap_camera.fx, colmap_camera.fy, colmap_camera.cx, colmap_camera.cy
            img_width, img_height = colmap_camera.width, colmap_camera.height
            loaded_cameras[colmap_image.camera_id] = SfmCameraMetadata(
                img_width=img_width,
                img_height=img_height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                camera_model=camera_model,
                distortion_coeffs=distortion_coeffs,
            )

    # Most papers use train/test splits based on sorted images so sort the images here
    sort_indices = np.argsort(image_file_names)
    image_world_to_cam_mats = [image_world_to_cam_mats[i] for i in sort_indices]
    image_camera_ids = [image_camera_ids[i] for i in sort_indices]
    image_colmap_ids = [image_colmap_ids[i] for i in sort_indices]
    image_file_names = [image_file_names[i] for i in sort_indices]
    image_mask_absolute_paths = [image_mask_absolute_paths[i] for i in sort_indices]
    image_absolute_paths = [image_absolute_paths[i] for i in sort_indices]

    # Compute the set of 3D points visible in each image
    if cache.has_file("visible_points_per_image"):
        key_meta = cache.get_file_metadata("visible_points_per_image")
        value_meta = key_meta["metadata"]
        if (
            key_meta.get("data_type", "pt") != "pt"
            or value_meta.get("num_points", 0) != len(scene_manager.points3D)
            or value_meta.get("num_images", 0) != num_images
        ):
            logger.info("Cached visible points per image do not match current scene. Recomputing...")
            cache.delete_file("visible_points_per_image")

    if cache.has_file("visible_points_per_image"):
        logger.info("Loading visible points per image from cache...")
        _, point_indices = cache.read_file("visible_points_per_image")
    else:
        logger.info("Computing and caching visible points per image...")
        # For each point, get the images that see it
        point_indices = dict()  # Map from image names to point indices
        for point_id, data in tqdm.tqdm(scene_manager.point3D_id_to_images.items()):
            # For each image that sees this point, add the index of the point
            # to a list of points corresponding to that image
            for image_id, _ in data:
                point_idx = scene_manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_id, []).append(point_idx)
        point_indices = {k: np.array(v).astype(np.int32) for k, v in point_indices.items()}
        cache.write_file(
            name="visible_points_per_image",
            data=point_indices,
            metadata={
                "num_points": len(scene_manager.points3D),
                "num_images": num_images,
            },
            data_type="pt",
        )

    # Create ColmapImageMetadata objects for each image
    loaded_images = [
        SfmPosedImageMetadata(
            world_to_camera_matrix=image_world_to_cam_mats[i].copy(),
            camera_to_world_matrix=np.linalg.inv(image_world_to_cam_mats[i]).copy(),
            camera_id=image_camera_ids[i],
            camera_metadata=loaded_cameras[image_camera_ids[i]],
            image_path=str(image_absolute_paths[i].absolute()),
            mask_path=image_mask_absolute_paths[i],
            point_indices=(
                point_indices[image_colmap_ids[i]].copy()
                if image_colmap_ids[i] in point_indices
                else np.empty((0,), dtype=np.int32)
            ),
            image_id=i,
        )
        for i in range(len(image_file_names))
    ]

    # Transform the points to the normalized coordinate system and cast to the right types
    # Note: we do not normalize the point errors or colors, they are already in the correct format.
    # Note: we don't transform the point errors
    points = scene_manager.points3D.astype(np.float32)  # type: ignore (num_points, 3)
    points_err = scene_manager.point3D_errors.astype(np.float32)  # type: ignore
    points_rgb = scene_manager.point3D_colors.astype(np.uint8)  # type: ignore

    return loaded_cameras, loaded_images, points, points_err, points_rgb, cache
    # return sfm_scene
