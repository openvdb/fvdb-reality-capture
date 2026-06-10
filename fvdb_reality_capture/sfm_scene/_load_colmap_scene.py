# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import hashlib
import logging
import pathlib

import numpy as np
import tqdm

from .adapter import COLMAPAdapter
from .sfm_cache import SfmCache
from .sfm_metadata import SfmPosedImageMetadata


def _point_id_order_hash(point3D_ids: np.ndarray) -> str:
    point3D_ids = np.ascontiguousarray(point3D_ids, dtype=np.uint64)
    return hashlib.sha1(point3D_ids.view(np.uint8)).hexdigest()


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
    adapter = COLMAPAdapter(colmap_path)
    colmap_image_ids = adapter.registered_image_ids()
    num_images = len(colmap_image_ids)

    (
        points3D,
        point3D_ids,
        point3D_colors,
        point3D_errors,
        point3D_id_to_point3D_idx,
        point3D_id_to_images,
    ) = adapter.points_from_scene()
    point3D_id_order_hash = _point_id_order_hash(point3D_ids)

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
    for colmap_image_id in colmap_image_ids:
        colmap_camera_id = adapter.image_camera_id(colmap_image_id)
        image_file_name = adapter.image_name(colmap_image_id)
        image_world_to_cam_mats.append(adapter.world_to_camera_matrix(colmap_image_id))
        image_camera_ids.append(colmap_camera_id)
        image_colmap_ids.append(colmap_image_id)
        image_file_names.append(image_file_name)
        image_absolute_paths.append(colmap_images_path / image_file_name)

        if colmap_masks_path.exists():
            image_mask_path = colmap_masks_path / image_file_name
            if image_mask_path.exists():
                image_mask_absolute_paths.append(str(image_mask_path.absolute()))
            elif image_mask_path.with_suffix(".png").exists():
                image_mask_absolute_paths.append(str(image_mask_path.with_suffix(".png").absolute()))
            else:
                image_mask_absolute_paths.append("")
        else:
            image_mask_absolute_paths.append("")

        if colmap_camera_id not in loaded_cameras:
            loaded_cameras[colmap_camera_id] = adapter.camera_metadata(colmap_camera_id)

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
            or value_meta.get("num_points", 0) != len(points3D)
            or value_meta.get("num_images", 0) != num_images
            or value_meta.get("loader") != adapter.visibility_cache_loader
            or value_meta.get("point3D_id_order_hash") != point3D_id_order_hash
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
        for point_id, data in tqdm.tqdm(point3D_id_to_images.items()):
            # For each image that sees this point, add the index of the point
            # to a list of points corresponding to that image
            for image_id, _ in data:
                point_idx = point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(int(image_id), []).append(point_idx)
        point_indices = {k: np.array(v).astype(np.int32) for k, v in point_indices.items()}
        cache.write_file(
            name="visible_points_per_image",
            data=point_indices,
            metadata={
                "num_points": len(points3D),
                "num_images": num_images,
                "loader": adapter.visibility_cache_loader,
                "point3D_id_order_hash": point3D_id_order_hash,
            },
            data_type="pt",
        )

    # Create SfmPosedImageMetadata objects for each image
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
    points = points3D.astype(np.float32)  # type: ignore (num_points, 3)
    points_err = point3D_errors.astype(np.float32)  # type: ignore
    points_rgb = point3D_colors.astype(np.uint8)  # type: ignore

    return loaded_cameras, loaded_images, points, points_err, points_rgb, cache
