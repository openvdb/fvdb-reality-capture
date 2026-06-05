# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import hashlib
import logging
import pathlib
from typing import Any

import numpy as np
import pycolmap
import tqdm
from fvdb import CameraModel

from .sfm_cache import SfmCache
from .sfm_metadata import SfmCameraMetadata, SfmPosedImageMetadata


_COLMAP_CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
}

_VISIBLE_POINTS_CACHE_LOADER = "pycolmap"


def _colmap_model_to_name(colmap_model: Any) -> str:
    if isinstance(colmap_model, str):
        return colmap_model

    model_name = getattr(colmap_model, "name", None)
    if model_name is not None:
        return str(model_name)

    model_value = getattr(colmap_model, "value", colmap_model)
    if isinstance(model_value, np.integer):
        model_value = int(model_value)

    if isinstance(model_value, int) and model_value in _COLMAP_CAMERA_MODEL_NAMES:
        return _COLMAP_CAMERA_MODEL_NAMES[model_value]

    raise ValueError(f"Unknown COLMAP camera model {colmap_model}")


def _camera_model_name(cam: pycolmap.Camera) -> str:
    model_name = getattr(cam, "model_name", None)
    if model_name:
        return str(model_name)
    return _colmap_model_to_name(cam.model)


def _camera_intrinsics(cam: pycolmap.Camera) -> tuple[float, float, float, float]:
    params = np.asarray(cam.params, dtype=np.float64)
    camera_model = _camera_model_name(cam)

    if camera_model == "SIMPLE_PINHOLE":
        fx, cx, cy = params[:3]
        return float(fx), float(fx), float(cx), float(cy)
    if camera_model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    if camera_model == "SIMPLE_RADIAL":
        fx, cx, cy = params[:3]
        return float(fx), float(fx), float(cx), float(cy)
    if camera_model == "RADIAL":
        fx, cx, cy = params[:3]
        return float(fx), float(fx), float(cx), float(cy)
    if camera_model in ("OPENCV", "OPENCV_FISHEYE"):
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)

    raise ValueError(f"Unsupported COLMAP camera model {camera_model}")


def _camera_model_and_distortion_coeffs_from_colmap_camera(cam: pycolmap.Camera) -> tuple[CameraModel, np.ndarray]:
    """
    Convert a COLMAP camera into the canonical FVDB camera model and packed distortion coefficients.

    Args:
        cam (pycolmap.Camera): The COLMAP camera object.

    Returns:
        tuple[CameraModel, np.ndarray]: The canonical camera model and distortion coefficients in
            FVDB packed layout ``[k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]``.
    """
    camera_model = _camera_model_name(cam)
    params = np.asarray(cam.params, dtype=np.float64)
    coeffs = np.zeros((12,), dtype=np.float32)

    if camera_model == "SIMPLE_PINHOLE":
        return CameraModel.PINHOLE, np.empty((0,), dtype=np.float32)
    if camera_model == "PINHOLE":
        return CameraModel.PINHOLE, np.empty((0,), dtype=np.float32)
    if camera_model == "SIMPLE_RADIAL":
        coeffs[0] = params[3]
        return CameraModel.OPENCV_RADTAN_5, coeffs
    if camera_model == "RADIAL":
        coeffs[0] = params[3]
        coeffs[1] = params[4]
        return CameraModel.OPENCV_RADTAN_5, coeffs
    if camera_model == "OPENCV":
        coeffs[0] = params[4]
        coeffs[1] = params[5]
        coeffs[6] = params[6]
        coeffs[7] = params[7]
        return CameraModel.OPENCV_RADTAN_5, coeffs
    if camera_model == "OPENCV_FISHEYE":
        raise ValueError("COLMAP OPENCV_FISHEYE cameras are not supported by fvdb.CameraModel")
    raise ValueError(f"Unsupported COLMAP camera model {camera_model}")


def _world_to_camera_matrix(colmap_image: pycolmap.Image) -> np.ndarray:
    world_to_camera = np.asarray(colmap_image.cam_from_world().matrix(), dtype=np.float64)
    return np.vstack([world_to_camera, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)])


def _registered_image_ids(reconstruction: pycolmap.Reconstruction) -> list[int]:
    try:
        return [int(image_id) for image_id in reconstruction.reg_image_ids()]
    except AttributeError:
        return [int(image_id) for image_id, image in reconstruction.images.items() if image.has_pose]


def _point_id_order_hash(point3D_ids: np.ndarray) -> str:
    point3D_ids = np.ascontiguousarray(point3D_ids, dtype=np.uint64)
    return hashlib.sha1(point3D_ids.view(np.uint8)).hexdigest()


def _points3D_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int], dict[int, np.ndarray]]:
    point3D_items = sorted((int(point3D_id), point3D) for point3D_id, point3D in reconstruction.points3D.items())
    num_points = len(point3D_items)

    point3D_ids = np.empty(num_points, dtype=np.uint64)
    points3D = np.empty((num_points, 3), dtype=np.float64)
    point3D_colors = np.empty((num_points, 3), dtype=np.uint8)
    point3D_errors = np.empty(num_points, dtype=np.float64)
    point3D_id_to_point3D_idx: dict[int, int] = {}
    point3D_id_to_images: dict[int, np.ndarray] = {}

    for point3D_idx, (point3D_id, point3D) in enumerate(point3D_items):
        point3D_ids[point3D_idx] = point3D_id
        points3D[point3D_idx] = np.asarray(point3D.xyz, dtype=np.float64)
        point3D_colors[point3D_idx] = np.asarray(point3D.color, dtype=np.uint8)
        point3D_errors[point3D_idx] = float(point3D.error)
        point3D_id_to_point3D_idx[point3D_id] = point3D_idx
        point3D_id_to_images[point3D_id] = np.array(
            [(int(track_el.image_id), int(track_el.point2D_idx)) for track_el in point3D.track.elements],
            dtype=np.uint32,
        ).reshape(-1, 2)

    return (
        points3D,
        point3D_ids,
        point3D_colors,
        point3D_errors,
        point3D_id_to_point3D_idx,
        point3D_id_to_images,
    )


def _load_colmap_internal(colmap_path: pathlib.Path) -> pycolmap.Reconstruction:
    """
    Internal call to load colmap data into a `pycolmap.Reconstruction` which encodes the raw colmap information
    before we extract an `SfmScene` from it.

    Args:
        colmap_path (pathlib.Path): The path to the COLMAP dataset directory.

    Returns:
        reconstruction (pycolmap.Reconstruction): An internal object holding metadata about a COLMAP run.
    """

    if not colmap_path.exists():
        raise FileNotFoundError(f"COLMAP directory {colmap_path} does not exist.")

    colmap_sparse_path = colmap_path / "sparse" / "0"
    if not colmap_sparse_path.exists():
        colmap_sparse_path = colmap_path / "sparse"
    if not colmap_sparse_path.exists():
        raise FileNotFoundError(f"COLMAP directory {colmap_sparse_path} does not exist.")

    return pycolmap.Reconstruction(colmap_sparse_path)


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
    reconstruction = _load_colmap_internal(colmap_path)
    colmap_image_ids = _registered_image_ids(reconstruction)
    num_images = len(colmap_image_ids)

    (
        points3D,
        point3D_ids,
        point3D_colors,
        point3D_errors,
        point3D_id_to_point3D_idx,
        point3D_id_to_images,
    ) = _points3D_from_reconstruction(reconstruction)
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
        colmap_image = reconstruction.images[colmap_image_id]
        colmap_camera_id = int(colmap_image.camera_id)
        image_world_to_cam_mats.append(_world_to_camera_matrix(colmap_image))
        image_camera_ids.append(colmap_camera_id)
        image_colmap_ids.append(colmap_image_id)
        image_file_names.append(colmap_image.name)
        image_absolute_paths.append(colmap_images_path / colmap_image.name)

        if colmap_masks_path.exists():
            image_mask_path = colmap_masks_path / colmap_image.name
            if image_mask_path.exists():
                image_mask_absolute_paths.append(str(image_mask_path.absolute()))
            elif image_mask_path.with_suffix(".png").exists():
                image_mask_absolute_paths.append(str(image_mask_path.with_suffix(".png").absolute()))
            else:
                image_mask_absolute_paths.append("")
        else:
            image_mask_absolute_paths.append("")

        if colmap_camera_id not in loaded_cameras:
            colmap_camera = reconstruction.cameras[colmap_camera_id]
            camera_model, distortion_coeffs = _camera_model_and_distortion_coeffs_from_colmap_camera(colmap_camera)
            fx, fy, cx, cy = _camera_intrinsics(colmap_camera)
            img_width, img_height = int(colmap_camera.width), int(colmap_camera.height)
            loaded_cameras[colmap_camera_id] = SfmCameraMetadata(
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
            or value_meta.get("num_points", 0) != len(points3D)
            or value_meta.get("num_images", 0) != num_images
            or value_meta.get("loader") != _VISIBLE_POINTS_CACHE_LOADER
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
                "loader": _VISIBLE_POINTS_CACHE_LOADER,
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
