# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib

import cv2
import numpy as np
import pye57
import tqdm
from scipy.spatial.transform import Rotation

from . import SfmCache, SfmCameraMetadata, SfmCameraType, SfmImageMetadata


def _load_e57_scan(
    e57_file: pye57.E57,
    camera_metadata: dict[int, SfmCameraMetadata],
    image_metadata: list[SfmImageMetadata],
    cum_num_points: int,
    cache: SfmCache,
    total_images: int,
    point_downsample_factor: int = 1,
):
    """
    Load points and images from a single E57 file into the provided metadata structures.

    Args:
        e57_file (pye57.E57): The E57 file to load.
        camera_metadata (dict[int, SfmCameraMetadata]): Existing camera metadata to append to.
        image_metadata (list[SfmImageMetadata]): Existing image metadata to append to.
        cum_num_points (int): Cumulative number of points loaded so far (used for indexing).
        cache (SfmCache): Cache to use for storing images.
        total_images (int): Total number of images expected (used for zero-padding).
        point_downsample_factor (int): Factor by which to downsample input points (since E57 files can be large).
    Returns:
        tuple: (image_metadata, camera_metadata, points, points_rgb, points_intensity) where
            image_metadata (list[SfmImageMetadata]): Updated list of image metadata.
            camera_metadata (dict[int, SfmCameraMetadata]): Updated camera metadata dictionary.
            points (np.ndarray): Loaded points of shape (N, 3).
            points_rgb (np.ndarray): Loaded point colors of shape (N, 3).
            points_intensity (np.ndarray): Loaded point intensities of shape (N,).
    """
    num_scans: int = e57_file.scan_count
    e57_path: str = e57_file.path

    if num_scans <= 0:
        raise ValueError(f"No scans found in {e57_path}")

    image_file = e57_file.image_file.root()

    if not image_file.isDefined("images2D"):
        raise ValueError(f"No images2D found in {e57_path}")
    image_nodes = image_file["images2D"]

    if len(image_nodes) <= 0:
        raise ValueError(f"No images found in {e57_path}")

    num_zeropad = len(str(total_images)) + 2

    scans = {}
    for scan_idx in range(num_scans):
        scan_meta = e57_file.root["data3D"][scan_idx]
        if not scan_meta.isDefined("guid"):
            raise ValueError(f"No guid found for scan {scan_idx} in {e57_path}")
        scan_guid = scan_meta["guid"].value()
        scan = e57_file.read_scan(scan_idx, intensity=True, colors=True, row_column=True, ignore_missing_fields=True)

        if "cartesianX" not in scan:
            raise ValueError(f"No cartesianX found for scan {scan_idx} in {e57_path}")
        if "cartesianY" not in scan:
            raise ValueError(f"No cartesianY found for scan {scan_idx} in {e57_path}")
        if "cartesianZ" not in scan:
            raise ValueError(f"No cartesianZ found for scan {scan_idx} in {e57_path}")

        if "colorRed" not in scan:
            raise ValueError(f"No colorRed found for scan {scan_idx} in {e57_path}")
        if "colorGreen" not in scan:
            raise ValueError(f"No colorGreen found for scan {scan_idx} in {e57_path}")
        if "colorBlue" not in scan:
            raise ValueError(f"No colorBlue found for scan {scan_idx} in {e57_path}")

        points = np.stack(
            [
                scan["cartesianX"],
                scan["cartesianY"],
                scan["cartesianZ"],
            ],
            axis=1,
        ).astype(np.float32)

        points_rgb = np.stack(
            [
                scan["colorRed"],
                scan["colorGreen"],
                scan["colorBlue"],
            ],
            axis=1,
        ).astype(np.uint8)

        points_intensity = scan["intensity"] if "intensity" in scan else np.zeros_like(scan["cartesianX"])

        scans[scan_guid] = {
            "points": points[::point_downsample_factor],
            "points_rgb": points_rgb[::point_downsample_factor],
            "points_intensity": points_intensity[::point_downsample_factor],
            "guid": scan_guid,
        }

    sorted_keys = sorted(scans.keys())
    points = []
    points_rgb = []
    points_intensity = []
    total_points = 0
    cum_num_points_per_scan = {}
    for key in sorted_keys:
        points.append(scans[key]["points"])
        points_rgb.append(scans[key]["points_rgb"])
        points_intensity.append(scans[key]["points_intensity"])
        cum_num_points_per_scan[key] = total_points
        total_points += scans[key]["points"].shape[0]

    points = np.concatenate(points, axis=0)
    points_rgb = np.concatenate(points_rgb, axis=0)
    points_intensity = np.concatenate(points_intensity, axis=0)
    assert points.shape[0] == total_points
    assert points_rgb.shape[0] == total_points
    assert points_intensity.shape[0] == total_points
    assert points.shape[1] == 3
    assert points_rgb.shape[1] == 3
    assert points_intensity.ndim == 1

    start_image_id = len(image_metadata)
    start_camera_id = len(camera_metadata)

    for i in range(len(image_nodes)):
        image_node = image_nodes[i]

        camera_id = start_camera_id + i
        image_id = start_image_id + i

        if image_node.isDefined("pinholeRepresentation"):
            representation = image_node["pinholeRepresentation"]
        elif image_node.isDefined("sphericalRepresentation"):
            raise NotImplementedError("Spherical representation not supported yet")
        else:
            raise ValueError(f"No supported representation found for image {i} in {e57_path}")

        if not representation.isDefined("focalLength"):
            raise ValueError(f"No focalLength found for image {i} in {e57_path}")
        if not representation.isDefined("pixelWidth"):
            raise ValueError(f"No pixelWidth found for image {i} in {e57_path}")
        if not representation.isDefined("pixelHeight"):
            raise ValueError(f"No pixelHeight found for image {i} in {e57_path}")
        if not representation.isDefined("principalPointX"):
            raise ValueError(f"No principalPointX found for image {i} in {e57_path}")
        if not representation.isDefined("principalPointY"):
            raise ValueError(f"No principalPointY found for image {i} in {e57_path}")
        if not representation.isDefined("imageWidth"):
            raise ValueError(f"No imageWidth found for image {i} in {e57_path}")
        if not representation.isDefined("imageHeight"):
            raise ValueError(f"No imageHeight found for image {i} in {e57_path}")

        if not representation.isDefined("jpegImage"):
            raise ValueError(f"No jpegImage found for image {i} in {e57_path}")

        if not image_node.isDefined("pose"):
            raise ValueError(f"No pose found for image {i} in {e57_path}")

        if not image_node.isDefined("associatedData3DGuid"):
            raise ValueError(f"No associatedData3DGuid found for image {i} in {e57_path}")

        associated_scan_guid = image_node["associatedData3DGuid"].value()
        if associated_scan_guid not in scans:
            raise ValueError(f"Associated scan guid {associated_scan_guid} not found for image {i} in {e57_path}")

        # Get focal length (in meters) and pixel dimensions (in meters)
        focal_length_m = representation["focalLength"].value()
        pixel_width_m = representation["pixelWidth"].value()
        pixel_height_m = representation["pixelHeight"].value()

        # Convert focal length from meters to pixels
        fx = focal_length_m / pixel_width_m
        fy = focal_length_m / pixel_height_m

        # Get the principal point (in pixels)
        cx = representation["principalPointX"].value()
        cy = representation["principalPointY"].value()

        # Get the sensor resolution
        image_width = representation["imageWidth"].value()
        image_height = representation["imageHeight"].value()

        camera_metadata[camera_id] = SfmCameraMetadata(
            img_width=image_width,
            img_height=image_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_type=SfmCameraType.PINHOLE,
            distortion_parameters=np.array([]),
        )

        # Load the image data
        image_filename = f"e57_image_{image_id:0{num_zeropad}d}"
        if not cache.has_file(image_filename):
            jpeg_image = representation["jpegImage"]
            jpeg_image_data = np.zeros(shape=jpeg_image.byteCount(), dtype=np.uint8)
            jpeg_image.read(jpeg_image_data, 0, jpeg_image.byteCount())
            image = cv2.imdecode(jpeg_image_data, cv2.IMREAD_COLOR)
            assert image is not None, f"Failed to decode JPEG image for image {i} in {e57_path}"

            cache.write_file(image_filename, image, "jpg", quality=98)

        cache_file_metadata = cache.get_file_metadata(image_filename)
        assert cache_file_metadata is not None
        image_path = cache_file_metadata["path"]

        pose_node = image_node["pose"]
        if not pose_node.isDefined("rotation"):
            raise ValueError(f"No rotation found for image {i} in {e57_path}")
        if not pose_node.isDefined("translation"):
            raise ValueError(f"No translation found for image {i} in {e57_path}")

        rot_node = pose_node["rotation"]
        if not rot_node.isDefined("w"):
            raise ValueError(f"No rotation.w found for image {i} in {e57_path}")
        if not rot_node.isDefined("x"):
            raise ValueError(f"No rotation.x found for image {i} in {e57_path}")
        if not rot_node.isDefined("y"):
            raise ValueError(f"No rotation.y found for image {i} in {e57_path}")
        if not rot_node.isDefined("z"):
            raise ValueError(f"No rotation.z found for image {i} in {e57_path}")

        translation_node = pose_node["translation"]
        if not translation_node.isDefined("x"):
            raise ValueError(f"No translation.x found for image {i} in {e57_path}")
        if not translation_node.isDefined("y"):
            raise ValueError(f"No translation.y found for image {i} in {e57_path}")
        if not translation_node.isDefined("z"):
            raise ValueError(f"No translation.z found for image {i} in {e57_path}")

        # Load the world-to-camera matrix
        rot_quat = np.array(
            [
                rot_node["w"].value(),
                rot_node["x"].value(),
                rot_node["y"].value(),
                rot_node["z"].value(),
            ]
        )
        rot_matrix = Rotation.from_quat(rot_quat, scalar_first=True).as_matrix()
        translation = np.array(
            [
                translation_node["x"].value(),
                translation_node["y"].value(),
                translation_node["z"].value(),
            ]
        )
        cam_to_world_matrix = np.eye(4)
        cam_to_world_matrix[:3, :3] = rot_matrix
        cam_to_world_matrix[:3, 3] = translation
        world_to_cam_matrix = np.linalg.inv(cam_to_world_matrix)

        # The E57 standard specifies a Z-up camera coordinate system whereas our colmap code expects Y-up.
        # This is a 180-degree rotation around the X-axis, which flips the Y and Z axes.
        cv_to_e57_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        cam_to_world_matrix = cam_to_world_matrix @ cv_to_e57_transform

        # Compute which points in the scan project into the image.
        # These are the set of visible points for this image.
        # We will store these indices in the SfmImageMetadata for this image.
        projection_matrix = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )
        associated_scan_points = scans[associated_scan_guid]["points"]
        associated_scan_points_cam = (
            world_to_cam_matrix[:3, :3] @ associated_scan_points.T + world_to_cam_matrix[:3, 3:4]
        )
        associated_scan_points_clip = projection_matrix @ associated_scan_points_cam
        associated_scan_z = associated_scan_points_cam[2, :].T
        associated_scan_points_ndc = (associated_scan_points_clip[:2, :] / associated_scan_points_clip[2:3, :]).T
        visible_point_indices = (
            np.where(
                (associated_scan_z < 0)
                & (associated_scan_points_ndc[:, 0] >= 0)
                & (associated_scan_points_ndc[:, 0] < image_width)
                & (associated_scan_points_ndc[:, 1] >= 0)
                & (associated_scan_points_ndc[:, 1] < image_height)
            )[0]
            + cum_num_points_per_scan[associated_scan_guid]
        ) + cum_num_points

        image_metadata.append(
            SfmImageMetadata(
                world_to_camera_matrix=world_to_cam_matrix,
                camera_to_world_matrix=cam_to_world_matrix,
                camera_metadata=camera_metadata[camera_id],
                camera_id=camera_id,
                image_path=image_path,
                mask_path="",
                point_indices=visible_point_indices.astype(np.int32),
                image_id=image_id,
            )
        )

    return image_metadata, camera_metadata, points, points_rgb, points_intensity


def load_e57_dataset(dataset_path: pathlib.Path, point_downsample_factor: int = 1):
    """
    Load a dataset of posed images and points from a directory of E57 files.

    Args:
        dataset_path (pathlib.Path): Path to the directory containing E57 files.
        point_downsample_factor (int): Factor by which to downsample input points (since E57 files can be large).

    Returns:
        tuple: (camera_metadata, image_metadata, points, points_rgb, points_intensity, cache) where
            camera_metadata (dict[int, SfmCameraMetadata]): Dictionary mapping camera IDs to camera metadata.
            image_metadata (list[SfmImageMetadata]): List of image metadata.
            points (np.ndarray): Loaded points of shape (N, 3).
            points_rgb (np.ndarray): Loaded point colors of shape (N, 3).
            points_intensity (np.ndarray): Loaded point intensities of shape (N,).
    """
    e57_files = sorted(list(dataset_path.glob("*.e57")))
    if len(e57_files) == 0:
        raise ValueError(f"No E57 files found in {dataset_path}")

    cache = SfmCache.get_cache(dataset_path / "_cache", "e57_dataset_cache", "Cache for e57 dataset")

    total_images = 0
    total_scans = 0
    for e57_path in e57_files:
        e57_file: pye57.E57 = pye57.E57(str(e57_path))
        image_file = e57_file.image_file.root()

        if not image_file.isDefined("images2D"):
            raise ValueError(f"No images2D found in {e57_path}")

        total_images += len(image_file["images2D"])
        total_scans += e57_file.scan_count

    if cache.num_files != total_images:
        cache.clear_current_folder()

    cum_num_points = 0
    image_metadata: list[SfmImageMetadata] = []
    camera_metadata: dict[int, SfmCameraMetadata] = {}
    points = []
    points_rgb = []
    points_intensity = []
    for e57_path in tqdm.tqdm(e57_files, desc="Loading E57 files"):
        e57_file: pye57.E57 = pye57.E57(str(e57_path))
        (
            image_metadata,
            camera_metadata,
            scan_points,
            scan_points_rgb,
            scan_points_intensity,
        ) = _load_e57_scan(
            e57_file,
            camera_metadata,
            image_metadata,
            cum_num_points,
            cache,
            total_images,
            point_downsample_factor,
        )
        points.append(scan_points)
        points_rgb.append(scan_points_rgb)
        points_intensity.append(scan_points_intensity)
        cum_num_points += scan_points.shape[0]

    points = np.concatenate(points, axis=0)
    points_rgb = np.concatenate(points_rgb, axis=0)
    points_intensity = np.concatenate(points_intensity, axis=0)

    return camera_metadata, image_metadata, points, points_rgb, points_intensity, cache
