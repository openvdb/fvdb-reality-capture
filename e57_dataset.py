# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pye57
import pyproj
import torch
import tqdm
from torch.utils.data import Dataset

from ._transform_utils import (
    camera_similarity_normalization_transform,
    geo_ecef2enu_normalization_transform,
    pca_normalization_transform,
    transform_cam_to_world_matrix,
    transform_point_cloud,
)


def normalized_quat_to_rotmat(quat_: np.ndarray) -> np.ndarray:
    """Convert normalized quaternion to rotation matrix.

    Args:
        quat: Normalized quaternion in wxyz convension. (..., 4)

    Returns:
        Rotation matrix (..., 3, 3)
    """
    assert quat_.shape[-1] == 4, quat_.shape

    w, x, y, z = np.split(quat_, 4, axis=-1)
    w, x, y, z = w.squeeze(-1), x.squeeze(-1), y.squeeze(-1), z.squeeze(-1)

    mat = np.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        axis=-1,
    )
    return mat.reshape(quat_.shape[:-1] + (3, 3))


class E57Scan:
    """
    Class representing a single scan from an E57 file.
    """

    def __init__(self, e57: pye57.E57, index: int):
        self.e57 = e57
        self.index = index

        self._scan_read = False

    def _get_data(self):
        if not self._scan_read:
            self.scan_data = self.e57.read_scan(self.index, colors=True)
            self._scan_read = True
        return self.scan_data

    def get_points(self) -> np.ndarray:
        data = self._get_data()
        return np.column_stack(
            [
                data["cartesianX"],
                data["cartesianY"],
                data["cartesianZ"],
            ]
        )

    def get_colors(self) -> np.ndarray:
        data = self._get_data()
        return np.column_stack(
            [
                data["colorRed"],
                data["colorGreen"],
                data["colorBlue"],
            ]
        )


class E57File:
    """
    Class representing a single E57 file which contains a set of scans and a set of images and camera metadata.
    """

    def __init__(self, file_path: Union[str, Path], image_downsample_factor: int = 1):
        self.file_path = Path(file_path)
        self.e57 = pye57.E57(str(file_path))
        self.image_downsample_factor = image_downsample_factor

    def __len__(self):
        return self.e57.scan_count

    def _get_header(self, index: int) -> pye57.ScanHeader:
        return self.e57.get_header(index)

    def get_scan(self, index: int) -> E57Scan:
        return E57Scan(self.e57, index)

    def get_num_images(self) -> int:
        imf = self.e57.image_file
        root = imf.root()
        images = root["images2D"]
        return len(images)

    def get_image(self, index: int) -> np.ndarray:  # [H, W, 3]
        imf = self.e57.image_file
        root = imf.root()
        images = root["images2D"]
        image_node = images[index]

        if image_node.isDefined("pinholeRepresentation"):
            representation = image_node["pinholeRepresentation"]
        elif image_node.isDefined("sphericalRepresentation"):
            representation = image_node["sphericalRepresentation"]
        else:
            raise ValueError("No image representation found")
        jpeg_image = representation["jpegImage"]
        jpeg_image_data = np.zeros(shape=jpeg_image.byteCount(), dtype=np.uint8)
        jpeg_image.read(jpeg_image_data, 0, jpeg_image.byteCount())
        image = cv2.imdecode(jpeg_image_data, cv2.IMREAD_COLOR)
        if self.image_downsample_factor > 1:
            h, w = image.shape[:2]
            new_h, new_w = h // self.image_downsample_factor, w // self.image_downsample_factor
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def get_intrinsics(self, index: int) -> np.ndarray:
        imf = self.e57.image_file
        root = imf.root()
        images = root["images2D"]
        image_node = images[index]

        if image_node.isDefined("pinholeRepresentation"):
            representation = image_node["pinholeRepresentation"]

            # Get focal length (in meters) and pixel dimensions (in meters)
            focal_length_m = representation["focalLength"].value()
            pixel_width_m = representation["pixelWidth"].value()
            pixel_height_m = representation["pixelHeight"].value()
            principal_point = representation["principalPointX"].value(), representation["principalPointY"].value()

            # Convert focal length from meters to pixels
            fx = focal_length_m / pixel_width_m
            fy = focal_length_m / pixel_height_m

            proj_matrix = np.array(
                [
                    [fx, 0, principal_point[0]],
                    [0, fy, principal_point[1]],
                    [0, 0, 1],
                ]
            )
            if self.image_downsample_factor > 1:
                proj_matrix[:2, :] /= self.image_downsample_factor
            return proj_matrix
        elif image_node.isDefined("sphericalRepresentation"):
            raise ValueError("Spherical representation not supported")
        else:
            raise ValueError("No camera representation found")

    def get_camera_to_world_matrix(self, index: int) -> np.ndarray:
        imf = self.e57.image_file
        root = imf.root()
        images = root["images2D"]
        image_node = images[index]
        pose_node = image_node["pose"]
        rot_node = pose_node["rotation"]
        trans_node = pose_node["translation"]

        rot = np.array([rot_node["w"].value(), rot_node["x"].value(), rot_node["y"].value(), rot_node["z"].value()])
        trans = np.array([trans_node["x"].value(), trans_node["y"].value(), trans_node["z"].value()])

        rot_mat = normalized_quat_to_rotmat(rot)

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_mat
        transform_matrix[:3, 3] = trans

        # The E57 standard specifies a Z-up camera coordinate system whereas our colmap code expects Y-up.
        # This is a 180-degree rotation around the X-axis, which flips the Y and Z axes.
        cv_to_e57_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        transform_matrix = transform_matrix @ cv_to_e57_transform

        return transform_matrix


class E57Scene:
    """
    Class representing an E57 scene which consists of a set of posed images taken from a set of cameras along with
    a set of 3D scanned points in the scene. The scene is normalized to a canonical coordinate system using one of several
    heuristics.
    """

    def __init__(
        self,
        dataset_path: str,
        pointcloud_downsample_factor: int = 1,
        image_downsample_factor: int = 1,
        normalization_type: str = "pca",
        max_workers: Optional[int] = None,
    ):
        self.dataset_path = dataset_path
        self.pointcloud_downsample_factor = pointcloud_downsample_factor
        self.image_downsample_factor = image_downsample_factor
        logging.debug(f"Loading E57 files from {dataset_path}")
        self.e57_files = [E57File(path, image_downsample_factor) for path in sorted(Path(dataset_path).glob("*.e57"))]
        if len(self.e57_files) == 0:
            raise ValueError(f"No E57 files found in {dataset_path}")

        valid_normalization_types = {"none", "pca", "ecef2enu", "similarity"}
        if normalization_type not in valid_normalization_types:
            raise ValueError(
                f"Unknown normalization type {normalization_type}. Must be one of {valid_normalization_types}"
            )

        if os.path.exists(os.path.join(dataset_path, f"points_rgb_{pointcloud_downsample_factor}.npz")):

            cache = np.load(os.path.join(dataset_path, f"points_rgb_{pointcloud_downsample_factor}.npz"))
            self.points = cache["points"]
            self.points_rgb = cache["rgb"]
            logging.debug(f"Loaded {self.points.shape[0]} points and {self.points_rgb.shape[0]} colors")
        else:
            logging.debug("Gathering points and colors in parallel")

            # Create a list of all scan tasks
            scan_tasks = []
            for e57_file in self.e57_files:
                for i in range(len(e57_file)):
                    scan_tasks.append((e57_file, i))

            logging.debug(f"Loading {len(scan_tasks)} scans using parallel processing")

            # Use ThreadPoolExecutor to parallelize scan loading
            points_list = []
            colors_list = []

            # Determine optimal number of workers
            if max_workers is None:
                max_workers = min(8, len(scan_tasks), os.cpu_count() or 4)

            print(f"Loading {len(scan_tasks)} scans using {max_workers} parallel workers...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_scan = {
                    executor.submit(self._load_scan_data, e57_file, scan_idx): (e57_file, scan_idx)
                    for e57_file, scan_idx in scan_tasks
                }

                # Create progress bar
                progress_bar = tqdm.tqdm(
                    total=len(scan_tasks),
                    desc="Loading scans",
                    unit="scan",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )

                # Collect results as they complete
                for future in as_completed(future_to_scan):
                    try:
                        points_data, colors_data = future.result()
                        points_data = points_data[::pointcloud_downsample_factor]
                        colors_data = colors_data[::pointcloud_downsample_factor]
                        points_list.append(points_data)
                        colors_list.append(colors_data)

                        # Update progress bar with additional info
                        progress_bar.set_postfix(
                            {
                                "Points": f"{points_data.shape[0]:,}",
                                "Total_pts": f"{sum(p.shape[0] for p in points_list):,}",
                            }
                        )
                        progress_bar.update(1)

                    except Exception as e:
                        e57_file, scan_idx = future_to_scan[future]
                        progress_bar.close()
                        logging.error(f"Failed to load scan {scan_idx} from {e57_file.file_path}: {e}")
                        raise

                progress_bar.close()

            # Concatenate all the data
            self.points = np.concatenate(points_list, axis=0)
            logging.debug(f"Points shape: {self.points.shape}")

            self.points_rgb = np.concatenate(colors_list, axis=0)
            logging.debug(f"Colors shape: {self.points_rgb.shape[0]} colors")

            # save  points to file
            np.savez_compressed(
                os.path.join(dataset_path, f"points_rgb_{pointcloud_downsample_factor}.npz"),
                points=self.points,
                rgb=self.points_rgb,
            )

        logging.debug("Gathering world-to-camera matrices")
        cam_to_world_mats = np.stack(
            [e57_file.get_camera_to_world_matrix(i) for e57_file, i in self._iter_all_images()],
            axis=0,
        )
        world_to_cam_mats = np.linalg.inv(cam_to_world_mats)
        logging.debug(f"World-to-camera matrices shape: {world_to_cam_mats.shape}")

        logging.debug("Computing normalization transform")
        self.normalization_transform = self._compute_normalization_transform(
            self.points, world_to_cam_mats, normalization_type
        )

        self.points = transform_point_cloud(self.normalization_transform, self.points).astype(
            np.float32
        )  # (num_points, 3)

        self.cam_to_world_mats = transform_cam_to_world_matrix(self.normalization_transform, cam_to_world_mats)
        self.world_to_cam_mats = np.linalg.inv(self.cam_to_world_mats)

        self.intrinsics = np.stack(
            [e57_file.get_intrinsics(i) for e57_file, i in self._iter_all_images()],
            axis=0,
        )

        self.scene_scale = self._compute_scene_scale(self.cam_to_world_mats)

    @staticmethod
    def _load_scan_data(e57_file: E57File, scan_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load point and color data for a single scan. This function is designed to be thread-safe.

        Args:
            e57_file: The E57 file to load from
            scan_idx: Index of the scan to load

        Returns:
            Tuple of (points, colors) as numpy arrays
        """
        scan = e57_file.get_scan(scan_idx)
        points = scan.get_points()
        colors = scan.get_colors()
        return points, colors

    @property
    def num_images(self) -> int:
        return sum([e57_file.get_num_images() for e57_file in self.e57_files])

    def _camera_index_to_camera(self, index: int) -> Tuple[E57File, int]:
        """Return the image/camera at the given index."""
        for e57_file in self.e57_files:
            if index < e57_file.get_num_images():
                return e57_file, index
            index -= e57_file.get_num_images()
        raise IndexError(f"Index {index} out of range for dataset with {self.num_images} images")

    def get_image(self, camera_index: int) -> np.ndarray:
        e57_file, local_index = self._camera_index_to_camera(camera_index)
        return e57_file.get_image(local_index)

    def get_world_to_camera_matrix(self, camera_index: int) -> np.ndarray:
        return self.world_to_cam_mats[camera_index]

    def get_cam_to_world_matrix(self, camera_index: int) -> np.ndarray:
        return self.cam_to_world_mats[camera_index]

    def get_intrinsics(self, camera_index: int) -> np.ndarray:
        return self.intrinsics[camera_index]

    @staticmethod
    def _compute_normalization_transform(
        points: np.ndarray,
        world_to_cam_mats: Optional[np.ndarray] = None,
        normalization_type: str = "pca",
    ):
        """
        Computes an affine transformatrion matrix which normalizes a scene using one of several heuristics

        Args:
            points: 3D points to use for normalization
            world_to_cam_mats: 4x4 camera-to-world transformation matrices
            normalization_type: Type of normalization to apply. Options are "pca", "similarity", "ecef2enu", or "none".
        Returns:
            normalization_transform: 4x4 transformation matrix for normalizing the scene
        """
        # Normalize the world space.
        if normalization_type == "pca":
            normalization_transform = pca_normalization_transform(points)
        elif normalization_type == "ecef2enu":
            centroid = np.median(points, axis=0)
            tform_ecef2lonlat = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
            pt_lonlat = tform_ecef2lonlat.transform(centroid[0], centroid[1], centroid[2])

            normalization_transform = geo_ecef2enu_normalization_transform(
                points, pt_lonlat[0], pt_lonlat[1], centroid[0], centroid[1], centroid[2]
            )
        elif normalization_type == "similarity":
            # For similarity transform, we need camera poses
            if world_to_cam_mats:
                cam_to_world_mats = np.linalg.inv(world_to_cam_mats)
                normalization_transform = camera_similarity_normalization_transform(cam_to_world_mats)
            else:
                normalization_transform = np.eye(4)
        elif normalization_type == "none":
            normalization_transform = np.eye(4)
        else:
            raise RuntimeError(f"Unknown normalization type {normalization_type}")

        return normalization_transform

    @staticmethod
    def _compute_scene_scale(cam_to_world_mats: np.ndarray) -> float:
        """
        Calculate the maximum distance from the average point of the scene to any point
        which defines a notion of scene scale

        Args:
            cam_to_world_mats: 4x4 camera-to-world transformation matrices

        Returns:
            scene_scale: The maximum distance from the average point of the scene to any point
        """
        camera_locations = cam_to_world_mats[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        scene_scale = np.max(dists)

        return scene_scale

    def _iter_all_images(self):
        """Generator that yields (e57_file, image_index) pairs for all images across all E57 files."""
        for e57_file in self.e57_files:
            for i in range(e57_file.get_num_images()):
                yield e57_file, i


class E57Dataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        normalization_type: str = "pca",  # or "ecef2enu"
        image_downsample_factor: int = 1,
        pointcloud_downsample_factor: int = 1,
        test_every: int = 8,  # or 100
        split: str = "train",  # "train", "test", or "all"
        image_indices: Optional[List] = None,
        mask_path: Optional[str] = None,
        max_workers: Optional[int] = None,
    ):
        self.dataset_path = dataset_path
        self.normalization_type = normalization_type
        self.image_downsample_factor = image_downsample_factor
        self.test_every = test_every
        self.split = split

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        self.e57_scene = E57Scene(
            dataset_path,
            pointcloud_downsample_factor=pointcloud_downsample_factor,
            image_downsample_factor=image_downsample_factor,
            normalization_type=normalization_type,
            max_workers=max_workers,
        )

        indices = np.arange(self.e57_scene.num_images) if image_indices is None else np.array(image_indices)
        if self.split == "train":
            self.indices = indices[indices % self.test_every != 0]
        elif self.split == "test":
            self.indices = indices[indices % self.test_every == 0]
        elif self.split == "all":
            self.indices = indices
        else:
            raise ValueError(f"Split must be one of 'train', 'test', or 'all'. Got {self.split}.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_index = self.indices[index]
        image = self.e57_scene.get_image(image_index)

        item = {
            "K": torch.from_numpy(self.e57_scene.get_intrinsics(image_index)).float(),
            "worldtocam": torch.from_numpy(self.e57_scene.get_world_to_camera_matrix(image_index)).float(),
            "camtoworld": torch.from_numpy(self.e57_scene.get_cam_to_world_matrix(image_index)).float(),
            "image": image,
            "mask": np.max(image, axis=-1) < 5,  # wherever the image is black, the mask is true
            "image_id": image_index,
        }
        return item

    @property
    def points(self) -> np.ndarray:
        """Return the points in the dataset."""
        return self.e57_scene.points

    @property
    def points_rgb(self) -> np.ndarray:
        """Return the colors in the dataset."""
        return self.e57_scene.points_rgb

    @property
    def scene_scale(self) -> float:
        """Return the scene scale for optimization."""
        points = self.points
        if len(points) > 0:
            # Simple scene scale calculation based on point cloud extent
            scene_center = np.mean(points, axis=0)
            dists = np.linalg.norm(points - scene_center, axis=1)
            return float(np.max(dists) * 1.1)
        else:
            return 1.0


def main():
    file_path = "/home/jswartz/Downloads/kantonalschule/20042020-kantonalschule-_Setip_001.e57"
    e57_file = E57File(file_path)
    for i in range(len(e57_file)):
        scan = e57_file.get_scan(i)
        points = scan.get_points()
        colors = scan.get_colors()
        print(points.shape)
        print(colors.shape)
        print("RGB min/max:", colors.min(), colors.max())
        print("Points min/max:", points.min(), points.max())

    for i in range(e57_file.get_num_images()):
        image = e57_file.get_image(i)
        print(image.shape)
        intrinsics = e57_file.get_intrinsics(i)
        print(intrinsics)
        camera_to_world_matrix = e57_file.get_camera_to_world_matrix(i)
        print(camera_to_world_matrix)

    e57_dataset = E57Dataset(os.path.dirname(file_path), pointcloud_downsample_factor=100)
    for i in range(len(e57_dataset)):
        item = e57_dataset[i]
        print(item["image"].shape)
        print(item["K"])
        print(item["worldtocam"])
        print(item["camtoworld"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().name = "e57_dataset"
    main()
