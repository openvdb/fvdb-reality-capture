# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib

import numpy as np
import tqdm

from ..image_dataset_cache import ImageDatasetCache
from ._colmap_utils import Camera as ColmapCamera
from ._colmap_utils import Image as ColmapImage
from ._colmap_utils import SceneManager
from .base_dataset_reader import BaseDatasetReader
from .sfm_metadata import SfmCameraMetadata, SfmCameraType, SfmImageMetadata


class ColmapDatasetReader(BaseDatasetReader):
    """
    A DatasetReader for COLMAP-style datasets.

    This class reads COLMAP datasets and provides methods to access camera metadata, image metadata, and 3D points.

    It implements the interface defined in `BaseDatasetReader` and provides methods to load the dataset,
    including cameras, images, and points.
    """

    def __init__(self, colmap_path: pathlib.Path):
        """
        Initialize the ColmapDatasetReader with the path to the COLMAP dataset.
        Args:
            colmap_path (pathlib.Path): Path to the COLMAP dataset directory.
        """
        super().__init__()
        if not colmap_path.exists():
            raise FileNotFoundError(f"COLMAP directory {colmap_path} does not exist.")
        self._colmap_path = colmap_path

        colmap_sparse_path = self._colmap_path / "sparse" / "0"
        if not colmap_sparse_path.exists():
            colmap_sparse_path = self._colmap_path / "sparse"
        if not colmap_sparse_path.exists():
            raise FileNotFoundError(f"COLMAP directory {colmap_sparse_path} does not exist.")

        self._scene_manager = SceneManager(f"{colmap_sparse_path}/")  # Need the trailing slash for the SceneManager
        self._scene_manager.load_cameras()
        self._scene_manager.load_images()
        self._scene_manager.load_points3D()
        self._cache = ImageDatasetCache(self._colmap_path, num_images=self.num_images)

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    @property
    def cache(self) -> ImageDatasetCache:
        """
        Return the ImageDatasetCache associated with this dataset reader.
        This cache is used to store and retrieve image data efficiently.
        Returns:
            ImageDatasetCache: The cache used by this dataset reader.
        """
        return self._cache

    @staticmethod
    def _colmap_camera_type_to_str(colmap_camera_type: int) -> SfmCameraType:
        """
        Convert a COLMAP camera type integer to an SfmCameraType enum.

        Args:
            colmap_camera_type (int): The COLMAP camera type integer.

        Returns:
            SfmCameraType: The corresponding SfmCameraType enum.
        """
        if colmap_camera_type == 0:
            return SfmCameraType.SIMPLE_PINHOLE
        elif colmap_camera_type == 1:
            return SfmCameraType.PINHOLE
        elif colmap_camera_type == 2:
            return SfmCameraType.SIMPLE_RADIAL
        elif colmap_camera_type == 3:
            return SfmCameraType.RADIAL
        elif colmap_camera_type == 4:
            return SfmCameraType.OPENCV
        elif colmap_camera_type == 5:
            return SfmCameraType.OPENCV_FISHEYE
        else:
            raise ValueError(f"Unknown COLMAP camera type {colmap_camera_type}")

    @staticmethod
    def _distortion_params_from_camera_type(cam: ColmapCamera) -> np.ndarray:
        """
        Get distotion model parameters (to use with cv2.initUndistortRectifyMap) from the specified camera type.
        We store these so we can distort images from non pinhole camera models and use a pinhole camera model.

        Args:
            cam (ColmapCamera): The COLMAP camera object.

        Returns:
            np.ndarray: An array of distortion parameters.
            The shape and content of the array depend on the camera type.
            For example, for a radial camera, it returns [k1, k2, 0.0, 0.0].
            For an OpenCV camera, it returns [k1, k2, p1, p2].
            For a simple pinhole camera, it returns an empty array.
            For a pinhole camera, it also returns an empty array.
            For a fisheye camera, it raises a NotImplementedError.
        """
        if cam.camera_type == 0 or cam.camera_type == "SIMPLE_PINHOLE":
            return np.empty(0, dtype=np.float32)
        elif cam.camera_type == 1 or cam.camera_type == "PINHOLE":
            return np.empty(0, dtype=np.float32)
        elif cam.camera_type == 2 or cam.camera_type == "SIMPLE_RADIAL":
            return np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
        elif cam.camera_type == 3 or cam.camera_type == "RADIAL":
            return np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
        elif cam.camera_type == 4 or cam.camera_type == "OPENCV":
            return np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
        elif cam.camera_type == 5 or cam.camera_type == "OPENCV_FISHEYE":
            raise NotImplementedError("Fisheye cameras are not currently supported")
            return np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
        else:
            raise ValueError(f"Unknown camera type {cam.camera_type}")

    @property
    def num_images(self) -> int:
        """
        Return the number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self._scene_manager.images)

    def load_data(
        self,
    ) -> tuple[dict[int, SfmCameraMetadata], list[SfmImageMetadata], np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the dataset into a format consumable by `SfmScene`

        Returns:
            camera_metadata (dict[int, SfmCameraMetadata]): A dictionary mapping camera IDs to `SfmCameraMetadata` objects.
            image_metadata (list[SfmImageMetadata]): A list of `SfmImageMetadata` objects.
            points (np.ndarray): An Nx3 array of 3D points in the scene, where N is the number of points.
            points_err (np.ndarray): An array of shape (N,) representing the error or uncertainty of each point in `points`.
            points_rgb (np.ndarray): An Nx3 uint8 array of RGB color values for each point in the scene, where N is the number of points.
        """
        image_world_to_cam_mats = []
        image_camera_ids = []
        image_colmap_ids = []
        image_file_names = []
        image_absolute_paths = []
        image_mask_absolute_paths = []
        loaded_cameras = dict()
        colmap_images_path = self._colmap_path / "images"
        colmap_masks_path = self._colmap_path / "masks"
        for colmap_image_id in self._scene_manager.images:
            colmap_image: ColmapImage = self._scene_manager.images[colmap_image_id]
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
                colmap_camera: ColmapCamera = self._scene_manager.cameras[colmap_image.camera_id]
                distortion_parameters = ColmapDatasetReader._distortion_params_from_camera_type(colmap_camera)
                fx, fy, cx, cy = colmap_camera.fx, colmap_camera.fy, colmap_camera.cx, colmap_camera.cy
                img_width, img_height = colmap_camera.width, colmap_camera.height
                colmap_camera_type_enum = ColmapDatasetReader._colmap_camera_type_to_str(colmap_camera.camera_type)
                loaded_cameras[colmap_image.camera_id] = SfmCameraMetadata(
                    img_width, img_height, fx, fy, cx, cy, colmap_camera_type_enum, distortion_parameters
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
        if "visible_points_per_image" in self._cache:
            key_meta, value_meta = self._cache.get_property_metadata("visible_points_per_image")
            if (
                key_meta["scope"] != "dataset"
                or key_meta.get("data_type", "pt") != "pt"
                or value_meta.get("num_points", 0) != len(self._scene_manager.points3D)
                or value_meta.get("num_images", 0) != self.num_images
            ):
                self._logger.info("Cached visible points per image do not match current scene. Recomputing...")
                self._cache.delete_property("visible_points_per_image")

        if "visible_points_per_image" in self._cache:
            self._logger.info("Loading visible points per image from cache...")
            point_indices, _ = self._cache.get_dataset_property("visible_points_per_image", default_value=None)
            assert point_indices is not None, "Visible points per image not found in cache."
        else:
            self._logger.info("Computing and caching visible points per image...")
            # For each point, get the images that see it
            point_indices = dict()  # Map from image names to point indices
            for point_id, data in tqdm.tqdm(self._scene_manager.point3D_id_to_images.items()):
                # For each image that sees this point, add the index of the point
                # to a list of points corresponding to that image
                for image_id, _ in data:
                    point_idx = self._scene_manager.point3D_id_to_point3D_idx[point_id]
                    point_indices.setdefault(image_id, []).append(point_idx)
            point_indices = {k: np.array(v).astype(np.int32) for k, v in point_indices.items()}
            self._cache.set_dataset_property(
                "visible_points_per_image",
                data_type="pt",
                data=point_indices,
                description="Which points are visible from each image. This is a dictionary mapping image names to arrays of point indices.",
                metadata={
                    "num_points": len(self._scene_manager.points3D),
                    "num_images": self.num_images,
                },
            )

        # Create ColmapImageMetadata objects for each image
        loaded_images = [
            SfmImageMetadata(
                world_to_camera_matrix=image_world_to_cam_mats[i].copy(),
                camera_to_world_matrix=np.linalg.inv(image_world_to_cam_mats[i]).copy(),
                camera_id=image_camera_ids[i],
                camera_metadata=loaded_cameras[image_camera_ids[i]],
                image_path=str(image_absolute_paths[i].absolute()),
                mask_path=image_mask_absolute_paths[i],
                point_indices=point_indices[image_colmap_ids[i]].copy(),
                image_id=i,
            )
            for i in range(len(image_file_names))
        ]

        # Transform the points to the normalized coordinate system and cast to the right types
        # Note: we do not normalize the point errors or colors, they are already in the correct format.
        # Note: we don't transform the point errors
        points = self._scene_manager.points3D.astype(np.float32)  # type: ignore (num_points, 3)
        points_err = self._scene_manager.point3D_errors.astype(np.float32)  # type: ignore
        points_rgb = self._scene_manager.point3D_colors.astype(np.uint8)  # type: ignore

        return loaded_cameras, loaded_images, points, points_err, points_rgb
