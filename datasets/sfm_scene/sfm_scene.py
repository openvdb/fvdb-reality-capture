# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from .base_dataset_reader import BaseDatasetReader
from .sfm_metadata import SfmCameraMetadata, SfmImageMetadata


class SfmScene:
    """
    Class representing a scene extracted from a structure-from-motion (SFM) pipeline such as COLMAP or GLOMAP.
    The scene consists of:
        - cameras: A dictionary mapping unique integer camera identifiers to `SfmCameraMetadata` objects
                   which contain information about each camera used to capture the scene (e.g. focal length,
                   distortion parameters).
        - images: A list of `SfmImageMetadata` objects containing metadata for each posed image in the scene (e.g. camera ID,
                  image path, view transform, etc.).
        - points: An Nx3 array of 3D points in the scene, where N is the number of points.
        - points_err: An array of shape (N,) representing the error or uncertainty of each point in `points`.
        - points_rgb: An Nx3 uint8 array of RGB color values for each point in the scene, where N is the number of points.

    The scene can be transformed using a 4x4 transformation matrix, which applies to both the camera poses and the 3D points in the scene.
    The scene also provides properties to access the world-to-camera and camera-to-world matrices,
    the scale of the scene, and the number of images and cameras.
    """

    def __init__(
        self,
        cameras: dict[int, SfmCameraMetadata],
        images: list[SfmImageMetadata],
        points: np.ndarray,
        points_err: np.ndarray,
        points_rgb: np.ndarray,
    ):
        """
        Initialize the SfmScene with cameras, images, and points.

        Args:
            cameras (dict[int, SfmCameraMetadata]): A dictionary mapping camera IDs to `SfmCameraMetadata` objects
                                                     containing information about each camera used to capture the
                                                     scene (e.g. focal length, distortion parameters, etc.).
            images (list[SfmImageMetadata]): A list of `SfmImageMetadata` objects containing metadata for each image
                                              in the scene (e.g. camera ID, image path, view transform, etc.).
            points (np.ndarray): An Nx3 array of 3D points in the scene,
                                 where N is the number of points.
            points_err (np.ndarray): An array of shape (N,) representing the error or uncertainty
                                     of each point in `points`.
            points_rgb (np.ndarray): An Nx3 uint8 array of RGB color values for each point in the scene,
                                     where N is the number of points.
        """
        self._cameras = cameras
        self._images = images
        self._points = points
        self._points_err = points_err
        self._points_rgb = points_rgb

        # Calculate the maximum distance from the average point of the scene to any point
        # which defines a notion of scene scale
        camera_locations = np.stack([img.origin for img in self._images], axis=0)
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self._scene_scale = np.max(dists)

    @classmethod
    def from_dataset_reader(
        cls,
        dataset_reader: BaseDatasetReader,
    ) -> "SfmScene":
        """
        Create an SfmScene instance from a dataset reader.

        Args:
            dataset_path: Path to the dataset directory.
            dataset_reader: An instance of BaseDatasetReader that provides the data.

        Returns:
            An instance of SfmScene containing the loaded data.
        """
        (
            cameras,
            images,
            points,
            points_err,
            points_rgb,
        ) = dataset_reader.load_data()

        return cls(cameras, images, points, points_err, points_rgb)

    def filter_points(self, mask: np.ndarray) -> "SfmScene":
        """
        Filter the points in the scene based on a boolean mask.

        Args:
            mask (np.ndarray): A boolean array of shape (N,) where N is the number of points.
                               True values indicate that the corresponding point should be kept.

        Returns:
            SfmScene: A new SfmScene instance with filtered points and corresponding metadata.
        """
        visible_point_indices = set(np.argwhere(mask)[0].tolist())
        remap_indices = np.cumsum(mask, dtype=int)
        filtered_images = []
        image_meta: SfmImageMetadata
        for image_meta in self._images:
            old_visible_points = set(image_meta.point_indices.tolist())
            old_visible_points_filtered = old_visible_points.intersection(visible_point_indices)
            remapped_points = remap_indices[np.array(list(old_visible_points_filtered), dtype=np.int64)]
            filtered_images.append(
                SfmImageMetadata(
                    world_to_camera_matrix=image_meta.world_to_camera_matrix,
                    camera_to_world_matrix=image_meta.camera_to_world_matrix,
                    camera_metadata=image_meta.camera_metadata,
                    camera_id=image_meta.camera_id,
                    image_path=image_meta.image_path,
                    mask_path=image_meta.mask_path,
                    point_indices=remapped_points,
                    image_id=image_meta.image_id,
                )
            )

        filtered_points = self._points[mask]
        filtered_points_err = self._points_err[mask]
        filtered_points_rgb = self._points_rgb[mask]

        return SfmScene(
            cameras=self._cameras,
            images=filtered_images,
            points=filtered_points,
            points_err=filtered_points_err,
            points_rgb=filtered_points_rgb,
        )

    def transform(self, transformation_matrix: np.ndarray) -> "SfmScene":
        """
        Apply a transformation to the scene using a 4x4 transformation matrix.

        The transformation applies to the camera poses and the 3D points in the scene.

        Note: This does not modify the original scene, but returns a new SfmScene instance with the transformed data.

        Args:
            transformation_matrix (np.ndarray): A 4x4 transformation matrix to apply to the scene.

        Returns:
            SfmScene: A new SfmScene instance with the transformed cameras and points.
        """
        camera_locations = []
        transformed_images = []
        for image in self._images:
            transformed_images.append(image.transform(transformation_matrix))
            camera_locations.append(image.origin)

        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be a 4x4 matrix.")

        transformed_points = self._points @ transformation_matrix[:3, :3].T + transformation_matrix[:3, 3]

        return SfmScene(
            cameras=self._cameras,
            images=transformed_images,
            points=transformed_points,
            points_err=self._points_err,
            points_rgb=self._points_rgb,
        )

    @property
    def world_to_camera_matrices(self) -> np.ndarray:
        """
        Return the world-to-camera matrices for each image in the scene.

        Returns:
            np.ndarray: A (N, 4, 4) array representing the world-to-camera transformation matrices.
        """
        return np.stack([image.world_to_camera_matrix for image in self._images], axis=0)

    @property
    def camera_to_world_matrices(self) -> np.ndarray:
        """
        Return the camera-to-world matrices for each image in the scene.

        Returns:
            np.ndarray: A (N, 4, 4) array representing the camera-to-world transformation matrices.
        """
        return np.stack([image.camera_to_world_matrix for image in self._images], axis=0)

    @property
    def scene_scale(self) -> float:
        """
        Return the scale of the scene, defined as the maximum distance from the average point of the scene to any point.

        Returns:
            float: The scale of the scene.
        """
        return self._scene_scale

    @property
    def num_images(self) -> int:
        """
        Return the total number of images used to capture the scene.

        Returns:
            int: The number of images in the scene.
        """
        return len(self._images)

    @property
    def num_cameras(self) -> int:
        """
        Return the total number of cameras used to capture the scene.

        Returns:
            int: The number of cameras in the scene.
        """
        return len(self._cameras)

    @property
    def cameras(self) -> dict[int, SfmCameraMetadata]:
        """
        Return a dictionary mapping unique (integer) camera identifiers to `SfmCameraMetadata` objects
        which contain information about each camera used to capture the scene
        (e.g. its focal length, projection matrix, etc.).

        Returns:
            dict[int, SfmCameraMetadata]: A dictionary mapping camera IDs to `SfmCameraMetadata` objects.
        """
        return self._cameras

    @property
    def images(self) -> list[SfmImageMetadata]:
        """
        Get a list of image metadata objects (`SfmImageMetadata`) with information about each image
        in the scene (e.g. it's camera ID, path on the filesystem, etc.).

        Returns:
            list[SfmImageMetadata]: A list of `SfmImageMetadata` objects containing metadata
                                    for each image in the scene.
        """
        return self._images

    @property
    def points(self) -> np.ndarray:
        """
        Get the 3D points reconstructed in the scene as a numpy array of shape (N, 3),

        Note: The points are in the same coordinate system as the camera poses.

        Returns:
            np.ndarray: An Nx3 array of 3D points in the scene.
        """
        return self._points

    @property
    def points_err(self) -> np.ndarray:
        """
        Return an un-normalized confidence value for each point (seel `points`) in the scene.

        The error is a measure of the uncertainty in the 3D point position, typically derived from the SFM pipeline.

        Returns:
            np.ndarray: An array of shape (N,) where N is the number of points in the scene.
                        Each value represents the error or uncertainty of the corresponding point in `points`.
        """
        return self._points_err

    @property
    def points_rgb(self) -> np.ndarray:
        """
        Return the RGB color values for each point in the scene as a uint8 array of shape (N, 3) where N is the number of points.

        Returns:
            np.ndarray: An Nx3 uint8 array of RGB color values for each point in the scene where N is the number of points.
        """
        return self._points_rgb
