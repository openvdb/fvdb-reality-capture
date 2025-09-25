# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
from typing import Sequence

import numpy as np

from ._load_colmap_scene import load_colmap_scene
from .sfm_cache import SfmCache
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
        - scene_bbox: An array of shape (6,) representing a bounding box containing the scene. In the form
            (bbmin_x, bbmin_y, bbmin_z, bbmax_x, bbmax_y, bbmax_z)
        - transformation_matrix: A 4x4 matrix encoding a transformation from some canonical coordinate space
            to scene coordinates.

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
        scene_bbox: np.ndarray | None,
        transformation_matrix: np.ndarray | None,
        cache: SfmCache,
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
            scene_bbox (np.ndarray): A (6)-shaped array of the form [bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z]
                defining the bounding box of the scene. If None is passed in, it will default to
                [-inf, -inf, -inf, inf, inf, inf] (i.e. all of R^3)
            transformation_matrix (np.ndarray): A 4x4 transformation matrix encoding the transformation from a reference coordinate
                system to the scene's coordinate system.
                Note that this is not applied to the scene but simply stored to track transformations applied
                to the scene (e.g. via apply_transformation_matrix).
        """
        self._cameras = cameras
        self._images = images
        self._points = points
        self._points_err = points_err
        self._points_rgb = points_rgb
        self._transformation_matrix = transformation_matrix if transformation_matrix is not None else np.eye(4)
        self._scene_bbox = scene_bbox
        self._cache = cache

    @classmethod
    def from_colmap(cls, colmap_path: str | pathlib.Path) -> "SfmScene":
        """
        Load an `SfmScene` (with a cache to store derived quantities) from the output of a COLMAP
        structure-from-motion (SfM) pipeline. COLMAP produces a directory of images, a set of
        correspondence points, as well as a lightweight SqLite database containing image poses
        (camera to world matrices), camera intrinsics (projection matrices, camera type, etc.), and
        indices of which points are seen from which images.

        Args:
            colmap_path (str | pathlib.Path): The path to the output of a COLMAP run.
        """

        if isinstance(colmap_path, str):
            colmap_path = pathlib.Path(colmap_path)

        cameras, images, points, points_err, points_rgb, cache = load_colmap_scene(colmap_path)
        return cls(
            cameras=cameras,
            images=images,
            points=points,
            points_err=points_err,
            points_rgb=points_rgb,
            scene_bbox=None,
            transformation_matrix=None,
            cache=cache,
        )

    @classmethod
    def from_e57(cls, e57_path: str | pathlib.Path, point_downsample_factor: int = 1) -> "SfmScene":
        """
        Load an `SfmScene` (with a cache to store derived quantities) from a set of E57 files.

        Args:
            e57_path (str | pathlib.Path): The path to a directory containing E57 files.
            point_downsample_factor (int): Factor by which to downsample the points loaded from the E57 files.
                Defaults to 1 (i.e. no downsampling).
        """

        if isinstance(e57_path, str):
            e57_path = pathlib.Path(e57_path)

        from ._load_e57_scene import load_e57_dataset

        cameras, images, points, points_rgb, points_err, cache = load_e57_dataset(e57_path, point_downsample_factor)
        return cls(
            cameras=cameras,
            images=images,
            points=points,
            points_err=points_err,
            points_rgb=points_rgb,
            scene_bbox=None,
            transformation_matrix=None,
            cache=cache,
        )

    def filter_points(self, mask: np.ndarray | Sequence[bool]) -> "SfmScene":
        """
        Filter the points in the scene based on a boolean mask.

        Args:
            mask (np.ndarray | Sequence[bool]): A boolean array of shape (N,) where N is the number of points.
                               True values indicate that the corresponding point should be kept.

        Returns:
            SfmScene: A new SfmScene instance with filtered points and corresponding metadata.
        """
        visible_point_indices = set(np.argwhere(mask).ravel().tolist())
        remap_indices = np.cumsum(mask, dtype=int)
        filtered_images = []
        image_meta: SfmImageMetadata
        for image_meta in self._images:
            old_visible_points = set(image_meta.point_indices.tolist())
            old_visible_points_filtered = old_visible_points.intersection(visible_point_indices)
            remapped_points = remap_indices[np.array(list(old_visible_points_filtered), dtype=np.int64)] - 1
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
            scene_bbox=self._scene_bbox,
            transformation_matrix=self._transformation_matrix,
            cache=self.cache,
        )

    def filter_images(self, mask: np.ndarray | Sequence[bool]) -> "SfmScene":
        """
        Filter the images in the scene based on a Boolean mask.

        Args:
            mask (np.ndarray | Sequence[bool]): A Boolean array of shape (M,) where M is the number of images.
                               True values indicate that the corresponding image should be kept.

        Returns:
            SfmScene: A new SfmScene instance with filtered images and corresponding metadata.
        """
        filtered_images = [img for img, keep in zip(self._images, mask) if keep]
        return SfmScene(
            cameras=self._cameras,
            images=filtered_images,
            points=self._points,
            points_err=self._points_err,
            points_rgb=self._points_rgb,
            scene_bbox=self._scene_bbox,
            transformation_matrix=self._transformation_matrix,
            cache=self.cache,
        )

    def select_images(self, indices: np.ndarray | Sequence[int]) -> "SfmScene":
        """
        Select specific images from the scene based on their indices.

        Args:
            indices (np.ndarray | Sequence[int]): An array of integer indices specifying which images to select.

        Returns:
            SfmScene: A new SfmScene instance with the selected images and corresponding metadata.
        """
        filtered_images = [self._images[i] for i in indices]
        return SfmScene(
            cameras=self._cameras,
            images=filtered_images,
            points=self._points,
            points_err=self._points_err,
            points_rgb=self._points_rgb,
            scene_bbox=self._scene_bbox,
            transformation_matrix=self._transformation_matrix,
            cache=self.cache,
        )

    def apply_transformation_matrix(self, transformation_matrix: np.ndarray) -> "SfmScene":
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
        transformation_matrix = transformation_matrix @ self._transformation_matrix
        bbox = self._scene_bbox
        if self._scene_bbox is not None:
            bbmin = self._scene_bbox[:3]
            bbmax = self._scene_bbox[3:]
            bbmin = transformation_matrix[:3, :3] @ bbmin + transformation_matrix[:3, 3]
            bbmax = transformation_matrix[:3, :3] @ bbmax + transformation_matrix[:3, 3]
            bbox = np.concatenate([bbmin, bbmax])

        return SfmScene(
            cameras=self._cameras,
            images=transformed_images,
            points=transformed_points,
            points_err=self._points_err,
            points_rgb=self._points_rgb,
            scene_bbox=bbox,
            transformation_matrix=transformation_matrix,
            cache=self.cache,
        )

    @property
    def cache(self) -> SfmCache:
        return self._cache

    @property
    def image_centers(self):
        """
        Returns the position where each image was captured in the scene.

        Returns:
            np.ndarray: A (N, 3) array representing the 3D positions of the image centers.
        """

        if not self._images:
            return np.zeros((0, 3))
        return np.stack([img.origin for img in self.images])

    @property
    def image_sizes(self) -> np.ndarray:
        """
        Return the dimensions of each image in the scene as a numpy array of shape (N, 2)
        where N is the number of images and each entry is (height, width).

        Returns:
            np.ndarray: A (N, 2) array representing the dimensions of each image in the scene.
        """
        if not self._images:
            return np.zeros((0, 2), dtype=int)
        return np.array([[img.camera_metadata.height, img.camera_metadata.width] for img in self._images])

    @property
    def transformation_matrix(self) -> np.ndarray:
        """
        Return the 4x4 transformation matrix for the scene which encodes the transformation
        from some reference coordinate system to the scene's coordinates.
        """
        return self._transformation_matrix

    @property
    def world_to_camera_matrices(self) -> np.ndarray:
        """
        Return the world-to-camera matrices for each image in the scene.

        Returns:
            np.ndarray: A (N, 4, 4) array representing the world-to-camera transformation matrices.
        """
        if not self._images:
            return np.zeros((0, 4, 4))
        return np.stack([image.world_to_camera_matrix for image in self._images], axis=0)

    @property
    def camera_to_world_matrices(self) -> np.ndarray:
        """
        Return the camera-to-world matrices for each image in the scene.

        Returns:
            np.ndarray: A (N, 4, 4) array representing the camera-to-world transformation matrices.
        """
        if not self._images:
            return np.zeros((0, 4, 4))
        return np.stack([image.camera_to_world_matrix for image in self._images], axis=0)

    @property
    def projection_matrices(self) -> np.ndarray:
        """
        Return the projection matrices for each image in the scene.

        Returns:
            np.ndarray: A (N, 3, 3) array representing the projection matrices.
        """
        if not self._images:
            return np.zeros((0, 3, 3))
        return np.stack([image.camera_metadata.projection_matrix for image in self._images], axis=0)

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

    @property
    def scene_bbox(self) -> np.ndarray:
        """
        Return the clip bounds of the scene as a numpy array of shape (6,) in the form
        [xmin, ymin, zmin, xmax, ymax, zmax].

        By default, the clip bounds are [-inf, -inf, -inf, inf, inf, inf].

        Returns:
            np.ndarray: A 1D array of shape (6,) representing the bounding box of the scene.
                        If the bounding box has not been computed, it returns [-inf, -inf, -inf, inf, inf, inf].
        """
        if self._scene_bbox is None:
            # Calculate the bounding box of the scene if not already computed
            return np.array([-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf])
        else:
            return self._scene_bbox
