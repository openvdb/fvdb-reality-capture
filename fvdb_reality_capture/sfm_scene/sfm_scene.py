# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
from typing import Any, Sequence

import numpy as np

from ._load_colmap_scene import load_colmap_scene
from .sfm_cache import SfmCache
from .sfm_metadata import SfmCameraMetadata, SfmPosedImageMetadata


class SfmScene:
    """
    Lightweight in-memory representation of a scene extracted from a structure-from-motion (SFM) pipeline such as
    `COLMAP <https://github.com/colmap/colmap>`_ or `GLOMAP <https://github.com/colmap/glomap>`_.

    This class does not load large data, but instead stores metadata about the scene, including camera parameters,
    image paths, and 3D points. It also provides methods to manipulate and transform the scene, such as filtering points
    or images, applying transformations, and accessing camera and image properties.

    .. note:: The :class:`SfmScene` class is *immutable*. Methods that modify the scene (e.g. filtering points or images,
            applying transformations) return new instances rather than modifying the existing one.

    .. note:: The :class:`SfmScene` class does not load or store the actual image data. It only stores
              paths to the images on disk.


    In general, an :class:`SfmScene` consists of the following components:

    **Cameras** (:obj:`cameras`): A dictionary mapping unique integer camera identifiers to
    :class:`SfmCameraMetadata` objects which contain information about each camera used to capture the scene
    (e.g. focal length, distortion parameters).

    **Posed Images** (:obj:`images`): A list of :class:`SfmImageMetadata` objects containing metadata for each posed
    image in the scene (e.g. the ID of the camera that captured it, the path to the image on disk, the camera to
    world matrix, etc.).

    **Points** (:obj:`points`): An Nx3 array of 3D points in the scene, where ``N`` is the number of points.
    **Point Errors** (:obj:`points_err`): An array of shape ``(N,)`` representing the error or uncertainty of each
    point where ``N`` is the number of points.

    **Point RGB Colors** (:obj:`points_rgb`): An Nx3 uint8 array of RGB color values for each point in the
    scene, where ``N`` is the number of points.

    **Scene Bounding Box** (:obj:`scene_bbox`): An array of shape (6,) representing a bounding box containing the scene.
    In the form ``(bbmin_x, bbmin_y, bbmin_z, bbmax_x, bbmax_y, bbmax_z)``.

    **Transformation Matrix** (:obj:`transformation_matrix`): A 4x4 matrix encoding a transformation from some canonical
    coordinate space to scene coordinates.

    **Cache** (:obj:`cache`): An :class:`SfmCache` object representing a cache folder for storing quantities
    derived from the scene (e.g. depth maps, feature matches, etc.).
    """

    def __init__(
        self,
        cameras: dict[int, SfmCameraMetadata],
        images: Sequence[SfmPosedImageMetadata],
        points: np.ndarray,
        points_err: np.ndarray,
        points_rgb: np.ndarray,
        scene_bbox: np.ndarray | None,
        transformation_matrix: np.ndarray | None,
        cache: SfmCache,
    ):
        """
        Initialize an :class:`SfmScene` instance from the given components.

        Args:
            cameras (dict[int, SfmCameraMetadata]): A dictionary mapping camera IDs to :class:`SfmCameraMetadata`
                objects containing information about each camera used to capture the scene
                (e.g. focal length, distortion parameters, etc.).
            images (Sequence[SfmImageMetadata]): A sequence of :class:`SfmImageMetadata` objects containing metadata
                for each image in the scene (e.g. camera ID, image path, view transform, etc.).
            points (np.ndarray): An ``(N, 3)``-shaped array of 3D points in the scene, where ``N`` is the number of points.
            points_err (np.ndarray): An array of shape  ``(N,)`` representing the error or uncertainty of each point
                in ``points``.
            points_rgb (np.ndarray): An ``(N,3)``-shaped uint8 array of RGB color values for each point in the scene,
                where ``N`` is the number of points.
            scene_bbox (np.ndarray | None): A ``(6,)``-shaped array of the form
                ``[bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z]`` defining the bounding box of the scene.
                If ``None`` is passed in, it will default to ``[-inf, -inf, -inf, inf, inf, inf]``
                (i.e. all of :math:`\\mathbb{R}^3`)
            transformation_matrix (np.ndarray | None): A 4x4 transformation matrix encoding the transformation from a reference
                coordinate system to the scene's coordinate system. Note that this is not applied to the scene but simply
                stored to track transformations applied to the scene (e.g. via :meth:`apply_transformation_matrix`).
                If ``None`` is passed in, it will default to the identity matrix.
        """
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._cameras = cameras
        self._images = list(images)
        self._points = points
        self._points_err = points_err
        self._points_rgb = points_rgb
        self._transformation_matrix = transformation_matrix if transformation_matrix is not None else np.eye(4)
        self._scene_bbox = scene_bbox
        self._cache = cache

        # Validate the scene_bbox
        if self._scene_bbox is not None and np.any(np.isinf(self._scene_bbox)):
            if not np.all(np.isinf(self._scene_bbox)):
                self._logger.warning(
                    "scene_bbox contains a mix of finite and infinite values. Setting scene_bbox to None."
                )
            self._scene_bbox = None

        # If there are images, check if they have point indices
        # Images can either all have point indices or none of them should
        if len(images) > 0:
            all_images_have_same_visibility = len(set([img.point_indices is None for img in images])) == 1
            if not all_images_have_same_visibility or ():
                raise ValueError("All images must either have point indices or none of them should.")
            self._has_point_indices = images[0].point_indices is not None if images else False
        else:
            # In the case, where there are no images, we default to saying there are no point indices
            # which makes some semantic sense because images are required to have point indices
            self._has_point_indices = False

    @classmethod
    def from_colmap(cls, colmap_path: str | pathlib.Path) -> "SfmScene":
        """
        Load an :class:`SfmScene` (with a cache to store derived quantities) from the output of a
        `COLMAP <https://github.com/colmap/colmap>`_  structure-from-motion (SfM) pipeline.
        COLMAP produces a directory of images, a set of 3D correspondence points, as well as a lightweight SqLite
        database containing image poses (camera to world matrices), camera intrinsics
        (projection matrices, camera type, etc.), and indices of which points are seen from which images.

        These are loaded into memory as an :class:`SfmScene` object which provides easy access to the camera parameters,
        image paths, and 3D points, as well as methods to manipulate and transform the scene.

        Args:
            colmap_path (str | pathlib.Path): The path to the output of a COLMAP run.

        Returns:
            scene (SfmScene): An in-memory representation of the loaded scene.
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
        Load an :class:`SfmScene` (with a cache to store derived quantities) from a set of E57 files.

        Args:
            e57_path (str | pathlib.Path): The path to a directory containing E57 files.
            point_downsample_factor (int): Factor by which to downsample the points loaded from the E57 files.
                Defaults to 1 (i.e. no downsampling).

        Returns:
            scene (SfmScene): An in-memory representation of the loaded scene.
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

    @classmethod
    def from_simple_directory(cls, data_path: str | pathlib.Path) -> "SfmScene":
        """
        Load an :class:`SfmScene` (with a cache to store derived quantities) from a simple directory structure
        containing images, camera parameters (stored as JSON), and 3D points (stored as a PLY).

        The directory should contain:

        - **images/**: A directory of images.

        - **cameras.json**: A JSON file containing camera parameters.
          The cameras.json file is a list of dictionaries, each containing the following keys:

            - ``"camera_name"``: The name of the image file.
            - ``"width"``: The width of the image.
            - ``"height"``: The height of the image.
            - ``"camera_intrinsics"``: The perspective projection matrix
            - ``"world_to_camera"``: The world-to-camera transformation matrix.
            - ``"image_path"``: The path to the image file relative to the images directory.

        - **points.ply**: A PLY file containing 3D points.

        Args:
            data_path (str | pathlib.Path): The path to the data directory.

        Returns:
            scene (SfmScene): An in-memory representation of the loaded scene.
        """

        if isinstance(data_path, str):
            data_path = pathlib.Path(data_path)

        from ._load_simple_scene import load_simple_scene

        cameras, images, points, points_rgb, points_err, cache = load_simple_scene(data_path)
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

    def state_dict(self) -> dict[str, Any]:
        """
        Get a state dictionary representing the SfmScene. This can be used to serialize the scene to disk
        or to create a new SfmScene instance via :meth:`from_state_dict`.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing the state of the SfmScene.
        """
        return {
            "cameras": {k: v.state_dict() for k, v in self._cameras.items()},
            "images": [img.state_dict() for img in self._images],
            "points": self._points,
            "points_err": self._points_err,
            "points_rgb": self._points_rgb,
            "scene_bbox": self._scene_bbox.tolist() if self._scene_bbox is not None else None,
            "transformation_matrix": (
                self._transformation_matrix.tolist() if self._transformation_matrix is not None else None
            ),
            "cache_path": self._cache.db_path.parent.absolute().as_posix(),
            "cache_name": self._cache.cache_name,
            "cache_description": self._cache.cache_description,
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "SfmScene":
        """
        Create an :class:`SfmScene` from a state dictionary previously obtained via :meth:`state_dict`.

        Args:
            state_dict (dict[str, Any]): A state dictionary representing the SfmScene originally created with :meth:`state_dict`.

        Returns:
            scene (SfmScene): An in-memory representation of the loaded scene.
        """
        if "images" not in state_dict:
            raise KeyError("State dictionary is missing 'images' key.")
        if "cameras" not in state_dict:
            raise KeyError("State dictionary is missing 'cameras' key.")
        if "points" not in state_dict:
            raise KeyError("State dictionary is missing 'points' key.")
        if "points_err" not in state_dict:
            raise KeyError("State dictionary is missing 'points_err' key.")
        if "points_rgb" not in state_dict:
            raise KeyError("State dictionary is missing 'points_rgb' key.")
        if "scene_bbox" not in state_dict:
            raise KeyError("State dictionary is missing 'scene_bbox' key.")
        if "transformation_matrix" not in state_dict:
            raise KeyError("State dictionary is missing 'transformation_matrix' key.")
        if "cache_path" not in state_dict:
            raise KeyError("State dictionary is missing 'cache_path' key.")
        if "cache_name" not in state_dict:
            raise KeyError("State dictionary is missing 'cache_name' key.")
        if "cache_description" not in state_dict:
            raise KeyError("State dictionary is missing 'cache_description' key.")

        cache_path = pathlib.Path(state_dict["cache_path"])
        if not cache_path.exists():
            raise ValueError(f"Cache path {cache_path} does not exist.")
        cache_name = state_dict["cache_name"]
        cache_description = state_dict["cache_description"]
        if not isinstance(cache_name, str):
            raise ValueError("Cache name must be a string.")
        if not isinstance(cache_description, str):
            raise ValueError("Cache description must be a string.")
        cache = SfmCache.get_cache(cache_path, name=cache_name, description=cache_description)

        cameras = {int(k): SfmCameraMetadata.from_state_dict(v) for k, v in state_dict["cameras"].items()}
        images = [SfmPosedImageMetadata.from_state_dict(img_dict, cameras) for img_dict in state_dict["images"]]
        points = np.array(state_dict["points"], dtype=np.float32)
        points_err = np.array(state_dict["points_err"], dtype=np.float32)
        points_rgb = np.array(state_dict["points_rgb"], dtype=np.uint8)
        scene_bbox = (
            np.array(state_dict["scene_bbox"], dtype=np.float32) if state_dict["scene_bbox"] is not None else None
        )
        transformation_matrix = (
            np.array(state_dict["transformation_matrix"], dtype=np.float32)
            if state_dict["transformation_matrix"] is not None
            else None
        )
        return cls(
            cameras,
            images,
            points,
            points_err,
            points_rgb,
            scene_bbox,
            transformation_matrix,
            cache,
        )

    def filter_points(self, mask: np.ndarray | Sequence[bool]) -> "SfmScene":
        """
        Return a new :class:`SfmScene` instance containing only the points for which the mask is ``True``.

        Args:
            mask (np.ndarray | Sequence[bool]): A boolean array of shape ``(N,)`` where ``N`` is the number of points.
                ``True`` values indicate that the corresponding point should be kept.

        Returns:
            SfmScene: A new :class:`SfmScene` instance with filtered points and corresponding metadata.
        """
        visible_point_indices = set(np.argwhere(mask).ravel().tolist())
        remap_indices = np.cumsum(mask, dtype=int)
        filtered_images = []
        image_meta: SfmPosedImageMetadata
        for image_meta in self._images:
            new_point_indices = None
            if image_meta.point_indices is not None:
                old_visible_points = set(image_meta.point_indices.tolist())
                old_visible_points_filtered = old_visible_points.intersection(visible_point_indices)
                new_point_indices = remap_indices[np.array(list(old_visible_points_filtered), dtype=np.int64)] - 1
            filtered_images.append(
                SfmPosedImageMetadata(
                    world_to_camera_matrix=image_meta.world_to_camera_matrix,
                    camera_to_world_matrix=image_meta.camera_to_world_matrix,
                    camera_metadata=image_meta.camera_metadata,
                    camera_id=image_meta.camera_id,
                    image_path=image_meta.image_path,
                    mask_path=image_meta.mask_path,
                    point_indices=new_point_indices,
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
        Return a new :class:`SfmScene` instance containing only the images for which the mask is ``True``.

        Args:
            mask (np.ndarray | Sequence[bool]): A Boolean array of shape ``(I,)`` where ``I`` is the number of images.
                ``True`` values indicate that the corresponding image should be kept.
        Returns:
            SfmScene: A new :class:`SfmScene` instance with filtered images and corresponding metadata.
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
        Return a new :class:`SfmScene` instance containing only the images specified by the given indices.

        Args:
            indices (np.ndarray | Sequence[int]): An array of integer indices specifying which images to select.
                The indices should be in the range ``[0, num_images - 1]``.

        Returns:
            SfmScene: A new :class:`SfmScene` instance with the selected images and corresponding metadata.
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
        Return a new :class:`SfmScene` instance with the transformation applied.

        The transformation applies to the camera poses and the 3D points in the scene.

        Args:
            transformation_matrix (np.ndarray): A 4x4 transformation matrix to apply to the scene.

        Returns:
            SfmScene: A new :class:`SfmScene` instance with the transformed cameras and points.
        """
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be a 4x4 matrix.")

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
        """
        Return an :class:`SfmCache` object representing a cache folder for storing quantities derived from the scene.

        Returns:
            cache (SfmCache): The :class:`SfmCache` object associated with this scene.
        """
        return self._cache

    @property
    def has_visible_point_indices(self) -> bool:
        """
        Return whether the images in the scene have point indices indicating which 3D points are visible in each image.

        Returns:
            has_visible_point_indices (bool): ``True`` if the images have point indices, ``False`` otherwise.
        """
        return self._has_point_indices

    @property
    def median_depth_per_image(self) -> np.ndarray:
        """
        Return an array containing the median depth of the points observed in each image.

        Returns:
            median_depth_per_image (np.ndarray): An array of shape ``(I,)`` where ``I`` is the number of images.
                Each value represents the median depth of the points observed in the corresponding image.
                If this scene does not have visible points per image (*i.e* :obj:`has_visible_point_indices` is ``False``),
                an array of ``np.nan`` values is returned.
        """
        if not self._has_point_indices:
            return np.full((len(self._images),), np.nan, dtype=np.float32)

        if len(self._images) == 0:
            return np.zeros((0,), dtype=np.float32)

        median_depths = np.full((len(self._images),), np.nan, dtype=np.float32)
        for i, img in enumerate(self._images):
            if img.point_indices is None or len(img.point_indices) == 0:
                continue
            points_in_image = self._points[img.point_indices]
            camera_origin = img.origin.reshape(1, 3)
            depths = np.linalg.norm(points_in_image - camera_origin, axis=1)
            if len(depths) > 0:
                median_depths[i] = np.median(depths)
        return median_depths

    @property
    def image_camera_positions(self) -> np.ndarray:
        """
        Returns the position where each posed image was captured in the scene (i.e. the position of the camera
        when it captured the image).

        Returns:
            image_camera_positions (np.ndarray): A ``(I, 3)``-shaped array representing the 3D positions of the camera positions that captured
                each posed image in the scene, where ``I`` is the number of images.
        """

        if not self._images:
            return np.zeros((0, 3))
        return np.stack([img.origin for img in self.images])

    @property
    def image_sizes(self) -> np.ndarray:
        """
        Return the resolution of each posed image in the scene as a numpy array of shape (N, 2)
        where ``N`` is the number of images and each entry is (height, width).

        Returns:
            image_sizes (np.ndarray): A ``(I, 2)``-shaped array representing the resolution of each posed
                image in the scene, where ``I`` is the number of images, `image_sizes[i, 0]` is the height of image ``i``,
                and `image_sizes[i, 1]` is the width of image ``i``.
        """
        if not self._images:
            return np.zeros((0, 2), dtype=int)
        return np.array([[img.camera_metadata.height, img.camera_metadata.width] for img in self._images])

    @property
    def transformation_matrix(self) -> np.ndarray:
        """
        Return the 4x4 transformation matrix for the scene. This matrix encodes the transformation
        from the coordinate system the scene was loaded in to the current scene's coordinates.

        Returns:
            transformation_matrix (np.ndarray): A 4x4 numpy array representing the transformation matrix from
                the original coordinate system to the current scene's coordinate system.
        """
        return self._transformation_matrix

    @property
    def world_to_camera_matrices(self) -> np.ndarray:
        """
        Return the world-to-camera matrices for each posed image in the scene.

        Returns:
            world_to_camera_matrices (np.ndarray): An ``(I, 4, 4)``-shaped array representing the world-to-camera
                transformation matrix of each posed image in the scene, where `I` is the number of images.
        """
        if not self._images:
            return np.zeros((0, 4, 4))
        return np.stack([image.world_to_camera_matrix for image in self._images], axis=0)

    @property
    def camera_to_world_matrices(self) -> np.ndarray:
        """
        Return the camera-to-world matrices for each posed image in the scene.

        Returns:
            camera_to_world_matrices (np.ndarray): An ``(I, 4, 4)``-shaped array representing the camera-to-world
                transformation matrix of each posed image in the scene, where `I` is the number of images.
        """
        if not self._images:
            return np.zeros((0, 4, 4))
        return np.stack([image.camera_to_world_matrix for image in self._images], axis=0)

    @property
    def projection_matrices(self) -> np.ndarray:
        """
        Return the projection matrices for each posed image in the scene.

        Returns:
            projection_matrices (np.ndarray): An ``(I, 3, 3)``-shaped array representing the projection matrix of each
                posed image in the scene, where `I` is the number of images. The projection matrix maps 3D points in camera
                coordinates to 2D points in pixel coordinates.
        """
        if not self._images:
            return np.zeros((0, 3, 3))
        return np.stack([image.camera_metadata.projection_matrix for image in self._images], axis=0)

    @property
    def num_images(self) -> int:
        """
        Return the total number of posed images in the scene.

        Returns:
            num_images (int): The number of posed images in the scene.
        """
        return len(self._images)

    @property
    def num_cameras(self) -> int:
        """
        Return the total number of cameras used to capture the scene.

        Returns:
            num_cameras (int): The number of cameras in the scene.
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
    def images(self) -> list[SfmPosedImageMetadata]:
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
        Get the 3D points in the scene as a numpy array of shape ``(N, 3)``.

        Note: The points are in the same coordinate system as the camera poses.

        Returns:
            points (np.ndarray): An ``(N, 3)``-shaped array of 3D points in the scene where ``N`` is the number of points.
        """
        return self._points

    @property
    def points_err(self) -> np.ndarray:
        """
        Return an un-normalized confidence value for each point in the scene (see :obj:`points`).

        The error is a measure of the uncertainty in the 3D point position, typically derived from the SFM pipeline.

        Returns:
            points_err (np.ndarray): An array of shape ``(N,)`` where ``N`` is the number of points in the scene.
                        ``points_err[i]`` encodes the error or uncertainty of the i-th corresponding
                        point in :obj:`points`.
        """
        return self._points_err

    @property
    def points_rgb(self) -> np.ndarray:
        """
        Return the RGB color values for each point in the scene as a uint8 array of shape ``(N, 3)`` where ``N`` is the number of points.

        Returns:
            points_rgb (np.ndarray): An ``(N, 3)``-shaped uint8 array of RGB color values for each point in the scene where ``N`` is the number of points.
        """
        return self._points_rgb

    @property
    def scene_bbox(self) -> np.ndarray:
        """
        Return the clip bounds of the scene as a numpy array of shape ``(6,)`` in the form
        ```[xmin, ymin, zmin, xmax, ymax, zmax]``.

        If the scene was not constructed with a bounding box, the default clip bounds are ``[-inf, -inf, -inf, inf, inf, inf]``.

        Returns:
            scene_bbox (np.ndarray): A 1D array of shape ``(6,)`` representing the bounding box of the scene.
                        If the scene was not constructed with a bounding box, then return ``[-inf, -inf, -inf, inf, inf, inf]``.
        """
        if self._scene_bbox is None:
            # Calculate the bounding box of the scene if not already computed
            return np.array([-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf])
        else:
            return self._scene_bbox
