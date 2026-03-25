# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Any

from fvdb import CameraModel
import numpy as np

_DISTORTION_COEFFS_SHAPE = (12,)
"""
Shape of the canonical packed FVDB distortion coefficient vector.

The packed layout is ``[k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]``.
"""


def _as_packed_distortion_coeffs(
    coeffs: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """
    Coerce distortion coefficients into a NumPy array in the canonical packed FVDB layout.

    This helper exists so callers can pass a convenient Python sequence or array-like object
    while `SfmCameraMetadata` stores one canonical in-memory representation. Centralizing the
    coercion and validation here keeps the constructor logic simple and ensures all call sites
    produce a float32 NumPy array with a consistent packed layout.

    Args:
        coeffs: Distortion coefficients in packed FVDB order
            ``[k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]`` or an empty sequence.

    Returns:
        np.ndarray: A float32 NumPy array containing either 12 packed coefficients or an empty array.
    """
    coeff_array = np.asarray(coeffs, dtype=np.float32)
    if coeff_array.ndim > 1:
        raise ValueError(f"distortion_coeffs must have shape {_DISTORTION_COEFFS_SHAPE}, got {coeff_array.shape}")
    coeff_array = coeff_array.reshape(-1)
    if coeff_array.size == 0:
        return np.empty((0,), dtype=np.float32)
    if coeff_array.size != _DISTORTION_COEFFS_SHAPE[0]:
        raise ValueError(f"distortion_coeffs must have shape {_DISTORTION_COEFFS_SHAPE}, got {coeff_array.shape}")
    return coeff_array


def _legacy_camera_type_to_camera_model(camera_type: str) -> CameraModel:
    """
    Map a legacy serialized ``camera_type`` string onto the canonical FVDB camera model.

    This exists only to preserve backwards compatibility when loading older scene metadata
    that predated the move from ``SfmCameraType`` to ``fvdb.CameraModel``.

    Args:
        camera_type: Legacy serialized camera type string.

    Returns:
        CameraModel: The closest matching canonical FVDB camera model.
    """
    if camera_type in ("PINHOLE", "SIMPLE_PINHOLE"):
        return CameraModel.PINHOLE
    if camera_type in ("SIMPLE_RADIAL", "RADIAL", "OPENCV"):
        return CameraModel.OPENCV_RADTAN_5
    raise ValueError(f"Unsupported legacy camera_type {camera_type}")


def _legacy_distortion_parameters_to_coeffs(camera_type: str, distortion_parameters: np.ndarray) -> np.ndarray:
    """
    Convert legacy serialized distortion parameters into packed FVDB distortion coefficients.

    Older checkpoints/scenes stored distortion using ``SfmCameraType``-specific layouts. This
    helper translates those layouts into the canonical FVDB packed representation
    ``[k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]``, zero-filling unused entries.

    Args:
        camera_type: Legacy serialized camera type string.
        distortion_parameters: Legacy distortion parameter array.

    Returns:
        np.ndarray: Packed FVDB distortion coefficients or an empty array for undistorted cameras.
    """
    params = np.asarray(distortion_parameters, dtype=np.float32).reshape(-1)
    if camera_type in ("PINHOLE", "SIMPLE_PINHOLE"):
        return np.empty((0,), dtype=np.float32)
    coeffs = np.zeros(_DISTORTION_COEFFS_SHAPE, dtype=np.float32)
    if camera_type == "SIMPLE_RADIAL":
        coeffs[0] = params[0]
        return coeffs
    if camera_type == "RADIAL":
        coeffs[0] = params[0]
        coeffs[1] = params[1]
        return coeffs
    if camera_type == "OPENCV":
        coeffs[0] = params[0]
        coeffs[1] = params[1]
        if params.size >= 5:
            coeffs[2] = params[4]
        coeffs[6] = params[2]
        coeffs[7] = params[3]
        return coeffs
    raise ValueError(f"Unsupported legacy camera_type {camera_type}")


class SfmCameraMetadata:
    """
    This class encodes metadata about a camera used to capture images in an :class:`SfmScene`.

    It contains information about the camera's intrinsic parameters (focal length, principal point, etc.),
    the canonical :class:`fvdb.CameraModel`, and packed distortion coefficients if applicable.

    The camera metadata is used to project 3D points into 2D pixel coordinates for a single scene pixel space.
    """

    def __init__(
        self,
        img_width: int,
        img_height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_model: CameraModel,
        distortion_coeffs: np.ndarray,
    ):
        """
        Create a new :class:`SfmCameraMetadata` object.

        Args:
            img_width (int): The width of the camera image in pixel units (must be a positive integer).
            img_height (int): The height of the camera image in pixel units (must be a positive integer).
            fx (float): The focal length in the x direction in pixel units.
            fy (float): The focal length in the y direction in pixel units.
            cx (float): The x-coordinate of the principal point (optical center) in pixel units.
            cy (float): The y-coordinate of the principal point (optical center) in pixel units.
            camera_model (CameraModel): The canonical camera model used throughout the library.
            distortion_coeffs (np.ndarray): Distortion coefficients in FVDB packed layout
                ``[k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]`` or an empty
                array if no distortion is present.
        """

        if img_width <= 0 or img_height <= 0:
            raise ValueError("Image dimensions must be positive integers.")

        self._width = img_width
        self._height = img_height
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._camera_model = camera_model
        self._distortion_coeffs = _as_packed_distortion_coeffs(distortion_coeffs)
        self._projection_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def state_dict(self) -> dict[str, Any]:
        """
        Return a state dictionary representing the camera metadata.

        This dictionary can be used to serialize and deserialize the camera metadata.
        The dictionary stores the camera intrinsics for the pixel space represented by this
        metadata object.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing the camera metadata.
        """
        return {
            "img_width": self.width,
            "img_height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "camera_model": self.camera_model.name,
            "distortion_coeffs": self.distortion_coeffs.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "SfmCameraMetadata":
        """
        Create a new :class:`SfmCameraMetadata` object from a state dictionary originally created by :meth:`state_dict`.

        Args:
            state_dict (dict[str, Any]): A dictionary containing the camera metadata.

        Returns:
            SfmCameraMetadata: A new :class:`SfmCameraMetadata` object.
        """
        for key in ("img_width", "img_height", "fx", "fy", "cx", "cy"):
            if key not in state_dict:
                raise KeyError(f"{key} is missing from state_dict")
        img_width = int(state_dict["img_width"])
        img_height = int(state_dict["img_height"])
        fx = float(state_dict["fx"])
        fy = float(state_dict["fy"])
        cx = float(state_dict["cx"])
        cy = float(state_dict["cy"])
        if "camera_model" in state_dict:
            serialized_camera_model = state_dict["camera_model"]
            if isinstance(serialized_camera_model, str):
                camera_model = CameraModel[serialized_camera_model]
            else:
                camera_model = CameraModel(serialized_camera_model)
            distortion_coeffs = np.array(state_dict.get("distortion_coeffs", []), dtype=np.float32)
        else:
            if "camera_type" not in state_dict:
                raise KeyError("camera_model is missing from state_dict")
            if "distortion_parameters" not in state_dict:
                raise KeyError("distortion_parameters is missing from state_dict")
            camera_type = str(state_dict["camera_type"])
            camera_model = _legacy_camera_type_to_camera_model(camera_type)
            distortion_coeffs = _legacy_distortion_parameters_to_coeffs(
                camera_type,
                np.array(state_dict["distortion_parameters"], dtype=np.float32),
            )

        return cls(
            img_width=img_width,
            img_height=img_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_model=camera_model,
            distortion_coeffs=distortion_coeffs,
        )

    @property
    def projection_matrix(self) -> np.ndarray:
        """
        Return the camera projection matrix.

        The projection matrix is a 3x3 matrix that maps 3D points in camera coordinates to 2D
        points in pixel coordinates for the single pixel space represented by this metadata object.

        Returns:
            projection_matrix (np.ndarray): The camera projection matrix as a 3x3 numpy array.
        """
        return self._projection_matrix

    @property
    def fx(self) -> float:
        """
        Return the focal length in the x direction in pixel units.

        Returns:
            fx (float): The focal length in the x direction in pixel units.
        """
        return self._fx

    @property
    def fy(self) -> float:
        """
        Return the focal length in the y direction in pixel units.

        Returns:
            fy (float): The focal length in the y direction in pixel units.
        """
        return self._fy

    @property
    def cx(self) -> float:
        """
        Return the x-coordinate of the principal point (optical center) in pixel units.

        Returns:
            cx (float): The x-coordinate of the principal point in pixel units.
        """
        return self._cx

    @property
    def cy(self) -> float:
        """
        Return the y-coordinate of the principal point (optical center) in pixel units.

        Returns:
            cy (float): The y-coordinate of the principal point in pixel units.
        """
        return self._cy

    @property
    def fovx(self) -> float:
        """
        Return the horizontal field of view in radians.

        Returns:
            fovx (float): The horizontal field of view in radians.
        """
        return self._focal2fov(self.fx, self.width)

    @property
    def fovy(self) -> float:
        """
        Return the vertical field of view in radians.

        Returns:
            fovy (float): The vertical field of view in radians.
        """
        return self._focal2fov(self.fy, self.height)

    @property
    def width(self) -> int:
        """
        Return the width of the camera image in pixel units.

        Returns:
            width (int): The width of the camera image in pixels.
        """
        return self._width

    @property
    def height(self) -> int:
        """
        Return the height of the camera image in pixel units.

        Returns:
            height (int): The height of the camera image in pixels.
        """
        return self._height

    @property
    def camera_model(self) -> CameraModel:
        """
        Return the canonical camera model used to capture the image.

        Returns:
            camera_model (CameraModel): The canonical FVDB camera model.
        """
        return self._camera_model

    @property
    def aspect(self) -> float:
        """
        Return the aspect ratio of the camera image.

        The aspect ratio is defined as the width divided by the height.

        Returns:
            aspect (float): The aspect ratio of the camera image.
        """
        return self.width / self.height

    @property
    def distortion_coeffs(self) -> np.ndarray:
        """
        Return the packed distortion coefficients of the camera.

        The coefficients follow the FVDB packed layout
        ``[k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]``.

        Returns:
            np.ndarray: An array of distortion coefficients.
        """
        return self._distortion_coeffs

    @property
    def can_undistort(self) -> bool:
        """
        Return whether :class:`UndistortImages` can handle this camera.

        Returns:
            bool: True if the camera is already undistorted, in which case the transform is a
                no-op, or if it uses the local OpenCV radtan undistortion path.
        """
        return self._distortion_coeffs.size == 0 or self._camera_model == CameraModel.OPENCV_RADTAN_5

    def resize(self, new_width: int, new_height: int) -> "SfmCameraMetadata":
        """
        Return a new :class:`SfmCameraMetadata` object with the camera parameters resized to the new image dimensions.

        The resize is applied to the pixel space represented by this metadata object.

        Args:
            new_width (int): The new width of the camera image (must be a positive integer)
            new_height (int): The new height of the camera image (must be a positive integer)

        Returns:
            SfmCameraMetadata: A new :class:`SfmCameraMetadata` object with the resized camera parameters.
        """
        if new_width <= 0 or new_height <= 0:
            raise ValueError("New size must be positive integers.")

        rescale_w = self.width / new_width
        rescale_h = self.height / new_height
        new_fx = self.fx / rescale_w
        new_fy = self.fy / rescale_h
        new_cx = self.cx / rescale_w
        new_cy = self.cy / rescale_h

        return SfmCameraMetadata(
            new_width,
            new_height,
            new_fx,
            new_fy,
            new_cx,
            new_cy,
            self.camera_model,
            self.distortion_coeffs,
        )

    @staticmethod
    def _focal2fov(focal: float, pixels: float) -> float:
        """
        Convert a focal length in pixel units to a field of view in radians.

        Args:
            focal (float): The focal length in pixel units.
            pixels (float): The number of pixels corresponding to the field of view.

        Returns:
            float: The field of view in radians.
        """
        return 2 * np.arctan(pixels / (2 * focal))


class SfmPosedImageMetadata:
    """
    This class encodes metadata about a single posed image in an :class:`SfmScene`.

    It contains information about the camera pose (world-to-camera and camera-to-world matrices),
    a reference to the metadata for the camera that captured the image (see :class:`SfmCameraMetadata`),
    and the image and (optionally) mask file paths.
    """

    def __init__(
        self,
        world_to_camera_matrix: np.ndarray,
        camera_to_world_matrix: np.ndarray,
        camera_metadata: SfmCameraMetadata,
        camera_id: int,
        image_path: str,
        mask_path: str,
        point_indices: np.ndarray | None,
        image_id: int,
    ):
        """
        Create a new :class:`SfmImageMetadata` object.

        Args:
            world_to_camera_matrix (np.ndarray): A 4x4 matrix representing the transformation from world coordinates to camera coordinates.
            camera_to_world_matrix (np.ndarray): A 4x4 matrix representing the transformation from camera coordinates to world coordinates.
            camera_metadata (SfmCameraMetadata): The metadata for the camera that captured this image.
            camera_id (int): The unique identifier for the camera that captured this image.
            image_path (str): The file path to the image on the filesystem.
            mask_path (str): The file path to the mask image on the filesystem (can be an empty string if no mask is available).
            point_indices (np.ndarray | None): An optional array of point indices that are visible in this image (can be None if not available).
            image_id (int): The unique identifier for the image.
        """
        self._world_to_camera_matrix = world_to_camera_matrix
        self._camera_to_world_matrix = camera_to_world_matrix
        self._camera_id = camera_id
        self._image_path = image_path
        self._mask_path = mask_path
        self._point_indices = point_indices
        self._camera_metadata = camera_metadata
        self._image_id = image_id

    def state_dict(self) -> dict[str, Any]:
        """
        Return a state dictionary representing the image metadata.

        This dictionary can be used to serialize and deserialize the image metadata.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing the image metadata.
        """
        return {
            "world_to_camera_matrix": self.world_to_camera_matrix.tolist(),
            "camera_to_world_matrix": self.camera_to_world_matrix.tolist(),
            "camera_id": self.camera_id,
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "point_indices": self.point_indices.tolist() if self.point_indices is not None else None,
            "image_id": self.image_id,
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        camera_metadata: dict[int, SfmCameraMetadata],
    ) -> "SfmPosedImageMetadata":
        """
        Create a new :class:`SfmImageMetadata` object from a state dictionary and camera metadata (see :meth:`state_dict`).

        Args:
            state_dict (dict[str, Any]): A dictionary containing the image metadata.
            camera_metadata (dict[int, SfmCameraMetadata]): A dictionary mapping camera IDs to :class:`SfmCameraMetadata` objects.
        Returns:
            SfmImageMetadata: A new :class:`SfmImageMetadata` object.
        """
        if "world_to_camera_matrix" not in state_dict:
            raise KeyError("world_to_camera_matrix is missing from state_dict")
        if "camera_to_world_matrix" not in state_dict:
            raise KeyError("camera_to_world_matrix is missing from state_dict")
        if "camera_id" not in state_dict:
            raise KeyError("camera_id is missing from state_dict")
        if "image_path" not in state_dict:
            raise KeyError("image_path is missing from state_dict")
        if "mask_path" not in state_dict:
            raise KeyError("mask_path is missing from state_dict")
        if "image_id" not in state_dict:
            raise KeyError("image_id is missing from state_dict")

        world_to_camera_matrix = np.array(state_dict["world_to_camera_matrix"])
        camera_to_world_matrix = np.array(state_dict["camera_to_world_matrix"])
        camera_id = int(state_dict["camera_id"])
        image_path = str(state_dict["image_path"])
        mask_path = str(state_dict["mask_path"])
        point_indices = (
            np.array(state_dict["point_indices"])
            if "point_indices" in state_dict and state_dict["point_indices"] is not None
            else None
        )
        image_id = int(state_dict["image_id"])

        if camera_id not in camera_metadata:
            raise KeyError(f"Camera ID {camera_id} not found in camera_metadata")

        return cls(
            world_to_camera_matrix=world_to_camera_matrix,
            camera_to_world_matrix=camera_to_world_matrix,
            camera_metadata=camera_metadata[camera_id],
            camera_id=camera_id,
            image_path=image_path,
            mask_path=mask_path,
            point_indices=point_indices,
            image_id=image_id,
        )

    def transform(self, transformation_matrix: np.ndarray) -> "SfmPosedImageMetadata":
        """
        Return a new :class:`SfmImageMetadata` object with the camera pose transformed by the given transformation matrix.

        This transformation applies to the left of the camera to world transformation matrix,
        meaning it transforms the camera in world space.

        *i.e.* ``new_camera_to_world_matrix = transformation_matrix @ self.camera_to_world_matrix``
        Args:
            transformation_matrix (np.ndarray): A 4x4 transformation matrix to apply.

        Returns:
            SfmImageMetadata: A new :class:`SfmImageMetadata` object with the transformed matrices.
        """
        new_camera_to_world_matrix = transformation_matrix @ self.camera_to_world_matrix
        new_world_to_camera_matrix = np.linalg.inv(new_camera_to_world_matrix)

        return SfmPosedImageMetadata(
            world_to_camera_matrix=new_world_to_camera_matrix,
            camera_to_world_matrix=new_camera_to_world_matrix,
            camera_metadata=self.camera_metadata,
            camera_id=self.camera_id,
            image_path=self.image_path,
            mask_path=self.mask_path,
            point_indices=self.point_indices,
            image_id=self.image_id,
        )

    @property
    def world_to_camera_matrix(self) -> np.ndarray:
        """
        Return the world-to-camera transformation matrix for this posed image.

        This matrix transforms points from world coordinates to camera coordinates.

        Returns:
            world_to_camera_matrix (np.ndarray): The world-to-camera transformation matrix as a 4x4 numpy array.
        """
        return self._world_to_camera_matrix

    @property
    def camera_to_world_matrix(self) -> np.ndarray:
        """
        Return the camera-to-world transformation matrix for this posed image.

        This matrix transforms points from camera coordinates to world coordinates.

        Returns:
            camera_to_world_matrix (np.ndarray): The camera-to-world transformation matrix as a 4x4 numpy array.
        """
        return self._camera_to_world_matrix

    @property
    def camera_id(self) -> int:
        """
        Return the unique identifier for the camera that captured this image.

        Returns:
            camera_id (int): The camera ID.
        """
        return self._camera_id

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Return the resolution of the posed image in pixels as a tuple of the form ``(height, width)``

        Returns:
            image_size (tuple[int, int]): The image resolution as ``(height, width)``.
        """
        return self._camera_metadata.height, self._camera_metadata.width

    @property
    def image_path(self) -> str:
        """
        Return the file path to color image for this posed image.

        Returns:
            image_path (str): The path to the color image file for this posed image.
        """
        return self._image_path

    @property
    def mask_path(self) -> str:
        """
        Return the file path to the mask for this posed image.

        The mask image is used to indicate which pixels in the image are valid (e.g., not occluded).

        An empty string indicates that no mask is available.

        Returns:
            mask_path (str): The path to the posed mask image file.
        """
        return self._mask_path

    @property
    def point_indices(self) -> np.ndarray | None:
        """
        Return the indices of the 3D points that are visible in this posed image or ``None`` if the indices are not available.

        These indices correspond to the points in the :class:`SfmScene`'s point cloud that are visible in this posed image.

        Returns:
            point_indices (np.ndarray | None): An array of indices of the visible 3D points or ``None`` if not available.
        """
        return self._point_indices

    @property
    def camera_metadata(self) -> SfmCameraMetadata:
        """
        Return metadata about the camera that captured this posed image (see :class:`SfmCameraMetadata`).

        The camera metadata contains information about the camera's intrinsic parameters, such as focal length and distortion coefficients.

        Returns:
            SfmCameraMetadata: The camera metadata object.
        """
        return self._camera_metadata

    @property
    def image_id(self) -> int:
        """
        Return the unique identifier for this image.

        This ID is used to uniquely identify the image within the dataset.

        Returns:
            int: The image ID.
        """
        return self._image_id

    @property
    def lookat(self):
        """
        Return the camera lookat vector.

        The lookat vector is the direction the camera is pointing, which is the negative z-axis in the camera coordinate system.

        Returns:
            lookat (np.ndarray): The camera lookat vector as a 3D numpy array.
        """
        return self.camera_to_world_matrix[:3, 2]

    @property
    def origin(self):
        """
        Return the origin of the posed image. *i.e.* the position of the camera in world coordinates when it captured the image.

        The origin is the position of the camera in world coordinates, which is the translation part of the camera-to-world matrix.

        Returns:
            origin (np.ndarray): The camera origin as a 3D numpy array.
        """
        return self.camera_to_world_matrix[:3, 3]

    @property
    def up(self):
        """
        Return the camera up vector.

        The up vector is the direction that is considered "up" in the camera coordinate system, which is the negative y-axis in the camera coordinate system.

        Returns:
            up (np.ndarray): The camera up vector as a 3D numpy array.
        """
        return -self.camera_to_world_matrix[:3, 1]

    @property
    def right(self):
        """
        Return the camera right vector.

        The right vector is the direction that is considered "right" in the camera coordinate system, which is the x-axis in the camera coordinate system.

        Returns:
            right (np.ndarray): The camera right vector as a 3D numpy array.
        """
        return self.camera_to_world_matrix[:3, 0]
