# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from enum import Enum

import cv2
import numpy as np


class SfmCameraType(Enum):
    """
    Enum representing different camera types used in structure-from-motion (SFM) pipelines.
    """

    PINHOLE = "PINHOLE"
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"
    SIMPLE_RADIAL = "SIMPLE_RADIAL"
    RADIAL = "RADIAL"
    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"


class SfmCameraMetadata:
    """
    This class encodes metadata about a camera used to capture images in a structure-from-motion (SFM) pipeline.

    It contains information about the camera's intrinsic parameters (focal length, principal point, etc.),
    the camera type (e.g., pinhole, radial distortion), and distortion parameters if applicable.

    The camera metadata is used to project 3D points into 2D pixel coordinates and to undistort images captured by the camera.
    """

    def __init__(
        self,
        img_width: int,
        img_height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_type: SfmCameraType,
        distortion_parameters: np.ndarray,
    ):
        """
        Create a new `SfmCameraMetadata` object.

        Args:
            img_width (int): The width of the camera image in pixel units (must be a positive integer).
            img_height (int): The height of the camera image in pixel units (must be a positive integer).
            fx (float): The focal length in the x direction in pixel units.
            fy (float): The focal length in the y direction in pixel units.
            cx (float): The x-coordinate of the principal point (optical center) in pixel units.
            cy (float): The y-coordinate of the principal point (optical center) in pixel units.
            camera_type (SfmCameraType): The type of camera used to capture the image (e.g., "PINHOLE", "SIMPLE_PINHOLE", etc.).
            distortion_parameters (np.ndarray): An array of distortion coefficients corresponding to the camera type, or an empty array if no distortion is present.
        """

        # camera intrinsics assuming a perspective projection model
        projection_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        if len(distortion_parameters) != 0:
            undistorted_proj_mat, undistort_roi = cv2.getOptimalNewCameraMatrix(
                projection_matrix, distortion_parameters, (img_width, img_height), 0
            )
            undistort_map_x, undistort_map_y = cv2.initUndistortRectifyMap(
                projection_matrix, distortion_parameters, None, undistorted_proj_mat, (img_width, img_height), cv2.CV_32FC1  # type: ignore
            )

            self._projection_matrix = undistorted_proj_mat

            self._undistort_roi = tuple([v for v in undistort_roi])
            assert len(self._undistort_roi) == 4, "Undistort ROI must be a tuple of (x, y, width, height)"

            self._undistort_map_x = undistort_map_x
            self._undistort_map_y = undistort_map_y
        else:
            self._projection_matrix = projection_matrix
            self._undistort_roi = None
            self._undistort_map_x = None
            self._undistort_map_y = None

        self._fx = self._projection_matrix[0, 0]
        self._fy = self._projection_matrix[1, 1]
        self._cx = self._projection_matrix[0, 2]
        self._cy = self._projection_matrix[1, 2]
        self._width = img_width
        self._height = img_height
        self._camera_type = camera_type
        self._distortion_parameters = distortion_parameters

    @property
    def projection_matrix(self) -> np.ndarray:
        """
        Return the camera projection matrix.

        The projection matrix is a 3x3 matrix that maps 3D points in camera coordinates to 2D points in pixel coordinates.

        Returns:
            np.ndarray: The camera projection matrix as a 3x3 numpy array.
        """
        return self._projection_matrix

    @property
    def fx(self) -> float:
        """
        Return the focal length in the x direction in pixel units.

        Returns:
            float: The focal length in the x direction in pixel units.
        """
        return self._fx

    @property
    def fy(self) -> float:
        """
        Return the focal length in the y direction in pixel units.

        Returns:
            float: The focal length in the y direction in pixel units.
        """
        return self._fy

    @property
    def cx(self) -> float:
        """
        Return the x-coordinate of the principal point (optical center) in pixel units.

        Returns:
            float: The x-coordinate of the principal point in pixel units.
        """
        return self._cx

    @property
    def cy(self) -> float:
        """
        Return the y-coordinate of the principal point (optical center) in pixel units.

        Returns:
            float: The y-coordinate of the principal point in pixel units.
        """
        return self._cy

    @property
    def fovx(self) -> float:
        """
        Return the horizontal field of view in radians.

        Returns:
            float: The horizontal field of view in radians.
        """
        return self._focal2fov(self.fx, self.width)

    @property
    def fovy(self) -> float:
        """
        Return the vertical field of view in radians.

        Returns:
            float: The vertical field of view in radians.
        """
        return self._focal2fov(self.fy, self.height)

    @property
    def width(self) -> int:
        """
        Return the width of the camera image in pixel units.

        Returns:
            int: The width of the camera image in pixels.
        """
        return self._width

    @property
    def height(self) -> int:
        """
        Return the height of the camera image in pixel units.

        Returns:
            int: The height of the camera image in pixels.
        """
        return self._height

    @property
    def camera_type(self) -> SfmCameraType:
        """
        Return the type of camera used to capture the image.

        Returns:
            SfmCameraType: The camera type (e.g., "PINHOLE", "SIMPLE_PINHOLE", etc.).
        """
        return self._camera_type

    @property
    def aspect(self) -> float:
        """
        Return the aspect ratio of the camera image.

        The aspect ratio is defined as the width divided by the height.

        Returns:
            float: The aspect ratio of the camera image.
        """
        return self.width / self.height

    @property
    def distortion_parameters(self) -> np.ndarray:
        """
        Return the distortion parameters of the camera.

        The distortion parameters are used to correct lens distortion in the captured images.

        Returns:
            np.ndarray: An array of distortion coefficients.
        """
        return self._distortion_parameters

    def resize(self, new_width, new_height) -> "SfmCameraMetadata":
        """
        Resize the camera metadata to a new resolution

        Args:
            new_width (int): The new width of the camera image (must be a positive integer)
            new_height (int): The new height of the camera image (must be a positive integer)

        Returns:
            A new `SfmCameraMetadata` object with the resized camera parameters.
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
            new_width, new_height, new_fx, new_fy, new_cx, new_cy, self.camera_type, self.distortion_parameters
        )

    @property
    def undistort_roi(self) -> tuple[int, int, int, int] | None:
        """
        Return the region of interest (ROI) for undistorted images.
        The ROI is defined as a tuple of (x, y, width, height) that specifies the valid region of the undistorted image.
        If the camera does not have distortion parameters, this will be None.

        Returns:
            tuple[int, int, int, int] | None: The ROI for undistorted images or None if no distortion parameters are present.
        """
        if self._undistort_roi is not None:
            assert len(self._undistort_roi) == 4, "Undistort ROI must be a tuple of (x, y, width, height)"
        return self._undistort_roi

    @property
    def undistort_map_x(self) -> np.ndarray | None:
        """
        Return the undistortion map for the x-coordinates of the image.
        The undistortion map is used to remap the pixel coordinates of the image to correct for lens distortion.
        If the camera does not have distortion parameters, this will be None.

        Returns:
            np.ndarray | None: The undistortion map for the x-coordinates or None if no distortion parameters are present.
        """
        return self._undistort_map_x

    @property
    def undistort_map_y(self) -> np.ndarray | None:
        """
        Return the undistortion map for the y-coordinates of the image.
        The undistortion map is used to remap the pixel coordinates of the image to correct for lens distortion.
        If the camera does not have distortion parameters, this will be None.
        Returns:
            np.ndarray | None: The undistortion map for the y-coordinates or None if no distortion parameters are present.
        """
        return self._undistort_map_y

    @staticmethod
    def _focal2fov(focal: float, pixels: float) -> float:
        return 2 * np.arctan(pixels / (2 * focal))

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        if self.undistort_map_x is not None and self.undistort_map_y is not None:
            image_remap = cv2.remap(image, self.undistort_map_x, self.undistort_map_y, interpolation=cv2.INTER_LINEAR)
            assert self.undistort_roi is not None
            x, y, w, h = self.undistort_roi
            return image_remap[y : y + h, x : x + w]
        else:
            return image


class SfmImageMetadata:
    """
    This class encodes metadata about a single posed image captured by a camera in a structure-from-motion (SFM) pipeline.

    It contains information about the camera pose (world-to-camera and camera-to-world matrices),
    the camera metadata (intrinsics, distortion parameters, etc.), for the camera that captured this image,
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
        point_indices: np.ndarray,
        image_id: int,
    ):
        self._world_to_camera_matrix = world_to_camera_matrix
        self._camera_to_world_matrix = camera_to_world_matrix
        self._camera_id = camera_id
        self._image_path = image_path
        self._mask_path = mask_path
        self._point_indices = point_indices
        self._camera_metadata = camera_metadata
        self._image_id = image_id

    def transform(self, transformation_matrix: np.ndarray) -> "SfmImageMetadata":
        """
        Apply a transformation to the world-to-camera matrix and camera-to-world matrix of this image.

        This transformation applies to the left of the camera to world transformation matrix,
        meaning it transforms the camera in world space.

        _i.e._
            new_camera_to_world_matrix = transformation_matrix @ self.camera_to_world_matrix
        Args:
            transformation_matrix (np.ndarray): A 4x4 transformation matrix to apply.

        Returns:
            SfmImageMetadata: A new `SfmImageMetadata` object with the transformed matrices.
        """
        new_camera_to_world_matrix = transformation_matrix @ self.camera_to_world_matrix
        new_world_to_camera_matrix = np.linalg.inv(new_camera_to_world_matrix)

        return SfmImageMetadata(
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
        Return the world-to-camera transformation matrix.

        This matrix transforms points from world coordinates to camera coordinates.

        Returns:
            np.ndarray: The world-to-camera transformation matrix as a 4x4 numpy array.
        """
        return self._world_to_camera_matrix

    @property
    def camera_to_world_matrix(self) -> np.ndarray:
        """
        Return the camera-to-world transformation matrix.

        This matrix transforms points from camera coordinates to world coordinates.

        Returns:
            np.ndarray: The camera-to-world transformation matrix as a 4x4 numpy array.
        """
        return self._camera_to_world_matrix

    @property
    def camera_id(self) -> int:
        """
        Return the unique identifier for the camera that captured this image.

        Returns:
            int: The camera ID.
        """
        return self._camera_id

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Return the resolution of the image in pixels as a tuple of the form (height, width)

        Returns:
            tuple[int, int]: The image resolution as (height, width).
        """
        return self._camera_metadata.height, self._camera_metadata.width

    @property
    def image_path(self) -> str:
        """
        Return the file path to the image.

        Returns:
            str: The path to the image file.
        """
        return self._image_path

    @property
    def mask_path(self) -> str:
        """
        Return the file path to the mask image.

        The mask image is used to indicate which pixels in the image are valid (e.g., not occluded).

        Returns:
            str: The path to the mask image file.
        """
        return self._mask_path

    @property
    def point_indices(self) -> np.ndarray:
        """
        Return the indices of the 3D points that are visible in this image.

        These indices correspond to the points in the point cloud that are visible in this image.

        Returns:
            np.ndarray: An array of indices of the visible 3D points.
        """
        return self._point_indices

    @property
    def camera_metadata(self) -> SfmCameraMetadata:
        """
        Return the camera metadata associated with this image.

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
            np.ndarray: The camera lookat vector as a 3D numpy array.
        """
        return self.camera_to_world_matrix[:3, 2]

    @property
    def origin(self):
        """
        Return the origin of the camera.

        The origin is the position of the camera in world coordinates, which is the translation part of the camera-to-world matrix.

        Returns:
            np.ndarray: The camera origin as a 3D numpy array.
        """
        return self.camera_to_world_matrix[:3, 3]

    @property
    def up(self):
        """
        Return the camera up vector.

        The up vector is the direction that is considered "up" in the camera coordinate system, which is the negative y-axis in the camera coordinate system.

        Returns:
            np.ndarray: The camera up vector as a 3D numpy array.
        """
        return -self.camera_to_world_matrix[:3, 1]

    @property
    def right(self):
        """
        Return the camera right vector.

        The right vector is the direction that is considered "right" in the camera coordinate system, which is the x-axis in the camera coordinate system.

        Returns:
            np.ndarray: The camera right vector as a 3D numpy array.
        """
        return self.camera_to_world_matrix[:3, 0]
