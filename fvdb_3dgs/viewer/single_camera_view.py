# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from collections.abc import Sequence

import numpy as np
import torch
import viser

from .viewer_handle import ViewerHandle


class SingleCameraView:
    """
    A view representing a single visualized camera in the viewer.
    It renders the camera's frustum, axes, and optionally an image.
    """

    def __init__(
        self,
        name: str,
        viewer_handle: ViewerHandle,
        cam_to_world_matrix: np.ndarray | torch.Tensor,
        projection_matrix: np.ndarray | torch.Tensor,
        image_dimensions: Sequence[int | float],
        image: np.ndarray | torch.Tensor | None,
        axis_length: float,
        axis_thickness: float,
        frustum_line_width: float,
        frustum_scale: float,
        frustum_color: np.ndarray,
        show_image: bool,
        visible: bool,
    ):
        """
        Create a new SingleCameraView instance.

        A SingleCameraView represents a single camera in the viewer, displaying its frustum, axes, and optionally an image.

        Args:
            name (str): The name of the camera view.
            viewer_handle (ViewerHandle): The viewer handle to which this camera view will be registered.
            cam_to_world_matrix (np.ndarray | torch.Tensor): The 4x4 camera to world transformation matrix.
            projection_matrix (np.ndarray | torch.Tensor): The 3x3 projection matrix.
            image_dimensions (Sequence[int | float]): The dimensions of the image as a sequence of two integers (height, width).
            image (np.ndarray | torch.Tensor | None): The image to display in the camera view. If None, no image is shown.
            axis_length (float): The length of the axis lines in the camera frustum view.
            axis_thickness (float): The world-unit thickness (2x the radius) of the axis lines in the camera frustum view.
            frustum_line_width (float): The width of the frustum lines in pixels.
            frustum_scale (float): The scale factor for the frustum in the camera frustum view.
            frustum_color (np.ndarray): The RGB color of the frustum lines as a 3-element array.
            show_image (bool): Whether to show the image in the camera view.
            visible (bool): If True, the camera frustum and axes will be visible in the viewer.
        """
        self._name = name
        self._viewer_handle = viewer_handle

        self._axis_length = axis_length
        self._axis_radius = axis_thickness / 2
        self._show_image = show_image
        self._initial_axis_length = axis_length
        self._frustum_line_width = frustum_line_width
        self._frustum_scale = frustum_scale
        self._frustum_color = frustum_color

        if not isinstance(frustum_color, (torch.Tensor, np.ndarray)):
            raise TypeError("frustum_color must be a torch.Tensor or np.ndarray")

        if frustum_color.shape != (3,):
            raise ValueError("frustum_color must be a 3-element array representing RGB color")

        if np.any(frustum_color < 0) or np.any(frustum_color > 1):
            raise ValueError("frustum_color values must be in the range [0, 1]")

        if not isinstance(cam_to_world_matrix, (torch.Tensor, np.ndarray)):
            raise TypeError("cam_to_world_matrix must be a torch.Tensor or np.ndarray")

        if not isinstance(projection_matrix, (torch.Tensor, np.ndarray)):
            raise TypeError("projection_matrix must be a torch.Tensor or np.ndarray")

        if isinstance(cam_to_world_matrix, torch.Tensor):
            cam_to_world_matrix = cam_to_world_matrix.cpu().numpy()

        if isinstance(projection_matrix, torch.Tensor):
            projection_matrix = projection_matrix.cpu().numpy()

        if cam_to_world_matrix.ndim != 2 or cam_to_world_matrix.shape != (4, 4):
            raise ValueError("cam_to_world_matrix must be a 2D tensor with shape (4, 4)")

        if projection_matrix.ndim != 2 or projection_matrix.shape != (3, 3):
            raise ValueError("projection_matrix must be a 2D tensor with shape (3, 3)")

        self._projection_matrix: np.ndarray = projection_matrix
        self._cam_to_world_matrix: np.ndarray = cam_to_world_matrix

        if len(image_dimensions) != 2:
            raise ValueError("image_dimensions must be a sequence of two integers (height, width)")

        self._image_size: tuple[int, int] = (int(image_dimensions[0]), int(image_dimensions[1]))

        if image is not None:
            if not isinstance(image, (torch.Tensor, np.ndarray)):
                raise TypeError("image must be a torch.Tensor or np.ndarray")
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()

            if image.ndim not in (2, 3):
                raise ValueError("images must be a 2D or 3D tensor with shape (H, W) or (H, W, D)")

            if image.ndim == 2:
                image = image[..., np.newaxis]

            if self._image_size != image.shape[0:2]:
                raise ValueError(
                    f"Image dimensions {self._image_size} do not match the shape of the image {image.shape[0:2]}"
                )

        self._image: np.ndarray | None = image

        scene: viser.SceneApi = self._viewer_handle.scene

        position, quaternion = self._transformation_matrix_to_position_and_quat(self._cam_to_world_matrix)
        fov, aspect_ratio = self._projection_matrix_to_fov_aspect(self._projection_matrix, *self._image_size)

        self._frame_scene_handle: viser.FrameHandle = scene.add_frame(
            name=f"{self._name}_camera_frame",
            position=position,
            wxyz=quaternion,
            axes_length=self._axis_length,
            axes_radius=self._axis_radius,
            origin_radius=2.0 * self._axis_radius,
            visible=visible,
        )

        self._frustum_scene_handle: viser.CameraFrustumHandle = scene.add_camera_frustum(
            name=f"{self._name}_camera_frame/frustum",
            fov=fov,
            aspect=aspect_ratio,
            image=self._image if self._show_image else None,
            format="jpeg",
            jpeg_quality=50,
            line_width=self._frustum_line_width,
            scale=self._frustum_scale,
            color=self._frustum_color,
            visible=visible,
        )

    @property
    def name(self) -> str:
        """
        Returns the name of the camera view.

        Returns:
            str: The name of the camera view.
        """
        return self._name

    @property
    def visible(self) -> bool:
        """
        Returns whether the camera view is visible.

        Returns:
            bool: True if the camera view is visible, False otherwise.
        """
        assert (
            self._frame_scene_handle.visible == self._frustum_scene_handle.visible
        ), "Frame and frustum visibility should be the same."
        return self._frame_scene_handle.visible

    @visible.setter
    def visible(self, value: bool):
        """
        Sets whether the camera view is visible.

        Args:
            value (bool): True to enable the camera view, False to disable it.
        """
        if not isinstance(value, bool):
            raise TypeError("enabled must be a boolean value.")

        self._frame_scene_handle.visible = value
        self._frustum_scene_handle.visible = value

    @property
    def cam_to_world_matrix(self) -> np.ndarray:
        """
        Returns the camera to world transformation matrix.

        Returns:
            np.ndarray: The 4x4 camera to world transformation matrix.
        """
        return self._cam_to_world_matrix

    @cam_to_world_matrix.setter
    def cam_to_world_matrix(self, value: np.ndarray | torch.Tensor):
        """
        Sets the camera to world 4x4 transformation matrix.

        Args:
            value (np.ndarray | torch.Tensor): The new camera to world 4x4 transformation matrix.
        """
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        if not isinstance(value, np.ndarray) or value.shape != (4, 4):
            raise ValueError("cam_to_world_matrix must be a 4x4 numpy array.")
        self._cam_to_world_matrix = value

        position, quaternion = self._transformation_matrix_to_position_and_quat(self._cam_to_world_matrix)

        self._frame_scene_handle.position = position
        self._frame_scene_handle.wxyz = quaternion

    @property
    def projection_matrix(self) -> np.ndarray:
        """
        Returns the projection matrix.

        Returns:
            np.ndarray: The 3x3 projection matrix.
        """
        return self._projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value: np.ndarray | torch.Tensor):
        """
        Sets the projection matrix.

        Args:
            value (np.ndarray | torch.Tensor): The new 3x3 projection matrix.
        """
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        if not isinstance(value, np.ndarray) or value.shape != (3, 3):
            raise ValueError("projection_matrix must be a 3x3 numpy array.")
        self._projection_matrix = value

        fov, aspect_ratio = self._projection_matrix_to_fov_aspect(self._projection_matrix, *self._image_size)
        self._frustum_scene_handle.fov = fov
        self._frustum_scene_handle.aspect = aspect_ratio

    @property
    def image(self) -> np.ndarray | None:
        """
        Returns the image associated with the camera view.

        Returns:
            np.ndarray | None: The image data, or None if no image is set.
        """
        return self._image

    @image.setter
    def image(self, value: np.ndarray | torch.Tensor):
        """
        Sets the image associated with the camera view.

        Args:
            value (np.ndarray | torch.Tensor): The new image data
        """

        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        if not isinstance(value, np.ndarray):
            raise TypeError("image must be a numpy array or torch tensor.")

        if value.ndim not in (2, 3):
            raise ValueError("Image must be a 2D or 3D array with shape (H, W) or (H, W, D).")

        if value.ndim == 2:
            value = value[..., np.newaxis]

        if self._image_size != value.shape[0:2]:
            raise ValueError(
                f"Image dimensions {self._image_size} do not match the shape of the image {value.shape[0:2]}"
            )

        self._image = value
        self._image_size = value.shape[0:2]
        self._frustum_scene_handle.image = self._image

    @property
    def show_image(self) -> bool:
        """
        Returns whether the camera view is set to show images.

        Returns:
            bool: True if images are shown, False otherwise.
        """
        return self._show_image

    @show_image.setter
    def show_image(self, value: bool):
        """
        Sets whether the camera view should show images.

        Args:
            value (bool): True to show images, False to hide them.
        """
        if value:
            self._frustum_scene_handle.image = self._image
        else:
            self._frustum_scene_handle.image = None
        self._show_image = value

    @property
    def axis_length(self) -> float:
        """
        Returns the length of the axis lines in the camera frustum view.

        Returns:
            float: The length of the axis lines.
        """
        return self._axis_length

    @axis_length.setter
    def axis_length(self, value: float):
        """
        Sets the length of the axis lines in the camera frustum view.

        Args:
            value (float): The new length for the axis lines.
        """
        if value <= 0:
            raise ValueError("axis_length must be positive.")

        self._axis_length = value
        self._frame_scene_handle.axes_length = value

    @property
    def axis_thickness(self) -> float:
        """
        Returns the world-unit thickness (2x the radius) of the axis lines in the camera frustum view.

        Returns:
            float: The thickness (2x the radius) of the axis lines (in world units).
        """
        return self._axis_radius * 2.0

    @axis_thickness.setter
    def axis_thickness(self, value: float):
        """
        Sets the world-unit thickness (2x the radius) of the axis lines in the camera frustum view.

        Args:
            value (float): The new world-unit thickness for the axis lines.
        """
        if value <= 0:
            raise ValueError("axis_radius must be positive.")

        self._axis_radius = value / 2.0
        self._frame_scene_handle.axes_radius = self._axis_radius
        self._frame_scene_handle.origin_radius = 2.0 * self._axis_radius

    @property
    def frustum_line_width(self) -> float:
        """
        Returns the width of the frustum lines in the camera frustum view.

        Returns:
            float: The width of the frustum lines in pixels.
        """
        return self._frustum_line_width

    @frustum_line_width.setter
    def frustum_line_width(self, value: float):
        """
        Sets the width of the frustum lines in the camera frustum view.

        Args:
            value (float): The new width for the frustum lines in pixels.
        """
        if value <= 0:
            raise ValueError("frustum_line_width must be positive.")

        self._frustum_line_width = value
        self._frustum_scene_handle.line_width = value

    @property
    def frustum_scale(self) -> float:
        """
        Returns the scale factor for the frustum in the camera frustum view.

        Returns:
            float: The scale factor for the frustum.
        """
        return self._frustum_scale

    @frustum_scale.setter
    def frustum_scale(self, value: float):
        """
        Sets the scale factor for the frustum in the camera frustum view.

        Args:
            value (float): The new scale factor for the frustum.
        """
        if value <= 0:
            raise ValueError("frustum_scale must be positive.")

        self._frustum_scale = value
        self._frustum_scene_handle.scale = value

    def _projection_matrix_to_fov_aspect(
        self, projection_matrix: np.ndarray, img_h: int, img_w: int
    ) -> tuple[float, float]:
        """
        Convert a projection matrix and image dimensions to field of view and aspect ratio.

        Args:
            projection_matrix (np.ndarray): A 3x3 projection matrix.
            img_h (int): Height of the image.
            img_w (int): Width of the image.

        Returns:
            fov (float): The field of view in radians.
            aspect_ratio (float): The aspect ratio of the projection
        """
        if projection_matrix.shape != (3, 3):
            raise ValueError("Projection matrix must be 3x3.")

        fy = projection_matrix[1, 1]
        fov = 2 * np.arctan2(img_h / 2.0, fy)
        aspect_ratio = img_w / img_h

        return fov, aspect_ratio

    def _transformation_matrix_to_position_and_quat(self, T: np.ndarray):
        """
        Convert a 4x4 transformation matrix to position and quaternion.

        Args:
            T (np.ndarray): A 4x4 transformation matrix.

        Returns:
            tuple: A tuple containing the position (3-element array) and quaternion (4-element array).
        """
        if T.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4.")

        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)

        return position, quaternion

    def _rotation_matrix_to_quaternion(self, R):
        """

        Convert a 3x3 rotation matrix to a unit quaternion.

        Args:
            R (np.ndarray): A 3x3 rotation matrix.

        Returns:
            np.ndarray: A unit quaternion represented as a 4-element array [w, x, y, z].
        """
        # Ensure R is a proper rotation matrix
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = -R

        # Convert to quaternion
        q = np.zeros(4)
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            q[0] = 0.25 * S
            q[1] = (R[2, 1] - R[1, 2]) / S
            q[2] = (R[0, 2] - R[2, 0]) / S
            q[3] = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S
            q[3] = 0.25 * S
        return q
