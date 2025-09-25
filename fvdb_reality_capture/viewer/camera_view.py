# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from collections.abc import Sequence

import numpy as np
import torch
import viser

from .single_camera_view import SingleCameraView
from .viewer_handle import ViewerHandle


class CameraView:
    """
    A view representing a set of cameras in the fVDB viewer's GUI.

    This view displays camera frustums, view axes, and optional images for each camera.

    It also provides GUI controls to adjust the appearance of the camera frustums,
    such as axis length, axis thickness, frustum line width, and whether to show images.
    """

    def __init__(
        self,
        name: str,
        viewer_handle: ViewerHandle,
        cam_to_world_matrices: Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
        projection_matrices: Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
        image_dimensions: np.ndarray | torch.Tensor,
        images: Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None,
        axis_length: float,
        axis_thickness: float,
        frustum_line_width: float,
        frustum_scale: float,
        frustum_color: Sequence[float] | np.ndarray,
        show_images: bool,
        enabled: bool = True,
    ):
        """
        Create a new `CameraView` for displaying a set of cameras in the fVDB viewer's GUI.

        A `CameraView` represents a set of visualized cameras, each with its own frustum, view axes, and optional image.

        Note: You should not create this view directly. Instead, use the viewer's
        `register_camera_view` method to create and manage instances of this view.

        Args:
            name (str): The name of the view, used as the header title in the GUI.
            viewer_handle (ViewerHandle): The handle to the viewer.
            cam_to_world_matrices (Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray): A sequence of N 4x4 tensors or numpy arrays
                representing the camera-to-world transformation matrices, where N is the number of cameras.
            projection_matrices (Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray): A sequence of N 3x3 tensors or numpy arrays
                representing the projection matrices for the cameras, where N is the number of cameras.
            image_dimensions (np.ndarray | torch.Tensor): An array or Tensor of shape (N, 2) containing the height and width of each camera image.
            images (Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray): A sequence of N images (as numpy arrays or tensors) corresponding to the cameras.
                If None, no images will be displayed in the camera frustum view.
            axis_length (float): The length of the axis lines in the camera frustum view.
            axis_thickness (float): The thickness (in world coordinates) of the axis lines in the camera frustum view.
            frustum_line_width (float): The width (in pixels) of the frustum lines in the camera frustum view.
            show_images (bool): Whether to show images in the camera frustum view.
            enabled (bool): If True, the camera view UI is enabled and the cameras will be rendered.
                If False, the camera view UI is disabled and the cameras will not be rendered.
        """
        self._name = name
        self._viewer_handle = viewer_handle

        self._axis_length = axis_length
        self._axis_thickness = axis_thickness
        self._show_images = show_images
        self._frustum_line_width = frustum_line_width
        self._frustum_color = np.asarray(frustum_color, dtype=np.float32)
        self._frustum_scale = frustum_scale
        self._enabled = enabled

        if not isinstance(image_dimensions, (np.ndarray, torch.Tensor)):
            raise ValueError("image_dimensions must be a numpy array or torch tensor.")
        if isinstance(image_dimensions, torch.Tensor):
            image_dimensions = image_dimensions.cpu().numpy()

        self._camera_views: dict[int, SingleCameraView] = {}

        if len(cam_to_world_matrices) != len(projection_matrices):
            raise ValueError("The number of camera-to-world matrices must match the number of projection matrices.")
        if images is not None:
            if len(images) != len(projection_matrices):
                raise ValueError("The number of images must match the number of projection matrices.")

        for i in range(len(cam_to_world_matrices)):
            cam_to_world_matrix: np.ndarray | torch.Tensor = cam_to_world_matrices[i]
            projection_matrix: np.ndarray | torch.Tensor = projection_matrices[i]
            image: np.ndarray | torch.Tensor | None = images[i] if images is not None else None
            if image is not None:
                if not isinstance(image, (np.ndarray, torch.Tensor)):
                    raise ValueError(f"Image {i} must be a numpy array or torch tensor. Got {type(image)} instead.")
                if not (image.ndim == 3 or image.ndim == 2):
                    raise ValueError(
                        f"Image {i} must be a 2D (grayscale) or 3D (color) array. Got {image.ndim}D instead."
                    )

                if image.shape[0] != image_dimensions[i, 0] or image.shape[1] != image_dimensions[i, 1]:
                    raise ValueError(
                        f"Image {i} dimensions {image.shape} do not match the specified image width/height {image_dimensions[i]}."
                    )
                if image.ndim == 3 and image.shape[2] not in (3, 4):
                    raise ValueError(f"Image {i} must have 3 (RGB) or 4 (RGBA) channels. Got {image.shape[2]} instead.")

            self._camera_views[i] = SingleCameraView(
                name=f"{self._name} Camera {i}",
                viewer_handle=viewer_handle,
                cam_to_world_matrix=cam_to_world_matrix,
                projection_matrix=projection_matrix,
                image_dimensions=image_dimensions[i],
                image=image,
                axis_length=self._axis_length,
                axis_thickness=self.axis_thickness,
                frustum_line_width=self._frustum_line_width,
                frustum_scale=self._frustum_scale,
                frustum_color=self._frustum_color,
                show_image=self._show_images,
                visible=self._enabled,
            )

    def layout_gui(self):
        gui = self._viewer_handle.gui

        # Constants for controlling range of gui sliders
        self._MIN_FRUSTUM_LINE_WIDTH = 1.0 * self._frustum_line_width
        self._MAX_FRUSTUM_LINE_WIDTH = 5.0 * self._frustum_line_width
        self._FRUSTUM_LINE_WIDTH_INCR = 0.5 * self._frustum_line_width
        self._MIN_FRUSTUM_SCALE = 0.1 * self._frustum_scale
        self._MAX_FRUSTUM_SCALE = 100.0 * self._frustum_scale
        self._FRUSTUM_SCALE_INCR = 0.1 * self._frustum_scale

        # These are factors because they multiply the initial values
        self._AXIS_MIN_FACTOR = 0.1
        self._AXIS_MAX_FACTOR = 3.0
        self._AXIS_INCR_FACTOR = 0.1

        with gui.add_folder(self._name) as self._name_gui_handle:
            self._enabled_gui_handle = gui.add_checkbox(
                "Enabled",
                self._enabled,
            )

            self._axis_length_gui_handle = gui.add_slider(
                "Axis Length",
                self._AXIS_MIN_FACTOR * self._axis_length,
                self._AXIS_MAX_FACTOR * self._axis_length,
                self._AXIS_INCR_FACTOR * self._axis_length,
                self._axis_length,
                disabled=not self._enabled,
            )
            self._axis_thickness_gui_handle = gui.add_slider(
                "Axis Thickness",
                self._AXIS_MIN_FACTOR * self.axis_thickness,
                self._AXIS_MAX_FACTOR * self.axis_thickness,
                self._AXIS_INCR_FACTOR * self.axis_thickness,
                self.axis_thickness,
                disabled=not self._enabled,
            )
            self._frustum_line_width_gui_handle = gui.add_slider(
                "Frustum Line Width",
                self._MIN_FRUSTUM_LINE_WIDTH,
                self._MAX_FRUSTUM_LINE_WIDTH,
                self._FRUSTUM_LINE_WIDTH_INCR,
                self._frustum_line_width,
                disabled=not self._enabled,
            )
            self._frustum_scale_gui_handle = gui.add_slider(
                "Frustum Scale",
                self._MIN_FRUSTUM_SCALE,
                self._MAX_FRUSTUM_SCALE,
                self._FRUSTUM_SCALE_INCR,
                self._frustum_scale,
                disabled=not self._enabled,
            )

            self._show_images_gui_handle = gui.add_checkbox("Show Images", self._show_images)

        self._enabled_gui_handle.on_update(self._enabled_update)
        self._axis_length_gui_handle.on_update(self._axis_length_update)
        self._axis_thickness_gui_handle.on_update(self._axis_thickness_update)
        self._show_images_gui_handle.on_update(self._show_images_update)
        self._frustum_line_width_gui_handle.on_update(self._frustum_line_width_update)
        self._frustum_scale_gui_handle.on_update(self._frustum_scale_update)

    def __getitem__(self, key: int) -> SingleCameraView:
        """
        Get a specific camera view by its index.

        Args:
            key (int): The index of the camera view to retrieve.

        Returns:
            SingleCameraView: The camera view at the specified index.
        """
        return self._camera_views[key]

    def __len__(self) -> int:
        """
        Get the number of camera views in this CameraView.

        Returns:
            int: The number of camera views.
        """
        return len(self._camera_views)

    def items(self):
        """
        Get an iterable view of the camera views in this CameraView.

        Returns:
            An iterable view of the camera views, similar to `dict.items()`.
        """
        return self._camera_views.items()

    def keys(self):
        """
        Get an iterable view of the keys (indices) of the camera views in this CameraView.

        Returns:
            An iterable view of the keys (indices) of the camera views, similar to `dict.keys()`.
        """
        return self._camera_views.keys()

    def values(self):
        """
        Get a list of the camera views in this CameraView.

        Returns:
            list[SingleCameraView]: A list of the camera views.
        """
        return list(self._camera_views.values())

    def __iter__(self):
        """
        Get an iterator over the camera views in this CameraView.

        Returns:
            An iterator over the camera view keys.
        """
        return iter(self._camera_views.keys())

    @property
    def enabled(self) -> bool:
        """
        Get whether the camera view UI is enabled.

        Returns:
            bool: True if the camera view UI is enabled, False otherwise.
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """
        Set whether the camera view UI is enabled.

        Args:
            value (bool): If True, the camera view UI is enabled and the cameras will be rendered.
                If False, the camera view UI is disabled and the cameras will not be rendered.
        """
        self._enabled = value
        self._axis_length_gui_handle.disabled = not value
        self._axis_thickness_gui_handle.disabled = not value
        self._show_images_gui_handle.disabled = not value
        self._frustum_line_width_gui_handle.disabled = not value
        self._frustum_scale_gui_handle.disabled = not value

        for camera_view in self._camera_views.values():
            camera_view.visible = value

    @property
    def name(self) -> str:
        """
        Get the name of the camera frustum view.

        Returns:
            str: The name of the camera frustum view.
        """
        return self._name

    @property
    def axis_thickness(self) -> float:
        """
        Get the thickness of the axis lines in the camera frustum view.

        Returns:
            float: The thickness of the axis lines.
        """
        return self._axis_thickness

    @axis_thickness.setter
    def axis_thickness(self, value: float):
        """
        Set the thickness of the axis lines in the camera frustum view.

        Args:
            value (float): The new thickness of the axis lines.
        """
        if value <= 0:
            raise ValueError("Axis thickness must be a positive number.")
        if value < self._axis_thickness_gui_handle.min or value > self._axis_thickness_gui_handle.max:
            raise ValueError(
                f"Axis thickness must be between {self._axis_thickness_gui_handle.min} and {self._axis_thickness_gui_handle.max}. ({self._AXIS_MIN_FACTOR} to {self._AXIS_MAX_FACTOR} times the initial value of {self.axis_thickness})"
            )
        self._axis_thickness = value
        self._axis_thickness_gui_handle.value = value
        for camera_view in self._camera_views.values():
            camera_view.axis_thickness = value

    @property
    def axis_length(self) -> float:
        """
        Get the length of the axis lines in the camera frustum view.

        Returns:
            float: The length of the axis lines.
        """
        return self._axis_length

    @axis_length.setter
    def axis_length(self, value: float):
        """
        Set the length of the axis lines in the camera frustum view.

        Args:
            value (float): The new length of the axis lines.
        """
        if value <= 0:
            raise ValueError("Axis length must be a positive number.")

        if value < self._axis_length_gui_handle.min or value > self._axis_length_gui_handle.max:
            raise ValueError(
                f"Axis length must be between {self._axis_length_gui_handle.min} and {self._axis_length_gui_handle.max}. ({self._AXIS_MIN_FACTOR} to {self._AXIS_MAX_FACTOR} times the initial value of {self.axis_length})"
            )
        self._axis_length = value
        self._axis_length_gui_handle.value = self._axis_length
        for camera_view in self._camera_views.values():
            camera_view.axis_length = self._axis_length

    @property
    def frustum_line_width(self) -> float:
        """
        Get the width of the frustum lines in the camera frustum view.

        Returns:
            float: The width of the frustum lines.
        """
        return self._frustum_line_width

    @frustum_line_width.setter
    def frustum_line_width(self, value: float):
        """
        Set the width of the frustum lines in the camera frustum view.

        Args:
            value (float): The new width of the frustum lines.
        """
        if value <= 0:
            raise ValueError("Frustum line width must be a positive number.")
        if value < self._frustum_line_width_gui_handle.min or value > self._frustum_line_width_gui_handle.max:
            raise ValueError(
                f"Frustum line width must be between {self._frustum_line_width_gui_handle.min} and {self._frustum_line_width_gui_handle.max}."
            )
        self._frustum_line_width = value
        self._frustum_line_width_gui_handle.value = self._frustum_line_width
        for camera_view in self._camera_views.values():
            camera_view.frustum_line_width = self._frustum_line_width

    @property
    def frustum_scale(self) -> float:
        """
        Get the scale of the frustum in the camera frustum view.

        Returns:
            float: The scale of the frustum.
        """
        return self._frustum_scale

    @frustum_scale.setter
    def frustum_scale(self, value: float):
        """
        Set the scale of the frustum in the camera frustum view.

        Args:
            value (float): The new scale of the frustum.
        """
        if value < self._frustum_scale_gui_handle.min or value > self._frustum_scale_gui_handle.max:
            raise ValueError(
                f"Frustum scale must be between {self._frustum_scale_gui_handle.min} and {self._frustum_scale_gui_handle.max}."
            )

        self._frustum_scale = value
        self._frustum_scale_gui_handle.value = self._frustum_scale
        for camera_view in self._camera_views.values():
            camera_view.frustum_scale = self._frustum_scale

    @property
    def show_images(self) -> bool:
        """
        Get whether images are currently shown in the camera frustum view.

        Returns:
            bool: True if images are shown, False otherwise.
        """
        assert self._show_images_gui_handle.value == self._show_images
        return self._show_images

    @show_images.setter
    def show_images(self, value: bool):
        """
        Set whether to show images in the camera frustum view.

        Args:
            value (bool): If True, images will be shown; if False, they will not.
        """
        self._show_images = value
        self._show_images_gui_handle.value = value
        for camera_view in self._camera_views.values():
            camera_view.show_image = value

    def _enabled_update(self, event: viser.GuiEvent):
        """
        Update the enabled state of the camera view based on the GUI checkbox value.

        Args:
            event (viser.GuiEvent): The GUI event containing the new enabled state.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiCheckboxHandle), "Expected a GuiCheckboxHandle for enabled state."
        self.enabled = target_handle.value

    def _frustum_scale_update(self, event: viser.GuiEvent):
        """
        Update the scale of the frustum based on the GUI slider value.

        Args:
            event (viser.GuiEvent): The GUI event containing the new scale value.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiSliderHandle), "Expected a GuiSliderHandle for frustum scale."
        self._frustum_scale = target_handle.value
        for camera_view in self._camera_views.values():
            camera_view.frustum_scale = self._frustum_scale

    def _frustum_line_width_update(self, event: viser.GuiEvent):
        """
        Update the width of the frustum lines based on the GUI slider value.

        Args:
            event (viser.GuiEvent): The GUI event containing the new line width.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiSliderHandle), "Expected a GuiSliderHandle for frustum line width."
        self._frustum_line_width = target_handle.value
        for camera_view in self._camera_views.values():
            camera_view.frustum_line_width = self._frustum_line_width

    def _show_images_update(self, event: viser.GuiEvent):
        """
        Update the visibility of images in the camera frustum view based on the GUI checkbox value.

        Args:
            event (viser.GuiEvent): The GUI event containing the new visibility state.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiCheckboxHandle), "Expected a GuiCheckboxHandle for image visibility."
        self._show_images = target_handle.value
        for camera_view in self._camera_views.values():
            camera_view.show_image = self._show_images

    def _axis_length_update(self, event: viser.GuiEvent):
        """
        Update the axis length based on the GUI slider value.

        Args:
            event (viser.GuiEvent): The GUI event containing the new axis length.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiSliderHandle), "Expected a GuiSliderHandle for axis length update."
        self._axis_length = target_handle.value
        for camera_view in self._camera_views.values():
            camera_view.axis_length = self._axis_length

    def _axis_thickness_update(self, event: viser.GuiEvent):
        """
        Update the axis thickness (2x the radius) based on the GUI slider value.

        Args:
            event (viser.GuiEvent): The GUI event containing the new axis radius.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiSliderHandle), "Expected a GuiSliderHandle for axis length update."
        self._axis_thickness = target_handle.value
        for camera_view in self._camera_views.values():
            camera_view.axis_thickness = self._axis_thickness
