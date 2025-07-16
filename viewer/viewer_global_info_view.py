# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Literal

import viser

from .viewer_handle import ViewerHandle

CameraUpAxis = Literal["+x", "+y", "+z", "-x", "-y", "-z"]


class ViewerGlobalInfoView:
    """
    A view for displaying and controlling global information about a scene rendered by an fVDB viewer.

    Note: This view is used internally by the viewer and we don't expose it directly to the user.
    """

    def __init__(self, viewer_handle: ViewerHandle, camera_up_axis: CameraUpAxis = "-z"):
        """
        Initializes the `ViewerGlobalInfoView` with a ViewerHandle and an up axis.

        Args:
            viewer_handle (ViewerHandle): The handle to the viewer.
            camera_up_axis (Literal["+x", "+y", "+z", "-x", "-y", "-z"]): The up axis for cameras viewing the scene. Defaults to '-z'.
        """
        self._viewer_handle: ViewerHandle = viewer_handle
        self._camera_up_axis: CameraUpAxis = camera_up_axis
        self._low_res_render_width = 64  # Default target pixels per frame
        self._target_framerate = 30.0  # Default target framerate
        self._max_render_width = 2048  # Default maximum image width
        self._max_image_width_min = 64  # Minimum possible value a user can set for the maximum image width
        self._max_image_width_max = 2048  # Maximum possible value a user can set for the maximum image width

    @property
    def camera_up_axis(self) -> CameraUpAxis:
        """
        Returns the current camera up axis.
        """
        return self._camera_up_axis

    @property
    def target_framerate(self) -> float:
        """
        Returns the target framerate for rendering.
        """
        return self._target_framerate

    @property
    def low_res_render_width(self) -> int:
        """
        Returns the low resolution rendering width.
        This is used to set the target pixels per frame for low resolution rendering in background threads.
        """
        return self._low_res_render_width

    @property
    def max_render_width(self) -> int:
        """
        Returns the maximum render width for rendering.
        This is used to limit the width of the rendered frames.
        """
        return self._max_render_width

    @property
    def max_render_width_upper_bound(self) -> int:
        """
        Returns the upper bound for the maximum render width.
        This is used to set the maximum value for the max image width slider.
        """
        return self._max_image_width_max

    def layout_gui(self):
        """
        Define the GUI layout for the `ViewerGlobalInfoView`.
        """
        gui: viser.GuiApi = self._viewer_handle.gui

        with gui.add_folder("fVDB Viewer", visible=True):
            self._camera_up_axis_selector_handle = gui.add_dropdown(
                "Camera Up Axis", ["+x", "+y", "+z", "-x", "-y", "-z"], self._camera_up_axis
            )
            self._max_image_width_handle = gui.add_slider(
                "Max Render Width",
                min=self._max_image_width_min,
                max=self._max_image_width_max,
                step=1,
                initial_value=self._max_render_width,
            )
            self._image_width_range_handle = gui.add_number(
                "Max Render Width Upper Bound",
                max=8192,
                min=64,
                initial_value=self._max_image_width_max,
                step=64,
            )
            self._low_res_image_width_handle = gui.add_slider(
                "Low Res Render Width",
                min=32,
                max=512,
                step=1,
                initial_value=self._low_res_render_width,
            )

        self._camera_up_axis_selector_handle.on_update(self._camera_up_axis_update)
        self._max_image_width_handle.on_update(self._max_image_width_update)
        self._image_width_range_handle.on_update(self._max_image_width_range_update)
        self._low_res_image_width_handle.on_update(self._low_res_image_width_update)

    def _low_res_image_width_update(self, event: viser.GuiEvent):
        """
        Callback function for when the low resolution image width slider is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the slider update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiSliderHandle)

        target_pixels_per_frame = (target_handle.value**2) * self._target_framerate
        self._low_res_render_width = target_handle.value
        self._viewer_handle.set_target_pixels_per_frame(event, target_pixels_per_frame)

    def _max_image_width_update(self, event: viser.GuiEvent):
        """
        Callback function for when the max image width slider is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the slider update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiSliderHandle)
        self._max_render_width = target_handle.value
        self._viewer_handle.set_max_image_width(event, self._max_render_width)

    def _max_image_width_range_update(self, event: viser.GuiEvent):
        """
        Callback function for when the range of the max image width slider is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the slider update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiNumberHandle)

        self._max_image_width_handle.max = target_handle.value

    def _camera_up_axis_update(self, event: viser.GuiEvent):
        """
        Callback function for when the Up Axis dropdown is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the dropdown update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiDropdownHandle)
        self._camera_up_axis = target_handle.value
        self._viewer_handle.set_up_direction(event, self._camera_up_axis)
