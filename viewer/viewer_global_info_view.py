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

    @property
    def camera_up_axis(self) -> CameraUpAxis:
        """
        Returns the current camera up axis.
        """
        return self._camera_up_axis

    def layout_gui(self):
        """
        Define the GUI layout for the `ViewerGlobalInfoView`.
        """
        gui: viser.GuiApi = self._viewer_handle.gui

        with gui.add_folder("fVDB Viewer", visible=True):
            self._camera_up_axis_selector_handle = gui.add_dropdown(
                "Camera Up Axis", ["+x", "+y", "+z", "-x", "-y", "-z"], self._camera_up_axis
            )

        self._camera_up_axis_selector_handle.on_update(self._camera_up_axis_update)

    def _camera_up_axis_update(self, event: viser.GuiEvent):
        """
        Callback function for when the Up Axis dropdown is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the dropdown update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiDropdownHandle)
        self._camera_up_axis = target_handle.value
        self._viewer_handle.set_up_direction(event)
