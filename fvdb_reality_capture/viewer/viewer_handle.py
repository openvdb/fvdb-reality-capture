# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import dataclasses
from enum import Enum
from typing import Any, Callable, Literal

import viser


class ViewerAction(Enum):
    """
    Enum representing actions that can be performed on a viewer.
    This is used to communicate between views in the viewer and the viewer itself.
    """

    # Notify background rendering threads that the viewer has changed and they should re-render
    NOTIFY_GAUSSIAN_THREADS = 1

    # Pause background rendering threads
    PAUSE_GAUSSIAN_THREADS = 2

    # Resume background rendering threads
    RESUME_GAUSSIAN_THREADS = 3

    # Notify the viewer to rerender its GUI (e.g. after adding a new GUI element)
    RERENDER_GUI = 4

    # Set the up direction for all clients attached to the viewer.
    SET_UP_DIRECTION = 5

    # Set the target number of pixels to render per frame.
    SET_TARGET_PIXELS_PER_FRAME = 6

    # Set the maximum width of the rendered image.
    SET_MAX_IMAGE_WIDTH = 7

    # Set the far plane distance for the camera
    SET_CAMERA_FAR = 8

    # Set the near plane distance for the camera
    SET_CAMERA_NEAR = 9


@dataclasses.dataclass(kw_only=True)
class ViewerEvent:
    """
    Event that is sent to the viewer to notify it of an action that should be performed.
    This gets passed by views to the viewer's event handler.

    Warning: You should not create this event directly. Instead, use the ViewerHandle
    to perform actions on the viewer. This ensures that the viewer's event handler is
    properly set up and that the viewer is ready to handle the action.

    Attributes:
        viewer_server (viser.ViserServer): The unique ViserServer instance associated with the viewer.
        action (ViewerAction): The action to be performed on the viewer.
        gui_event (viser.GuiEvent | None): Optional GUI event associated with the action.
            This is used to pass additional information about the action, such as which GUI element triggered it.
    """

    viewer_server: viser.ViserServer  # The unique ViserServer instance associated with the viewer
    action: ViewerAction  # Action to be performed on the viewer
    gui_event: viser.GuiEvent | None = None  # Optional GUI event associated with the action


class ViewerHandle:
    """
    A handle class for interacting with a viewer.

    Views do not access the viewer directly, but instead use this handle
    to perform actions on the viewer. This allows for better separation of concerns and
    makes it easier to manage the viewer's state and behavior.
    """

    def __init__(self, viser_server: viser.ViserServer, event_handler: Callable[[ViewerEvent, Any], None]):
        """
        Create a new ViewerHandle associated with a specific viewer.
        This constructor should only be called from the viewer itself.

        The viewer creates this handle with its server (which is unique per viewer) and
        registers an event handler that will be called when actions are requested on the viewer.

        Args:
            viser_server (viser.ViserServer): The ViserServer instance associated with the viewer.
            event_handler (Callable[[ViewerEvent], None]): A callable that takes a ViewerEvent and handles it.
        """
        self._viser_server = viser_server
        self._event_handler = event_handler

    @property
    def scene(self) -> viser.SceneApi:
        """
        Get a handle to the 3D scene associated with the viewer.
        This allows views to modify the 3D scene, such as adding or removing objects.

        Returns:
            viser.SceneApi: The scene API for the viewer. This can be used to create and manage
                3D objects in the viewer's scene.
        """
        return self._viser_server.scene

    @property
    def gui(self) -> viser.GuiApi:
        """
        Get the GUI API associated with the viewer's server.
        This allows views to interact with the viewer's GUI.

        Returns:
            viser.GuiApi: The GUI API for the viewer. This can be used to create and manage
                GUI elements associated with the viewer.
        """
        return self._viser_server.gui

    def notify_render_threads(self) -> None:
        """
        Tell the viewer to notify its background rendering threads that something has changed
        and they should re-render the scene.
        """
        event = ViewerEvent(viewer_server=self._viser_server, action=ViewerAction.NOTIFY_GAUSSIAN_THREADS)
        self._event_handler(event, None)

    def pause_gaussian_render_threads(self) -> None:
        """
        Tell the viewer to pause its background rendering threads.
        """
        event = ViewerEvent(viewer_server=self._viser_server, action=ViewerAction.PAUSE_GAUSSIAN_THREADS)
        self._event_handler(event, None)

    def resume_gaussian_render_threads(self) -> None:
        """
        Tell the viewer to resume its background rendering threads.
        """
        event = ViewerEvent(viewer_server=self._viser_server, action=ViewerAction.RESUME_GAUSSIAN_THREADS)
        self._event_handler(event, None)

    def rerender_gui(self, gui_event: viser.GuiEvent | None = None) -> None:
        """
        Tell the viewer that the GUI layout has changed and it should re-render its GUI.

        Args:
            gui_event (viser.GuiEvent | None): Optional GUI event associated with the re-rendering.
        """
        event = ViewerEvent(viewer_server=self._viser_server, action=ViewerAction.RERENDER_GUI, gui_event=gui_event)
        self._event_handler(event, None)

    def set_up_direction(
        self, gui_event: viser.GuiEvent, up_direction: Literal["+x", "+y", "+z", "-x", "-y", "-z"]
    ) -> None:
        """
        Set the up direction for all clients attached to the viewer.
        This is used to change the up direction of the scene.
        Args:
            gui_event (viser.GuiEvent): The GUI event that triggered this action.
                This is used to pass additional information about the action, such as which GUI element triggered it.
            up_direction (Literal["+x", "+y", "+z", "-x", "-y", "-z"]): The up direction to set for the viewer.
                This should be one of the six cardinal directions.
                It determines the orientation of the camera and objects in the scene.
        """
        event = ViewerEvent(viewer_server=self._viser_server, action=ViewerAction.SET_UP_DIRECTION, gui_event=gui_event)
        self._event_handler(event, up_direction)

    def set_target_pixels_per_frame(self, gui_event: viser.GuiEvent, target_pixels_per_frame: int) -> None:
        """
        Set the target number of pixels to render per frame.
        This is used to adjust the resolution of the rendered images to achieve a specific target pixel count.

        Args:
            gui_event (viser.GuiEvent): The GUI event that triggered this action.
                This is used to pass additional information about the action, such as which GUI element triggered it.
        """
        self._event_handler(
            ViewerEvent(
                viewer_server=self._viser_server, action=ViewerAction.SET_TARGET_PIXELS_PER_FRAME, gui_event=gui_event
            ),
            target_pixels_per_frame,
        )

    def set_max_image_width(self, gui_event: viser.GuiEvent, max_image_width: int) -> None:
        """
        Set the maximum width of the rendered image.
        This is used to ensure that the rendered images do not exceed a certain width.

        Args:
            gui_event (viser.GuiEvent): The GUI event that triggered this action.
                This is used to pass additional information about the action, such as which GUI element triggered it.
        """
        self._event_handler(
            ViewerEvent(viewer_server=self._viser_server, action=ViewerAction.SET_MAX_IMAGE_WIDTH, gui_event=gui_event),
            max_image_width,
        )

    def set_camera_far(self, gui_event: viser.GuiEvent, camera_far: float) -> None:
        """
        Set the far plane distance for the camera.
        This is used to define the maximum distance at which objects are rendered.

        Args:
            gui_event (viser.GuiEvent): The GUI event that triggered this action.
                This is used to pass additional information about the action, such as which GUI element triggered it.
            camera_far (float): The far plane distance for the camera.
        """

        self._event_handler(
            ViewerEvent(viewer_server=self._viser_server, action=ViewerAction.SET_CAMERA_FAR, gui_event=gui_event),
            camera_far,
        )

    def set_camera_near(self, gui_event: viser.GuiEvent, camera_near: float) -> None:
        """
        Set the near plane distance for the camera.
        This is used to define the minimum distance at which objects are rendered.

        Args:
            gui_event (viser.GuiEvent): The GUI event that triggered this action.
                This is used to pass additional information about the action, such as which GUI element triggered it.
            camera_near (float): The near plane distance for the camera.
        """

        self._event_handler(
            ViewerEvent(viewer_server=self._viser_server, action=ViewerAction.SET_CAMERA_NEAR, gui_event=gui_event),
            camera_near,
        )


# Only expose the ViewerHandle class to the outside world.
__all__ = ["ViewerHandle"]
