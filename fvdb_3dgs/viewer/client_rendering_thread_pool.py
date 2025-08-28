# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import dataclasses
import os
import sys
import threading
import time
import traceback
from enum import Enum
from typing import Literal

import numpy as np
import torch
import viser
import viser.transforms as vt

from .client_thread_view import ClientThreadRenderingView


class ClientRenderThreadInterrupted(Exception):
    pass


class set_trace_context(object):
    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, *_, **__):
        sys.settrace(None)


class RenderState(Enum):
    """
    Enum representing the different rendering states of the of the scene.
    """

    LOW_RES_MOVING = 1
    LOW_RES_STATIC = 2
    HIGH_RES = 3


class RenderAction(Enum):
    """
    Enum representing the different actions that can be taken by the rendering thread.
    - STAY_PUT: The camera is not moving, and we can render in high resolution.
    - MOVE_CAMERA: The camera is moving, and we need to render in low resolution.
    - UPDATE_STATE: The state of the scene has changed, and we need to update the rendering.
    """

    STAY_PUT = 1
    MOVE_CAMERA = 2
    UPDATE_STATE = 3


@dataclasses.dataclass(kw_only=True)
class ThreadLocalCamera(object):
    """
    A thread-local camera representation for rendering.

    We use this to avoid having the rendering thread read the viewer's camera state directly
    which may change underneath us while we're rendering.

    Attributes:
        fov (float): Field of view of the camera in radians.
        aspect (float): Aspect ratio of the camera (W/H).
        rotation_quat (np.ndarray): Rotation of the camera as a quaternion (w, x, y, z).
        position (np.ndarray): Position of the camera in world coordinates.
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.
        camera_model (Literal["perspective", "orthographic"]): The camera model used for rendering.
            Defaults to "perspective". This is used to determine how the camera projects points in
            3D space onto the 2D image plane.
            Currently, we only support "perspective" cameras because viser does not support orthographic cameras.
    """

    fov: float
    aspect: float
    rotation_quat: np.ndarray
    position: np.ndarray
    near: float
    far: float
    camera_model: Literal["perspective", "orthographic"] = "perspective"

    @staticmethod
    def from_viser_camera(camera: viser.CameraHandle) -> "ThreadLocalCamera":
        """
        Create a ThreadLocalCamera from a viser.CameraHandle.

        Args:
            camera (viser.CameraHandle): The camera handle from the viser client.
        """
        return ThreadLocalCamera(
            fov=camera.fov,
            aspect=camera.aspect,
            rotation_quat=camera.wxyz,
            position=camera.position,
            near=camera.near,
            far=camera.far,
        )

    @torch.no_grad()
    def cam_to_world_matrix(self) -> torch.Tensor:
        """
        Build the camera-to-world transformation matrix from the camera's position and rotation.

        Returns:
            torch.Tensor: A 4x4 transformation matrix that transforms points from camera space to world space.
        """
        return torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate([vt.SO3(self.rotation_quat).as_matrix(), self.position[:, None]], 1),
                    [[0, 0, 0, 1]],
                ],
                0,
            )
        ).to(torch.float32)

    @torch.no_grad()
    def world_to_cam_matrix(self) -> torch.Tensor:
        """
        Build the world-to-camera transformation matrix from the camera's position and rotation.

        Returns:
            torch.Tensor: A 4x4 transformation matrix that transforms points from world space to camera space.
        """
        return torch.linalg.inv(self.cam_to_world_matrix()).contiguous()

    @torch.no_grad()
    def projection_matrix(self, img_w: int, img_h: int) -> torch.Tensor:
        """
        Build the projection matrix for the camera and specified image resolution.

        Args:
            img_w (int): The width of the image.
            img_h (int): The height of the image.

        Returns:
            torch.Tensor: A 3x3 projection matrix.
        """
        focal_length = img_h / 2.0 / np.tan(self.fov / 2.0)
        return torch.from_numpy(
            np.array(
                [
                    [focal_length, 0.0, img_w / 2.0],
                    [0.0, focal_length, img_h / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        ).to(torch.float32)


class ClientRenderThread(threading.Thread):
    """
    A background thread that renders images on the server for a specific web client
    connected to the viewer. This is useful when you have a custom type of view that you want
    to render in the viewer for which the viewer does not have a built-in rendering
    implementation, such as a Gaussian splat scene.

    We do this in a thread to avoid blocking the main UI thread for the viewer.

    The thread will render scenes in low resolution while the user is interacting with the scene
    to achieve a targe framerate, and switch to high resolution rendering when the camera is not moving.

    This is handled by the `RenderState` and `RenderAction` enums which define a state machine with
    three states and three actions.

    - The state machine has three **states**:
        - `HIGH_RES`: The camera is not moving, and we can render in high resolution.
        - `LOW_RES_MOVING`: The camera is moving, and we need to render in low resolution.
        - `LOW_RES_STATIC`: The camera is not moving, but we are rendering in low resolution.
    - The **actions** that can be taken are:
        - `STAY_PUT`: The camera is not moving, and we can switch to rendering in high resolution.
        - `MOVE_CAMERA`: The camera is moving, and we need to switch to rendering in low resolution.
        - `UPDATE_STATE`: The state of the scene has changed, and we need to update the rendering so
        render the next frame in low resolution.

    """

    def __init__(
        self,
        client: viser.ClientHandle,
        render_views: dict[str, tuple[int, ClientThreadRenderingView]],
        viewer_lock: threading.Lock,
        target_pixels_per_frame: int = 100_000,
        target_framerate: float = 30,
        max_image_width: int = 2048,
    ):
        """
        Initialize the rendering thread. The rendering thread tracks which client it's rendering for,
        and the set of scenes it's responsible for rendering.

        Args:
            client (viser.ClientHandle): The client handle for the viewer client this thread is rendering for.
            render_views (dict[str, type[ClientThreadRenderingView]]): A dictionary mapping scene names to their corresponding ClientThreadRenderingView instances.
            viewer_lock (threading.Lock): A global lock shared with the viewer to ensure thread-safe access to shared resources.
        """
        super().__init__(daemon=True)

        self._scenes: dict[str, tuple[int, ClientThreadRenderingView]] = render_views
        self._client: viser.ClientHandle = client
        self._lock: threading.Lock = viewer_lock
        self._running: bool = True
        self._last_img: np.ndarray | None = None  # Store the last image this thread rendered
        self._paused = False

        self._update_event = threading.Event()
        self._state: RenderState = RenderState.LOW_RES_STATIC
        self._current_camera: ThreadLocalCamera = ThreadLocalCamera.from_viser_camera(client.camera)
        self._task = (RenderAction.STAY_PUT, self._current_camera)

        # Track the number of pixels rendered per second for this scene
        self._pixels_per_second: float = 0.0

        self._may_interrupt_render = False

        self._target_pixels_per_frame: int = target_pixels_per_frame
        self._target_framerate: float = target_framerate
        self._max_image_width: int = max_image_width

    @property
    def target_pixels_per_frame(self) -> int:
        """
        Returns the target number of pixels per frame for this thread.
        This is used to determine the resolution of the rendered images based on the current state.

        Returns:
            int: The target number of pixels per frame.
        """
        return self._target_pixels_per_frame

    @target_pixels_per_frame.setter
    def target_pixels_per_frame(self, value: int):
        """
        Set the target number of pixels per frame for this thread.
        This is used to determine the resolution of the rendered images based on the current state.

        Args:
            value (int): The target number of pixels per frame.
        """
        self._target_pixels_per_frame = value

    @property
    def target_framerate(self) -> float:
        """
        Returns the target framerate for this thread.
        This is used to determine the resolution of the rendered images based on the current state.

        Returns:
            float: The target framerate.
        """
        return self._target_framerate

    @target_framerate.setter
    def target_framerate(self, value: float):
        """
        Set the target framerate for this thread.
        This is used to determine the resolution of the rendered images based on the current state.

        Args:
            value (int): The target framerate.
        """
        self._target_framerate = value

    @property
    def max_image_width(self) -> int:
        """
        Returns the maximum image width for this thread.
        This is used to determine the resolution of the rendered images based on the current state.

        Returns:
            int: The maximum image width.
        """
        return self._max_image_width

    @max_image_width.setter
    def max_image_width(self, value: int):
        """
        Set the maximum image width for this thread.
        This is used to determine the resolution of the rendered images based on the current state.

        Args:
            value (int): The maximum image width.
        """
        self._max_image_width = value

    def stop(self):
        """
        Signal to the thread to stop running. This will cause the thread to exit gracefully.
        It sets the running flag to False and signals the update event to wake up the thread if it is waiting.
        """
        self._running = False
        self._update_event.set()

    def pause(self):
        """
        Pause rendering for this thread. If the thread is already paused, this will not block.
        """
        self._paused = True

    def resume(self, camera: viser.CameraHandle):
        """
        Resume rendering for this thread. If the thread is not paused, this will not block.
        """
        self._paused = False
        self.update_state(camera)

    def move_camera(self, camera: viser.CameraHandle):
        """
        Called by the viewer thread to indicate that the client's camera has moved.

        Args:
            camera (viser.CameraHandle): The new camera to use for rendering.
        """
        self._task = (RenderAction.MOVE_CAMERA, ThreadLocalCamera.from_viser_camera(camera))
        if self._state == RenderState.HIGH_RES:
            self._may_interrupt_render = True
        self._update_event.set()

    def update_state(self, camera: viser.CameraHandle | None = None):
        """
        Called by the viewer thread to indicate that the state of the scene has changed, or the camera has moved.

        Args:
            camera (viser.CameraHandle | None): The new camera to use for rendering. If None, the current camera is used.
            If provided, this will update the current camera state for rendering.
        """
        cam = ThreadLocalCamera.from_viser_camera(camera) if camera is not None else self._current_camera
        self._task = (RenderAction.UPDATE_STATE, cam)
        if self._state == RenderState.HIGH_RES:
            self._may_interrupt_render = True
        self._update_event.set()

    def _image_resolution_for_current_camera(
        self, camera: ThreadLocalCamera, render_view: "ClientThreadRenderingView"
    ) -> tuple[int, int]:
        """
        Determine at what resolution to render the images based on the current state of the viewer and the
        specified maximum resolution in the `ClientThreadRenderingView`.

        If the viewer is rendering in low resolution mode, we also use the target pixels per frame and target framerate
        specified in the `ClientThreadRenderingView` to determine the image resolution.

        Args:
            camera (ThreadLocalCamera): The current camera state for rendering.
            splat_view (ClientThreadRenderingView): The view containing rendering parameters.
        """
        if self._state == RenderState.HIGH_RES:
            img_height = self.max_image_width
            img_width = int(img_height * camera.aspect)
            if img_width > self.max_image_width:
                img_width = self.max_image_width
                img_height = int(img_width / camera.aspect)
        elif self._state in [RenderState.LOW_RES_MOVING, RenderState.LOW_RES_STATIC]:
            target_view_rays_per_sec = self.target_pixels_per_frame
            target_fps = self.target_framerate
            num_viewer_rays = target_view_rays_per_sec / target_fps
            img_height = (num_viewer_rays / camera.aspect) ** 0.5
            img_height = int(round(img_height, -1))
            img_height = max(min(self.max_image_width, img_height), 30)
            img_width = int(img_height * camera.aspect)
            if img_width > self.max_image_width:
                img_width = self.max_image_width
                img_height = int(img_width / camera.aspect)
        else:
            raise ValueError(f"Invalid state: {self._state}.")
        return img_width, img_height

    @property
    def pixels_per_second(self) -> float:
        """
        Returns the number of pixels per second rendered by this thread.
        This is updated every time a frame gets rendered.

        Returns:
            float: The number of pixels per second rendered by this thread.
        """
        return self._pixels_per_second

    def _may_interrupt_trace(self, frame, event, arg):
        if event == "line":
            if self._may_interrupt_render:
                self._may_interrupt_render = False
                raise ClientRenderThreadInterrupted
        return self._may_interrupt_trace

    def _next_state(self, action: RenderAction) -> RenderState:
        """
        Move the thread from its current state to the next state based on the action taken.

        The state machine has three states:
        - `HIGH_RES`: The camera is not moving, and we can render in high resolution.
        - `LOW_RES_MOVING`: The camera is moving, and we need to render in low resolution.
        - `LOW_RES_STATIC`: The camera is not moving, but we are rendering in low resolution.

        The actions that can be taken are:
        - `STAY_PUT`: The camera is not moving, and we can switch to rendering in high resolution.
        - `MOVE_CAMERA`: The camera is moving, and we need to switch to rendering in low resolution.
        - `UPDATE_STATE`: The state of the scene has changed, and we need to update the rendering so render the next frame in low resolution.

        Args:
            action (RenderAction): The action taken by the viewer thread.

        Returns:
            RenderState: The next state of the rendering thread based on the current state and action taken.
        """
        if self._state == RenderState.HIGH_RES and action == RenderAction.STAY_PUT:
            # If we're in high res mode and the camera is not moving, we can skip rendering.
            return RenderState.HIGH_RES
        elif self._state == RenderState.HIGH_RES and action == RenderAction.MOVE_CAMERA:
            # If we're in high res mode and the camera is moving, we need to switch to low res mode.
            return RenderState.LOW_RES_MOVING
        elif self._state == RenderState.HIGH_RES and action == RenderAction.UPDATE_STATE:
            # If we're in high res mode and the camera is moving, we need to switch to low res mode.
            return RenderState.LOW_RES_STATIC

        elif self._state == RenderState.LOW_RES_MOVING and action == RenderAction.STAY_PUT:
            # If we're in low res moving mode and the camera is not moving, we need to switch to low res static.
            return RenderState.LOW_RES_STATIC
        elif self._state == RenderState.LOW_RES_MOVING and action == RenderAction.MOVE_CAMERA:
            # If we're in low res moving mode and the camera is still moving, we stay in low res moving.
            return RenderState.LOW_RES_MOVING
        elif self._state == RenderState.LOW_RES_MOVING and action == RenderAction.UPDATE_STATE:
            # If we're in low res moving mode and the camera is still moving, we stay in low res moving.
            return RenderState.LOW_RES_MOVING

        elif self._state == RenderState.LOW_RES_STATIC and action == RenderAction.MOVE_CAMERA:
            # If we're in low res static mode and the camera is moving, we need to switch to low res moving.
            return RenderState.LOW_RES_MOVING
        elif self._state == RenderState.LOW_RES_STATIC and action == RenderAction.UPDATE_STATE:
            # If we're in low res static mode and the camera is moving, we need to switch to low res moving.
            return RenderState.LOW_RES_STATIC
        elif self._state == RenderState.LOW_RES_STATIC and action == RenderAction.STAY_PUT:
            # If we're in low res static mode and the camera is not moving, we need to switch to high res.
            return RenderState.HIGH_RES
        else:
            raise ValueError(f"Unknown state: {self._state} and action: {action}.")

    def run(self):
        while self._running:
            # Try to acquire the paused lock. If we can't then the viewer must be holding it
            # and we should wait until it releases it.
            rendered_paused_image = False
            while self._paused:
                if self._last_img is not None and not rendered_paused_image:
                    rendered_paused_image = True
                    self._client.scene.set_background_image(
                        self._last_img * 0.5,
                        format="jpeg",
                        jpeg_quality=40,
                        depth=None,
                    )
                time.sleep(0.1)

            if not self._update_event.wait(0.2):
                self._task = (RenderAction.STAY_PUT, ThreadLocalCamera.from_viser_camera(self._client.camera))
                self._update_event.set()

            self._update_event.clear()
            action, self._current_camera = self._task
            next_state = self._next_state(action)
            if next_state == self._state == RenderState.HIGH_RES:
                # If we're already in high res mode, we can skip rendering.
                continue
            self._state = next_state

            try:
                composited_image: torch.Tensor | None = None
                composited_depth: torch.Tensor | None = None
                tic = time.time()

                with self._lock, set_trace_context(self._may_interrupt_trace):
                    sorted_scenes = sorted(self._scenes.items(), key=lambda x: x[1][0])
                    for scene_name, id_and_scene in sorted_scenes:
                        scene_id, scene = id_and_scene
                        camera = self._current_camera

                        img_w, img_h = self._image_resolution_for_current_camera(camera, scene)

                        world_to_cam_matrix = camera.world_to_cam_matrix().to(dtype=torch.float32)
                        projection_matrix = camera.projection_matrix(img_w, img_h).to(dtype=torch.float32)

                        with torch.no_grad():
                            composited_image, composited_depth = scene._render(
                                composited_image,
                                composited_depth,
                                world_to_cam_matrix,
                                projection_matrix,
                                img_w,
                                img_h,
                                camera.near,
                                camera.far,
                                camera.camera_model,
                            )
                    if composited_image is None:
                        self._pixels_per_second = 0.0
                    else:
                        self._pixels_per_second = (len(self._scenes.items()) * composited_image.numel() / 3) / (
                            time.time() - tic
                        )

                composited_image_np = composited_image.cpu().numpy() if composited_image is not None else None
                composited_depth_np = composited_depth.cpu().numpy() if composited_depth is not None else None
                self._last_img = composited_image_np
                self._client.scene.set_background_image(
                    composited_image_np,
                    format="jpeg",
                    jpeg_quality=70 if action == RenderAction.STAY_PUT else 40,
                    depth=composited_depth_np,
                )
            except ClientRenderThreadInterrupted:
                # If we got an interrupt, we just skip this frame and continue.
                continue
            except Exception:
                traceback.print_exc()
                os._exit(1)


class ClientRenderingThreadPool:
    def __init__(
        self,
        viser_server: viser.ViserServer,
        render_views: dict[str, tuple[int, ClientThreadRenderingView]],
        lock: threading.Lock,
        target_pixels_per_frame: int = 100_000,
        target_framerate: float = 30,
        max_image_width: int = 2048,
    ):
        self._client_rendering_threads: dict[int, ClientRenderThread] = {}
        self._viser_server: viser.ViserServer = viser_server
        self._render_views: dict[str, tuple[int, ClientThreadRenderingView]] = render_views
        self._lock: threading.Lock = lock
        self._target_pixels_per_frame: int = target_pixels_per_frame
        self._target_framerate: float = target_framerate
        self._max_image_width: int = max_image_width

    @property
    def target_pixels_per_frame(self) -> int:
        """
        Returns the target number of pixels per frame for all threads in the pool.
        This is used to determine the resolution of the rendered images based on the current state.

        Returns:
            int: The target number of pixels per frame.
        """
        return self._target_pixels_per_frame

    @target_pixels_per_frame.setter
    def target_pixels_per_frame(self, value: int):
        """
        Set the target number of pixels per frame for all threads in the pool.
        This is used to determine the resolution of the rendered images based on the current state.

        Args:
            value (int): The target number of pixels per frame.
        """
        self._target_pixels_per_frame = value
        for client_id in self._client_rendering_threads:
            client = self._viser_server.get_clients()[client_id]
            thread = self._client_rendering_threads[client_id]
            thread.target_pixels_per_frame = value
            thread.update_state(client.camera)

    @property
    def target_framerate(self) -> float:
        """
        Returns the target framerate for all threads in the pool.
        This is used to determine the resolution of the rendered images based on the current state.

        Returns:
            float: The target framerate.
        """
        return self._target_framerate

    @target_framerate.setter
    def target_framerate(self, value: float):
        """
        Set the target framerate for all threads in the pool.
        This is used to determine the resolution of the rendered images based on the current state.

        Args:
            value (float): The target framerate.
        """
        self._target_framerate = value
        for client_id in self._client_rendering_threads:
            client = self._viser_server.get_clients()[client_id]
            thread = self._client_rendering_threads[client_id]
            thread.target_framerate = value
            thread.update_state(client.camera)

    @property
    def max_image_width(self) -> int:
        """
        Returns the maximum image width for all threads in the pool.
        This is used to determine the resolution of the rendered images based on the current state.

        Returns:
            int: The maximum image width.
        """
        return self._max_image_width

    @max_image_width.setter
    def max_image_width(self, value: int):
        """
        Set the maximum image width for all threads in the pool.
        This is used to determine the resolution of the rendered images based on the current state.

        Args:
            value (int): The maximum image width.
        """
        self._max_image_width = value
        for client_id in self._client_rendering_threads:
            client = self._viser_server.get_clients()[client_id]
            thread = self._client_rendering_threads[client_id]
            thread.max_image_width = value
            thread.update_state(client.camera)

    def unregister_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._client_rendering_threads[client_id].stop()
        self._client_rendering_threads.pop(client_id)

    def register_client(self, client: viser.ClientHandle):
        client_render_thread = ClientRenderThread(
            client=client, render_views=self._render_views, viewer_lock=self._lock
        )
        self._client_rendering_threads[client.client_id] = client_render_thread
        client_render_thread.start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            with self._viser_server.atomic():
                client_render_thread.move_camera(client.camera)

    def notify_threads(self):
        for client_id in self._client_rendering_threads:
            client = self._viser_server.get_clients()[client_id]
            thread = self._client_rendering_threads[client_id]
            thread.update_state(client.camera)

    def pause_threads(self):
        for thread in self._client_rendering_threads.values():
            thread.pause()

    def resume_threads(self):
        for thread in self._client_rendering_threads.values():
            thread.resume(self._viser_server.get_clients()[thread._client.client_id].camera)
