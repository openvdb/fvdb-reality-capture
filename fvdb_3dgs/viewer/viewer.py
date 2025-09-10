# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import threading
from typing import Any, Literal, Sequence

import numpy as np
import torch
import viser

from fvdb import GaussianSplat3d

from .camera_view import CameraView
from .client_rendering_thread_pool import ClientRenderingThreadPool
from .client_thread_view import ClientThreadRenderingView
from .custom_render_view import CustomRenderView, RenderCallBack
from .dict_label_view import DictLabelView
from .gaussian_splat_3d_view import GaussianSplat3dView
from .viewer_global_info_view import ViewerGlobalInfoView
from .viewer_handle import ViewerAction, ViewerEvent, ViewerHandle


class Viewer(object):
    """
    This is the main fVDB viewer class that provides a GUI for visualizing and interacting with
    3D data such as Gaussian splats and point clouds as well as utilities for adding GUI widgets.

    A viewer spins up a small web server on the specified port and serves a web-based GUI
    that can be accessed from a browser. The viewer allows for the registration of various
    3D objects and UI components.
    """

    def __init__(
        self,
        port: int = 8080,
        verbose: bool = False,
        camera_up_axis: Literal["+x", "+y", "+z", "-x", "-y", "-z"] = "-z",
    ):
        """
        Create a new Viewer instance. A viewer spins up a small web server on the specified port
        and serves a web-based GUI that can be accessed from a browser. The viewer allows for the
        registration of various 3D objects and UI components.

        Args:
            port (int): The port on which the viewer's web server will run. Defaults to 8080.
            verbose (bool): If True, enables verbose logging for the viewer. Defaults to False.
            camera_up_axis (Literal["+x", "+y", "+z", "-x", "-y", "-z"]): The up axis for the camera.
                This determines the orientation of the camera in the viewer. Defaults to "-z".
        """
        self._viser_server = viser.ViserServer(port=port, verbose=verbose)

        self._handle = ViewerHandle(viser_server=self._viser_server, event_handler=self._view_event_handler)
        self._lock = threading.Lock()

        self._gaussian_splat_3d_views: dict[str, tuple[int, ClientThreadRenderingView]] = {}
        self._dict_label_views: dict[str, DictLabelView] = {}
        self._camera_frustum_views: dict[str, CameraView] = {}
        self._background_rendering_view_counter = 0

        self._global_info_view: ViewerGlobalInfoView = ViewerGlobalInfoView(
            viewer_handle=self._handle, camera_up_axis=camera_up_axis
        )
        self._gaussian_client_render_thread_pool = ClientRenderingThreadPool(
            self._viser_server, self._gaussian_splat_3d_views, self._lock
        )

        self._viser_server.on_client_disconnect(self._on_client_disconnect)
        self._viser_server.on_client_connect(self._on_client_connect)

        self._layout_gui()
        self.set_up_direction(camera_up_axis)

    @property
    def viser_server(self) -> viser.ViserServer:
        """
        Get the Viser server instance associated with this viewer.

        Returns:
            viser.ViserServer: The Viser server instance.
        """
        return self._viser_server

    def set_up_direction(self, up_axis: Literal["+x", "+y", "+z", "-x", "-y", "-z"]):
        """
        Sets the camera up direction for all web clients connected to this viewer.

        Args:
            up_axis (Literal["+x", "+y", "+z", "-x", "-y", "-z"]): The up axis to set.
        """
        self._global_info_view.camera_up_axis = up_axis

    def register_camera_view(
        self,
        name: str,
        cam_to_world_matrices: Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
        projection_matrices: Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
        image_sizes: np.ndarray | torch.Tensor,
        images: Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None,
        axis_length: float = 0.5,
        axis_thickness: float = 0.0125,
        frustum_line_width: float = 2.0,
        frustum_scale: float = 1.0,
        frustum_color: Sequence[float] | np.ndarray = (0.0, 1.0, 0.0),
        show_images: bool = True,
        enabled: bool = True,
    ):
        """
        Register a new camera view to be visualized in the viewer.

        A camera view displays a set of 3D camera frustums (with optional images) in the viewer.

        Args:
            name (str): The name of the camera view.
            cam_to_world_matrices (Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray):
                A sequence of camera-to-world transformation matrices, or a single tensor/array.
                Each matrix should have shape (4, 4) and represent the transformation from camera space
                to world space. If a single tensor/array is provided, it should have shape (N, 4, 4) where N is the number of cameras.
            projection_matrices (Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray):
                A sequence of projection matrices, or a single tensor/array.
                Each matrix should have shape (3, 3) and represent the projection transformation from camera space to image space.
                If a single tensor/array is provided, it should have shape (N, 3, 3) where N is the number of cameras.
            image_sizes (np.ndarray): An array of shape (N, 2) containing the height and width of each camera image.
            images (Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None):
                An optional sequence of images (as numpy arrays or tensors) viewed by each camera.
                If None, no images will be displayed in the camera frustum view.
                Each image should have shape (H, W, C) for RGB images or (H, W) for grayscale images.
                If a single tensor/array is provided, it should have shape (N, H, W, C) or (N, H, W) where N is the number of cameras.
                The width and height of each image should match the corresponding size in `image_sizes`.
            axis_length (float): The length of the camera axes in the viewer in world units. Defaults to 0.5.
            axis_thickness (float): The thickness (diameter) of the camera axis lines in world units. Defaults to 0.0125.
            frustum_line_width (float): The width of the lines representing the camera frustums in pixel units. Defaults to 2.0.
            frustum_scale (float): The length of the optical axis of the camera frustums in world units
                (_i.e._ the distance from the camera position to the view plane). Defaults to 1.0.
            frustum_color (Sequence[float] | np.ndarray): The RGB color of the camera frustum lines.
            show_images (bool): If True, the camera images will be displayed in the viewer. Defaults to True.
            enabled: (bool): If True, the camera view UI is enabled and the cameras will be rendered.
                If False, the camera view UI is disabled and the cameras will not be rendered. Defaults to True.
        """
        if name in self._camera_frustum_views:
            raise ValueError(f"A camera view with the name '{name}' is already registered.")

        if not isinstance(image_sizes, (np.ndarray, torch.Tensor)):
            raise ValueError("image_sizes must be a numpy array or torch tensor.")
        if isinstance(image_sizes, torch.Tensor):
            image_sizes = image_sizes.cpu().numpy()
        if image_sizes.ndim != 2 or image_sizes.shape[1] != 2:
            raise ValueError("image_sizes must have shape (N, 2) where N is the number of cameras.")

        if image_sizes.shape[0] != len(cam_to_world_matrices) or image_sizes.shape[0] != len(projection_matrices):
            raise ValueError("The number of image sizes must match the number of camera matrices provided.")

        camer_frustum_view = CameraView(
            name=name,
            viewer_handle=self._handle,
            cam_to_world_matrices=cam_to_world_matrices,
            projection_matrices=projection_matrices,
            images=images,
            image_dimensions=image_sizes,
            axis_length=axis_length,
            axis_thickness=axis_thickness,
            frustum_line_width=frustum_line_width,
            frustum_scale=frustum_scale,
            frustum_color=frustum_color,
            show_images=show_images,
            enabled=enabled,
        )
        self._camera_frustum_views[name] = camer_frustum_view
        self._layout_gui()

        return camer_frustum_view

    def register_gaussian_splat_3d(
        self,
        name: str,
        gaussian_scene: GaussianSplat3d,
        eps_2d: float = 0.3,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        antialias: bool = True,
        sh_degree: int = -1,
        render_output_type: Literal["rgb", "depth"] = "rgb",
        enabled: bool = True,
    ) -> GaussianSplat3dView:
        """
        Register a new Gaussian Splat radiance field to be visualized in the viewer.
        The viewer will show a GUI allowing the user to control the rendering of the Gaussian splats,
        such as changing the SH degree, tile size, and other rendering parameters.

        Note: The Gaussian splat scene is rendered in the background by a separate thread.

        Args:
            name (str): The name of the Gaussian splat scene.
            gaussian_scene (fvdb.GaussianSplat3d): The Gaussian splat scene to be registered.
            eps_2d (float): The epsilon value for 2D rendering. Defaults to 0.3.
            tile_size (int): The size of the tiles used for rendering. Defaults to 16.
            min_radius_2d (float): The minimum radius for 2D rendering. Defaults to 0.0.
            antialias (bool): Whether to enable antialiasing. Defaults to False.
            sh_degree (int): The SH degree for the Gaussian splats. -1 means use all available SH bands.
                Must be less than the total available SH bands in the Gaussian splat scene.
                Defaults to -1.
            render_output_type (Literal["rgb", "depth"]): The type of output to render.
                Can be either "rgb" for color images or "depth" for colorized depth images.
            enabled (bool): If True, the Gaussian splat scene is enabled and will be rendered.
                If False, the Gaussian splat scene is disabled and will not be rendered.
        Returns:
            GaussianSplat3dView: A `GaussianSplat3dView`, giving the caller the ability to
                set render parameters for the registered splats from code which are reflected in the GUI.
        """
        gaussian_splat_3d_view = GaussianSplat3dView(
            name=name,
            viewer_handle=self._handle,
            sh_degree=sh_degree,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            gaussian_scene=gaussian_scene,
            render_output_type=render_output_type,
            enable_depth_compositing=False,  # Depth compositing is slow so disable by default
            enabled=enabled,
        )
        with self._lock:
            self._gaussian_splat_3d_views[name] = (self._background_rendering_view_counter, gaussian_splat_3d_view)
            self._background_rendering_view_counter += 1

        self._layout_gui()
        self._notify_render_threads()

        return gaussian_splat_3d_view

    def register_custom_render_view(
        self,
        name: str,
        render_callback: RenderCallBack,
        ui_schema: list[dict[str, float | int | bool | str]] = [],
        enabled: bool = True,
    ) -> CustomRenderView:
        """
        Register a new custom render view to the viewer.

        A custom render view allows you to define a custom rendering callback that will be called
        every frame to render images in the scene as well as define custom UI components to affect
        the rendering process.

        The callback has the form:
        ```python
        def render_callback(
            current_frame: torch.Tensor | None,
            current_depth: torch.Tensor | None,
            world_to_cam_matrix: torch.Tensor,
            projection_matrix: torch.Tensor,
            img_width: int,
            img_height: int,
            near: float,
            far: float,
            camera_model: str,
            ui_state: dict[str, Any],
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ```
        Where `current_frame` and `current_depth` are the current frame and depth image being rendered,
        (you can use these to composite your rendered frame on top of these). If they are `None`, it means
        this is the first render pass.
        The `world_to_cam_matrix` and `projection_matrix` are the camera transformation matrices for the current frame.
        The `img_width` and `img_height` are the dimensions of the image being rendered.
        The `near` and `far` parameters are the near and far clipping planes for the camera.
        The `camera_model` is the camera model being used for rendering (e.g., "perspective", "orthographic").
        The `ui_state` is a dictionary containing the current state of the UI components defined in the `ui_schema`.
        The callback should return a tuple of `(rendered_image, rendered_depth)`, where `rendered_image` is a tensor of shape
        `(H, W, C)` representing the rendered image and `rendered_depth` is a tensor of shape `(H, W)` representing the depth image.
        If you do not want to render a depth image, you can return `None` for the `rendered_depth`.

        UI Components can be defined in the `ui_schema` parameter, which is a sequence of dictionaries.
        Each dictionary should contain the following keys:
        - "type": The type of the UI component (e.g., "slider", "checkbox", "markdown").
        - "name": The name of the UI component.
        - "default_value": The default value for the UI component.

        If the component is a slider, it must also contain:
        - "min": The minimum value for the slider.
        - "max": The maximum value for the slider.
        - "step": The step size for the slider.

        Args:
            name (str): The name of the custom render view.
            render_callback (RenderCallBack): The callback function that will be called every frame to render
                the images in the scene. This function should take the parameters described above and return
                a tuple of `(rendered_image, rendered_depth)`.
            ui_schema (Sequence[dict[str, float | int | bool | str]]): A sequence of dictionaries defining the UI components
                to be displayed in the custom render view. Each dictionary should contain the following keys:
                - "type": The type of the UI component (e.g., "slider", "checkbox", "markdown").
                - "name": The name of the UI component.
                - "default_value": The default value for the UI component.
            enabled (bool): If True, the custom render view is enabled and will be rendered.
                If False, the custom render view is disabled and will not be rendered. Defaults to True
        Returns:
            CustomRenderView: A `CustomRenderView` instance, which can be used to render
                custom images in the viewer and interact with the UI components defined in the `ui_schema`.
        """
        custom_render_view = CustomRenderView(
            name=name, viewer_handle=self._handle, render_callback=render_callback, ui_schema=ui_schema, enabled=enabled
        )

        with self._lock:
            self._gaussian_splat_3d_views[name] = (self._background_rendering_view_counter, custom_render_view)
            self._background_rendering_view_counter += 1

        self._layout_gui()
        self._notify_render_threads()

        return custom_render_view

    def register_point_cloud(self, name, points):
        raise NotImplementedError("Point cloud rendering is not implemented yet. Please use Gaussian splats for now.")

    def register_voxel_grid(self, name, voxel_grid):
        raise NotImplementedError("Voxel grid rendering is not implemented yet. Please use Gaussian splats for now.")

    def register_dictionary_label(self, name: str, label_dict: dict[str, str | int | float] = {}) -> DictLabelView:
        """
        Register a new UI component which logs a set of key-value pairs to the viewer's GUI.

        Args:
            name (str): The name of the dictionary label UI component.
            label_dict (dict[str, str | int | float]): A dictionary containing the information to
                be displayed in the viewer's GUI. The keys must be strings, and the values can
                be strings, integers, or floats.
        Returns:
            DictLabelView: A `DictLabelView`, which can be used to update the
                information displayed in the viewer's GUI from code for the registered dict label.
        """
        for key, value in label_dict.items():
            if not isinstance(key, str):
                raise ValueError(f"Key {key} in label_dict must be a string.")
            if not isinstance(value, (str, int, float)):
                raise ValueError(f"Unsupported type {type(value)} for key {key} in label_dict.")
        dict_label_view = DictLabelView(viewer_handle=self._handle, name=name, label_dict=label_dict)

        self._dict_label_views[name] = dict_label_view

        self._layout_gui()

        return dict_label_view

    @property
    def lock(self) -> threading.Lock:
        """
        Get a lock for the viewer.
        This lock is used to block background Gaussian rendering threads while any Gaussian
        splat they are rendering is being modified, or to prevent over-using memory
        and compute ressources (since the background threads are Python threads which run in the same process).
        """
        return self._lock

    def acquire_lock(self):
        """
        Acquire a lock on the viewer. This will block any background Gaussian rendering threads
        until the lock is released.
        """
        self._lock.acquire()

    def release_lock(self):
        """
        Release the lock on the viewer. This will allow background rendering threads
        to continue rendering.
        """
        self._lock.release()

    def _layout_gui(self):
        """
        Re-layout the entire GUI of the viewer.

        This function is called when the viewer's GUI layout needs to be updated. For example, when a new GUI component
        is added or when the viewer's state changes in a way that affects the GUI layout.

        It resets the GUI and re-adds all the components that are currently registered with the viewer.
        """
        gui: viser.GuiApi = self._viser_server.gui
        gui.reset()

        self._viser_server.gui.configure_theme(dark_mode=True, control_layout="fixed")
        self._global_info_view.layout_gui()

        if len(self._gaussian_splat_3d_views) > 0:
            with gui.add_folder("Gaussian Scenes", visible=True):
                for _, splat_id_and_view in self._gaussian_splat_3d_views.items():
                    splat_id_and_view[1].layout_gui()

        if len(self._camera_frustum_views) > 0:
            with gui.add_folder("Cameras Views", visible=True):
                for _, camera_view in self._camera_frustum_views.items():
                    camera_view.layout_gui()

        if len(self._dict_label_views) > 0:
            for _, dict_label_view in self._dict_label_views.items():
                dict_label_view.layout_gui()

    def _notify_render_threads(self):
        """
        Notify all background threads that render the Gaussian splats
        that they should re-render the scene.
        """
        self._gaussian_client_render_thread_pool.notify_threads()

    def _view_event_handler(self, event: ViewerEvent, value: Any):
        """
        Handle events from the views registered to the viewer.

        Modules use a `ViewerHandle` to send events to the viewer. Those events are handled here.

        Args:
            event (ViewerEvent): The event to handle. This can be an action like notifying Gaussian
                render threads to re-render, re-rendering the GUI, or setting up the camera up direction.
        """
        if event.action == ViewerAction.NOTIFY_GAUSSIAN_THREADS:
            self._gaussian_client_render_thread_pool.notify_threads()
        elif event.action == ViewerAction.RERENDER_GUI:
            self._layout_gui()
        elif event.action == ViewerAction.SET_UP_DIRECTION:
            assert event.gui_event is not None, "GuiEvent must be provided for setting up direction."
            if value not in ["+x", "+y", "+z", "-x", "-y", "-z"]:
                raise ValueError(f"Invalid up direction: {value}. Must be one of '+x', '+y', '+z', '-x', '-y', '-z'.")
            clients = self._viser_server.get_clients()
            for client_id in clients:
                clients[client_id].scene.set_up_direction(value)
            self._notify_render_threads()
        elif event.action == ViewerAction.PAUSE_GAUSSIAN_THREADS:
            self._gaussian_client_render_thread_pool.pause_threads()
        elif event.action == ViewerAction.RESUME_GAUSSIAN_THREADS:
            self._gaussian_client_render_thread_pool.resume_threads()
        elif event.action == ViewerAction.SET_TARGET_PIXELS_PER_FRAME:
            assert isinstance(value, (int, float)), "Target pixels per frame must be a number."
            self._gaussian_client_render_thread_pool.target_pixels_per_frame = int(value)
        elif event.action == ViewerAction.SET_MAX_IMAGE_WIDTH:
            assert event.gui_event is not None, "GuiEvent must be provided for setting up direction."
            assert isinstance(value, (int, float)), "Max image width must be a number."
            self._gaussian_client_render_thread_pool.max_image_width = int(value)
        elif event.action == ViewerAction.SET_CAMERA_FAR:
            assert event.gui_event is not None, "GuiEvent must be provided for setting camera far."
            assert isinstance(value, (int, float)), "Camera far must be a number."
            clients = self._viser_server.get_clients()
            for client_id in clients:
                clients[client_id].camera.far = float(value)
            self._notify_render_threads()
        elif event.action == ViewerAction.SET_CAMERA_NEAR:
            assert event.gui_event is not None, "GuiEvent must be provided for setting camera near."
            assert isinstance(value, (int, float)), "Camera near must be a number."
            clients = self._viser_server.get_clients()
            for client_id in clients:
                clients[client_id].camera.near = float(value)
            self._notify_render_threads()
        else:
            raise ValueError(f"Unknown action: {event.action}. Cannot handle this event.")

    @property
    def camera_far(self) -> float:
        """
        Returns the far plane distance for the camera.
        This is used to define the maximum distance at which objects are rendered.
        """
        return self._global_info_view.camera_far

    @camera_far.setter
    def camera_far(self, camera_far: float):
        """
        Set the far plane distance for the camera.
        This is used to define the maximum distance at which objects are rendered.

        Args:
            camera_far (float): The far plane distance for the camera.
        """
        self._global_info_view.camera_far = camera_far

    @property
    def camera_near(self) -> float:
        """
        Returns the near plane distance for the camera.
        This is used to define the minimum distance at which objects are rendered.
        """
        return self._global_info_view.camera_near

    @camera_near.setter
    def camera_near(self, camera_near: float):
        """
        Set the near plane distance for the camera.
        This is used to define the minimum distance at which objects are rendered.

        Args:
            camera_near (float): The near plane distance for the camera.
        """
        self._global_info_view.camera_near = camera_near

    def _on_client_disconnect(self, client: viser.ClientHandle):
        """
        Callback function which is executed when a client disconnects from the viewer's viser server.

        This function unregisters the background rendering thread for that client.

        Args:
            client (viser.ClientHandle): The client that disconnected.
        """
        self._gaussian_client_render_thread_pool.unregister_client(client)

    def _on_client_connect(self, client: viser.ClientHandle):
        """
        Callback function which is executed when a client connects to the viewer's viser server.

        This function does the following:
          - Sets the camera up direction for that client to the viewer's global up direction.
          - Creates a new background rendering thread to render Gaussian splats for that client.
          - Notifies the new background rendering thread to render its first frame.

        Args:
            client (viser.ClientHandle): The client that connected.
        """
        with self._viser_server.atomic():
            client.scene.set_up_direction(self._global_info_view.camera_up_axis)
            client.camera.near = self._global_info_view.camera_near
            client.camera.far = self._global_info_view.camera_far
        self._gaussian_client_render_thread_pool.register_client(client)
        self._notify_render_threads()
