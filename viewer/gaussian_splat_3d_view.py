# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Literal

import torch
import viser

from fvdb import GaussianSplat3d

from .viewer_handle import ViewerHandle


class GaussianSplat3dView:
    """
    A view that tracks a single `GaussianSplat3d` instance in the viewer.

    It allows the user to control various Gaussian rendering parameters such as spherical harmonics degree,
    tile size, etc. from code and have those changes reflected in the viewer GUI.

    Attributes:
        name (str): The name of the scene, used as the header title in the GUI.
        sh_degree (int): The spherical harmonics degree used to render this Gaussian scene.
            A value of -1 means we will use all available spherical harmonics.
        tile_size (int): The tile size for rendering this Gaussian scene.
        min_radius_2d (float): The minimum projected pixel radius below which Gaussians will not be rendered.
        eps_2d (float): The 2D epsilon value for this Gaussian scene.
        antialias (bool): Whether to enable antialiasing for this Gaussian scene.
        max_image_width (int): The maximum image width for rendering this Gaussian scene in the viewer.
        target_pixels_per_frame (float): The target number of pixels to render per frame for this Gaussian scene.
            The viewer will dynamically adjust the rendering resolution to achieve this target.
        target_framerate (float): The target framerate for rendering this Gaussian scene.
            The viewer will dynamically adjust the rendering resolution to achieve this target.
        gaussian_scene (GaussianSplat3d): The `GaussianSplat3d` instance to be tracked by this view.
            This is the scene that will be rendered in the viewer.
        render_output_type (Literal["rgb", "depth"]): The type of output to render.
        enable_depth_compositing (bool): Whether to enable depth compositing for the rendered output.
            Depth compositing uses the gaussian depths to blend with other objects in the scene, but is slow to render.
    Note:
        You should not create this view directly. Instead, use the viewer's
        `register_gaussian_splat_3d` method to request one from the viewer.
    """

    def __init__(
        self,
        name: str,
        viewer_handle: ViewerHandle,
        sh_degree: int,  # -1 means no spherical harmonics
        tile_size: int,
        min_radius_2d: float,
        eps_2d: float,
        antialias: bool,
        max_image_width: int,
        target_pixels_per_frame: float,
        target_framerate: float,
        gaussian_scene: GaussianSplat3d,
        render_output_type: Literal["rgb", "depth"],
        enable_depth_compositing: bool,
        enabled: bool,
    ):
        """
        Create a new `GaussianSplat3dView` associated with a specific GaussianSplat3d instance.

        Note: You should not create this view directly. Instead, use the viewer's
        `register_gaussian_splat_scene` method.

        Args:
            name (str): The name of the scene, used as the header title in the GUI.
            viewer_handle (ViewerHandle): The handle to the viewer.
            sh_degree (int): The spherical harmonics degree used to render this Gaussian scene.
                A value of -1 means we will use all available spherical harmonics.
            tile_size (int): The tile size for rendering this Gaussian scene.
            min_radius_2d (float): The minimum projected pixel radius below which Gaussians will not be rendered.
            eps_2d (float): The 2D epsilon value for this Gaussian scene.
            antialias (bool): Whether to enable antialiasing for this Gaussian scene.
            max_image_width (int): The maximum image width for rendering this Gaussian scene in the viewer.
            target_pixels_per_frame (float): The target number of pixels to render per frame for this Gaussian scene.
                The viewer will dynamically adjust the rendering resolution to achieve this target.
            target_framerate (float): The target framerate for rendering this Gaussian scene.
                The viewer will dynamically adjust the rendering resolution to achieve this target.
            gaussian_scene (GaussianSplat3d): The `GaussianSplat3d` instance to be tracked by this view.
                This is the scene that will be rendered in the viewer.
            render_output_type (Literal["rgb", "depth"]): The type of output to render.
                Defaults to "rgb".
            enable_depth_compositing (bool): Whether to enable depth compositing for the rendered output.
                This will make the viewer play more nicely with other geometry but is slow

        """
        self._name: str = name
        self._viewer_handle: ViewerHandle = viewer_handle
        self._sh_degree: int = sh_degree  # -1 means no spherical harmonics
        self._tile_size: int = tile_size
        self._min_radius_2d: float = min_radius_2d
        self._eps_2d: float = eps_2d
        self._antialias: bool = antialias
        self._max_image_width: int = max_image_width
        self._target_pixels_per_frame: float = target_pixels_per_frame
        self._target_framerate: float = target_framerate
        self._gaussian_scene: GaussianSplat3d = gaussian_scene
        if render_output_type not in ["rgb", "depth"]:
            raise ValueError(f"Invalid render output type: {render_output_type}")
        self._render_output_type: Literal["rgb", "depth"] = render_output_type
        self._enable_depth_compositing: bool = enable_depth_compositing
        self._enabled = enabled
        self._allow_enable_in_gui = True

    def layout_gui(self):
        gui = self._viewer_handle.gui
        with gui.add_folder(self._name, visible=True) as self._name_gui_handle:
            self._enabled_gui_handle = gui.add_checkbox(
                "Enabled", self._enabled, disabled=not self._allow_enable_in_gui
            )
            disabled = not self._enabled
            self._eps_2d_gui_handle = gui.add_number("Eps 2d", self._eps_2d, 0.01, 0.5, 0.05, disabled=disabled)
            self._tile_size_gui_handle = gui.add_number("Tile Size", self._tile_size, 1, 64, 1, disabled=disabled)
            self._antialias_gui_handle = gui.add_checkbox("Antialias", self._antialias, disabled=disabled)
            self._sh_degree_gui_handle = gui.add_slider(
                "SH Degree",
                min=-1,
                max=self._gaussian_scene.sh_degree,
                step=1,
                initial_value=self._sh_degree,
                disabled=disabled,
            )
            self._min_radius_2d_gui_handle = gui.add_number(
                "Min Projected Pixel Radius", self._min_radius_2d, 0.0, 3.0, 0.1, disabled=disabled
            )
            self._max_image_width_gui_handle = gui.add_slider(
                "Max Image Width", min=64, max=2048, step=1, initial_value=2048, disabled=disabled
            )
            self._render_output_type_gui_handle = gui.add_dropdown(
                "Render Output Type", ["rgb", "depth"], initial_value="rgb", disabled=disabled
            )
            self._depth_compositing_update_handle = gui.add_checkbox(
                "Enable Depth Compositing", self._enable_depth_compositing, disabled=disabled
            )

        self._enabled_gui_handle.on_update(self._enabled_update)
        self._eps_2d_gui_handle.on_update(self._eps2d_update)
        self._tile_size_gui_handle.on_update(self._tile_size_update)
        self._antialias_gui_handle.on_update(self._antialias_update)
        self._min_radius_2d_gui_handle.on_update(self._min_radius_2d_update)
        self._max_image_width_gui_handle.on_update(self._max_image_width_update)
        self._render_output_type_gui_handle.on_update(self._render_output_type_update)
        self._depth_compositing_update_handle.on_update(self._depth_compositing_update)
        self._sh_degree_gui_handle.on_update(self._sh_degree_update)

    @property
    def allow_enable_in_viewer(self) -> bool:
        """
        Return whether the user can enable/disable this component in the viewer GUI.

        Returns:
            bool: True if the user can enable/disable this component in the viewer GUI, False otherwise.
        """
        return self._allow_enable_in_gui

    @allow_enable_in_viewer.setter
    def allow_enable_in_viewer(self, value: bool):
        """
        Set whether the user can enable/disable this component in the viewer GUI.

        Args:
            value (bool): True to allow the user to enable/disable this component in the viewer GUI, False otherwise.
        """
        self._allow_enable_in_gui = value
        self._enabled_gui_handle.disabled = not value

    @property
    def enabled(self) -> bool:
        """
        Returns whether this GaussianSplat3dView is enabled.

        Returns:
            bool: True if the view is enabled, False otherwise.
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """
        Sets whether this GaussianSplat3dView is enabled and updates the viewer.

        Args:
            value (bool): True to enable the view, False to disable it.
        """
        if not isinstance(value, bool):
            raise TypeError("enabled must be a boolean value.")
        self._enabled = value
        self._enabled_gui_handle.value = value
        if not value:
            self._viewer_handle.pause_gaussian_render_threads()
        else:
            self._viewer_handle.resume_gaussian_render_threads()

    @property
    def viewer_handle(self) -> ViewerHandle:
        """
        Returns the ViewerHandle associated with this GaussianSplat3dView.

        Returns:
            ViewerHandle: The viewer handle for this view.
        """
        return self._viewer_handle

    @property
    def render_output_type(self) -> Literal["rgb", "depth"]:
        """
        Returns the render output type for this GaussianSplat3dView.

        Returns:
            Literal["rgb", "depth"]: The render output type for this view.
        """
        return self._render_output_type

    @render_output_type.setter
    def render_output_type(self, value: Literal["rgb", "depth"]):
        """
        Sets the render output type for this GaussianSplat3dView and updates the viewer.

        Args:
            value (Literal["rgb", "depth"]): The new render output type for this view.

        """
        if value not in ["rgb", "depth"]:
            raise ValueError(f"Invalid render output type: {value}")
        self._render_output_type = value
        self._render_output_type_gui_handle.value = value
        self._viewer_handle.notify_gaussian_render_threads()

    @property
    def enable_depth_compositing(self) -> bool:
        """
        Returns whether depth compositing is enabled for this GaussianSplat3dView.

        Depth compositing uses the gaussian depths to blend with other objects in the scene.
        It generally causes the viewer to render more slowly, but it allows the viewer to play
        more nicely with other geometry.

        Returns:
            bool: True if depth compositing is enabled, False otherwise.
        """
        return self._enable_depth_compositing

    @enable_depth_compositing.setter
    def enable_depth_compositing(self, value: bool):
        """
        Sets whether depth compositing is enabled for this GaussianSplat3dView.

        Depth compositing uses the gaussian depths to blend with other objects in the scene.
        It generally causes the viewer to render more slowly, but it allows the viewer to play
        more nicely with other geometry.

        Args:
            value (bool): True to enable depth compositing, False to disable it.
        """
        if not isinstance(value, bool):
            raise TypeError("enable_depth_compositing must be a boolean value.")
        self._enable_depth_compositing = value
        self._depth_compositing_update_handle.value = value
        self._viewer_handle.notify_gaussian_render_threads()

    @property
    def name(self) -> str:
        """
        Returns the name of the scene associated with this GaussianSplat3dView.

        Returns:
            str: The name of the scene.
        """
        return self._name

    @property
    def sh_degree(self) -> int:
        """
        Returns the spherical harmonics degree used to render this Gaussian scene.

        Note: A value of -1 means we will use all available spherical harmonics.

        Returns:
            int: The spherical harmonics degree. -1 means all available spherical harmonics will be used.
        """
        return self._sh_degree

    @sh_degree.setter
    def sh_degree(self, value: int):
        """
        Sets the spherical harmonics degree for this Gaussian scene and update the UI.

        Note: A value of -1 means we will use all available spherical harmonics.

        Args:
            value (int): The new spherical harmonics degree. -1 means all available spherical harmonics will be used.
        """
        if value > self._gaussian_scene.sh_degree:
            raise ValueError(
                f"Cannot set spherical harmonics degree to {value} because the scene only has spherical harmonics of degree {self._gaussian_scene.sh_degree}."
            )
        self._sh_degree = value
        self._sh_degree_gui_handle.value = value
        self._viewer_handle.notify_gaussian_render_threads()

    @property
    def tile_size(self) -> int:
        """
        Returns the tile size for rendering this Gaussian scene.

        Returns:
            int: The tile size for rendering this Gaussian scene.
        """
        return self._tile_size

    @tile_size.setter
    def tile_size(self, value: int):
        """
        Sets the tile size for rendering this Gaussian scene and update the UI.

        Args:
            value (int): The new tile size for rendering this Gaussian scene.
        """
        if not isinstance(value, int):
            raise TypeError("tile_size must be an integer.")
        if value <= 0:
            raise ValueError("tile_size must be greater than 0.")
        if value > 64:
            raise ValueError("tile_size cannot be greater than 64 pixels.")
        self._tile_size = value
        self._tile_size_gui_handle.value = value
        self._viewer_handle.notify_gaussian_render_threads()

    @property
    def min_radius_2d(self) -> float:
        """
        Returns the minimum projected pixel radius below which Gaussians will not be rendered.

        Returns:
            float: The minimum projected pixel radius below which Gaussians will not be rendered.
        """
        return self._min_radius_2d

    @min_radius_2d.setter
    def min_radius_2d(self, value: float):
        """
        Sets the minimum projected pixel radius below which Gaussians will not be rendered and update the UI.

        Args:
            value (float): The new minimum projected pixel radius below which Gaussians will not be rendered.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("min_radius_2d must be a number.")
        if value < 0:
            raise ValueError("min_radius_2d must be greater than or equal to 0.")
        self._min_radius_2d = value
        self._min_radius_2d_gui_handle.value = value
        self._viewer_handle.notify_gaussian_render_threads()

    @property
    def eps_2d(self) -> float:
        """
        Returns the 2D epsilon value for this Gaussian scene.

        Returns:
            float: The 2D epsilon value.
        """
        return self._eps_2d

    @eps_2d.setter
    def eps_2d(self, value: float):
        """
        Sets the 2D epsilon value for this Gaussian scene and update the UI.

        Args:
            value (float): The new 2D epsilon value.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("eps_2d must be a number.")
        if value <= 0:
            raise ValueError("eps_2d must be greater than 0.")
        self._eps_2d = value
        self._eps_2d_gui_handle.value = value
        self._viewer_handle.notify_gaussian_render_threads()

    @property
    def antialias(self) -> bool:
        """
        Returns whether antialiasing is enabled for this Gaussian scene.

        Returns:
            bool: True if antialiasing is enabled, False otherwise.
        """
        return self._antialias

    @antialias.setter
    def antialias(self, value: bool):
        """
        Sets whether antialiasing is enabled for this Gaussian scene and update the UI.
        Args:
            value (bool): True to enable antialiasing, False to disable it.
        """
        if not isinstance(value, bool):
            raise TypeError("antialias must be a boolean value.")
        self._antialias = value
        self._antialias_gui_handle.value = value
        self._viewer_handle.notify_gaussian_render_threads()

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the Gaussian scene is located.

        Returns:
            torch.device: The device of the Gaussian scene.
        """
        return self._gaussian_scene.device

    @property
    def max_image_width(self) -> int:
        """
        Returns the maximum image width for rendering this Gaussian scene in the viewer.

        Returns:
            int: The maximum image width.
        """
        return self._max_image_width

    @max_image_width.setter
    def max_image_width(self, value: int):
        """
        Sets the maximum image width for rendering this Gaussian scene in the viewer and updates the UI.

        Args:
            value (int): The new maximum image width.
        """
        if not isinstance(value, int):
            raise TypeError("max_image_width must be an integer.")
        if value <= 0:
            raise ValueError("max_image_width must be greater than 0.")
        if value < 64:
            raise ValueError("max_image_width cannot be less than 64 pixels.")
        if value > 8192:
            raise ValueError("max_image_width cannot be greater than 8192 pixels.")
        self._max_image_width = value
        self._max_image_width_gui_handle.value = value
        self._viewer_handle.notify_gaussian_render_threads()

    @property
    def target_pixels_per_frame(self) -> float:
        """
        Returns the target number of pixels to render per frame for this Gaussian scene.
        The viewer will dynamically adjust the rendering resolution to achieve this target.

        Returns:
            float: The target number of pixels to render per frame.
        """
        return self._target_pixels_per_frame

    @target_pixels_per_frame.setter
    def target_pixels_per_frame(self, value: float):
        """
        Sets the target number of pixels to render per frame for this Gaussian scene and updates the UI.
        The viewer will dynamically adjust the rendering resolution to achieve this target.

        Args:
            value (float): The new target number of pixels to render per frame.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("target_pixels_per_frame must be a number.")
        if value <= 0:
            raise ValueError("target_pixels_per_frame must be greater than 0.")
        self._target_pixels_per_frame = value
        self._viewer_handle.notify_gaussian_render_threads()

    @property
    def target_framerate(self) -> float:
        """
        Returns the target framerate for rendering this Gaussian scene.
        The viewer will dynamically adjust the rendering resolution to achieve this target framerate.

        Returns:
            float: The target framerate.
        """
        return self._target_framerate

    @target_framerate.setter
    def target_framerate(self, value: float):
        """
        Sets the target framerate for rendering this Gaussian scene and updates the UI.
        The viewer will dynamically adjust the rendering resolution to achieve this target framerate.

        Args:
            value (float): The new target framerate.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("target_framerate must be a number.")
        if value <= 0:
            raise ValueError("target_framerate must be greater than 0.")
        if value > 60:
            raise ValueError("target_framerate cannot be greater than 60 FPS.")
        self._target_framerate = value
        self._viewer_handle.notify_gaussian_render_threads()

    @property
    def gaussian_scene(self) -> GaussianSplat3d:
        """
        Returns the GaussianSplat3d instance associated with this view.

        Returns:
            GaussianSplat3d: The Gaussian scene.
        """
        return self._gaussian_scene

    @gaussian_scene.setter
    def gaussian_scene(self, value: GaussianSplat3d):
        """
        Sets the GaussianSplat3d instance for this view and updates the viewer.

        Args:
            value (GaussianSplat3d): The new Gaussian scene.
        """
        if not isinstance(value, GaussianSplat3d):
            raise TypeError("gaussian_scene must be an instance of GaussianSplat3d.")
        self._gaussian_scene = value
        self._viewer_handle.rerender_gui()
        self._viewer_handle.notify_gaussian_render_threads()

    def _eps2d_update(self, event: viser.GuiEvent):
        """
        Callback function for when the epsilon 2D slider is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the slider update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiNumberHandle)
        self._eps_2d = target_handle.value
        self._viewer_handle.notify_gaussian_render_threads()

    def _tile_size_update(self, event: viser.GuiEvent):
        """
        Callback function for when the tile size slider is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the slider update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiNumberHandle)
        self._tile_size = int(target_handle.value)
        self._viewer_handle.notify_gaussian_render_threads()

    def _antialias_update(self, event: viser.GuiEvent):
        """
        Callback function for when the antialias checkbox is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the checkbox update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiCheckboxHandle)
        self._antialias = target_handle.value
        self._viewer_handle.notify_gaussian_render_threads()

    def _min_radius_2d_update(self, event: viser.GuiEvent):
        """
        Callback function for when the min radius 2D slider is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the slider update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiNumberHandle)
        self._min_radius_2d = target_handle.value
        self._viewer_handle.notify_gaussian_render_threads()

    def _max_image_width_update(self, event: viser.GuiEvent):
        """
        Callback function for when the max image width slider is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the slider update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiSliderHandle)
        self._max_image_width = int(target_handle.value)
        self._viewer_handle.notify_gaussian_render_threads()

    def _render_output_type_update(self, event: viser.GuiEvent):
        """
        Callback function for when the render output type dropdown is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the dropdown update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiDropdownHandle)
        if target_handle.value not in ["rgb", "depth"]:
            raise ValueError(f"Invalid render output type: {target_handle.value}")
        value = target_handle.value
        self._render_output_type = value
        self._viewer_handle.notify_gaussian_render_threads()

    def _depth_compositing_update(self, event: viser.GuiEvent):
        """
        Callback function for when the depth compositing checkbox is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the checkbox update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiCheckboxHandle)
        self._enable_depth_compositing = target_handle.value
        self._viewer_handle.notify_gaussian_render_threads()

    def _sh_degree_update(self, event: viser.GuiEvent):
        """
        Callback function for when the SH degree slider is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the slider update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiSliderHandle)
        value = int(target_handle.value)
        if value > self._gaussian_scene.sh_degree:
            raise ValueError(
                f"Cannot set spherical harmonics degree to {value} because the scene only has spherical harmonics of degree {self._gaussian_scene.sh_degree}."
            )
        self._sh_degree = value
        self._viewer_handle.notify_gaussian_render_threads()

    def _enabled_update(self, event: viser.GuiEvent):
        """
        Callback function for when the enabled checkbox is updated.

        Args:
            event (viser.GuiEvent): The event triggered by the checkbox update.
        """
        target_handle = event.target
        assert isinstance(target_handle, viser.GuiCheckboxHandle)
        self._enabled = target_handle.value
        disabled = not self._enabled
        self._eps_2d_gui_handle.disabled = disabled
        self._tile_size_gui_handle.disabled = disabled
        self._antialias_gui_handle.disabled = disabled
        self._min_radius_2d_gui_handle.disabled = disabled
        self._max_image_width_gui_handle.disabled = disabled
        self._render_output_type_gui_handle.disabled = disabled
        self._depth_compositing_update_handle.disabled = disabled
        self._sh_degree_gui_handle.disabled = disabled
        if not self._enabled:
            self._viewer_handle.pause_gaussian_render_threads()
        else:
            self._viewer_handle.resume_gaussian_render_threads()
