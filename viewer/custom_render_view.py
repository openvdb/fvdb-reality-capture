# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import copy
import functools
from typing import Callable

import torch
import viser
from torchvision.transforms.functional import resize

from .client_thread_view import ClientThreadRenderingView
from .viewer_handle import ViewerHandle

UIComponent = viser.GuiSliderHandle | viser.GuiCheckboxHandle | viser.GuiMarkdownHandle

UiComponentSchema = dict[str, str | int | float | bool]


class UIComponentView:
    """
    A view containing a set of UI components that can be used to interact with the viewer.
    This view allows you to create and manage UI components such as sliders, checkboxes, and labels.

    The UI components are defined by a schema, which is a list of dictionaries.
    Each dictionary should have the following:
    - "name": The name of the component (string).
    - "type": The type of the component (string, one of "slider", "checkbox", "label").
    - "default_value": The initial value of the component (int, float, bool, or str).
    - For "slider" components, you must also provide "min", "max", and "step" keys:
        - "min": The minimum value of the slider (int or float).
        - "max": The maximum value of the slider (int or float).
        - "step": The step size of the slider (int or float).
    """

    def __init__(self, viewer_handle: ViewerHandle, ui_schema: list[UiComponentSchema], enabled: bool):
        """
        Initializes the UIComponentView with a ViewerHandle and a schema for the UI components.

        Args:
            viewer_handle (ViewerHandle): The handle to the viewer that this UI component view is associated
            ui_schema (list[UiComponentSchema]): A list of dictionaries defining the UI components.
                Each dictionary should have the keys "name", "type", and "default_value".
                For "slider" components, you must also provide "min", "max", and "step" keys.
        """
        self._viewer_handle = viewer_handle
        self._ui_schema: list[UiComponentSchema] = ui_schema

        self._components: dict[str, UIComponent] = {}
        self._component_values: dict[str, str | int | float | bool] = {}
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """
        Returns whether this UI component view is enabled.
        If enabled, the UI components will be interactive and can be used to control the viewer.
        """
        assert self._enabled == all(
            component.disabled is False
            for component in self._components.values()
            if not isinstance(component, viser.GuiMarkdownHandle)
        )
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """
        Set the enabled state of this UI component view.
        If enabled, the UI components will be interactive and can be used to control the viewer.

        Args:
            value (bool): The new enabled state for this UI component view.
        """
        for component in self._components.values():
            if not isinstance(component, viser.GuiMarkdownHandle):
                component.disabled = not value
        self._enabled = value
        self._viewer_handle.notify_render_threads()

    def layout_gui(self):
        """
        Function to layout the GUI components defined in the schema.
        This will create the UI components in the viewer's GUI based on the schema provided.
        It will raise an error if the schema is invalid or if any component is not properly defined.
        """
        gui: viser.GuiApi = self._viewer_handle.gui

        self._enabled_checkbox = gui.add_checkbox(
            label="Enabled",
            initial_value=self._enabled,
        )
        self._enabled_checkbox.on_update(self._enabled_update)

        for component_schma in self._ui_schema:
            if "type" not in component_schma:
                raise ValueError(f"Component schema must have 'type' key, got {component_schma}")
            if "name" not in component_schma:
                raise ValueError(f"Component schema must have 'name' key, got {component_schma}")
            if "default_value" not in component_schma:
                raise ValueError(f"Component schema must have 'default_value' key, got {component_schma}")

            if not isinstance(component_schma["name"], str):
                raise TypeError(f"Component name must be a string, got name = {type(component_schma['name'])}")

            if not isinstance(component_schma["type"], str):
                raise TypeError(
                    f"Component type must be a string, got type = {type(component_schma['type'])} for name = {component_schma['name']}"
                )
            if not isinstance(component_schma["default_value"], (int, float, bool, str)):
                raise TypeError(
                    f"Component default_value must be a number, boolean, or string, got "
                    f"{type(component_schma['default_value'])} for component '{component_schma['name']}'"
                )

            comp_type: str = component_schma["type"]
            comp_name: str = component_schma["name"]
            if comp_name == "Enabled":
                raise ValueError("Component name 'Enabled' is reserved for the enabled checkbox and cannot be used.")

            comp_value: str | int | float | bool = component_schma["default_value"]
            if comp_name in self._component_values:
                comp_value = self._component_values[comp_name]

            if comp_type == "slider":
                if "min" not in component_schma or "max" not in component_schma or "step" not in component_schma:
                    raise ValueError(f"Slider component must have 'min', 'max', and 'step' keys, got {component_schma}")
                if not isinstance(component_schma["min"], (int, float)):
                    raise TypeError(f"Slider 'min' must be a number, got {type(component_schma['min'])}")
                if not isinstance(component_schma["max"], (int, float)):
                    raise TypeError(f"Slider 'max' must be a number, got {type(component_schma['max'])}")
                if not isinstance(component_schma["step"], (int, float)):
                    raise TypeError(f"Slider 'step' must be a number, got {type(component_schma['step'])}")

                if not isinstance(comp_value, (int, float)):
                    raise TypeError(
                        f"Slider initial value must be a number, got {type(comp_value)} for component '{comp_name}'"
                    )
                gui_slider = gui.add_slider(
                    label=comp_name,
                    min=component_schma["min"],
                    max=component_schma["max"],
                    step=component_schma["step"],
                    initial_value=comp_value,
                )
                gui_slider.on_update(functools.partial(self._on_update, comp_name))
                self._components[comp_name] = gui_slider
            elif comp_type == "checkbox":
                if not isinstance(comp_value, bool):
                    raise TypeError(
                        f"Checkbox initial value must be a boolean, got {type(comp_value)} for component '{comp_name}'"
                    )
                gui_checkbox = gui.add_checkbox(label=comp_name, initial_value=comp_value)
                gui_checkbox.on_update(functools.partial(self._on_update, comp_name))
                self._components[comp_name] = gui_checkbox
            elif comp_type == "label":
                if not isinstance(comp_value, str):
                    raise TypeError(
                        f"Label initial value must be a string, got {type(comp_value)} for component '{comp_name}'"
                    )
                gui_label = gui.add_markdown(content=f"**{comp_name}**: {comp_value}")
                self._components[comp_name] = gui_label
            else:
                raise ValueError(
                    f"Unsupported component type '{comp_type}' for component '{comp_name}'. Must be one of 'slider', 'checkbox', 'label'."
                )
            self._component_values[comp_name] = comp_value

    def _enabled_update(self, gui_event: viser.GuiEvent):
        """
        Callback function for when the enabled checkbox is updated.
        This will enable or disable all components based on the checkbox state.

        Args:
            gui_event (viser.GuiEvent): The event triggered by the checkbox update.
        """
        assert isinstance(gui_event.target, viser.GuiCheckboxHandle), "Expected GuiCheckboxHandle for enabled checkbox"
        self.enabled = gui_event.target.value
        self._viewer_handle.notify_render_threads()

    def _on_update(self, name: str, gui_event: viser.GuiEvent):
        """
        Internal method to handle updates to the component values.
        This is called when a GUI component is updated.

        Args:
            name (str): The name of the component that was updated.
            gui_event (viser.GuiEvent): The GUI event that triggered the update.
        """
        if name not in self._components:
            raise ValueError(f"Component with name '{name}' not found in GUI components")
        component = self._components[name]
        if isinstance(component, viser.GuiSliderHandle):
            assert isinstance(gui_event.target, viser.GuiSliderHandle), "Expected GuiSliderHandle for slider component"
            self._component_values[name] = gui_event.target.value
        elif isinstance(component, viser.GuiCheckboxHandle):
            assert isinstance(
                gui_event.target, viser.GuiCheckboxHandle
            ), "Expected GuiCheckboxHandle for checkbox component"
            self._component_values[name] = gui_event.target.value
        self._viewer_handle.notify_render_threads()

    def __getitem__(self, name: str) -> str | int | float | bool:
        """
        Get the value of a GUI component by its name.

        Args:
            name (str): The name of the component to retrieve.

        Returns:
            str | int | float | bool: The current value of the component.
        """
        if name not in self._components:
            raise ValueError(f"Component with name '{name}' not found in GUI components")
        handle = self._components[name]
        if isinstance(handle, viser.GuiSliderHandle):
            assert handle.value == self._component_values[name]
            return handle.value
        elif isinstance(handle, viser.GuiCheckboxHandle):
            assert handle.value == self._component_values[name]
            return handle.value
        elif isinstance(handle, viser.GuiMarkdownHandle):
            assert handle.content == self._component_values[name]
            return handle.content
        else:
            raise TypeError(f"Unsupported GUI handle type: {type(handle)} for component '{name}'")

    def __setitem__(self, name: str, value: str | int | float | bool):
        """
        Set the value of a GUI component by its name.

        Args:
            name (str): The name of the component to set.
            value (str | int | float | bool): The value to set for the component.
        """
        if name not in self._components:
            raise ValueError(f"Component with name '{name}' not found in GUI components")
        handle = self._components[name]
        if isinstance(handle, viser.GuiSliderHandle):
            assert isinstance(value, (int, float)), "Slider value must be a number"
            self._component_values[name] = value
            handle.value = value
        elif isinstance(handle, viser.GuiCheckboxHandle):
            assert isinstance(value, bool), "Checkbox value must be a boolean"
            self._component_values[name] = value
            handle.value = value
        elif isinstance(handle, viser.GuiMarkdownHandle):
            assert isinstance(value, str), "Markdown content must be a string"
            self._component_values[name] = value
            handle.content = value
        else:
            raise TypeError(f"Unsupported GUI handle type: {type(handle)} for component '{name}'")

        self._viewer_handle.notify_render_threads()

    def __contains__(self, name: str) -> bool:
        """
        Check if a component with the given name exists in the UI components.

        Args:
            name (str): The name of the component to check.
        """
        return name in self._components

    def items(self):
        """
        Return the component names and values as an iterable of tuples.
        Each tuple contains the component name and its current value.

        Yields:
            tuple[str, str | int | float | bool]: A tuple containing the component name and its current value.
        """
        for name, handle in self._components.items():
            if isinstance(handle, viser.GuiSliderHandle):
                yield name, handle.value
            elif isinstance(handle, viser.GuiCheckboxHandle):
                yield name, handle.value
            elif isinstance(handle, viser.GuiMarkdownHandle):
                yield name, handle.content
            else:
                raise TypeError(f"Unsupported GUI handle type: {type(handle)} for component '{name}'")

    def keys(self):
        """
        Return the component names as an iterable.

        Returns:
            Iterable[str]: An iterable containing the names of the UI components.
        """
        return self._components.keys()

    def values(self):
        """
        Return the component values as an iterable.

        Yields:
            str | int | float | bool: The current value of each UI component.
        """
        for handle in self._components.values():
            if isinstance(handle, viser.GuiSliderHandle):
                yield handle.value
            elif isinstance(handle, viser.GuiCheckboxHandle):
                yield handle.value
            elif isinstance(handle, viser.GuiMarkdownHandle):
                yield handle.content
            else:
                raise TypeError(f"Unsupported GUI handle type: {type(handle)}")

    def state(self) -> dict[str, str | int | float | bool]:
        """
        Returns a copy of the the current state of the UI components as a dictionary.
        The keys are the component names and the values are their current values.

        Note: Modifying the values will not update the UI components.

        Returns:
            dict[str, str | int | float | bool]: A dictionary containing the current state of the UI components.
        """
        return copy.deepcopy(self._component_values)

    def __iter__(self):
        return iter(self._components)


RenderCallBack = Callable[
    [
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        float,
        float,
        str,
        dict[str, str | int | float | bool],
    ],
    tuple[torch.Tensor | None, torch.Tensor | None],
]


class CustomRenderView(ClientThreadRenderingView):
    def __init__(
        self,
        name: str,
        viewer_handle: ViewerHandle,
        render_callback: RenderCallBack,
        ui_schema: list[dict[str, str | int | float | bool]],
        enabled: bool,
    ):
        """
        Initializes a custom render view with a name, viewer handle, render callback, and UI schema.

        The render callback is a function that will be called to render the frame with the given camera and light settings.

        The UI schema defines the UI components that will be available in this custom render view (See `UIComponentView` for details).

        Args:
            name (str): The name of the custom render view.
            viewer_handle (ViewerHandle): The handle to the viewer that this custom render view is associated
            render_callback (RenderCallBack): A callback function that will be called to render the frame.
                The callback should accept the following parameters:
                - current_frame (torch.Tensor | None): The last rendered frame, or None if no frame has been rendered yet.
                - current_depth (torch.Tensor | None): The last rendered depth map, or None if no depth has been rendered yet.
                - world_to_cam_matrix (torch.Tensor): The transformation matrix from world coordinates to camera coordinates.
                - projection_matrix (torch.Tensor): The projection matrix for the camera.
                - img_width (int): The width of the image to render.
                - img_height (int): The height of the image to render.
                - near (float): The near clipping plane distance.
                - far (float): The far clipping plane distance.
                - camera_model (str): The camera model being used.
                - ui_state (dict[str, str | int | float | bool]): The current state of the UI components.
            ui_schema (list[dict[str, str | int | float | bool]]): A list of dictionaries defining the UI components.
                Each dictionary should have the keys "name", "type", and "default_value".
                For "slider" components, you must also provide "min", "max", and "step" keys.
            enabled (bool): Whether this custom render view is enabled by default.
        """
        self._name = name
        self._viewer_handle = viewer_handle
        self._render_callback = render_callback
        self._ui_components = UIComponentView(viewer_handle, ui_schema, enabled)
        self._last_rendered_image = None

    @property
    def name(self) -> str:
        """
        Returns the name of the custom render view.

        Returns:
            str: The name of the custom render view.
        """
        return self._name

    @property
    def ui(self) -> UIComponentView:
        """
        Returns the UI components for this custom render view.

        Returns:
            UIComponentView: The UI components for this custom render view.
        """
        return self._ui_components

    def layout_gui(self):
        """
        Layout the GUI components for this custom render view.
        This will create the UI components defined in the schema.
        """
        gui: viser.GuiApi = self._viewer_handle.gui

        with gui.add_folder(self._name, visible=True):
            self._ui_components.layout_gui()

    def _render(
        self,
        current_frame: torch.Tensor | None,
        current_depth: torch.Tensor | None,
        world_to_cam_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
        img_width: int,
        img_height: int,
        near: float,
        far: float,
        camera_model: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Internal method to render the frame with the given camera and light settings.

        This method calls the render callback function with the current frame, depth, camera matrices,
        image dimensions, near and far clipping planes, camera model, and the current UI state.

        Args:
            current_frame (torch.Tensor | None): The last rendered frame, or None if no frame has been rendered yet.
            current_depth (torch.Tensor | None): The last rendered depth map, or None if no depth has been rendered yet.
            world_to_cam_matrix (torch.Tensor): The transformation matrix from world coordinates to camera coordinates.
            projection_matrix (torch.Tensor): The projection matrix for the camera.
            img_width (int): The width of the image to render.
            img_height (int): The height of the image to render.
            near (float): The near clipping plane distance.
            far (float): The far clipping plane distance.
            camera_model (str): The camera model being used.

        Returns:
            tuple[torch.Tensor | None, torch.Tensor | None]: A tuple containing the rendered image and depth map.
            If the render callback returns None for either, it indicates that no rendering was performed.
        """
        if not self._ui_components.enabled:
            if self._last_rendered_image is not None:
                # If the UI is disabled, return the last rendered image at half resolution
                last_img = self._last_rendered_image
                if self._last_rendered_image.shape[1] != img_width or self._last_rendered_image.shape[0] != img_height:
                    # Resize the last rendered image to the current dimensions
                    last_img = resize(self._last_rendered_image.permute(2, 0, 1), [img_height, img_width]).permute(
                        1, 2, 0
                    )
                return last_img * 0.5, None
            return None, None

        self._last_rendered_image, ret_depth = self._render_callback(
            current_frame,
            current_depth,
            world_to_cam_matrix,
            projection_matrix,
            img_width,
            img_height,
            near,
            far,
            camera_model,
            self._ui_components.state(),
        )

        return self._last_rendered_image, ret_depth
