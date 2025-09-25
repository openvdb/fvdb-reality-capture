# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import viser

from .viewer_handle import ViewerHandle

InfoDict = dict[str, str | int | float]


class DictLabelView:
    """
    A view for displaying a set of key-value data in the fVDB viewer's GUI.

    Supports dictionary-like operations such as getting, setting, deleting items,
    and iterating over keys, values, and items. The view automatically updates the GUI
    when the internal dictionary is modified.

    Attributes:
        name (str): The name of the view displayed as the header in the GUI.
        label_dict (dict[str, str | int | float]): The internal dictionary containing the
            key-value pairs to be displayed. Assigning this updates the GUI automatically.

    Note:
        You should not create this view directly. Instead, use the viewer's
        `register_dict_label` method to create and manage instances of this view.
    """

    def __init__(self, name: str, viewer_handle: ViewerHandle, label_dict: InfoDict):
        """
        Create a new `DictLabelView` for displaying a set of key-value data in the fVDB viewer's GUI.

        Supports dictionary-like operations such as getting, setting, deleting items,
        and iterating over keys, values, and items. The view automatically updates the GUI
        when the internal dictionary is modified.

        Note: You should not create this view directly. Instead, use the viewer's
        `register_dict_label` method.

        Args:
            name (str): The name of the view, used as the header title in the GUI.
            viewer_handle (ViewerHandle): The handle to the viewer.
            label_dict (InfoDict): A dictionary containing the information to be displayed in the view.

        """

        self._viewer_handle: ViewerHandle = viewer_handle
        self._name: str = name
        self._label_dict: InfoDict = label_dict

    def __getitem__(self, key: str) -> str | int | float:
        """
        Gets the value of an item from the label_dict.

        Args:
            key (str): The key of the item to get.
        """
        if key not in self._label_dict:
            raise KeyError(f"Key {key} not found in label_dict.")
        return self._label_dict[key]

    def __setitem__(self, key: str, value: str | int | float):
        """
        Sets the value of an item in the label_dict and updates the GUI.

        Args:
            key (str): The key of the item to set.
            value (str | int | float): The value of the item to set.
        """
        if not isinstance(key, (str, int, float)):
            raise TypeError(f"Value for key {key} must be a string, int, or float.")
        self._label_dict[key] = value
        self._label_dict_gui_handle.content = self._render_markdown()

    def __delitem__(self, key: str):
        """
        Deletes an item from the label_dict and updates the GUI.

        Args:
            key (str): The key of the item to delete.
        """
        if key not in self._label_dict:
            raise KeyError(f"Key {key} not found in label_dict.")
        del self._label_dict[key]
        self._label_dict_gui_handle.content = self._render_markdown()

    def __len__(self) -> int:
        """
        Returns the number of key-value pairs in the label_dict.

        Returns:
            int: The number of key-value pairs in the label_dict.
            This is the same as `len(dict)`.
        """
        return len(self._label_dict)

    def __contains__(self, key: str) -> bool:
        """
        Checks if a key is in the label_dict.

        Args:
            key (str): The key to check for existence in the label_dict.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self._label_dict

    def items(self):
        """
        Returns an iterable view of the key-value pairs in the label_dict.

        Returns:
            An iterable view of the key-value pairs in the label_dict. The same as `dict.items()`.
        """
        return self._label_dict.items()

    def keys(self):
        """
        Returns an iterable view of the keys in the label_dict.

        Returns:
            An iterable view of the keys in the label_dict. The same as `dict.keys()`.
        """
        return self._label_dict.keys()

    def values(self):
        """
        Returns a list of the values in the label_dict.

        Returns:
            list[str | int | float]: A list of the values in the label_dict.
        """
        return [v for v in self._label_dict.values()]

    @property
    def label_dict(self) -> InfoDict:
        """
        Returns the internal dictionary containing the label_dict data.

        Returns:
            dict[str, str | int | float]: The internal dictionary with key-value pairs.
        """
        return self._label_dict

    @label_dict.setter
    def label_dict(self, value: dict[str, str | int | float]):
        """
        Sets the internal dictionary to a new value and updates the GUI.

        Args:
            value (dict[str, str | int | float]): The new dictionary to set as the internal label_dict.
                Keys must be strings, and values can be strings, integers, or floats.
        """
        if not isinstance(value, dict):
            raise TypeError("label_dict must be a dictionary.")
        self._label_dict = value
        self._label_dict_gui_handle.content = self._render_markdown()

    @property
    def name(self) -> str:
        """
        Returns the name of the view. The name corresponds to the header in the GUI.
        """
        return self._name

    def _render_markdown(self) -> str:
        """
        Helper function to render the label_dict as a Markdown string.

        Returns:
            str: A Markdown formatted string representing the key-value pairs in the label_dict.
        """
        res = ""
        for key, value in self._label_dict.items():
            if isinstance(value, float):
                value = f"{value:.3f}"
            elif isinstance(value, int):
                value = f"{value:,}"
            res += f"**{key}**: {value}\n\n"
        return res

    def layout_gui(self):
        gui: viser.GuiApi = self._viewer_handle.gui
        with gui.add_folder(self._name, visible=True) as self._name_gui_handle:
            self._label_dict_gui_handle = gui.add_markdown(self._render_markdown())
