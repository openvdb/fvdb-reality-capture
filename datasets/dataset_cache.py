# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import json
import pathlib
import re
import shutil
import warnings
from typing import Any

import imageio
import numpy as np
import torch

__SECRET__ = object()


class DatasetCache:
    known_data_types = {
        "jpg",
        "png",
        "pt",
        "npy",
        "json",
        "txt",
    }

    metadata_filename = "cache_metadata.pt"
    magic_number = 0xAFAFAFAF  # Arbitrary magic number to identify the cache format
    version = "1.0.0"

    def _validate_name(self, key: str, name_type: str = "File") -> None:
        """
        Validate the key for the cache. The key must be a nonempty string consisting only of alphanumeric characters and underscores.
        """
        if not isinstance(key, str):
            raise TypeError(f"{name_type} name must be a string, got {type(key)}")
        if not re.match(r"^[a-zA-Z0-9_]+$", key):
            raise ValueError(
                f"{name_type} name '{key}' contains invalid characters. "
                f"{name_type}s must only contain alphanumeric characters and underscores."
            )
        if len(key) == 0:
            raise ValueError(f"{name_type} name cannot be an empty string.")

    @property
    def root_path(self) -> pathlib.Path:
        """
        Get the absolute path to the directory of the root cache.
        If this cache is not a child cache, this is the same as the cache path.
        If this cache is a child cache, this is the absolute path to the root cache directory
        (i.e., the cache directory where the root cache metadata file is stored).

        Returns:
            pathlib.Path: The absolute path to the root cache directory.
                This path is guaranteed to be absolute and will not change.
        """
        assert self._cache_root.is_absolute(), "Cache root path must be absolute."
        return self._cache_root

    @property
    def cache_path(self) -> pathlib.Path:
        """
        Get the absolute path to the cache directory.
        This is the directory where all cache files are stored.

        Returns:
            pathlib.Path: The absolute path to the cache directory.
        """
        return self._cache_root / self._current_folder

    @property
    def metadata_path(self) -> pathlib.Path:
        """
        Get the absolute path to the cache metadata file.
        This file contains metadata about the cache, such as the description, magic number, version, subcache name, and child caches.

        Returns:
            pathlib.Path: The absolute path to the cache metadata file.
                The file is named `cache_metadata.pt` and is stored in the cache directory.
                The metadata file is a PyTorch file that contains a dictionary with the following keys:
                - ".description": A human-readable description of the cache.
                - ".magic_number": The magic number for the cache format, used to identify the cache format.
                - ".version": The version of the cache format, used to identify the cache format.
                - ".current_folder": The name of the current folder this cache is in.
                - ".parent_folder": The name of the parent cache, if this is a child cache.
                - ".folders": A set containing the names of folders in the cache, initially empty. This is updated when a child cache is created.
        """
        return self.cache_path / DatasetCache.metadata_filename

    @property
    def description(self) -> str:
        """
        Get the description of the cache.

        Returns:
            str: The description of the cache, as stored in the metadata file.
        """
        return self._cache_metadata.get(".description", "")

    @staticmethod
    def get_cache(cache_root: pathlib.Path, description: str = "") -> "DatasetCache":
        return DatasetCache(
            cache_root, current_folder="", parent_folder="", description=description, _private=__SECRET__
        )

    def __init__(
        self,
        cache_root: pathlib.Path,
        current_folder: str,
        parent_folder: str,
        description: str = "",
        _private: Any = None,
    ):
        if _private != __SECRET__:
            raise ValueError(
                "Do not create a `DatasetCache` instance directly. Instead use `DatasetCache.get_cache()` to create a cache "
                "or make_folder to make a folder within a cache."
            )
        if not isinstance(cache_root, pathlib.Path):
            raise TypeError(f"Cache path must be a pathlib.Path, got {type(cache_root)}")
        if not isinstance(current_folder, str):
            raise TypeError(f"Subcache must be a string, got {type(current_folder)}")

        self._cache_root = cache_root.absolute()
        self._current_folder = current_folder
        self._parent_folder = parent_folder

        if not self.cache_path.exists():
            # We're creating the cache for the first time, so we need to create the directory and metadata file
            if not isinstance(description, str):
                raise TypeError(f"Description must be a string, got {type(description)}")

            self.cache_path.mkdir(parents=True, exist_ok=True)
            self._cache_metadata: dict[str, Any] = {
                ".description": description,
                ".magic_number": DatasetCache.magic_number,
                ".version": DatasetCache.version,
                ".current_folder": self._current_folder,
                ".parent_folder": parent_folder,
                ".folders": set(),  # Set of folder names, initially empty
            }
            torch.save(self._cache_metadata, self.metadata_path)
        else:
            # The cache directory already exists, so we need to load the metadata
            if not self.metadata_path.exists():
                raise ValueError(
                    f"Cache directory {self.cache_path} exists but does not contain a metadata file. "
                    "Please delete the cache directory and rebuild."
                )
            self._cache_metadata: dict[str, Any] = torch.load(self.metadata_path, weights_only=False)
            if ".description" not in self._cache_metadata:
                raise ValueError(
                    "Cache metadata does not contain a description. " "Please delete the cache directory and rebuild."
                )
            if (
                ".magic_number" not in self._cache_metadata
                or self._cache_metadata[".magic_number"] != DatasetCache.magic_number
            ):
                raise ValueError(
                    f"Cache metadata has an invalid magic number. "
                    f"Expected {DatasetCache.magic_number}, got {self._cache_metadata['.magic_number']}. "
                    "Please delete the cache directory and rebuild."
                )
            if ".version" not in self._cache_metadata or self._cache_metadata[".version"] != DatasetCache.version:
                raise ValueError(
                    f"Cache metadata has an invalid version. "
                    f"Expected {DatasetCache.version}, got {self._cache_metadata['.version']}. "
                    "Please delete the cache directory and rebuild."
                )
            if ".current_folder" not in self._cache_metadata:
                raise ValueError(
                    "Cache metadata does not contain .current_folder. Please delete the cache directory and rebuild."
                )
            if ".parent_folder" not in self._cache_metadata:
                raise ValueError(
                    "Cache metadata does not contain .parent_folder. Please delete the cache directory and rebuild."
                )
            self._current_folder = self._cache_metadata[".current_folder"]
            self._parent_folder = self._cache_metadata[".parent_folder"]

    def write_file(
        self, key: str, data: Any, data_type: str, metadata: dict = {}, quality: int = 100
    ) -> dict[str, Any]:
        """
        Create or update a file with the given name in the cache. This will write the data to a file in the cache directory
        with the name `{key}.{data_type}`. The data type must be one of the known data types (See `Cache2.known_data_types`).
        The metadata is optional and can be used to store additional information about the data.

        Args:
            key (str): The key for the property. This must be a nonempty string consisting only of alphanumeric characters and underscores.
            data (Any): The data to store in the cache. The type of data must match the registered data type for this key.
            data_type (str): The type of data stored for this property. Must be one of the known data types (See `Cache2.known_data_types`).
            metadata (dict): Optional additional metadata for the property. This can be used to store additional information about the property.
            quality (int): Optional quality parameter for image data types (e.g., JPEG). Defaults to 100.

        Returns:
            key_metadata (dict[str, Any]): A dictionary containing the data type, metadata, and path of the property in the cache.
        """

        self._validate_name(key)
        if data_type not in DatasetCache.known_data_types:
            raise ValueError(
                f"Unknown data type {data_type} for property {key}. Must be one of {DatasetCache.known_data_types}"
            )

        file_path = self.cache_path / f"{key}.{data_type}"
        if data_type == "jpg" or data_type == "png":
            imageio.imwrite(file_path, data, quality=quality)
        elif data_type == "pt":
            torch.save(data, file_path)
        elif data_type == "npy":
            np.save(file_path, data)
        elif data_type == "json":
            with open(file_path, "w") as f:
                json.dump(data, f)
        elif data_type == "txt":
            with open(file_path, "w") as f:
                f.write(data)
        else:
            raise ValueError(f"Unknown data type {data_type} for property {key}")

        self._cache_metadata[key] = {
            "data_type": data_type,
            "metadata": metadata,
        }
        torch.save(self._cache_metadata, self.metadata_path)

        return self.get_file_metadata(key)

    def read_file(self, filename: str) -> tuple[dict[str, Any], Any]:
        """
        Read the data in a cached file with the given name.

        Args:
            filename (str): The name of the file to read. This must be a nonempty string consisting only of alphanumeric characters and underscores.

        Returns:
            file_metadata (dict[str, Any]): A tuple containing metadata about the read file.
            data (Any): The data stored for the given key. The type of data will depend on the registered data type for this key.
        """

        file_metadata = self.get_file_metadata(filename)

        data_type = file_metadata["data_type"]
        file_path = file_metadata["path"]

        data = None
        if data_type == "jpg" or data_type == "png":
            data = imageio.imread(file_path)
        elif data_type == "pt":
            data = torch.load(file_path, weights_only=False)
        elif data_type == "npy":
            data = np.load(file_path)
        elif data_type == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
        elif data_type == "txt":
            with open(file_path, "r") as f:
                data = f.read()
        else:
            raise ValueError(
                f"Unknown data type {data_type} for image property. Must be one of {DatasetCache.known_data_types}"
            )

        return file_metadata, data

    def delete_file(self, filename: str) -> None:
        """
        Delete the cached value for a given key.

        This will remove the file from the cache directory and delete the key from the metadata.

        Args:
            key (str): The key for the property. This must be a nonempty string consisting only of alphanumeric characters and underscores.
        """
        file_metadata = self.get_file_metadata(filename)

        file_path = file_metadata["path"]

        if not file_path.exists():
            raise FileNotFoundError(f"File for property {filename} not found in cache at {file_path}")

        file_path.unlink(missing_ok=True)

        del self._cache_metadata[filename]

        torch.save(self._cache_metadata, self.metadata_path)

    def get_file_metadata(self, filename: str):
        """
        Get the metadata for a given file in the cache.

        The metadata is a dictionary with the following keys:
        - "data_type": The type of data stored for this file. Must be one of the known data types (See `Cache2.known_data_types`).
        - "metadata": A dict of additional metadata for the file, which can be used to store additional information about the file.
        - "path": The path to the file where the data for this file is stored. This is a pathlib.Path object.

        Args:
            filename (str): The filename for the property. This must be a nonempty string consisting only of alphanumeric characters and underscores.

        Returns:
            key_metadata (dict[str, Any]): A dictionary containing the data type, metadata, and path of the property in the cache.
        """
        self._validate_name(filename)
        file_metadata = self._cache_metadata.get(filename, None)
        if file_metadata is None:
            raise ValueError(f"Key {filename} not found in cache.")
        if "data_type" not in file_metadata:
            raise ValueError(
                f"Metadata for {filename} does not have a data type specified in the cache metadata. The cache may be corrupted."
            )
        if "metadata" not in file_metadata:
            raise ValueError(
                f"Metadata for {filename} does not have metadata specified in the cache metadata. The cache may be corrupted."
            )

        return {
            "data_type": file_metadata["data_type"],
            "metadata": file_metadata["metadata"],
            "path": self.cache_path / f"{filename}.{file_metadata['data_type']}",
        }

    def get_parent_cache(self) -> "DatasetCache":
        """
        Get the parent cache of this cache.

        If this cache is a child cache, this will return the parent cache.
        If this cache is the root cache, this will return itself.

        Returns:
            DatasetCache: The parent cache of this cache.
        """
        if self._parent_folder == "":
            return self
        return DatasetCache(
            self._cache_root,
            current_folder=self._parent_folder,
            parent_folder="",
            description="",
            _private=__SECRET__,
        )

    def make_folder(self, foldername: str, description: str = "") -> "DatasetCache":
        """
        Create or return a folder with the given name.
        If no folder with that name exists, this will create a new cache directory
        at the path `self.cache_path / name`.
        This will return a cache object for that directory.

        Args:
            foldername (str): The name of the folder to create. This must be a nonempty string
                consisting only of alphanumeric characters and underscores.
            description (str): A human-readable description of the folder. This is optional and can be
                used to provide additional information about the folder.
        Returns:
            DatasetCache: A new DatasetCache object for the created folder.
        """
        self._validate_name(foldername, "Folder")
        child_subcache = f"{self._current_folder}/{foldername}" if self._current_folder else foldername
        ret = DatasetCache(
            self._cache_root,
            current_folder=child_subcache,
            parent_folder=self._current_folder,
            description=description,
            _private=__SECRET__,
        )
        self._cache_metadata[".folders"].add(foldername)
        torch.save(self._cache_metadata, self.metadata_path)
        return ret

    def has_folder(self, foldername: str) -> bool:
        """
        Check if a folder with the given name exists in the cache.

        Args:
            foldername (str): The name of the child cache to check for.

        Returns:
            bool: True if the child cache exists, False otherwise.
        """
        self._validate_name(foldername, "Folder")
        return foldername in self._cache_metadata[".folders"]

    def has_file(self, filename: str) -> bool:
        """
        Check if a given key exists in the cache.

        Args:
            key (str): The key for the property. This must be a nonempty string consisting only of alphanumeric characters and underscores.

        Returns:
            bool: True if the key exists in the cache, False otherwise.
        """
        self._validate_name(filename)
        return filename in self._cache_metadata

    @property
    def num_files(self) -> int:
        """
        Get the number of keys in the cache.

        Returns:
            int: The number of keys in the cache.
        """
        return len([key for key in self._cache_metadata if not key.startswith(".")])

    @property
    def num_folders(self) -> int:
        """
        Get the number of child caches in the cache.

        Returns:
            int: The number of child caches in the cache.
        """
        return len(self._cache_metadata[".folders"])

    def clear_all(self) -> None:
        """
        Clear the cache by deleting all keys and subcaches in the cache directory.
        This does not delete the cache directory itself.
        """
        for item in self.cache_path.iterdir():
            if item.name == DatasetCache.metadata_filename:
                continue
            if not item.is_relative_to(self.root_path):
                raise ValueError(
                    f"Attempting to delete a directory {item} that is not relative to the cache root {self.root_path}. "
                    "This may be a security issue."
                )

            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        for key in list(self._cache_metadata.keys()):
            if key.startswith("."):
                continue
            del self._cache_metadata[key]
        self._cache_metadata[".folders"] = set()

        torch.save(self._cache_metadata, self.metadata_path)
