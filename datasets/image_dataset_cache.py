# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import json
import math
import os
import pathlib
import re
import shutil
from typing import Any

import imageio
import numpy as np
import torch


class ImageDatasetCache:
    """
    A class to manage a persistent cache for a dataset. This class is used to store per-image
    data such as derived feature images (E.g. by running CLIP, SAM, etc.), downsampled images, etc.
    """

    known_data_types = {
        "jpg",
        "png",
        "pt",
        "npy",
        "json",
        "txt",
    }

    metadata_filename = "cache_metadata.pt"

    def __init__(self, dataset_path: pathlib.Path, num_images: int = 0, prefix: str = ""):
        """
        Initialize the dataset cache given the path to the dataset directory. If there is no _cache directory,
        in the dataset directory, it will be created and the specified number of images will be registered.
        If prefix is passed, it will be used as a prefix for the cache keys and file names. _i.e._
        when accessing the key "mykey", the cache will use the key "prefix.mykey" and look for files of the form
        `_cache/prefix.mykey*`.

        If there is a _cache directory, the metadata will be loaded from the `cache_metadata.pt` file.

        The `cache_metadata.pt` file is a dictionary with one key called ".num_images" which contains the number of images,
        and one key for each registered image property. Each image property key maps to a dictionary with the following keys:
        - "data_type": The type of data stored for this image property. Must be one of the known data types (See `ImageDatasetCache.known_data_types`).
        - "description": A human-readable description of the image property.
        - "scope": Whether this is a per-image ("image") or per-dataset ("dataset") property. This is used to determine how to store the data in the cache.
        - "metadata": Optional additional metadata for the image property. This can be used to store additional information about the image property.

        Args:
            dataset_path (pathlib.Path): The path to the dataset directory where the cache is stored (in the dataset_path/_cache).
            num_images (int): The number of images in the dataset if the cache is being created. This is used to create a zero-padded image ID for each image.
                              If the dataset cache already exists, this value is ignored and the number of images is read from the cache metadata.
        """
        self._cache_path: pathlib.Path = (dataset_path / pathlib.Path("_cache")).resolve()
        self._num_images: int = num_images

        if not isinstance(prefix, str):
            raise TypeError(f"Prefix must be a string, got {type(prefix)}")
        if not re.match(r"^[a-zA-Z0-9_]*$", prefix):
            raise ValueError(
                f"Prefix {prefix} contains invalid characters. "
                "Prefixes must only contain alphanumeric characters and underscores."
            )
        self._cache_prefix: str = prefix

        if not self._cache_path.exists():
            if num_images <= 0:
                raise ValueError(
                    "Cannot create a cache directory without a valid number of images. "
                    "Please specify a positive number of images. You passed in `num_images` = {num_images}."
                )
            self._cache_path.mkdir(exist_ok=True)
            self._cache_image_metadata: dict[str, tuple[dict[str, str], dict] | int] = {}
            self._cache_image_metadata[".num_images"] = self._num_images
            torch.save(self._cache_image_metadata, self._cache_path / ImageDatasetCache.metadata_filename)
        else:
            self._cache_image_metadata: dict[str, tuple[dict[str, str], dict] | int] = torch.load(
                self._cache_path / ImageDatasetCache.metadata_filename, weights_only=False
            )
            if ".num_images" not in self._cache_image_metadata:
                raise RuntimeError(
                    "Dataset cache metadata does not contain the number of images. "
                    "Please delete the cache directory and rebuild the dataset."
                )
            num_images_read = self._cache_image_metadata[".num_images"]
            if not isinstance(num_images_read, int) or num_images_read <= 0:
                raise ValueError(
                    f"Invalid number of images in dataset cache: {num_images_read}. "
                    "Please delete the cache directory and rebuild the dataset."
                )
            self._num_images = num_images_read

    def _zeropad_image_id(self, image_id: int) -> str:
        """
        Zero-pad the image ID to so all image data has enough leading zeros to be sorted correctly.
        e.g. if there are 75 images, we'll pad with two zeros 1 -> "001", 12 -> "012".

        Args:
            image_id (int): The image ID to zero-pad.

        Returns:
            str: The zero-padded image ID as a string.
        """
        order_of_magnitude = int(math.log10(self._num_images)) + 1 if self._num_images > 0 else 1

        return f"{image_id:0{2*order_of_magnitude}d}"

    def _image_key_path(self, key: str) -> pathlib.Path:
        """
        Get the path to the image key directory for a given key.
        This directory is where all files for the given image key will be stored.
        It has the format `_cache/image_{key}` where `key` is the image property key.

        Args:
            key (str): The key for the image property.

        Returns:
            pathlib.Path: The path to the image key directory.
        """
        return self._cache_path / f"image_{key}"

    def _dataset_key_path(self, key: str, data_type: str) -> pathlib.Path:
        """
        Get the path to the dataset key file for a given key.
        This file is where the data for the given dataset key will be stored.
        It has the format `_cache/{key}.{data_type}` where `key` is the dataset property key.

        Args:
            key (str): The key for the dataset property.
            data_type (str): The type of data stored for this dataset property. Must be one of the known data types.

        Returns:
            pathlib.Path: The path to the dataset key file.
        """
        return self._cache_path / f"{key}.{data_type}"

    def _image_key_file_path(self, key: str, image_id: int, data_type: str) -> pathlib.Path:
        """
        Get the path to the file for a given image key and image ID.

        The file is stored in the `_cache/image_{key}` directory with the name `zeropad({image_id}).{data_type}`.
        i.e. `_cache/image_{key}/zeropad({image_id}).{data_type}`

        Args:
            key (str): The key for the image property.
            image_id (int): The ID of the image.
            data_type (str): The type of data stored for this image property. Must be one of the known data types.

        Returns:
            pathlib.Path: The path to the file for the given image key and image ID.
        """
        return self._image_key_path(key) / f"{self._zeropad_image_id(image_id)}.{data_type}"

    def _get_key_metadata(self, key: str) -> tuple[str, str, str, dict]:
        """
        Gets the metadata for a given property in the cache and validates it.

        Args:
            key (str): The key for the image property.
        Returns:
            data_type (str): The type of data stored for this image property. Must be one of the known data types.
            scope (str): The scope of the property, either "image" or "dataset". This determines where the data is stored.
            description (str): A human-readable description of the image property.
        """
        assert key in self._cache_image_metadata, f"Property {key} not found in dataset cache."

        value = self._cache_image_metadata[key]
        assert isinstance(value, tuple), f"Property {key} metadata is not a tuple. Got {type(value)} instead."

        key_meta, value_meta = value
        assert "data_type" in key_meta, f"Property {key} does not have a data type specified in the dataset cache."
        assert "description" in key_meta, f"Property {key} does not have a description specified in the dataset cache."
        assert (
            "scope" in key_meta
        ), f"Property {key} does not have a scope specified in the dataset cache. Must be 'image' or 'dataset'."

        scope: str = key_meta["scope"]
        if scope not in {"image", "dataset"}:
            raise ValueError(f"Property {key} has an invalid scope {scope}. Must be either 'image' or 'dataset'.")
        data_type: str = key_meta["data_type"]
        if data_type not in ImageDatasetCache.known_data_types:
            raise ValueError(
                f"Unknown data type {data_type} for property {key}. "
                f"Must be one of {ImageDatasetCache.known_data_types}"
            )
        description: str = key_meta["description"]
        if not isinstance(description, str):
            raise TypeError(f"Description for property {key} must be a string, got {type(description)}")

        if not isinstance(value_meta, dict):
            raise TypeError(f"Value metadata for property {key} must be a dictionary, got {type(value_meta)}")
        return data_type, scope, description, value_meta

    def _get_image_property_path_and_data_type(self, key: str, image_id: int) -> tuple[pathlib.Path, str, dict]:
        """
        Get the file path and data type for a given image property key and image ID.
        This method checks if the key is registered in the dataset cache and corresponds to an image property.
        If so, it retrieves the file path and data type for the image property. Otherwise, it throws an exception.

        Args:
            key (str): The key for the image property.
            image_id (int): The ID of the image.
        Returns:
            file_path (pathlib.Path): The path to the file for the given image key and image ID.
            data_type (str): The type of data stored for this image property. Must be one of the known data types.
        """
        if key not in self._cache_image_metadata:
            raise KeyError(
                f"Image property {key} not found in dataset cache. Please register it first with `register_image_property`."
            )

        data_type, scope, description, value_meta = self._get_key_metadata(key)

        if scope != "image":
            raise ValueError(
                f"Image property {key} is registered as a dataset property but is being accessed as an image property. "
                "Please register it as an image property with `register_image_property`."
            )

        file_path = self._image_key_file_path(key, image_id, data_type)

        return file_path, data_type, value_meta

    def _load_cache_data(self, file_path: pathlib.Path, data_type: str):
        """
        Read data from the cache for a given property (image or dataset) from the path specified by `file_path`.

        Args:
            file_path (pathlib.Path): The path to the file where the data is stored.
            data_type (str): The type of data stored for this image property. Must be one of the known data types (See `ImageDatasetCache.known_data_types`).
        """
        if data_type == "jpg" or data_type == "png":
            return imageio.imread(file_path)
        elif data_type == "pt":
            return torch.load(file_path, weights_only=False)
        elif data_type == "npy":
            return np.load(file_path)
        elif data_type == "json":
            with open(file_path, "r") as f:
                return json.load(f)
        elif data_type == "txt":
            with open(file_path, "r") as f:
                return f.read()
        else:
            raise ValueError(
                f"Unknown data type {data_type} for image property. Must be one of {ImageDatasetCache.known_data_types}"
            )

    def _save_cache_data(
        self, key: str, scope: str, file_path: pathlib.Path, data: Any, data_type: str, quality: int | None = None
    ) -> None:
        """
        Write data to the cache for a given property (image or dataset) to the path specified by `file_path`.

        Args:
            key (str): The key for the image property. This must be a nonempty string consisting only of alphanumeric characters and underscores.
            scope (str): The scope of the property, either "image" or "dataset". This determines where the data is stored.
            file_path (pathlib.Path): The path to the file where the data will be saved.
            data (Any): The data to save for the image property. The type of data must match the registered data type for this key.
            data_type (str): The type of data stored for this image property. Must be one of the known data types (See `ImageDatasetCache.known_data_types`).
        """

        assert scope in {"image", "dataset"}, f"Scope must be either 'image' or 'dataset', got {scope}"

        if data_type == "jpg" or data_type == "png":
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    f"Data for {'image' if scope == 'image' else 'dataset'} property {key} must be a numpy array for {data_type} type."
                )
            if quality is not None and not isinstance(quality, int):
                raise TypeError(
                    f"Quality must be an integer if specified, got {type(quality)} for image property {key}."
                )
            if quality is not None:
                imageio.imwrite(file_path, data, quality=quality)
            else:
                imageio.imwrite(file_path, data)
        elif data_type == "pt":
            torch.save(data, file_path)
        elif data_type == "npy":
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    f"Data for {'image' if scope == 'image' else 'dataset'} property {key} must be a numpy array for {data_type} type."
                )
            np.save(file_path, data)
        elif data_type == "json":
            json_str = json.dumps(data)
            with open(file_path, "w") as f:
                f.write(json_str)
        elif data_type == "txt":
            if not isinstance(data, str):
                raise TypeError(
                    f"Data for {'image' if scope == 'image' else 'dataset'} property {key} must be a string for {data_type} type."
                )
            with open(file_path, "w") as f:
                f.write(data)
        else:
            raise ValueError(f"Unknown data type {data_type} for image property {key}")

    def _validate_key_description_and_datatype(self, key: str, data_type: str, description: str) -> None:
        """
        Validate the key, data type, and description for a dataset property.
        This method checks that the key is a nonempty string consisting only of alphanumeric characters and underscores,
        that the description is a string, and that the data type is one of the known data types.

        If any of these conditions are not met, it raises an appropriate exception.

        Args:
            key (str): The key for the image property. This must be a nonempty string consisting only of alphanumeric characters and underscores.
            data_type (str): The type of data stored for this image property. Must be one of the known data types (See `ImageDatasetCache.known_data_types`).
            description (str): A brief description of the image property.
        """
        # Ensure key is a nonempty string consisting only of alphanumeric characters and underscores
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}")
        if not re.match(r"^[a-zA-Z0-9_]+$", key):
            raise ValueError(
                f"Key {key} contains invalid characters. "
                "Keys must only contain alphanumeric characters and underscores."
            )
        if len(key) == 0:
            raise ValueError("Key cannot be an empty string.")

        # Ensure description is a string
        if not isinstance(description, str):
            raise TypeError(f"Description must be a string, got {type(description)}")

        # Ensure data_dtype is a known data type
        if data_type not in ImageDatasetCache.known_data_types:
            raise ValueError(
                f"Unknown data type {data_type} for image property {key}. Must be one of {ImageDatasetCache.known_data_types}"
            )

    def _prefix_key(self, key: str) -> str:
        """
        Prefix the key with the cache prefix to ensure unique keys across different datasets.

        Args:
            key (str): The key for the image property.

        Returns:
            str: The prefixed key.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}")
        return f"{self._cache_prefix}.{key}" if self._cache_prefix else key

    def __contains__(self, key: str) -> bool:
        """
        Check if a given key is registered in the dataset cache.

        Args:
            key (str): The key for the image property.

        Returns:
            bool: True if the key is registered in the dataset cache, False otherwise.
        """
        key = self._prefix_key(key)
        return key in self._cache_image_metadata

    def num_values_for_image_property(self, key: str) -> int:
        """
        Get the number of values stored for a given image property key in the dataset cache.

        This method checks if the key is registered in the dataset cache and corresponds to an image property.
        If so, it returns the number of images in the dataset. Otherwise, it raises an exception.

        Args:
            key (str): The key for the image property.

        Returns:
            int: The number of values stored for the given image property key.
        """
        key = self._prefix_key(key)

        if key not in self._cache_image_metadata:
            raise KeyError(f"Image property {key} not found in dataset cache.")

        data_type, scope, description, value_meta = self._get_key_metadata(key)
        if scope != "image":
            raise ValueError(
                f"Image property {key} is registered as a dataset property but is being accessed as an image property. "
                "Please register it as an image property with `register_image_property`."
            )

        # If the directory doesn't exist, it means there are no values for this image property
        if not self._image_key_path(key).exists():
            self._image_key_path(key).mkdir(exist_ok=True)
        return len(os.listdir(self._image_key_path(key)))

    def register_image_property(self, key: str, data_type: str, description: str = "", metadata: dict = {}) -> None:
        """
        Register a new image property in the dataset cache.

        An image property maps to a directory in the cache where all files for this image property will be stored.

        If we register a new image property, we create a new directory in the cache with the name `image_{key}`.

        Args:
            key (str): The key for the image property. This must be a nonempty string consisting only of alphanumeric characters and underscores.
            data_type (str): The type of data stored for this image property. Must be one of the known data types (See `ImageDatasetCache.known_data_types`).
            description (str): A brief description of the image property.
        """

        self._validate_key_description_and_datatype(key, data_type, description)

        key = self._prefix_key(key)

        # Ensure this key doesn't already exist in the cache
        if key in self._cache_image_metadata:
            raise ValueError(
                f"Image property {key} already exists in the dataset cache. "
                f"Please use a different key or delete the existing property."
            )

        # Get the path on the filesystem where the data for this key will be stored
        key_path = self._image_key_path(key)

        # Ensure the key path does not already exist
        if key_path.exists():
            raise FileExistsError(
                f"Files for image property {key} already exists in the dataset cache but key does not exist. "
                f"Someone probably modified the cache directory manually and it is now in a bad state. "
                f"You should delete it and rebuild. Note: Cache directory is {self._cache_path}"
            )

        # Try to create the directory for this key and register the key in the metadata
        try:
            self._cache_image_metadata[key] = {
                "data_type": data_type,
                "description": description,
                "scope": "image",  # This is an image property, not a dataset property
            }, metadata
            torch.save(self._cache_image_metadata, self._cache_path / ImageDatasetCache.metadata_filename)
            key_path.mkdir(exist_ok=False)
        except Exception as e:
            # If we fail to create the directory, we'll try to rollback the changes
            # by deleting the key from the metadata and removing the directory
            # There are still many ways the rollback can fail (e.g if any of the below calls raise
            # an exception) and the cache can get in a bad state, but this should work in a lot of cases.
            if key in self._cache_image_metadata:
                del self._cache_image_metadata[key]
            torch.save(self._cache_image_metadata, self._cache_path / ImageDatasetCache.metadata_filename)

            if key_path.exists() and key_path.resolve().is_relative_to(self._cache_path):
                shutil.rmtree(key_path, ignore_errors=True)
            raise e

    def delete_property(self, key: str) -> None:
        """
        Delete a property from the dataset cache.

        This method will remove the property from the metadata and delete the directory where the data for this property is stored.

        Args:
            key (str): The key for the property. This must be a nonempty string consisting only of alphanumeric characters and underscores.
        """
        key = self._prefix_key(key)

        # Ensure the key exists in the cache
        if key not in self._cache_image_metadata:
            raise KeyError(f"Image property {key} not found in dataset cache.")

        # Get the scope of the key so we know how to delete it
        data_type, scope, _, _ = self._get_key_metadata(key)

        if scope == "dataset":
            # If the scope is "dataset", we just delete the file
            value_path = self._dataset_key_path(key, data_type)
            if value_path.exists():
                os.remove(value_path)
        elif scope == "image":
            # Otherwise if it's an image, we need to delete the whole image
            # Get the path to the directory where the data for this key is stored
            value_path = self._image_key_path(key)

            # Remove the directory where the data for this key is stored
            if value_path.exists() and value_path.resolve().is_relative_to(self._cache_path):
                shutil.rmtree(value_path, ignore_errors=True)

        # Remove the key from the metadata
        del self._cache_image_metadata[key]

        torch.save(self._cache_image_metadata, self._cache_path / ImageDatasetCache.metadata_filename)

    def get_property_metadata(self, key: str) -> tuple[dict[str, str], dict]:
        """
        Get the metadata for a given property in the cache.

        Args:
            key (str): The key for the image property.

        Returns:
            key_meta (dict[str, str]): A dictionary containing the data type, scope, and description of the image property.
            value_meta (dict): A dictionary containing any additional metadata associated with the image property.
        """
        key = self._prefix_key(key)

        if key not in self._cache_image_metadata:
            raise KeyError(f"Image property {key} not found in dataset cache.")

        data_type, scope, description, value_meta = self._get_key_metadata(key)

        return {"data_type": data_type, "scope": scope, "description": description}, value_meta

    def set_image_property(self, key: str, image_id: int, data: Any, quality: int | None = None) -> None:
        """
        Save an image property for a given image ID in the dataset cache.

        This method will save the data to a file in the `_cache/image_{key}` directory with the name `zeropad({image_id}).{data_type}`.

        Args:
            key (str): The key for the image property. This must be a nonempty string consisting only of alphanumeric characters and underscores.
            image_id (int): The ID of the image to save the property for. This should be a nonnegative integer.
            data (Any): The data to save for the image property. The type of data must match the registered data type for this key.
            quality (int, optional): The quality of the image to save if the data type is "jpg" or "png". Defaults to None.
        """
        key = self._prefix_key(key)
        file_path, data_type, _ = self._get_image_property_path_and_data_type(key, image_id)

        self._save_cache_data(
            key=key, scope="image", file_path=file_path, data=data, data_type=data_type, quality=quality
        )

    def get_image_property(self, key: str, image_id: int, default_value: Any = None) -> tuple[Any, dict]:
        """
        Read an image property for a given image ID from the dataset cache.

        This method will read the data from a file in the `_cache/image_{key}` directory with the name `zeropad({image_id}).{data_type}`.

        If the file does not exist, it will return the `default_value` provided.

        Args:
            key (str): The key for the image property. This must be a nonempty string consisting only of alphanumeric characters and underscores.
            image_id (int): The ID of the image to read the property for. This should be a nonnegative integer.
            default_value (Any): The value to return if the image property file does not exist. Defaults to None.
        """
        key = self._prefix_key(key)
        file_path, data_type, value_meta = self._get_image_property_path_and_data_type(key, image_id)
        if not file_path.exists():
            return default_value

        return self._load_cache_data(file_path, data_type), value_meta

    def set_dataset_property(
        self, key: str, data_type: str, data: Any, description: str = "", metadata: dict = {}
    ) -> None:
        """
        Set a dataset property at the scope of a whole dataset in the cache.

        If the property does not exist, it will be created. If it does exist, it will be overwritten.

        Args:
            key (str): The key for the dataset property. This must be a nonempty string consisting only of alphanumeric characters and underscores.
            data_type (str): The type of data stored for this dataset property. Must be one of the known data types (See `ImageDatasetCache.known_data_types`).
            data (Any): The data to save for the dataset property. The type of data must match the registered data type for this key.
            description (str): A brief description of the dataset property. Defaults to an empty string.
            metadata (dict): Optional additional metadata for the dataset property. This can be used to store additional information about the dataset property.
        """
        self._validate_key_description_and_datatype(key, data_type, description)

        key = self._prefix_key(key)

        value_path = self._dataset_key_path(key, data_type)

        # Try to create the directory for this key and register the key in the metadata
        try:
            self._cache_image_metadata[key] = {
                "data_type": data_type,
                "description": description,
                "scope": "dataset",
                "metadata": metadata,
            }, metadata
            self._save_cache_data(key=key, scope="dataset", file_path=value_path, data=data, data_type=data_type)
            torch.save(self._cache_image_metadata, self._cache_path / ImageDatasetCache.metadata_filename)
        except Exception as e:
            # If we fail to create the directory, we'll try to rollback the changes
            # by deleting the key from the metadata and removing the directory
            # There are still many ways the rollback can fail (e.g if any of the below calls raise
            # an exception) and the cache can get in a bad state, but this should work in a lot of cases.
            if key in self._cache_image_metadata:
                del self._cache_image_metadata[key]
            torch.save(self._cache_image_metadata, self._cache_path / ImageDatasetCache.metadata_filename)
            if value_path.exists():
                os.remove(value_path)
            raise e

    def get_dataset_property(self, key: str, default_value: Any = None) -> tuple[Any, dict]:
        """
        Read a dataset property from the dataset cache. This method is used to retrieve properties that
        are not specific to a single image, but rather apply to the entire dataset.

        Args:
            key (str): The key for the dataset property. This must be a nonempty string consisting only of alphanumeric characters and underscores.
            default_value (Any): The value to return if the dataset property file does not exist. Defaults to None.

        Returns:
            Any: The data stored for the dataset property, or the `default_value` if the property does not exist.
        """
        key = self._prefix_key(key)
        if key not in self._cache_image_metadata:
            return default_value

        data_type, scope, description, metadata = self._get_key_metadata(key)
        if scope != "dataset":
            raise ValueError(
                f"Dataset property {key} is registered as an image property but is being accessed as a dataset property. "
                "Please register it as a dataset property with `register_image_property`."
            )
        file_path = self._cache_path / f"{key}.{data_type}"

        # There's no file associated with the key, delete it and return the default value
        if not file_path.exists():
            del self._cache_image_metadata[key]
            torch.save(self._cache_image_metadata, self._cache_path / ImageDatasetCache.metadata_filename)
            return default_value

        return self._load_cache_data(file_path, data_type), metadata

    def get_subcache(self, prefix: str) -> "ImageDatasetCache":
        """
        Get a subcache of the current dataset cache with a given prefix.

        This method returns a new `ImageDatasetCache` instance that is a subcache of the current cache.
        The new cache will have the same dataset path and number of images, but will append the prefix to this cache's prefix.

        _i.e_ if this cache has prefix "main" and the subcache is created with prefix "sub",
        the new cache will have prefix "main.sub".

        Args:
            prefix (str): The prefix to use for the subcache. This must be a nonempty string consisting only of alphanumeric characters and underscores.

        Returns:
            ImageDatasetCache: A new `ImageDatasetCache` instance that is a subcache of the current cache.
        """
        prefix = f"{self._cache_prefix}.{prefix}" if self._cache_prefix else prefix
        return ImageDatasetCache(self._cache_path.parent, self._num_images, prefix=prefix)
