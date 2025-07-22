# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from abc import ABC, abstractmethod

import numpy as np

from .sfm_metadata import SfmCameraMetadata, SfmImageMetadata


class BaseDatasetReader(ABC):
    """
    Abstract base class for dataset readers.

    A DatasetReader is responsible for loading an SFM scene (see `SfmScene`) from a specific dataset format.

    In particular, it should implement the `load_data` method to load the cameras, images, and points from the dataset.
    The `num_images` property should return the number of images in the dataset.
    The `load_data` method should return a tuple containing:
        - A dictionary mapping camera IDs to `SfmCameraMetadata` objects.
        - A list of `SfmImageMetadata` objects.
        - An Nx3 array of 3D points in the scene, where N is the number of points.
        - An array of shape (N,) representing the error or uncertainty of each point in the 3D points.
        - An Nx3 uint8 array of RGB color values for each point in the scene, where N is the number of points.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the dataset reader.
        This constructor can be extended by subclasses to initialize specific dataset parameters.
        """
        pass

    @abstractmethod
    def load_data(
        self,
    ) -> tuple[dict[int, SfmCameraMetadata], list[SfmImageMetadata], np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the dataset from the specified path.
        This method should be implemented by subclasses.

        It is responsible for loading the cameras, images, and points from the dataset.

        Returns:
            camera_metadata (dict[int, SfmCameraMetadata]): A dictionary mapping camera IDs to `SfmCameraMetadata` objects.
            image_metadata (list[SfmImageMetadata]): A list of `SfmImageMetadata` objects.
            points (np.ndarray): An Nx3 array of 3D points in the scene, where N is the number of points.
            points_err (np.ndarray): An array of shape (N,) representing the error or uncertainty of each point in `points`.
            points_rgb (np.ndarray): An Nx3 uint8 array of RGB color values for each point in the scene, where N is the number of points.
        """
        pass

    @property
    @abstractmethod
    def num_images(self) -> int:
        """
        Return the number of images in the dataset.
        This property should be implemented by subclasses.

        Returns:
            int: The number of images in the dataset.
        """
        pass
