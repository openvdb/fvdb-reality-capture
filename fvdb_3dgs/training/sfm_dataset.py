# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from collections.abc import Iterable
from typing import Any, Dict, Sequence

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision

from ..sfm_scene import SfmCameraMetadata, SfmImageMetadata, SfmScene


class SfmDataset(torch.utils.data.Dataset, Iterable):
    """
    A torch dataset encoding posed images from a Structure from Motion (SfM) pipeline.

    This class provides an interface to load and manipulate datasets from SfM pipelines
    (e.g. those generated from COLMAP).

    Each item in the dataset is an image with a corresponding camera pose, projection matrix,
    and optionally mask and depth information.

    The class also provides methods to access camera to world matrices, projection matrices,
    scene scale, and 3D points within the SFM scene.

    The dataset provides an API for common transformations on this kind of data used in reality capture.
    In particular it supports normalization of the scene, filtering points based on percentiles, and downsampling images.
    """

    def __init__(
        self,
        sfm_scene: SfmScene,
        dataset_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        patch_size: int | None = None,
        return_visible_points: bool = False,
    ):
        """
        Create a new SfmDataset instance.

        Args:
            sfm_scene: The SfmScene for this dataset
            dataset_indices: Indices of images to include in the dataset. If None, all images will be used.
            patch_size: If not None, images will be randomly cropped to this size.
            return_visible_points: If True, depths of visible points will be loaded and included in each datum.
        """
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._sfm_scene = sfm_scene

        self.patch_size = patch_size
        self._return_visible_points = return_visible_points

        # If you specified image indices, we'll filter the dataset to only include those images.
        if dataset_indices is None:
            dataset_indices = np.arange(self._sfm_scene.num_images)
        elif isinstance(dataset_indices, torch.Tensor):
            dataset_indices = dataset_indices.cpu().numpy()
        else:
            dataset_indices = np.asarray(dataset_indices)
        if dataset_indices.dtype not in (np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64):
            raise ValueError("Dataset indices must be integers")
        dataset_indices = dataset_indices.astype(np.int64)

        self._indices: np.ndarray = dataset_indices

    @property
    def sfm_scene(self) -> SfmScene:
        """
        Returns the SfmScene associated with this dataset.

        Returns:
            sfm_scene (SfmScene): The SfmScene associated with this dataset.
        """
        return self._sfm_scene

    @property
    def indices(self) -> np.ndarray:
        """
        Return the indices of the images in the SfmScene used in the dataset.

        Returns:
            np.ndarray: The indices of the images in the SfmScene used in the dataset.
        """
        return self._indices

    @property
    def scene_bbox(self) -> np.ndarray:
        """
        Get the bounding box of the scene.

        The bounding box is defined as a tensor of shape (2, 3) where the first row is the minimum
        corner and the second row is the maximum corner of the bounding box.

        Returns:
            torch.Tensor: A tensor of shape (2, 3) representing the bounding box of the scene.
        """
        return self._sfm_scene.scene_bbox.reshape([2, 3])

    @property
    def camera_to_world_matrices(self) -> np.ndarray:
        """
        Get the camera to world matrices for all images in the dataset.

        This returns the camera to world matrices as a numpy array of shape (N, 4, 4) where N is the number of images.

        Returns:
            np.ndarray: An Nx4x4 array of camera to world matrices for the cameras in the dataset.
        """
        return self.sfm_scene.camera_to_world_matrices[self._indices]

    @property
    def projection_matrices(self) -> np.ndarray:
        """
        Get the projection matrices mapping camera to pixel coordinates for all images in the dataset.

        This returns the undistorted projection matrices as a numpy array of shape (N, 3, 3) where N is the number of images.

        Returns:
            np.ndarray: An Nx3x3 array of projection matrices for the cameras in the dataset.
        """
        # in fvdb_3dgs/training/sfm_dataset.py
        return np.stack(
            [self._sfm_scene.images[i].camera_metadata.projection_matrix for i in self._indices],
            axis=0,
        )

    @property
    def image_sizes(self) -> np.ndarray:
        """
        Get the image sizes for all images in the dataset.

        This returns the image sizes as a numpy array of shape (N, 2) where N is the number of images.
        Each row contains the height and width of the corresponding image.

        Returns:
            np.ndarray: An Nx2 array of image sizes for the cameras in the dataset.
        """
        return self.sfm_scene.image_sizes[self._indices]

    @property
    def points(self) -> np.ndarray:
        """
        Get the 3D points in the scene.
        This returns the points in world coordinates as a numpy array of shape (N, 3) where N is the number of points.

        Returns:
            np.ndarray: An Nx3 array of 3D points in the scene.
        """
        return self.sfm_scene.points

    @property
    def visible_point_indices(self) -> np.ndarray:
        """
        Return the indices of all points that are visible by some camera in the dataset.
        This is useful for filtering points that are not visible in any image.

        Returns:
            np.ndarray: An array of point indices that are visible in at least one image.
        """
        visible_points = set()
        for idx in self._indices:
            image_meta: SfmImageMetadata = self._sfm_scene.images[idx]
            visible_points.update(image_meta.point_indices.tolist())
        return np.array(list(visible_points))

    @property
    def points_rgb(self) -> np.ndarray:
        """
        Return the RGB colors of the points in the scene as a uint8 numpy array.
        The shape of the array is (N, 3) where N is the number of points.

        Returns:
            np.ndarray: An Nx3 array of uint8 RGB colors for the points in the scene.
        """
        return self._sfm_scene.points_rgb

    def __iter__(self):
        """
        Iterate over the dataset

        Yields:
            The next image in the dataset.
        """
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """
        Get the number of images in the dataset.
        This is the number of images that will be returned by the dataset iterator.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self._indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.

        An item is a dictionary with the following keys:
         - projection: The projection matrix for the camera.
         - camera_to_world: The camera to world transformation matrix.
         - world_to_camera: The world to camera transformation matrix.
         - image: The image tensor.
         - image_id: The index of the image in the dataset.
         - image_path: The file path of the image.
         - points (Optional): The projected points in the image (if return_visible_points is True).
         - depths (Optional): The depths of the projected points (if return_visible_points is True).
         - mask (Optional): The mask tensor (if available).
         - mask_path (Optional): The file path of the mask (if available).

        Returns:
            Dict[str, Any]: A dictionary containing the image data and metadata.
        """
        index = self._indices[item]

        image_meta: SfmImageMetadata = self._sfm_scene.images[index]
        camera_meta: SfmCameraMetadata = image_meta.camera_metadata

        if image_meta.image_path.endswith(".jpg") or image_meta.image_path.endswith(".jpeg"):
            data = torchvision.io.read_file(image_meta.image_path)
            image = torchvision.io.decode_jpeg(data, device="cpu").permute(1, 2, 0).numpy()
        elif image_meta.image_path.endswith(".png"):
            data = torchvision.io.read_file(image_meta.image_path)
            image = torchvision.io.decode_png(data, device="cpu").permute(1, 2, 0).numpy()
        else:
            image = cv2.imread(image_meta.image_path, cv2.IMREAD_UNCHANGED)
            assert image is not None, f"Failed to load image: {image_meta.image_path}"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.ndim == 2:
            image = image[:, :, None]
        image = camera_meta.undistort_image(image)

        projection_matrix = camera_meta.projection_matrix.copy()  # undistorted projection matrix
        camera_to_world_matrix = image_meta.camera_to_world_matrix.copy()
        world_to_camera_matrix = image_meta.world_to_camera_matrix.copy()

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            projection_matrix[0, 2] -= x
            projection_matrix[1, 2] -= y

        data = {
            "projection": torch.from_numpy(projection_matrix).float(),
            "camera_to_world": torch.from_numpy(camera_to_world_matrix).float(),
            "world_to_camera": torch.from_numpy(world_to_camera_matrix).float(),
            "image": image,
            "image_id": item,  # the index of the image in the dataset
            "image_path": image_meta.image_path,
        }

        # If you passed in masks, we'll set set these in the data dictionary
        if image_meta.mask_path != "":
            mask = cv2.imread(image_meta.mask_path, cv2.IMREAD_GRAYSCALE)
            assert mask is not None, f"Failed to load mask: {image_meta.mask_path}"
            mask = mask > 127

            data["mask_path"] = image_meta.mask_path
            data["mask"] = mask

        # If you asked to load depths, we'll load the depths of visible colmap points
        if self._return_visible_points:
            # projected points to image plane to get depths
            points_world = self._sfm_scene.points[image_meta.point_indices]
            points_cam = (world_to_camera_matrix[:3, :3] @ points_world.T + world_to_camera_matrix[:3, 3:4]).T
            points_proj = (projection_matrix @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            if self.patch_size is not None:
                points[:, 0] -= x
                points[:, 1] -= y
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()
        return data


__all__ = ["SfmDataset"]
