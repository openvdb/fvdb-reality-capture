# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
from collections.abc import Iterable
from typing import Any, Dict, List, Literal

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.utils.data
import tqdm

from .dataset_cache import DatasetCache
from .sfm_scene import SfmCameraMetadata, SfmImageMetadata, SfmScene, load_colmap_scene
from .transforms import BaseTransform, Identity


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
        dataset_path: pathlib.Path,
        test_every: int = 100,
        split: Literal["train", "test", "all"] = "train",
        transform: BaseTransform = Identity(),
        image_indices: List[int] | None = None,
        patch_size: int | None = None,
        return_visible_points: bool = False,
    ):
        """
        Create a new SfmDataset instance.

        Args:
            dataset_path: Path to the SfM dataset directory.
            test_every: If > 0, every Nth image will be used for testing.
            split: The split of the dataset to use. Options are "train", "test", or "all". If "train", only
                training images will be used. If "test", only testing images will be used.
                If "all", all images will be used.
            image_indices: Optional list of image indices to include in the dataset. If None, all images will be used.
            patch_size: If not None, images will be randomly cropped to this size.
            return_visible_points: If True, depths of visible points will be loaded and included in each datum.
        """
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._dataset_path = dataset_path

        sfm_scene: SfmScene
        base_cache: DatasetCache
        sfm_scene, base_cache = load_colmap_scene(dataset_path=dataset_path)

        self._transform = transform
        self._sfm_scene, self._cache = self._transform(sfm_scene, base_cache)

        self._test_every = test_every
        self._split: Literal["train", "test", "all"] = split
        self.patch_size = patch_size
        self._return_visible_points = return_visible_points

        # If you specified image indices, we'll filter the dataset to only include those images.
        indices = np.arange(self._sfm_scene.num_images) if image_indices is None else np.array(image_indices)
        if self._split == "train":
            self._indices = indices[indices % self._test_every != 0]
        elif self._split == "test":
            self._indices = indices[indices % self._test_every == 0]
        elif self._split == "all":
            self._indices = indices
        else:
            raise ValueError(f"Split must be one of 'train', 'test', or 'all'. Got {self._split}.")

    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state of the dataset as a dictionary.

        This is useful for saving the dataset state to disk or for debugging purposes.

        Returns:
            Dict[str, Any]: A dictionary containing the dataset state.
        """
        return {
            "dataset_path": str(self._dataset_path),
            "test_every": self._test_every,
            "split": self._split,
            "transform": self._transform.state_dict(),
            "image_indices": self._indices.tolist(),
            "patch_size": self.patch_size,
            "return_visible_points": self._return_visible_points,
        }

    @staticmethod
    def from_state_dict(state_dict: Dict[str, Any], map_path: pathlib.Path | None = None) -> "SfmDataset":
        """
        Create a new SfmDataset instance from a state dictionary.

        Args:
            state_dict: A dictionary containing the dataset state.
            map_path: Optional path to the dataset directory. If provided, this will override the path in the state_dict.

        Returns:
            SfmDataset: A new SfmDataset instance with the state loaded from the dictionary.
        """
        dataset_path = map_path if map_path is not None else pathlib.Path(state_dict["dataset_path"])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        dataset = SfmDataset(
            dataset_path=map_path if map_path is not None else pathlib.Path(state_dict["dataset_path"]),
            test_every=state_dict["test_every"],
            split=state_dict["split"],
            transform=BaseTransform.from_state_dict(state_dict["transform"]),
            image_indices=state_dict["image_indices"],
            patch_size=state_dict["patch_size"],
            return_visible_points=state_dict["return_visible_points"],
        )
        return dataset

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
    def transform(self) -> BaseTransform:
        """
        Get the transform applied to the dataset.

        This is useful if you want to access the transform directly or modify it.

        Returns:
            BaseTransform: The transform applied to the dataset.
        """
        return self._transform

    @property
    def split(self) -> Literal["train", "test", "all"]:
        """
        Get the split of the dataset (train, test, or all).

        This is useful for understanding how the dataset is split and can be used to filter images based on the split.

        Returns:
            Literal["train", "test", "all"]: The split of the dataset.
        """
        return self._split

    @property
    def cache(self) -> DatasetCache:
        """
        Get the image dataset cache for this dataset.
        This is useful if you're building new properties for the dataset or want to access the cache directly.

        Returns:
            DatasetCache: The image dataset cache for this dataset.
        """
        return self._cache

    @property
    def camera_to_world_matrices(self) -> np.ndarray:
        """
        Get the camera to world matrices for all images in the dataset.

        This returns the camera to world matrices as a numpy array of shape (N, 4, 4) where N is the number of images.

        Returns:
            np.ndarray: An Nx4x4 array of camera to world matrices for the cameras in the dataset.
        """
        return np.stack([self[i]["camera_to_world"].numpy() for i in range(len(self))], axis=0)

    @property
    def projection_matrices(self) -> np.ndarray:
        """
        Get the projection matrices mapping camera to pixel coordinates for all images in the dataset.

        This returns the undistorted projection matrices as a numpy array of shape (N, 3, 3) where N is the number of images.

        Returns:
            np.ndarray: An Nx3x3 array of projection matrices for the cameras in the dataset.
        """
        return np.stack([self[i]["projection"].numpy() for i in range(len(self))], axis=0)

    @property
    def image_sizes(self) -> np.ndarray:
        """
        Get the image sizes for all images in the dataset.

        This returns the image sizes as a numpy array of shape (N, 2) where N is the number of images.
        Each row contains the height and width of the corresponding image.

        Returns:
            np.ndarray: An Nx2 array of image sizes for the cameras in the dataset.
        """
        return np.array([self[i]["image"].shape[:2] for i in range(len(self))], dtype=np.int32)

    @property
    def scene_scale(self) -> float:
        """
        Get the scale of the scene defined as the maximum distance of any image pose origin from the
        median of all image pose origins after normalization.

        This is useful for understanding the scale of the scene and can be used for scaling the points
        in the scene to a unit sphere or other normalization.

        Returns:
            float: The scale of the scene.
        """
        return self._sfm_scene.scene_scale

    @property
    def points(self) -> np.ndarray:
        """
        Get the 3D points in the scene.
        This returns the points in world coordinates as a numpy array of shape (N, 3) where N is the number of points.

        Returns:
            np.ndarray: An Nx3 array of 3D points in the scene.
        """
        return self._sfm_scene.points

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

        image = imageio.imread(image_meta.image_path)[..., :3]
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
            mask = imageio.imread(image_meta.mask_path)
            mask = mask < 127

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

if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    dataset = SfmDataset(
        dataset_path=args.data_dir,
        test_every=8,
        split="train",
        return_visible_points=True,
    )
    print(f"Dataset: {len(dataset)} images.")

    imsize = None
    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):  # type: ignore
        image = data["image"].numpy().astype(np.uint8)
        # Make sure all images we write are the same size. We use the first image to determine the size of the video.
        # This is done because some images have slightly different sizes due to undistortion.
        imsize = image.shape if imsize is None else imsize
        if image.shape != imsize:
            new_image = np.zeros(imsize, dtype=np.uint8)
            new_image[: image.shape[0], : image.shape[1]] = image[: imsize[0], : imsize[1]]
            image = new_image
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:  # type: ignore
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
