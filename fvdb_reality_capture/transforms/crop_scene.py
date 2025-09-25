# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
from typing import Literal, Sequence

import cv2
import numpy as np
import tqdm
from scipy.spatial import ConvexHull

from ..sfm_scene import SfmCache, SfmImageMetadata, SfmScene
from .base_transform import BaseTransform, transform


@transform
class CropScene(BaseTransform):
    """
    Crop the scene points to a specified bounding box and update masks to nullify
    pixels corresponding to regions outside the cropped scene.
    """

    version = "1.0.0"

    def __init__(
        self,
        bbox: Sequence[float] | np.ndarray,
        mask_format: Literal["png", "jpg", "npy"] = "png",
        composite_with_existing_masks: bool = True,
    ):
        """
        Initialize the Crop transform with a bounding box.

        Args:
            bbox (tuple): A tuple defining the bounding box in the format (min_x, min_y, min_z, max_x, max_y, max_z).
            mask_format (Literal["png", "jpg", "npy"]): The format to save the masks in. Defaults to "png".
            composite_with_existing_masks (bool): Whether to composite the masks generated into existing masks for
                pixels corresponding to regions outside the cropped scene. If set to True, existing masks
                will be loaded and composited with the new mask. Defaults to True.
        """
        super().__init__()
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        if not len(bbox) == 6:
            raise ValueError("Bounding box must be a tuple of the form (min_x, min_y, min_z, max_x, max_y, max_z).")
        self._bbox = np.asarray(bbox).astype(np.float32)
        self._mask_format = mask_format
        if self._mask_format not in ["png", "jpg", "npy"]:
            raise ValueError(
                f"Unsupported mask format: {self._mask_format}. Supported formats are 'png', 'jpg', and 'npy'."
            )
        self._composite_with_existing_masks = composite_with_existing_masks

    @staticmethod
    def name() -> str:
        """
        Return the name of the transform.

        Returns:
            str: The name of the transform.
        """
        return "Crop"

    @staticmethod
    def from_state_dict(state_dict: dict) -> "CropScene":
        """
        Create a Crop transform from a state dictionary.

        Args:
            state (dict): The state dictionary containing the bounding box.

        Returns:
            Crop: An instance of the Crop transform.
        """
        bbox = state_dict.get("bbox", None)
        if bbox is None:
            raise ValueError("State dictionary must contain 'bbox' key with bounding box coordinates.")
        if not isinstance(bbox, np.ndarray) or len(bbox) != 6:
            raise ValueError(
                "Bounding box must be a tuple or array of the form (min_x, min_y, min_z, max_x, max_y, max_z)."
            )
        return CropScene(bbox)

    def state_dict(self) -> dict:
        """
        Return the state dictionary of the Crop transform.

        Returns:
            dict: A dictionary containing the bounding box.
        """
        return {
            "name": self.name(),
            "version": self.version,
            "bbox": self._bbox,
            "mask_format": self._mask_format,
            "composite_into_existing_masks": self._composite_with_existing_masks,
        }

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Apply the cropping transform to the scene.

        Args:
            scene (SfmScene): The scene to be cropped.

        Returns:
            SfmScene: The cropped scene.
        """
        # Ensure the bounding box is a numpy array of length 6
        bbox = np.asarray(self._bbox, dtype=np.float32)
        if bbox.shape != (6,):
            raise ValueError("Bounding box must be a 1D array of shape (6,)")

        self._logger.info(f"Cropping scene to bounding box: {self._bbox}")

        input_cache: SfmCache = input_scene.cache

        output_cache_prefix = f"{self.name()}_{self._bbox[0]}_{self._bbox[1]}_{self._bbox[2]}_{self._bbox[3]}_{self._bbox[4]}_{self._bbox[5]}_{self._mask_format}_{self._composite_with_existing_masks}"
        output_cache_prefix = output_cache_prefix.replace(" ", "_")  # Ensure no spaces in the cache prefix
        output_cache_prefix = output_cache_prefix.replace(".", "_")  # Ensure no dots in the cache prefix
        output_cache_prefix = output_cache_prefix.replace("-", "neg")  # Ensure no dashes in the cache prefix
        output_cache = input_cache.make_folder(
            output_cache_prefix,
            description=f"Image masks ({self._mask_format}) for cropping to bounding box {self._bbox}",
        )

        # Create a mask over all the points which are inside the bounding box
        points_mask = np.logical_and.reduce(
            [
                input_scene.points[:, 0] > bbox[0],
                input_scene.points[:, 0] < bbox[3],
                input_scene.points[:, 1] > bbox[1],
                input_scene.points[:, 1] < bbox[4],
                input_scene.points[:, 2] > bbox[2],
                input_scene.points[:, 2] < bbox[5],
            ]
        )

        # Mask the scene using the points mask
        masked_scene = input_scene.filter_points(points_mask)

        # How many zeros to pad the image index in the mask file names
        num_zeropad = len(str(len(masked_scene.images))) + 2

        new_image_metadata = []

        regenerate_cache = False
        if output_cache.num_files != len(masked_scene.images) + 1:
            if output_cache.num_files == 0:
                self._logger.info(f"No masks found in the cache for cropping.")
            else:
                self._logger.info(
                    f"Inconsistent number of masks for images. Expected {len(masked_scene.images)}, found {output_cache.num_files}. "
                    f"Clearing cache and regenerating masks."
                )
            output_cache.clear_current_folder()
            regenerate_cache = True
        if output_cache.has_file("transform"):
            _, transform_data = output_cache.read_file("transform")
            cached_transform: np.ndarray | None = transform_data.get("transform", None)
            if cached_transform is None:
                self._logger.info(
                    f"Transform metadata does not match expected format. No 'transform' key in cached file."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
            elif not isinstance(cached_transform, np.ndarray) or cached_transform.shape != (4, 4):
                self._logger.info(
                    f"Transform metadata does not match expected format. Expected 'transform'."
                    f"Clearing the cache and regenerating transform."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
            elif not np.allclose(cached_transform, input_scene.transformation_matrix):
                self._logger.info(
                    f"Cached transform does not match input scene transform. Clearing the cache and regenerating transform."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
        else:
            self._logger.info("No transform found in cache, regenerating.")
            output_cache.clear_current_folder()
            regenerate_cache = True

        for image_id in range(len(masked_scene.images)):
            if regenerate_cache:
                break
            image_cache_filename = f"mask_{image_id:0{num_zeropad}}"
            image_meta = masked_scene.images[image_id]
            if not output_cache.has_file(image_cache_filename):
                self._logger.info(
                    f"Mask for image {image_id} not found in cache. Clearing cache and regenerating masks."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

            key_meta = output_cache.get_file_metadata(image_cache_filename)
            if key_meta.get("data_type", "") != self._mask_format:
                self._logger.info(
                    f"Output cache masks metadata does not match expected format. Expected '{self._mask_format}'."
                    f"Clearing the cache and regenerating masks."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break
            new_image_metadata.append(
                SfmImageMetadata(
                    world_to_camera_matrix=image_meta.world_to_camera_matrix,
                    camera_to_world_matrix=image_meta.camera_to_world_matrix,
                    camera_metadata=image_meta.camera_metadata,
                    camera_id=image_meta.camera_id,
                    image_id=image_meta.image_id,
                    image_path=image_meta.image_path,
                    mask_path=str(key_meta["path"]),
                    point_indices=image_meta.point_indices,
                )
            )

        if regenerate_cache:
            output_cache.write_file("transform", {"transform": input_scene.transformation_matrix}, data_type="pt")
            self._logger.info(f"Computing image masks for cropping and saving to cache.")
            new_image_metadata = []

            min_x, min_y, min_z, max_x, max_y, max_z = bbox

            # (8, 4)-shaped array representing the corners of the bounding cube containing the input points
            # in homogeneous coordinates
            cube_bounds_world_space_homogeneous = np.array(
                [
                    [min_x, min_y, min_z, 1.0],
                    [min_x, min_y, max_z, 1.0],
                    [min_x, max_y, min_z, 1.0],
                    [min_x, max_y, max_z, 1.0],
                    [max_x, min_y, min_z, 1.0],
                    [max_x, min_y, max_z, 1.0],
                    [max_x, max_y, min_z, 1.0],
                    [max_x, max_y, max_z, 1.0],
                ]
            )

            for image_meta in tqdm.tqdm(masked_scene.images, unit="imgs", desc="Computing image masks for cropping"):
                cam_meta = image_meta.camera_metadata

                # Transform the cube corners to camera space
                cube_bounds_cam_space = (
                    image_meta.world_to_camera_matrix @ cube_bounds_world_space_homogeneous.T
                )  # [4, 8]
                # Divide out the homogeneous coordinate -> [3, 8]
                cube_bounds_cam_space = cube_bounds_cam_space[:3, :] / cube_bounds_cam_space[-1, :]

                # Project the camera-space cube corners into image space [3, 3] * [8, 3] - > [8, 2]
                cube_bounds_pixel_space = cam_meta.projection_matrix @ cube_bounds_cam_space  # [3, 8]
                # Divide out the homogeneous coordinate and transpose -> [8, 2]
                cube_bounds_pixel_space = (cube_bounds_pixel_space[:2, :] / cube_bounds_pixel_space[2, :]).T

                # Compute the pixel-space convex hull of the cube corners
                convex_hull = ConvexHull(cube_bounds_pixel_space)
                # Each face of the convex hull is defined by a normal vector and an offset
                # These define a set of half spaces. We're going to check that we're on the inside of all of them
                # to determine if a pixel is inside the convex hull
                hull_normals = convex_hull.equations[:, :-1]  # [num_faces, 2]
                hull_offsets = convex_hull.equations[:, -1]  # [n_faces]

                # Generate a grid of pixel (u, v) coordinates of shape [image_height, image_width, 2]
                image_width = image_meta.camera_metadata.width
                image_height = image_meta.camera_metadata.height
                pixel_u, pixel_v = np.meshgrid(np.arange(image_width), np.arange(image_height), indexing="xy")
                pixel_coords = np.stack([pixel_u, pixel_v], axis=-1)  # [image_height, image_width, 2]

                # Shift and take the dot product between each pixel coordinate and the hull half-space normals
                # to get the shortest signed distance to each face of the convex hull
                # This produces an (image_height, image_width, num_faces)-shaped array
                # where each pixel has a signed distance to each face of the convex hull
                pixel_to_half_space_signed_distances = (
                    pixel_coords @ hull_normals.T + hull_offsets[np.newaxis, np.newaxis, :]
                )

                # A pixel lies inside the hull if it's signed distance to all faces is less than or equal to zero
                # This produces a boolean mask of shape [image_height, image_width]
                # where True indicates the pixel is inside the hull
                inside_mask = np.all(
                    pixel_to_half_space_signed_distances <= 0.0, axis=-1
                )  # [image_height, image_width]

                # If the mask already exists, load it and composite this one into it
                mask_to_save = inside_mask.astype(np.uint8) * 255  # Convert to uint8 mask
                if os.path.exists(image_meta.mask_path) and self._composite_with_existing_masks:
                    if image_meta.mask_path.strip().endswith(".npy"):
                        existing_mask = np.load(image_meta.mask_path)
                    elif image_meta.mask_path.strip().endswith(".png"):
                        existing_mask = cv2.imread(image_meta.mask_path, cv2.IMREAD_GRAYSCALE)
                        assert existing_mask is not None, f"Failed to load mask {image_meta.mask_path}"
                    elif image_meta.mask_path.strip().endswith(".jpg"):
                        existing_mask = cv2.imread(image_meta.mask_path, cv2.IMREAD_GRAYSCALE)
                        assert existing_mask is not None, f"Failed to load mask {image_meta.mask_path}"
                    else:
                        raise ValueError(f"Unsupported mask file format: {image_meta.mask_path}")
                    if existing_mask.ndim == 3:
                        # Ensure the mask is 3D to match the input mask
                        inside_mask = inside_mask[..., np.newaxis]
                    elif existing_mask.ndim != 2:
                        raise ValueError(f"Unsupported mask shape: {existing_mask.shape}. Must have 2D or 3D shape.")

                    if existing_mask.shape[:2] != inside_mask.shape[:2]:
                        raise ValueError(
                            f"Existing mask shape {existing_mask.shape[:2]} does not match computed mask shape {inside_mask.shape[:2]}."
                        )
                    mask_to_save = existing_mask * inside_mask

                cache_file_meta = output_cache.write_file(
                    name=f"mask_{image_meta.image_id:0{num_zeropad}}",
                    data=mask_to_save,
                    data_type=self._mask_format,
                )

                new_image_metadata.append(
                    SfmImageMetadata(
                        world_to_camera_matrix=image_meta.world_to_camera_matrix,
                        camera_to_world_matrix=image_meta.camera_to_world_matrix,
                        camera_metadata=image_meta.camera_metadata,
                        camera_id=image_meta.camera_id,
                        image_id=image_meta.image_id,
                        image_path=image_meta.image_path,
                        mask_path=str(cache_file_meta["path"]),
                        point_indices=image_meta.point_indices,
                    )
                )

        output_scene = SfmScene(
            cameras=masked_scene.cameras,
            images=new_image_metadata,
            points=masked_scene.points,
            points_rgb=masked_scene.points_rgb,
            points_err=masked_scene.points_err,
            scene_bbox=bbox,
            transformation_matrix=input_scene.transformation_matrix,
            cache=output_cache,
        )

        return output_scene
