# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
from typing import Literal, Sequence

import imageio
import numpy as np
import tqdm
from scipy.spatial import ConvexHull

from ..image_dataset_cache import ImageDatasetCache
from ..sfm_scene import SfmImageMetadata, SfmScene
from .base_transform import BaseTransform
from .transform_registry import transform


@transform
class CropScene(BaseTransform):
    """
    Crop the scene to a specified bounding box.
    """

    version = "1.0.0"

    def __init__(
        self,
        bbox: Sequence[float] | np.ndarray,
        mask_format: Literal["png", "jpg", "npy"] = "png",
        overwrite_existing_masks: bool = False,
    ):
        """
        Initialize the Crop transform with a bounding box.

        Args:
            bbox (tuple): A tuple defining the bounding box in the format (min_x, min_y, min_z, max_x, max_y, max_z).
            mask_format (Literal["png", "jpg", "npy"]): The format to save the masks in. Defaults to "png".
            overwrite_existing_masks (bool): Whether to overwrite existing masks. If set to False, existing masks
                will be loaded and composited with the new mask. Defaults to False.
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
        self._overwrite_existing_masks = overwrite_existing_masks

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
            "overwrite_existing_masks": self._overwrite_existing_masks,
        }

    def __call__(self, input_scene: SfmScene, input_cache: ImageDatasetCache) -> tuple[SfmScene, ImageDatasetCache]:
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

        output_cache_prefix = f"{self.name()}_{self._bbox[0]}_{self._bbox[1]}_{self._bbox[2]}_{self._bbox[3]}_{self._bbox[4]}_{self._bbox[5]}_{self._mask_format}_{self._overwrite_existing_masks}"
        output_cache_prefix = output_cache_prefix.replace(" ", "_")  # Ensure no spaces in the cache prefix
        output_cache_prefix = output_cache_prefix.replace(".", "_")  # Ensure no dots in the cache prefix
        output_cache_prefix = output_cache_prefix.replace("-", "neg")  # Ensure no dashes in the cache prefix
        output_cache = input_cache.get_subcache(output_cache_prefix)

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

        new_image_metadata = []
        if "masks" in output_cache:
            key_meta, value_meta = output_cache.get_property_metadata("masks")
            if key_meta.get("scope", "") != "image" or key_meta.get("data_type", "") != self._mask_format:
                self._logger.warning(
                    f"Output cache masks metadata does not match expected format. Expected scope 'image' and data type '{self._mask_format}'."
                    f"Deleting the key and regenerating masks."
                )
                output_cache.delete_property("masks")
            if key_meta.get("scope", "") == "image":
                if output_cache.num_values_for_image_property("masks") != len(masked_scene.images):
                    self._logger.info(
                        f"Inconsistent number of masks for images. Deleting the key and regenerating masks."
                    )
                    output_cache.delete_property("masks")

        if "masks" not in output_cache:
            self._logger.info(f"Computing image masks for cropping.")

            output_cache.register_image_property(
                "masks",
                data_type=self._mask_format,
                description=f"Image masks for cropping to bounding box {self._bbox}",
            )

            # Compute the bounding box of the masked points. We're going to use these to compute masks for the images images
            min_x, min_y, min_z, max_x, max_y, max_z = np.concatenate(
                (np.min(masked_scene.points, axis=0), np.max(masked_scene.points, axis=0))
            ).tolist()

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
                if os.path.exists(image_meta.mask_path) and not self._overwrite_existing_masks:
                    if image_meta.mask_path.strip().endswith(".npy"):
                        existing_mask = np.load(image_meta.mask_path)
                    elif image_meta.mask_path.strip().endswith(".png"):
                        existing_mask = imageio.imread(image_meta.mask_path)
                    elif image_meta.mask_path.strip().endswith(".jpg"):
                        existing_mask = imageio.imread(image_meta.mask_path)
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

                output_cache.set_image_property(
                    "masks",
                    image_meta.image_id,
                    mask_to_save,
                )
                crop_mask_path = output_cache.get_image_property_path("masks", image_meta.image_id)

                new_image_metadata.append(
                    SfmImageMetadata(
                        world_to_camera_matrix=image_meta.world_to_camera_matrix,
                        camera_to_world_matrix=image_meta.camera_to_world_matrix,
                        camera_metadata=image_meta.camera_metadata,
                        camera_id=image_meta.camera_id,
                        image_id=image_meta.image_id,
                        image_path=image_meta.image_path,
                        mask_path=str(crop_mask_path),
                        point_indices=image_meta.point_indices,
                    )
                )

        else:
            self._logger.info(f"Using cached image masks for cropping.")
            new_image_metadata = [
                SfmImageMetadata(
                    world_to_camera_matrix=image_meta.world_to_camera_matrix,
                    camera_to_world_matrix=image_meta.camera_to_world_matrix,
                    camera_metadata=image_meta.camera_metadata,
                    camera_id=image_meta.camera_id,
                    image_id=image_meta.image_id,
                    image_path=image_meta.image_path,
                    mask_path=str(output_cache.get_image_property_path("masks", image_meta.image_id)),
                    point_indices=image_meta.point_indices,
                )
                for image_meta in masked_scene.images
            ]

        output_scene = SfmScene(
            cameras=masked_scene.cameras,
            images=new_image_metadata,
            points=masked_scene.points,
            points_rgb=masked_scene.points_rgb,
            points_err=masked_scene.points_err,
        )

        return output_scene, output_cache
