# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
from typing import Literal

import cv2
import numpy as np
import torch
import tqdm
from fvdb.types import NumericMaxRank1, to_VecNf
from scipy.spatial import ConvexHull

from fvdb_reality_capture.sfm_scene import SfmCache, SfmPosedImageMetadata, SfmScene

from .base_transform import BaseTransform, transform


def _crop_scene_to_bbox(
    input_scene: SfmScene,
    transform_name: str,
    composite_with_existing_masks: bool,
    mask_format: str,
    bbox: np.ndarray,
    logger: logging.Logger,
):
    if bbox.shape != (6,):
        raise ValueError("Bounding box must be a 1D array of shape (6,)")

    output_cache_prefix = f"{transform_name}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{bbox[4]}_{bbox[5]}_{mask_format}_{composite_with_existing_masks}"
    output_cache_prefix = output_cache_prefix.replace(" ", "_")  # Ensure no spaces in the cache prefix
    output_cache_prefix = output_cache_prefix.replace(".", "_")  # Ensure no dots in the cache prefix
    output_cache_prefix = output_cache_prefix.replace("-", "neg")  # Ensure no dashes in the cache prefix

    input_cache: SfmCache = input_scene.cache

    output_cache = input_cache.make_folder(
        output_cache_prefix,
        description=f"Image masks ({mask_format}) for cropping to bounding box {bbox}",
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
            logger.info(f"No masks found in the cache for cropping.")
        else:
            logger.info(
                f"Inconsistent number of masks for images. Expected {len(masked_scene.images)}, found {output_cache.num_files}. "
                f"Clearing cache and regenerating masks."
            )
        output_cache.clear_current_folder()
        regenerate_cache = True
    if output_cache.has_file("transform"):
        _, transform_data = output_cache.read_file("transform")
        cached_transform: np.ndarray | None = transform_data.get("transform", None)
        if cached_transform is None:
            logger.info(f"Transform metadata does not match expected format. No 'transform' key in cached file.")
            output_cache.clear_current_folder()
            regenerate_cache = True
        elif not isinstance(cached_transform, np.ndarray) or cached_transform.shape != (4, 4):
            logger.info(
                f"Transform metadata does not match expected format. Expected 'transform'."
                f"Clearing the cache and regenerating transform."
            )
            output_cache.clear_current_folder()
            regenerate_cache = True
        elif not np.allclose(cached_transform, input_scene.transformation_matrix):
            logger.info(
                f"Cached transform does not match input scene transform. Clearing the cache and regenerating transform."
            )
            output_cache.clear_current_folder()
            regenerate_cache = True
    else:
        logger.info("No transform found in cache, regenerating.")
        output_cache.clear_current_folder()
        regenerate_cache = True

    for image_id in range(len(masked_scene.images)):
        if regenerate_cache:
            break
        image_cache_filename = f"mask_{image_id:0{num_zeropad}}"
        image_meta = masked_scene.images[image_id]
        if not output_cache.has_file(image_cache_filename):
            logger.info(f"Mask for image {image_id} not found in cache. Clearing cache and regenerating masks.")
            output_cache.clear_current_folder()
            regenerate_cache = True
            break

        key_meta = output_cache.get_file_metadata(image_cache_filename)
        if key_meta.get("data_type", "") != mask_format:
            logger.info(
                f"Output cache masks metadata does not match expected format. Expected '{mask_format}'."
                f"Clearing the cache and regenerating masks."
            )
            output_cache.clear_current_folder()
            regenerate_cache = True
            break
        new_image_metadata.append(
            SfmPosedImageMetadata(
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
        logger.info(f"Computing image masks for cropping and saving to cache.")
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
            cube_bounds_cam_space = image_meta.world_to_camera_matrix @ cube_bounds_world_space_homogeneous.T  # [4, 8]
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
            inside_mask = np.all(pixel_to_half_space_signed_distances <= 0.0, axis=-1)  # [image_height, image_width]

            # If the mask already exists, load it and composite this one into it
            mask_to_save = inside_mask.astype(np.uint8) * 255  # Convert to uint8 mask
            if os.path.exists(image_meta.mask_path) and composite_with_existing_masks:
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
                data_type=mask_format,
            )

            new_image_metadata.append(
                SfmPosedImageMetadata(
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


@transform
class CropScene(BaseTransform):
    """
    A :class:`~base_transform.BaseTransform` which crops the input
    :class:`~fvdb_reality_capture.sfm_scene.SfmScene` points to lie within a specified bounding box.
    This transform additionally and updates the scene's masks to nullify pixels whose rays do not intersect
    the bounding box.

    .. note::

        If the input scene already has masks, these new masks will be composited with the existing masks to ensure that
        pixels outside the cropped region are properly masked. This can be disabled by setting
        ``composite_with_existing_masks`` to ``False``.

    Example usage:

    .. code-block:: python

        # Example usage:
        from fvdb_reality_capture import transforms
        from fvdb_reality_capture.sfm_scene import SfmScene
        import numpy as np

        # Bounding box in the format (min_x, min_y, min_z, max_x, max_y, max_z)
        scene_transform = transforms.CropScene(bbox=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))

        input_scene: SfmScene = ...  # Load or create an SfmScene

        # The transformed scene will have points only within the bounding box, and posed images will have
        # masks updated to nullify pixels corresponding to regions outside the cropped scene.
        transformed_scene: SfmScene = scene_transform(input_scene)

    """

    version = "1.0.0"

    def __init__(
        self,
        bbox: NumericMaxRank1,
        mask_format: Literal["png", "jpg", "npy"] = "png",
        composite_with_existing_masks: bool = True,
    ):
        """
        Create a new :class:`CropScene` transform with a bounding box.

        Args:
            bbox (NumericMaxRank1): A bounding box in the format ``(min_x, min_y, min_z, max_x, max_y, max_z)``.
            mask_format (Literal["png", "jpg", "npy"]): The format to save the masks in. Defaults to "png".
            composite_with_existing_masks (bool): Whether to composite the masks generated into existing masks for
                pixels corresponding to regions outside the cropped scene. If set to ``True``, existing masks
                will be loaded and composited with the new mask. Defaults to ``True``. The resulting composited
                mask will allow a pixel to be valid if it is valid in both the existing and new mask.
        """
        super().__init__()
        bbox = to_VecNf(bbox, 6, dtype=torch.float64).numpy()
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
        Return the name of the :class:`CropScene` transform. **i.e.** ``"CropScene"``.

        Returns:
            str: The name of the :class:`CropScene` transform. **i.e.** ``"CropScene"``.
        """
        return "CropScene"

    @staticmethod
    def from_state_dict(state_dict: dict) -> "CropScene":
        """
        Create a :class:`CropScene` transform from a state dictionary created with :meth:`state_dict`.

        Args:
            state_dict (dict): The state dictionary for the transform.

        Returns:
            transform (CropScene): An instance of the :class:`CropScene` transform.
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
        Return the state of the :class:`CropScene` transform for serialization.

        You can use this state dictionary to recreate the transform using :meth:`from_state_dict`.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
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
        Return a new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` with points cropped to lie within the bounding
        box specified at initialization, and with masks updated to nullify pixels whose rays do not intersect the
        bounding box.

        Args:
            input_scene (SfmScene): The scene to be cropped.

        Returns:
            output_scene (SfmScene): The cropped scene.
        """
        # Ensure the bounding box is a numpy array of length 6
        bbox = np.asarray(self._bbox, dtype=np.float32)
        if bbox.shape != (6,):
            raise ValueError("Bounding box must be a 1D array of shape (6,)")

        self._logger.info(f"Cropping scene to bounding box: {self._bbox}")

        return _crop_scene_to_bbox(
            input_scene=input_scene,
            transform_name=self.name(),
            composite_with_existing_masks=self._composite_with_existing_masks,
            mask_format=self._mask_format,
            bbox=bbox,
            logger=self._logger,
        )


@transform
class CropSceneToPoints(BaseTransform):
    """
    A :class:`~base_transform.BaseTransform` which crops the input
    :class:`~fvdb_reality_capture.sfm_scene.SfmScene` points to lie within the bounding box around its points plus
    or minus a padding margin. This transform additionally and updates the scene's masks to nullify pixels whose rays
    do not intersect the bounding box.

    .. note::

        If the input scene already has masks, these new masks will be composited with the existing masks to ensure that
        pixels outside the cropped region are properly masked. This can be disabled by setting
        ``composite_with_existing_masks`` to ``False``.

    .. note::

        You may want to use this over :class:`CropScene` if you want the bounding box to depend on the input scene
        points rather than being fixed (*e.g.* if you don't know the bounding box ahead of time). This transform
        is also useful if you just want to apply conservative masking to the input scene based on its points.

    .. note::

        The margin is specified as a fraction of the bounding box size. For example, a margin of 0.1 will expand the
        bounding box by 10% (5% in all directions). So if the scene's bounding box is ``(0, 0, 0)`` to ``(1, 1, 1)``,
        a margin of ``0.1`` will result in a bounding box of ``(-0.05, -0.05, -0.05)`` to ``(1.05, 1.05, 1.05)``.
        The margin can also be negative to shrink the bounding box.

    Example usage:

    .. code-block:: python

        # Example usage:
        from fvdb_reality_capture import transforms
        from fvdb_reality_capture.sfm_scene import SfmScene
        import numpy as np

        # Crop the scene to be 0.1 times smaller than the bounding box around its points
        # (i.e. a margin of -0.1)
        scene_transform = transforms.CropSceneToPoints(margin=-0.1)

        input_scene: SfmScene = ...  # Load or create an SfmScene

        # The transformed scene will have points only within the bounding box of its points
        # minus a factor of 0.1 times the size. (i.e. a margin of -0.1).
        # Posed images will have masks updated to nullify pixels corresponding to regions outside the cropped scene.
        transformed_scene: SfmScene = scene_transform(input_scene)

    """

    version = "1.0.0"

    def __init__(
        self,
        margin: float = 0.0,
        mask_format: Literal["png", "jpg", "npy"] = "png",
        composite_with_existing_masks: bool = True,
    ):
        """
        Create a new :class:`CropSceneToPoints` transform with the given margin.

        Args:
            margin (float): The margin factor to apply around the bounding box of the points.
                Can be negative to shrink the bounding box. This is a fraction of the bounding box size.
                For example, a margin of ``0.1`` will expand the bounding box by 10% (5% in all directions),
                while a margin of ``-0.1`` will shrink the bounding box by 10% (-5% in all directions).
                Defaults to ``0.0``.
            mask_format (Literal["png", "jpg", "npy"]): The format to save the masks in. Defaults to "png".
            composite_with_existing_masks (bool): Whether to composite the masks generated into existing masks for
                pixels corresponding to regions outside the cropped scene. If set to True, existing masks
                will be loaded and composited with the new mask. Defaults to True.
        """
        super().__init__()
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._margin = margin

        self._mask_format = mask_format
        if self._mask_format not in ["png", "jpg", "npy"]:
            raise ValueError(
                f"Unsupported mask format: {self._mask_format}. Supported formats are 'png', 'jpg', and 'npy'."
            )
        self._composite_with_existing_masks = composite_with_existing_masks

    @staticmethod
    def name() -> str:
        """
        Return the name of the :class:`CropSceneToPoints` transform. *i.e.* ``"CropSceneToPoints"``.

        Returns:
            str: The name of the :class:`CropSceneToPoints` transform. *i.e.* ``"CropSceneToPoints"``.
        """
        return "CropSceneToPoints"

    @staticmethod
    def from_state_dict(state_dict: dict) -> "CropSceneToPoints":
        """
        Create a :class:`CropSceneToPoints` transform from a state dictionary generated with :meth:`state_dict`.

        Args:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.

        Returns:
            transform (:class:`CropSceneToPoints`): An instance of the :class:`CropSceneToPoints` transform loaded
                from the state dictionary.
        """
        margin = state_dict.get("margin", None)
        if margin is None:
            raise ValueError("State dictionary must contain 'margin' key with margin value.")
        if not isinstance(margin, (float, int)):
            raise ValueError("Margin must be a non-negative float.")

        mask_format = state_dict.get("mask_format", None)
        if mask_format is None:
            raise ValueError("State dictionary must contain 'mask_format' key with mask format value.")
        if mask_format is not None and mask_format not in ["png", "jpg", "npy"]:
            raise ValueError(f"Unsupported mask format: {mask_format}. Supported formats are 'png', 'jpg', and 'npy'.")

        composite_into_existing_masks = state_dict.get("composite_into_existing_masks", None)
        if composite_into_existing_masks is None:
            raise ValueError("State dictionary must contain 'composite_into_existing_masks' key with boolean value.")
        if not isinstance(composite_into_existing_masks, bool):
            raise ValueError("composite_into_existing_masks must be a boolean.")
        return CropSceneToPoints(
            margin=margin, mask_format=mask_format, composite_with_existing_masks=composite_into_existing_masks
        )

    def state_dict(self) -> dict:
        """
        Return the state of the :class:`CropSceneToPoints` transform for serialization.

        You can use this state dictionary to recreate the transform using :meth:`from_state_dict`.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {
            "name": self.name(),
            "version": self.version,
            "margin": self._margin,
            "mask_format": self._mask_format,
            "composite_into_existing_masks": self._composite_with_existing_masks,
        }

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Return a new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` with points cropped to lie within the
        bounding box of the input scene's points plus or minus the margin specified at initialization,
        and with masks updated to nullify pixels whose rays do not intersect the bounding box.

        Args:
            input_scene (SfmScene): The scene to be cropped.

        Returns:
            output_scene (SfmScene): The cropped scene.
        """
        points_min = input_scene.points.min(axis=0)
        points_max = input_scene.points.max(axis=0)
        box_size = points_max - points_min
        padding = self._margin * box_size / 0.5
        points_min -= padding
        points_max += padding
        bbox = np.array(
            [
                points_min[0],
                points_min[1],
                points_min[2],
                points_max[0],
                points_max[1],
                points_max[2],
            ],
            dtype=np.float32,
        )

        if bbox.shape != (6,):
            raise ValueError("Bounding box must be a 1D array of shape (6,)")

        self._logger.info(f"Cropping scene to point bounding box {bbox} using margin {self._margin}")

        return _crop_scene_to_bbox(
            input_scene=input_scene,
            transform_name=self.name(),
            composite_with_existing_masks=self._composite_with_existing_masks,
            mask_format=self._mask_format,
            bbox=bbox,
            logger=self._logger,
        )
