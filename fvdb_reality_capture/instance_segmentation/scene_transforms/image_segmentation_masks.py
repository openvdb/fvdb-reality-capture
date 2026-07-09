# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import hashlib
import logging
from typing import Any, Literal

import cv2
import numpy as np
import torch
import tqdm
from fvdb import CameraModel, GaussianSplat3d
from fvdb_reality_capture.foundation_models import SAM2Model
from fvdb_reality_capture.sfm_scene import SfmCache, SfmScene
from fvdb_reality_capture.transforms import BaseTransform, transform

from ..scene_attribute import (
    GARFVDB_MASK_ATTRIBUTE_NAME,
    GARFVDB_MASK_DATA_SCHEMA_VERSION,
    GARfVDBMaskAttribute,
)


@transform
class GenerateGARfVDBMasks(BaseTransform):
    """
    A transform that uses SAM2 to compute segmentation masks for each image.
    """

    version = "1.0.0"

    def __init__(
        self,
        gs3d: GaussianSplat3d | None = None,
        checkpoint: Literal["large", "small", "tiny", "base_plus"] = "large",
        points_per_side=40,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.80,
        device: torch.device | str = "cuda",
        gs3d_hash: str | None = None,
    ):
        """Create a transform that uses SAM2 to compute segmentation masks with scale information for each image.

        Args:
            gs3d (GaussianSplat3d | None): The GaussianSplat3d model to use for computing scales.
                If None, the transform can only be used with precomputed cached results (requires gs3d_hash).
            checkpoint (Literal["large", "small", "tiny", "base_plus"]): The checkpoint to use for the SAM2 model.
            points_per_side (int): The number of points to use per side for the segmentation mask.
            pred_iou_thresh (float): The IoU threshold for the segmentation mask.
            stability_score_thresh (float): The stability score threshold for the segmentation mask.
            device (torch.device | str): The device to use for the SAM2 model.
            gs3d_hash (str | None): Precomputed hash of gs3d.means for cache lookup.
                If provided, this is used instead of computing from gs3d. Useful when restoring
                from state_dict with precomputed cached results.
        """
        self._checkpoint = checkpoint
        self._gs3d = gs3d
        self._image_type = "pt"
        self._points_per_side = points_per_side
        self._pred_iou_thresh = pred_iou_thresh
        self._stability_score_thresh = stability_score_thresh
        self._device = device
        self._gs3d_hash = gs3d_hash

        # Only initialize SAM2 model if gs3d is provided (needed for computation)
        self._sam2: SAM2Model | None = None
        if gs3d is not None:
            self._sam2 = SAM2Model(
                checkpoint=checkpoint,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                device=device,
            )

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    @staticmethod
    def _smallest_int_dtype(values: torch.Tensor) -> torch.dtype:
        """Return the smallest signed integer dtype that can hold values in [min_val, max_val]."""
        min_val, max_val = int(values.min().item()), int(values.max().item())
        if min_val >= -128 and max_val <= 127:
            return torch.int8
        elif min_val >= -32768 and max_val <= 32767:
            return torch.int16
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return torch.int32
        else:
            return torch.int64

    @staticmethod
    def rle_encode(tensor: torch.Tensor) -> dict[str, Any]:
        flat = tensor.flatten()
        # Find where values change
        changes = torch.where(flat[1:] != flat[:-1])[0] + 1
        starts = torch.cat([torch.tensor([0]), changes])
        lengths = torch.diff(torch.cat([starts, torch.tensor([len(flat)])])).to(torch.int32)
        values = flat[starts]

        if values.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            # Use smallest dtype that can represent the values
            optimal_dtype = GenerateGARfVDBMasks._smallest_int_dtype(values)
            values = values.to(optimal_dtype)

        return {
            "values": values,
            "lengths": lengths.to(GenerateGARfVDBMasks._smallest_int_dtype(lengths)),
            "shape": tensor.shape,
            "dtype": tensor.dtype,
        }

    @staticmethod
    def rle_decode(encoded: dict[str, Any]) -> torch.Tensor:
        lengths = encoded["lengths"]
        if lengths.dtype in [torch.int8, torch.int16]:
            lengths = lengths.to(torch.int32)
        flat = torch.repeat_interleave(encoded["values"], lengths)
        if flat.dtype in [torch.int8, torch.int16]:
            flat = flat.to(torch.int32)
        return flat.reshape(encoded["shape"])  # .to(encoded["dtype"])

    def __call__(
        self,
        input_scene: SfmScene,
    ) -> SfmScene:
        """
        Perform the compute image segmentation masks transform on the input scene and cache.

        Args:
            input_scene (SfmScene): The input scene containing images to be used to compute segmentation masks.

        Returns:
            output_scene (SfmScene): A new SfmScene with paths to computed segmentation masks.
        """

        # input validation
        if len(input_scene.images) == 0:
            self._logger.warning("No images found in the SfmScene. Returning the input scene unchanged.")
            return input_scene
        if len(input_scene.cameras) == 0:
            self._logger.warning("No cameras found in the SfmScene. Returning the input scene unchanged.")
            return input_scene
        non_pinhole_cameras = [
            camera_id
            for camera_id, camera in input_scene.cameras.items()
            if camera.camera_model != CameraModel.PINHOLE or camera.distortion_coeffs.size != 0
        ]
        if non_pinhole_cameras:
            raise ValueError(
                "GenerateGARfVDBMasks requires an undistorted pinhole SfmScene. "
                "Apply UndistortImages before generating masks. "
                f"Non-pinhole camera IDs: {non_pinhole_cameras}"
            )

        input_cache: SfmCache = input_scene.cache

        # hash of the transform parameters
        # TODO: In PyTorch 2.9 we can use torch.hash_tensor instead
        if self._gs3d_hash is not None:
            # Use precomputed hash (e.g., from state_dict restoration)
            hash_str = self._gs3d_hash
        elif self._gs3d is not None:
            hash_str = hashlib.sha256(self._gs3d.means.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            self._gs3d_hash = hash_str  # Cache for state_dict
        else:
            raise RuntimeError(
                "Cannot compute cache hash: neither gs3d nor gs3d_hash was provided. "
                "Provide either a GaussianSplat3d model or restore from a state_dict that includes gs3d_hash."
            )

        cache_prefix = (
            f"garfvdb_masks_v{GARFVDB_MASK_DATA_SCHEMA_VERSION}_{hash_str}_p{self._points_per_side}_"
            f"i{int(self._pred_iou_thresh * 100)}_s{int(self._stability_score_thresh * 100)}"
        )
        output_cache = input_cache.make_folder(
            cache_prefix,
            description=f"Segmentation masks with scales using points per side {self._points_per_side}, pred iou threshold {self._pred_iou_thresh}, and stability score threshold {self._stability_score_thresh}",
        )

        self._logger.info(
            f"Calculating segmentation masks with scales using points per side {self._points_per_side}, "
            f"pred iou threshold {self._pred_iou_thresh}, "
            f"and stability score threshold {self._stability_score_thresh}"
        )

        self._logger.info(f"Attempting to load segmentation masks with scales from cache.")
        # How many zeros to pad the image index in the mask file names
        num_zeropad = len(str(len(input_scene.images))) + 2

        regenerate_cache = False

        if output_cache.num_files != input_scene.num_images:
            if output_cache.num_files == 0:
                self._logger.info(f"No segmentation masks found in the cache.")
            else:
                self._logger.info(
                    f"Inconsistent number of segmentation masks in the cache. "
                    f"Expected {input_scene.num_images}, found {output_cache.num_files}. "
                    f"Clearing cache and regenerating segmentation masks."
                )
            output_cache.clear_current_folder()
            regenerate_cache = True

        mask_paths: list[str] = []
        for image_id in range(input_scene.num_images):
            if regenerate_cache:
                break
            cache_image_filename = f"masks_{image_id:0{num_zeropad}}"
            image_meta = input_scene.images[image_id]
            if not output_cache.has_file(cache_image_filename):
                self._logger.info(
                    f"Masks {cache_image_filename} not found in the cache. " f"Clearing cache and regenerating."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

            cache_file_meta = output_cache.get_file_metadata(cache_image_filename)
            mask_paths.append(str(cache_file_meta["path"]))
            value_meta = cache_file_meta["metadata"]
            points_per_side = value_meta.get("points_per_side", -1)
            pred_iou_thresh = value_meta.get("pred_iou_thresh", -1)
            stability_score_thresh = value_meta.get("stability_score_thresh", -1)
            gs3d_hash = value_meta.get("gs3d_hash", -1)

            if (
                cache_file_meta.get("data_type", "") != self._image_type
                or points_per_side != self._points_per_side
                or pred_iou_thresh != self._pred_iou_thresh
                or stability_score_thresh != self._stability_score_thresh
                or gs3d_hash != hash_str
            ):
                self._logger.info(
                    f"Output cache mask metadata does not match expected format. "
                    f"Clearing the cache and regenerating masks."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

        if regenerate_cache:
            if self._gs3d is None or self._sam2 is None:
                raise RuntimeError(
                    "Cannot regenerate segmentation masks: gs3d was not provided. "
                    "Either provide a GaussianSplat3d when creating the transform, "
                    "or ensure the cache already contains valid precomputed results."
                )
            min = self._gs3d.means.min(dim=0)[0]
            max = self._gs3d.means.max(dim=0)[0]
            gs3d_extents = torch.abs(max - min)
            max_scale = gs3d_extents.max().item()

            self._logger.info(f"Generating segmentation masks with scales and saving to cache.")
            pbar = tqdm.tqdm(input_scene.images, unit="masks", desc="Generating segmentation masks with scales")
            mask_paths = []
            for image_index, image_meta in enumerate(pbar):
                image_path = image_meta.image_path
                img = cv2.imread(image_path)
                assert img is not None, f"Failed to load image {image_path}"

                scales, pixel_to_mask_id, mask_cdf = self._generate_segmentation_mask(
                    self._gs3d,
                    img,
                    image_meta.camera_metadata.projection_matrix,
                    image_meta.world_to_camera_matrix,
                    max_scale,
                )

                # Save the rescaled image to the cache
                cache_image_filename = f"masks_{image_index:0{num_zeropad}}"
                cache_file_meta = output_cache.write_file(
                    name=cache_image_filename,
                    data={
                        "schema_version": GARFVDB_MASK_DATA_SCHEMA_VERSION,
                        "scales": scales.detach().cpu(),
                        "pixel_to_mask_id": pixel_to_mask_id.to(self._smallest_int_dtype(pixel_to_mask_id))
                        .detach()
                        .cpu(),
                        "mask_cdf": mask_cdf.detach().cpu(),
                    },
                    data_type=self._image_type,
                    metadata={
                        "points_per_side": self._points_per_side,
                        "pred_iou_thresh": self._pred_iou_thresh,
                        "stability_score_thresh": self._stability_score_thresh,
                        "gs3d_hash": hash_str,
                    },
                )
                mask_paths.append(str(cache_file_meta["path"]))

            pbar.close()

            self._logger.info(f"Generated segmentation masks for {input_scene.num_images} images and saved to cache")

        return input_scene.with_attributes(
            **{
                GARFVDB_MASK_ATTRIBUTE_NAME: GARfVDBMaskAttribute(
                    paths=mask_paths,
                    provenance={
                        "generator": self.name(),
                        "generator_version": self.version,
                        "checkpoint": self._checkpoint,
                        "points_per_side": self._points_per_side,
                        "pred_iou_thresh": self._pred_iou_thresh,
                        "stability_score_thresh": self._stability_score_thresh,
                        "carrier_means_sha256": hash_str,
                    },
                )
            }
        )

    @torch.inference_mode()
    def _generate_segmentation_mask(
        self,
        gs3d: GaussianSplat3d,
        img: np.ndarray,
        projection_matrix: np.ndarray,
        world_to_camera_matrix: np.ndarray,
        max_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all the segmentation masks and correlated scale information for the given image using SAM2.

        Args:
            gs3d: GaussianSplat3d object containing the 3D model.
            img: Image to generate segmentation masks for.
            projection_matrix: Projection matrix for the image.
            world_to_camera_matrix: World to camera matrix for the image.
            max_scale: Maximum scale for the segmentation mask.

        Returns:
            scales: Scales for the segmentation masks.
            pixel_to_mask_id: Pixel to mask id mapping.
            mask_cdf: Mask CDF for the segmentation masks.
        """
        img = img.squeeze()  # [H, W, 3]
        h, w = img.shape[:2]
        intrinsics = torch.from_numpy(projection_matrix).to(self._device).squeeze()
        world_to_cam = torch.from_numpy(world_to_camera_matrix).to(self._device).squeeze()

        g_ids, _ = gs3d.render_contributing_gaussian_ids(
            top_k_contributors=1,
            world_to_camera_matrices=world_to_cam.unsqueeze(0).float(),
            projection_matrices=intrinsics.unsqueeze(0).float(),
            image_width=img.shape[1],
            image_height=img.shape[0],
            near=0.01,
            far=1e10,
        )
        # The JaggedTensor has h*w entries, but some may be empty (pixels with no contributing gaussians).
        # We need to create a full tensor with -1 for empty entries.
        num_pixels = h * w
        offsets = g_ids.joffsets  # [h*w + 1]

        # Create output tensor initialized with -1 (no gaussian)
        g_ids_full = torch.full((num_pixels,), -1, dtype=g_ids.jdata.dtype, device=g_ids.jdata.device)

        # Find which pixels have data (where the entry has length > 0)
        has_data = offsets[1:] > offsets[:-1]  # [h*w] bool tensor

        # Fill in the values where we have data
        if g_ids.jdata.numel() > 0:
            g_ids_full[has_data] = g_ids.jdata.view(-1)

        g_ids = g_ids_full.reshape(h, w, 1)  # [H, W, 1]

        self._logger.debug("g_ids.shape " + str(g_ids.shape))
        if g_ids.max() >= gs3d.means.shape[0]:
            self._logger.debug("g_ids.max() " + str(g_ids.max()))
            self._logger.debug("model.means.shape[0] " + str(gs3d.means.shape[0]))
            raise ValueError("g_ids.max() is greater than gs3d.means.shape[0]")

        invalid_mask = g_ids == -1
        if invalid_mask.any():
            self._logger.debug("Found %d invalid (-1) ids" % (invalid_mask.sum().item()))

        world_pts = gs3d.means[g_ids].squeeze(2)  # [H, W, 3]

        # Generate a set of masks for the current image using SAM2
        assert self._sam2 is not None  # Guaranteed by check in __call__
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sam_masks = self._sam2.predict_masks(img)
            sam_masks = sorted(sam_masks, key=(lambda x: x["area"]), reverse=True)
            sam_masks = torch.stack([torch.from_numpy(m["segmentation"]) for m in sam_masks]).to(
                self._device
            )  # [M, H, W]
        # Erode masks to remove noise at the boundary.
        # We're going to compute the scale of each mask by taking the standard deviation of the 3D points
        # within that mask, and the points at the boundary of masks are usually noisy.
        eroded_masks = torch.conv2d(
            sam_masks.unsqueeze(1).float(),
            torch.full((3, 3), 1.0, device=self._device).view(1, 1, 3, 3),
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze(1)  # [M, H, W]

        # mask out any pixels with invalid gaussian ids in the sam_masks
        eroded_masks = eroded_masks * (~invalid_mask.squeeze().unsqueeze(0))

        # Compute a 3D scale per mask which corresponds to the variance of the 3D points that fall within that mask
        # Filter out masks whose scale is too large since very scattered 3D points are likely noise
        self._logger.debug("world_pts " + str(world_pts.shape))
        self._logger.debug("scale " + str(world_pts[eroded_masks[1]].std(dim=0) * 2.0))
        scales = torch.stack([world_pts[mask].unique(dim=0).std(dim=0).norm() for mask in eroded_masks])  # [M]
        keep = scales < max_scale  # [M]
        eroded_masks = eroded_masks[keep]  # [M', H, W]
        scales = scales[keep]  # [M']

        # Compute a tensor that maps pixels to the set of masks which intersect that pixel (sorted by area)
        # i.e. pixel_to_mask_id[i, j] = [m1, m2, m3, ...] where m1, m2, ... are the integer ids of the masks
        # which contain pixel [i, j] and area(m1) <= area(m2) <= area(m3) <= ...
        max_masks = int(eroded_masks.sum(dim=0).max().item())
        pixel_to_mask_id = torch.full(
            (max_masks, eroded_masks.shape[1], eroded_masks.shape[2]), -1, dtype=torch.long, device=self._device
        )  # [MM, H, W]
        for m, mask in enumerate(eroded_masks):
            mask_clone = mask.clone()
            for i in range(max_masks):
                free = pixel_to_mask_id[i] == -1
                masked_area = mask_clone == 1
                right_index = free & masked_area
                if len(pixel_to_mask_id[i][right_index]) > 0:
                    pixel_to_mask_id[i][right_index] = m
                mask_clone[right_index] = 0
        pixel_to_mask_id = pixel_to_mask_id.permute(1, 2, 0)  # [H, W, MM]

        # We're going to use the SAM masks to group pixels for contrastive learning.
        # i.e. we're going to project features for each pixel into the image and push features corresponding to pixels
        #      with the same mask together, and pixels with different masks apart.
        # If we sample pixels, uniformly, we're going to overwhelmingly sample pixels in large masks, and small masks
        # will not get supervised. To fix this, we assign a weight to each mask which intersects a pixel. The weight
        # is proportional to the log probability of sampling that mask (under uniform sampling).
        # These weights are encoded as a CDF per-pixel which we use to choose which mask to use for loss computation
        # at training time

        # Get the unique ids of each mask, and the number of pixels each mask occupies (area)
        mask_ids, num_pix_per_mask = torch.unique(pixel_to_mask_id, return_counts=True)  # [N], [N]

        # Sort masks by their area
        mask_area_sort_ids = torch.argsort(num_pix_per_mask)
        mask_ids, num_pix_per_mask = mask_ids[mask_area_sort_ids], num_pix_per_mask[mask_area_sort_ids]  # [N], [N]
        num_pix_per_mask[0] = 0  # Remove the -1 mask which corresponds to no mask, [N]

        # The probability of any pixel landing in a mask is just the area of the mask over the area of the image
        probs = num_pix_per_mask / num_pix_per_mask.sum()  # [N]

        # Gather the probability values into pixel_to_mask_id, which produces a tensor where
        # each pixel has a list of probabilities that correspond to the masks that intersect that pixel
        mask_probs = torch.gather(probs, 0, pixel_to_mask_id.reshape(-1) + 1).view(pixel_to_mask_id.shape)  # [H, W, MM]

        # Compute a CDF for each pixel (which sums to 1) which weighs each mask by its log probability of being sampled
        # i.e. mask_cdf[i, j, k] is a cumulative probability weight used to select mask k for pixel [i, j]
        mask_cdf = torch.log(mask_probs)
        never_masked = mask_cdf.isinf()
        mask_cdf[never_masked] = 0.0
        mask_cdf = mask_cdf / (mask_cdf.sum(dim=-1, keepdim=True) + 1e-6)
        mask_cdf = torch.cumsum(mask_cdf, dim=-1)  # [H, W, MM]
        mask_cdf[never_masked] = 1.0

        return scales, pixel_to_mask_id, mask_cdf

    @staticmethod
    def name() -> str:
        """
        Return the name of the GenerateGARfVDBMasks transform.

        Returns:
            str: The name of the GenerateGARfVDBMasks transform.
        """
        return "GenerateGARfVDBMasks"

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the GenerateGARfVDBMasks transform for serialization.
        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {
            "name": self.name(),
            "version": self.version,
            "checkpoint": self._checkpoint,
            "points_per_side": self._points_per_side,
            "pred_iou_thresh": self._pred_iou_thresh,
            "stability_score_thresh": self._stability_score_thresh,
            "device": self._device,
            "gs3d_hash": self._gs3d_hash,
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "GenerateGARfVDBMasks":
        """
        Create a GenerateGARfVDBMasks transform from a state dictionary.

        When restored from state_dict, the transform does not have gs3d or the SAM2 model loaded.
        It can only be used with scenes that have precomputed cached results. The gs3d_hash
        stored in the state_dict is used to locate the correct cache folder.

        Args:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.

        Returns:
            GenerateGARfVDBMasks: A restored transform instance.
        """
        if state_dict["name"] != "GenerateGARfVDBMasks":
            raise ValueError(f"Expected state_dict with name 'GenerateGARfVDBMasks', got {state_dict['name']} instead.")
        if state_dict["version"] != GenerateGARfVDBMasks.version:
            raise ValueError(
                f"Expected state_dict with version '{GenerateGARfVDBMasks.version}', got {state_dict['version']} instead."
            )

        return GenerateGARfVDBMasks(
            gs3d=None,  # Not needed for cached results
            checkpoint=state_dict["checkpoint"],
            points_per_side=state_dict["points_per_side"],
            pred_iou_thresh=state_dict["pred_iou_thresh"],
            stability_score_thresh=state_dict["stability_score_thresh"],
            device=state_dict["device"],
            gs3d_hash=state_dict.get("gs3d_hash"),  # Use stored hash for cache lookup
        )
