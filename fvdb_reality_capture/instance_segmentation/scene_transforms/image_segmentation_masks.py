# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import hashlib
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Literal

import cv2
import numpy as np
import torch
import tqdm
from fvdb_reality_capture.enums import CameraModel
from fvdb_reality_capture.radiance_fields.gaussian_splatting import GaussianSplat3d

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
        points_per_batch=128,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.80,
        device: torch.device | str = "cuda:0",
        gs3d_hash: str | None = None,
    ):
        """Create a transform that uses SAM2 to compute segmentation masks with scale information for each image.

        Args:
            gs3d (GaussianSplat3d | None): The GaussianSplat3d model to use for computing scales.
                If None, the transform can only be used with precomputed cached results (requires gs3d_hash).
            checkpoint (Literal["large", "small", "tiny", "base_plus"]): The checkpoint to use for the SAM2 model.
            points_per_side (int): The number of points to use per side for the segmentation mask.
            points_per_batch (int): The number of point prompts run through the SAM2 mask decoder per
                forward pass. Higher values reduce the number of decoder invocations (faster mask
                generation) at the cost of more GPU memory. Does not affect the generated masks, so it
                is not part of the cache key.
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
        self._points_per_batch = points_per_batch
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
                points_per_batch=points_per_batch,
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
    def _camera_parameters_sha256(scene: SfmScene) -> str:
        """Hash ordered camera poses, intrinsics, image sizes, and stable image IDs."""
        digest = hashlib.sha256()
        arrays = (
            scene.camera_to_world_matrices,
            scene.projection_matrices,
            scene.image_sizes,
            np.asarray([image.image_id for image in scene.images], dtype=np.int64),
        )
        for values in arrays:
            array = np.ascontiguousarray(values)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()

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
        camera_parameters_hash = self._camera_parameters_sha256(input_scene)

        cache_prefix = (
            f"garfvdb_masks_v{GARFVDB_MASK_DATA_SCHEMA_VERSION}_{hash_str}_p{self._points_per_side}_"
            f"i{int(self._pred_iou_thresh * 100)}_s{int(self._stability_score_thresh * 100)}_"
            f"c{camera_parameters_hash}"
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
            cached_camera_parameters_hash = value_meta.get("camera_parameters_sha256", "")

            if (
                cache_file_meta.get("data_type", "") != self._image_type
                or points_per_side != self._points_per_side
                or pred_iou_thresh != self._pred_iou_thresh
                or stability_score_thresh != self._stability_score_thresh
                or gs3d_hash != hash_str
                or cached_camera_parameters_hash != camera_parameters_hash
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

            # Cache writes are disk-bound (~1 s/img) while mask generation is GPU-bound (~2.7 s/img), so we
            # hand each write off to a single background thread. That overlaps an image's write with the next
            # image's SAM2 forward pass instead of blocking on it. The GPU->CPU copies happen here on the main
            # thread so the worker only touches CPU tensors and the disk. A single worker keeps writes ordered
            # and avoids concurrent sqlite/disk contention; since writes are faster than SAM2 the queue stays
            # shallow. Futures are resolved in order below to build mask_paths and surface any write errors.
            write_futures: list[Future[str]] = []
            writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="garfvdb-mask-writer")

            def _submit_write(name: str, data: dict[str, Any]) -> None:
                metadata = {
                    "points_per_side": self._points_per_side,
                    "pred_iou_thresh": self._pred_iou_thresh,
                    "stability_score_thresh": self._stability_score_thresh,
                    "gs3d_hash": hash_str,
                    "camera_parameters_sha256": camera_parameters_hash,
                }
                write_futures.append(
                    writer.submit(
                        lambda: str(
                            output_cache.write_file(
                                name=name, data=data, data_type=self._image_type, metadata=metadata
                            )["path"]
                        )
                    )
                )

            try:
                for image_index, image_meta in enumerate(pbar):
                    image_path = image_meta.image_path
                    img = cv2.imread(image_path)
                    assert img is not None, f"Failed to load image {image_path}"
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    scales, pixel_to_mask_id = self._generate_segmentation_mask(
                        self._gs3d,
                        img,
                        image_meta.camera_metadata.projection_matrix,
                        image_meta.world_to_camera_matrix,
                        max_scale,
                    )

                    # Copy to CPU on the main thread, then hand the disk write to the background thread.
                    cache_image_filename = f"masks_{image_index:0{num_zeropad}}"
                    data = {
                        "schema_version": GARFVDB_MASK_DATA_SCHEMA_VERSION,
                        "scales": scales.detach().cpu(),
                        "pixel_to_mask_id": pixel_to_mask_id.to(self._smallest_int_dtype(pixel_to_mask_id))
                        .detach()
                        .cpu(),
                    }
                    _submit_write(cache_image_filename, data)

                # Drain the background writer, preserving image order for mask_paths and surfacing errors.
                mask_paths = [future.result() for future in write_futures]
            finally:
                writer.shutdown(wait=True)

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
                        "gaussian_means_sha256": hash_str,
                        "camera_parameters_sha256": camera_parameters_hash,
                    },
                )
            }
        )

    @staticmethod
    def _erode_masks(masks: torch.Tensor) -> torch.Tensor:
        """Erode binary masks by one pixel using a 3x3 square structuring element."""
        neighborhood_counts = torch.conv2d(
            masks.unsqueeze(1).float(),
            torch.ones((1, 1, 3, 3), dtype=torch.float32, device=masks.device),
            padding=1,
        )
        return (neighborhood_counts == 9).squeeze(1)

    @torch.inference_mode()
    def _generate_segmentation_mask(
        self,
        gs3d: GaussianSplat3d,
        img: np.ndarray,
        projection_matrix: np.ndarray,
        world_to_camera_matrix: np.ndarray,
        max_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate all the segmentation masks and correlated scale information for the given image using SAM2.

        Args:
            gs3d: GaussianSplat3d object containing the 3D model.
            img: Image to generate segmentation masks for.
            projection_matrix: Projection matrix for the image.
            world_to_camera_matrix: World to camera matrix for the image.
            max_scale: Maximum scale for the segmentation mask.

        Returns:
            scales: Scales for the segmentation masks.
            pixel_to_mask_id: Pixel to mask id mapping. The per-pixel mask-selection CDF is not
                returned or cached; it is recomputed from this tensor at load time via
                ``compute_mask_cdf`` since it is fully derived from the mask areas.
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
        eroded_masks = self._erode_masks(sam_masks)  # [M, H, W]

        # mask out any pixels with invalid gaussian ids in the sam_masks
        eroded_masks = eroded_masks * (~invalid_mask.squeeze().unsqueeze(0))

        # Compute a 3D scale per mask which corresponds to the variance of the 3D points that fall within that mask
        # Filter out masks whose scale is too large since very scattered 3D points are likely noise.
        # Multiple pixels in a mask can hit the same gaussian, so we deduplicate before taking the std. We
        # deduplicate on the (integer) gaussian id rather than the (float3) world point, which is both cheaper
        # and equivalent as long as distinct gaussians have distinct means.
        g_ids_2d = g_ids.squeeze(-1)  # [H, W]
        scales = torch.stack([gs3d.means[g_ids_2d[mask].unique()].std(dim=0).norm() for mask in eroded_masks])  # [M]
        keep = scales < max_scale  # [M]
        eroded_masks = eroded_masks[keep]  # [M', H, W]
        scales = scales[keep]  # [M']

        # Compute a tensor that maps pixels to the set of masks which intersect that pixel (sorted by area)
        # i.e. pixel_to_mask_id[i, j] = [m1, m2, m3, ...] where m1, m2, ... are the integer ids of the masks
        # which contain pixel [i, j] and area(m1) <= area(m2) <= area(m3) <= ...
        num_masks, mask_h, mask_w = eroded_masks.shape
        max_masks = int(eroded_masks.sum(dim=0).max().item()) if num_masks > 0 else 0
        pixel_to_mask_id = torch.full(
            (max_masks, mask_h, mask_w), -1, dtype=torch.long, device=self._device
        )  # [MM, H, W]
        # For each pixel, the masks covering it are packed into slots 0, 1, 2, ... in order of increasing
        # mask index (i.e. decreasing area, since sam_masks are area-sorted). The slot a mask occupies at a
        # covered pixel is the number of lower-indexed masks that also cover that pixel, which is exactly the
        # (exclusive) prefix sum over masks. This replaces the O(M * max_masks) Python loop (which also forced
        # a device sync per iteration) with a single cumsum + scatter.
        if num_masks > 0 and max_masks > 0:
            slot = torch.cumsum(eroded_masks.to(torch.long), dim=0) - 1  # [M, H, W]
            m_index, row, col = torch.where(eroded_masks)  # covered (mask, y, x) triples
            pixel_to_mask_id[slot[m_index, row, col], row, col] = m_index
        pixel_to_mask_id = pixel_to_mask_id.permute(1, 2, 0)  # [H, W, MM]

        # The per-pixel mask-selection CDF (used to weight masks for contrastive learning so small masks
        # aren't drowned out by large ones) is fully derived from pixel_to_mask_id, so we do NOT compute or
        # cache it here.
        return scales, pixel_to_mask_id

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
            "points_per_batch": self._points_per_batch,
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
            points_per_batch=state_dict.get("points_per_batch", 128),
            pred_iou_thresh=state_dict["pred_iou_thresh"],
            stability_score_thresh=state_dict["stability_score_thresh"],
            device=state_dict["device"],
            gs3d_hash=state_dict.get("gs3d_hash"),  # Use stored hash for cache lookup
        )
