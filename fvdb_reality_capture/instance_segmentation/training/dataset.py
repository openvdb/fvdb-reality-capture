# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from typing import Literal, NotRequired, Sequence, Sized, TypedDict, cast, overload

import fvdb
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from fvdb_reality_capture.radiance_fields import SfmDataset
from fvdb_reality_capture.instance_segmentation.scene_attribute import (
    GARFVDB_MASK_ATTRIBUTE_NAME,
    GARfVDBMaskAttribute,
)
from fvdb_reality_capture.sfm_scene import SfmScene


class SegmentationDataItem(TypedDict):
    """Type definition for a single item in the SegmentationDataset.

    Attributes:
        image: RGB image tensor of shape ``[H, W, 3]`` or ``[num_samples, 3]``.
        projection: Camera projection matrix of shape ``[3, 3]``.
        camera_to_world: Camera-to-world transformation of shape ``[4, 4]``.
        world_to_camera: World-to-camera transformation of shape ``[4, 4]``.
        scales: Per-mask scale values of shape ``[num_masks]``.
        mask_cdf: Cumulative distribution for mask selection, shape ``[H, W, max_masks]``.
        mask_ids: Per-pixel mask IDs of shape ``[H, W, max_masks]``, where -1 indicates
            no mask.
        image_h: Image height in pixels.
        image_w: Image width in pixels.
        image_full: Optional full-resolution image before pixel sampling.
        pixel_coords: Optional sampled pixel coordinates of shape ``[num_samples, 2]``.
    """

    image: torch.Tensor
    projection: torch.Tensor
    camera_to_world: torch.Tensor
    world_to_camera: torch.Tensor
    scales: torch.Tensor
    mask_cdf: torch.Tensor
    mask_ids: torch.Tensor
    image_h: int
    image_w: int
    image_full: NotRequired[torch.Tensor]
    pixel_coords: NotRequired[torch.Tensor]


class SegmentationDataset(SfmDataset):
    """Dataset for loading segmentation training data from an SfmScene.

    Extends SfmDataset to also load pre-computed segmentation masks and scale
    information. The loaded data includes images, camera parameters, and mask
    data that can be further processed by transforms.
    """

    def __init__(
        self,
        sfm_scene: SfmScene,
        dataset_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        cache_loaded_masks: bool = True,
        cache_images: bool = True,
        mask_attribute_name: str = GARFVDB_MASK_ATTRIBUTE_NAME,
    ):
        """
        Args:
            sfm_scene: The SfmScene containing images and camera data.
            dataset_indices: Optional indices to subset the dataset.
            cache_loaded_masks: If True, cache loaded mask data in memory to avoid
                repeated decoding. This significantly speeds up data loading at the cost
                of additional RAM usage. Default is True.
            cache_images: If True, cache decoded/undistorted images in memory. This
                eliminates repeated JPEG/PNG decoding and undistortion, which is
                especially beneficial for small datasets with large images. Default is True.
            mask_attribute_name: Name of the :class:`GARfVDBMaskAttribute` attached
                to ``sfm_scene``.
        """
        super().__init__(sfm_scene, dataset_indices)
        if not sfm_scene.has_attribute(mask_attribute_name):
            raise ValueError(
                f"SfmScene does not contain GARfVDB mask attribute {mask_attribute_name!r}. "
                "Run GenerateGARfVDBMasks after the standard scene transforms."
            )
        mask_attribute = sfm_scene.get_attribute(mask_attribute_name)
        if not isinstance(mask_attribute, GARfVDBMaskAttribute):
            raise TypeError(
                f"Scene attribute {mask_attribute_name!r} must be GARfVDBMaskAttribute, "
                f"got {type(mask_attribute).__name__}."
            )
        self._mask_attribute = mask_attribute
        self._cache_loaded_masks = cache_loaded_masks
        self._cache_images = cache_images
        self._loaded_masks_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._loaded_images_cache: dict[int, dict] = {}  # Cache for SfmDataset items (includes decoded image)

    def warmup_cache(self) -> None:
        """Pre-load all mask and image data into cache.

        Call this BEFORE creating a DataLoader with num_workers > 0.
        When workers are forked, they inherit the populated cache, eliminating
        disk I/O and image decoding during training.
        """
        import tqdm

        with nvtx.range("warmup_cache"):
            # Cache masks
            if self._cache_loaded_masks:
                for idx in tqdm.tqdm(range(len(self)), desc="Warming up mask cache"):
                    index = self._indices[idx]
                    if index not in self._loaded_masks_cache:
                        self._get_mask_data(index)

            # Cache images (decoded and undistorted)
            if self._cache_images:
                for idx in tqdm.tqdm(range(len(self)), desc="Warming up image cache"):
                    index = self._indices[idx]
                    if index not in self._loaded_images_cache:
                        # Call parent's __getitem__ to decode and undistort the image
                        sfm_item = super().__getitem__(idx)
                        self._loaded_images_cache[index] = sfm_item

    def _get_mask_data(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get mask data for the given index, using cache if enabled.

        Returns:
            Tuple of (mask_cdf, mask_ids, scales) tensors.
        """
        with nvtx.range("get_mask_data_cache_check"):
            if self._cache_loaded_masks and index in self._loaded_masks_cache:
                return self._loaded_masks_cache[index]

        # Read and decode from disk
        with nvtx.range("get_mask_data_disk_read"):
            data = self._mask_attribute.load(index)
        # NOTE: One other strategy we investigated was to RLE-encode the mask_cdf and mask_ids,
        # which are smaller (by about 2x) but are slower to decode on disk and would certainly
        # require caching to be performant.
        with nvtx.range("get_mask_data_decode"):
            mask_cdf = data["mask_cdf"]
            mask_ids = data["pixel_to_mask_id"]
            # Using pixel_to_mask_id as indices, torch requires the indices to be at least int32.
            if mask_ids.dtype in [torch.int8, torch.int16]:
                mask_ids = mask_ids.to(torch.int32)

            scales = data["scales"]

        # Cache if enabled
        if self._cache_loaded_masks:
            self._loaded_masks_cache[index] = (mask_cdf, mask_ids, scales)

        return mask_cdf, mask_ids, scales

    def _get_sfm_item(self, idx: int) -> dict:
        """Get SfmDataset item, using cache if enabled.

        Returns:
            Dictionary with image, projection, camera matrices, etc.
        """
        index = self._indices[idx]

        if self._cache_images and index in self._loaded_images_cache:
            return self._loaded_images_cache[index]

        # Decode from disk (slow path)
        sfm_item = super().__getitem__(idx)

        # Cache if enabled
        if self._cache_images:
            self._loaded_images_cache[index] = sfm_item

        return sfm_item

    def __getitem__(self, idx: int) -> SegmentationDataItem:  # pyright: ignore[reportIncompatibleMethodOverride]
        with nvtx.range("SegmentationDataset.__getitem__"):
            with nvtx.range("sfm_dataset_getitem"):
                sfm_item = self._get_sfm_item(idx)
            index = self._indices[idx]

            mask_cdf, mask_ids, scales = self._get_mask_data(index)

            with nvtx.range("build_segmentation_data_item"):
                return SegmentationDataItem(
                    image=torch.from_numpy(sfm_item["image"]),
                    projection=sfm_item["projection"],
                    camera_to_world=sfm_item["camera_to_world"],
                    world_to_camera=sfm_item["world_to_camera"],
                    scales=scales,
                    mask_cdf=mask_cdf,
                    mask_ids=mask_ids,
                    image_h=sfm_item["image"].shape[0],
                    image_w=sfm_item["image"].shape[1],
                )

    @property
    def scales(self) -> torch.Tensor:
        scales = []
        for index in self._indices:
            # Use cache if available, otherwise read from disk
            if self._cache_loaded_masks and index in self._loaded_masks_cache:
                scales.append(self._loaded_masks_cache[index][2])  # scales is third element
            else:
                data = self._mask_attribute.load(int(index))
                scales.append(data["scales"])
        return torch.cat(scales)

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
    def indices(self) -> np.ndarray:
        """
        Return the indices of the images in the SfmScene used in the dataset.

        Returns:
            np.ndarray: The indices of the images in the SfmScene used in the dataset.
        """
        return self._indices


class InfiniteSampler(torch.utils.data.Sampler):
    """Sampler that yields dataset indices infinitely without stopping.

    Unlike standard samplers, this never raises StopIteration, allowing the
    DataLoader iterator to run indefinitely. This avoids the performance
    overhead of recreating the iterator between epochs. Epoch boundaries
    must be tracked manually by counting samples processed.
    """

    def __init__(self, dataset: Sized, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        while True:
            if self.shuffle:
                # Use epoch as part of seed for reproducibility across restarts
                g = torch.Generator()
                g.manual_seed(self.seed + self._epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))

            yield from indices
            self._epoch += 1

    def __len__(self):
        """Length is undefined for InfiniteSampler.

        This sampler yields indices indefinitely and does not define epoch
        boundaries. Code must not rely on len(sampler) to determine the number
        of steps per epoch; track epochs/steps explicitly instead.
        """
        raise NotImplementedError(
            "InfiniteSampler does not define __len__; it yields indices infinitely. "
            "Do not rely on len(sampler) for epoch length."
        )


class GARfVDBInput(dict[str, torch.Tensor | fvdb.JaggedTensor | list[int] | None]):
    @overload
    def __getitem__(self, key: Literal["image_w"]) -> list[int]: ...
    @overload
    def __getitem__(self, key: Literal["image_h"]) -> list[int]: ...
    @overload
    def __getitem__(
        self,
        key: Literal["image", "projection", "camera_to_world", "world_to_camera", "scales", "mask_ids", "mask_cdf"],
    ) -> torch.Tensor: ...
    @overload
    def __getitem__(self, key: Literal["pixel_coords"]) -> torch.Tensor: ...
    @overload
    def __getitem__(self, key: str) -> torch.Tensor | fvdb.JaggedTensor | list[int] | None: ...

    def __getitem__(self, key: str) -> torch.Tensor | fvdb.JaggedTensor | list[int] | None:  # type: ignore[override]
        return super().__getitem__(key)

    @overload
    def get(self, key: Literal["pixel_coords"], default: None = None) -> torch.Tensor | None: ...
    @overload
    def get(
        self, key: str, default: torch.Tensor | fvdb.JaggedTensor | list[int] | None = None
    ) -> torch.Tensor | fvdb.JaggedTensor | list[int] | None: ...

    def get(self, key: str, default: torch.Tensor | fvdb.JaggedTensor | list[int] | None = None) -> torch.Tensor | fvdb.JaggedTensor | list[int] | None:  # type: ignore[override]
        return super().get(key, default)

    def __repr__(self):
        return f"GARfVDBInput({dict(self)!r})"

    def to(self, device: torch.device, non_blocking: bool = True) -> "GARfVDBInput":
        result = {}
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device, non_blocking=non_blocking)
            elif isinstance(v, fvdb.JaggedTensor):
                # JaggedTensor.to() doesn't support non_blocking
                result[k] = v.to(device)
            else:
                result[k] = v
        return GARfVDBInput(result)


def GARfVDBInputCollateFn(
    batch: list[SegmentationDataItem],
    collate_full_image: bool = False,
    include_mask_cdf: bool = False,
) -> GARfVDBInput:
    """Collate SegmentationDataItems into a batched GARfVDBInput.

    Args:
        batch: List of SegmentationDataItems to collate.
        collate_full_image: If True, also stack the full-resolution images.
        include_mask_cdf: If True, include mask_cdf for GPU-based mask selection.
            Set to True when using GPURandomSelectMaskIDAndScale.

    Returns:
        Batched input dictionary for the GARfVDB model.
    """
    with nvtx.range("GARfVDBInputCollateFn"):
        with nvtx.range("collate_stack_tensors"):
            kwargs: dict[str, torch.Tensor | fvdb.JaggedTensor | list[int] | None] = {
                "image": torch.stack([cast(torch.Tensor, b["image"]) for b in batch]),
                "projection": torch.stack([cast(torch.Tensor, b["projection"]) for b in batch]),
                "camera_to_world": torch.stack([cast(torch.Tensor, b["camera_to_world"]) for b in batch]),
                "world_to_camera": torch.stack([cast(torch.Tensor, b["world_to_camera"]) for b in batch]),
                "image_h": [cast(int, b["image_h"]) for b in batch],
                "image_w": [cast(int, b["image_w"]) for b in batch],
            }

        # mask_ids has shape [num_samples, MM] where MM (max masks per pixel) varies per image
        # Pad to max MM in batch for GPU transform path, or stack directly for CPU path
        if include_mask_cdf:
            # Pad mask_ids and mask_cdf to max MM dimension
            mask_ids_list = [cast(torch.Tensor, b["mask_ids"]) for b in batch]
            mask_cdf_list = [cast(torch.Tensor, b["mask_cdf"]) for b in batch]
            max_mm = max(m.shape[-1] for m in mask_ids_list)

            # Pad mask_ids with -1 (invalid mask), mask_cdf with 1.0 (never selected)
            padded_mask_ids = []
            padded_mask_cdf = []
            for mask_ids, mask_cdf in zip(mask_ids_list, mask_cdf_list):
                mm = mask_ids.shape[-1]
                if mm < max_mm:
                    pad_size = max_mm - mm
                    # Pad last dimension: (left, right) for 2D tensor's last dim
                    mask_ids = torch.nn.functional.pad(mask_ids, (0, pad_size), value=-1)
                    mask_cdf = torch.nn.functional.pad(mask_cdf, (0, pad_size), value=1.0)
                padded_mask_ids.append(mask_ids)
                padded_mask_cdf.append(mask_cdf)

            kwargs["mask_ids"] = torch.stack(padded_mask_ids)
            kwargs["mask_cdf"] = torch.stack(padded_mask_cdf)

            # scales has variable length per image (different number of masks), use JaggedTensor
            kwargs["scales"] = fvdb.JaggedTensor([cast(torch.Tensor, b["scales"]) for b in batch])
        else:
            # CPU transform path - transforms already applied, dimensions should be uniform
            kwargs["mask_ids"] = torch.stack([cast(torch.Tensor, b["mask_ids"]) for b in batch])
            kwargs["scales"] = torch.stack([cast(torch.Tensor, b["scales"]) for b in batch])

        if "pixel_coords" in batch[0]:
            kwargs["pixel_coords"] = torch.stack([cast(torch.Tensor, b.get("pixel_coords")) for b in batch])

        if collate_full_image and "image_full" in batch[0]:
            kwargs["image_full"] = torch.stack([cast(torch.Tensor, b.get("image_full")) for b in batch])

        return GARfVDBInput(**kwargs)
