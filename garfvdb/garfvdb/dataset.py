# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import random
from pathlib import Path
from typing import Dict, List, NotRequired, TypedDict, Union, cast

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import fvdb


class SegmentationDataItem(TypedDict):
    """Type definition for a single item in the SegmentationDataset for linting convenience."""

    image: torch.Tensor  # [H, W, 3] or [num_samples, 3]
    intrinsics: torch.Tensor  # [3, 3]
    cam_to_world: torch.Tensor  # [4, 4]
    scales: torch.Tensor  # [NM]
    mask_cdf: torch.Tensor  # [H, W, MM] or [num_samples, MM]
    mask_ids: torch.Tensor  # [H, W, MM] or [num_samples, MM]
    image_h: torch.Tensor  # [1]
    image_w: torch.Tensor  # [1]
    image_full: NotRequired[torch.Tensor]  # [H, W, 3]
    pixel_coords: NotRequired[torch.Tensor]  # [num_samples, 2]


class SegmentationDataset(Dataset):
    """Dataset for loading the SegmentationDataset which loads the images, intrinsics, cam_to_worlds,
    scales, mask_cdfs, mask_ids from disk.  Members of this class can then be modified by further
    data transforms."""

    def __init__(self, segmentation_dataset_path: Union[str, Path]):
        """
        Args:
            segmentation_dataset_path: Path to the segmentation dataset.
        """
        logging.info(f"Loading segmentation dataset from {segmentation_dataset_path}")
        data = torch.load(segmentation_dataset_path)

        # NM = num_masks, MM=max_masks
        self.images = data["images"]  # List([H, W, 3])
        self.intrinsics = data["intrinsics"]  # List([3, 3])
        self.cam_to_worlds = data["cam_to_worlds"]  # List([4, 4])
        self.scales = data["scales"]  # List( [NM] )  i.e. [0.6053, 0.4358, 0.2108, 0.2107, 0.2090, 0.1880, 0.1320,
        self.mask_cdfs = data["mask_cdfs"]  # List([H, W, MM]) Float32 i.e. [0.5014, 1., 1., 1.]
        self.mask_ids = data["mask_ids"]  # List([H, W, MM]) i.e. [12, 14, -1, -1],
        self.image_h, self.image_w = self.images[0].shape[0], self.images[0].shape[1]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> SegmentationDataItem:
        return SegmentationDataItem(
            image=self.images[idx],
            intrinsics=self.intrinsics[idx],
            cam_to_world=self.cam_to_worlds[idx],
            scales=self.scales[idx],
            mask_cdf=self.mask_cdfs[idx],
            mask_ids=self.mask_ids[idx],
            image_h=torch.tensor(self.images[idx].shape[0], dtype=torch.int32),
            image_w=torch.tensor(self.images[idx].shape[1], dtype=torch.int32),
        )


class TransformDataset(Dataset):
    """A dataset that applies transforms to the base dataset."""

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if self.transform:
            return self.transform(item)
        return item

    def __len__(self):
        return len(self.base_dataset)


class RandomSelectMaskIDAndScale:
    """A dataset transform that picks a per-image random mask ID based on the mask CDF and interpolates scale values.
    This is used to create a smooth transition between different mask groups."""

    def __call__(self, item: SegmentationDataItem) -> SegmentationDataItem:
        """Pick a per-image random mask ID based on the mask CDF and interpolate scale values.
        Args:
            item: SegmentationDataItem to pick a random mask ID and interpolate scale values from.
        Returns:
            SegmentationDataItem: SegmentationDataItem where mask_ids and scales are updated.
        """

        per_pixel_index = item["mask_ids"]  # [H, W, MM] or [num_samples, MM]
        random_vec_sampling = torch.full(per_pixel_index.shape[:-1], torch.rand((1,)).item()).unsqueeze(-1)  # [H, W, 1]
        random_vec_densify = torch.full(per_pixel_index.shape[:-1], torch.rand((1,)).item())  # [H, W] or [num_samples]

        random_index = torch.sum(random_vec_sampling > item["mask_cdf"], dim=-1)  # [H, W] dtype: torch.int64

        # `per_pixel_index` encodes the list of groups that each pixel belongs to.
        # If there's only one group, then `per_pixel_index` is a 1D tensor
        # -- this will mess up the future `gather` operations.
        if per_pixel_index.shape[-1] == 1:
            per_pixel_mask = per_pixel_index.squeeze()
        else:
            per_pixel_mask = torch.gather(
                per_pixel_index, -1, random_index.unsqueeze(-1)
            ).squeeze()  # [H, W] dtype: torch.int64
            # per_pixel_mask_ is a selection of the *previous* group in the list before the per_pixel_mask selection
            per_pixel_mask_ = torch.gather(
                per_pixel_index,
                -1,
                torch.max(random_index.unsqueeze(-1) - 1, torch.Tensor([0]).int()),
            ).squeeze()

        scales = item["scales"]  # [NM] dtype: torch.float32
        curr_scale = scales[per_pixel_mask]  # [H, W] dtype: torch.float32

        # For pixels in the first group (random_index == 0), randomly scale down their scale value
        # between 0 and the full scale. This creates a smooth transition from zero to the first group's scale,
        # similar to how we interpolate between groups for other indices.
        curr_scale[random_index == 0] = (
            scales[per_pixel_mask][random_index == 0] * random_vec_densify[random_index == 0]
        )
        # For each group, interpolate between the previous group's scale and the current group's scale,
        # based on the random_vec_densify value. This creates a smooth transition between groups.
        for j in range(1, item["mask_cdf"].shape[-1]):
            if (random_index == j).sum() == 0:
                continue
            curr_scale[random_index == j] = (
                scales[per_pixel_mask_][random_index == j]
                + (scales[per_pixel_mask][random_index == j] - scales[per_pixel_mask_][random_index == j])
                * random_vec_densify[random_index == j]
            ).squeeze()

        item["scales"] = curr_scale  # [rays_per_image] dtype: torch.float32

        item["mask_ids"] = per_pixel_mask  # [rays_per_image] dtype: torch.int64

        return item


class RandomSamplePixels:
    """A dataset transform that samples pixels from the image.
    Can use importance sampling based on scales to bias towards smaller scale pixels.
    Equivalent pixels will also be filtered out of mask_ids and mask_cdf and the original image
    will be preserved in 'image_full'.
    """

    def __init__(self, num_samples_per_image: int, scale_bias_strength: float = 0.0):
        """
        Args:
            num_samples_per_image: Number of pixels to sample per image.
            scale_bias_strength: Strength of bias towards smaller scales.
                                0.0 = uniform random sampling (default behavior)
                                > 0.0 = bias towards smaller scales (higher values = stronger bias)
        """
        self.num_samples_per_image = num_samples_per_image
        self.scale_bias_strength = scale_bias_strength

    def __call__(self, item: SegmentationDataItem) -> SegmentationDataItem:
        """Sample pixels from the image, optionally biased towards smaller scales.
        Args:
            item: SegmentationDataItem to sample pixels from.
        Returns:
            SegmentationDataItem: SegmentationDataItem where image, mask_ids, mask_cdf consist of only the sampled pixels whose original image coordinates are in 'pixel_coords'.
        """
        h, w = int(cast(torch.Tensor, item["image_h"]).item()), int(cast(torch.Tensor, item["image_w"]).item())

        if self.scale_bias_strength > 0.0 and "scales" in item:
            # Use importance sampling based on scales (smaller scales = higher probability)
            scales = item["scales"]  # [NM] - scale per mask
            mask_ids = item["mask_ids"]  # [H, W, MM] - mask IDs per pixel

            # Get scale values for each pixel by indexing scales with mask_ids
            # Handle invalid mask IDs (typically -1) by masking them out
            valid_mask = mask_ids >= 0  # [H, W, MM]

            # Vectorized computation of per-pixel scales
            # Clamp mask_ids to valid range to avoid index errors when accessing scales
            clamped_mask_ids = torch.clamp(mask_ids, 0, len(scales) - 1)  # [H, W, MM]

            # Get scale values for all pixels at once using advanced indexing
            pixel_scale_values = scales[clamped_mask_ids]  # [H, W, MM]

            # Mask out invalid entries (set to inf so they don't affect min operation)
            pixel_scale_values = torch.where(valid_mask, pixel_scale_values, float("inf"))

            # Get minimum scale per pixel across the MM dimension
            pixel_scales, _ = torch.min(pixel_scale_values, dim=-1)  # [H, W]

            # Handle pixels with no valid masks (where all scales were inf)
            inf_mask = pixel_scales == float("inf")
            if inf_mask.any():
                median_scale = torch.median(scales)
                pixel_scales[inf_mask] = median_scale

            # Convert scales to sampling probabilities (smaller scales = higher prob)
            inv_scales = 1.0 / (pixel_scales + 1e-8)  # Add small epsilon to avoid division by zero

            # Apply bias strength (higher strength = more bias towards small scales)
            if self.scale_bias_strength != 1.0:
                inv_scales = torch.pow(inv_scales, self.scale_bias_strength)

            # Flatten and normalize to get probabilities
            flat_probs = inv_scales.flatten()
            flat_probs = flat_probs / flat_probs.sum()

            # Determine sampling parameters
            total_pixels = h * w
            num_samples = min(self.num_samples_per_image, total_pixels)

            # Sample according to probabilities
            flat_indices = torch.multinomial(flat_probs, num_samples, replacement=False).tolist()
            pixels = torch.tensor([(idx // w, idx % w) for idx in flat_indices])  # (x, y) format
        else:
            # Fall back to uniform random sampling
            total_pixels = h * w
            num_samples = min(self.num_samples_per_image, total_pixels)
            flat_indices = random.sample(range(total_pixels), k=num_samples)
            pixels = torch.tensor([(idx // w, idx % w) for idx in flat_indices])  # (x, y) format

        item["image_full"] = item["image"]
        item["image"] = item["image"][pixels[:, 0], pixels[:, 1]]
        item["mask_ids"] = item["mask_ids"][pixels[:, 0], pixels[:, 1]]
        item["mask_cdf"] = item["mask_cdf"][pixels[:, 0], pixels[:, 1]]
        item["pixel_coords"] = pixels

        return item


class Resize:
    """A dataset transform that resizes the image and masks."""

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, item: SegmentationDataItem) -> SegmentationDataItem:
        """Resize the image and masks.
        Args:
            item: SegmentationDataItem to resize.
        Returns:
            SegmentationDataItem: SegmentationDataItem where image, mask_cdf, mask_ids, image_h, image_w are resized by 'scale'.
        """
        # Resize image from [H, W, 3] to [H * scale, W * scale, 3]
        item["image"] = (
            F.interpolate(
                item["image"].unsqueeze(0).permute(0, 3, 1, 2),  # [1, 3, H, W]
                scale_factor=self.scale,
                mode="nearest",
            )
            .permute(0, 2, 3, 1)
            .squeeze(0)
        )  # back to [H * scale, W * scale, 3]

        # Update dimensions
        item["image_h"] = torch.tensor(int(item["image_h"] * self.scale), dtype=torch.int32)
        item["image_w"] = torch.tensor(int(item["image_w"] * self.scale), dtype=torch.int32)

        # Resize masks similarly
        item["mask_cdf"] = (
            F.interpolate(item["mask_cdf"].unsqueeze(0).permute(0, 3, 1, 2), scale_factor=self.scale, mode="nearest")
            .permute(0, 2, 3, 1)
            .squeeze(0)
        )

        item["mask_ids"] = (
            TF.resize(
                item["mask_ids"].unsqueeze(0).permute(0, 3, 1, 2),
                size=[item["image"].shape[0], item["image"].shape[1]],
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            )
            .permute(0, 2, 3, 1)
            .squeeze(0)
        )

        # scale intrinsics for new image size
        fx = item["intrinsics"][0, 0]
        fy = item["intrinsics"][1, 1]
        cx = item["intrinsics"][0, 2]
        cy = item["intrinsics"][1, 2]
        new_fx = fx / self.scale
        new_fy = fy / self.scale
        new_cx = cx / self.scale
        new_cy = cy / self.scale
        item["intrinsics"] = torch.tensor([[new_fx, 0, new_cx], [0, new_fy, new_cy], [0, 0, 1]], dtype=torch.float32)

        return item


class GARfVDBInput(Dict[str, Union[torch.Tensor, fvdb.JaggedTensor, None]]):
    """Dictionary with custom behavior for 3D Gaussian splatting inputs."""

    def __repr__(self):
        return f"GARfVDBInput({super().__repr__()})"

    def to(self, device: torch.device) -> "GARfVDBInput":
        return GARfVDBInput(
            {k: v.to(device) if (type(v) in (torch.Tensor, fvdb.JaggedTensor)) else v for k, v in self.items()}
        )


def GARfVDBInputCollateFn(batch: List[SegmentationDataItem]) -> GARfVDBInput:
    """Collate function for a DataLoader to stack the SegmentationDataItems into a GARfVDBInput.
    Args:
        batch: List of SegmentationDataItems.
    Returns:
        GARfVDBInput: A dictionary of tensors that is expected as input to the GARfVDB model.
    """

    kwargs = {
        "image": torch.stack([cast(torch.Tensor, b["image"]) for b in batch]),
        "intrinsics": torch.stack([cast(torch.Tensor, b["intrinsics"]) for b in batch]),
        "cam_to_world": torch.stack([cast(torch.Tensor, b["cam_to_world"]) for b in batch]),
        "image_h": torch.tensor([cast(int, b["image_h"]) for b in batch]),
        "image_w": torch.tensor([cast(int, b["image_w"]) for b in batch]),
        "scales": torch.stack([cast(torch.Tensor, b["scales"]) for b in batch]),
        "mask_cdf": torch.nested.nested_tensor([cast(torch.Tensor, b["mask_cdf"]) for b in batch]),
        "mask_id": torch.stack([cast(torch.Tensor, b["mask_ids"]) for b in batch]),
    }

    if "image_full" in batch[0]:
        kwargs["image_full"] = torch.stack([cast(torch.Tensor, b.get("image_full")) for b in batch])

    if "pixel_coords" in batch[0]:
        kwargs["pixel_coords"] = torch.stack([cast(torch.Tensor, b.get("pixel_coords")) for b in batch])

    return GARfVDBInput(**kwargs)
