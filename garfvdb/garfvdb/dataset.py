# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import random
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

    def __init__(self, segmentation_dataset_path: str):
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
    """A dataset transform that randomly samples pixels from the image.
    Equivalent pixels will also be filtered out of mask_ids and mask_cdf and the original image
    will be preserved in 'image_full'.
    """

    def __init__(self, num_samples_per_image: int):
        self.num_samples_per_image = num_samples_per_image

    def __call__(self, item: SegmentationDataItem) -> SegmentationDataItem:
        """Sample random pixels from the image.
        Args:
            item: SegmentationDataItem to sample pixels from.
        Returns:
            SegmentationDataItem: SegmentationDataItem where image, mask_ids, mask_cdf consist of only the randomly sampled pixels whose original image coordinates are in 'pixel_coords'.
        """
        # sample random pixels
        h, w = int(cast(torch.Tensor, item["image_h"]).item()), int(cast(torch.Tensor, item["image_w"]).item())
        flat_indices = random.sample(range(h * w), k=self.num_samples_per_image)
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
