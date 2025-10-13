# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
from typing import Generator

import numpy as np
import torch


def crop_image_batch(
    image: torch.Tensor, mask: torch.Tensor | None, ncrops: int
) -> Generator[tuple[torch.Tensor, torch.Tensor | None, tuple[int, int, int, int], bool], None, None]:
    """
    Generator to iterate a minibatch of images (B, H, W, C) into disjoint patches patches (B, H_patch, W_patch, C).
    We use this function when training on very large images so that we can accumulate gradients over
    crops of each image.

    Args:
        image (torch.Tensor): Image minibatch (B, H, W, C)
        mask (torch.Tensor | None): Optional mask of shape (B, H, W) to apply to the image.
        ncrops (int): Number of chunks to split the image into (i.e. each crop will have shape (B, H/ncrops x W/ncrops, C).

    Yields: A crop of the input image and its coordinate
        image_patch (torch.Tensor): the patch with shape (B, H/ncrops, W/ncrops, C)
        mask_patch (torch.Tensor | None): the mask patch with shape (B, H/ncrops, W/ncrops) or None if no mask is provided
        crop (tuple[int, int, int, int]): the crop coordinates (x, y, w, h),
        is_last (bool): is true if this is the last crop in the iteration
    """
    h, w = image.shape[1:3]
    patch_w, patch_h = w // ncrops, h // ncrops
    patches = np.array(
        [
            [i * patch_w, j * patch_h, (i + 1) * patch_w, (j + 1) * patch_h]
            for i, j in itertools.product(range(ncrops), range(ncrops))
        ]
    )
    for patch_id in range(patches.shape[0]):
        x1, y1, x2, y2 = patches[patch_id]
        image_patch = image[:, y1:y2, x1:x2]
        mask_patch = None
        if mask is not None:
            mask_patch = mask[:, y1:y2, x1:x2]

        crop = (x1, y1, (x2 - x1), (y2 - y1))
        assert (x2 - x1) == patch_w and (y2 - y1) == patch_h
        is_last = patch_id == (patches.shape[0] - 1)
        yield image_patch, mask_patch, crop, is_last
