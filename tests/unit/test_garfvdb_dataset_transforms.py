# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import torch

from fvdb_reality_capture.instance_segmentation.training.dataset import SegmentationDataItem
from fvdb_reality_capture.instance_segmentation.training.dataset_transforms import (
    RandomSamplePixels,
    Resize,
    _sample_distinct_indices,
)


def test_sample_distinct_indices_has_no_duplicates():
    torch.manual_seed(0)
    # 4096 of 2M would contain a duplicate on ~98% of draws with replacement.
    for total, num_samples in [(778 * 519, 256), (2_000_000, 4096), (1000, 999), (10, 10)]:
        for _ in range(20):
            indices = _sample_distinct_indices(total, num_samples)
            assert indices.shape == (num_samples,)
            assert indices.unique().numel() == num_samples
            assert indices.min() >= 0 and indices.max() < total


def test_sample_distinct_indices_covers_small_ranges():
    torch.manual_seed(0)
    # Sampling every index must return exactly the full range.
    indices = _sample_distinct_indices(7, 7)
    torch.testing.assert_close(indices.sort().values, torch.arange(7))


def test_random_sample_pixels_returns_distinct_pixels():
    torch.manual_seed(0)
    h, w, num_samples = 12, 16, 100
    item: SegmentationDataItem = {
        "image": torch.arange(h * w * 3, dtype=torch.float32).reshape(h, w, 3),
        "projection": torch.eye(3),
        "camera_to_world": torch.eye(4),
        "world_to_camera": torch.eye(4),
        "scales": torch.tensor([0.1]),
        "mask_cdf": torch.ones((h, w, 1)),
        "mask_ids": torch.zeros((h, w, 1), dtype=torch.int32),
        "image_h": h,
        "image_w": w,
    }

    sampled = RandomSamplePixels(num_samples)(item)

    coords = sampled["pixel_coords"]
    assert coords.shape == (num_samples, 2)
    flat = coords[:, 0] * w + coords[:, 1]
    assert flat.unique().numel() == num_samples
    assert coords[:, 0].max() < h and coords[:, 1].max() < w
    assert sampled["image"].shape == (num_samples, 3)
    torch.testing.assert_close(sampled["image"], sampled["image_full"][coords[:, 0], coords[:, 1]])


def test_random_sample_pixels_scale_bias_favors_small_scale_masks():
    torch.manual_seed(0)
    h, w, num_samples = 20, 20, 50
    # Left half is covered by a small-scale mask, right half by a large-scale mask.
    mask_ids = torch.full((h, w, 1), 1, dtype=torch.int32)
    mask_ids[:, : w // 2] = 0
    item: SegmentationDataItem = {
        "image": torch.zeros((h, w, 3)),
        "projection": torch.eye(3),
        "camera_to_world": torch.eye(4),
        "world_to_camera": torch.eye(4),
        "scales": torch.tensor([0.01, 1.0]),
        "mask_cdf": torch.ones((h, w, 1)),
        "mask_ids": mask_ids,
        "image_h": h,
        "image_w": w,
    }

    sampled = RandomSamplePixels(num_samples, scale_bias_strength=1.0)(item)

    coords = sampled["pixel_coords"]
    assert coords.shape == (num_samples, 2)
    flat = coords[:, 0] * w + coords[:, 1]
    assert flat.unique().numel() == num_samples
    # With 100:1 weighting nearly every sample should land in the small-scale half.
    assert (coords[:, 1] < w // 2).float().mean() > 0.9


def test_resize_scales_intrinsics_by_actual_rounded_dimensions():
    projection = torch.tensor(
        [
            [70.0, 2.0, 3.5],
            [0.0, 50.0, 2.5],
            [0.0, 0.0, 1.0],
        ]
    )
    item: SegmentationDataItem = {
        "image": torch.zeros((5, 7, 3)),
        "projection": projection,
        "camera_to_world": torch.eye(4),
        "world_to_camera": torch.eye(4),
        "scales": torch.tensor([0.1]),
        "mask_cdf": torch.ones((5, 7, 1)),
        "mask_ids": torch.zeros((5, 7, 1), dtype=torch.int32),
        "image_h": 5,
        "image_w": 7,
    }

    resized = Resize(0.5)(item)

    assert resized["image"].shape == (2, 3, 3)
    assert resized["mask_cdf"].shape == (2, 3, 1)
    assert resized["mask_ids"].shape == (2, 3, 1)
    assert resized["image_h"] == 2
    assert resized["image_w"] == 3

    expected_projection = projection.clone()
    expected_projection[0, :] *= 3 / 7
    expected_projection[1, :] *= 2 / 5
    torch.testing.assert_close(resized["projection"], expected_projection)
