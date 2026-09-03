# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import torch

from fvdb_reality_capture.instance_segmentation.training.dataset import SegmentationDataItem
from fvdb_reality_capture.instance_segmentation.training.dataset_transforms import Resize


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
