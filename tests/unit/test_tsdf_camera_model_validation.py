# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest

import torch
from fvdb import CameraModel

from fvdb_reality_capture.tools import point_cloud_from_splats, tsdf_from_splats, tsdf_from_splats_dlnr


class _PlaceholderModel:
    def __init__(self, *, num_channels: int = 3):
        self.device = torch.device("cpu")
        self.num_channels = num_channels


class TsdfCameraModelValidationTests(unittest.TestCase):
    def setUp(self):
        self.camera_to_world = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        self.projection = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        self.image_sizes = torch.tensor([[4, 4]], dtype=torch.int32)

    def test_tsdf_from_splats_rejects_distorted_camera_models(self):
        with self.assertRaisesRegex(NotImplementedError, "only supports CameraModel.PINHOLE"):
            tsdf_from_splats(
                model=_PlaceholderModel(),  # type: ignore[arg-type]
                camera_to_world_matrices=self.camera_to_world,
                projection_matrices=self.projection,
                image_sizes=self.image_sizes,
                truncation_margin=0.1,
                camera_models=torch.tensor([int(CameraModel.OPENCV_RADTAN_5)], dtype=torch.int32),
                show_progress=False,
            )

    def test_tsdf_from_splats_rejects_orthographic_camera_models(self):
        with self.assertRaisesRegex(NotImplementedError, "only supports CameraModel.PINHOLE"):
            tsdf_from_splats(
                model=_PlaceholderModel(),  # type: ignore[arg-type]
                camera_to_world_matrices=self.camera_to_world,
                projection_matrices=self.projection,
                image_sizes=self.image_sizes,
                truncation_margin=0.1,
                camera_models=torch.tensor([int(CameraModel.ORTHOGRAPHIC)], dtype=torch.int32),
                show_progress=False,
            )

    def test_tsdf_from_splats_dlnr_rejects_non_pinhole_camera_models(self):
        with self.assertRaisesRegex(NotImplementedError, "only supports CameraModel.PINHOLE"):
            tsdf_from_splats_dlnr(
                model=_PlaceholderModel(num_channels=3),  # type: ignore[arg-type]
                camera_to_world_matrices=self.camera_to_world,
                projection_matrices=self.projection,
                image_sizes=self.image_sizes,
                truncation_margin=0.1,
                camera_models=torch.tensor([int(CameraModel.OPENCV_RADTAN_5)], dtype=torch.int32),
                show_progress=False,
                num_workers=0,
            )

    def test_point_cloud_from_splats_rejects_distorted_camera_models(self):
        with self.assertRaisesRegex(NotImplementedError, "only supports CameraModel.PINHOLE"):
            point_cloud_from_splats(
                model=_PlaceholderModel(num_channels=3),  # type: ignore[arg-type]
                camera_to_world_matrices=self.camera_to_world,
                projection_matrices=self.projection,
                image_sizes=self.image_sizes,
                camera_models=torch.tensor([int(CameraModel.OPENCV_RADTAN_5)], dtype=torch.int32),
                show_progress=False,
            )

    def test_point_cloud_from_splats_rejects_orthographic_camera_models(self):
        with self.assertRaisesRegex(NotImplementedError, "only supports CameraModel.PINHOLE"):
            point_cloud_from_splats(
                model=_PlaceholderModel(num_channels=3),  # type: ignore[arg-type]
                camera_to_world_matrices=self.camera_to_world,
                projection_matrices=self.projection,
                image_sizes=self.image_sizes,
                camera_models=torch.tensor([int(CameraModel.ORTHOGRAPHIC)], dtype=torch.int32),
                show_progress=False,
            )


if __name__ == "__main__":
    unittest.main()
