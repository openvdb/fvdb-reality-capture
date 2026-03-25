# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import tempfile
import unittest

import cv2
import numpy as np
from fvdb import CameraModel

from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset
from fvdb_reality_capture.sfm_scene import SfmCache, SfmCameraMetadata, SfmPosedImageMetadata, SfmScene


def _packed_radtan5_coeffs() -> np.ndarray:
    coeffs = np.zeros((12,), dtype=np.float32)
    coeffs[0] = 0.1
    coeffs[1] = -0.05
    coeffs[2] = 0.01
    coeffs[6] = 0.002
    coeffs[7] = -0.003
    return coeffs


class GaussianSplatDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_scene(self) -> tuple[SfmScene, SfmCameraMetadata]:
        image_path = self.root / "image.png"
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), image))

        camera_metadata = SfmCameraMetadata(
            img_width=10,
            img_height=8,
            fx=6.0,
            fy=6.5,
            cx=5.0,
            cy=4.0,
            camera_model=CameraModel.OPENCV_RADTAN_5,
            distortion_coeffs=_packed_radtan5_coeffs(),
        )
        image_metadata = SfmPosedImageMetadata(
            world_to_camera_matrix=np.eye(4, dtype=np.float32),
            camera_to_world_matrix=np.eye(4, dtype=np.float32),
            camera_metadata=camera_metadata,
            camera_id=1,
            image_path=str(image_path),
            mask_path="",
            point_indices=np.array([], dtype=np.int64),
            image_id=0,
        )
        cache = SfmCache.get_cache(self.root / "cache_root", "dataset_unit_test_cache", "Dataset unit test cache")
        scene = SfmScene(
            cameras={1: camera_metadata},
            images=[image_metadata],
            points=np.zeros((0, 3), dtype=np.float32),
            points_err=np.zeros((0,), dtype=np.float32),
            points_rgb=np.zeros((0, 3), dtype=np.uint8),
            scene_bbox=None,
            transformation_matrix=np.eye(4, dtype=np.float32),
            cache=cache,
        )
        return scene, camera_metadata

    def test_dataset_returns_camera_model_and_distortion_coeffs(self):
        scene, _ = self._make_scene()

        datum = SfmDataset(scene)[0]

        self.assertEqual(int(datum["camera_model"]), int(CameraModel.OPENCV_RADTAN_5))
        np.testing.assert_allclose(datum["distortion_coeffs"].numpy(), _packed_radtan5_coeffs())

    def test_dataset_distortion_coeffs_do_not_alias_camera_metadata(self):
        scene, camera_metadata = self._make_scene()

        datum = SfmDataset(scene)[0]
        datum["distortion_coeffs"][0] = 999.0

        self.assertAlmostEqual(float(camera_metadata.distortion_coeffs[0]), 0.1)
