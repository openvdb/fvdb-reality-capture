# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import unittest

import numpy as np
from fvdb import CameraModel

from fvdb_reality_capture.sfm_scene._colmap_utils import Camera
from fvdb_reality_capture.sfm_scene._load_colmap_scene import _camera_model_and_distortion_coeffs_from_colmap_camera


class LoadColmapSceneTests(unittest.TestCase):
    def test_supported_colmap_camera_models_map_to_expected_fvdb_models_and_coeffs(self):
        test_cases = [
            (
                "SIMPLE_PINHOLE",
                np.array([500.0, 320.0, 240.0], dtype=np.float32),
                CameraModel.PINHOLE,
                np.empty((0,), dtype=np.float32),
            ),
            (
                "PINHOLE",
                np.array([500.0, 505.0, 320.0, 240.0], dtype=np.float32),
                CameraModel.PINHOLE,
                np.empty((0,), dtype=np.float32),
            ),
            (
                "SIMPLE_RADIAL",
                np.array([500.0, 320.0, 240.0, 0.1], dtype=np.float32),
                CameraModel.OPENCV_RADTAN_5,
                np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
            (
                "RADIAL",
                np.array([500.0, 320.0, 240.0, 0.1, -0.2], dtype=np.float32),
                CameraModel.OPENCV_RADTAN_5,
                np.array([0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
            (
                "OPENCV",
                np.array([500.0, 505.0, 320.0, 240.0, 0.1, -0.2, 0.003, -0.004], dtype=np.float32),
                CameraModel.OPENCV_RADTAN_5,
                np.array([0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.003, -0.004, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
        ]

        for camera_type, params, expected_model, expected_coeffs in test_cases:
            with self.subTest(camera_type=camera_type):
                cam = Camera(camera_type, width_=640, height_=480, params=params)

                camera_model, distortion_coeffs = _camera_model_and_distortion_coeffs_from_colmap_camera(cam)

                self.assertEqual(camera_model, expected_model)
                np.testing.assert_allclose(distortion_coeffs, expected_coeffs)

    def test_opencv_fisheye_camera_is_rejected(self):
        cam = Camera(
            "OPENCV_FISHEYE",
            width_=640,
            height_=480,
            params=np.array([500.0, 505.0, 320.0, 240.0, 0.1, -0.2, 0.003, -0.004], dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "OPENCV_FISHEYE cameras are not supported"):
            _camera_model_and_distortion_coeffs_from_colmap_camera(cam)
