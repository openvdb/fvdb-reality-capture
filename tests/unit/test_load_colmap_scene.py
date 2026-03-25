# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import unittest

import numpy as np
from fvdb import CameraModel

from fvdb_reality_capture.sfm_scene._colmap_utils import Camera
from fvdb_reality_capture.sfm_scene._load_colmap_scene import _camera_model_and_distortion_coeffs_from_colmap_camera


class LoadColmapSceneTests(unittest.TestCase):
    def test_opencv_camera_maps_to_packed_radtan5_coeffs(self):
        cam = Camera(
            "OPENCV",
            width_=640,
            height_=480,
            params=np.array([500.0, 505.0, 320.0, 240.0, 0.1, -0.2, 0.003, -0.004], dtype=np.float32),
        )

        camera_model, distortion_coeffs = _camera_model_and_distortion_coeffs_from_colmap_camera(cam)

        expected = np.zeros((12,), dtype=np.float32)
        expected[0] = 0.1
        expected[1] = -0.2
        expected[6] = 0.003
        expected[7] = -0.004

        self.assertEqual(camera_model, CameraModel.OPENCV_RADTAN_5)
        np.testing.assert_allclose(distortion_coeffs, expected)
