# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import tempfile
import unittest
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
from fvdb import CameraModel

from fvdb_reality_capture.sfm_scene._colmap_utils import Camera
from fvdb_reality_capture.sfm_scene._load_colmap_scene import _camera_model_and_distortion_coeffs_from_colmap_camera
from fvdb_reality_capture.sfm_scene._load_colmap_scene import load_colmap_scene
from fvdb_reality_capture.sfm_scene._colmap_utils.image import Image
from fvdb_reality_capture.sfm_scene._colmap_utils.rotation import Quaternion


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

    def test_load_colmap_scene_preserves_sorted_images_camera_reuse_masks_and_visible_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            colmap_path = pathlib.Path(tmpdir)
            (colmap_path / "images").mkdir()
            (colmap_path / "masks").mkdir()
            (colmap_path / "masks" / "a.png").write_bytes(b"")
            (colmap_path / "masks" / "b.jpg").write_bytes(b"")

            scene_manager = type("FakeSceneManager", (), {})()
            scene_manager.cameras = {
                1: Camera("PINHOLE", width_=640, height_=480, params=np.array([500.0, 505.0, 320.0, 240.0])),
                2: Camera(
                    "OPENCV",
                    width_=800,
                    height_=600,
                    params=np.array([700.0, 710.0, 400.0, 300.0, 0.1, -0.2, 0.003, -0.004]),
                ),
            }
            scene_manager.images = OrderedDict(
                [
                    (42, Image("z.jpg", 1, Quaternion(), np.array([1.0, 2.0, 3.0]))),
                    (7, Image("a.jpg", 2, Quaternion(), np.array([0.0, 0.0, 1.0]))),
                    (100, Image("b.jpg", 1, Quaternion(), np.array([-1.0, 0.5, 2.0]))),
                ]
            )
            scene_manager.points3D = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
            scene_manager.point3D_errors = np.array([0.1, 0.2], dtype=np.float32)
            scene_manager.point3D_colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
            scene_manager.point3D_id_to_point3D_idx = {11: 0, 13: 1}
            scene_manager.point3D_id_to_images = {
                11: [(7, 0), (100, 3)],
                13: [(42, 1), (100, 2)],
            }

            with patch(
                "fvdb_reality_capture.sfm_scene._load_colmap_scene._load_colmap_internal", return_value=scene_manager
            ):
                loaded_cameras, loaded_images, points, points_err, points_rgb, cache = load_colmap_scene(colmap_path)

            self.assertEqual(set(loaded_cameras.keys()), {1, 2})
            self.assertEqual([image.image_id for image in loaded_images], [0, 1, 2])
            self.assertEqual(
                [pathlib.Path(image.image_path).name for image in loaded_images], ["a.jpg", "b.jpg", "z.jpg"]
            )
            self.assertEqual([image.camera_id for image in loaded_images], [2, 1, 1])
            self.assertIs(loaded_images[1].camera_metadata, loaded_images[2].camera_metadata)

            self.assertEqual(str(loaded_images[0].mask_path), str((colmap_path / "masks" / "a.png").absolute()))
            self.assertEqual(str(loaded_images[1].mask_path), str((colmap_path / "masks" / "b.jpg").absolute()))
            self.assertEqual(loaded_images[2].mask_path, "")

            np.testing.assert_array_equal(loaded_images[0].point_indices, np.array([0], dtype=np.int32))
            np.testing.assert_array_equal(loaded_images[1].point_indices, np.array([0, 1], dtype=np.int32))
            np.testing.assert_array_equal(loaded_images[2].point_indices, np.array([1], dtype=np.int32))

            self.assertEqual(loaded_images[0].camera_metadata.camera_model, CameraModel.OPENCV_RADTAN_5)
            np.testing.assert_allclose(
                loaded_images[0].camera_metadata.distortion_coeffs,
                np.array([0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.003, -0.004, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
            self.assertEqual(loaded_images[1].camera_metadata.camera_model, CameraModel.PINHOLE)

            np.testing.assert_allclose(points, scene_manager.points3D)
            np.testing.assert_allclose(points_err, scene_manager.point3D_errors)
            np.testing.assert_array_equal(points_rgb, scene_manager.point3D_colors)
            self.assertTrue((colmap_path / "_cache").exists())
            self.assertTrue(cache.has_file("visible_points_per_image"))
