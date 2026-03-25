# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import unittest

import numpy as np
from fvdb import CameraModel

from fvdb_reality_capture.sfm_scene import SfmCameraMetadata
from fvdb_reality_capture.sfm_scene.sfm_metadata import (
    _as_packed_distortion_coeffs,
    _legacy_camera_type_to_camera_model,
    _legacy_distortion_parameters_to_coeffs,
)


def _packed_radtan5_coeffs(
    k1: float = 0.1,
    k2: float = -0.05,
    k3: float = 0.01,
    p1: float = 0.002,
    p2: float = -0.003,
) -> np.ndarray:
    coeffs = np.zeros((12,), dtype=np.float32)
    coeffs[0] = k1
    coeffs[1] = k2
    coeffs[2] = k3
    coeffs[6] = p1
    coeffs[7] = p2
    return coeffs


class SfmMetadataHelperTests(unittest.TestCase):
    def test_as_packed_distortion_coeffs_accepts_list_tuple_and_array(self):
        coeffs_list = [float(i) for i in range(12)]
        coeffs_tuple = tuple(coeffs_list)
        coeffs_array = np.array(coeffs_list, dtype=np.float64)

        for coeffs in (coeffs_list, coeffs_tuple, coeffs_array):
            with self.subTest(type=type(coeffs).__name__):
                normalized = _as_packed_distortion_coeffs(coeffs)
                self.assertIsInstance(normalized, np.ndarray)
                self.assertEqual(normalized.dtype, np.float32)
                self.assertEqual(normalized.shape, (12,))
                np.testing.assert_allclose(normalized, np.array(coeffs_list, dtype=np.float32))

    def test_as_packed_distortion_coeffs_accepts_empty(self):
        normalized = _as_packed_distortion_coeffs([])
        self.assertIsInstance(normalized, np.ndarray)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(normalized.shape, (0,))

    def test_as_packed_distortion_coeffs_rejects_invalid_length(self):
        with self.assertRaisesRegex(ValueError, "distortion_coeffs must have shape"):
            _as_packed_distortion_coeffs([0.1, 0.2])

    def test_as_packed_distortion_coeffs_rejects_non_1d_array(self):
        coeffs = np.arange(12, dtype=np.float32).reshape(4, 3)

        with self.assertRaisesRegex(ValueError, "distortion_coeffs must have shape"):
            _as_packed_distortion_coeffs(coeffs)

    def test_legacy_camera_type_to_camera_model(self):
        self.assertEqual(_legacy_camera_type_to_camera_model("PINHOLE"), CameraModel.PINHOLE)
        self.assertEqual(_legacy_camera_type_to_camera_model("SIMPLE_PINHOLE"), CameraModel.PINHOLE)
        self.assertEqual(_legacy_camera_type_to_camera_model("SIMPLE_RADIAL"), CameraModel.OPENCV_RADTAN_5)
        self.assertEqual(_legacy_camera_type_to_camera_model("RADIAL"), CameraModel.OPENCV_RADTAN_5)
        self.assertEqual(_legacy_camera_type_to_camera_model("OPENCV"), CameraModel.OPENCV_RADTAN_5)
        with self.assertRaisesRegex(ValueError, "Unsupported legacy camera_type"):
            _legacy_camera_type_to_camera_model("OPENCV_FISHEYE")

    def test_legacy_distortion_parameters_to_coeffs(self):
        pinhole = _legacy_distortion_parameters_to_coeffs("PINHOLE", np.array([], dtype=np.float32))
        self.assertEqual(pinhole.shape, (0,))

        simple_radial = _legacy_distortion_parameters_to_coeffs("SIMPLE_RADIAL", np.array([0.15], dtype=np.float32))
        self.assertEqual(simple_radial.shape, (12,))
        self.assertAlmostEqual(float(simple_radial[0]), 0.15)
        self.assertTrue(np.all(simple_radial[1:] == 0.0))

        radial = _legacy_distortion_parameters_to_coeffs("RADIAL", np.array([0.15, -0.03], dtype=np.float32))
        self.assertAlmostEqual(float(radial[0]), 0.15)
        self.assertAlmostEqual(float(radial[1]), -0.03)
        self.assertTrue(np.all(radial[2:] == 0.0))

        opencv = _legacy_distortion_parameters_to_coeffs(
            "OPENCV",
            np.array([0.1, -0.2, 0.003, -0.004, 0.05], dtype=np.float32),
        )
        expected = np.zeros((12,), dtype=np.float32)
        expected[0] = 0.1
        expected[1] = -0.2
        expected[2] = 0.05
        expected[6] = 0.003
        expected[7] = -0.004
        np.testing.assert_allclose(opencv, expected)

        with self.assertRaisesRegex(ValueError, "Unsupported legacy camera_type"):
            _legacy_distortion_parameters_to_coeffs("OPENCV_FISHEYE", np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))


class SfmCameraMetadataTests(unittest.TestCase):
    @staticmethod
    def _make_camera(
        camera_model: CameraModel = CameraModel.PINHOLE,
        distortion_coeffs: np.ndarray | None = None,
        width: int = 12,
        height: int = 10,
    ) -> SfmCameraMetadata:
        return SfmCameraMetadata(
            img_width=width,
            img_height=height,
            fx=8.0,
            fy=7.5,
            cx=5.5,
            cy=4.5,
            camera_model=camera_model,
            distortion_coeffs=np.array([], dtype=np.float32) if distortion_coeffs is None else distortion_coeffs,
        )

    def test_pinhole_camera_has_expected_single_camera_state(self):
        camera = self._make_camera()
        self.assertEqual(camera.camera_model, CameraModel.PINHOLE)
        self.assertEqual(camera.distortion_coeffs.shape, (0,))
        self.assertTrue(camera.can_undistort)
        self.assertEqual(camera.width, 12)
        self.assertEqual(camera.height, 10)
        np.testing.assert_allclose(
            camera.projection_matrix,
            np.array([[8.0, 0.0, 5.5], [0.0, 7.5, 4.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        )

    def test_supported_distorted_camera_preserves_raw_pixel_space_metadata(self):
        camera = self._make_camera(CameraModel.OPENCV_RADTAN_5, _packed_radtan5_coeffs())
        self.assertTrue(camera.can_undistort)
        self.assertEqual(camera.distortion_coeffs.shape, (12,))
        self.assertEqual(camera.width, 12)
        self.assertEqual(camera.height, 10)
        np.testing.assert_allclose(
            camera.projection_matrix,
            np.array([[8.0, 0.0, 5.5], [0.0, 7.5, 4.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        )

    def test_unsupported_distorted_camera_reports_no_local_undistortion_support(self):
        camera = self._make_camera(CameraModel.OPENCV_RATIONAL_8, _packed_radtan5_coeffs())
        self.assertFalse(camera.can_undistort)
        np.testing.assert_allclose(
            camera.projection_matrix,
            np.array([[8.0, 0.0, 5.5], [0.0, 7.5, 4.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        )

    def test_state_dict_uses_new_schema(self):
        camera = self._make_camera(CameraModel.OPENCV_RADTAN_5, _packed_radtan5_coeffs())
        state = camera.state_dict()
        self.assertEqual(state["camera_model"], "OPENCV_RADTAN_5")
        self.assertIn("distortion_coeffs", state)
        self.assertNotIn("camera_type", state)
        self.assertNotIn("distortion_parameters", state)

    def test_state_dict_roundtrip_preserves_camera_metadata(self):
        camera = self._make_camera(CameraModel.OPENCV_RADTAN_5, _packed_radtan5_coeffs())
        loaded = SfmCameraMetadata.from_state_dict(camera.state_dict())
        self.assertEqual(loaded.camera_model, camera.camera_model)
        self.assertEqual(loaded.width, camera.width)
        self.assertEqual(loaded.height, camera.height)
        np.testing.assert_allclose(loaded.distortion_coeffs, camera.distortion_coeffs)
        np.testing.assert_allclose(loaded.projection_matrix, camera.projection_matrix)

    def test_from_state_dict_accepts_integer_camera_model_for_backwards_compatibility(self):
        state = self._make_camera(CameraModel.OPENCV_RADTAN_5, _packed_radtan5_coeffs()).state_dict()
        state["camera_model"] = int(CameraModel.OPENCV_RADTAN_5)
        loaded = SfmCameraMetadata.from_state_dict(state)
        self.assertEqual(loaded.camera_model, CameraModel.OPENCV_RADTAN_5)
        np.testing.assert_allclose(loaded.distortion_coeffs, _packed_radtan5_coeffs())

    def test_from_state_dict_rejects_non_1d_distortion_coeffs(self):
        state = self._make_camera(CameraModel.OPENCV_RADTAN_5, _packed_radtan5_coeffs()).state_dict()
        state["distortion_coeffs"] = np.arange(12, dtype=np.float32).reshape(4, 3).tolist()

        with self.assertRaisesRegex(ValueError, "distortion_coeffs must have shape"):
            SfmCameraMetadata.from_state_dict(state)

    def test_from_state_dict_migrates_legacy_pinhole_schema(self):
        state = {
            "img_width": 12,
            "img_height": 10,
            "fx": 8.0,
            "fy": 7.5,
            "cx": 5.5,
            "cy": 4.5,
            "camera_type": "PINHOLE",
            "distortion_parameters": [],
        }
        loaded = SfmCameraMetadata.from_state_dict(state)
        self.assertEqual(loaded.camera_model, CameraModel.PINHOLE)
        self.assertEqual(loaded.distortion_coeffs.shape, (0,))

    def test_from_state_dict_migrates_legacy_simple_radial_schema(self):
        state = {
            "img_width": 12,
            "img_height": 10,
            "fx": 8.0,
            "fy": 7.5,
            "cx": 5.5,
            "cy": 4.5,
            "camera_type": "SIMPLE_RADIAL",
            "distortion_parameters": [0.15],
        }
        loaded = SfmCameraMetadata.from_state_dict(state)
        self.assertEqual(loaded.camera_model, CameraModel.OPENCV_RADTAN_5)
        self.assertEqual(loaded.distortion_coeffs.shape, (12,))
        self.assertAlmostEqual(float(loaded.distortion_coeffs[0]), 0.15)
        self.assertTrue(np.all(loaded.distortion_coeffs[1:] == 0.0))

    def test_from_state_dict_migrates_legacy_radial_schema(self):
        state = {
            "img_width": 12,
            "img_height": 10,
            "fx": 8.0,
            "fy": 7.5,
            "cx": 5.5,
            "cy": 4.5,
            "camera_type": "RADIAL",
            "distortion_parameters": [0.15, -0.03],
        }
        loaded = SfmCameraMetadata.from_state_dict(state)
        self.assertEqual(loaded.camera_model, CameraModel.OPENCV_RADTAN_5)
        self.assertAlmostEqual(float(loaded.distortion_coeffs[0]), 0.15)
        self.assertAlmostEqual(float(loaded.distortion_coeffs[1]), -0.03)

    def test_from_state_dict_migrates_legacy_opencv_schema(self):
        state = {
            "img_width": 12,
            "img_height": 10,
            "fx": 8.0,
            "fy": 7.5,
            "cx": 5.5,
            "cy": 4.5,
            "camera_type": "OPENCV",
            "distortion_parameters": [0.1, -0.2, 0.003, -0.004, 0.05],
        }
        loaded = SfmCameraMetadata.from_state_dict(state)
        expected = _packed_radtan5_coeffs(k1=0.1, k2=-0.2, k3=0.05, p1=0.003, p2=-0.004)
        self.assertEqual(loaded.camera_model, CameraModel.OPENCV_RADTAN_5)
        np.testing.assert_allclose(loaded.distortion_coeffs, expected)

    def test_from_state_dict_rejects_unsupported_legacy_camera_type(self):
        state = {
            "img_width": 12,
            "img_height": 10,
            "fx": 8.0,
            "fy": 7.5,
            "cx": 5.5,
            "cy": 4.5,
            "camera_type": "OPENCV_FISHEYE",
            "distortion_parameters": [0.1, 0.2, 0.3, 0.4],
        }
        with self.assertRaisesRegex(ValueError, "Unsupported legacy camera_type"):
            SfmCameraMetadata.from_state_dict(state)

    def test_from_state_dict_requires_legacy_distortion_parameters_key(self):
        state = {
            "img_width": 12,
            "img_height": 10,
            "fx": 8.0,
            "fy": 7.5,
            "cx": 5.5,
            "cy": 4.5,
            "camera_type": "OPENCV",
        }
        with self.assertRaisesRegex(KeyError, "distortion_parameters is missing from state_dict"):
            SfmCameraMetadata.from_state_dict(state)

    def test_from_state_dict_rejects_unknown_camera_model_name(self):
        state = self._make_camera().state_dict()
        state["camera_model"] = "NOT_A_REAL_CAMERA_MODEL"
        with self.assertRaises(KeyError):
            SfmCameraMetadata.from_state_dict(state)

    def test_resize_scales_pinhole_intrinsics_and_preserves_metadata(self):
        camera = self._make_camera()
        resized = camera.resize(new_width=24, new_height=20)
        self.assertEqual(resized.camera_model, camera.camera_model)
        self.assertEqual(resized.width, 24)
        self.assertEqual(resized.height, 20)
        self.assertAlmostEqual(resized.projection_matrix[0, 0], 16.0)
        self.assertAlmostEqual(resized.projection_matrix[1, 1], 15.0)
        self.assertAlmostEqual(resized.projection_matrix[0, 2], 11.0)
        self.assertAlmostEqual(resized.projection_matrix[1, 2], 9.0)
        self.assertEqual(resized.distortion_coeffs.shape, (0,))

    def test_resize_preserves_distortion_coeffs(self):
        camera = self._make_camera(CameraModel.OPENCV_RADTAN_5, _packed_radtan5_coeffs())
        resized = camera.resize(new_width=24, new_height=20)
        self.assertEqual(resized.camera_model, CameraModel.OPENCV_RADTAN_5)
        np.testing.assert_allclose(resized.distortion_coeffs, camera.distortion_coeffs)
        self.assertEqual(resized.width, 24)
        self.assertEqual(resized.height, 20)
        self.assertAlmostEqual(resized.projection_matrix[0, 0], 16.0)
        self.assertAlmostEqual(resized.projection_matrix[1, 1], 15.0)

    def test_resize_rejects_invalid_dimensions(self):
        camera = self._make_camera()
        with self.assertRaisesRegex(ValueError, "positive integers"):
            camera.resize(new_width=0, new_height=10)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            camera.resize(new_width=10, new_height=-1)


if __name__ == "__main__":
    unittest.main()
