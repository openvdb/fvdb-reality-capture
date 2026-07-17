# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import tempfile
import unittest
from collections import OrderedDict
from unittest.mock import patch

import numpy as np

from fvdb_reality_capture import CameraModel
from fvdb_reality_capture.sfm_scene.adapter import COLMAPAdapter
from fvdb_reality_capture.sfm_scene._load_colmap_scene import load_colmap_scene


class FakeCamera:
    def __init__(self, model: str, width: int, height: int, params: np.ndarray):
        self.model = model
        self.width = width
        self.height = height
        self.params = np.asarray(params, dtype=np.float64)


class FakeTransform:
    def __init__(self, matrix: np.ndarray):
        self._matrix = np.asarray(matrix, dtype=np.float64)

    def matrix(self) -> np.ndarray:
        return self._matrix


class FakeImage:
    def __init__(self, name: str, camera_id: int, translation: np.ndarray):
        self.name = name
        self.camera_id = camera_id
        self.has_pose = True
        self._world_to_camera = np.eye(4, dtype=np.float64)
        self._world_to_camera[:3, 3] = translation

    def cam_from_world(self) -> FakeTransform:
        return FakeTransform(self._world_to_camera[:3])


class FakeTrackElement:
    def __init__(self, image_id: int, point2D_idx: int):
        self.image_id = image_id
        self.point2D_idx = point2D_idx


class FakeTrack:
    def __init__(self, elements: list[tuple[int, int]]):
        self.elements = [FakeTrackElement(image_id, point2D_idx) for image_id, point2D_idx in elements]


class FakePoint3D:
    def __init__(self, xyz: np.ndarray, color: np.ndarray, error: float, track: list[tuple[int, int]]):
        self.xyz = xyz
        self.color = color
        self.error = error
        self.track = FakeTrack(track)


class FakeReconstruction:
    def __init__(
        self,
        cameras: dict[int, FakeCamera],
        images: OrderedDict[int, FakeImage],
        points3D: dict[int, FakePoint3D],
    ):
        self.cameras = cameras
        self.images = images
        self.points3D = points3D

    def reg_image_ids(self) -> list[int]:
        return list(self.images.keys())


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
                cam = FakeCamera(camera_type, width=640, height=480, params=params)

                camera_model, distortion_coeffs = COLMAPAdapter.camera_model_and_distortion_coeffs(cam)

                self.assertEqual(camera_model, expected_model)
                np.testing.assert_allclose(distortion_coeffs, expected_coeffs)

    def test_opencv_fisheye_camera_is_rejected(self):
        cam = FakeCamera(
            "OPENCV_FISHEYE",
            width=640,
            height=480,
            params=np.array([500.0, 505.0, 320.0, 240.0, 0.1, -0.2, 0.003, -0.004], dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "OPENCV_FISHEYE cameras are not supported"):
            COLMAPAdapter.camera_model_and_distortion_coeffs(cam)

    def test_load_colmap_scene_preserves_sorted_images_camera_reuse_masks_and_visible_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            colmap_path = pathlib.Path(tmpdir)
            (colmap_path / "images").mkdir()
            (colmap_path / "masks").mkdir()
            (colmap_path / "masks" / "a.png").write_bytes(b"")
            (colmap_path / "masks" / "b.jpg").write_bytes(b"")

            points3D = {
                11: FakePoint3D(
                    xyz=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                    color=np.array([255, 0, 0], dtype=np.uint8),
                    error=0.1,
                    track=[(7, 0), (100, 3)],
                ),
                13: FakePoint3D(
                    xyz=np.array([4.0, 5.0, 6.0], dtype=np.float64),
                    color=np.array([0, 255, 0], dtype=np.uint8),
                    error=0.2,
                    track=[(42, 1), (100, 2)],
                ),
            }
            reconstruction = FakeReconstruction(
                cameras={
                    1: FakeCamera("PINHOLE", width=640, height=480, params=np.array([500.0, 505.0, 320.0, 240.0])),
                    2: FakeCamera(
                        "OPENCV",
                        width=800,
                        height=600,
                        params=np.array([700.0, 710.0, 400.0, 300.0, 0.1, -0.2, 0.003, -0.004]),
                    ),
                },
                images=OrderedDict(
                    [
                        (42, FakeImage("z.jpg", 1, np.array([1.0, 2.0, 3.0]))),
                        (7, FakeImage("a.jpg", 2, np.array([0.0, 0.0, 1.0]))),
                        (100, FakeImage("b.jpg", 1, np.array([-1.0, 0.5, 2.0]))),
                    ]
                ),
                points3D=points3D,
            )

            with patch(
                "fvdb_reality_capture.sfm_scene.adapter.COLMAPAdapter._load_reconstruction",
                return_value=reconstruction,
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

            np.testing.assert_allclose(points, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
            np.testing.assert_allclose(points_err, np.array([0.1, 0.2], dtype=np.float32))
            np.testing.assert_array_equal(points_rgb, np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8))
            self.assertTrue((colmap_path / "_cache").exists())
            self.assertTrue(cache.has_file("visible_points_per_image"))
