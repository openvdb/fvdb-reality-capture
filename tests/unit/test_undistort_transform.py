# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fvdb import CameraModel

from fvdb_reality_capture.sfm_scene import SfmCache, SfmCameraMetadata, SfmPosedImageMetadata, SfmScene
from fvdb_reality_capture.transforms import Compose, DownsampleImages, UndistortImages

from .common import sfm_scenes_match


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


class UndistortImagesTransformTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_scene(
        self,
        camera_model: CameraModel = CameraModel.OPENCV_RADTAN_5,
        distortion_coeffs: np.ndarray | None = None,
        with_mask: bool = True,
    ) -> tuple[SfmScene, np.ndarray, np.ndarray | None]:
        image_path = self.root / "image.png"
        mask_path = self.root / "mask.png"

        height, width = 14, 18
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.uint8), np.arange(height, dtype=np.uint8), indexing="xy")
        image = np.stack([grid_x * 10, grid_y * 15, (grid_x + grid_y) * 7], axis=-1).astype(np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[2:12, 3:15] = 255

        self.assertTrue(cv2.imwrite(str(image_path), image))
        if with_mask:
            self.assertTrue(cv2.imwrite(str(mask_path), mask))

        camera_metadata = SfmCameraMetadata(
            img_width=width,
            img_height=height,
            fx=11.0,
            fy=10.5,
            cx=8.2,
            cy=6.7,
            camera_model=camera_model,
            distortion_coeffs=np.array([], dtype=np.float32) if distortion_coeffs is None else distortion_coeffs,
        )
        image_metadata = SfmPosedImageMetadata(
            world_to_camera_matrix=np.eye(4, dtype=np.float32),
            camera_to_world_matrix=np.eye(4, dtype=np.float32),
            camera_metadata=camera_metadata,
            camera_id=1,
            image_path=str(image_path),
            mask_path=str(mask_path) if with_mask else "",
            point_indices=np.array([], dtype=np.int64),
            image_id=0,
        )
        cache = SfmCache.get_cache(self.root / "cache_root", "unit_test_cache", "Unit test cache")
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
        return scene, image, (mask if with_mask else None)

    def _make_multi_image_scene(
        self,
        image_ids: list[int],
        camera_model: CameraModel = CameraModel.OPENCV_RADTAN_5,
        distortion_coeffs: np.ndarray | None = None,
        with_mask: bool = True,
    ) -> SfmScene:
        height, width = 14, 18
        camera_metadata = SfmCameraMetadata(
            img_width=width,
            img_height=height,
            fx=11.0,
            fy=10.5,
            cx=8.2,
            cy=6.7,
            camera_model=camera_model,
            distortion_coeffs=np.array([], dtype=np.float32) if distortion_coeffs is None else distortion_coeffs,
        )
        images: list[SfmPosedImageMetadata] = []
        for i, image_id in enumerate(image_ids):
            image_path = self.root / f"image_{image_id}.png"
            mask_path = self.root / f"mask_{image_id}.png"
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.uint8), np.arange(height, dtype=np.uint8), indexing="xy"
            )
            image = np.stack([grid_x * 10 + i, grid_y * 15 + i, (grid_x + grid_y) * 7 + i], axis=-1).astype(np.uint8)
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[2:12, 3:15] = 255
            self.assertTrue(cv2.imwrite(str(image_path), image))
            if with_mask:
                self.assertTrue(cv2.imwrite(str(mask_path), mask))
            images.append(
                SfmPosedImageMetadata(
                    world_to_camera_matrix=np.eye(4, dtype=np.float32),
                    camera_to_world_matrix=np.eye(4, dtype=np.float32),
                    camera_metadata=camera_metadata,
                    camera_id=1,
                    image_path=str(image_path),
                    mask_path=str(mask_path) if with_mask else "",
                    point_indices=np.array([], dtype=np.int64),
                    image_id=image_id,
                )
            )
        cache = SfmCache.get_cache(self.root / "cache_root", "unit_test_cache_multi", "Unit test cache")
        return SfmScene(
            cameras={1: camera_metadata},
            images=images,
            points=np.zeros((0, 3), dtype=np.float32),
            points_err=np.zeros((0,), dtype=np.float32),
            points_rgb=np.zeros((0, 3), dtype=np.uint8),
            scene_bbox=None,
            transformation_matrix=np.eye(4, dtype=np.float32),
            cache=cache,
        )

    def test_private_packed_distortion_coeffs_to_opencv_radtan5_ordering(self):
        packed = _packed_radtan5_coeffs(k1=0.1, k2=-0.2, k3=0.05, p1=0.003, p2=-0.004)
        converted = UndistortImages._packed_distortion_coeffs_to_opencv_radtan5(packed)
        np.testing.assert_allclose(converted, np.array([0.1, -0.2, 0.003, -0.004, 0.05], dtype=np.float32))

    def test_undistort_transform_updates_camera_and_assets(self):
        scene, image, mask = self._make_scene(distortion_coeffs=_packed_radtan5_coeffs())
        transform = UndistortImages()

        transformed_scene = transform(scene)
        transformed_camera = transformed_scene.cameras[1]

        distortion = np.array([0.1, -0.05, 0.002, -0.003, 0.01], dtype=np.float32)
        expected_proj, expected_roi = cv2.getOptimalNewCameraMatrix(
            scene.cameras[1].projection_matrix,
            distortion,
            (scene.cameras[1].width, scene.cameras[1].height),
            0.0,
        )
        expected_map_x, expected_map_y = cv2.initUndistortRectifyMap(
            scene.cameras[1].projection_matrix,
            distortion,
            None,
            expected_proj,
            (scene.cameras[1].width, scene.cameras[1].height),
            cv2.CV_32FC1,  # type: ignore[arg-type]
        )
        roi_x, roi_y, roi_w, roi_h = (int(v) for v in expected_roi)
        expected_proj = expected_proj.copy()
        expected_proj[0, 2] -= roi_x
        expected_proj[1, 2] -= roi_y

        expected_image = cv2.remap(image, expected_map_x, expected_map_y, interpolation=cv2.INTER_LINEAR)
        expected_image = expected_image[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        assert mask is not None
        expected_mask = cv2.remap(mask, expected_map_x, expected_map_y, interpolation=cv2.INTER_NEAREST)
        expected_mask = expected_mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

        self.assertEqual(transformed_camera.camera_model, CameraModel.PINHOLE)
        self.assertEqual(transformed_camera.distortion_coeffs.shape, (0,))
        self.assertEqual(transformed_camera.width, roi_w)
        self.assertEqual(transformed_camera.height, roi_h)
        np.testing.assert_allclose(transformed_camera.projection_matrix, expected_proj)

        saved_image = cv2.imread(transformed_scene.images[0].image_path, cv2.IMREAD_UNCHANGED)
        saved_mask = cv2.imread(transformed_scene.images[0].mask_path, cv2.IMREAD_UNCHANGED)
        assert saved_image is not None
        assert saved_mask is not None
        np.testing.assert_array_equal(saved_image, expected_image)
        np.testing.assert_array_equal(saved_mask, expected_mask)

    def test_undistort_transform_can_be_composed_with_downsample_images(self):
        scene, _, _ = self._make_scene(distortion_coeffs=_packed_radtan5_coeffs())
        transformed_scene = Compose(UndistortImages(), DownsampleImages(2))(scene)

        transformed_camera = transformed_scene.cameras[1]
        self.assertEqual(transformed_camera.camera_model, CameraModel.PINHOLE)
        self.assertEqual(transformed_camera.distortion_coeffs.shape, (0,))
        saved_image = cv2.imread(transformed_scene.images[0].image_path, cv2.IMREAD_UNCHANGED)
        assert saved_image is not None
        self.assertEqual(saved_image.shape[1], transformed_camera.width)
        self.assertEqual(saved_image.shape[0], transformed_camera.height)

    def test_undistort_transform_reuses_cached_outputs(self):
        scene, _, _ = self._make_scene(distortion_coeffs=_packed_radtan5_coeffs())
        transform = UndistortImages()

        first_scene = transform(scene)
        second_scene = transform(scene)

        self.assertTrue(sfm_scenes_match(first_scene, second_scene))
        self.assertEqual(first_scene.images[0].image_path, second_scene.images[0].image_path)

    def test_undistort_transform_reuses_cached_outputs_for_noncontiguous_image_ids(self):
        scene = self._make_multi_image_scene(image_ids=[1, 3], distortion_coeffs=_packed_radtan5_coeffs())
        transform = UndistortImages()

        first_scene = transform(scene)
        with patch.object(
            UndistortImages,
            "_undistort_and_crop",
            side_effect=AssertionError("cache should be reused without regenerating undistortion outputs"),
        ):
            second_scene = transform(scene)

        self.assertTrue(sfm_scenes_match(first_scene, second_scene))
        self.assertEqual(
            [pathlib.Path(image.image_path).name for image in first_scene.images],
            [pathlib.Path(image.image_path).name for image in second_scene.images],
        )

    def test_undistort_transform_is_noop_for_pinhole_scenes(self):
        scene, _, _ = self._make_scene(
            camera_model=CameraModel.PINHOLE, distortion_coeffs=np.array([], dtype=np.float32)
        )
        transform = UndistortImages()
        transformed_scene = transform(scene)
        self.assertIs(transformed_scene, scene)

    def test_undistort_transform_rejects_unsupported_distorted_camera_models(self):
        scene, _, _ = self._make_scene(
            camera_model=CameraModel.OPENCV_RATIONAL_8, distortion_coeffs=_packed_radtan5_coeffs()
        )
        transform = UndistortImages()
        with self.assertRaisesRegex(NotImplementedError, "does not support image undistortion"):
            transform(scene)


if __name__ == "__main__":
    unittest.main()
