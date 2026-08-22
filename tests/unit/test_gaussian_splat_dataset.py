# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import tempfile
import unittest

import cv2
import numpy as np
import torch
import torch.utils.data

from fvdb_reality_capture import CameraModel
from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset
from fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction import _collate_cached_sfm_batch
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

    def _make_scene(self, with_mask: bool = False) -> tuple[SfmScene, SfmCameraMetadata]:
        image_path = self.root / "image.png"
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), image))
        mask_path = ""
        if with_mask:
            mask = np.zeros((8, 10), dtype=np.uint8)
            mask[:4] = 255
            mask_file = self.root / "mask.png"
            self.assertTrue(cv2.imwrite(str(mask_file), mask))
            mask_path = str(mask_file)

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
            mask_path=mask_path,
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

    def test_dataset_caches_images_and_masks_in_shared_memory(self):
        scene, _ = self._make_scene(with_mask=True)

        dataset = SfmDataset(scene, cache_images=True)
        first = dataset[0]

        self.assertTrue(dataset.images_cached)
        self.assertIsInstance(first["image"], torch.Tensor)
        self.assertIsInstance(first["mask"], torch.Tensor)
        self.assertTrue(first["image"].is_shared())
        self.assertTrue(first["mask"].is_shared())
        self.assertEqual(tuple(first["image"].shape), (8, 10, 3))
        self.assertEqual(tuple(first["mask"].shape), (8, 10))
        self.assertTrue(torch.all(first["image"] == 0))
        self.assertTrue(torch.all(first["mask"][:4]))
        self.assertTrue(torch.all(~first["mask"][4:]))

        # Changing the source files after construction must not change cached data.
        self.assertTrue(cv2.imwrite(scene.images[0].image_path, np.full((8, 10, 3), 255, dtype=np.uint8)))
        self.assertTrue(cv2.imwrite(scene.images[0].mask_path, np.zeros((8, 10), dtype=np.uint8)))
        second = dataset[0]
        self.assertEqual(first["image"].data_ptr(), second["image"].data_ptr())
        self.assertEqual(first["mask"].data_ptr(), second["mask"].data_ptr())
        self.assertTrue(torch.all(second["image"] == 0))
        self.assertTrue(torch.all(second["mask"][:4]))

        uncached = SfmDataset(scene)[0]
        self.assertIsInstance(uncached["image"], np.ndarray)
        self.assertTrue(np.all(uncached["image"] == 255))
        self.assertFalse(np.any(uncached["mask"]))

    def test_cached_batch_collation_does_not_copy_shared_rasters(self):
        scene, _ = self._make_scene(with_mask=True)
        datum = SfmDataset(scene, cache_images=True)[0]

        collated = _collate_cached_sfm_batch([datum])

        self.assertEqual(tuple(collated["image"].shape), (1, 8, 10, 3))
        self.assertEqual(tuple(collated["mask"].shape), (1, 8, 10))
        self.assertEqual(collated["image"].data_ptr(), datum["image"].data_ptr())
        self.assertEqual(collated["mask"].data_ptr(), datum["mask"].data_ptr())

    def test_dataloader_worker_reads_cached_shared_rasters(self):
        scene, _ = self._make_scene(with_mask=True)
        dataset = SfmDataset(scene, cache_images=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=_collate_cached_sfm_batch,
        )

        batches = list(dataloader)

        self.assertEqual(len(batches), 1)
        self.assertTrue(batches[0]["image"].is_shared())
        self.assertTrue(batches[0]["mask"].is_shared())
        self.assertTrue(torch.all(batches[0]["image"] == 0))
        self.assertTrue(torch.all(batches[0]["mask"][:, :4]))
