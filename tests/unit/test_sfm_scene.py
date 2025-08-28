# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import unittest

import cv2
from fvdb_3dgs.io import DatasetCache, load_colmap_dataset
from fvdb_3dgs.sfm_scene import SfmCameraMetadata, SfmImageMetadata, SfmScene
from fvdb_3dgs.transforms import DownsampleImages


class BasicSfmSceneTest(unittest.TestCase):
    def setUp(self):
        # TODO: Auto-download this dataset if it doesn't exist.
        # NOTE: For now, we assume you've downloaded this dataset. We'll do this automatically
        # when we have access to S3 buckets
        self.dataset_path = pathlib.Path(__file__).parent.parent.parent / "data" / "glomap_gettysburg_small_scaled"

        self.expected_num_images = 154
        self.expected_num_cameras = 5
        self.expected_image_resolutions = {
            1: (10630, 14179),
            2: (10628, 14177),
            3: (10631, 14180),
            4: (10630, 14180),
            5: (10628, 14177),
        }

    def test_dataset_exists(self):
        self.assertTrue(self.dataset_path.exists(), "Dataset path does not exist.")

    def test_sfm_scene_creation_creation(self):

        scene: SfmScene
        cache: DatasetCache
        scene, cache = load_colmap_dataset(self.dataset_path)

        self.assertEqual(len(scene.cameras), self.expected_num_cameras)
        self.assertEqual(len(scene.images), self.expected_num_images)

        for camera_id, camera_metadata in scene.cameras.items():
            self.assertIsInstance(camera_metadata, SfmCameraMetadata)
            expected_h = self.expected_image_resolutions[camera_id][0]
            expected_w = self.expected_image_resolutions[camera_id][1]
            self.assertEqual(camera_metadata.height, expected_h)
            self.assertEqual(camera_metadata.width, expected_w)

        for i, image_metadata in enumerate(scene.images):
            self.assertIsInstance(image_metadata, SfmImageMetadata)
            # These are big images so only test a few of them
            if i % 20 == 0:
                img = cv2.imread(image_metadata.image_path)
                self.assertTrue(img.shape[0] == image_metadata.camera_metadata.height)
                self.assertTrue(img.shape[1] == image_metadata.camera_metadata.width)

    def test_downsample_images(self):
        downsample_factor = 8
        transform = DownsampleImages(downsample_factor)

        scene: SfmScene
        cache: DatasetCache
        scene, cache = load_colmap_dataset(self.dataset_path)

        transformed_scene, cache = transform(scene, cache)

        self.assertIsInstance(transformed_scene, SfmScene)
        self.assertIsInstance(cache, DatasetCache)

        for camera_id, camera_metadata in transformed_scene.cameras.items():
            self.assertIsInstance(camera_metadata, SfmCameraMetadata)
            expected_h = int(self.expected_image_resolutions[camera_id][0] / downsample_factor)
            expected_w = int(self.expected_image_resolutions[camera_id][1] / downsample_factor)
            self.assertEqual(camera_metadata.height, expected_h)
            self.assertEqual(camera_metadata.width, expected_w)

        for i, image_metadata in enumerate(transformed_scene.images):
            self.assertIsInstance(image_metadata, SfmImageMetadata)
            # These are big images so only test a few of them
            if i % 20 == 0:
                img = cv2.imread(image_metadata.image_path)
                self.assertTrue(img.shape[0] == image_metadata.camera_metadata.height)
                self.assertTrue(img.shape[1] == image_metadata.camera_metadata.width)


if __name__ == "__main__":
    unittest.main()
