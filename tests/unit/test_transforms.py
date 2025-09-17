# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import unittest

import cv2
import numpy as np
from fvdb_3dgs.sfm_scene import SfmCameraMetadata, SfmImageMetadata, SfmScene
from fvdb_3dgs.transforms import DownsampleImages, FilterImagesWithLowPoints, PercentileFilterPoints, CropScene


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

    def test_downsample_images(self):
        downsample_factor = 16
        transform = DownsampleImages(downsample_factor)

        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        transformed_scene = transform(scene)

        self.assertIsInstance(transformed_scene, SfmScene)

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
                assert img is not None
                self.assertTrue(img.shape[0] == image_metadata.camera_metadata.height)
                self.assertTrue(img.shape[1] == image_metadata.camera_metadata.width)
    
    def test_filter_images_with_low_points(self):
        min_num_points = 300
        transform = FilterImagesWithLowPoints(min_num_points)

        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        transformed_scene = transform(scene)

        self.assertIsInstance(transformed_scene, SfmScene)

        for i, image_metadata in enumerate(transformed_scene.images):
            self.assertIsInstance(image_metadata, SfmImageMetadata)
            self.assertTrue(image_metadata.point_indices.shape[0] > min_num_points)

    def test_filter_images_with_low_points_delete_all_images(self):
        min_num_points = 1_000_000_000
        transform = FilterImagesWithLowPoints(min_num_points)

        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        transformed_scene = transform(scene)

        self.assertIsInstance(transformed_scene, SfmScene)
        self.assertEqual(len(transformed_scene.images), 0)
        for i, image_metadata in enumerate(transformed_scene.images):
            self.assertIsInstance(image_metadata, SfmImageMetadata)
            self.assertTrue(image_metadata.point_indices.shape[0] > min_num_points)

    def test_filter_images_with_low_points_no_images_removed(self):
        min_num_points = 0
        transform = FilterImagesWithLowPoints(min_num_points)

        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        transformed_scene = transform(scene)

        self.assertIsInstance(transformed_scene, SfmScene)
        self.assertEqual(len(transformed_scene.images), len(scene.images))
        for i, image_metadata in enumerate(transformed_scene.images):
            self.assertIsInstance(image_metadata, SfmImageMetadata)
            self.assertTrue(image_metadata.point_indices.shape[0] > min_num_points)

    def test_percentile_filter_points(self):
        percentile_min = (5, 5, 5)
        percentile_max = (95, 95, 95)
        transform = PercentileFilterPoints(percentile_min, percentile_max)

        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        transformed_scene = transform(scene)

        self.assertIsInstance(transformed_scene, SfmScene)

        min_x = np.percentile(scene.points[:, 0], percentile_min[0])
        max_x = np.percentile(scene.points[:, 0], percentile_max[0])
        min_y = np.percentile(scene.points[:, 1], percentile_min[1])
        max_y = np.percentile(scene.points[:, 1], percentile_max[1])
        min_z = np.percentile(scene.points[:, 2], percentile_min[2])
        max_z = np.percentile(scene.points[:, 2], percentile_max[2])

        self.assertTrue(transformed_scene.points.shape[0] < scene.points.shape[0])
        self.assertTrue(transformed_scene.points.shape[0] > 0)
        self.assertTrue(transformed_scene.points_err.shape[0] == transformed_scene.points.shape[0])
        for i, image_metadata in enumerate(transformed_scene.images):
            self.assertIsInstance(image_metadata, SfmImageMetadata)
            self.assertTrue(np.all(image_metadata.point_indices >= 0))
            self.assertTrue(np.all(image_metadata.point_indices < transformed_scene.points.shape[0]))

        self.assertTrue(np.all(transformed_scene.points[:, 0] > min_x) and np.all(transformed_scene.points[:, 0] < max_x))
        self.assertTrue(np.all(transformed_scene.points[:, 1] > min_y) and np.all(transformed_scene.points[:, 1] < max_y))
        self.assertTrue(np.all(transformed_scene.points[:, 2] > min_z) and np.all(transformed_scene.points[:, 2] < max_z))
    
    def test_percentile_filter_points_no_points_removed(self):
        percentile_min = (0, 0, 0)
        percentile_max = (100, 100, 100)
        transform = PercentileFilterPoints(percentile_min, percentile_max)

        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        transformed_scene = transform(scene)

        self.assertIsInstance(transformed_scene, SfmScene)

        self.assertTrue(transformed_scene.points.shape[0] == scene.points.shape[0])
        self.assertTrue(transformed_scene.points_err.shape[0] == scene.points_err.shape[0])
        for i, image_metadata in enumerate(transformed_scene.images):
            self.assertIsInstance(image_metadata, SfmImageMetadata)
            self.assertTrue(np.all(image_metadata.point_indices >= 0))
            self.assertTrue(np.all(image_metadata.point_indices < transformed_scene.points.shape[0]))
            self.assertTrue(np.all(image_metadata.point_indices == scene.images[i].point_indices))
            self.assertTrue(np.all(image_metadata.point_indices == scene.images[i].point_indices))
        self.assertTrue(np.all(transformed_scene.points == scene.points))

    def test_percentile_filter_points_all_points_removed_is_an_error(self):
        percentile_min = (10, 10, 10)
        percentile_max = (9, 9, 9)
        transform = PercentileFilterPoints(percentile_min, percentile_max)

        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        with self.assertRaises(ValueError):
            transformed_scene = transform(scene)

    def test_crop_scene(self):
        # These bounds were determined by looking at the point cloud in a 3D viewer
        # and finding a reasonable bounding box that would crop out some points
        # but still leave a good number of points.
        # NOTE: These bounds are specific to this dataset and won't work for other datasets.
        # The format is [min_x, min_y, min_z, max_x, max_y, max_z]
        # NOTE: The dataset is in EPSG:26917 (UTM zone 17N) so the bounds are in meters
        # and are quite large.
        min_bound = [ 1075540.25 , -4780800.5  ,  4043418.775] 
        max_bound = [ 1090150.75 , -4772843.5  ,  4058591.925]
        dowsample_transform = DownsampleImages(16) # Cropping with large images is very slow, so downsample first
        transform = CropScene(min_bound + max_bound)

        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)
        scene = dowsample_transform(scene)
        transformed_scene = transform(scene)

        self.assertIsInstance(transformed_scene, SfmScene)

        self.assertTrue(transformed_scene.points.shape[0] < scene.points.shape[0])
        self.assertTrue(transformed_scene.points.shape[0] > 0)
        self.assertTrue(transformed_scene.points_err.shape[0] == transformed_scene.points.shape[0])
        for i, image_metadata in enumerate(transformed_scene.images):
            self.assertIsInstance(image_metadata, SfmImageMetadata)
            self.assertEqual(scene.images[i].image_id, image_metadata.image_id)
            self.assertEqual(scene.images[i].mask_path, "")
            self.assertTrue(len(image_metadata.mask_path) > 0)
            self.assertTrue(np.all(image_metadata.point_indices >= 0))
            self.assertTrue(np.all(image_metadata.point_indices < transformed_scene.points.shape[0]))
            self.assertTrue(np.all(image_metadata.point_indices < scene.points.shape[0]))

if __name__ == "__main__":
    unittest.main()
