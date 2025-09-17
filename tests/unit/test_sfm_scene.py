# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import unittest

import cv2
from fvdb_3dgs.sfm_scene import SfmCameraMetadata, SfmImageMetadata, SfmScene
from fvdb_3dgs.transforms import DownsampleImages
import numpy as np
from scipy.spatial.transform import Rotation as R


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

    def test_transform_scene(self):
        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        self.assertEqual(len(scene.images), self.expected_num_images)

        translation = np.array([10.0, 0.0, 0.0])
        rotation = R.from_euler('xyz', 2.0 * np.pi * np.random.rand(3)).as_matrix()
        scaling = np.diag([1.0, 2.0, 1.0])
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation @ scaling
        transform_matrix[:3, 3] = translation

        transformed_scene = scene.apply_transformation_matrix(transform_matrix)

        self.assertTrue(np.allclose(transformed_scene.transformation_matrix, transform_matrix @ scene.transformation_matrix))
        self.assertEqual(len(transformed_scene.images), len(scene.images))
        self.assertEqual(len(transformed_scene.cameras), len(scene.cameras))
        self.assertEqual(len(transformed_scene.points), len(scene.points))

        expected_c2w = np.stack(
            [transform_matrix @ image.camera_to_world_matrix for image in scene.images], axis=0)
        expected_w2c = np.stack(
            [np.linalg.inv(expected_c2w[i]) for i in range(len(scene.images))], axis=0)

        self.assertTrue(np.allclose(transformed_scene.camera_to_world_matrices, expected_c2w))
        self.assertTrue(np.allclose(transformed_scene.world_to_camera_matrices, expected_w2c))
        self.assertTrue(np.allclose(transformed_scene.projection_matrices, scene.projection_matrices))

        expected_positions = expected_c2w[:, :3, 3]
        self.assertTrue(np.allclose(transformed_scene.image_centers, expected_positions))

        expected_points = (transform_matrix[:3, :3] @ scene.points.T + transform_matrix[:3, 3][:, None]).T
        self.assertTrue(np.allclose(transformed_scene.points, expected_points))

    def test_select_images(self):
        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        self.assertEqual(len(scene.images), self.expected_num_images)

        select_every_other = list(range(0, len(scene.images), 2))
        selected_scene = scene.select_images(select_every_other)

        self.assertTrue(np.all(scene.transformation_matrix == selected_scene.transformation_matrix))
        self.assertEqual(selected_scene.num_images, len(select_every_other))
        self.assertEqual(len(selected_scene.projection_matrices), len(select_every_other))
        self.assertEqual(len(selected_scene.image_centers), len(select_every_other))
        self.assertEqual(len(selected_scene.image_sizes), len(select_every_other))
        self.assertEqual(len(selected_scene.images), len(select_every_other))
        self.assertEqual(len(selected_scene.world_to_camera_matrices), len(select_every_other))
        self.assertEqual(len(selected_scene.camera_to_world_matrices), len(select_every_other))
        self.assertEqual(len(selected_scene.cameras), len(scene.cameras))
        self.assertEqual(selected_scene.num_cameras, scene.num_cameras)
        self.assertEqual(len(scene.points), len(selected_scene.points))
        self.assertEqual(len(scene.points_rgb), len(selected_scene.points_rgb))
        self.assertEqual(len(scene.points_err), len(selected_scene.points_err))
        self.assertTrue(np.all(selected_scene.scene_bbox == scene.scene_bbox))

    def test_select_images_with_duplicates(self):
        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        self.assertEqual(len(scene.images), self.expected_num_images)

        select_duplicated = list(range(0, len(scene.images), 2)) * 3
        selected_scene = scene.select_images(select_duplicated)

        self.assertTrue(np.all(scene.transformation_matrix == selected_scene.transformation_matrix))
        self.assertEqual(selected_scene.num_images, len(select_duplicated))
        self.assertEqual(len(selected_scene.projection_matrices), len(select_duplicated))
        self.assertEqual(len(selected_scene.image_centers), len(select_duplicated))
        self.assertEqual(len(selected_scene.image_sizes), len(select_duplicated))
        self.assertEqual(len(selected_scene.images), len(select_duplicated))
        self.assertEqual(len(selected_scene.world_to_camera_matrices), len(select_duplicated))
        self.assertEqual(len(selected_scene.camera_to_world_matrices), len(select_duplicated))
        self.assertEqual(len(selected_scene.cameras), len(scene.cameras))
        self.assertEqual(selected_scene.num_cameras, scene.num_cameras)
        self.assertEqual(len(scene.points), len(selected_scene.points))
        self.assertEqual(len(scene.points_rgb), len(selected_scene.points_rgb))
        self.assertEqual(len(scene.points_err), len(selected_scene.points_err))
        self.assertTrue(np.all(selected_scene.scene_bbox == scene.scene_bbox))

        for i, im in enumerate(selected_scene.images):
            expected_im = scene.images[select_duplicated[i]]
            self.assertEqual(im.image_id, expected_im.image_id)
            self.assertEqual(im.image_path, expected_im.image_path)
            self.assertTrue(np.all(im.camera_to_world_matrix == expected_im.camera_to_world_matrix))
            self.assertTrue(np.all(im.world_to_camera_matrix == expected_im.world_to_camera_matrix))
            self.assertTrue(np.all(im.image_size == expected_im.image_size))
            self.assertTrue(np.all(im.point_indices == expected_im.point_indices))
            self.assertEqual(im.camera_id, expected_im.camera_id)

    def test_filter_images(self):
        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        self.assertEqual(len(scene.images), self.expected_num_images)
        every_other_mask = np.array([i % 2 == 0 for i in range(len(scene.images))])
        filtered_scene = scene.filter_images(every_other_mask)

        self.assertTrue(np.all(scene.transformation_matrix == filtered_scene.transformation_matrix))
        self.assertEqual(filtered_scene.num_images, every_other_mask.sum())
        self.assertEqual(len(filtered_scene.projection_matrices), every_other_mask.sum())
        self.assertEqual(len(filtered_scene.image_centers), every_other_mask.sum())
        self.assertEqual(len(filtered_scene.image_sizes), every_other_mask.sum())
        self.assertEqual(len(filtered_scene.images), every_other_mask.sum())
        self.assertEqual(len(filtered_scene.world_to_camera_matrices), every_other_mask.sum())
        self.assertEqual(len(filtered_scene.camera_to_world_matrices), every_other_mask.sum())
        self.assertEqual(len(filtered_scene.cameras), len(scene.cameras))
        self.assertEqual(filtered_scene.num_cameras, scene.num_cameras)
        self.assertEqual(len(scene.points), len(filtered_scene.points))
        self.assertEqual(len(scene.points_rgb), len(filtered_scene.points_rgb))
        self.assertEqual(len(scene.points_err), len(filtered_scene.points_err))
        self.assertTrue(np.all(filtered_scene.scene_bbox == scene.scene_bbox))

    def test_filter_images_empty_mask(self):   
        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        self.assertEqual(len(scene.images), self.expected_num_images)
        empty_mask = np.zeros(len(scene.images), dtype=bool)
        filtered_scene = scene.filter_images(empty_mask)

        self.assertTrue(np.all(scene.transformation_matrix == filtered_scene.transformation_matrix))
        self.assertEqual(filtered_scene.num_images, empty_mask.sum())
        self.assertEqual(len(filtered_scene.projection_matrices), empty_mask.sum())
        self.assertEqual(len(filtered_scene.image_centers), empty_mask.sum())
        self.assertEqual(len(filtered_scene.image_sizes), empty_mask.sum())
        self.assertEqual(len(filtered_scene.images), empty_mask.sum())
        self.assertEqual(len(filtered_scene.world_to_camera_matrices), empty_mask.sum())
        self.assertEqual(len(filtered_scene.camera_to_world_matrices), empty_mask.sum())
        self.assertEqual(len(filtered_scene.cameras), len(scene.cameras))
        self.assertEqual(filtered_scene.num_cameras, scene.num_cameras)
        self.assertEqual(len(scene.points), len(filtered_scene.points))
        self.assertEqual(len(scene.points_rgb), len(filtered_scene.points_rgb))
        self.assertEqual(len(scene.points_err), len(filtered_scene.points_err))
        self.assertTrue(np.all(filtered_scene.scene_bbox == scene.scene_bbox))

    def test_filter_points(self):
        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

        self.assertEqual(len(scene.images), self.expected_num_images)
        self.assertTrue(scene.points is not None)
        original_num_points = len(scene.points)

        every_other_mask = np.array([i % 2 == 0 for i in range(original_num_points)])

        filtered_scene = scene.filter_points(every_other_mask)

        self.assertTrue(filtered_scene.points is not None)
        self.assertLessEqual(len(filtered_scene.points), every_other_mask.sum())
        self.assertLessEqual(len(filtered_scene.points_rgb), every_other_mask.sum())
        self.assertLessEqual(len(filtered_scene.points_err), every_other_mask.sum())
        for image in filtered_scene.images:
            self.assertTrue(np.all(image.point_indices >= 0))
            self.assertTrue(np.all(image.point_indices < filtered_scene.points.shape[0]))

    def test_load_from_colmap(self):
        scene: SfmScene = SfmScene.from_colmap(self.dataset_path)

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
                assert img is not None
                self.assertTrue(img.shape[0] == image_metadata.camera_metadata.height)
                self.assertTrue(img.shape[1] == image_metadata.camera_metadata.width)


if __name__ == "__main__":
    unittest.main()
