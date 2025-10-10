# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
import tempfile
import unittest

import numpy as np

import fvdb_reality_capture as frc


class GaussianSplatReconstructionTests(unittest.TestCase):
    def setUp(self):
        # Auto-download this dataset if it doesn't exist.
        self.dataset_path = pathlib.Path(__file__).parent.parent.parent / "data" / "360_v2" / "counter"
        if not self.dataset_path.exists():
            frc.tools.download_example_data("mipnerf360", self.dataset_path.parent)

        self.sfm_scene = frc.SfmScene.from_colmap(self.dataset_path)
        self.scene_transform = frc.transforms.Compose(
            frc.transforms.NormalizeScene("pca"),
            frc.transforms.DownsampleImages(4),
        )
        self.sfm_scene = self.scene_transform(self.sfm_scene)
        self.sfm_scene = self.sfm_scene.filter_images(np.arange(0, len(self.sfm_scene.images), 4))

    def test_run_training_with_no_saving(self):

        short_config = frc.training.SceneOptimizationConfig(
            max_epochs=1,
            refine_start_epoch=5,
            eval_at_percent=[],
        )

        runner = frc.GaussianSplatReconstruction.from_sfm_scene(
            self.sfm_scene,
            config=short_config,
            use_every_n_as_val=2,
        )

        runner.train()

        self.assertEqual(runner.model.num_gaussians, self.sfm_scene.points.shape[0])

    def test_run_training_with_saving(self):

        short_config = frc.training.SceneOptimizationConfig(
            max_epochs=2,
            refine_start_epoch=5,
            eval_at_percent=[50, 100],
            save_at_percent=[100],
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            results_path = pathlib.Path(tmpdirname)
            runner = frc.GaussianSplatReconstruction.from_sfm_scene(
                self.sfm_scene,
                config=short_config,
                use_every_n_as_val=2,
                results_path=results_path,
            )
            num_val = len(np.arange(0, len(self.sfm_scene.images), 2))
            num_train = len(self.sfm_scene.images) - num_val
            self.assertEqual(len(runner.training_dataset), num_train)
            self.assertEqual(len(runner.validation_dataset), num_val)

            self.assertTrue(results_path.exists())
            self.assertTrue(runner.stats_path is not None and runner.stats_path.exists())
            self.assertTrue(runner.checkpoints_path is not None and runner.checkpoints_path.exists())
            self.assertTrue(runner.image_render_path is None)

            assert runner.stats_path is not None
            assert runner.checkpoints_path is not None
            num_stats_files = sum(1 for item in runner.stats_path.iterdir() if item.is_file())
            num_ckpt_files = sum(1 for item in runner.checkpoints_path.iterdir() if item.is_file())
            self.assertEqual(num_stats_files, 0)
            self.assertEqual(num_ckpt_files, 0)

            runner.train()

            num_stats_files = sum(1 for item in runner.stats_path.iterdir() if item.is_file())
            num_ckpt_files = sum(1 for item in runner.checkpoints_path.iterdir() if item.is_file())
            self.assertEqual(num_stats_files, 2)
            self.assertEqual(num_ckpt_files, 4)

        self.assertEqual(runner.model.num_gaussians, self.sfm_scene.points.shape[0])

    def test_run_training_with_saving_and_image_rendering(self):

        short_config = frc.training.SceneOptimizationConfig(
            max_epochs=2,
            refine_start_epoch=5,
            eval_at_percent=[50, 100],
            save_at_percent=[100],
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            results_path = pathlib.Path(tmpdirname)
            runner = frc.GaussianSplatReconstruction.from_sfm_scene(
                self.sfm_scene,
                config=short_config,
                use_every_n_as_val=2,
                results_path=results_path,
                save_eval_images=True,
            )
            num_val = len(np.arange(0, len(self.sfm_scene.images), 2))
            num_train = len(self.sfm_scene.images) - num_val
            self.assertEqual(len(runner.training_dataset), num_train)
            self.assertEqual(len(runner.validation_dataset), num_val)

            self.assertTrue(results_path.exists())
            self.assertTrue(runner.stats_path is not None and runner.stats_path.exists())
            self.assertTrue(runner.checkpoints_path is not None and runner.checkpoints_path.exists())
            self.assertTrue(runner.image_render_path is not None and runner.image_render_path.exists())

            assert runner.stats_path is not None
            assert runner.checkpoints_path is not None
            assert runner.image_render_path is not None
            num_stats_files = len([item for item in runner.stats_path.iterdir() if item.is_file()])
            num_ckpt_files = len([item for item in runner.checkpoints_path.iterdir() if item.is_file()])
            num_image_files = len([item for item in runner.image_render_path.iterdir() if item.is_file()])

            self.assertEqual(num_stats_files, 0)
            self.assertEqual(num_ckpt_files, 0)
            self.assertEqual(num_image_files, 0)

            runner.train()

            num_stats_files = len([item for item in runner.stats_path.iterdir() if item.is_file()])
            num_ckpt_files = len([item for item in runner.checkpoints_path.iterdir() if item.is_file()])
            num_image_files = len([item for item in runner.image_render_path.iterdir() if item.is_file()])
            num_image_directories = len([item for item in runner.image_render_path.iterdir() if item.is_dir()])

            self.assertEqual(num_stats_files, 2)
            self.assertEqual(num_ckpt_files, 4)  # 2 numbered + 2 symlinks
            self.assertEqual(num_image_files, 0)  # all images in subdirectories
            self.assertEqual(num_image_directories, 2)  # one per eval

            for item in runner.image_render_path.iterdir():
                if item.is_dir():
                    num_image_files = sum(1 for subitem in item.iterdir() if subitem.is_file())
                    self.assertEqual(num_image_files, num_val)

        self.assertEqual(runner.model.num_gaussians, self.sfm_scene.points.shape[0])

    def test_checkpoint_loading(self):

        short_config = frc.training.SceneOptimizationConfig(
            max_epochs=4,
            refine_start_epoch=5,
            eval_at_percent=[],
            save_at_percent=[25, 50, 75, 100],
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            results_path = pathlib.Path(tmpdirname)
            runner = frc.GaussianSplatReconstruction.from_sfm_scene(
                self.sfm_scene,
                config=short_config,
                use_every_n_as_val=2,
                results_path=results_path,
                save_eval_images=False,
            )
            num_val = len(np.arange(0, len(self.sfm_scene.images), 2))
            num_train = len(self.sfm_scene.images) - num_val
            self.assertEqual(len(runner.training_dataset), num_train)
            self.assertEqual(len(runner.validation_dataset), num_val)

            self.assertTrue(results_path.exists())
            self.assertTrue(runner.stats_path is not None and runner.stats_path.exists())
            self.assertTrue(runner.checkpoints_path is not None and runner.checkpoints_path.exists())
            self.assertTrue(runner.image_render_path is None)

            assert runner.stats_path is not None
            assert runner.checkpoints_path is not None
            num_stats_files = len([item for item in runner.stats_path.iterdir() if item.is_file()])
            num_ckpt_files = len([item for item in runner.checkpoints_path.iterdir() if item.is_file()])

            self.assertEqual(num_stats_files, 0)
            self.assertEqual(num_ckpt_files, 0)

            runner.train()

            all_saved_files_before_delete = [item for item in runner.checkpoints_path.iterdir() if item.is_file()]

            num_ckpt_files = len([item for item in runner.checkpoints_path.iterdir() if item.is_file()])
            self.assertEqual(num_stats_files, 0)
            self.assertEqual(num_ckpt_files, 10)  # 1 ply + 1 checkpoint per save + 2 symlinks

            # Delete all but the first checkpoint
            lowest_checkpoint_path = sorted(
                [
                    item
                    for item in all_saved_files_before_delete
                    if item.suffix == ".pt" and not item.name.endswith("_final.pt")
                ]
            )[0]
            lowest_ply_path = sorted(
                [
                    item
                    for item in all_saved_files_before_delete
                    if item.suffix == ".ply" and not item.name.endswith("_final.ply")
                ]
            )[0]

            for item in runner.checkpoints_path.iterdir():
                if item in (lowest_checkpoint_path, lowest_ply_path):
                    continue
                if item.is_file():
                    item.unlink()

            num_ckpt_files = len([item for item in runner.checkpoints_path.iterdir() if item.is_file()])
            self.assertEqual(num_ckpt_files, 2)  # 1 ply + 1 checkpoint per save + 2 symlinks

            # Now try to load the checkpoint
            runner2 = frc.GaussianSplatReconstruction.from_checkpoint(
                lowest_checkpoint_path, device=runner.model.device, run_name_suffix="_resumed"
            )

            assert runner2.checkpoints_path is not None
            assert runner2.stats_path is not None
            assert runner2.image_render_path is None
            expected_results_path = results_path / (runner._run_name + "_resumed")
            self.assertEqual(str(runner2.results_path), str(expected_results_path))
            self.assertEqual(len(runner2.training_dataset), num_train)
            self.assertEqual(len(runner2.validation_dataset), num_val)
            num_stats_files = len([item for item in runner2.stats_path.iterdir() if item.is_file()])
            num_ckpt_files = len([item for item in runner2.checkpoints_path.iterdir() if item.is_file()])
            self.assertEqual(num_ckpt_files, 0)
            self.assertEqual(num_stats_files, 0)

            runner2.train()

            num_stats_files = len([item for item in runner2.stats_path.iterdir() if item.is_file()])
            num_ckpt_files = len([item for item in runner2.checkpoints_path.iterdir() if item.is_file()])
            self.assertEqual(num_ckpt_files, 8)
            self.assertEqual(num_stats_files, 0)

            result_set = {item.name for item in runner2.checkpoints_path.iterdir() if item.is_file()}
            start_set = {lowest_checkpoint_path.name, lowest_ply_path.name}
            result_set = result_set.union(start_set)

            expected_set = {item.name for item in all_saved_files_before_delete}
            self.assertEqual(expected_set.intersection(result_set), result_set)

        self.assertEqual(runner.model.num_gaussians, self.sfm_scene.points.shape[0])
