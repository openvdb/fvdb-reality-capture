# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
import unittest
from typing import Any

import fvdb
import numpy as np
import torch

import fvdb_reality_capture as frc
from fvdb_reality_capture import radiance_fields


class MockWriter(radiance_fields.GaussianSplatReconstructionBaseWriter):
    def __init__(self):
        super().__init__()
        self.metric_log: list[tuple[int, str, float]] = []
        self.checkpoint_log: list[tuple[int, str, dict[str, float]]] = []
        self.ply_log: list[tuple[int, str]] = []
        self.image_log: list[tuple[int, str, torch.Size, torch.dtype]] = []

    def log_metric(self, global_step: int, metric_name: str, metric_value: float) -> None:
        self.metric_log.append((global_step, metric_name, metric_value))

    def save_checkpoint(self, global_step: int, checkpoint_name: str, checkpoint: dict[str, Any]) -> None:
        self.checkpoint_log.append((global_step, checkpoint_name, checkpoint))

    def save_ply(self, global_step: int, ply_name: str, model: fvdb.GaussianSplat3d, metadata: dict[str, Any]) -> None:
        self.ply_log.append((global_step, ply_name))

    def save_image(self, global_step: int, image_name: str, image: torch.Tensor, jpeg_quality: int = 98) -> None:
        self.image_log.append((global_step, image_name, image.shape, image.dtype))


class GaussianSplatReconstructionTests(unittest.TestCase):
    def setUp(self):
        # Auto-download this dataset if it doesn't exist.
        self.dataset_root = pathlib.Path(__file__).parent.parent.parent / "data"
        print("datasets root is ", self.dataset_root)
        self.dataset_path = self.dataset_root / "360_v2" / "counter"
        print("dataset path is ", self.dataset_path)
        if not self.dataset_path.exists():
            frc.tools.download_example_data("mipnerf360", self.dataset_root)

        self.sfm_scene = frc.sfm_scene.SfmScene.from_colmap(self.dataset_path)
        self.scene_transform = frc.transforms.Compose(
            frc.transforms.NormalizeScene("pca"),
            frc.transforms.DownsampleImages(4),
        )
        self.sfm_scene = self.scene_transform(self.sfm_scene)
        self.sfm_scene = self.sfm_scene.select_images(np.arange(0, len(self.sfm_scene.images), 4))

    def test_run_training_with_no_saving(self):

        short_config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            max_epochs=1,
            refine_start_epoch=5,
            eval_at_percent=[],
        )

        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            self.sfm_scene,
            config=short_config,
            use_every_n_as_val=2,
        )

        runner.optimize()

        self.assertEqual(runner.model.num_gaussians, self.sfm_scene.points.shape[0])

    def test_run_training_with_mcmc_optimizer_no_refine(self):
        if not torch.cuda.is_available():
            self.skipTest("GaussianSplatOptimizerMCMC uses CUDA-only ops")

        short_config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            max_epochs=1,
            refine_start_epoch=10_000,  # never refine
            refine_stop_epoch=10_000,
            eval_at_percent=[],
            save_at_percent=[],
            optimize_camera_poses=False,  # keep this test lightweight
        )

        mcmc_opt_config = frc.radiance_fields.GaussianSplatOptimizerMCMCConfig(
            noise_lr=0.0,  # disable stochastic noise in step()
            insertion_rate=1.0,  # no insertion in refine() (which we also skip)
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )

        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            self.sfm_scene,
            config=short_config,
            optimizer_config=mcmc_opt_config,
            use_every_n_as_val=2,
        )
        self.assertIsInstance(runner.optimizer, frc.radiance_fields.GaussianSplatOptimizerMCMC)

        n_before = runner.model.num_gaussians
        runner.optimize()
        n_after = runner.model.num_gaussians
        self.assertEqual(n_after, n_before)

    def test_run_training_with_mcmc_optimizer_with_refine_small_epoch(self):
        """
        Integration-style test: run a very short epoch (few images) and ensure the training loop
        actually calls optimizer.refine() for the MCMC optimizer, causing insertion to occur.
        """
        if not torch.cuda.is_available():
            self.skipTest("GaussianSplatOptimizerMCMC uses CUDA-only ops")

        # Make the "epoch" short by using only a few images.
        num_images = min(4, len(self.sfm_scene.images))
        small_scene = self.sfm_scene.select_images(np.arange(num_images))

        # With batch_size=1 and 4 images:
        # num_steps_per_epoch == 4
        # refine_every_step == int(0.5 * 4) == 2, so refine triggers at global_step==2.
        short_config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            max_epochs=1,
            batch_size=1,
            refine_start_epoch=0,
            refine_stop_epoch=1,
            refine_every_epoch=0.5,
            eval_at_percent=[],
            save_at_percent=[],
            optimize_camera_poses=False,
        )

        mcmc_opt_config = frc.radiance_fields.GaussianSplatOptimizerMCMCConfig(
            noise_lr=0.0,
            insertion_rate=1.0001,  # small, fast insertion
            deletion_opacity_threshold=0.0,  # avoid relocation in this test; isolate insertion via refine()
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )

        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            small_scene,
            config=short_config,
            optimizer_config=mcmc_opt_config,
            use_every_n_as_val=-1,
        )
        self.assertIsInstance(runner.optimizer, frc.radiance_fields.GaussianSplatOptimizerMCMC)

        n_before = runner.model.num_gaussians
        runner.optimize(show_progress=False)
        n_after = runner.model.num_gaussians

        self.assertGreater(runner.optimizer.state_dict().get("refine_count", 0), 0)
        self.assertGreater(n_after, n_before)

    def test_run_training_with_saving(self):
        short_config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            max_epochs=2,
            refine_start_epoch=5,
            eval_at_percent=[50, 100],
            save_at_percent=[100],
        )

        writer = MockWriter()

        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            self.sfm_scene,
            config=short_config,
            use_every_n_as_val=2,
            writer=writer,
        )
        num_val = len(np.arange(0, len(self.sfm_scene.images), 2))
        num_train = len(self.sfm_scene.images) - num_val
        self.assertEqual(len(runner.training_dataset), num_train)
        self.assertEqual(len(runner.validation_dataset), num_val)

        self.assertEqual(len(writer.metric_log), 0)
        self.assertEqual(len(writer.checkpoint_log), 0)
        self.assertEqual(len(writer.ply_log), 0)
        self.assertEqual(len(writer.image_log), 0)

        runner.optimize()

        self.assertGreater(len(writer.metric_log), 0)
        self.assertEqual(len(writer.checkpoint_log), 1)  # One per save
        self.assertEqual(len(writer.ply_log), 1)  # One per save
        self.assertEqual(
            len(writer.image_log), 2 * len(runner.validation_dataset) * 2
        )  # Two images (predicted and ground truth) per validation view per eval

    def test_resuming_from_checkpoint(self):

        short_config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            max_epochs=2,
            refine_start_epoch=5,
            eval_at_percent=[50, 100],
            save_at_percent=[50, 100],
        )

        writer = MockWriter()

        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            self.sfm_scene,
            config=short_config,
            use_every_n_as_val=2,
            writer=writer,
        )
        num_val = len(np.arange(0, len(self.sfm_scene.images), 2))
        num_train = len(self.sfm_scene.images) - num_val
        self.assertEqual(len(runner.training_dataset), num_train)
        self.assertEqual(len(runner.validation_dataset), num_val)

        self.assertEqual(len(writer.metric_log), 0)
        self.assertEqual(len(writer.checkpoint_log), 0)
        self.assertEqual(len(writer.ply_log), 0)
        self.assertEqual(len(writer.image_log), 0)

        runner.optimize()

        num_metric_logs = len(writer.metric_log)
        print(writer.metric_log)
        self.assertGreater(num_metric_logs, 0)
        self.assertEqual(len(writer.checkpoint_log), 2)  # One per save
        self.assertEqual(len(writer.ply_log), 2)  # One per save
        self.assertEqual(
            len(writer.image_log), 2 * len(runner.validation_dataset) * 2
        )  # Two images (predicted and ground truth) per validation view per eval

        # Now let's grab one of the middle checkpoints and load the runner from that
        ckpt_step, ckpt_name, ckpt_dict = writer.checkpoint_log[0]

        # We'll create a runner from this checkpoint, but use the same writer so things get appended
        runner2 = frc.radiance_fields.GaussianSplatReconstruction.from_state_dict(
            ckpt_dict, device=runner.model.device, writer=writer
        )

        self.assertEqual(len(runner2.training_dataset), num_train)
        self.assertEqual(len(runner2.validation_dataset), num_val)

        # This should pick up from where we left off (50% through 2 epochs is epoch 1)
        # and save and evalute at 100% again
        runner2.optimize()

        print(writer.metric_log)
        self.assertEqual(len(writer.metric_log), num_metric_logs + num_metric_logs // 2)
        self.assertEqual(len(writer.checkpoint_log), 3)  # One more per save
        self.assertEqual(len(writer.ply_log), 3)  # One more per save
        self.assertEqual(
            len(writer.image_log), 3 * len(runner.validation_dataset) * 2
        )  # Two more images (predicted and ground truth) per validation view per eval
