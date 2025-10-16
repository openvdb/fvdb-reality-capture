# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
import tempfile
import unittest
from typing import Any

import fvdb
import numpy as np
import torch

import fvdb_reality_capture as frc
from fvdb_reality_capture import training


class MockWriter(training.GaussianSplatReconstructionBaseWriter):
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

        short_config = frc.training.GaussianSplatReconstructionConfig(
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
        short_config = frc.training.GaussianSplatReconstructionConfig(
            max_epochs=2,
            refine_start_epoch=5,
            eval_at_percent=[50, 100],
            save_at_percent=[100],
        )

        writer = MockWriter()

        runner = frc.GaussianSplatReconstruction.from_sfm_scene(
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

        runner.train()

        self.assertGreater(len(writer.metric_log), 0)
        self.assertEqual(len(writer.checkpoint_log), 1)  # One per save
        self.assertEqual(len(writer.ply_log), 1)  # One per save
        self.assertEqual(
            len(writer.image_log), 2 * len(runner.validation_dataset) * 2
        )  # Two images (predicted and ground truth) per validation view per eval

    def test_resuming_from_checkpoint(self):

        short_config = frc.training.GaussianSplatReconstructionConfig(
            max_epochs=2,
            refine_start_epoch=5,
            eval_at_percent=[50, 100],
            save_at_percent=[50, 100],
        )

        writer = MockWriter()

        runner = frc.GaussianSplatReconstruction.from_sfm_scene(
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

        runner.train()

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
        runner2 = frc.GaussianSplatReconstruction.from_state_dict(ckpt_dict, device=runner.model.device, writer=writer)

        self.assertEqual(len(runner2.training_dataset), num_train)
        self.assertEqual(len(runner2.validation_dataset), num_val)

        # This should pick up from where we left off (50% through 2 epochs is epoch 1)
        # and save and evalute at 100% again
        runner2.train()

        print(writer.metric_log)
        self.assertEqual(len(writer.metric_log), num_metric_logs + num_metric_logs // 2)
        self.assertEqual(len(writer.checkpoint_log), 3)  # One more per save
        self.assertEqual(len(writer.ply_log), 3)  # One more per save
        self.assertEqual(
            len(writer.image_log), 3 * len(runner.validation_dataset) * 2
        )  # Two more images (predicted and ground truth) per validation view per eval
