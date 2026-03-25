# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import fvdb
import numpy as np
import torch

import fvdb_reality_capture as frc
from fvdb_reality_capture import radiance_fields
from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset


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

    def test_dataset_image_id_uses_global_scene_index(self):
        dataset = SfmDataset(self.sfm_scene, dataset_indices=np.array([1, 3, 5], dtype=np.int64))
        self.assertEqual(dataset[0]["image_id"], 1)
        self.assertEqual(dataset[1]["image_id"], 3)
        self.assertEqual(dataset[2]["image_id"], 5)

    def test_dataset_exposes_camera_model_and_distortion_coeffs(self):
        dataset = SfmDataset(self.sfm_scene, dataset_indices=np.array([0], dtype=np.int64))

        datum = dataset[0]
        image_meta = self.sfm_scene.images[0]

        self.assertEqual(int(datum["camera_model"]), int(image_meta.camera_metadata.camera_model))
        self.assertEqual(datum["camera_model"].dtype, torch.int32)
        self.assertEqual(tuple(datum["distortion_coeffs"].shape), (12,))

        expected_distortion_coeffs = (
            image_meta.camera_metadata.distortion_coeffs
            if image_meta.camera_metadata.distortion_coeffs.size != 0
            else np.zeros((12,), dtype=np.float32)
        )
        np.testing.assert_allclose(datum["distortion_coeffs"].numpy(), expected_distortion_coeffs)

    def test_pose_optimization_warns_with_holdout_and_uses_scene_global_pose_table(self):
        if not torch.cuda.is_available():
            self.skipTest("Camera pose optimization test requires CUDA")

        short_config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            max_epochs=1,
            refine_start_epoch=10_000,
            refine_stop_epoch=10_000,
            eval_at_percent=[],
            save_at_percent=[],
            optimize_camera_poses=True,
        )

        with self.assertLogs(
            "fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction.GaussianSplatReconstruction",
            level="WARNING",
        ) as logs:
            runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
                self.sfm_scene,
                config=short_config,
                use_every_n_as_val=2,
            )

        pose_adjust_model = runner.pose_adjust_model
        self.assertIsNotNone(pose_adjust_model)
        assert pose_adjust_model is not None
        self.assertEqual(pose_adjust_model.num_poses, self.sfm_scene.num_images)
        self.assertTrue(any("holdout set" in message.lower() for message in logs.output))

    def test_pose_optimization_scheduler_uses_training_step_horizon(self):
        config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            max_epochs=2,
            refine_start_epoch=10_000,
            refine_stop_epoch=10_000,
            eval_at_percent=[],
            save_at_percent=[],
            optimize_camera_poses=True,
            pose_opt_lr_decay=0.25,
        )

        with patch(
            "fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction.make_render_backend"
        ) as make_render_backend:
            make_render_backend.return_value.validate_scene_cameras.return_value = None
            runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
                self.sfm_scene,
                config=config,
                use_every_n_as_val=2,
                device="cpu",
            )

        pose_adjust_model = runner.pose_adjust_model
        pose_adjust_scheduler = runner.pose_adjust_scheduler
        self.assertIsNotNone(pose_adjust_model)
        self.assertIsNotNone(pose_adjust_scheduler)
        assert pose_adjust_model is not None
        assert pose_adjust_scheduler is not None

        self.assertEqual(pose_adjust_model.num_poses, self.sfm_scene.num_images)

        num_steps_per_epoch = int(np.ceil(len(runner.training_dataset) / config.batch_size))
        expected_total_pose_steps = max(
            1, int((config.pose_opt_stop_epoch - config.pose_opt_start_epoch) * num_steps_per_epoch)
        )
        expected_gamma = config.pose_opt_lr_decay ** (1.0 / expected_total_pose_steps)
        self.assertAlmostEqual(pose_adjust_scheduler.gamma, expected_gamma)

    def test_from_state_dict_restores_cpu_loaded_legacy_pose_checkpoint_on_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("Legacy pose checkpoint restore test requires CUDA")

        short_config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            max_epochs=1,
            refine_start_epoch=10_000,
            refine_stop_epoch=10_000,
            eval_at_percent=[],
            save_at_percent=[],
            optimize_camera_poses=True,
        )

        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            self.sfm_scene,
            config=short_config,
            use_every_n_as_val=2,
        )
        pose_adjust_model = runner.pose_adjust_model
        self.assertIsNotNone(pose_adjust_model)
        assert pose_adjust_model is not None

        checkpoint = runner.state_dict()
        train_indices = torch.as_tensor(runner.training_dataset.indices, dtype=torch.long, device=runner.device)
        legacy_pose_weights = (
            checkpoint["pose_adjust_model"]["pose_embeddings.weight"].index_select(0, train_indices).clone()
        )
        checkpoint["num_training_poses"] = len(runner.training_dataset)
        checkpoint["pose_adjust_model"]["pose_embeddings.weight"] = legacy_pose_weights

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pt", delete=True) as temp_file:
            torch.save(checkpoint, temp_file.name)
            cpu_loaded_checkpoint = torch.load(temp_file.name, map_location="cpu", weights_only=False)

        with self.assertLogs(
            "fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction.GaussianSplatReconstruction",
            level="WARNING",
        ) as logs:
            restored = frc.radiance_fields.GaussianSplatReconstruction.from_state_dict(
                cpu_loaded_checkpoint, device="cuda"
            )

        self.assertEqual(restored.device.type, "cuda")
        self.assertEqual(restored.model.device.type, "cuda")
        restored_pose_adjust_model = restored.pose_adjust_model
        self.assertIsNotNone(restored_pose_adjust_model)
        assert restored_pose_adjust_model is not None
        self.assertEqual(restored_pose_adjust_model.num_poses, self.sfm_scene.num_images)
        self.assertEqual(restored_pose_adjust_model.pose_embeddings.weight.device.type, "cuda")
        self.assertTrue(any("legacy checkpoint" in message.lower() for message in logs.output))

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
