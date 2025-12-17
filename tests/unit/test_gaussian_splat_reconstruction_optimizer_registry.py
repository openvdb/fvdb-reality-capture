# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import unittest
from typing import Any

import torch

import fvdb_reality_capture as frc
from tests.unit.common import load_gettysburg_scene_and_dataset


class MockWriter(frc.radiance_fields.GaussianSplatReconstructionBaseWriter):
    """Writer that records nothing; used to avoid filesystem side effects in tests."""

    def __init__(self):
        super().__init__()

    def log_metric(self, global_step: int, metric_name: str, metric_value: float) -> None:  # noqa: ARG002
        return

    def save_checkpoint(
        self, global_step: int, checkpoint_name: str, checkpoint: dict[str, Any]
    ) -> None:  # noqa: ARG002
        return

    def save_ply(self, global_step: int, ply_name: str, model, metadata: dict[str, Any]) -> None:  # noqa: ARG002
        return

    def save_image(
        self, global_step: int, image_name: str, image: torch.Tensor, jpeg_quality: int = 98
    ) -> None:  # noqa: ARG002
        return


class GaussianSplatReconstructionOptimizerRegistryTests(unittest.TestCase):
    def setUp(self):
        _, dataset = load_gettysburg_scene_and_dataset()
        # Use the transformed scene from the dataset so the tests match other unit tests.
        self.sfm_scene = dataset.sfm_scene
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Keep these tests lightweight: no eval, no saving, no refinement, no pose optimization.
        self.recon_config = frc.radiance_fields.GaussianSplatReconstructionConfig(
            max_epochs=1,
            refine_start_epoch=10_000,
            refine_stop_epoch=10_000,
            eval_at_percent=[],
            save_at_percent=[],
            optimize_camera_poses=False,
        )

    def test_from_sfm_scene_uses_classic_optimizer_for_classic_config(self):
        optimizer_config = frc.radiance_fields.GaussianSplatOptimizerConfig(
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
            use_screen_space_scales_for_refinement_until=0,
            use_scales_for_deletion_after_n_refinements=-1,
        )
        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            self.sfm_scene,
            config=self.recon_config,
            optimizer_config=optimizer_config,
            writer=MockWriter(),
            device=self.device,
        )
        self.assertIsInstance(runner.optimizer, frc.radiance_fields.GaussianSplatOptimizer)

    def test_from_sfm_scene_uses_mcmc_optimizer_for_mcmc_config(self):
        if self.device != "cuda":
            self.skipTest("GaussianSplatOptimizerMCMC uses CUDA-only ops")
        optimizer_config = frc.radiance_fields.GaussianSplatOptimizerMCMCConfig(
            noise_lr=0.0,  # deterministic for tests
            insertion_rate=1.0,
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            self.sfm_scene,
            config=self.recon_config,
            optimizer_config=optimizer_config,
            writer=MockWriter(),
            device=self.device,
        )
        self.assertIsInstance(runner.optimizer, frc.radiance_fields.GaussianSplatOptimizerMCMC)

    def test_checkpoint_roundtrip_preserves_classic_optimizer_type(self):
        optimizer_config = frc.radiance_fields.GaussianSplatOptimizerConfig(
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
            use_screen_space_scales_for_refinement_until=0,
            use_scales_for_deletion_after_n_refinements=-1,
        )
        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            self.sfm_scene,
            config=self.recon_config,
            optimizer_config=optimizer_config,
            writer=MockWriter(),
            device=self.device,
        )
        ckpt = runner.state_dict()
        runner2 = frc.radiance_fields.GaussianSplatReconstruction.from_state_dict(
            ckpt,
            writer=MockWriter(),
            device=runner.model.device,
        )
        self.assertIsInstance(runner2.optimizer, frc.radiance_fields.GaussianSplatOptimizer)
        self.assertEqual(runner2.optimizer.state_dict().get("name"), frc.radiance_fields.GaussianSplatOptimizer.name())

    def test_checkpoint_roundtrip_preserves_mcmc_optimizer_type(self):
        if self.device != "cuda":
            self.skipTest("GaussianSplatOptimizerMCMC uses CUDA-only ops")
        optimizer_config = frc.radiance_fields.GaussianSplatOptimizerMCMCConfig(
            noise_lr=0.0,
            insertion_rate=1.0,
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
            self.sfm_scene,
            config=self.recon_config,
            optimizer_config=optimizer_config,
            writer=MockWriter(),
            device=self.device,
        )
        ckpt = runner.state_dict()
        runner2 = frc.radiance_fields.GaussianSplatReconstruction.from_state_dict(
            ckpt,
            writer=MockWriter(),
            device=runner.model.device,
        )
        self.assertIsInstance(runner2.optimizer, frc.radiance_fields.GaussianSplatOptimizerMCMC)
        self.assertEqual(
            runner2.optimizer.state_dict().get("name"),
            frc.radiance_fields.GaussianSplatOptimizerMCMC.name(),
        )


if __name__ == "__main__":
    unittest.main()
