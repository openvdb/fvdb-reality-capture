# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import tempfile
import unittest

import torch
from fvdb import GaussianSplat3d

import fvdb_reality_capture as frc
from tests.unit.common import GettysburgGaussianSplatTestCase


class GaussianSplatOptimizerMCMCTests(GettysburgGaussianSplatTestCase, unittest.TestCase):

    def test_serialize_optimizer_mcmc(self):
        if self.device != "cuda":
            self.skipTest("GaussianSplatOptimizerMCMC uses CUDA-only ops (add_noise_to_means / relocate_gaussians)")

        model_1 = self.model
        max_steps = 200 * len(self.training_dataset)
        config = frc.radiance_fields.GaussianSplatOptimizerMCMCConfig(
            noise_lr=0.0,  # disable stochasticity for this determinism/round-trip test
            insertion_rate=1.0,  # avoid insertions for determinism
            max_gaussians=-1,
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        optimizer_1 = frc.radiance_fields.GaussianSplatOptimizerMCMC.from_model_and_scene(
            model=model_1,
            sfm_scene=self.training_dataset.sfm_scene,
            config=config,
        )
        optimizer_1.reset_learning_rates_and_decay(batch_size=1, expected_steps=max_steps)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pt", delete=True) as temp_file:
            torch.save(model_1.state_dict(), temp_file.name + ".model")
            torch.save(optimizer_1.state_dict(), temp_file.name)

            # Run one step of refine + step
            optimizer_1.zero_grad()
            gt_img_1, pred_img_1, _ = self.render_one_image(model_1)
            loss_1 = torch.nn.functional.l1_loss(pred_img_1, gt_img_1)
            loss_1.backward()
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            optimizer_1.refine()
            optimizer_1.step()
            optimizer_1.zero_grad()

            # Post-step render
            gt_img_2, pred_img_2, _ = self.render_one_image(model_1)
            loss_2 = torch.nn.functional.l1_loss(pred_img_2, gt_img_2)

            # Load model + optimizer and replay the same operations
            model_2 = GaussianSplat3d.from_state_dict(torch.load(temp_file.name + ".model", map_location=self.device))
            loaded_state_dict = torch.load(temp_file.name, map_location=self.device, weights_only=False)
            optimizer_2 = frc.radiance_fields.GaussianSplatOptimizerMCMC.from_state_dict(model_2, loaded_state_dict)

            optimizer_2.zero_grad()
            gt_img_3, pred_img_3, _ = self.render_one_image(model_2)
            self.assertTrue(torch.allclose(pred_img_1, pred_img_3))
            loss_3 = torch.nn.functional.l1_loss(pred_img_3, gt_img_3)
            self.assertAlmostEqual(loss_1.item(), loss_3.item(), places=3)
            loss_3.backward()
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            optimizer_2.refine()
            optimizer_2.step()
            optimizer_2.zero_grad()

            gt_img_4, pred_img_4, _ = self.render_one_image(model_2)
            self.assertTrue(torch.allclose(pred_img_2, pred_img_4, atol=1e-3))
            loss_4 = torch.nn.functional.l1_loss(pred_img_4, gt_img_4)
            self.assertAlmostEqual(loss_2.item(), loss_4.item(), places=3)

    def test_refine_relocates_dead_gaussians(self):
        if self.device != "cuda":
            self.skipTest("GaussianSplatOptimizerMCMC uses CUDA-only ops (relocate_gaussians)")

        model = self.model
        config = frc.radiance_fields.GaussianSplatOptimizerMCMCConfig(
            noise_lr=0.0,
            insertion_rate=1.0,  # no add; isolate relocate behavior
            max_gaussians=-1,
            deletion_opacity_threshold=0.05,  # init opacity is ~0.1, so only explicitly-deadened gaussians are relocated
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        optimizer = frc.radiance_fields.GaussianSplatOptimizerMCMC.from_model_and_scene(
            model=model,
            sfm_scene=self.training_dataset.sfm_scene,
            config=config,
        )
        optimizer.reset_learning_rates_and_decay(batch_size=1, expected_steps=10)

        # Force some dead gaussians by lowering their opacities below the deletion threshold.
        num_dead = max(1, model.num_gaussians // 10)
        dead_indices = torch.randperm(model.num_gaussians, device=model.device)[:num_dead]
        means_before = model.means[dead_indices].clone()
        with torch.no_grad():
            model.logit_opacities[dead_indices] = torch.logit(torch.tensor([0.001], device=model.device)).item()

        n_before = model.num_gaussians
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        refine_stats = optimizer.refine()
        self.assertEqual(refine_stats["num_relocated"], num_dead)
        self.assertEqual(refine_stats["num_added"], 0)
        self.assertEqual(model.num_gaussians, n_before)

        means_after = model.means[dead_indices]
        self.assertFalse(torch.allclose(means_before, means_after))
        # Relocation does *not* guarantee opacity > deletion threshold (ratios can reduce opacity);
        # it *does* clamp to deletion_opacity_threshold.
        relocated_opacities = torch.sigmoid(model.logit_opacities[dead_indices])
        self.assertTrue(torch.all(relocated_opacities >= config.deletion_opacity_threshold - 1e-7))

    def test_refine_adds_gaussians(self):
        if self.device != "cuda":
            self.skipTest("GaussianSplatOptimizerMCMC uses CUDA-only ops (relocate_gaussians)")

        model = self.model
        config = frc.radiance_fields.GaussianSplatOptimizerMCMCConfig(
            noise_lr=0.0,
            insertion_rate=1.1,
            max_gaussians=-1,
            deletion_opacity_threshold=0.0,  # ensure none are dead
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        optimizer = frc.radiance_fields.GaussianSplatOptimizerMCMC.from_model_and_scene(
            model=model,
            sfm_scene=self.training_dataset.sfm_scene,
            config=config,
        )
        optimizer.reset_learning_rates_and_decay(batch_size=1, expected_steps=10)

        n_before = model.num_gaussians
        expected_target = int(config.insertion_rate * n_before)
        expected_added = max(0, expected_target - n_before)

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        stats = optimizer.refine()

        self.assertEqual(stats["num_relocated"], 0)
        self.assertEqual(stats["num_added"], expected_added)
        self.assertEqual(model.num_gaussians, n_before + expected_added)

        # Optimizer param groups should now match model tensor sizes.
        for pg in optimizer._optimizer.param_groups:
            p = pg["params"][0]
            self.assertEqual(p.shape[0], model.num_gaussians, f"param_group {pg['name']} not resized correctly")


if __name__ == "__main__":
    unittest.main()
