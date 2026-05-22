# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""CPU-only coverage for the dense-depth supervision plumbing.

The full training loop is exercised by the GPU tests in ``test_training.py``;
here we cover the two pure pieces of new logic without needing a model or GPU:

- ``_needs_depth_render`` -- the predicate that makes the render backend emit a
  depth channel for either sparse or dense depth supervision.
- ``_scale_shift_invariant_l1`` -- the RELATIVE-mode loss, which must be
  invariant to a per-image affine between prediction and target.
"""

import unittest

import torch

from fvdb_reality_capture.radiance_fields._gaussian_rendering import _needs_depth_render
from fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction import (
    GaussianSplatReconstructionConfig,
    _scale_shift_invariant_l1,
)


class TestNeedsDepthRender(unittest.TestCase):
    def test_disabled_when_no_depth_supervision(self):
        cfg = GaussianSplatReconstructionConfig(sparse_depth_reg=0.0, dense_depth_reg=0.0)
        self.assertFalse(_needs_depth_render(cfg))

    def test_enabled_by_sparse_depth(self):
        cfg = GaussianSplatReconstructionConfig(sparse_depth_reg=0.1, dense_depth_reg=0.0)
        self.assertTrue(_needs_depth_render(cfg))

    def test_enabled_by_dense_depth(self):
        cfg = GaussianSplatReconstructionConfig(sparse_depth_reg=0.0, dense_depth_reg=0.5)
        self.assertTrue(_needs_depth_render(cfg))

    def test_enabled_by_both(self):
        cfg = GaussianSplatReconstructionConfig(sparse_depth_reg=0.2, dense_depth_reg=0.5)
        self.assertTrue(_needs_depth_render(cfg))


class TestScaleShiftInvariantL1(unittest.TestCase):
    def test_identical_is_zero(self):
        target = torch.rand(1, 8, 8) + 0.5
        valid = torch.ones(1, 8, 8, dtype=torch.bool)
        self.assertAlmostEqual(_scale_shift_invariant_l1(target.clone(), target, valid).item(), 0.0, places=6)

    def test_invariant_to_affine(self):
        # An arbitrary positive scale + shift of the target must yield ~zero loss.
        target = torch.rand(1, 16, 16) + 0.5
        valid = torch.ones(1, 16, 16, dtype=torch.bool)
        pred = 3.7 * target + 2.1
        self.assertLess(_scale_shift_invariant_l1(pred, target, valid).item(), 1e-5)

    def test_detects_non_affine_difference(self):
        # A genuinely different (non-affine) prediction must produce a positive loss.
        target = torch.rand(1, 16, 16) + 0.5
        valid = torch.ones(1, 16, 16, dtype=torch.bool)
        pred = target**2
        self.assertGreater(_scale_shift_invariant_l1(pred, target, valid).item(), 1e-3)

    def test_respects_valid_mask(self):
        # Garbage in the invalid region must not affect the loss.
        target = torch.rand(1, 8, 8) + 0.5
        valid = torch.ones(1, 8, 8, dtype=torch.bool)
        valid[0, :, 4:] = False
        pred = 2.0 * target + 1.0  # affine-related on valid pixels
        clean = _scale_shift_invariant_l1(pred, target, valid).item()

        corrupted = target.clone()
        corrupted[0, :, 4:] = 1e6  # only invalid pixels corrupted
        pred_corrupt = 2.0 * corrupted + 1.0
        # Recompute pred from the (corrupted) target on invalid pixels only;
        # valid pixels are identical, so the masked loss must match.
        pred_corrupt[0, :, :4] = pred[0, :, :4]
        masked = _scale_shift_invariant_l1(pred_corrupt, target, valid).item()
        self.assertAlmostEqual(clean, masked, places=5)

    def test_empty_mask_returns_zero(self):
        target = torch.rand(1, 8, 8) + 0.5
        pred = torch.rand(1, 8, 8) + 0.5
        valid = torch.zeros(1, 8, 8, dtype=torch.bool)
        self.assertEqual(_scale_shift_invariant_l1(pred, target, valid).item(), 0.0)

    def test_is_differentiable(self):
        target = torch.rand(1, 8, 8) + 0.5
        valid = torch.ones(1, 8, 8, dtype=torch.bool)
        pred = (torch.rand(1, 8, 8) + 0.5).requires_grad_(True)
        loss = _scale_shift_invariant_l1(pred, target, valid)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_batched_matches_per_image_mean(self):
        # Batch>1 with differing valid-pixel counts per element: result must equal the
        # mean of the per-image losses computed independently.
        torch.manual_seed(0)
        pred = torch.rand(3, 8, 8) + 0.5
        target = torch.rand(3, 8, 8) + 0.5
        valid = torch.ones(3, 8, 8, dtype=torch.bool)
        valid[1, :, 5:] = False  # element 1 has fewer valid pixels
        valid[2] = False  # element 2 fully invalid -> contributes 0

        batched = _scale_shift_invariant_l1(pred, target, valid).item()
        per_image = [
            _scale_shift_invariant_l1(pred[i : i + 1], target[i : i + 1], valid[i : i + 1]).item() for i in range(3)
        ]
        self.assertAlmostEqual(batched, sum(per_image) / 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
