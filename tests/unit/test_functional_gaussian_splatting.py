# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the composable Gaussian splatting pipeline in ``fvdb_reality_capture.functional``.

The pipeline is checked against :class:`GaussianSplat3d`, which composes the same stages, and
against itself across the dense, sparse, cropped and world-space paths.
"""

import unittest

import numpy as np
import torch
from fvdb import JaggedTensor
from fvdb.utils.tests import get_fvdb_test_data_path

import fvdb_reality_capture.functional as F
from fvdb_reality_capture import CameraModel, GaussianRenderMode, GaussianSplat3d, ProjectionMethod


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class FunctionalPipelineTestCase(unittest.TestCase):
    """Load the garden scene once per test and expose its tensors."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("Gaussian splatting requires a CUDA device")
        torch.manual_seed(0)
        np.random.seed(0)
        self.device = torch.device("cuda:0")
        data = np.load(get_fvdb_test_data_path() / "gsplat" / "test_garden_cropped.npz")

        self.means = torch.from_numpy(data["means3d"]).float().to(self.device)
        self.quats = torch.from_numpy(data["quats"]).float().to(self.device)
        self.log_scales = torch.log(torch.from_numpy(data["scales"]).float().to(self.device))
        self.logit_opacities = torch.logit(torch.from_numpy(data["opacities"]).float().to(self.device))
        colors = torch.from_numpy(data["colors"]).float().to(self.device)
        self.W = int(data["width"].item())
        self.H = int(data["height"].item())
        all_w2c = torch.from_numpy(data["viewmats"]).float().to(self.device)
        all_K = torch.from_numpy(data["Ks"]).float().to(self.device)
        self.w2c = all_w2c[:2].contiguous()
        self.K = all_K[:2].contiguous()
        self.C = self.w2c.shape[0]

        N = self.means.shape[0]
        self.sh_degree = 2
        sh = torch.zeros(N, (self.sh_degree + 1) ** 2, 3, device=self.device)
        sh[:, 0] = rgb_to_sh(colors)
        sh[:, 1:] = torch.randn_like(sh[:, 1:]) * 0.05
        self.sh0 = sh[:, :1].contiguous()
        self.shN = sh[:, 1:].contiguous()

    def _params(self, requires_grad: bool = False):
        tensors = [self.means, self.quats, self.log_scales, self.logit_opacities, self.sh0, self.shN]
        return [t.detach().clone().requires_grad_(requires_grad) for t in tensors]

    def _model(self, params) -> GaussianSplat3d:
        means, quats, log_scales, logit_opacities, sh0, shN = params
        return GaussianSplat3d.from_tensors(
            means=means, quats=quats, log_scales=log_scales, logit_opacities=logit_opacities, sh0=sh0, shN=shN
        )

    def _render_functional(self, params, render_mode=GaussianRenderMode.FEATURES, **kwargs):
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H, **kwargs)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, render_mode=render_mode)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        tiles = F.intersect_gaussian_tiles(projected, opacities)
        return F.rasterize_screen_space_gaussians(projected, features, opacities, tiles)


class TestStageOutputs(FunctionalPipelineTestCase):
    def test_projection_contract(self):
        params = self._params()
        projected = F.project_gaussians(*params[:3], self.w2c, self.K, self.W, self.H, antialias=True)
        C, N = self.C, self.means.shape[0]
        self.assertEqual(tuple(projected.radii.shape), (C, N, 2))
        self.assertEqual(tuple(projected.means2d.shape), (C, N, 2))
        self.assertEqual(tuple(projected.depths.shape), (C, N))
        self.assertEqual(tuple(projected.conics.shape), (C, N, 3))
        self.assertEqual(tuple(projected.compensations.shape), (C, N))
        self.assertEqual((projected.num_cameras, projected.num_gaussians), (C, N))
        self.assertEqual(projected.projection_method, ProjectionMethod.ANALYTIC)
        self.assertTrue(projected.is_differentiable)
        with self.assertRaises(Exception):
            projected.radii = None  # frozen
        # Equality is identity, so the dataclasses can be compared and hashed despite holding tensors.
        self.assertEqual(projected, projected)
        self.assertNotEqual(projected, without := F.project_gaussians(*params[:3], self.w2c, self.K, self.W, self.H))
        self.assertEqual(len({projected, without}), 2)

        without = F.project_gaussians(*params[:3], self.w2c, self.K, self.W, self.H, antialias=False)
        self.assertIsNone(without.compensations)

    def test_render_modes(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        depth = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, render_mode=GaussianRenderMode.DEPTH)
        both = F.evaluate_gaussian_sh(
            means, sh0, shN, self.w2c, projected, render_mode=GaussianRenderMode.FEATURES_AND_DEPTH
        )
        self.assertEqual(features.shape[-1], 3)
        self.assertEqual(depth.shape[-1], 1)
        self.assertEqual(both.shape[-1], 4)
        torch.testing.assert_close(both[..., :3], features)
        torch.testing.assert_close(both[..., 3:], depth)
        # Both the projection and the features stage zero the depth of culled Gaussians and agree on the rest.
        visible = (projected.radii > 0).all(-1)
        torch.testing.assert_close(depth[..., 0][visible], projected.depths[visible])
        self.assertTrue(bool((projected.depths[~visible] == 0).all()))
        # Lower degrees are allowed, higher than available are not.
        F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, sh_degree_to_use=0)
        with self.assertRaises(ValueError):
            F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, sh_degree_to_use=self.sh_degree + 1)

    def test_tile_intersection_contract(self):
        params = self._params()
        projected = F.project_gaussians(*params[:3], self.w2c, self.K, self.W, self.H)
        tiles = F.intersect_gaussian_tiles(projected, F.compute_gaussian_opacities(params[3], projected), tile_size=16)
        self.assertEqual(tuple(tiles.tile_offsets.shape), (self.C, tiles.num_tiles_h, tiles.num_tiles_w))
        self.assertEqual(tiles.num_tiles_h, -(-self.H // 16))
        self.assertEqual(tiles.num_tiles_w, -(-self.W // 16))
        # Culling with opacities can only remove intersections relative to the bounding-box test.
        loose = F.intersect_gaussian_tiles(projected, None, tile_size=16)
        self.assertLessEqual(tiles.tile_gaussian_ids.numel(), loose.tile_gaussian_ids.numel())


class TestMatchesGaussianSplat3d(FunctionalPipelineTestCase):
    def test_forward_matches_oo(self):
        params = self._params()
        images, alphas = self._render_functional(params)
        images_oo, alphas_oo = self._model(params).render_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        torch.testing.assert_close(images, images_oo, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alphas, alphas_oo, atol=1e-5, rtol=1e-5)

    def test_backward_matches_oo(self):
        params_fn = self._params(requires_grad=True)
        params_oo = self._params(requires_grad=True)
        images, alphas = self._render_functional(params_fn)
        (images.square().mean() + alphas.mean()).backward()
        images_oo, alphas_oo = self._model(params_oo).render_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        (images_oo.square().mean() + alphas_oo.mean()).backward()
        for p_fn, p_oo in zip(params_fn, params_oo):
            self.assertIsNotNone(p_fn.grad)
            self.assertGreater(float(p_fn.grad.abs().max()), 0.0)
            # Both paths run the same atomicAdd kernels, so agreement is up to summation order.
            torch.testing.assert_close(p_fn.grad, p_oo.grad, atol=1e-5, rtol=1e-4)

    def test_depth_and_features_and_depth_match_oo(self):
        params = self._params()
        model = self._model(params)
        depth, alpha = self._render_functional(params, render_mode=GaussianRenderMode.DEPTH)
        depth_oo, alpha_oo = model.render_depths(self.w2c, self.K, self.W, self.H, 0.01, 1e10, min_radius_2d=0.0)
        torch.testing.assert_close(depth, depth_oo, atol=1e-5, rtol=1e-5)
        both, _ = self._render_functional(params, render_mode=GaussianRenderMode.FEATURES_AND_DEPTH)
        both_oo, _ = model.render_images_and_depths(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        torch.testing.assert_close(both, both_oo, atol=1e-5, rtol=1e-5)

    def test_world_space_matches_oo_and_is_differentiable_through_ut(self):
        params = self._params(requires_grad=True)
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(
            means, quats, log_scales, self.w2c, self.K, self.W, self.H, projection_method=ProjectionMethod.UNSCENTED
        )
        self.assertFalse(projected.is_differentiable)
        self.assertFalse(projected.means2d.requires_grad)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        tiles = F.intersect_gaussian_tiles(projected, opacities)
        images, alphas = F.rasterize_world_space_gaussians(
            means, quats, log_scales, projected, features, opacities, self.w2c, self.K, tiles
        )
        images.mean().backward()
        for p in (means, quats, log_scales, logit_opacities, sh0):
            self.assertGreater(float(p.grad.abs().max()), 0.0)

        # OpenCV models need explicit distortion coefficients; zeros would silently render pinhole rays.
        opencv = F.project_gaussians(
            means.detach(),
            quats.detach(),
            log_scales.detach(),
            self.w2c,
            self.K,
            self.W,
            self.H,
            camera_model=CameraModel.OPENCV_RADTAN_5,
            distortion_coeffs=torch.zeros(self.C, 12, device=self.device),
        )
        with self.assertRaises(RuntimeError):
            F.rasterize_world_space_gaussians(
                means, quats, log_scales, opencv, features.detach(), opacities.detach(), self.w2c, self.K, tiles
            )
        # The coefficients are checked before the kernel reads twelve per camera.
        world_args = (means, quats, log_scales, opencv, features.detach(), opacities.detach(), self.w2c, self.K, tiles)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            F.rasterize_world_space_gaussians(*world_args, distortion_coeffs=torch.zeros(self.C, 5, device=self.device))
        with self.assertRaisesRegex(RuntimeError, "must be on"):
            F.rasterize_world_space_gaussians(*world_args, distortion_coeffs=torch.zeros(self.C, 12))
        # Pinhole cameras ignore the coefficients, whatever is passed.
        F.rasterize_world_space_gaussians(
            means,
            quats,
            log_scales,
            projected,
            features.detach(),
            opacities.detach(),
            self.w2c,
            self.K,
            tiles,
            distortion_coeffs=torch.zeros(self.C, 5),
        )

        params_oo = self._params()
        images_oo, alphas_oo = self._model(params_oo).render_images_from_world(
            self.w2c, self.K, self.W, self.H, 0.01, 1e10, projection_method=ProjectionMethod.UNSCENTED
        )
        torch.testing.assert_close(images.detach(), images_oo, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alphas.detach(), alphas_oo, atol=1e-5, rtol=1e-5)

    def test_depth_is_differentiable_through_the_unscented_projection(self):
        means, quats, log_scales, logit_opacities, sh0, shN = self._params(requires_grad=True)
        analytic = F.project_gaussians(
            means.detach(), quats.detach(), log_scales.detach(), self.w2c, self.K, self.W, self.H
        )
        projected = F.project_gaussians(
            means, quats, log_scales, self.w2c, self.K, self.W, self.H, projection_method=ProjectionMethod.UNSCENTED
        )
        self.assertFalse(projected.depths.requires_grad)
        # Depth is linear in the center, so the recomputed depth matches the projection's exactly and
        # carries the gradient the unscented kernel cannot provide.
        depth = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, render_mode=GaussianRenderMode.DEPTH)
        self.assertTrue(depth.requires_grad)
        # Each projection zeroes the depth of what it culls, and the two cull slightly differently.
        visible = (analytic.radii.amin(-1) > 0) & (projected.radii.amin(-1) > 0)
        torch.testing.assert_close(depth[..., 0][visible], analytic.depths[visible], atol=1e-4, rtol=1e-4)
        depth.sum().backward()
        self.assertGreater(float(means.grad.abs().max()), 0.0)
        both = F.evaluate_gaussian_sh(
            means, sh0, shN, self.w2c, projected, render_mode=GaussianRenderMode.FEATURES_AND_DEPTH
        )
        self.assertTrue(both[..., 3:].requires_grad)
        # The depth channel is the same function of its inputs with or without autograd.
        with torch.no_grad():
            same = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, render_mode=GaussianRenderMode.DEPTH)
        self.assertTrue(torch.equal(same, both[..., 3:].detach()))
        visible = (projected.radii > 0).all(-1)
        torch.testing.assert_close(same[..., 0][visible], projected.depths[visible])
        self.assertFalse(F.requires_distortion_coeffs(CameraModel.PINHOLE))
        self.assertTrue(F.requires_distortion_coeffs(CameraModel.OPENCV_RADTAN_5))

    def test_training_loop_reduces_loss(self):
        target, _ = self._render_functional(self._params())
        params = self._params(requires_grad=True)
        means, quats, log_scales, logit_opacities, sh0, shN = params
        with torch.no_grad():
            sh0.add_(0.3 * torch.randn_like(sh0))
        optimizer = torch.optim.Adam([sh0, shN, logit_opacities], lr=1e-2)
        losses = []
        for _ in range(15):
            optimizer.zero_grad()
            images, _ = self._render_functional(params)
            loss = torch.nn.functional.l1_loss(images, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], 0.5 * losses[0])


class TestSparseAndCrop(FunctionalPipelineTestCase):
    def _pixels(self, with_duplicates: bool) -> JaggedTensor:
        rows = torch.randint(0, self.H, (200,), device=self.device)
        cols = torch.randint(0, self.W, (200,), device=self.device)
        px = torch.stack([rows, cols], dim=1)
        px = torch.unique(px, dim=0)
        if with_duplicates:
            px = torch.cat([px, px[:37], px[5:6].repeat(3, 1)])
        return JaggedTensor([px, px.flip(0)])

    def test_sparse_matches_dense_at_requested_pixels(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        dense, dense_alpha = F.rasterize_screen_space_gaussians(
            projected, features, opacities, F.intersect_gaussian_tiles(projected, opacities)
        )
        for with_duplicates in (False, True):
            pixels = self._pixels(with_duplicates)
            sparse_tiles = F.intersect_gaussian_tiles_sparse(pixels, projected, opacities)
            self.assertEqual(sparse_tiles.has_duplicates, with_duplicates)
            rendered, alphas = F.rasterize_screen_space_gaussians_sparse(projected, features, opacities, sparse_tiles)
            self.assertEqual(len(rendered), self.C)
            all_tiles = torch.ones_like(sparse_tiles.active_tile_mask)
            same, _ = F.rasterize_screen_space_gaussians_sparse(
                projected, features, opacities, sparse_tiles, tile_masks=all_tiles
            )
            torch.testing.assert_close(same.jdata, rendered.jdata)
            # A per-pixel mask handed in as a tile mask is refused instead of skipping the wrong tiles.
            with self.assertRaisesRegex(ValueError, "per-tile"):
                F.rasterize_screen_space_gaussians_sparse(
                    projected, features, opacities, sparse_tiles, tile_masks=torch.ones(self.C, self.H, self.W)
                )
            for c in range(self.C):
                px = pixels[c].jdata
                self.assertEqual(tuple(rendered[c].jdata.shape), (px.shape[0], 3))
                torch.testing.assert_close(rendered[c].jdata, dense[c, px[:, 0], px[:, 1]], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(alphas[c].jdata, dense_alpha[c, px[:, 0], px[:, 1]], atol=1e-5, rtol=1e-5)

    def test_sparse_matches_oo_and_backward_runs(self):
        params = self._params(requires_grad=True)
        means, quats, log_scales, logit_opacities, sh0, shN = params
        pixels = self._pixels(with_duplicates=True)
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        sparse_tiles = F.intersect_gaussian_tiles_sparse(pixels, projected, opacities)
        rendered, alphas = F.rasterize_screen_space_gaussians_sparse(projected, features, opacities, sparse_tiles)
        rendered_oo, alphas_oo = self._model(self._params()).sparse_render_images(
            pixels, self.w2c, self.K, self.W, self.H, 0.01, 1e10
        )
        torch.testing.assert_close(rendered.jdata.detach(), rendered_oo.jdata, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alphas.jdata.detach(), alphas_oo.jdata, atol=1e-5, rtol=1e-5)
        rendered.jdata.mean().backward()
        self.assertGreater(float(means.grad.abs().max()), 0.0)
        self.assertGreater(float(sh0.grad.abs().max()), 0.0)

    def test_crop_is_a_slice_of_the_full_render(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        tiles = F.intersect_gaussian_tiles(projected, opacities)
        full, full_alpha = F.rasterize_screen_space_gaussians(projected, features, opacities, tiles)
        ox, oy, w, h = 37, 21, 90, 60
        crop, crop_alpha = F.rasterize_screen_space_gaussians(
            projected, features, opacities, tiles, crop=(ox, oy, w, h)
        )
        self.assertEqual(tuple(crop.shape), (self.C, h, w, 3))
        torch.testing.assert_close(crop, full[:, oy : oy + h, ox : ox + w], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(crop_alpha, full_alpha[:, oy : oy + h, ox : ox + w], atol=1e-5, rtol=1e-5)
        # Clamped to the image, invalid crops rejected.
        clamped, _ = F.rasterize_screen_space_gaussians(
            projected, features, opacities, tiles, crop=(self.W - 10, self.H - 5, 100, 100)
        )
        self.assertEqual(tuple(clamped.shape[1:3]), (5, 10))
        for bad in ((-1, 0, 10, 10), (0, 0, 0, 10)):
            with self.assertRaises(ValueError):
                F.rasterize_screen_space_gaussians(projected, features, opacities, tiles, crop=bad)
        # A crop entirely outside the image clips to nothing rather than raising.
        empty, _ = F.rasterize_screen_space_gaussians(projected, features, opacities, tiles, crop=(self.W, 0, 10, 10))
        self.assertEqual(tuple(empty.shape), (self.C, 0, 0, 3))

        # The OO crop path agrees, including a mask given in crop coordinates.
        model = self._model(params)
        pg = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        crop_oo, _ = model.render_from_projected_gaussians(
            pg, crop_width=w, crop_height=h, crop_origin_w=ox, crop_origin_h=oy
        )
        torch.testing.assert_close(crop_oo, crop, atol=1e-5, rtol=1e-5)
        mask = torch.zeros(self.C, h, w, dtype=torch.bool, device=self.device)
        mask[:, : h // 2] = True
        masked, masked_alpha = model.render_from_projected_gaussians(
            pg, crop_width=w, crop_height=h, crop_origin_w=ox, crop_origin_h=oy, masks=mask
        )
        torch.testing.assert_close(masked[:, : h // 2], crop[:, : h // 2], atol=1e-5, rtol=1e-5)
        self.assertEqual(float(masked[:, h // 2 :].abs().max()), 0.0)
        self.assertEqual(float(masked_alpha[:, h // 2 :].abs().max()), 0.0)
        # A crop past the image edge keeps its requested size: the part inside the image is the render,
        # the rest is background with zero alpha. A crop-space mask of the requested size still applies.
        edge_w, edge_h = 50, 40
        edge_mask = torch.ones(self.C, edge_h, edge_w, dtype=torch.bool, device=self.device)
        edge, edge_alpha = model.render_from_projected_gaussians(
            pg,
            crop_width=edge_w,
            crop_height=edge_h,
            crop_origin_w=self.W - 30,
            crop_origin_h=self.H - 25,
            masks=edge_mask,
        )
        self.assertEqual(tuple(edge.shape[1:3]), (edge_h, edge_w))
        self.assertEqual(tuple(edge_alpha.shape[1:]), (edge_h, edge_w, 1))
        torch.testing.assert_close(edge[:, :25, :30], full[:, self.H - 25 :, self.W - 30 :], atol=1e-5, rtol=1e-5)
        self.assertEqual(float(edge[:, 25:].abs().max()), 0.0)
        self.assertEqual(float(edge[:, :, 30:].abs().max()), 0.0)
        self.assertEqual(float(edge_alpha[:, 25:].abs().max()), 0.0)
        self.assertEqual(float(edge_alpha[:, :, 30:].abs().max()), 0.0)
        # Non-boolean masks are accepted with and without a crop.
        float_mask = torch.ones(self.C, self.H, self.W, device=self.device)
        with_float, _ = F.rasterize_screen_space_gaussians(
            projected, features, opacities, tiles, masks=float_mask, crop=(ox, oy, w, h)
        )
        torch.testing.assert_close(with_float, crop, atol=1e-5, rtol=1e-5)
        # With a crop the stage also takes a crop-space mask through crop_masks, and pools the skipped
        # tiles from it; masks stays in image coordinates, so neither shape is ever guessed.
        stage_masked, _ = F.rasterize_screen_space_gaussians(
            projected, features, opacities, tiles, crop=(ox, oy, w, h), crop_masks=mask
        )
        torch.testing.assert_close(stage_masked, masked, atol=1e-5, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, "not both"):
            F.rasterize_screen_space_gaussians(
                projected, features, opacities, tiles, masks=float_mask, crop=(ox, oy, w, h), crop_masks=mask
            )
        with self.assertRaisesRegex(ValueError, "needs a crop"):
            F.rasterize_screen_space_gaussians(projected, features, opacities, tiles, crop_masks=mask)
        # A crop mask of any other size is rejected rather than read from its top-left corner; the class
        # method hands its crop-space mask to the same check.
        with self.assertRaisesRegex(ValueError, "crop_masks must match the crop .* clipped size"):
            F.rasterize_screen_space_gaussians(
                projected, features, opacities, tiles, crop=(ox, oy, w, h), crop_masks=mask[:, :-1]
            )
        with self.assertRaisesRegex(ValueError, "clipped size"):
            model.render_from_projected_gaussians(
                pg, crop_width=w, crop_height=h, crop_origin_w=ox, crop_origin_h=oy, masks=mask[:, :-1]
            )
        # A mask on the wrong device is rejected at the same point, not deep inside torch.
        cpu_full_mask = torch.ones(self.C, self.H, self.W, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "must be on"):
            F.rasterize_screen_space_gaussians(projected, features, opacities, tiles, masks=cpu_full_mask)
        with self.assertRaisesRegex(ValueError, "must be on"):
            model.render_from_projected_gaussians(
                pg, crop_width=w, crop_height=h, crop_origin_w=ox, crop_origin_h=oy, masks=mask.cpu()
            )

    def test_analysis_matches_oo(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        model = self._model(params)
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        tiles = F.intersect_gaussian_tiles(projected, opacities)
        counts, alphas = F.rasterize_num_contributing_gaussians(projected, opacities, tiles)
        counts_oo, alphas_oo = model.render_num_contributing_gaussians(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        self.assertTrue(torch.equal(counts, counts_oo))
        torch.testing.assert_close(alphas, alphas_oo)

        ids, weights = F.rasterize_contributing_gaussian_ids(projected, opacities, tiles)
        ids_oo, weights_oo = model.render_contributing_gaussian_ids(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        self.assertTrue(torch.equal(ids.jdata, ids_oo.jdata))
        self.assertEqual(ids.ldim, 2)
        self.assertEqual(int(ids.jdata.numel()), int(counts.sum()))
        top_ids, _ = F.rasterize_contributing_gaussian_ids(projected, opacities, tiles, top_k_contributors=3)
        self.assertLessEqual(int(top_ids.jdata.numel()), int(counts.clamp(max=3).sum()))

    def test_sparse_analysis_with_duplicates(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        tiles = F.intersect_gaussian_tiles(projected, opacities)
        counts_dense, _ = F.rasterize_num_contributing_gaussians(projected, opacities, tiles)
        ids_dense, _ = F.rasterize_contributing_gaussian_ids(projected, opacities, tiles)

        pixels = self._pixels(with_duplicates=True)
        sparse_tiles = F.intersect_gaussian_tiles_sparse(pixels, projected, opacities)
        counts, _ = F.rasterize_num_contributing_gaussians_sparse(projected, opacities, sparse_tiles)
        ids, weights = F.rasterize_contributing_gaussian_ids_sparse(projected, opacities, sparse_tiles)
        self.assertEqual(ids.ldim, 2)
        for c in range(self.C):
            px = pixels[c].jdata
            self.assertTrue(torch.equal(counts[c].jdata, counts_dense[c, px[:, 0], px[:, 1]]))

            def flat(x):
                return x.jdata if isinstance(x, JaggedTensor) else x

            expected = [flat(ids_dense[c][int(r) * self.W + int(q)]) for r, q in px.tolist()]
            got = [flat(t) for t in ids[c].unbind()]
            self.assertEqual(len(got), len(expected))
            for g, e in zip(got, expected):
                self.assertTrue(torch.equal(g, e))

    def test_sparse_analysis_keeps_a_trailing_camera_with_no_requested_pixels(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        w2c, K = self.w2c[:2], self.K[:2]
        projected = F.project_gaussians(means, quats, log_scales, w2c, K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        busy = torch.tensor([[self.H // 2, self.W // 2], [self.H // 2 + 3, self.W // 2 + 3]], device=self.device)
        empty = torch.empty(0, 2, dtype=torch.int64, device=self.device)
        # Camera 0 has a duplicate (so the expansion path runs); camera 1 requests nothing.
        pixels = JaggedTensor([torch.cat([busy, busy[:1]]), empty])
        sparse_tiles = F.intersect_gaussian_tiles_sparse(pixels, projected, opacities)
        self.assertTrue(sparse_tiles.has_duplicates)
        for result in F.rasterize_contributing_gaussian_ids_sparse(projected, opacities, sparse_tiles):
            self.assertEqual(len(result), 2)
            self.assertEqual(len(result[0].unbind()), 3)
            self.assertEqual(len(result[1].unbind()), 0)
        counts, _ = F.rasterize_num_contributing_gaussians_sparse(projected, opacities, sparse_tiles)
        self.assertEqual(len(counts), 2)
        self.assertEqual(counts[1].jdata.numel(), 0)

    def test_single_camera_sparse_analysis_keeps_camera_nesting(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        w2c, K = self.w2c[:1], self.K[:1]
        projected = F.project_gaussians(means, quats, log_scales, w2c, K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        pixels = self._pixels(with_duplicates=True)[0]  # one camera, duplicates included
        pixels = JaggedTensor([pixels.jdata])
        sparse_tiles = F.intersect_gaussian_tiles_sparse(pixels, projected, opacities)
        ids, weights = F.rasterize_contributing_gaussian_ids_sparse(projected, opacities, sparse_tiles)
        self.assertEqual(ids.ldim, 2)
        self.assertEqual(len(ids), 1)
        self.assertEqual(len(ids[0].unbind()), pixels.jdata.shape[0])
        counts, _ = F.rasterize_num_contributing_gaussians_sparse(projected, opacities, sparse_tiles)
        self.assertEqual(int(ids.jdata.numel()), int(counts.jdata.sum()))

    def test_opacities_are_validated_and_made_contiguous(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(
            means, quats, log_scales, self.w2c, self.K, image_width=self.W, image_height=self.H
        )
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        self.assertEqual(tuple(opacities.shape), (self.C, means.shape[0]))
        tiles = F.intersect_gaussian_tiles(projected, opacities, tile_size=16)
        a, _ = F.rasterize_screen_space_gaussians(projected, features, opacities, tiles)
        # Every stage checks the opacities against the projection's camera and Gaussian counts.
        with self.assertRaises(ValueError):
            F.rasterize_screen_space_gaussians(projected, features, opacities[:1], tiles)
        with self.assertRaises(ValueError):
            F.intersect_gaussian_tiles(projected, opacities[:, :-1], tile_size=16)
        with self.assertRaises(ValueError):
            F.rasterize_num_contributing_gaussians(projected, torch.sigmoid(logit_opacities), tiles)
        # A non-contiguous (expanded) opacity tensor is accepted and materialized, not handed to the kernel.
        expanded = torch.sigmoid(logit_opacities).unsqueeze(0).expand(self.C, -1)
        self.assertFalse(expanded.is_contiguous())
        c, _ = F.rasterize_screen_space_gaussians(projected, features, expanded, tiles)
        torch.testing.assert_close(c, a)

    def test_projected_splats_snapshot_opacities_at_projection(self):
        model = self._model(self._params(requires_grad=True))
        pg = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        # Opacities and features come from the same projection, so they carry a gradient together.
        self.assertTrue(pg.opacities.requires_grad)
        self.assertTrue(pg.render_quantities.requires_grad)
        expected = torch.sigmoid(model.logit_opacities.detach()).repeat(self.C, 1)
        torch.testing.assert_close(pg.opacities.detach(), expected)
        # An optimizer step between projection and render must not leak into the projection.
        with torch.no_grad():
            model.logit_opacities.add_(1.0)
        torch.testing.assert_close(pg.opacities.detach(), expected)
        self.assertIs(pg.opacities, pg.opacities)
        # A projection made without a graph renders without one, for both quantities.
        with torch.no_grad():
            preview = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        self.assertFalse(preview.opacities.requires_grad)
        self.assertFalse(preview.render_quantities.requires_grad)

    def test_deduplicate_pixels_keeps_request_order_and_never_merges_out_of_image_pixels(self):
        w, h = 64, 48
        # (0, 64) linearizes to the same key as (1, 0) in a 64-wide image; it must not be merged into it.
        coords = torch.tensor([[3, 7], [1, 2], [3, 7], [0, 64], [1, 0], [5, 5], [1, 2]], dtype=torch.int64)
        coords = coords.to(self.device)
        unique, inverse, has_duplicates = F.deduplicate_pixels(JaggedTensor([coords]), w, h)
        self.assertTrue(has_duplicates)
        self.assertTrue(torch.equal(unique.jdata, coords[[0, 1, 3, 4, 5]]), "first occurrences, in request order")
        self.assertTrue(torch.equal(unique.jdata[inverse], coords))

    def test_as_pixel_jagged_validates_shape_and_dtype(self):
        with self.assertRaises(TypeError):
            F.as_pixel_jagged(torch.zeros(1, 4, 2, device=self.device))
        with self.assertRaises(ValueError):
            F.as_pixel_jagged(torch.zeros(0, 4, 2, dtype=torch.int64, device=self.device))
        with self.assertRaises(ValueError):
            F.as_pixel_jagged(JaggedTensor([torch.zeros(4, 3, dtype=torch.int64, device=self.device)]))
        pixels = F.as_pixel_jagged(torch.zeros(self.C, 4, 2, dtype=torch.int32, device=self.device))
        self.assertEqual(pixels.num_tensors, self.C)

    def test_projected_splats_render_crops_outside_the_image_as_background(self):
        model = self._model(self._params())
        pg = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        self.assertIs(pg.opacities, pg.opacities)
        # Tile intersections are computed on demand, not held by the projection.
        self.assertIsNot(pg.tile_intersection(16), pg.tile_intersection(16))
        background = torch.tensor([[0.2, 0.4, 0.6]] * self.C, device=self.device)
        for origin_w, origin_h in ((self.W, 0), (0, self.H), (self.W + 5, self.H + 5)):
            images, alphas = model.render_from_projected_gaussians(
                pg,
                crop_width=10,
                crop_height=10,
                crop_origin_w=origin_w,
                crop_origin_h=origin_h,
                backgrounds=background,
            )
            self.assertEqual(tuple(images.shape), (self.C, 10, 10, 3))
            torch.testing.assert_close(images, background[:, None, None, :].expand_as(images))
            self.assertEqual(float(alphas.abs().max()), 0.0)

    def test_crops_outside_the_image_stay_differentiable(self):
        params = self._params(requires_grad=True)
        model = self._model(params)
        outside = dict(crop_width=10, crop_height=10, crop_origin_w=self.W, crop_origin_h=0)
        pg = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        images, alphas = model.render_from_projected_gaussians(pg, **outside)
        self.assertTrue(images.requires_grad)
        self.assertTrue(alphas.requires_grad)
        # Nothing was rendered, so backward runs and every gradient it produces is zero.
        (images.sum() + alphas.sum()).backward()
        grads = [p.grad for p in params if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(float(g.abs().max()) == 0.0 for g in grads))
        with torch.no_grad():
            frozen = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
            images, alphas = model.render_from_projected_gaussians(frozen, **outside)
        self.assertFalse(images.requires_grad)
        self.assertFalse(alphas.requires_grad)
        # The stage function has the same contract minus the padding: an all-outside crop is empty.
        empty, empty_alpha = F.rasterize_screen_space_gaussians(
            pg.projected_gaussians,
            pg.render_quantities,
            pg.opacities,
            pg.tile_intersection(16),
            crop=(self.W, 0, 10, 10),
        )
        self.assertEqual(tuple(empty.shape), (self.C, 0, 0, 3))
        self.assertEqual(tuple(empty_alpha.shape), (self.C, 0, 0, 1))
        # ... and short-circuits the rasterizer while staying connected to the features.
        self.assertTrue(empty.requires_grad)
        # A mask of the wrong shape is rejected for an outside crop just as for any other crop.
        bad_mask = torch.ones(self.C, 3, 3, dtype=torch.bool, device=self.device)
        with self.assertRaisesRegex(ValueError, "clipped size"):
            model.render_from_projected_gaussians(pg, masks=bad_mask, **outside)
        good_mask = torch.ones(self.C, 10, 10, dtype=torch.bool, device=self.device)
        images, _ = model.render_from_projected_gaussians(pg, masks=good_mask, **outside)
        self.assertEqual(tuple(images.shape), (self.C, 10, 10, 3))
        # A crop mask the size of the image at a nonzero origin is read in crop coordinates, since crop_masks
        # is a separate argument from the image-coordinate masks.
        crop_mask = torch.zeros(self.C, self.H, self.W, dtype=torch.bool, device=self.device)
        crop_mask[:, :20, :20] = True
        shifted, _ = F.rasterize_screen_space_gaussians(
            pg.projected_gaussians,
            pg.render_quantities,
            pg.opacities,
            pg.tile_intersection(16),
            crop=(5, 7, self.W, self.H),
            crop_masks=crop_mask,
        )
        plain, _ = F.rasterize_screen_space_gaussians(
            pg.projected_gaussians,
            pg.render_quantities,
            pg.opacities,
            pg.tile_intersection(16),
            crop=(5, 7, self.W, self.H),
        )
        torch.testing.assert_close(shifted[:, :20, :20], plain[:, :20, :20])
        self.assertEqual(float(shifted[:, 20:].detach().abs().max()), 0.0)

    def test_empty_selection(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        empty = JaggedTensor([torch.empty(0, 2, dtype=torch.int64, device=self.device) for _ in range(self.C)])
        sparse_tiles = F.intersect_gaussian_tiles_sparse(empty, projected, opacities)
        rendered, alphas = F.rasterize_screen_space_gaussians_sparse(projected, features, opacities, sparse_tiles)
        self.assertEqual(tuple(rendered.jdata.shape), (0, 3))
        self.assertEqual(tuple(alphas.jdata.shape), (0, 1))
        counts, _ = F.rasterize_num_contributing_gaussians_sparse(projected, opacities, sparse_tiles)
        self.assertEqual(counts.jdata.numel(), 0)


if __name__ == "__main__":
    unittest.main()
