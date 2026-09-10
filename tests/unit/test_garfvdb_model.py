# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import unittest

import fvdb
import torch

from fvdb_reality_capture import GaussianSplat3d
from fvdb_reality_capture.instance_segmentation import GARfVDBConfig
from fvdb_reality_capture.instance_segmentation.model import GARfVDBModel


def _make_model(device: str) -> GARfVDBModel:
    generator = torch.Generator().manual_seed(7)
    num_gaussians = 64
    means = torch.rand((num_gaussians, 3), generator=generator)
    quats = torch.rand((num_gaussians, 4), generator=generator)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    log_scales = torch.full((num_gaussians, 3), -2.0)
    logit_opacities = torch.full((num_gaussians,), 2.0)
    sh0 = torch.rand((num_gaussians, 1, 3), generator=generator)
    shN = torch.zeros((num_gaussians, 0, 3))
    gaussians = GaussianSplat3d.from_tensors(means, quats, log_scales, logit_opacities, sh0, shN).to(device)
    config = GARfVDBConfig(
        depth_samples=8,
        num_grids=4,
        grid_feature_dim=2,
        mlp_hidden_dim=8,
        mlp_num_layers=1,
        mlp_output_dim=4,
    )
    return GARfVDBModel(gaussians, torch.tensor([0.05, 0.1, 0.2, 0.4], device=device), config, device=device)


def _make_input(device: str, pixel_coords: torch.Tensor) -> dict:
    # OpenCV camera above the unit cube of Gaussian means, looking down -z.
    camera_to_world = torch.eye(4, device=device)
    camera_to_world[1, 1] = -1.0
    camera_to_world[2, 2] = -1.0
    camera_to_world[:3, 3] = torch.tensor([0.5, 0.5, 3.0], device=device)
    world_to_camera = torch.linalg.inv(camera_to_world).contiguous()
    projection = torch.tensor(
        [[24.0, 0.0, 16.0], [0.0, 24.0, 16.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    return {
        "image_w": [32],
        "image_h": [32],
        "projection": projection[None],
        "world_to_camera": world_to_camera[None],
        "camera_to_world": camera_to_world[None],
        "pixel_coords": pixel_coords[None].to(device),
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GARfVDBEncodedFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0"
        self.model = _make_model(self.device)

    def test_duplicate_pixels_reduce_per_pixel(self):
        grid = torch.stack(torch.meshgrid(torch.arange(4, 32, 6), torch.arange(4, 32, 6), indexing="ij"), -1)
        pixels = grid.reshape(-1, 2)
        pixels = torch.cat([pixels, pixels[:3]])  # duplicates
        features = self.model.get_encoded_features(_make_input(self.device, pixels))
        # [B, R, S, F] with the depth samples reduced to one per pixel.
        self.assertEqual(features.shape[:3], (1, pixels.shape[0], 1))
        self.assertTrue(features.abs().sum() > 0, "camera setup hit no Gaussians")
        torch.testing.assert_close(features[0, -3:], features[0, :3])

    def test_one_level_render_result_is_rejected(self):
        pixels = torch.tensor([[10, 10], [16, 16], [20, 12]])
        original = self.model.gs_model.sparse_render_contributing_gaussian_ids

        def collapsed(*args, **kwargs):
            ids, weights = original(*args, **kwargs)
            # Collapse to one row per pixel, the shape a broken render would return.
            n = pixels.shape[0]
            offsets = torch.tensor([0, n], device=ids.device, dtype=torch.long)
            ids_1 = fvdb.JaggedTensor.from_data_and_offsets(ids.jdata[:n], offsets)
            weights_1 = fvdb.JaggedTensor.from_data_and_offsets(weights.jdata[:n], offsets)
            return ids_1, weights_1

        self.model.gs_model.sparse_render_contributing_gaussian_ids = collapsed
        with self.assertRaisesRegex(RuntimeError, "1-level JaggedTensor"):
            self.model.get_encoded_features(_make_input(self.device, pixels))


if __name__ == "__main__":
    unittest.main()
