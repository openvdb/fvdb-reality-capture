# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import functools
import multiprocessing
import pathlib
import shutil
import tempfile
import unittest

import numpy as np
import torch
from fvdb import GaussianSplat3d
from scipy.spatial import cKDTree  # type: ignore

import fvdb_reality_capture as frc


class BasicCacheTest(unittest.TestCase):
    @staticmethod
    def _init_model(
        device: torch.device | str,
        training_dataset: frc.training.SfmDataset,
    ):
        """
        Initialize a Gaussian Splatting model with random parameters based on the training dataset.

        Args:
            device: The device to run the model on (e.g., "cuda" or "cpu").
            training_dataset: The dataset used for training, which provides the initial points and RGB values
                            for the Gaussians.
        """

        initial_covariance_scale = 1.0
        initial_opacity = 0.1
        sh_degree = 3

        def _knn(x_np: np.ndarray, k: int = 4) -> torch.Tensor:
            kd_tree = cKDTree(x_np)  # type: ignore
            distances, _ = kd_tree.query(x_np, k=k)
            return torch.from_numpy(distances).to(device=device, dtype=torch.float32)

        def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
            C0 = 0.28209479177387814
            return (rgb - 0.5) / C0

        num_gaussians = training_dataset.points.shape[0]

        dist2_avg = (_knn(training_dataset.points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        log_scales = torch.log(dist_avg * initial_covariance_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

        means = torch.from_numpy(training_dataset.points).to(device=device, dtype=torch.float32)  # [N, 3]
        quats = torch.rand((num_gaussians, 4), device=device)  # [N, 4]
        logit_opacities = torch.logit(torch.full((num_gaussians,), initial_opacity, device=device))  # [N,]

        rgbs = torch.from_numpy(training_dataset.points_rgb / 255.0).to(device=device, dtype=torch.float32)  # [N, 3]
        sh_0 = _rgb_to_sh(rgbs).unsqueeze(1)  # [N, 1, 3]

        sh_n = torch.zeros((num_gaussians, (sh_degree + 1) ** 2 - 1, 3), device=device)  # [N, K-1, 3]

        model = GaussianSplat3d(means, quats, log_scales, logit_opacities, sh_0, sh_n, True)
        model.requires_grad = True

        model.accumulate_max_2d_radii = False

        return model

    @staticmethod
    def _compute_scene_scale(sfm_scene: frc.SfmScene) -> float:
        median_depth_per_camera = []
        for image_meta in sfm_scene.images:
            # Don't use cameras that don't see any points in the estimate
            if len(image_meta.point_indices) == 0:
                continue
            points = sfm_scene.points[image_meta.point_indices]
            dist_to_points = np.linalg.norm(points - image_meta.origin, axis=1)
            median_dist = np.median(dist_to_points)
            median_depth_per_camera.append(median_dist)
        return float(np.median(median_depth_per_camera))

    def setUp(self):
        # Auto-download this dataset if it doesn't exist.
        self.dataset_path = pathlib.Path(__file__).parent.parent.parent / "data" / "gettysburg"
        if not self.dataset_path.exists():
            frc.tools.download_example_data("gettysburg", self.dataset_path.parent)

        scene = frc.SfmScene.from_colmap(self.dataset_path)
        transform = frc.transforms.Compose(
            frc.transforms.NormalizeScene("pca"),
            frc.transforms.DownsampleImages(4),
        )
        self.training_dataset = frc.training.SfmDataset(transform(scene))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._init_model(self.device, self.training_dataset)
        self.scene_scale = self._compute_scene_scale(scene)

    def test_create_optimizer(self):
        max_steps = 200 * len(self.training_dataset)
        optimizer = frc.training.GaussianSplatOptimizer(
            self.model,
            scene_scale=self.scene_scale * 1.1,
            mean_lr_decay_exponent=0.01 ** (1.0 / max_steps),
        )
        state_dict = optimizer.state_dict()
        self.assertEqual(state_dict["mean_lr_decay_exponent"], 0.01 ** (1.0 / max_steps))
        self.assertEqual(state_dict["prune_opacity_threshold"], 0.005)
        self.assertEqual(state_dict["prune_scale3d_threshold"], 0.1)
        self.assertEqual(state_dict["prune_scale2d_threshold"], 0.15)
        self.assertEqual(state_dict["grow_grad2d_threshold"], None)
        self.assertEqual(state_dict["grow_grad_2d_percentile_threshold"], 0.9)
        self.assertEqual(state_dict["scale_3d_threshold_relative_units"], True)
        self.assertEqual(state_dict["grow_scale3d_threshold"], 0.01)
        self.assertEqual(state_dict["grow_scale2d_threshold"], 0.05)
        self.assertEqual(state_dict["absgrad"], False)
        self.assertEqual(state_dict["adaptive_grad2d_threshold"], False)
        self.assertEqual(state_dict["did_first_refinement"], False)
        self.assertEqual(state_dict["revised_opacity"], False)
        self.assertEqual(state_dict["version"], 3)

    def test_can_only_use_percentiles_or_absolute_grow_grad_2d(self):
        max_steps = 200 * len(self.training_dataset)
        with self.assertRaises(ValueError):
            optimizer = frc.training.GaussianSplatOptimizer(
                self.model,
                scene_scale=self.scene_scale * 1.1,
                mean_lr_decay_exponent=0.01 ** (1.0 / max_steps),
                grow_grad_2d_percentile_threshold=0.9,
                grow_grad2d_threshold=0.001,
            )
        with self.assertRaises(ValueError):
            optimizer = frc.training.GaussianSplatOptimizer(
                self.model,
                scene_scale=self.scene_scale * 1.1,
                mean_lr_decay_exponent=0.01 ** (1.0 / max_steps),
                grow_grad_2d_percentile_threshold=None,
                grow_grad2d_threshold=None,
            )

        optimizer = frc.training.GaussianSplatOptimizer(
            self.model,
            scene_scale=self.scene_scale * 1.1,
            mean_lr_decay_exponent=0.01 ** (1.0 / max_steps),
            grow_grad_2d_percentile_threshold=0.9,
            grow_grad2d_threshold=None,
        )
        optimizer = frc.training.GaussianSplatOptimizer(
            self.model,
            scene_scale=self.scene_scale * 1.1,
            mean_lr_decay_exponent=0.01 ** (1.0 / max_steps),
            grow_grad_2d_percentile_threshold=None,
            grow_grad2d_threshold=0.0002,
        )


if __name__ == "__main__":
    unittest.main()
