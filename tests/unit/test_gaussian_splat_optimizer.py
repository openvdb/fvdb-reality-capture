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

    def render_one_image(self, model):
        data_item = self.training_dataset[0]
        projection_matrix = data_item["projection"].to(device=self.device).unsqueeze(0)
        world_to_camera_matrix = data_item["world_to_camera"].to(device=self.device).unsqueeze(0)
        gt_image = torch.from_numpy(data_item["image"]).to(device=self.device).unsqueeze(0).float() / 255.0

        pred_image, alphas = model.render_images(
            world_to_camera_matrices=world_to_camera_matrix,
            projection_matrices=projection_matrix,
            image_width=gt_image.shape[2],
            image_height=gt_image.shape[1],
            near=0.1,
            far=1e10,
        )
        return gt_image, pred_image, alphas

    def test_serialize_optimizer(self):
        model_1 = self.model
        max_steps = 200 * len(self.training_dataset)
        config = frc.training.GaussianSplatOptimizerConfig()
        optimizer_1 = frc.training.GaussianSplatOptimizer.from_model_and_config(
            model=self.model,
            config=config,
            batch_size=1,
            means_lr_decay_exponent=0.01 ** (1.0 / max_steps),
        )

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pt", delete=True) as temp_file:
            # Save the state dict of the optimizer and model
            torch.save(self.model.state_dict(), temp_file.name + ".model")
            torch.save(optimizer_1.state_dict(), temp_file.name)

            # Run one step of the optimization and refinement
            optimizer_1.zero_grad()
            gt_img_1, pred_img_1, _ = self.render_one_image(model_1)
            loss_1 = torch.nn.functional.l1_loss(pred_img_1, gt_img_1)
            loss_1.backward()
            optimizer_1.refine_gaussians(True)
            optimizer_1.step()
            optimizer_1.zero_grad()
            num_gaussians_after_refine = model_1.num_gaussians

            # Compute the rendered image and loss after one step
            gt_img_2, pred_img_2, _ = self.render_one_image(model_1)
            loss_2 = torch.nn.functional.l1_loss(pred_img_2, gt_img_2)
            # print(f"loss_2 = {loss_2.item()}")

            # Load the original model and optimizer from the saved state dict
            model_2 = GaussianSplat3d.from_state_dict(torch.load(temp_file.name + ".model", map_location=self.device))
            loaded_state_dict = torch.load(temp_file.name, map_location=self.device, weights_only=False)
            optimizer_2 = frc.training.GaussianSplatOptimizer.from_state_dict(model_2, loaded_state_dict)

            # Run one step of of optimization and refinement with the loaded optimizer and model
            # and check that the results match the previous results
            optimizer_2.zero_grad()
            gt_img_3, pred_img_3, _ = self.render_one_image(model_2)
            self.assertTrue(torch.allclose(pred_img_1, pred_img_3))
            loss_3 = torch.nn.functional.l1_loss(pred_img_3, gt_img_3)
            self.assertAlmostEqual(loss_1.item(), loss_3.item(), places=3)
            loss_3.backward()
            optimizer_2.refine_gaussians(True)
            optimizer_2.step()
            optimizer_2.zero_grad()
            self.assertEqual(model_2.num_gaussians, num_gaussians_after_refine)

            # Compute the rendered image and loss after one step and check that it matches the previous result
            gt_img_4, pred_img_4, _ = self.render_one_image(model_2)
            self.assertTrue(torch.allclose(pred_img_2, pred_img_4, atol=1e-3))
            loss_4 = torch.nn.functional.l1_loss(pred_img_4, gt_img_4)
            self.assertAlmostEqual(loss_2.item(), loss_4.item(), places=3)


if __name__ == "__main__":
    unittest.main()
