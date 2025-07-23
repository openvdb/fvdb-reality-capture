# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import csv
import itertools
import json
import logging
import pathlib
import random
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm
import tyro
import yaml
from datasets import SfmDataset
from datasets.transforms import (
    Compose,
    CropScene,
    DownsampleImages,
    NormalizeScene,
    PercentileFilterPoints,
)
from fvdb.optim import GaussianSplatOptimizer
from sklearn.neighbors import NearestNeighbors
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import gaussian_means_outside_bbox
from viewer import Viewer

from fvdb import GaussianSplat3d


@dataclass
class Config:
    # Random seed
    seed: int = 42

    #
    # Training duration and evaluation parameters
    #

    # Number of training epochs -- i.e. number of times we will visit each image in the dataset
    max_epochs: int = 200
    # Percentage of total epochs at which we perform evaluation on the validation set. i.e. 10 means perform evaluation after 10% of the epochs.
    eval_at_percent: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 75, 100])
    # Percentage of total epochs at which we save the model checkpoint. i.e. 10 means save a checkpoint after 10% of the epochs.
    save_at_percent: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 75, 100])

    #
    # Gaussian Optimization Parameters
    #

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # If you're using very large images, run the forward pass on crops and accumulate gradients
    crops_per_image: int = 1
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this many epochs
    increase_sh_degree_every_epoch: int = 5
    # Initial opacity of each Gaussian
    initial_opacity: float = 0.1
    # Initial scale of each Gaussian
    initial_covariance_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Which network to use for LPIPS loss
    lpips_net: Literal["vgg", "alex"] = "alex"
    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0
    # Use random background for training to discourage transparency
    random_bkgd: bool = False
    # When to start refining (split/duplicate/merge) Gaussians during optimization
    refine_start_epoch: int = 3
    # When to stop refining (split/duplicate/merge) Gaussians during optimization
    refine_stop_epoch: int = 100
    # How often to refine (split/duplicate/merge) Gaussians during optimization
    refine_every_epoch: float = 0.75
    # How often to reset the opacities of the Gaussians during optimization
    reset_opacities_every_epoch: int = 16
    # When to stop using the 2d projected scale for refinement (default of 0 is to never use it)
    refine_using_scale2d_stop_epoch: int = 0

    #
    # Pose optimization parameters
    #

    # Flag to enable camera pose optimization.
    optimize_camera_poses: bool = True
    # Learning rate for camera pose optimization.
    pose_opt_lr: float = 1e-5
    # Weight for regularization of camera pose optimization.
    pose_opt_reg: float = 1e-6
    # Learning rate decay factor for camera pose optimization (will decay to this fraction of initial lr)
    pose_opt_lr_decay: float = 1.0
    # Which epoch to stop optimizing camera postions. Default matches max training epochs.
    pose_opt_stop_epoch: int = max_epochs
    # Standard devation for the normal distribution used for camera pose optimization's random iniitilaization
    pose_opt_init_std: float = 1e-4

    #
    # Gaussian Rendering Parameters
    #

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10
    # Minimum screen space radius below which Gaussians are ignored after projection
    min_radius_2d: float = 0.0
    # Blur amount for anti-aliasing
    eps_2d: float = 0.3
    # Whether to use anti-aliasing or not
    antialias: bool = False
    # Size of tiles to use during rasterization
    tile_size: int = 16


def crop_image_batch(image: torch.Tensor, mask: Optional[torch.Tensor], ncrops: int):
    """
    Generator to iterate a minibatch of images (B, H, W, C) into disjoint patches patches (B, H_patch, W_patch, C).
    We use this function when training on very large images so that we can accumulate gradients over
    crops of each image.

    Args:
        image: Image minibatch (B, H, W, C)
        ncrops: Number of chunks to split the image into (i.e. each crop will have shape (B, H/ncrops x W/ncrops, C).

    Yields: A crop of the input image and its coordinate
        image_patch (torch.Tensor): the patch with shape (B, H/ncrops, W/ncrops, C)
        crop (tuple[int, int, int, int]): the crop coordinates (x, y, w, h),
        is_last (bool): is true if this is the last crop in the iteration
    """
    h, w = image.shape[1:3]
    patch_w, patch_h = w // ncrops, h // ncrops
    patches = np.array(
        [
            [i * patch_w, j * patch_h, (i + 1) * patch_w, (j + 1) * patch_h]
            for i, j in itertools.product(range(ncrops), range(ncrops))
        ]
    )
    for patch_id in range(patches.shape[0]):
        x1, y1, x2, y2 = patches[patch_id]
        image_patch = image[:, y1:y2, x1:x2]
        mask_patch = None
        if mask is not None:
            mask_patch = mask[:, y1:y2, x1:x2]

        crop = (x1, y1, (x2 - x1), (y2 - y1))
        assert (x2 - x1) == patch_w and (y2 - y1) == patch_h
        is_last = patch_id == (patches.shape[0] - 1)
        yield image_patch, mask_patch, crop, is_last


class CameraPoseAdjustment(torch.nn.Module):
    """Camera pose optimization module for 3D Gaussian Splatting.

    This module enables optimization of camera poses defined by their Camera-to-World
    transform. It's generally used to jointly optimize camera poses during training of a
    3D Gaussian Splatting model.

    The model learns a transformation *delta* which applies to the original camera-to-world
    transforms in a dataset.

    The delta is represented as a 9D vector `[dx, dy, dz, r1, r2, r3, r4, r5, r6]`
    which encodes a change in translation and a change in rotation.
    The nine components of the vector are:
    - `[dx, dy, dz]`: translation deltas in world coordinates
    - `[r1, r2, r3, r4, r5, r6]`: 6D rotation representation for stable optimization in machine
    learning, as described in "On the Continuity of Rotation Representations in Neural Networks"
    (Zhou et al., 2019). This representation is preferred over Euler angles or quaternions for
    optimization stability and avoids singularities.

    The module uses an embedding layer to learn these deltas for a fixed number of cameras
    specified at initialization. Generally, this is the number of cameras in the training dataset.\

    You apply this module to a batch of camera-to-world transforms by passing the transforms
    and their corresponding camera indices (in the range `[0, num_cameras-1]`() to the
    `forward` method. The module will return the updated camera-to-world transforms
    after applying the learned deltas.

    Attributes:
        pose_embeddings (torch.nn.Embedding): Embedding layer for learning camera pose deltas.
    """

    def __init__(self, num_cameras: int, init_std: float = 1e-4):
        """
        Create a new `CameraPoseAdjustment` module for storing changes in camera-to-world transforms
        for a fixed number of cameras (`num_cameras`).

        Args:
            num_cameras (int): Number of cameras to learn deltas for.
            init_std (float): Standard deviation for the normal distribution used to initialize
                the pose embeddings.
        """
        super().__init__()

        # Change in positions (3D) + Change in rotations (6D)
        self.pose_embeddings: torch.nn.Embedding = torch.nn.Embedding(num_cameras, 9)

        torch.nn.init.normal_(self.pose_embeddings.weight, std=init_std)

        # Identity rotation in 6D representation
        self.register_buffer("_identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    @property
    def num_cameras(self) -> int:
        """Number of cameras this module is initialized for."""
        return self.pose_embeddings.num_embeddings

    @staticmethod
    def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation described in [1] to a rotation matrix.

        This method uses the Gram-Schmid orthogonalization schemed described in Section B of [1].

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035

        Args:
            d6 (torch.Tensor): 6D rotation representation tensor with shape (*, 6)

        Returns:
            torch.Tensor: batch of rotation matrices with shape (*, 3, 3)
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def forward(self, cam_to_world_matrices: torch.Tensor, camera_ids: torch.Tensor) -> torch.Tensor:
        """Adjust camera pose based on deltas.

        Args:
            cam_to_world_matrices (torch.Tensor): A batch of camera to world transformations
                to adjust. Tnsor of shape (*, 4, 4) where B is the batch size.
            camera_ids (torch.Tensor): Indices of cameras in the batch in the range
            `[0, self.num_cameras -1]`. Tensor of shape (*,).

        Returns:
            torch.Tensor: A batch of updated cam_to_world_matrices where we've applied the
                learned deltas for camera ids to the input camera-to-world transforms.
                i.e. `output[i] = cam_to_world_matrices[i] @ transform[camera_ids[i]]`
        """
        if cam_to_world_matrices.shape[:-2] != camera_ids.shape:
            raise ValueError("`cam_to_world_matrices` and `camera_ids` must have the same batch shape.")
        if cam_to_world_matrices.shape[-2:] != (4, 4):
            raise ValueError("`cam_to_world_matrices` must have shape (..., 4, 4).")

        batch_shape = cam_to_world_matrices.shape[:-2]
        pose_deltas = self.pose_embeddings(camera_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = self._rotation_6d_to_matrix(drot + self._identity.expand(*batch_shape, -1))  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(cam_to_world_matrices, transform)


class TensorboardLogger:
    """
    A utility class to log training metrics to TensorBoard.
    """

    def __init__(self, log_dir: pathlib.Path, log_every_step: int = 100, log_images_to_tensorboard: bool = False):
        """
        Create a new `TensorboardLogger` instance which is used to track training and evaluation progress in tensorboard.

        Args:
            log_dir (pathlib.Path): Directory to save TensorBoard logs.
            log_every_step (int): Log every `log_every_step` steps.
            log_images_to_tensorboard (bool): Whether to log images to TensorBoard.
        """
        self._log_every_step = log_every_step
        self._log_dir = log_dir
        self._log_images_to_tensorboard = log_images_to_tensorboard
        self._tb_writer = SummaryWriter(log_dir=log_dir)

    def log_training_iteration(
        self,
        step: int,
        num_gaussians: int,
        loss: float,
        l1loss: float,
        ssimloss: float,
        mem: float,
        gt_img: torch.Tensor,
        pred_img: torch.Tensor,
        pose_loss: float | None,
    ):
        """
        Log training metrics to TensorBoard.

        Args:
            step: Current training step.
            num_gaussians: Number of Gaussians in the model.
            loss: Total loss value.
            l1loss: L1 loss value.
            ssimloss: SSIM loss value.
            mem: Maximum GPU memory allocated in GB.
            pose_loss: Pose optimization loss, if applicable.
            gt_img: Ground truth image for visualization.
            pred_img: Predicted image for visualization.
        """
        if self._log_every_step > 0 and step % self._log_every_step == 0 and self._tb_writer is not None:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            self._tb_writer.add_scalar("train/loss", loss, step)
            self._tb_writer.add_scalar("train/l1loss", l1loss, step)
            self._tb_writer.add_scalar("train/ssimloss", ssimloss, step)
            self._tb_writer.add_scalar("train/num_gaussians", num_gaussians, step)
            self._tb_writer.add_scalar("train/mem", mem, step)
            # Log pose optimization metrics
            if pose_loss is not None:
                # Log individual components of pose parameters
                self._tb_writer.add_scalar("train/pose_reg_loss", pose_loss, step)
            if self._log_images_to_tensorboard:
                canvas = torch.cat([gt_img, pred_img], dim=2).detach().cpu().numpy()
                canvas = canvas.reshape(-1, *canvas.shape[2:])
                self._tb_writer.add_image("train/render", canvas, step)
            self._tb_writer.flush()

    def log_evaluation_iteration(
        self,
        step: int,
        psnr: float,
        ssim: float,
        lpips: float,
        avg_time_per_image: float,
        num_gaussians: int,
    ):
        """
        Log evaluation metrics to TensorBoard.

        Args:
            step: The training step after which the evaluation was performed.
            psnr: Peak Signal-to-Noise Ratio for the evaluation (averaged over all images in the validation set).
            ssim: Structural Similarity Index Measure for the evaluation (averaged over all images in the validation set).
            lpips: Learned Perceptual Image Patch Similarity for the evaluation (averaged over all images in the validation set).
            avg_time_per_image: Average time taken to evaluate each image.
            num_gaussians: Number of Gaussians in the model at this evaluation step.
        """

        self._tb_writer.add_scalar("eval/psnr", psnr, step)
        self._tb_writer.add_scalar("eval/ssim", ssim, step)
        self._tb_writer.add_scalar("eval/lpips", lpips, step)
        self._tb_writer.add_scalar("eval/avg_time_per_image", avg_time_per_image, step)
        self._tb_writer.add_scalar("eval/num_gaussians", num_gaussians, step)


class ViewerLogger:
    """
    A utility class to visualize the scene being trained and log training statistics and model state to the viewer.
    """

    def __init__(
        self,
        splat_scene: GaussianSplat3d,
        train_dataset: SfmDataset,
        viewer_port: int = 8080,
        verbose: bool = False,
    ):
        """
        Create a new `ViewerLogger` instance which is used to track training and evaluation progress through the viewer.

        Args:
            splat_scene: The GaussianSplat3d scene to visualize.
            train_dataset: The dataset containing camera frames and images.
            viewer_port: The port on which the viewer will run.
            verbose: If True, print additional information about the viewer.
        """

        cam_to_world_matrices, projection_matrices, images = [], [], []
        camera_positions = []
        for data in train_dataset:
            cam_to_world_matrices.append(data["camtoworld"])
            projection_matrices.append(data["K"])
            camera_positions.append(data["camtoworld"][:3, 3].cpu().numpy())
            images.append(data["image"])

        scene_center = np.mean(camera_positions, axis=0)
        scene_radius = np.max(np.linalg.norm(camera_positions - scene_center, axis=1))

        cam_to_world_matrices = np.stack(cam_to_world_matrices, axis=0)
        projection_matrices = np.stack(projection_matrices, axis=0)
        images = np.stack(images, axis=0)

        self.viewer = Viewer(port=viewer_port, verbose=verbose)

        self._splat_model_view = self.viewer.register_gaussian_splat_3d(name="Model", gaussian_scene=splat_scene)

        self._train_camera_view = self.viewer.register_camera_view(
            name="Training Cameras",
            cam_to_world_matrices=cam_to_world_matrices,
            projection_matrices=projection_matrices,
            images=images,
            axis_length=0.05 * scene_radius,
            axis_thickness=0.1 * 0.05 * scene_radius,
            show_images=False,
            enabled=False,
        )

        self._training_metrics_view = self.viewer.register_dictionary_label(
            "Training Metrics",
            {
                "Current Iteration": 0,
                "Current SH Degree": 0,
                "Num Gaussians": 0,
                "Loss": 0.0,
                "SSIM Loss": 0.0,
                "LPIPS Loss": 0.0,
                "GPU Memory Usage": 0,
                "Pose Regularization": 0.0,
            },
        )

        self._evaluation_metrics_view = self.viewer.register_dictionary_label(
            "Evaluation Metrics",
            {
                "Last Evaluation Step": 0,
                "PSNR": 0.0,
                "SSIM": 0.0,
                "LPIPS": 0.0,
                "Evaluation Time": 0.0,
                "Num Gaussians": 0,
            },
        )

    @torch.no_grad
    def pause_for_eval(self):
        self._splat_model_view.allow_enable_in_viewer = False
        self._splat_model_view.enabled = False
        self._training_metrics_view["Status"] = "**Paused for Evaluation**"

    @torch.no_grad
    def resume_after_eval(self):
        self._splat_model_view.allow_enable_in_viewer = True
        self._splat_model_view.enabled = True
        del self._training_metrics_view["Status"]

    @torch.no_grad
    def set_sh_basis_to_view(self, sh_degree: int):
        """
        Set the degree of the spherical harmonics to use in the viewer.

        Args:
            sh_degree: The spherical harmonics degree to view.
        """
        self._splat_model_view.sh_degree = sh_degree

    @torch.no_grad
    def update_camera_poses(self, cam_to_world_matrices: torch.Tensor, image_ids: torch.Tensor):
        """
        Update camera poses in the viewer corresponding to the given image IDs

        Args:
            cam_to_world_matrices: A tensor of shape (B, 4, 4) containing camera-to-world matrices.
            image_ids: A tensor of shape (B,) containing image IDs of the cameras in the training set to update.
        """
        for i in range(len(cam_to_world_matrices)):
            cam_to_world_matrix = cam_to_world_matrices[i].cpu().numpy()
            image_id = int(image_ids[i].item())
            self._train_camera_view[image_id].cam_to_world_matrix = cam_to_world_matrix

    @torch.no_grad
    def log_evaluation_iteration(
        self, step: int, psnr: float, ssim: float, lpips: float, average_time_per_img: float, num_gaussians: int
    ):
        """
        Log data for a single evaluation step to the viewer.

        Args:
            step: The training step after which the evaluation was performed.
            psnr: Peak Signal-to-Noise Ratio for the evaluation (averaged over all images in the validation set).
            ssim: Structural Similarity Index Measure for the evaluation (averaged over all images in the validation set).
            lpips: Learned Perceptual Image Patch Similarity for the evaluation (averaged over all images in the validation set).
            average_time_per_img: Average time taken to evaluate each image.
            num_gaussians: Number of Gaussians in the model at this evaluation step.
        """
        self._evaluation_metrics_view["Last Evaluation Step"] = step
        self._evaluation_metrics_view["PSNR"] = psnr
        self._evaluation_metrics_view["SSIM"] = ssim
        self._evaluation_metrics_view["LPIPS"] = lpips
        self._evaluation_metrics_view["Average Time Per Image (s)"] = average_time_per_img
        self._evaluation_metrics_view["Num Gaussians"] = num_gaussians

    @torch.no_grad
    def log_training_iteration(
        self,
        step: int,
        loss: float,
        l1loss: float,
        ssimloss: float,
        mem: float,
        num_gaussians: int,
        current_sh_degree: int,
        pose_regulation: float | None,
    ):
        """
        Log data for a single training step to the viewer.

        Args:
            step: The current training step.
            loss: Total loss value for the training step.
            l1loss: L1 loss value for the training step.
            ssimloss: SSIM loss value for the training step.
            mem: Maximum GPU memory allocated in GB during this step.
            num_gaussians: Number of Gaussians in the model at this training step.
            current_sh_degree: Current degree of spherical harmonics used in the
            pose_regulation: Pose optimization regularization loss, if applicable.
        """

        self._training_metrics_view["Current Iteration"] = step
        self._training_metrics_view["Current SH Degree"] = current_sh_degree
        self._training_metrics_view["Num Gaussians"] = num_gaussians
        self._training_metrics_view["Loss"] = loss
        self._training_metrics_view["SSIM Loss"] = ssimloss
        self._training_metrics_view["LPIPS Loss"] = l1loss
        self._training_metrics_view["GPU Memory Usage"] = f"{mem:3.2f} GiB"
        if pose_regulation is not None:
            self._training_metrics_view["Pose Regularization"] = f"{pose_regulation:.3e}"
        else:
            if "Pose Regularization" in self._training_metrics_view:
                # Remove the pose regularization key if it was previously set
                del self._training_metrics_view["Pose Regularization"]


class Runner:
    """Engine for training and testing."""

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["splats"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # pose optimization
            if self.cfg.optimize_camera_poses and "pose_optimizer" in checkpoint:
                assert (
                    self.pose_optimizer is not None
                ), "Pose optimizer should be initialized if pose optimization is enabled."
                self.pose_optimizer.load_state_dict(checkpoint["pose_optimizer"])
        self.config = Config(*checkpoint["config"])

        return checkpoint["step"]

    def save_checkpoint(self, step: int):
        if self.checkpoints_path is None:
            self.logger.info("No checkpoints path specified, skipping checkpoint save.")
            return
        checkpoint_path = self.checkpoints_path / pathlib.Path(f"ckpt_{step:04d}.pt")
        ply_path = self.checkpoints_path / pathlib.Path(f"ckpt_{step:04d}.ply")

        checkpoint_path = self.checkpoints_path / pathlib.Path(f"ckpt_{step:04d}.pt")
        ply_path = self.checkpoints_path / pathlib.Path(f"ckpt_{step:04d}.ply")

        self.logger.info(f"Save checkpoint at step {step} to path {checkpoint_path}.")

        checkpoint_data = {
            "step": step,
            "splats": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "transform": self.transform.state_dict(),
            "config": vars(self.cfg),
        }

        # pose optimization
        if self.cfg.optimize_camera_poses:
            assert (
                self.pose_optimizer is not None
            ), "Pose optimizer should be initialized if pose optimization is enabled."
            checkpoint_data["pose_adjust"] = self.adjust_camera_poses.state_dict()
            checkpoint_data["pose_optimizer"] = self.pose_optimizer.state_dict()

        torch.save(checkpoint_data, checkpoint_path)
        self.model.save_ply(str(ply_path))

    def save_statistics(self, step: int, stage: str, stats: dict):
        if self.stats_path is None:
            self.logger.info("No stats path specified, skipping statistics save.")
            return
        stats_path = self.stats_path / pathlib.Path(f"stats_{stage}_{step:04d}.json")

        self.logger.info(f"Saving {stage} statistics at step {step} to path {stats_path}.")

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

    def save_rendered_image(
        self, step: int, stage: str, image_name: str, predicted_image: torch.Tensor, ground_truth_image: torch.Tensor
    ):
        if self.image_render_path is None:
            self.logger.debug("No image render path specified, skipping image save.")
            return
        eval_render_directory_path = self.image_render_path / pathlib.Path(f"{stage}_{step:04d}")
        eval_render_directory_path.mkdir(parents=True, exist_ok=True)
        image_path = eval_render_directory_path / pathlib.Path(image_name)
        self.logger.info(f"Saving {stage} image at step {step} to {image_path}")
        canvas = torch.cat([predicted_image, ground_truth_image], dim=2).squeeze(0).cpu().numpy()
        imageio.imwrite(
            str(image_path),
            (canvas * 255).astype(np.uint8),
        )

    def make_results_directories(
        self, save_results: bool, results_base_path: pathlib.Path, render_eval_images: bool
    ) -> tuple[str | None, pathlib.Path | None, pathlib.Path | None, pathlib.Path | None, pathlib.Path | None]:
        if not save_results:
            return None, None, None, None, None

        results_base_path.mkdir(exist_ok=True)

        attempts = 0
        run_name = f"run_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        while attempts < 50:
            results_path = results_base_path / run_name
            if not (results_path / run_name).exists():
                break
            run_name = f"run_{time.strftime('%Y-%m-%d-%H-%M-%S')}_{attempts+1:02d}"
        if attempts >= 50:
            raise RuntimeError("Failed to generate a unique results directory name after 50 attempts.")

        self.logger.info(f"Creating new training run with name {run_name}.")
        self.logger.info(f"Results will be saved to {results_path.absolute()}.")
        results_path.mkdir(exist_ok=True)
        eval_render_path = None
        if render_eval_images:
            eval_render_path = results_path / pathlib.Path("eval_renders")
            eval_render_path.mkdir(exist_ok=True)

        stats_path = results_path / pathlib.Path("stats")
        stats_path.mkdir(exist_ok=True)

        checkpoints_path = results_path / pathlib.Path("checkpoints")
        checkpoints_path.mkdir(exist_ok=True)

        tensorboard_path = results_path / pathlib.Path("tb")
        tensorboard_path.mkdir(exist_ok=True)

        # Dump config to file
        config_path = results_path / pathlib.Path("cfg.yml")
        with open(config_path, "w") as f:
            yaml.dump(vars(self.cfg), f)

        return run_name, eval_render_path, stats_path, checkpoints_path, tensorboard_path

    def init_model(
        self,
        training_dataset: SfmDataset,
    ):
        def _knn(x_np: np.ndarray, k: int = 4) -> torch.Tensor:
            model = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x_np)
            distances, _ = model.kneighbors(x_np)
            return torch.from_numpy(distances).to(device=self.device, dtype=torch.float32)

        def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
            C0 = 0.28209479177387814
            return (rgb - 0.5) / C0

        num_gaussians = training_dataset.points.shape[0]

        dist2_avg = (_knn(training_dataset.points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        log_scales = torch.log(dist_avg * self.cfg.initial_covariance_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

        means = torch.from_numpy(training_dataset.points).to(device=self.device, dtype=torch.float32)  # [N, 3]
        quats = torch.rand((num_gaussians, 4), device=self.device)  # [N, 4]
        logit_opacities = torch.logit(
            torch.full((num_gaussians,), self.cfg.initial_opacity, device=self.device)
        )  # [N,]

        rgbs = torch.from_numpy(training_dataset.points_rgb / 255.0).to(
            device=self.device, dtype=torch.float32
        )  # [N, 3]
        sh_0 = _rgb_to_sh(rgbs).unsqueeze(1)  # [N, 1, 3]

        sh_n = torch.zeros((num_gaussians, (self.cfg.sh_degree + 1) ** 2 - 1, 3), device=self.device)  # [N, K-1, 3]

        model = GaussianSplat3d(means, quats, log_scales, logit_opacities, sh_0, sh_n, True)
        model.requires_grad = True

        if self.cfg.refine_using_scale2d_stop_epoch > 0:
            model.accumulate_max_2d_radii = True

        return model

    def __init__(
        self,
        cfg: Config,
        data_path: str,
        image_downsample_factor: int = 4,
        points_percentile_filter: float = 0.0,
        normalization_type: Literal["none", "pca", "ecef2enu", "similarity"] = "pca",
        crop_bbox: tuple[float, float, float, float, float, float] | None = None,
        results_path: pathlib.Path = pathlib.Path("results"),
        device: Union[str, torch.device] = "cuda",
        use_every_n_as_test: int = 100,
        disable_viewer: bool = False,
        log_tensorboard_every: int = 100,
        log_images_to_tensorboard: bool = False,
        render_eval_images: bool = False,
        save_results: bool = True,
        disable_masks: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.disable_masks = disable_masks

        self.logger = logging.getLogger("Runner")

        # Setup output directories.
        self.run_name, self.image_render_path, self.stats_path, self.checkpoints_path, self.tensorboard_path = (
            self.make_results_directories(
                save_results=save_results, results_base_path=results_path, render_eval_images=render_eval_images
            )
        )

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Tensorboard
        self.tensorboard_logger = None
        if self.tensorboard_path is not None:
            self.tensorboard_logger = TensorboardLogger(
                log_dir=self.tensorboard_path,
                log_every_step=log_tensorboard_every,
                log_images_to_tensorboard=log_images_to_tensorboard,
            )

        # Dataset transform
        transforms = [
            NormalizeScene(normalization_type=normalization_type),
            PercentileFilterPoints(
                percentile_min=np.full((3,), points_percentile_filter),
                percentile_max=np.full((3,), 100.0 - points_percentile_filter),
            ),
            DownsampleImages(
                image_downsample_factor=image_downsample_factor,
            ),
        ]
        if crop_bbox is not None:
            transforms.append(CropScene(crop_bbox))
        self.transform = Compose(*transforms)

        # Store the crop bounding box and whether to prune Gaussians outside the crop
        # This is used to filter out Gaussians that are outside the crop bounding box during
        # training.
        self.crop_bbox = crop_bbox

        self.trainset = SfmDataset(
            dataset_path=pathlib.Path(data_path),
            test_every=use_every_n_as_test,
            split="train",
            transform=self.transform,
        )
        self.valset = SfmDataset(
            dataset_path=pathlib.Path(data_path),
            test_every=use_every_n_as_test,
            split="test",
            transform=self.transform,
        )
        self.logger.info(
            f"Created dataset training and test datasets with {len(self.trainset)} training images and {len(self.valset)} test images."
        )

        # Initialize model
        self.model = self.init_model(self.trainset)
        self.logger.info(f"Model initialized with {self.model.num_gaussians:,} Gaussians")

        # Viewer
        self.viewer_logger = ViewerLogger(self.model, self.trainset) if not disable_viewer else None

        # Initialize optimizer
        max_steps = cfg.max_epochs * len(self.trainset)
        self.optimizer = GaussianSplatOptimizer(
            self.model, scene_scale=self.trainset.scene_scale * 1.1, mean_lr_decay_exponent=0.01 ** (1.0 / max_steps)
        )

        # Optional camera position optimizer
        self.pose_optimizer = None
        if self.cfg.optimize_camera_poses:
            # Module to adjust camera poses during training
            self.adjust_camera_poses = CameraPoseAdjustment(len(self.trainset), init_std=cfg.pose_opt_init_std).to(
                self.device
            )

            # Increase learning rate for pose optimization and add gradient clipping
            self.pose_optimizer = torch.optim.Adam(
                self.adjust_camera_poses.parameters(),
                lr=self.cfg.pose_opt_lr * 100.0,
                weight_decay=self.cfg.pose_opt_reg,
            )

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.adjust_camera_poses.parameters(), max_norm=1.0)

            # Add learning rate scheduler for pose optimization
            self.pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.pose_optimizer, gamma=self.cfg.pose_opt_lr_decay ** (1.0 / max_steps)
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

    def train(self, start_step: int = 0):
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        # TODO: doesn't account for batch size
        total_steps: int = int(self.cfg.max_epochs * len(self.trainset))
        refine_start_step: int = int(self.cfg.refine_start_epoch * len(self.trainset))
        refine_stop_step: int = int(self.cfg.refine_stop_epoch * len(self.trainset))
        refine_every_step: int = int(self.cfg.refine_every_epoch * len(self.trainset))
        reset_opacities_every_step: int = int(self.cfg.reset_opacities_every_epoch * len(self.trainset))
        refine_using_scale2d_stop_step: int = int(self.cfg.refine_using_scale2d_stop_epoch * len(self.trainset))
        increase_sh_degree_every_step: int = int(self.cfg.increase_sh_degree_every_epoch * len(self.trainset))
        pose_opt_stop_step: int = int(self.cfg.pose_opt_stop_epoch * len(self.trainset))

        pbar = tqdm.tqdm(range(start_step, total_steps), unit="imgs", desc="Training")

        for epoch in range(self.cfg.max_epochs):
            for minibatch in trainloader:
                current_step = pbar.n
                if current_step < start_step:
                    pbar.set_description(f"Skipping step {current_step:,} (before start step {start_step:,})")
                    continue
                if self.viewer_logger is not None:
                    self.viewer_logger.viewer.acquire_lock()

                cam_to_world_mats: torch.Tensor = minibatch["camtoworld"].to(self.device)  # [B, 4, 4]
                world_to_cam_mats: torch.Tensor = minibatch["worldtocam"].to(self.device)  # [B, 4, 4]

                # Camera pose optimization
                image_ids = minibatch["image_id"].to(self.device)  # [B]
                if self.cfg.optimize_camera_poses:
                    if current_step < pose_opt_stop_step:
                        cam_to_world_mats = self.adjust_camera_poses(cam_to_world_mats, image_ids)
                    else:
                        # After pose_opt_stop_iter, don't track gradients through pose adjustment
                        with torch.no_grad():
                            cam_to_world_mats = self.adjust_camera_poses(cam_to_world_mats, image_ids)

                projection_mats = minibatch["K"].to(self.device)  # [B, 3, 3]
                image = minibatch["image"]  # [B, H, W, 3]
                mask = minibatch["mask"] if "mask" in minibatch and not self.disable_masks else None
                image_height, image_width = image.shape[1:3]

                # Progressively use higher spherical harmonic degree as we optimize
                sh_degree_to_use = min(current_step // increase_sh_degree_every_step, self.cfg.sh_degree)
                projected_gaussians = self.model.project_gaussians_for_images(
                    world_to_cam_mats,
                    projection_mats,
                    image_width,
                    image_height,
                    self.cfg.near_plane,
                    self.cfg.far_plane,
                    "perspective",
                    sh_degree_to_use,
                    self.cfg.min_radius_2d,
                    self.cfg.eps_2d,
                    self.cfg.antialias,
                )

                # If you have very large images, you can iterate over disjoint crops and accumulate gradients
                # If cfg.crops_per_image is 1, then this just returns the image
                for pixels, mask_pixels, crop, is_last in crop_image_batch(image, mask, self.cfg.crops_per_image):
                    # Actual pixels to compute the loss on, normalized to [0, 1]
                    pixels = pixels.to(self.device) / 255.0  # [1, H, W, 3]

                    # Render an image from the gaussian splats
                    # possibly using a crop of the full image
                    crop_origin_w, crop_origin_h, crop_w, crop_h = crop
                    colors, alphas = self.model.render_from_projected_gaussians(
                        projected_gaussians, crop_w, crop_h, crop_origin_w, crop_origin_h, self.cfg.tile_size
                    )
                    # If you want to add random background, we'll mix it in here
                    if self.cfg.random_bkgd:
                        bkgd = torch.rand(1, 3, device=self.device)
                        colors = colors + bkgd * (1.0 - alphas)

                    if mask_pixels is not None:
                        # set the ground truth pixel values to match render, thus loss is zero at mask pixels and not updated
                        mask_pixels = mask_pixels.to(self.device)
                        pixels[~mask_pixels] = colors.detach()[~mask_pixels]

                    # Image losses
                    l1loss = F.l1_loss(colors, pixels)
                    ssimloss = 1.0 - self.ssim(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2))
                    loss = l1loss * (1.0 - self.cfg.ssim_lambda) + ssimloss * self.cfg.ssim_lambda

                    # Rgularize opacity to ensure Gaussian's don't become too opaque
                    if self.cfg.opacity_reg > 0.0:
                        loss = loss + self.cfg.opacity_reg * torch.abs(self.model.opacities).mean()

                    # Regularize scales to ensure Gaussians don't become too large
                    if self.cfg.scale_reg > 0.0:
                        loss = loss + self.cfg.scale_reg * torch.abs(self.model.scales).mean()

                    # If you're optimizing poses, regularize the pose parameters so the poses
                    # don't drift too far from the initial values
                    if self.adjust_camera_poses is not None and current_step < pose_opt_stop_step:
                        pose_params = self.adjust_camera_poses.pose_embeddings(image_ids)
                        pose_reg = torch.mean(torch.abs(pose_params))
                        loss = loss + self.cfg.pose_opt_reg * pose_reg

                    # If we're splitting into crops, accumulate gradients, so pass retain_graph=True
                    # for every crop but the last one
                    loss.backward(retain_graph=not is_last)

                # Update the log in the progress bar
                pbar.set_description(f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| ")

                # Refine the gaussians via splitting/duplication/pruning
                if (
                    current_step > refine_start_step
                    and current_step % refine_every_step == 0
                    and current_step < refine_stop_step
                ):
                    num_gaussians_before: int = self.model.num_gaussians
                    use_scales_for_refinement: bool = current_step > reset_opacities_every_step
                    use_screen_space_scales_for_refinement: bool = current_step < refine_using_scale2d_stop_step
                    if not use_screen_space_scales_for_refinement:
                        self.model.accumulate_max_2d_radii = False
                    num_dup, num_split, num_prune = self.optimizer.refine_gaussians(
                        use_scales=use_scales_for_refinement,
                        use_screen_space_scales=use_screen_space_scales_for_refinement,
                    )
                    self.logger.info(
                        f"Step {current_step:,}: Refinement: {num_dup:,} duplicated, {num_split:,} split, {num_prune:,} pruned. "
                        f"Num Gaussians: {self.model.num_gaussians:,} (before: {num_gaussians_before:,})"
                    )
                    # If you specified a crop bounding box, clip the Gaussians that are outside the crop
                    # bounding box. This is useful if you want to train on a subset of the scene
                    # and don't want to waste resources on Gaussians that are outside the crop.
                    if self.crop_bbox is not None:
                        ng_prior = self.model.num_gaussians
                        bad_mask = gaussian_means_outside_bbox(self.crop_bbox, self.model)
                        self.optimizer.remove_gaussians(bad_mask)
                        ng_post = self.model.num_gaussians
                        nclip = ng_prior - ng_post
                        self.logger.info(f"Clipped {nclip:,} Gaussians outside the crop bounding box {self.crop_bbox}.")

                # Reset the opacity parameters every so often
                if current_step % reset_opacities_every_step == 0:
                    self.optimizer.reset_opacities()

                # Step the Gaussian optimizer
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # If you enabled pose optimization, step the pose optimizer if we performed a
                # pose update this iteration
                if self.cfg.optimize_camera_poses and current_step < pose_opt_stop_step:
                    assert (
                        self.pose_optimizer is not None
                    ), "Pose optimizer should be initialized if pose optimization is enabled."
                    self.pose_optimizer.step()
                    self.pose_optimizer.zero_grad(set_to_none=True)
                    # Step the scheduler
                    self.pose_scheduler.step()

                # Log to tensorboard if you requested it
                if self.tensorboard_logger is not None:
                    self.tensorboard_logger.log_training_iteration(
                        current_step,
                        self.model.num_gaussians,
                        loss.item(),
                        l1loss.item(),
                        ssimloss.item(),
                        torch.cuda.max_memory_allocated() / 1024**3,
                        pose_loss=pose_reg.item() if self.cfg.optimize_camera_poses else None,
                        gt_img=pixels,
                        pred_img=colors,
                    )

                # Update the viewer
                if self.viewer_logger is not None:
                    self.viewer_logger.viewer.release_lock()
                    self.viewer_logger.log_training_iteration(
                        current_step,
                        loss=loss.item(),
                        l1loss=l1loss.item(),
                        ssimloss=ssimloss.item(),
                        mem=torch.cuda.max_memory_allocated() / 1024**3,
                        num_gaussians=self.model.num_gaussians,
                        current_sh_degree=sh_degree_to_use,
                        pose_regulation=pose_reg.item() if self.cfg.optimize_camera_poses else None,
                    )
                    if self.cfg.optimize_camera_poses:
                        self.viewer_logger.update_camera_poses(cam_to_world_mats, image_ids)
                    if current_step % increase_sh_degree_every_step == 0 and sh_degree_to_use < self.cfg.sh_degree:
                        self.viewer_logger.set_sh_basis_to_view(sh_degree_to_use)

                pbar.update(self.cfg.batch_size)

            # Save the model if we've reached a percentage of the total epochs specified in save_at_percent
            if epoch in [(pct * self.cfg.max_epochs // 100) - 1 for pct in self.cfg.save_at_percent]:
                self.save_checkpoint(pbar.n - 1)

            # Run evaluation if we've reached a percentage of the total epochs specified in eval_at_percent
            if epoch in [(pct * self.cfg.max_epochs // 100) - 1 for pct in self.cfg.eval_at_percent]:
                if self.viewer_logger is not None:
                    self.viewer_logger.pause_for_eval()
                self.eval(pbar.n - 1)
                if self.viewer_logger is not None:
                    self.viewer_logger.resume_after_eval()

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        self.logger.info("Running evaluation...")
        device = self.device

        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=1)
        evaluation_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            world_to_cam_matrices = data["worldtocam"].to(device)
            projection_matrices = data["K"].to(device)
            ground_truth_image = data["image"].to(device) / 255.0
            mask_pixels = data["mask"] if "mask" in data and not self.disable_masks else None

            height, width = ground_truth_image.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()

            predicted_image, _ = self.model.render_images(
                world_to_cam_matrices,
                projection_matrices,
                width,
                height,
                self.cfg.near_plane,
                self.cfg.far_plane,
                "perspective",
                self.cfg.sh_degree,
                self.cfg.tile_size,
                self.cfg.min_radius_2d,
                self.cfg.eps_2d,
                self.cfg.antialias,
            )
            predicted_image = torch.clamp(predicted_image, 0.0, 1.0)
            # depths = colors[..., -1:] / alphas.clamp(min=1e-10)
            # depths = (depths - depths.min()) / (depths.max() - depths.min())
            # depths = depths / depths.max()

            torch.cuda.synchronize()

            evaluation_time += time.time() - tic

            if mask_pixels is not None:
                # set the ground truth pixel values to match render, thus loss is zero at mask pixels and not updated
                mask_pixels = mask_pixels.to(self.device)
                ground_truth_image[~mask_pixels] = predicted_image.detach()[~mask_pixels]

            # write images
            self.save_rendered_image(step, stage, f"image_{i:04d}", predicted_image, ground_truth_image)

            ground_truth_image = ground_truth_image.permute(0, 3, 1, 2)  # [1, 3, H, W]
            predicted_image = predicted_image.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(predicted_image, ground_truth_image))
            metrics["ssim"].append(self.ssim(predicted_image, ground_truth_image))
            metrics["lpips"].append(self.lpips(predicted_image, ground_truth_image))

        evaluation_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        self.logger.info(f"Evaluation for stage {stage} completed. Average time per image: {evaluation_time:.3f}s")
        self.logger.info(f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f}")

        # Save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "evaluation_time": evaluation_time,
            "num_gaussians": self.model.num_gaussians,
        }
        self.save_statistics(step, stage, stats)

        # Log to tensorboard if enabled
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_evaluation_iteration(
                step, psnr.item(), ssim.item(), lpips.item(), evaluation_time, self.model.num_gaussians
            )

        # Upate the viewer with evaluation results
        if self.viewer_logger is not None:
            self.viewer_logger.log_evaluation_iteration(
                step, psnr.item(), ssim.item(), lpips.item(), evaluation_time, self.model.num_gaussians
            )


def train(
    data_path: str,
    cfg: Config = Config(),
    image_downsample_factor: int = 4,
    points_percentile_filter: float = 0.0,
    normalization_type: Literal["none", "pca", "ecef2enu", "similarity"] = "pca",
    crop_bbox: tuple[float, float, float, float, float, float] | None = None,
    results_path: pathlib.Path = pathlib.Path("results"),
    device: Union[str, torch.device] = "cuda",
    use_every_n_as_val: int = 8,
    disable_viewer: bool = False,
    log_tensorboard_every: int = 100,
    log_images_to_tensorboard: bool = False,
    save_results: bool = True,
    render_eval_images: bool = False,
    disable_masks: bool = False,
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    runner = Runner(
        cfg=cfg,
        data_path=data_path,
        image_downsample_factor=image_downsample_factor,
        points_percentile_filter=points_percentile_filter,
        normalization_type=normalization_type,
        crop_bbox=crop_bbox,
        results_path=results_path,
        device=device,
        use_every_n_as_test=use_every_n_as_val,
        disable_viewer=disable_viewer,
        log_tensorboard_every=log_tensorboard_every,
        log_images_to_tensorboard=log_images_to_tensorboard,
        save_results=save_results,
        render_eval_images=render_eval_images,
        disable_masks=disable_masks,
    )
    runner.train()

    logger = logging.getLogger(__name__)
    if not disable_viewer:
        logger.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(train)
