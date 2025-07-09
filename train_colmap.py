# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import csv
import itertools
import json
import logging
import os
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
from datasets import ColmapDataset
from fvdb.optim import GaussianSplatOptimizer
from sklearn.neighbors import NearestNeighbors
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import CameraOptModule, gaussian_means_outside_bbox
from viewer import Viewer

from fvdb import GaussianSplat3d


@dataclass
class Config:
    # Random seed
    seed: int = 42

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1

    # If you're using very large images, run the forward pass on crops and accumulate gradients
    crops_per_image: int = 1

    # Number of training steps (TODO: Scale with dataset size)
    max_steps: int = 30_000
    # Steps to evaluate the model (TODO: Scale with dataset size)
    eval_steps: List[int] = field(default_factory=lambda: [3_500, 7_000, 30_000])
    # Steps to save the model (TODO: Scale with dataset size)
    save_steps: List[int] = field(default_factory=lambda: [3_500, 7_000, 30_000])

    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps (TODO: Scale with dataset size)
    increase_sh_degree_every: int = 1000
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

    # When to start refining (split/duplicate/merge) Gaussians during optimization
    refine_start_step: int = 500
    # When to stop refining (split/duplicate/merge) Gaussians during optimization
    refine_stop_step: int = 15_000
    # How often to refine (split/duplicate/merge) Gaussians during optimization
    refine_every: int = 100

    # How often to reset the opacities of the Gaussians during optimization
    reset_opacities_every: int = 3000

    # When to stop using the 2d projected scale for refinement (default of 0 is to never use it)
    refine_using_scale2d_stop_iter: int = 0

    # Flag to enable camera pose optimization.
    pose_opt: bool = True
    # Learning rate for camera pose optimization.
    pose_opt_lr: float = 1e-5

    # Weight for regularization of camera pose optimization.
    pose_opt_reg: float = 1e-6

    # Learning rate decay factor for camera pose optimization (will decay to this fraction of initial lr)
    pose_opt_lr_decay: float = 1.0

    # When to stop optimizing camera postions. Default matches max training steps.
    pose_opt_stop_iter: int = max_steps

    # Standard devation for the normal distribution used for camera pose optimization's random iniitilaization
    pose_opt_init_std: float = 1e-4


def crop_image_batch(image: torch.Tensor, mask: Optional[torch.Tensor], ncrops: int):
    """
    Generator to iterate a minibatch of images (B, H, W, C) into disjoint patches patches (B, H_patch, W_patch, C).
    We use this function when training on very large images so that we can accumulate gradients over
    crops of each image.

    Args:
        image: Image minibatch (B, H, W, C)
        ncrops: Number of chunks to split the image into (i.e. each crop will have shape (B, H/ncrops x W/ncrops, C).

    Yields: A crop of the input image and its coordinate
        image_patch: the patch with shape (B, H/ncrops, W/ncrops, C)
        crop: the crop coordinates (x, y, w, h),
        is_last: is true if this is the last crop in the iteration
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


class TensorboardLogger:
    """
    A utility class to log training metrics to TensorBoard.
    """

    def __init__(self, log_dir: str, log_every: int = 100):
        """
        Create a new `TensorboardLogger` instance which is used to track training and evaluation progress in tensorboard.

        Args:
            log_dir: Directory to save TensorBoard logs.
            log_every: Log every `log_every` steps.
        """
        self._log_every = log_every
        self._log_dir = log_dir
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
        show_images_in_tensorboard: bool = False,
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
            show_images_in_tensorboard: Whether to show images in TensorBoard.
        """
        if self._log_every > 0 and step % self._log_every == 0 and self._tb_writer is not None:
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
            if show_images_in_tensorboard:
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
        ellapsed_time: float,
        num_gaussians: int,
    ):

        self._tb_writer.add_scalar("eval/psnr", psnr, step)
        self._tb_writer.add_scalar("eval/ssim", ssim, step)
        self._tb_writer.add_scalar("eval/lpips", lpips, step)
        self._tb_writer.add_scalar("eval/ellapsed_time", ellapsed_time, step)
        self._tb_writer.add_scalar("eval/num_gaussians", num_gaussians, step)


class ViewerLogger:
    """
    A utility class to visualize the scene being trained and log training statistics and model state to the viewer.
    """

    def __init__(
        self,
        splat_scene: GaussianSplat3d,
        train_dataset: ColmapDataset,
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
                "Pose Loss": 0.0,
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
        self, step: int, psnr: float, ssim: float, lpips: float, evaluation_time: float, num_gaussians: int
    ):
        """
        Log data for a single evaluation step to the viewer.

        Args:
            step: The training step after which the evaluation was performed.
            psnr: Peak Signal-to-Noise Ratio for the evaluation (averaged over all images in the validation set).
            ssim: Structural Similarity Index Measure for the evaluation (averaged over all images in the validation set).
            lpips: Learned Perceptual Image Patch Similarity for the evaluation (averaged over all images in the validation set).
            evaluation_time: Time taken for the evaluation in seconds.
            num_gaussians: Number of Gaussians in the model at this evaluation step.
        """
        self._evaluation_metrics_view["Last Evaluation Step"] = step
        self._evaluation_metrics_view["PSNR"] = psnr
        self._evaluation_metrics_view["SSIM"] = ssim
        self._evaluation_metrics_view["LPIPS"] = lpips
        self._evaluation_metrics_view["Evaluation Time"] = evaluation_time
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
        pose_loss: float | None,
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
        """

        self._training_metrics_view["Current Iteration"] = step
        self._training_metrics_view["Current SH Degree"] = current_sh_degree
        self._training_metrics_view["Num Gaussians"] = num_gaussians
        self._training_metrics_view["Loss"] = loss
        self._training_metrics_view["SSIM Loss"] = ssimloss
        self._training_metrics_view["LPIPS Loss"] = l1loss
        self._training_metrics_view["GPU Memory Usage"] = f"{mem:3.2f} GiB"
        if pose_loss is not None:
            self._training_metrics_view["Pose Loss"] = pose_loss


class Runner:
    """Engine for training and testing."""

    def save_checkpoint(self, step):
        if self.no_save:
            return
        mem = torch.cuda.max_memory_allocated() / 1024**3
        stats = {
            "mem": mem,
            "ellapsed_time": time.time() - self.train_start_time,
            "num_gaussians": self.model.num_gaussians,
        }
        checkpoint_path = f"{self.checkpoint_dir}/ckpt_{step:04d}.pt"
        self.logger.info(f"Save checkpoint at step {step} to path {checkpoint_path}. Stats: {stats}.")
        with open(
            f"{self.stats_dir}/train_step{step:04d}.json",
            "w",
        ) as f:
            json.dump(stats, f)
        data = {
            "step": step,
            "splats": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": vars(self.cfg),
        }

        # pose optimization
        if self.cfg.pose_opt:
            assert (
                self.pose_optimizer is not None
            ), "Pose optimizer should be initialized if pose optimization is enabled."
            data["pose_adjust"] = self.pose_adjust.state_dict()
            data["pose_optimizer"] = self.pose_optimizer.state_dict()

        torch.save(data, f"{self.checkpoint_dir}/ckpt_{step:04d}.pt")
        self.model.save_ply(f"{self.checkpoint_dir}/ckpt_{step:04d}.ply")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["splats"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            ##pose optimization
            if self.cfg.pose_opt and "pose_optimizer" in checkpoint:
                assert (
                    self.pose_optimizer is not None
                ), "Pose optimizer should be initialized if pose optimization is enabled."
                self.pose_optimizer.load_state_dict(checkpoint["pose_optimizer"])
        self.config = Config(*checkpoint["config"])

        return checkpoint["step"]

    def make_results_dir(self):
        if self.no_save:
            self.output_dir = None
            self.render_dir = None
            self.stats_dir = None
            self.checkpoint_dir = None
            self.tensorboard_dir = None
            return

        if self.results_path is None:
            os.makedirs("results", exist_ok=True)
            results_name = f"run_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
            tenative_results_dir = os.path.join("results", results_name)
            # If for some reason you have multiple runs at the same second, add a number to the directory
            num_retries = 0
            while os.path.exists(tenative_results_dir) and num_retries < 10:
                results_name = f"run_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                tenative_results_dir = os.path.join("results", f"{results_name}_{num_retries}")
                num_retries += 1
        else:
            tenative_results_dir = os.path.normpath(self.results_path)

        self.output_dir = tenative_results_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.render_dir = os.path.join(self.output_dir, "render")
        os.makedirs(self.render_dir, exist_ok=True)
        self.stats_dir = os.path.join(self.output_dir, "stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.tensorboard_dir = os.path.join(self.output_dir, "tb")
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        # Dump config to file
        with open(f"{self.output_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(self.cfg), f)

        self.logger.info(f"Saving results to {self.output_dir}")

    def init_model(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ):
        def _knn(x_np: np.ndarray, k: int = 4) -> torch.Tensor:
            model = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x_np)
            distances, _ = model.kneighbors(x_np)
            return torch.from_numpy(distances).to(device=self.device, dtype=torch.float32)

        def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
            C0 = 0.28209479177387814
            return (rgb - 0.5) / C0

        num_gaussians = points.shape[0]

        dist2_avg = (_knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        log_scales = torch.log(dist_avg * self.cfg.initial_covariance_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

        means = torch.from_numpy(points).to(device=self.device, dtype=torch.float32)  # [N, 3]
        quats = torch.rand((num_gaussians, 4), device=self.device)  # [N, 4]
        logit_opacities = torch.logit(
            torch.full((num_gaussians,), self.cfg.initial_opacity, device=self.device)
        )  # [N,]

        rgbs = torch.from_numpy(colors / 255.0).to(device=self.device, dtype=torch.float32)  # [N, 3]
        sh_0 = _rgb_to_sh(rgbs).unsqueeze(1)  # [N, 1, 3]

        sh_n = torch.zeros((num_gaussians, (self.cfg.sh_degree + 1) ** 2 - 1, 3), device=self.device)  # [N, K-1, 3]

        self.model = GaussianSplat3d(means, quats, log_scales, logit_opacities, sh_0, sh_n, True)

        if self.cfg.refine_using_scale2d_stop_iter > 0:
            self.model.track_max_2d_radii_for_grad = True

    def __init__(
        self,
        cfg: Config,
        data_path: str,
        data_scale_factor: int = 4,
        results_path: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        use_every_n_as_test: int = 100,
        disable_viewer: bool = False,
        log_tensorboard_every: int = 100,
        log_images_to_tensorboard: bool = False,
        no_save: bool = False,
        no_save_renders: bool = False,
        normalize_ecef2enu: bool = False,
        use_masks: bool = False,
        point_ids_split_path: Optional[str] = None,
        image_ids_split_path: Optional[str] = None,
        split_masks_path: Optional[str] = None,
    ) -> None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.cfg = cfg
        self.disable_viewer = disable_viewer
        self.device = device
        self.log_tensorboard_every = log_tensorboard_every
        self.log_images_to_tensorboard = log_images_to_tensorboard
        self.results_path = results_path
        self.no_save = no_save
        self.no_save_renders = no_save_renders
        self.use_masks = use_masks

        self.logger = logging.getLogger(__name__)

        # Setup output directories.
        self.make_results_dir()

        # Tensorboard
        assert self.tensorboard_dir is not None, "Tensorboard directory should be set."
        self.tensorboard_logger = (
            TensorboardLogger(log_dir=self.tensorboard_dir, log_every=log_tensorboard_every)
            if not self.no_save
            else None
        )

        # Load data: Training data should contain initial points and colors.
        normalization_type = "ecef2enu" if normalize_ecef2enu else "pca"

        image_ids = None
        if image_ids_split_path is not None:
            image_ids = []
            with open(image_ids_split_path, "r") as fp:
                reader = csv.reader(fp)
                hdr = next(reader)
                if hdr[0] != "imageId":
                    raise ValueError("header of image split file should be: imageId")
                else:
                    for row in reader:
                        image_ids.append(int(row[0]))

        self.trainset = ColmapDataset(
            dataset_path=data_path,
            normalization_type=normalization_type,
            image_downsample_factor=data_scale_factor,
            test_every=use_every_n_as_test,
            split="train",
            image_indices=image_ids,
            mask_path=split_masks_path,
        )
        self.valset = ColmapDataset(
            dataset_path=data_path,
            normalization_type=normalization_type,
            image_downsample_factor=data_scale_factor,
            test_every=use_every_n_as_test,
            split="test",
            image_indices=image_ids,
            mask_path=split_masks_path,
        )

        # Initialize model
        if point_ids_split_path is not None:
            point_ids = []
            with open(point_ids_split_path, "r") as fp:
                reader = csv.reader(fp)
                hdr = next(reader)
                if hdr[0] != "pointId":
                    raise ValueError("header of point split file should be: pointId")
                else:
                    for row in reader:
                        point_ids.append(int(row[0]))
            point_ids = np.array(point_ids)
            points = self.trainset.points[point_ids, :]
            points_rgb = self.trainset.points_rgb[point_ids, :]

            # Calculate the point bounds from split and save for trimming later
            self.clip_bounds = np.concatenate((np.min(points, axis=0), np.max(points, axis=0)))
            self.clip_bounds = torch.from_numpy(self.clip_bounds).float().to(self.device)

            ## TODO doesn't work well typically when from SFM due to outliers, user should percentile clean
            # Calculate scene bound from points provided by user
            scene_center = np.mean(points, axis=0)
            dists = np.linalg.norm(points - scene_center, axis=1)
            scene_scale = np.max(dists) * 1.1

        else:
            points = self.trainset.points
            points_rgb = self.trainset.points_rgb
            self.clip_bounds = None
            scene_scale = self.trainset.colmap_scene.scene_scale * 1.1

        self.logger.info(f"Created dataset. Scene scale = {scene_scale}")

        # Initialize model
        self.init_model(points, points_rgb)
        self.logger.info(f"Model initialized with {self.model.num_gaussians} Gaussians")

        # Initialize optimizer
        self.optimizer = GaussianSplatOptimizer(
            self.model, scene_scale=scene_scale, mean_lr_decay_exponent=0.01 ** (1.0 / cfg.max_steps)
        )

        # camera position optimizer
        self.pose_optimizer = None
        if self.cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            # Initialize with small random values instead of zeros
            self.pose_adjust.random_init(std=cfg.pose_opt_init_std)  # Small initial values to start with
            # Increase learning rate for pose optimization and add gradient clipping
            self.pose_optimizer = torch.optim.Adam(
                self.pose_adjust.parameters(),
                lr=self.cfg.pose_opt_lr * 100.0,
                weight_decay=self.cfg.pose_opt_reg,
            )

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.pose_adjust.parameters(), max_norm=1.0)

            # Add learning rate scheduler for pose optimization
            self.pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.pose_optimizer, gamma=self.cfg.pose_opt_lr_decay ** (1.0 / self.cfg.max_steps)
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

        # Viewer
        self.viewer_logger = ViewerLogger(self.model, self.trainset) if not self.disable_viewer else None

    def train(self, start_step: int = 0):
        # We keep cycling through every image in a random order until we reach
        # the specified number of optimization steps. We can't use itertools.cycle
        # because it caches each minibatch element in memory which can quickly
        # exhaust the amount of available RAM
        def cycle(dataloader):
            while True:
                for minibatch in dataloader:
                    yield minibatch

        trainloader = cycle(
            torch.utils.data.DataLoader(
                self.trainset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
            )
        )

        self.train_start_time = time.time()
        pbar = tqdm.tqdm(range(start_step, self.cfg.max_steps))
        for step in pbar:
            if self.viewer_logger is not None:
                self.viewer_logger.viewer.acquire_lock()

            minibatch = next(trainloader)
            cam_to_world_mats = minibatch["camtoworld"].to(self.device)  # [B, 4, 4]
            world_to_cam_mats = minibatch["worldtocam"].to(self.device)  # [B, 4, 4]

            # Camera pose optimization
            image_ids = minibatch["image_id"].to(self.device)  # [B]
            if self.cfg.pose_opt and step < self.cfg.pose_opt_stop_iter:
                cam_to_world_mats = self.pose_adjust(cam_to_world_mats, image_ids)
            else:
                # After pose_opt_stop_iter, use original camera poses
                cam_to_world_mats = minibatch["camtoworld"].to(self.device)
            projection_mats = minibatch["K"].to(self.device)  # [B, 3, 3]
            image = minibatch["image"]  # [B, H, W, 3]
            mask = minibatch["mask"] if "mask" in minibatch else None
            image_height, image_width = image.shape[1:3]

            # Progressively use higher spherical harmonic degree as we optimize
            sh_degree_to_use = min(step // self.cfg.increase_sh_degree_every, self.cfg.sh_degree)
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

                if self.use_masks:
                    if mask_pixels is None:
                        raise ValueError("use masks set, but no mask images available")
                    else:
                        # set the ground truth pixel values to match render, thus loss is zero at mask pixels and not updated
                        mask_pixels = mask_pixels.to(self.device)
                        pixels[mask_pixels] = colors.detach()[mask_pixels]

                # Image losses
                l1loss = F.l1_loss(colors, pixels)
                ssimloss = 1.0 - self.ssim(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2))
                loss = l1loss * (1.0 - self.cfg.ssim_lambda) + ssimloss * self.cfg.ssim_lambda

                # Regularization losses
                if self.cfg.opacity_reg > 0.0:
                    loss = loss + self.cfg.opacity_reg * torch.abs(self.model.opacities).mean()
                if self.cfg.scale_reg > 0.0:
                    loss = loss + self.cfg.scale_reg * torch.abs(self.model.scales).mean()
                    # Add pose regularization to encourage small pose changes
                if self.cfg.pose_opt and step < self.cfg.pose_opt_stop_iter:
                    pose_params = self.pose_adjust.pose_embeddings(image_ids)
                    pose_reg = torch.mean(torch.abs(pose_params))
                    loss = loss + self.cfg.pose_opt_reg * pose_reg
                # If we're splitting into crops, accumulate gradients
                loss.backward(retain_graph=not is_last)

            # Update the log in the progress bar
            pbar.set_description(f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| ")

            # Log to tensorboard if you requested it
            if self.tensorboard_logger is not None:
                self.tensorboard_logger.log_training_iteration(
                    step,
                    self.model.num_gaussians,
                    loss.item(),
                    l1loss.item(),
                    ssimloss.item(),
                    torch.cuda.max_memory_allocated() / 1024**3,
                    pose_loss=pose_reg.item() if self.cfg.pose_opt else None,
                    gt_img=pixels,
                    pred_img=colors,
                    show_images_in_tensorboard=self.log_images_to_tensorboard,
                )

            # save checkpoint before updating the model; clip if provided cliping bounds
            if step in [i - 1 for i in self.cfg.save_steps] or step == self.cfg.max_steps - 1:
                if self.viewer_logger is not None:
                    self.viewer_logger.pause_for_eval()
                self.save_checkpoint(step)
                if self.viewer_logger is not None:
                    self.viewer_logger.resume_after_eval()

            # Refine the gaussians via splitting/duplication/pruning
            if (
                step > self.cfg.refine_start_step
                and step % self.cfg.refine_every == 0
                and step < self.cfg.refine_stop_step
            ):
                num_gaussians_before = self.model.num_gaussians
                use_scales_for_refinement = step > self.cfg.reset_opacities_every
                use_screen_space_scales_for_refinement = step < self.cfg.refine_using_scale2d_stop_iter
                if not use_screen_space_scales_for_refinement:
                    self.model.track_max_2d_radii_for_grad = False
                num_dup, num_split, num_prune = self.optimizer.refine_gaussians(
                    use_scales=use_scales_for_refinement, use_screen_space_scales=use_screen_space_scales_for_refinement
                )
                self.logger.info(
                    f"Step {step}: Refinement: {num_dup} duplicated, {num_split} split, {num_prune} pruned. "
                    f"Num Gaussians: {self.model.num_gaussians} (before: {num_gaussians_before})"
                )
                if self.clip_bounds is not None:
                    ng_prior = self.model.num_gaussians
                    bad_mask = gaussian_means_outside_bbox(self.clip_bounds, self.model)
                    self.optimizer.remove_gaussians(bad_mask)
                    ng_post = self.model.num_gaussians
                    nclip = ng_prior - ng_post
                    self.logger.info(f"Clipped {nclip} Gaussians using clip bounds")

            # Reset the opacity parameters every so often
            if step % self.cfg.reset_opacities_every == 0:
                self.optimizer.reset_opacities()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # pose optimization
            if self.cfg.pose_opt and step < self.cfg.pose_opt_stop_iter:
                assert (
                    self.pose_optimizer is not None
                ), "Pose optimizer should be initialized if pose optimization is enabled."
                self.pose_optimizer.step()
                self.pose_optimizer.zero_grad(set_to_none=True)
                # Step the scheduler
                self.pose_scheduler.step()

            # Run evaluation every eval_steps
            if step in [i - 1 for i in self.cfg.eval_steps]:
                if self.viewer_logger is not None:
                    self.viewer_logger.pause_for_eval()
                self.eval(step)
                if self.viewer_logger is not None:
                    self.viewer_logger.resume_after_eval()

            # Update the viewer
            if self.viewer_logger is not None:
                self.viewer_logger.viewer.release_lock()
                self.viewer_logger.log_training_iteration(
                    step,
                    loss=loss.item(),
                    l1loss=l1loss.item(),
                    ssimloss=ssimloss.item(),
                    mem=torch.cuda.max_memory_allocated() / 1024**3,
                    num_gaussians=self.model.num_gaussians,
                    current_sh_degree=sh_degree_to_use,
                    pose_loss=pose_reg.item() if self.cfg.pose_opt else None,
                )
                if self.cfg.pose_opt:
                    self.viewer_logger.update_camera_poses(cam_to_world_mats, image_ids)
                if step % self.cfg.increase_sh_degree_every == 0 and sh_degree_to_use < self.cfg.sh_degree:
                    self.viewer_logger.set_sh_basis_to_view(sh_degree_to_use)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        self.logger.info("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=1)
        evaluation_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            world_to_cam_mats = data["worldtocam"].to(device)
            projection_mats = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            mask_pixels = data["mask"] if "mask" in data else None

            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()

            colors, _ = self.model.render_images(
                world_to_cam_mats,
                projection_mats,
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
            colors = torch.clamp(colors, 0.0, 1.0)
            # depths = colors[..., -1:] / alphas.clamp(min=1e-10)
            # depths = (depths - depths.min()) / (depths.max() - depths.min())
            # depths = depths / depths.max()

            torch.cuda.synchronize()

            evaluation_time += time.time() - tic

            if self.use_masks:
                if mask_pixels is None:
                    raise ValueError("use masks set, but no mask images available")
                else:
                    # set the ground truth pixel values to match render, thus loss is zero at mask pixels and not updated
                    mask_pixels = mask_pixels.to(self.device)
                    pixels[mask_pixels] = colors.detach()[mask_pixels]

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            if not self.no_save:
                if not self.no_save_renders:
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_{i:04d}.png",
                        (canvas * 255).astype(np.uint8),
                    )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        evaluation_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        self.logger.info(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {evaluation_time:.3f}s/image "
            f"Number of Gaussians: {self.model.num_gaussians}"
        )
        # Save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "evaluation_time": evaluation_time,
            "num_gaussians": self.model.num_gaussians,
        }
        if not self.no_save:
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)

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
    data_scale_factor: int = 4,
    results_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    use_every_n_as_test: int = 8,
    disable_viewer: bool = False,
    log_tensorboard_every: int = 100,
    log_images_to_tensorboard: bool = False,
    no_save: bool = False,
    no_save_renders: bool = False,
    normalize_ecef2enu: bool = False,
    use_masks: bool = False,
    point_ids_split_path: Optional[str] = None,
    image_ids_split_path: Optional[str] = None,
    split_masks_path: Optional[str] = None,
):
    logging.basicConfig(level=logging.INFO)
    runner = Runner(
        cfg,
        data_path,
        data_scale_factor,
        results_path,
        device,
        use_every_n_as_test,
        disable_viewer,
        log_tensorboard_every,
        log_images_to_tensorboard,
        no_save,
        no_save_renders,
        normalize_ecef2enu,
        use_masks,
        point_ids_split_path,
        image_ids_split_path,
        split_masks_path,
    )
    runner.train()
    if not disable_viewer:
        runner.logger.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(train)
