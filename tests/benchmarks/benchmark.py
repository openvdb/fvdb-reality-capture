# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import sys
from typing import Literal, Optional, Union

import pytest
import torch
import torch.nn.functional as F
import torch.utils.data
import yaml
from fvdb_3dgs.io import Cache, load_colmap_dataset
from fvdb_3dgs.sfm_scene import SfmScene
from fvdb_3dgs.training import Checkpoint, Config, SfmDataset

from fvdb import GaussianSplat3d

logger = logging.getLogger("Benchmark 3dgs")


def load_benchmark_config(config_path: str = "benchmark_config.yaml") -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class Benchmark3dgs:
    def __init__(
        self,
        data_path: str,
        checkpoint_path: str,
        results_path: Optional[pathlib.Path] = None,
        image_downsample_factor: int = 4,
        points_percentile_filter: float = 0.0,
        normalization_type: Literal["none", "pca", "ecef2enu", "similarity"] = "pca",
        crop_bbox: tuple[float, float, float, float, float, float] | None = None,
        device: Union[str, torch.device] = "cuda",
    ):
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.results_path = pathlib.Path(checkpoint_path).parent.parent if results_path is None else results_path

        # Load the checkpoint
        self.checkpoint = Checkpoint.load(pathlib.Path(checkpoint_path), device=device)

        sfm_scene: SfmScene
        cache: Cache
        sfm_scene, cache = load_colmap_dataset(self.checkpoint.dataset_path)
        sfm_scene, cache = self.checkpoint.dataset_transform(sfm_scene, cache)

        if "train" not in self.checkpoint.dataset_splits:
            raise ValueError("No training dataset found in checkpoint")
        train_indices = self.checkpoint.dataset_splits["train"]

        self.config = Config(**self.checkpoint.config)
        self.train_dataset = SfmDataset(sfm_scene, train_indices)

        step = self.checkpoint.step if self.checkpoint.step is not None else 0

        trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # for benchmarking always use the same order of the dataset
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        minibatch = next(iter(trainloader))

        self.cam_to_world_mats: torch.Tensor = minibatch["camera_to_world"].to(device)  # [B, 4, 4]
        self.world_to_cam_mats: torch.Tensor = minibatch["world_to_camera"].to(device)  # [B, 4, 4]
        self.projection_mats = minibatch["projection"].to(device)  # [B, 3, 3]
        self.image = minibatch["image"]  # [B, H, W, 3]
        self.mask = minibatch["mask"] if "mask" in minibatch else None
        self.image_height, self.image_width = self.image.shape[1:3]

        # Actual pixels to compute the loss on, normalized to [0, 1]
        self.pixels = self.image.to(device) / 255.0  # [1, H, W, 3]

        # Progressively use higher spherical harmonic degree as we optimize
        increase_sh_degree_every_step: int = int(self.config.increase_sh_degree_every_epoch * len(self.train_dataset))
        self.sh_degree_to_use = min(step // increase_sh_degree_every_step, self.config.sh_degree)

        # run pipeline once to warm up and enable running the benchmarks in any order (or filtered)
        self.run_project_gaussians()
        self.run_render_gaussians()
        self.run_backward()

    def run_project_gaussians(self):
        self.projected_gaussians = self.checkpoint.splats.project_gaussians_for_images(
            self.world_to_cam_mats,
            self.projection_mats,
            self.image_width,
            self.image_height,
            self.config.near_plane,
            self.config.far_plane,
            "perspective",
            self.sh_degree_to_use,
            self.config.min_radius_2d,
            self.config.eps_2d,
            self.config.antialias,
        )

    def run_render_gaussians(self):
        # Render an image from the gaussian splats
        # possibly using a crop of the full image
        self.colors, self.alphas = self.checkpoint.splats.render_from_projected_gaussians(
            self.projected_gaussians, self.image_width, self.image_height, 0, 0, self.config.tile_size
        )

    def run_forward(self):
        self.run_project_gaussians()
        self.run_render_gaussians()

    def run_backward(self):
        # Compute loss and backward pass with retain_graph=True to allow multiple calls
        loss = F.l1_loss(self.colors, self.pixels)
        loss.backward(retain_graph=True)


def create_benchmark_params():
    """Create benchmark parameters from YAML configuration."""
    config = load_benchmark_config()
    params = []

    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        dataset_path = dataset_config["path"]

        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Dataset path: {dataset_path}")

        # Use checkpoint paths if available, otherwise use default pattern
        if "checkpoint_paths" in dataset_config and dataset_config["checkpoint_paths"]:
            logger.info(f"Checkpoint paths: {dataset_config['checkpoint_paths']}")
            checkpoint_paths = dataset_config["checkpoint_paths"]
        else:
            # Fallback to default pattern if no checkpoint paths are specified
            checkpoint_paths = [
                f"results/benchmark/{dataset_name}/run_*/checkpoints/ckpt_00400.pt",
                f"results/benchmark/{dataset_name}/run_*/checkpoints/ckpt_04000.pt",
                f"results/benchmark/{dataset_name}/run_*/checkpoints/ckpt_20000.pt",
            ]

        for checkpoint_path in checkpoint_paths:
            params.append((dataset_path, checkpoint_path))

    return params


@pytest.fixture(
    scope="module",
    params=create_benchmark_params(),
    ids=lambda param: f"{param[0].split('/')[-1]}_ckpt_{param[1].split('_')[-1].split('.')[0]}",
)
def benchmark_3dgs(request):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    data_path, checkpoint_path = request.param
    return Benchmark3dgs(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
    )


@pytest.mark.benchmark(
    group="3dgs",
    warmup=True,
    warmup_iterations=3,
)
def test_project_gaussians(benchmark, benchmark_3dgs):
    benchmark(benchmark_3dgs.run_project_gaussians)


@pytest.mark.benchmark(
    group="3dgs",
    warmup=True,
    warmup_iterations=3,
)
def test_render_gaussians(benchmark, benchmark_3dgs):
    benchmark(benchmark_3dgs.run_render_gaussians)


@pytest.mark.benchmark(
    group="3dgs",
    warmup=True,
    warmup_iterations=3,
)
def test_forward(benchmark, benchmark_3dgs):
    benchmark(benchmark_3dgs.run_forward)


@pytest.mark.benchmark(
    group="3dgs",
    warmup=True,
    warmup_iterations=3,
)
def test_backward(benchmark, benchmark_3dgs):
    benchmark(benchmark_3dgs.run_backward)
