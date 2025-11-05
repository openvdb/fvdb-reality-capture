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

from fvdb_reality_capture.radiance_fields import (
    GaussianSplatReconstruction,
    GaussianSplatReconstructionConfig,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
    SfmDataset,
)
from fvdb_reality_capture.sfm_scene import SfmScene

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
        self.results_path = pathlib.Path(checkpoint_path).parent.parent.parent if results_path is None else results_path
        run_name = pathlib.Path(checkpoint_path).parent.parent.parent.name

        # Load the checkpoint
        checkpoint_state = torch.load(pathlib.Path(checkpoint_path), map_location=device, weights_only=False)

        writer_config = GaussianSplatReconstructionWriterConfig(
            save_images=False,
            save_metrics=False,
            save_plys=False,
            save_checkpoints=False,
            use_tensorboard=False,
        )

        print(f"Checkpoint path: {checkpoint_path}")
        print(f"Results path: {self.results_path}")
        print(f"Run name: {run_name}")

        writer = GaussianSplatReconstructionWriter(
            run_name=run_name, save_path=pathlib.Path(self.results_path), config=writer_config, exist_ok=True
        )

        # print checkpoint keys
        print(f"Checkpoint step: {checkpoint_state['step']}")

        self.runner = GaussianSplatReconstruction.from_state_dict(checkpoint_state, writer=writer, device=device)

        step = checkpoint_state["step"]

        trainloader = torch.utils.data.DataLoader(
            self.runner.training_dataset,
            batch_size=self.runner.config.batch_size,
            shuffle=False,  # for benchmarking always use the same order of the dataset
            num_workers=8,
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
        increase_sh_degree_every_step: int = int(
            self.runner.config.increase_sh_degree_every_epoch * len(self.runner.training_dataset)
        )
        self.sh_degree_to_use = min(step // increase_sh_degree_every_step, self.runner.config.sh_degree)

        # run pipeline once to warm up and enable running the benchmarks in any order (or filtered)
        self.run_project_gaussians()
        self.run_render_gaussians()
        self.run_backward()

    def run_project_gaussians(self):
        self.projected_gaussians = self.runner.model.project_gaussians_for_images(
            self.world_to_cam_mats,
            self.projection_mats,
            self.image_width,
            self.image_height,
            self.runner.config.near_plane,
            self.runner.config.far_plane,
            "perspective",
            self.sh_degree_to_use,
            self.runner.config.min_radius_2d,
            self.runner.config.eps_2d,
            self.runner.config.antialias,
        )

    def run_render_gaussians(self):
        # Render an image from the gaussian splats
        # possibly using a crop of the full image
        self.colors, self.alphas = self.runner.model.render_from_projected_gaussians(
            self.projected_gaussians,
            crop_width=self.image_width,
            crop_height=self.image_height,
            crop_origin_w=0,
            crop_origin_h=0,
            tile_size=self.runner.config.tile_size,
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
        run_path = dataset_config["run_directory"]

        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Dataset path: {dataset_path}")

        # Use checkpoint paths if available, otherwise use default pattern
        if "checkpoint_paths" in dataset_config and dataset_config["checkpoint_paths"]:
            logger.info(f"Checkpoint paths: {dataset_config['checkpoint_paths']}")
            checkpoint_paths = dataset_config["checkpoint_paths"]
        else:
            raise ValueError(f"No checkpoint paths specified for dataset: {dataset_name}")

        for checkpoint_path in checkpoint_paths:
            params.append((dataset_path, run_path, checkpoint_path))

    return params


@pytest.fixture(
    scope="module",
    params=create_benchmark_params(),
    ids=lambda param: f"{param[0].rstrip('/').split('/')[-1]}-{param[2].rstrip('/').split('/')[-2]}",
)
def benchmark_3dgs(request):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    data_path, run_path, checkpoint_path = request.param
    return Benchmark3dgs(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        results_path=run_path,
    )


# We append an ordinal to the benchmark group name so that the report comes out in logical order
# rather than alphabetical order.


@pytest.mark.benchmark(
    group="1: 3dgs:project_gaussians",
    warmup=True,
    warmup_iterations=3,
)
def test_project_gaussians(benchmark, benchmark_3dgs):
    benchmark(benchmark_3dgs.run_project_gaussians)


@pytest.mark.benchmark(
    group="2: 3dgs:render_gaussians",
    warmup=True,
    warmup_iterations=3,
)
def test_render_gaussians(benchmark, benchmark_3dgs):
    benchmark(benchmark_3dgs.run_render_gaussians)


@pytest.mark.benchmark(
    group="3: 3dgs:forward",
    warmup=True,
    warmup_iterations=3,
)
def test_forward(benchmark, benchmark_3dgs):
    benchmark(benchmark_3dgs.run_forward)


@pytest.mark.benchmark(
    group="4: 3dgs:backward",
    warmup=True,
    warmup_iterations=3,
)
def test_backward(benchmark, benchmark_3dgs):
    benchmark(benchmark_3dgs.run_backward)
