# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import json
import pathlib

import cv2
import numpy as np
import point_cloud_utils as pcu
import pytest
import torch

from fvdb_reality_capture.radiance_fields import (
    GaussianSplatOptimizerConfig,
    GaussianSplatOptimizerMCMCConfig,
    GaussianSplatReconstruction,
    GaussianSplatReconstructionConfig,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
    SpatialScaleMode,
)
from fvdb_reality_capture.sfm_scene import SfmScene

from . import contract


def _minimal_checkpoint_state() -> dict:
    recon_cfg = GaussianSplatReconstructionConfig()
    opt_cfg = GaussianSplatOptimizerConfig()
    return {
        "magic": "GaussianSplattingCheckpoint",
        "version": GaussianSplatReconstruction.version,
        "step": 0,
        "config": vars(recon_cfg),
        "sfm_scene": {},
        "model": {},
        "optimizer": {"config": vars(opt_cfg)},
        "train_indices": [],
        "val_indices": [],
        "num_training_poses": None,
        "pose_adjust_model": None,
        "pose_adjust_optimizer": None,
        "pose_adjust_scheduler": None,
    }


def test_contract_matches_dataclasses():
    contract._assert_contract_matches_dataclasses()


def test_checkpoint_contract_accepts_minimal_state():
    state = _minimal_checkpoint_state()
    contract.validate_checkpoint_contract(state)


def test_checkpoint_contract_rejects_unknown_config_key():
    state = _minimal_checkpoint_state()
    state["config"]["opacity_reg"] = 0.0
    with pytest.raises(ValueError, match="Checkpoint config key mismatch"):
        contract.validate_checkpoint_contract(state)


def test_checkpoint_contract_accepts_mcmc_optimizer_config():
    state = _minimal_checkpoint_state()
    mcmc_cfg = GaussianSplatOptimizerMCMCConfig()
    state["optimizer"]["config"] = vars(mcmc_cfg)
    contract.validate_checkpoint_contract(state)


def test_benchmark_config_yaml_matches_contract():
    config_path = pathlib.Path(__file__).parent / "benchmark_config.yaml"
    config = contract.load_benchmark_yaml(str(config_path))
    contract.validate_benchmark_yaml(config, require_run_paths=True)


def test_benchmark_yaml_rejects_unknown_key():
    config_path = pathlib.Path(__file__).parent / "benchmark_config.yaml"
    config = contract.load_benchmark_yaml(str(config_path))
    config["optimization_config"]["reconstruction_config"]["opacity_reg"] = 0.0
    with pytest.raises(ValueError, match="Unknown reconstruction_config keys"):
        contract.validate_benchmark_yaml(config, require_run_paths=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmark smoke test.")
def test_benchmark_smoke_pipeline_on_gpu(tmp_path: pathlib.Path):
    """
    Minimal GPU smoke test for the benchmark harness pipeline:
    - create tiny SfmScene from a simple directory
    - build reconstruction, run project/render/backward
    - verify state_dict round-trip works
    """
    data_dir = tmp_path / "simple_scene"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)

    # Create tiny image
    image = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    image_path = images_dir / "000.png"
    cv2.imwrite(str(image_path), image)

    # Create cameras.json
    fx = fy = 4.0
    cx = cy = 3.5
    cameras = [
        {
            "camera_name": "cam0",
            "width": 8,
            "height": 8,
            "camera_intrinsics": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
            "world_to_camera": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "image_path": "000.png",
        }
    ]
    cameras_path = data_dir / "cameras.json"
    cameras_path.write_text(json.dumps(cameras))

    # Create pointcloud.ply
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pcu.save_mesh_vc(str(data_dir / "pointcloud.ply"), points, colors)

    # Build scene
    sfm_scene = SfmScene.from_simple_directory(data_dir)

    writer_cfg = GaussianSplatReconstructionWriterConfig(
        save_images=False,
        save_metrics=False,
        save_plys=False,
        save_checkpoints=False,
        use_tensorboard=False,
    )
    writer = GaussianSplatReconstructionWriter(run_name="smoke", save_path=tmp_path, config=writer_cfg, exist_ok=True)

    config = GaussianSplatReconstructionConfig(batch_size=1, max_epochs=1)
    optimizer_config = GaussianSplatOptimizerConfig(spatial_scale_mode=SpatialScaleMode.ABSOLUTE_UNITS)
    runner = GaussianSplatReconstruction.from_sfm_scene(
        sfm_scene=sfm_scene,
        writer=writer,
        config=config,
        optimizer_config=optimizer_config,
        device="cuda",
    )

    # Pull a single sample directly (avoid dataloader multiprocessing in tests)
    sample = runner.training_dataset[0]
    world_to_camera = sample["world_to_camera"].unsqueeze(0).cuda()
    projection = sample["projection"].unsqueeze(0).cuda()
    image = torch.from_numpy(sample["image"]).unsqueeze(0).cuda() / 255.0
    height, width = image.shape[1:3]

    projected = runner.model.project_gaussians_for_images(
        world_to_camera,
        projection,
        width,
        height,
        runner.config.near_plane,
        runner.config.far_plane,
        "perspective",
        runner.config.sh_degree,
        runner.config.min_radius_2d,
        runner.config.eps_2d,
        runner.config.antialias,
    )
    colors, _alphas = runner.model.render_from_projected_gaussians(
        projected,
        crop_width=width,
        crop_height=height,
        crop_origin_w=0,
        crop_origin_h=0,
        tile_size=runner.config.tile_size,
    )
    loss = torch.nn.functional.l1_loss(colors, image)
    loss.backward()

    # Round-trip checkpoint load
    state = runner.state_dict()
    contract.validate_checkpoint_contract(state)
    checkpoint_path = tmp_path / "smoke_ckpt.pt"
    torch.save(state, checkpoint_path)
    loaded_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract.validate_checkpoint_contract(loaded_state)
    reloaded = GaussianSplatReconstruction.from_state_dict(
        state,
        override_sfm_scene=sfm_scene,
        writer=writer,
        device="cuda",
    )
    assert reloaded.model.num_gaussians == runner.model.num_gaussians
