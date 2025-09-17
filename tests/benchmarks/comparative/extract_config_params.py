#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Extract training parameters from benchmark_config.yaml for use in both FVDB and GSplat training.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Set up logger
logger = logging.getLogger("extract_config_params")


def count_dataset_images(data_dir: str) -> int:
    """Count dataset images using fvdb_3dgs SfmScene loader (no external deps)."""
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        logger.info(f"Added repo root to sys.path: {repo_root}")

    # Local import after sys.path update
    from fvdb_3dgs.sfm_scene.sfm_scene import SfmScene  # type: ignore

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    scene = SfmScene.from_colmap(data_path)
    num_images = scene.num_images
    logger.info(f"Found {num_images} images in dataset using fvdb_3dgs SfmScene")
    return num_images


def extract_training_params(config_path: str, scene: str) -> dict:
    """Extract training parameters from benchmark config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Find the dataset config for the specified scene
    dataset_config = None
    available_scenes = []
    for dataset in config.get("datasets", []):
        available_scenes.append(dataset.get("name"))
        if dataset.get("name") == scene:
            dataset_config = dataset
            break

    if not dataset_config:
        logger.error(f"Scene '{scene}' not found in config. Available scenes: {available_scenes}")
        raise ValueError(f"Scene '{scene}' not found in config")

    logger.info(f"Processing scene: {scene}")

    training_config = config.get("training", {}).get("config", {})
    training_params = config.get("training_params", {})

    # Extract key parameters
    params = {
        "image_downsample_factor": training_params.get("image_downsample_factor", 4),
        "max_epochs": training_config.get("max_epochs", 200),
        "refine_start_epoch": training_config.get("refine_start_epoch", 3),
        "refine_stop_epoch": training_config.get("refine_stop_epoch", 100),
        "refine_every_epoch": training_config.get("refine_every_epoch", 0.75),
        "sh_degree": training_config.get("sh_degree", 3),
        "initial_opacity": training_config.get("initial_opacity", 0.1),
        "initial_covariance_scale": training_config.get("initial_covariance_scale", 1.0),
        "batch_size": training_config.get("batch_size", 1),
        "ssim_lambda": training_config.get("ssim_lambda", 0.2),
        "lpips_net": training_config.get("lpips_net", "alex"),
        "near_plane": training_config.get("near_plane", 0.01),
        "far_plane": training_config.get("far_plane", 1e10),
        "antialias": training_config.get("antialias", False),
        "tile_size": training_config.get("tile_size", 16),
        "eps_2d": training_config.get("eps_2d", 0.3),
        "min_radius_2d": training_config.get("min_radius_2d", 0.0),
        "random_bkgd": training_config.get("random_bkgd", False),
        "optimize_camera_poses": training_config.get("optimize_camera_poses", True),
        "pose_opt_lr": training_config.get("pose_opt_lr", 1e-5),
        "pose_opt_reg": training_config.get("pose_opt_reg", 1e-6),
    }

    # Calculate steps based on dataset size
    # For both FVDB and GSplat: use the same validation split logic as FVDB
    # FVDB uses: training_images = total_images - validation_images
    # where validation_images = ceil(total_images / use_every_n_as_val)

    # Use the data path from config, with fallback for different environments
    data_base = config.get("paths", {}).get("data_base", "/workspace/data")
    data_dir = f"{data_base}/360_v2/{scene}"
    total_images = count_dataset_images(data_dir)

    # Get use_every_n_as_val from training_params
    use_every_n_as_val = training_params.get("use_every_n_as_val", 8)

    # Calculate validation images using FVDB's logic
    # Validation images are at indices: 0, use_every_n_as_val, 2*use_every_n_as_val, ...
    validation_images = (total_images + use_every_n_as_val - 1) // use_every_n_as_val

    # Calculate training images (excluding validation images) - same for both frameworks
    training_images = total_images - validation_images

    # Calculate steps: epochs * number of training images (same for both frameworks)
    max_epochs = params["max_epochs"]

    # Both FVDB and GSplat use the same step calculation for fair comparison
    max_steps = max_epochs * training_images
    refine_start_steps = int(params["refine_start_epoch"] * training_images)
    refine_stop_steps = int(params["refine_stop_epoch"] * training_images)
    refine_every_steps = int(params["refine_every_epoch"] * training_images)

    # Add calculated step values
    params.update(
        {
            "total_images": total_images,
            "training_images": training_images,
            "use_every_n_as_val": use_every_n_as_val,
            "max_steps": max_steps,
            "refine_start_steps": refine_start_steps,
            "refine_stop_steps": refine_stop_steps,
            "refine_every_steps": refine_every_steps,
            # For backward compatibility
            "num_images": total_images,
            "fvdb_training_images": training_images,
            "fvdb_max_steps": max_steps,
            "fvdb_refine_start_steps": refine_start_steps,
            "fvdb_refine_stop_steps": refine_stop_steps,
            "fvdb_refine_every_steps": refine_every_steps,
            "gsplat_max_steps": max_steps,
            "gsplat_refine_start_steps": refine_start_steps,
            "gsplat_refine_stop_steps": refine_stop_steps,
            "gsplat_refine_every_steps": refine_every_steps,
        }
    )

    return params


def main():
    parser = argparse.ArgumentParser(description="Extract training parameters from benchmark config")
    parser.add_argument("--config", required=True, help="Path to benchmark_config.yaml")
    parser.add_argument("--scene", required=True, help="Scene name")
    parser.add_argument("--format", choices=["bash", "json"], default="bash", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging to match comparison_benchmark.py format
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        params = extract_training_params(args.config, args.scene)

        if args.format == "bash":
            # Output as bash variable assignments
            for key, value in params.items():
                if isinstance(value, bool):
                    # For GSplat, we need to pass boolean flags differently
                    if value:
                        print(f"{key.upper()}=true")
                    else:
                        print(f"{key.upper()}=false")
                elif isinstance(value, float):
                    print(f"{key.upper()}={value}")
                elif isinstance(value, int):
                    print(f"{key.upper()}={value}")
                else:
                    print(f"{key.upper()}={value}")
        elif args.format == "json":
            # Output as JSON
            import json

            print(json.dumps(params, indent=2))

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
