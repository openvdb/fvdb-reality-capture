# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

"""
Generate Gaussian splatting checkpoints for the benchmark.
"""

import logging
import pathlib
import subprocess
import sys
import time
from typing import Dict, List

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from fvdb_reality_capture.training import Config, SceneOptimizationRunner

logger = logging.getLogger("train benchmark checkpoints")


def load_config(config_path: str = "benchmark_config.yaml") -> Dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict, config_path: str = "benchmark_config.yaml") -> None:
    """Save benchmark configuration to YAML file."""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def find_all_checkpoint_files(checkpoints_dir: pathlib.Path) -> List[str]:
    """Find all checkpoint files in the directory."""
    checkpoint_paths = []

    # Expects filenames of the form ckpt_<string>.pt. Typically the string is a number or "final".
    for file in checkpoints_dir.iterdir():
        if file.is_file() and file.name.startswith("ckpt_") and file.name.endswith(".pt"):
            checkpoint_paths.append(str(file))

    # Separate numeric and non-numeric checkpoints, sort numerics by number, then sort and append
    # non-numerics at the end
    numeric_ckpts = []
    non_numeric_ckpts = []
    for x in checkpoint_paths:
        try:
            # Try to extract the numeric part after the last underscore and before .pt
            num = int(x.split("_")[-1].split(".")[0])
            numeric_ckpts.append((num, x))
        except (ValueError, IndexError):
            non_numeric_ckpts.append(x)
    # Sort numeric checkpoints by their extracted number
    numeric_ckpts.sort(key=lambda pair: pair[0])
    non_numeric_ckpts.sort()
    # Rebuild the list: numerics first (in order), then non-numerics
    checkpoint_paths = [x for _, x in numeric_ckpts] + non_numeric_ckpts
    return checkpoint_paths


def get_run_name(results_path: pathlib.Path) -> str:
    """Get a unique run name for the current git commit hash."""
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    # Generate a run name based on the commit hash, appending an index if the directory exists
    base_run_name = f"run_{commit_hash}"
    run_name = base_run_name
    idx = 1
    while (results_path / run_name).exists():
        run_name = f"{base_run_name}_{idx:02d}"
        idx += 1
    return run_name


def main(
    run_name: str | None = None,
    results_base_path: pathlib.Path = pathlib.Path("results"),
    config_path: str = "benchmark_config.yaml",
):

    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    # Load configuration
    config = load_config(config_path)

    # Get the current git commit hash of the repository
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    logger.info(f"Current git commit hash: {commit_hash}")

    # Extract configuration
    # save_at_percent = config["training"]["save_at_percent"]
    datasets = config["datasets"]
    training_params = config["optimization_config"]["training_arguments"]

    # Create base Config object
    base_config = Config()

    # Override config with values from YAML
    for key, value in config["optimization_config"]["optimization_config"].items():
        if hasattr(base_config, key):
            # Handle boolean conversion explicitly
            if isinstance(value, str) and value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            setattr(base_config, key, value)

    # Set save percentages
    # base_config.save_at_percent = save_at_percent

    # Create the results directory if it doesn't exist
    results_base_path.mkdir(parents=True, exist_ok=True)

    # Track generated checkpoint paths
    generated_checkpoints = {}

    for dataset_config in datasets:
        dataset_name = dataset_config["name"]
        dataset_path = pathlib.Path(dataset_config["path"])

        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Dataset path: {dataset_path}")

        # Create the dataset directory if it doesn't exist
        dataset_results_path = results_base_path / dataset_name
        dataset_results_path.mkdir(parents=True, exist_ok=True)

        train = run_name is None

        if train:
            run_name = get_run_name(dataset_results_path)

            # Create the runner (this sets up datasets/transforms/cache) without including it in training time
            logger.info(f"Preparing training run for {dataset_name} (initializing datasets/transforms/cache)...")
            runner = SceneOptimizationRunner.new_run(
                config=base_config,
                dataset_path=dataset_path,
                run_name=run_name,
                image_downsample_factor=training_params["image_downsample_factor"],
                points_percentile_filter=training_params["points_percentile_filter"],
                normalization_type=training_params["normalization_type"],
                crop_bbox=training_params["crop_bbox"],
                # crop_to_points=training_params["crop_to_points"],
                min_points_per_image=training_params["min_points_per_image"],
                results_path=dataset_results_path,
                device=training_params["device"],
                use_every_n_as_val=training_params["use_every_n_as_val"],
                disable_viewer=training_params["disable_viewer"],
                log_tensorboard_every=training_params["log_tensorboard_every"],
                log_images_to_tensorboard=training_params["log_images_to_tensorboard"],
                save_results=training_params["save_results"],
                save_eval_images=training_params["save_eval_images"],
            )

            # Start training-only timer
            logger.info(f"Starting training for {dataset_name}...")
            start_time = time.time()
            runner.train()
            training_time = time.time() - start_time
            logger.info(f"Training completed for {dataset_name} in {training_time:.2f} seconds")
        else:
            logger.info(f"Skipping training for {dataset_name}")

        # Find the generated run directory (timestamped)
        run_dir = dataset_results_path / run_name

        # Find checkpoint files
        checkpoints_dir = run_dir / "checkpoints"

        if checkpoints_dir.exists():
            checkpoint_paths = find_all_checkpoint_files(checkpoints_dir)
            generated_checkpoints[dataset_name] = {
                "run_directory": str(run_dir),
                "checkpoint_paths": checkpoint_paths,
            }

            logger.info(f"Found {len(checkpoint_paths)} checkpoint files for {dataset_name}")
            for path in checkpoint_paths:
                logger.info(f"  - {path}")
        else:
            logger.warning(f"Checkpoints directory not found: {checkpoints_dir}")
            # Still create the entry even if no checkpoints found
            generated_checkpoints[dataset_name] = {
                "run_directory": str(run_dir),
                "checkpoint_paths": [],
            }

    # Update the configuration with generated checkpoint paths
    for dataset_config in datasets:
        dataset_name = dataset_config["name"]
        if dataset_name in generated_checkpoints:
            dataset_config["run_directory"] = generated_checkpoints[dataset_name]["run_directory"]
            dataset_config["checkpoint_paths"] = generated_checkpoints[dataset_name]["checkpoint_paths"]

    # Save updated configuration
    save_config(config, config_path)
    logger.info(f"Updated configuration saved to benchmark_config.yaml")
    logger.info("Checkpoint paths have been added to the configuration file.")


if __name__ == "__main__":
    import argparse

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Generate Gaussian splatting checkpoints for the benchmark.")
        parser.add_argument(
            "--config",
            default="benchmark_config.yaml",
            help="Path to the benchmark configuration file (default: benchmark_config.yaml)",
        )
        parser.add_argument(
            "--find-checkpoints-run-name",
            help=(
                "Skip training and look for checkpoints in the specified run directory name, "
                "and populate benchmark_config.yaml with the paths to the checkpoints."
            ),
        )

        args = parser.parse_args()
        main(run_name=args.find_checkpoints_run_name, config_path=args.config)
