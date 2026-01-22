# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import sys
import time
from typing import Any

import yaml

from ._common import extract_training_metrics, run_command


def run_fvdb_training(
    scene_name: str,
    run_dir: pathlib.Path,
    matrix_config_path: pathlib.Path,
    opt_config_path: pathlib.Path,
    fvdb_results_base_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    """
    Run fVDB training using the matrix configuration.

    Executes the fVDB training pipeline for a specific scene by invoking
    `generate_benchmark_checkpoints.py` with a filtered configuration containing
    only the specified scene. Captures training metrics and timing information.

    Args:
        scene_name (str): The name of the scene to train on.
        run_dir (pathlib.Path): Directory to save the run results and logs.
        matrix_config_path (pathlib.Path): Path to the matrix configuration YAML file
            (contains datasets, opt_configs, and runs definitions).
        opt_config_path (pathlib.Path): Path to the optimization configuration YAML file
            (contains fVDB training parameters).
        fvdb_results_base_path (pathlib.Path | None): Base directory for fVDB results.
            If None, defaults to run_dir / "fvdb_results".

    Returns:
        dict[str, Any]: Training results containing:
            - "success" (bool): Whether training completed successfully (exit_code == 0)
            - "total_time" (float): Total wall-clock time in seconds
            - "training_time" (float): Pure training time in seconds (excluding setup)
            - "exit_code" (int): Process exit code (0 = success)
            - "metrics" (dict[str, Any]): Extracted training metrics (PSNR, SSIM, etc.)
            - "result_dir" (str): Path to the run results directory
    """
    logging.info(f"Starting FVDB training for scene: {scene_name} with config: {opt_config_path.name}")

    # Create results directory for this run
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create log file for capturing output
    log_file = run_dir / "training.log"

    # Start timing
    start_time = time.time()

    # Create a temporary config file with only the specific scene
    temp_config_path = run_dir / "fvdb_config.yaml"

    # Load the original config
    with open(matrix_config_path, "r") as f:
        run_config = yaml.safe_load(f)

    # Filter to only include the current scene
    run_config["datasets"] = [dataset for dataset in run_config["datasets"] if dataset["name"] == scene_name]

    # Load the optimization config
    with open(opt_config_path, "r") as f:
        opt_config = yaml.safe_load(f)

    run_config["optimization_config"] = opt_config

    # Save the filtered config
    with open(temp_config_path, "w") as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)

    # Run FVDB training using the temporary config
    # Use absolute path for the config file since we're changing working directory
    if fvdb_results_base_path is None:
        fvdb_results_base_path = run_dir / "fvdb_results"
    fvdb_results_base_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tests/benchmarks/generate_benchmark_checkpoints.py",
        "--config",
        str(temp_config_path.absolute()),
        "--results-base-path",
        str(fvdb_results_base_path.absolute()),
    ]

    # Run from fvdb-reality-capture repo root (contains tests/benchmarks/generate_benchmark_checkpoints.py)
    repo_root = None
    for candidate in [
        (pathlib.Path(__file__).resolve().parents[3] if len(pathlib.Path(__file__).resolve().parents) >= 4 else None),
        pathlib.Path("/workspace/openvdb/fvdb-reality-capture"),
        pathlib.Path("/workspace/benchmark").parent,  # if running from /workspace/benchmark
    ]:
        if (
            candidate
            and candidate.exists()
            and (candidate / "tests/benchmarks/generate_benchmark_checkpoints.py").exists()
        ):
            repo_root = candidate
            break
    if repo_root is None:
        raise FileNotFoundError(
            "Could not locate fvdb-reality-capture repo root containing tests/benchmarks/generate_benchmark_checkpoints.py"
        )
    exit_code, stdout, stderr = run_command(cmd, cwd=str(repo_root), log_file=str(log_file))

    # End timing
    end_time = time.time()
    wall_time = end_time - start_time

    # Extract metrics from output
    metrics = extract_training_metrics(stdout, wall_time)

    # Always include both total (wall clock) and training times
    metrics["wall_time"] = wall_time
    training_time = metrics.get("training_time", wall_time)

    return {
        "success": exit_code == 0,
        "total_time": wall_time,  # Total time including dataset loading/rescaling
        "training_time": training_time,  # Pure training time
        "exit_code": exit_code,
        "metrics": metrics,
        "result_dir": str(run_dir),
    }
