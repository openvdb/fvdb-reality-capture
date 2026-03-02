# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import sys
import time
from typing import Any

import yaml

import fvdb_reality_capture as frc

from ._common import extract_training_metrics, run_command

# Maps FVDB/benchmark config names -> GSplat CLI flags
# Parameters in YAML config that have a mapping here will be passed to GSplat's simple_trainer.py
# Parameters without a mapping are NOT passed and will use GSplat's defaults
GSPLAT_PARAM_MAPPING: dict[str, str] = {
    # Initialization parameters
    "initial_opacity": "--init_opa",
    "initial_covariance_scale": "--init_scale",
    # Regularization (FVDB naming -> GSplat CLI flags)
    "opacity_regularization": "--opacity_reg",
    "scale_regularization": "--scale_reg",
    # Rendering parameters
    "near_plane": "--near_plane",
    "far_plane": "--far_plane",
    "antialias": "--antialiased",  # Note: different naming convention
    "random_bkgd": "--random_bkgd",
    "ssim_lambda": "--ssim_lambda",
    "sh_degree": "--sh_degree",
    # Training parameters
    "batch_size": "--batch_size",
    # Camera pose optimization
    "optimize_camera_poses": "--pose_opt",  # Note: different naming convention
    "pose_opt_lr": "--pose_opt_lr",
    "pose_opt_reg": "--pose_opt_reg",
    # Learning rates
    "means_lr": "--means_lr",
    "scales_lr": "--scales_lr",
    "opacities_lr": "--opacities_lr",
    "quats_lr": "--quats_lr",
    "sh0_lr": "--sh0_lr",
    "shN_lr": "--shN_lr",
}


def build_gsplat_cli_args(opt_config: dict[str, Any]) -> list[str]:
    """
    Build CLI arguments from opt_config using the parameter mapping.

    Extracts training config values from the optimization config and converts them
    to GSplat CLI flags using GSPLAT_PARAM_MAPPING. Boolean values are handled
    specially - only True values result in the flag being added.

    Args:
        opt_config: The optimization configuration dictionary loaded from YAML.
            Expected structure: {"training": {"config": {...parameters...}}}

    Returns:
        List of CLI argument strings ready to be appended to the command.
    """
    args: list[str] = []
    training_config = opt_config.get("training", {}).get("config", {})

    for config_key, cli_flag in GSPLAT_PARAM_MAPPING.items():
        if config_key in training_config:
            value = training_config[config_key]
            # Handle boolean flags - only add if True.
            # Note: All current boolean params (antialiased, random_bkgd, pose_opt) default
            # to False in GSplat, so not passing them when False achieves the desired behavior.
            if isinstance(value, bool):
                if value:
                    args.append(cli_flag)
            else:
                args.extend([cli_flag, str(value)])

    return args


def run_gsplat_training(
    scene_name: str,
    run_dir: pathlib.Path,
    matrix_config_path: pathlib.Path,
    opt_config_path: pathlib.Path,
    extra_cli_args: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run GSplat training using the matrix configuration.

    Executes the GSplat training pipeline for a specific scene by invoking
    `simple_trainer.py` with computed densification parameters matched to fVDB.
    Handles image downsampling via symlinks and extracts training metrics including
    rendering performance and final quality scores.

    Args:
        scene_name (str): The name of the scene to train on.
        run_dir (pathlib.Path): Directory to save the run results and logs.
        matrix_config_path (pathlib.Path): Path to the matrix configuration YAML file
            (contains datasets, opt_configs, and runs definitions).
        opt_config_path (pathlib.Path): Path to the optimization configuration YAML file
            (contains GSplat training parameters and mode selection).
        extra_cli_args (list[str] | None): Additional command-line arguments to pass
            to GSplat's simple_trainer.py. If None, only default arguments are used.

    Returns:
        dict[str, Any]: Training results containing:
            - "success" (bool): Whether training completed successfully (exit_code == 0)
            - "total_time" (float): Total wall-clock time in seconds
            - "training_time" (float): Pure training time in seconds (excluding setup),
              measured from first "Step N" log entry to completion
            - "exit_code" (int): Process exit code (0 = success)
            - "metrics" (dict[str, Any]): Extracted training metrics (PSNR, SSIM, etc.)
            - "result_dir" (str): Path to the run results directory
    """
    logging.info(f"Starting GSplat training for scene: {scene_name}")

    # Start timing
    start_time = time.time()

    # Load the benchmark config
    with open(matrix_config_path, "r") as f:
        run_config = yaml.safe_load(f)

    # Filter to only include the current scene
    run_config["datasets"] = [dataset for dataset in run_config["datasets"] if dataset["name"] == scene_name]
    scene_path = run_config["datasets"][0]["path"]

    # Load the optimization config
    with open(opt_config_path, "r") as f:
        opt_config = yaml.safe_load(f)

    # Create results directory
    config_name = opt_config["name"]
    gsplat_result_dir = run_dir / f"{scene_name}_{config_name}"
    gsplat_result_dir.mkdir(parents=True, exist_ok=True)

    # Create log file for capturing output
    log_file = gsplat_result_dir / "training.log"

    # Create a temporary config file with only the specific scene
    temp_config_path = gsplat_result_dir / "temp_config.yaml"

    run_config["optimization_config"] = opt_config

    # Calculate densification parameters to match FVDB
    # Import the extraction logic to compute parameters dynamically
    from extract_config_params import extract_training_params

    # Extract parameters for this scene
    params = extract_training_params(run_config, scene_name)

    # Extract the computed densification parameters
    max_steps = params["max_steps"]
    refine_start_steps = params["refine_start_steps"]
    refine_stop_steps = params["refine_stop_steps"]
    refine_every_steps = params["refine_every_steps"]
    sh_degree_interval_steps = params["sh_degree_interval_steps"]
    reset_every_steps = params["reset_every_steps"]
    refine_scale2d_stop_steps = params["refine_scale2d_stop_steps"]

    # Save the filtered config
    with open(temp_config_path, "w") as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)

    logging.info(f"GSplat densification parameters for {scene_name}:")
    logging.info(f"  max_steps: {max_steps}")
    logging.info(f"  refine_start_steps: {refine_start_steps}")
    logging.info(f"  refine_stop_steps: {refine_stop_steps}")
    logging.info(f"  refine_every_steps: {refine_every_steps}")
    logging.info(f"  sh_degree_interval_steps: {sh_degree_interval_steps}")
    logging.info(f"  reset_every_steps: {reset_every_steps}")
    logging.info(f"  refine_scale2d_stop_steps: {refine_scale2d_stop_steps}")
    logging.info(f"  Training images: {params.get('training_images', 'N/A')}")
    logging.info(f"  Total images: {params.get('total_images', 'N/A')}")

    data_base_path = run_config.get("paths", {}).get("data_base", "/workspace/data")
    scene_path = pathlib.Path(data_base_path) / scene_path

    # Create symlinks for both "images_{factor}" and "images_{factor}_png" pointing to "images", if they don't exist
    ds_factor = params.get("image_downsample_factor", 4)
    for suffix in ["", "_png"]:
        rescaled_images_path = scene_path / f"images_{ds_factor}{suffix}"
        if not rescaled_images_path.exists():
            rescaled_images_path.symlink_to(scene_path / "images")
            logging.info(f"Created symlink to {rescaled_images_path} from {scene_path / 'images'}")
        else:
            logging.info(f"Rescaled images path already exists: {rescaled_images_path}")
            logging.info(f"Skipping symlink creation")

    # Build GSplat command with computed parameters
    gsplat_mode = opt_config.get("mode", "default")
    if gsplat_mode not in ("default", "mcmc"):
        raise ValueError(f"Unsupported gsplat mode: {gsplat_mode}")

    # Convert eval_at_percent to eval_steps
    eval_at_percent = opt_config.get("training", {}).get("config", {}).get("eval_at_percent", [100])
    eval_steps = [int(pct * max_steps / 100) for pct in eval_at_percent]
    logging.info(f"  Eval at percent: {eval_at_percent}")
    logging.info(f"  Eval steps: {eval_steps}")

    cmd = [
        sys.executable,
        "simple_trainer.py",
        gsplat_mode,
        "--eval_steps",
    ]
    # Add eval steps as space-separated values
    cmd.extend([str(step) for step in eval_steps])
    cmd.extend(
        [
            "--disable_viewer",
            "--disable_video",  # Disable video generation to avoid rendering errors
            "--data_factor",
            str(ds_factor),
            "--render_traj_path",
            "ellipse",
            "--data_dir",
            str(scene_path),
            "--result_dir",
            str(gsplat_result_dir),
            "--max_steps",
            str(max_steps),  # Full training
            # SH degree schedule to match FVDB
            "--sh_degree_interval",
            str(sh_degree_interval_steps),
            # densification parameters to match FVDB
            "--strategy.refine_start_iter",
            str(refine_start_steps),
            "--strategy.refine_stop_iter",
            str(refine_stop_steps),
            "--strategy.refine_every",
            str(refine_every_steps),
            "--strategy.verbose",
            "--global_scale",
            "1.0",
        ]
    )
    if gsplat_mode == "default":
        cmd.extend(
            [
                "--strategy.reset_every",
                str(reset_every_steps),
                "--strategy.refine_scale2d_stop_iter",
                str(refine_scale2d_stop_steps),
            ]
        )

    # Add parameters from YAML config using the parameter mapping
    mapped_args = build_gsplat_cli_args(opt_config)
    if mapped_args:
        cmd.extend(mapped_args)
        logging.info(f"GSplat mapped parameters from config: {' '.join(mapped_args)}")
    else:
        logging.info("No additional parameters mapped from config")

    if extra_cli_args:
        if not isinstance(extra_cli_args, list) or not all(isinstance(x, str) for x in extra_cli_args):
            raise ValueError("extra_cli_args must be a list[str]")
        cmd.extend(extra_cli_args)

    logging.info(f"GSplat command: {' '.join(cmd)}")

    # Start a background watcher to detect the first training step in the log
    import os as _os
    import re as _re
    import threading as _threading  # local import to avoid polluting module scope

    first_step_time: dict = {"t": None}
    stop_event = _threading.Event()

    def _watch_training_start(log_path: str, pattern: str, started_flag: dict, stop_evt: _threading.Event):
        r"""
        Background thread that monitors the training log for the first training step.

        Watches the training log file for a line matching the given pattern (typically
        the first "Step N" output) and records the timestamp when training actually starts.
        Used to separate setup/initialization time from pure training time.

        Args:
            log_path (str): Path to the training log file to monitor.
            pattern (str): Regex pattern to search for (e.g., ``r"Step\s+\d+"``).

            started_flag (dict): Dictionary with "t" key to store the training start timestamp.
                Set to None initially, updated when pattern is found.
            stop_evt (_threading.Event): Event to signal thread to stop monitoring.
        """
        # Wait until the file exists
        while not stop_evt.is_set() and not _os.path.exists(log_path):
            time.sleep(0.05)
        if stop_evt.is_set():
            return
        try:
            with open(log_path, "r") as f:
                # Read from the beginning to catch early lines
                while not stop_evt.is_set():
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        time.sleep(0.05)
                        f.seek(pos)
                        continue
                    if started_flag["t"] is None and _re.search(pattern, line):
                        started_flag["t"] = time.time()
                        # We can keep running until stop to avoid extra synchronization
        except Exception:
            pass

    watcher = _threading.Thread(
        target=_watch_training_start,
        args=(str(log_file), r"Step\s+\d+", first_step_time, stop_event),
        daemon=True,
    )
    watcher.start()

    gsplat_base = run_config.get("paths", {}).get(
        "gsplat_base", "../../../../3d_gaussian_splatting/benchmark/gsplat/examples"
    )
    if not pathlib.Path(gsplat_base).exists():
        logging.error(f"GSplat base not found: {gsplat_base}. Skipping GSplat training for {scene_name}.")
        return {
            "success": False,
            "total_time": 0.0,
            "training_time": 0.0,
            "exit_code": -1,
            "metrics": {},
            "result_dir": str(gsplat_result_dir),
        }
    exit_code, stdout, stderr = run_command(cmd, cwd=gsplat_base, log_file=str(log_file))
    stop_event.set()
    # Give watcher a brief moment to exit
    try:
        watcher.join(timeout=1.0)
    except Exception:
        pass

    # End timing
    end_time = time.time()
    wall_time = end_time - start_time

    # Extract metrics from output
    metrics = extract_training_metrics(stdout, wall_time)

    # Always include both total (wall clock) and training times
    metrics["wall_time"] = wall_time
    if first_step_time["t"] is not None and first_step_time["t"] >= start_time and first_step_time["t"] <= end_time:
        training_time = end_time - first_step_time["t"]
        metrics["training_time"] = training_time
    else:
        training_time = wall_time  # Fall back to wall time if we can't extract training time

    return {
        "success": exit_code == 0,
        "total_time": wall_time,  # Total time including dataset loading/rescaling
        "training_time": training_time,  # Pure training time
        "exit_code": exit_code,
        "metrics": metrics,
        "result_dir": str(gsplat_result_dir),
    }
