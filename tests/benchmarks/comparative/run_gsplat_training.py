# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import sys
import time
from typing import Any

from common import extract_training_metrics, run_command


def run_gsplat_training(scene_name: str, result_dir: pathlib.Path, config: dict[str, Any]) -> dict[str, Any]:
    """Run GSplat training using the simplified basic benchmark approach."""
    logging.info(f"Starting GSplat training for scene: {scene_name}")

    # Create results directory
    gsplat_result_dir = result_dir / f"{scene_name}_gsplat"
    gsplat_result_dir.mkdir(parents=True, exist_ok=True)

    # Create log file for capturing output
    log_file = gsplat_result_dir / "training.log"

    # Start timing
    start_time = time.time()

    # Calculate densification parameters to match FVDB
    # Import the extraction logic to compute parameters dynamically
    from extract_config_params import extract_training_params

    # Load the config and extract parameters for this scene
    config_path = "benchmark_config.yaml"
    params = extract_training_params(config_path, scene_name)

    # Extract the computed densification parameters
    max_steps = params["max_steps"]
    refine_start_steps = params["refine_start_steps"]
    refine_stop_steps = params["refine_stop_steps"]
    refine_every_steps = params["refine_every_steps"]

    # Calculate reset_every_steps (convert reset_opacities_every_epoch to steps)
    reset_opacities_every_epoch = 16  # From benchmark_config.yaml
    training_images = params["training_images"]
    reset_every_steps = int(reset_opacities_every_epoch * training_images)

    logging.info(f"GSplat densification parameters for {scene_name}:")
    logging.info(f"  max_steps: {max_steps}")
    logging.info(f"  refine_start_steps: {refine_start_steps}")
    logging.info(f"  refine_stop_steps: {refine_stop_steps}")
    logging.info(f"  refine_every_steps: {refine_every_steps}")
    logging.info(f"  reset_every_steps: {reset_every_steps}")
    logging.info(f"  Training images: {training_images}")
    logging.info(f"  Total images: {params.get('total_images', 'N/A')}")

    # Build GSplat command with computed parameters
    cmd = [
        sys.executable,
        "simple_trainer.py",
        "default",
        "--eval_steps",
        str(max_steps),  # Evaluate at final step
        "--disable_viewer",
        "--disable_video",  # Disable video generation to avoid rendering errors
        "--data_factor",
        str(scene_info["data_factor"]),  # TODO: Load this from config
        "--render_traj_path",
        "ellipse",
        "--data_dir",
        f"{config.get('paths', {}).get('data_base', '/workspace/data')}/360_v2/{scene_name}/",
        "--result_dir",
        str(gsplat_result_dir),
        "--max_steps",
        str(max_steps),  # Full training
        # Add densification parameters to match FVDB using tyro nested syntax
        "--strategy.refine_start_iter",
        str(refine_start_steps),
        "--strategy.refine_stop_iter",
        str(refine_stop_steps),
        "--strategy.refine_every",
        str(refine_every_steps),
        "--strategy.reset_every",
        str(reset_every_steps),
        "--strategy.pause_refine_after_reset",
        "0",  # Don't pause refinement after reset
        "--strategy.verbose",  # Enable verbose output to see refinement info
        "--global_scale",
        "0.909",  # Compensate for GSplat's 1.1x scene scale multiplier to match FVDB
        "--strategy.refine_scale2d_stop_iter",
        "1",  # Disable 2D scale-based splitting to match FVDB behavior
    ]

    logging.info(f"GSplat command: {' '.join(cmd)}")

    # Start a background watcher to detect the first training step in the log
    import os as _os
    import re as _re
    import threading as _threading  # local import to avoid polluting module scope

    first_step_time: dict = {"t": None}
    stop_event = _threading.Event()

    def _watch_training_start(log_path: str, pattern: str, started_flag: dict, stop_evt: _threading.Event):
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

    gsplat_base = config.get("paths", {}).get(
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
