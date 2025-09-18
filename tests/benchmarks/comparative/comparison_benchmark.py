#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Comparative Benchmark Script

This script runs training for both FVDB and GSplat, and compares the results.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Colors
fvdb_color = "#76B900"
gsplat_color = "#767676"

# Global variable to track subprocesses created by this benchmark
active_processes = []


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("benchmark.log")],
    )


def load_config(config_path: str) -> Dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_scene_info(scene: str, config: Dict) -> Dict:
    """Extract scene-specific information from config."""
    # Find the scene in datasets
    scene_config = None
    for dataset in config.get("datasets", []):
        if dataset.get("name") == scene:
            scene_config = dataset
            break

    if not scene_config:
        raise ValueError(f"Scene '{scene}' not found in config")

    return {
        "name": scene,
        "path": scene_config["path"],
        "data_factor": config.get("training_params", {}).get("image_downsample_factor", 4),
    }


def get_available_scenes(config: Dict) -> List[str]:
    """Get list of available scenes from config."""
    return [dataset.get("name") for dataset in config.get("datasets", [])]


def run_command(
    cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict] = None, log_file: Optional[str] = None
) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    logging.info(f"Running command: {' '.join(cmd)}")
    if cwd:
        logging.info(f"Working directory: {cwd}")

    # Set up environment
    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    # Add CUDA architecture setting that helped
    process_env["TORCH_CUDA_ARCH_LIST"] = "8.9"
    # Force unbuffered output for Python
    process_env["PYTHONUNBUFFERED"] = "1"

    try:
        if log_file:
            # If log file is specified, use tee to capture output while displaying it
            # This preserves the progress bar while also saving output for metrics
            tee_cmd = ["tee", log_file]
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout to capture all output
                text=True,
                bufsize=0,  # Force unbuffered output
                universal_newlines=True,
                preexec_fn=os.setsid,  # Create a new process group for better signal handling
            )

            # Register the process for cleanup
            process_name = f"{cmd[0]} {' '.join(cmd[1:3])}"  # First few args for identification
            active_processes.append({"process": process, "name": process_name})

            # Use tee to display output in real-time while also capturing it
            tee_process = subprocess.Popen(
                tee_cmd,
                stdin=process.stdout,
                stdout=None,  # Display to terminal
                stderr=subprocess.STDOUT,  # Let stderr go directly to terminal
                text=True,
                bufsize=0,  # Force unbuffered output
                universal_newlines=True,
            )

            # Register the tee process for cleanup
            active_processes.append({"process": tee_process, "name": "tee"})

            # Close the pipe from the main process to tee
            if process.stdout is not None:
                process.stdout.close()

            try:
                # Wait for both processes to complete
                return_code = process.wait()
                tee_process.wait()

                # Clean up process registration
                active_processes[:] = [p for p in active_processes if p["process"] not in [process, tee_process]]

                # Read the log file for metrics
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        captured_output = f.read()
                    return return_code, captured_output, ""
                else:
                    return return_code, "Process completed but no log file found", ""

            except KeyboardInterrupt:
                logging.info("Received interrupt signal, terminating processes...")
                # Use direct process termination
                process.terminate()
                tee_process.terminate()
                try:
                    process.wait(timeout=3)
                    tee_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    tee_process.kill()

                # Clean up process registration
                active_processes[:] = [p for p in active_processes if p["process"] not in [process, tee_process]]
                raise
        else:
            # Fallback to direct execution without capturing output
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=process_env,
                stdout=None,  # Don't capture stdout - let it display directly
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid,  # Create a new process group for better signal handling
            )

            # Register the process for cleanup
            process_name = f"{cmd[0]} {' '.join(cmd[1:3])}"  # First few args for identification
            active_processes.append({"process": process, "name": process_name})

            try:
                # Wait for the process to complete
                return_code = process.wait()

                # Clean up process registration
                active_processes[:] = [p for p in active_processes if p["process"] != process]

                # Since we're not capturing stdout/stderr, we can't get the output
                # But we can check if the process completed successfully
                if return_code == 0:
                    return return_code, "Process completed successfully", ""
                else:
                    return return_code, "", f"Process failed with exit code {return_code}"

            except KeyboardInterrupt:
                logging.info("Received interrupt signal, terminating process...")
                # Use direct process termination
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()

                # Clean up process registration
                active_processes[:] = [p for p in active_processes if p["process"] != process]
                raise

    except subprocess.TimeoutExpired:
        logging.error("Command timed out after 2 hours")
        if "process" in locals():
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        if "tee_process" in locals():
            os.killpg(os.getpgid(tee_process.pid), signal.SIGKILL)
        return -1, "", "Command timed out"
    except Exception as e:
        logging.error(f"Command failed with exception: {e}")
        return -1, "", str(e)


def run_fvdb_training(scene_info: Dict, result_dir: str, config: Dict) -> Dict:
    """Run FVDB training using the simplified approach."""
    logging.info(f"Starting FVDB training for scene: {scene_info['name']}")

    # Create results directory
    fvdb_result_dir = Path(result_dir) / f"{scene_info['name']}_fvdb"
    fvdb_result_dir.mkdir(parents=True, exist_ok=True)

    # Create log file for capturing output
    log_file = fvdb_result_dir / "training.log"

    # Start timing
    start_time = time.time()

    # Create a temporary config file with only the specific scene
    temp_config_path = fvdb_result_dir / "temp_config.yaml"

    # Load the original config
    with open("benchmark_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Filter to only include the current scene
    config["datasets"] = [dataset for dataset in config["datasets"] if dataset["name"] == scene_info["name"]]

    # Save the filtered config
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Run FVDB training using the temporary config
    # Use absolute path for the config file since we're changing working directory
    cmd = [
        sys.executable,
        "tests/benchmarks/generate_benchmark_checkpoints.py",
        "--config",
        str(temp_config_path.absolute()),
    ]

    # Run from fvdb-realitycapture repo root (contains tests/benchmarks/generate_benchmark_checkpoints.py)
    repo_root = None
    for candidate in [
        (Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) >= 4 else None),
        Path("/workspace/fvdb-realitycapture"),
        Path("/workspace/benchmark").parent,  # if running from /workspace/benchmark
    ]:
        if candidate and candidate.exists() and (candidate / "tests/benchmarks/generate_benchmark_checkpoints.py").exists():
            repo_root = candidate
            break
    if repo_root is None:
        raise FileNotFoundError("Could not locate fvdb-realitycapture repo root containing tests/benchmarks/generate_benchmark_checkpoints.py")
    exit_code, stdout, stderr = run_command(cmd, cwd=str(repo_root), log_file=str(log_file))

    # Clean up temporary config
    temp_config_path.unlink(missing_ok=True)

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
        "result_dir": str(fvdb_result_dir),
    }


def run_gsplat_training(scene_info: Dict, result_dir: str, config: Dict) -> Dict:
    """Run GSplat training using the simplified basic benchmark approach."""
    logging.info(f"Starting GSplat training for scene: {scene_info['name']}")

    # Create results directory
    gsplat_result_dir = Path(result_dir) / f"{scene_info['name']}_gsplat"
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
    params = extract_training_params(config_path, scene_info["name"])

    # Extract the computed densification parameters
    max_steps = params["max_steps"]
    refine_start_steps = params["refine_start_steps"]
    refine_stop_steps = params["refine_stop_steps"]
    refine_every_steps = params["refine_every_steps"]

    # Calculate reset_every_steps (convert reset_opacities_every_epoch to steps)
    reset_opacities_every_epoch = 16  # From benchmark_config.yaml
    training_images = params["training_images"]
    reset_every_steps = int(reset_opacities_every_epoch * training_images)

    logging.info(f"GSplat densification parameters for {scene_info['name']}:")
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
        str(scene_info["data_factor"]),
        "--render_traj_path",
        "ellipse",
        "--data_dir",
        f"{config.get('paths', {}).get('data_base', '/workspace/data')}/360_v2/{scene_info['name']}/",
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
    import threading as _threading  # local import to avoid polluting module scope
    import re as _re
    import os as _os

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
    if not Path(gsplat_base).exists():
        logging.error(f"GSplat base not found: {gsplat_base}. Skipping GSplat training for {scene_info['name']}.")
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


def extract_training_metrics(output: str, total_time: float) -> Dict[str, Any]:
    """Extract training metrics from command output."""
    metrics: Dict[str, Any] = {
        "loss_values": [],
        "step_times": [],
        "loss_steps": [],  # Track which steps correspond to loss values
    }

    # Extract loss values and their corresponding steps from output
    import re

    # Extract loss values with their step context
    loss_pattern = r"loss=([0-9.]+)"
    losses = re.findall(loss_pattern, output)
    metrics["loss_values"] = [float(loss) for loss in losses]

    # Extract step numbers from progress indicators that appear with losses
    # Pattern: "| 1234/42000 [" - captures the current step from progress bars
    step_progress_pattern = r"\|\s*(\d+)/\d+\s*\["
    step_matches = re.findall(step_progress_pattern, output)

    # Convert to integers and ensure we have the same number as losses
    if step_matches and len(step_matches) >= len(losses):
        # Take the first len(losses) step numbers to match with losses
        metrics["loss_steps"] = [int(step) for step in step_matches[: len(losses)]]
    else:
        # Fallback: create evenly spaced step numbers
        if losses:
            max_steps = 42000  # Default expected steps
            metrics["loss_steps"] = [int(i * max_steps / len(losses)) for i in range(len(losses))]

    # Extract step information
    # Handle multiple formats from both FVDB and GSplat logs
    step_patterns = [
        r"Step ([\d,]+):",  # "Step 1234:" or "Step 1,234:" (refinement steps)
        r"Step:\s+([\d,]+)",  # "Step: 1234" (GSplat format)
        r"step ([\d,]+)",  # "step 42000" (FVDB final step)
        r"(\d+)/\d+.*\[",  # "41999/42000 [12:51<00:00" (progress indicators)
        r"ckpt_([\d,]+)\.pt",  # "ckpt_42000.pt" (checkpoint filenames)
    ]

    all_steps = []
    for pattern in step_patterns:
        steps = re.findall(pattern, output)
        all_steps.extend(steps)

    if all_steps:
        # Remove commas and convert to int, then find the maximum step
        step_numbers = [int(step.replace(",", "")) for step in all_steps]
        metrics["final_step"] = max(step_numbers)

    # Extract evaluation metrics (PSNR, SSIM, LPIPS)
    psnr_pattern = r"PSNR: ([0-9.]+)"
    ssim_pattern = r"SSIM: ([0-9.]+)"
    lpips_pattern = r"LPIPS: ([0-9.]+)"

    psnr_matches = re.findall(psnr_pattern, output)
    ssim_matches = re.findall(ssim_pattern, output)
    lpips_matches = re.findall(lpips_pattern, output)

    if psnr_matches:
        metrics["psnr"] = float(psnr_matches[-1])  # Use the last (most recent) PSNR value
    if ssim_matches:
        metrics["ssim"] = float(ssim_matches[-1])  # Use the last (most recent) SSIM value
    if lpips_matches:
        metrics["lpips"] = float(lpips_matches[-1])  # Use the last (most recent) LPIPS value

    # Extract training-only time from FVDB helper logs if available
    training_time_pattern = r"Training completed for .* in ([0-9.]+) seconds"
    import re as _re

    _m = _re.search(training_time_pattern, output)
    if _m:
        try:
            metrics["training_time"] = float(_m.group(1))
        except Exception:
            pass

    # Extract final Gaussian count
    # Try GSplat pattern first: "Now having X GSs"
    gaussian_pattern_gsplat = r"Now having (\d+) GSs"
    gaussian_matches = re.findall(gaussian_pattern_gsplat, output)
    if gaussian_matches:
        metrics["final_gaussian_count"] = int(gaussian_matches[-1])  # Use the last (most recent) count
    else:
        # Try FVDB pattern: "Num Gaussians: X (before: Y)"
        gaussian_pattern_fvdb = r"Num Gaussians: ([\d,]+) \(before:"
        gaussian_matches = re.findall(gaussian_pattern_fvdb, output)
        if gaussian_matches:
            # Remove commas from the number
            count_str = gaussian_matches[-1].replace(",", "")
            metrics["final_gaussian_count"] = int(count_str)

    # Calculate final metrics
    if metrics["loss_values"]:
        metrics["final_loss"] = metrics["loss_values"][-1]
        metrics["min_loss"] = min(metrics["loss_values"])

    return metrics


def run_evaluation(scene_info: Dict, result_dir: str, config: Dict) -> Dict:
    """Run evaluation for both frameworks."""
    logging.info(f"Running evaluation for scene: {scene_info['name']}")

    results = {}

    # FVDB evaluation
    fvdb_result_dir = Path(result_dir) / f"{scene_info['name']}_fvdb"
    if fvdb_result_dir.exists():
        logging.info("Running FVDB evaluation")
        # Find checkpoint files
        checkpoint_dirs = list(fvdb_result_dir.glob("run_*/checkpoints"))
        if checkpoint_dirs:
            latest_checkpoint_dir = max(checkpoint_dirs, key=lambda x: x.stat().st_mtime)
            checkpoints = list(latest_checkpoint_dir.glob("ckpt_*.pt"))
            if checkpoints:
                # Sort by checkpoint number
                checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
                latest_checkpoint = checkpoints[-1]

                # Evaluation uses local repo if available; skip if script not present
                bench_script = Path(__file__).resolve().parents[3] / "benchmark_3dgs.py"
                if bench_script.exists():
                    cmd = [
                        sys.executable,
                        str(bench_script),
                        "--data_path",
                        f"data/360_v2/{scene_info['name']}",
                        "--checkpoint_path",
                        str(latest_checkpoint),
                        "--results_path",
                        str(fvdb_result_dir),
                    ]
                    exit_code, stdout, stderr = run_command(cmd)
                else:
                    logging.warning("benchmark_3dgs.py not found locally. Skipping FVDB eval step.")
                    exit_code, stdout, stderr = 0, "", ""
                results["fvdb_eval"] = {
                    "success": exit_code == 0,
                    "checkpoint": str(latest_checkpoint),
                    "output": stdout,
                    "error": stderr,
                }

    # GSplat evaluation
    gsplat_result_dir = Path(result_dir) / f"{scene_info['name']}_gsplat"
    if gsplat_result_dir.exists():
        logging.info("Running GSplat evaluation")
        # Find checkpoint files
        checkpoint_dir = gsplat_result_dir / "ckpts"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)

                cmd = [
                    sys.executable,
                    "simple_trainer.py",
                    "default",
                    "--disable_viewer",
                    "--data_factor",
                    str(scene_info["data_factor"]),
                    "--render_traj_path",
                    "ellipse",
                    "--data_dir",
                    f"../../../../3d_gaussian_splatting/data/360_v2/{scene_info['name']}/",
                    "--result_dir",
                    str(gsplat_result_dir),
                    "--ckpt",
                    str(latest_checkpoint),
                ]

                gsplat_base = config.get("paths", {}).get(
                    "gsplat_base", "../../../../3d_gaussian_splatting/benchmark/gsplat/examples"
                )
                if Path(gsplat_base).exists():
                    exit_code, stdout, stderr = run_command(cmd, cwd=gsplat_base)
                else:
                    logging.warning(f"GSplat base not found: {gsplat_base}. Skipping GSplat eval step.")
                    exit_code, stdout, stderr = 0, "", ""
                results["gsplat_eval"] = {
                    "success": exit_code == 0,
                    "checkpoint": str(latest_checkpoint),
                    "output": stdout,
                    "error": stderr,
                }

    return results


def generate_comparison_report(
    scene: str, fvdb_results: Dict, gsplat_results: Dict, eval_results: Dict, result_dir: str
) -> None:
    """Generate a comparison report."""
    report_file = Path(result_dir) / f"{scene}_comparison_report.json"

    report = {
        "scene": scene,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fvdb_training": fvdb_results,
        "gsplat_training": gsplat_results,
        "evaluation": eval_results,
        "comparison": {
            "fvdb_success": fvdb_results["success"],
            "gsplat_success": gsplat_results["success"],
            "fvdb_total_time": fvdb_results["total_time"],
            "fvdb_training_time": fvdb_results.get("training_time", fvdb_results["total_time"]),
            "gsplat_total_time": gsplat_results["total_time"],
            "gsplat_training_time": gsplat_results.get("training_time", gsplat_results["total_time"]),
            "total_time_ratio": (
                gsplat_results["total_time"] / fvdb_results["total_time"]
                if fvdb_results["total_time"] > 0
                else float("inf")
            ),
            "training_time_ratio": (
                gsplat_results.get("training_time", gsplat_results["total_time"])
                / fvdb_results.get("training_time", fvdb_results["total_time"])
                if fvdb_results.get("training_time", fvdb_results["total_time"]) > 0
                else float("inf")
            ),
            "fvdb_final_loss": fvdb_results["metrics"].get("final_loss", None),
            "gsplat_final_loss": gsplat_results["metrics"].get("final_loss", None),
        },
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    logging.info("=== COMPARISON SUMMARY ===")
    logging.info(f"Scene: {scene}")

    fvdb_total = fvdb_results["total_time"]
    fvdb_training = fvdb_results.get("training_time", fvdb_total)
    gsplat_total = gsplat_results["total_time"]
    gsplat_training = gsplat_results.get("training_time", gsplat_total)

    logging.info(
        f"FVDB Training: {'SUCCESS' if fvdb_results['success'] else 'FAILED'} "
        f"(Total: {fvdb_total:.2f}s, Training: {fvdb_training:.2f}s)"
    )
    logging.info(
        f"GSplat Training: {'SUCCESS' if gsplat_results['success'] else 'FAILED'} "
        f"(Total: {gsplat_total:.2f}s, Training: {gsplat_training:.2f}s)"
    )

    if fvdb_results["success"] and gsplat_results["success"]:
        total_ratio = gsplat_total / fvdb_total
        training_ratio = gsplat_training / fvdb_training
        logging.info(f"Total Time Ratio (GSplat/FVDB): {total_ratio:.2f}x")
        logging.info(f"Training Time Ratio (GSplat/FVDB): {training_ratio:.2f}x")

        if "final_loss" in fvdb_results["metrics"] and "final_loss" in gsplat_results["metrics"]:
            fvdb_loss = fvdb_results["metrics"]["final_loss"]
            gsplat_loss = gsplat_results["metrics"]["final_loss"]
            logging.info(f"FVDB Final Loss: {fvdb_loss:.6f}")
            logging.info(f"GSplat Final Loss: {gsplat_loss:.6f}")

    logging.info(f"Detailed report saved to: {report_file}")


def load_existing_results(result_dir: str, scene: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load existing benchmark results from a directory."""
    result_path = Path(result_dir)

    # Look for FVDB results
    fvdb_result_dir = result_path / f"{scene}_fvdb"
    fvdb_results = None
    if fvdb_result_dir.exists():
        log_file = fvdb_result_dir / "training.log"
        if log_file.exists():
            with open(log_file, "r") as f:
                output = f.read()

            # Extract timing from log file or use file modification time
            import time

            mtime = log_file.stat().st_mtime
            fvdb_results = {
                "success": True,
                "total_time": 0,  # We'll estimate this
                "metrics": extract_training_metrics(output, 0),
                "result_dir": str(fvdb_result_dir),
            }

    # Look for GSplat results
    gsplat_result_dir = result_path / f"{scene}_gsplat"
    gsplat_results = None
    if gsplat_result_dir.exists():
        log_file = gsplat_result_dir / "training.log"
        if log_file.exists():
            with open(log_file, "r") as f:
                output = f.read()

            gsplat_results = {
                "success": True,
                "total_time": 0,  # We'll estimate this
                "metrics": extract_training_metrics(output, 0),
                "result_dir": str(gsplat_result_dir),
            }

    # Also try to load from comparison report if it exists
    comparison_report = result_path / f"{scene}_comparison_report.json"
    if comparison_report.exists():
        try:
            with open(comparison_report, "r") as f:
                report_data = json.load(f)

            # Prioritize comparison report data for plot-only mode (it has corrected metrics)
            if "fvdb_training" in report_data and report_data["fvdb_training"].get("success"):
                # Re-extract metrics with updated patterns to get corrected step counts
                if fvdb_results and "result_dir" in fvdb_results:
                    log_file = Path(fvdb_results["result_dir"]) / "training.log"
                    if log_file.exists():
                        with open(log_file, "r") as f:
                            output = f.read()
                        updated_metrics = extract_training_metrics(output, report_data["fvdb_training"]["total_time"])
                        # Use timing from report but updated metrics from log
                        fvdb_results = {**report_data["fvdb_training"], "metrics": updated_metrics}
                    else:
                        fvdb_results = report_data["fvdb_training"]
                else:
                    fvdb_results = report_data["fvdb_training"]

            if "gsplat_training" in report_data and report_data["gsplat_training"].get("success"):
                # Re-extract metrics with updated patterns to get corrected step counts
                if gsplat_results and "result_dir" in gsplat_results:
                    log_file = Path(gsplat_results["result_dir"]) / "training.log"
                    if log_file.exists():
                        with open(log_file, "r") as f:
                            output = f.read()
                        updated_metrics = extract_training_metrics(output, report_data["gsplat_training"]["total_time"])
                        # Use timing from report but updated metrics from log
                        gsplat_results = {**report_data["gsplat_training"], "metrics": updated_metrics}
                    else:
                        gsplat_results = report_data["gsplat_training"]
                else:
                    gsplat_results = report_data["gsplat_training"]
        except Exception as e:
            logging.warning(f"Could not load comparison report: {e}")

    return fvdb_results, gsplat_results


def extract_gaussian_count_from_logs(result_dir: str, scene: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract final Gaussian count from existing training logs."""
    fvdb_result_dir = Path(result_dir) / f"{scene}_fvdb"
    gsplat_result_dir = Path(result_dir) / f"{scene}_gsplat"

    fvdb_count = None
    gsplat_count = None

    # Extract from FVDB log
    if fvdb_result_dir.exists():
        log_file = fvdb_result_dir / "training.log"
        if log_file.exists():
            with open(log_file, "r") as f:
                output = f.read()
            import re

            # Try GSplat pattern first: "Now having X GSs"
            gaussian_pattern_gsplat = r"Now having (\d+) GSs"
            gaussian_matches = re.findall(gaussian_pattern_gsplat, output)
            if gaussian_matches:
                fvdb_count = int(gaussian_matches[-1])
            else:
                # Try FVDB pattern: "Num Gaussians: X (before: Y)"
                gaussian_pattern_fvdb = r"Num Gaussians: ([\d,]+) \(before:"
                gaussian_matches = re.findall(gaussian_pattern_fvdb, output)
                if gaussian_matches:
                    # Remove commas from the number
                    count_str = gaussian_matches[-1].replace(",", "")
                    fvdb_count = int(count_str)

    # Extract from GSplat log
    if gsplat_result_dir.exists():
        log_file = gsplat_result_dir / "training.log"
        if log_file.exists():
            with open(log_file, "r") as f:
                output = f.read()
            import re

            # Try GSplat pattern first: "Now having X GSs"
            gaussian_pattern_gsplat = r"Now having (\d+) GSs"
            gaussian_matches = re.findall(gaussian_pattern_gsplat, output)
            if gaussian_matches:
                gsplat_count = int(gaussian_matches[-1])
            else:
                # Try FVDB pattern: "Num Gaussians: X (before: Y)"
                gaussian_pattern_fvdb = r"Num Gaussians: ([\d,]+) \(before:"
                gaussian_matches = re.findall(gaussian_pattern_fvdb, output)
                if gaussian_matches:
                    # Remove commas from the number
                    count_str = gaussian_matches[-1].replace(",", "")
                    gsplat_count = int(count_str)

    return fvdb_count, gsplat_count


def generate_comparative_plots(fvdb_results: Dict, gsplat_results: Dict, result_dir: str, scene_name: str = "") -> None:
    """Generate comprehensive comparative plots and analysis."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Create plots directory
    plots_dir = Path(result_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    fvdb_losses = fvdb_results.get("metrics", {}).get("loss_values", [])
    gsplat_losses = gsplat_results.get("metrics", {}).get("loss_values", [])
    fvdb_steps = fvdb_results.get("metrics", {}).get("loss_steps", [])
    gsplat_steps = gsplat_results.get("metrics", {}).get("loss_steps", [])
    fvdb_time = fvdb_results.get("total_time", 0)
    gsplat_time = gsplat_results.get("total_time", 0)

    # Fallback to indices if step mapping is not available
    if not fvdb_steps:
        fvdb_steps = list(range(1, len(fvdb_losses) + 1))
    if not gsplat_steps:
        gsplat_steps = list(range(1, len(gsplat_losses) + 1))

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Comparative Analysis: FVDB vs GSplat", fontsize=16, fontweight="bold")

    # Plot 1: Loss vs Steps (using actual step numbers)
    ax1.plot(fvdb_steps, fvdb_losses, color=fvdb_color, label="FVDB", linewidth=3, alpha=0.9)
    ax1.plot(gsplat_steps, gsplat_losses, color=gsplat_color, label="GSplat", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs Training Steps")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss vs Time
    fvdb_times = np.linspace(0, fvdb_time, len(fvdb_losses))
    gsplat_times = np.linspace(0, gsplat_time, len(gsplat_losses))
    ax2.plot(fvdb_times, fvdb_losses, color=fvdb_color, label="FVDB", linewidth=3, alpha=0.9)
    ax2.plot(gsplat_times, gsplat_losses, color=gsplat_color, label="GSplat", linewidth=3, alpha=0.7)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss vs Training Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Training Time Comparison
    frameworks = ["FVDB", "GSplat"]
    times = [fvdb_time, gsplat_time]
    colors = [fvdb_color, gsplat_color]
    bars = ax3.bar(frameworks, times, color=colors, alpha=0.7)
    ax3.set_ylabel("Training Time (seconds)")
    ax3.set_title("Training Time Comparison")
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"{time:.1f}s", ha="center", va="bottom")

    # Plot 4: Gaussian Count Comparison
    fvdb_gaussians = fvdb_results.get("metrics", {}).get("final_gaussian_count", 0)
    gsplat_gaussians = gsplat_results.get("metrics", {}).get("final_gaussian_count", 0)
    gaussian_values = [fvdb_gaussians, gsplat_gaussians]

    bars = ax4.bar(frameworks, gaussian_values, color=colors, alpha=0.7)
    ax4.set_ylabel("Final Gaussian Count")
    ax4.set_title("Final Gaussian Count Comparison")
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, count in zip(bars, gaussian_values):
        height = bar.get_height()
        if count > 1000000:
            label = f"{count/1000000:.1f}M"
        elif count > 1000:
            label = f"{count/1000:.1f}K"
        else:
            label = f"{count}"
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height + height * 0.01, label, ha="center", va="bottom")

    plt.tight_layout()
    # Include scene name in filename if provided
    filename = f"comparative_analysis_{scene_name}.png" if scene_name else "comparative_analysis.png"
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()

    # Generate summary statistics
    summary_stats = {
        "training_time": {
            "fvdb": fvdb_time,
            "gsplat": gsplat_time,
            "ratio": gsplat_time / fvdb_time if fvdb_time > 0 else float("inf"),
        },
        "loss_metrics": {
            "fvdb_final": fvdb_results.get("metrics", {}).get("final_loss", 0),
            "gsplat_final": gsplat_results.get("metrics", {}).get("final_loss", 0),
            "fvdb_min": min(fvdb_losses) if fvdb_losses else 0,
            "gsplat_min": min(gsplat_losses) if gsplat_losses else 0,
        },
        "evaluation_metrics": {
            "fvdb_psnr": fvdb_results.get("metrics", {}).get("psnr", None),
            "gsplat_psnr": gsplat_results.get("metrics", {}).get("psnr", None),
            "fvdb_ssim": fvdb_results.get("metrics", {}).get("ssim", None),
            "gsplat_ssim": gsplat_results.get("metrics", {}).get("ssim", None),
            "fvdb_lpips": fvdb_results.get("metrics", {}).get("lpips", None),
            "gsplat_lpips": gsplat_results.get("metrics", {}).get("lpips", None),
        },
        "gaussian_count": {
            "fvdb": fvdb_results.get("metrics", {}).get("final_gaussian_count", None),
            "gsplat": gsplat_results.get("metrics", {}).get("final_gaussian_count", None),
            "ratio": (
                gsplat_results.get("metrics", {}).get("final_gaussian_count", 0)
                / fvdb_results.get("metrics", {}).get("final_gaussian_count", 1)
                if fvdb_results.get("metrics", {}).get("final_gaussian_count", 0) > 0
                else float("inf")
            ),
        },
        "convergence": {
            "fvdb_steps": len(fvdb_losses),
            "gsplat_steps": len(gsplat_losses),
            "fvdb_steps_per_second": len(fvdb_losses) / fvdb_time if fvdb_time > 0 else 0,
            "gsplat_steps_per_second": len(gsplat_losses) / gsplat_time if gsplat_time > 0 else 0,
        },
    }

    # Save summary statistics
    with open(plots_dir / "summary_statistics.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Training Time:")
    print(f"  FVDB:   {fvdb_time:.2f}s")
    print(f"  GSplat: {gsplat_time:.2f}s")
    print(f"  Ratio:  {summary_stats['training_time']['ratio']:.2f}x (GSplat/FVDB)")
    print()
    print(f"Final Loss:")
    fvdb_final = fvdb_results.get("metrics", {}).get("final_loss", 0)
    gsplat_final = gsplat_results.get("metrics", {}).get("final_loss", 0)
    print(f"  FVDB:   {fvdb_final:.4f}")
    print(f"  GSplat: {gsplat_final:.4f}")
    print(f"  Diff:   {abs(fvdb_final - gsplat_final):.4f}")
    print()
    print(f"Training Efficiency:")
    print(f"  FVDB:   {summary_stats['convergence']['fvdb_steps_per_second']:.1f} steps/sec")
    print(f"  GSplat: {summary_stats['convergence']['gsplat_steps_per_second']:.1f} steps/sec")
    print()
    print(f"Loss Quality:")
    print(
        f"  FVDB - Final: {summary_stats['loss_metrics']['fvdb_final']:.4f}, Min: {summary_stats['loss_metrics']['fvdb_min']:.4f}"
    )
    print(
        f"  GSplat - Final: {summary_stats['loss_metrics']['gsplat_final']:.4f}, Min: {summary_stats['loss_metrics']['gsplat_min']:.4f}"
    )
    print()
    print(f"Final Gaussian Count:")
    fvdb_gaussians = summary_stats["gaussian_count"]["fvdb"]
    gsplat_gaussians = summary_stats["gaussian_count"]["gsplat"]
    if fvdb_gaussians is not None:
        print(f"  FVDB:   {fvdb_gaussians:,}")
    if gsplat_gaussians is not None:
        print(f"  GSplat:  {gsplat_gaussians:,}")
    if fvdb_gaussians is not None and gsplat_gaussians is not None:
        ratio = summary_stats["gaussian_count"]["ratio"]
        print(f"  Ratio:   {ratio:.2f}x (GSplat/FVDB)")
    print()

    # Print evaluation metrics if available
    fvdb_psnr = summary_stats["evaluation_metrics"]["fvdb_psnr"]
    gsplat_psnr = summary_stats["evaluation_metrics"]["gsplat_psnr"]
    fvdb_ssim = summary_stats["evaluation_metrics"]["fvdb_ssim"]
    gsplat_ssim = summary_stats["evaluation_metrics"]["gsplat_ssim"]

    if fvdb_psnr is not None or gsplat_psnr is not None:
        print(f"Image Quality Metrics:")
        if fvdb_psnr is not None:
            print(
                f"  FVDB - PSNR: {fvdb_psnr:.2f}dB, SSIM: {fvdb_ssim:.4f}"
                if fvdb_ssim
                else f"  FVDB - PSNR: {fvdb_psnr:.2f}dB"
            )
        if gsplat_psnr is not None:
            print(
                f"  GSplat - PSNR: {gsplat_psnr:.2f}dB, SSIM: {gsplat_ssim:.4f}"
                if gsplat_ssim
                else f"  GSplat - PSNR: {gsplat_psnr:.2f}dB"
            )

    print("=" * 60)


def generate_enhanced_comparative_report(scenes: List[str], result_dir: str) -> None:
    """Generate enhanced comparative report with scene names, checkpoint links, and detailed statistics."""
    import pandas as pd

    # Create summary directory
    summary_dir = Path(result_dir) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Collect detailed data from all scenes
    detailed_data = []

    for scene in scenes:
        # Load comparison report for this scene
        report_file = Path(result_dir) / f"{scene}_comparison_report.json"
        if not report_file.exists():
            logging.warning(f"No comparison report found for {scene}, skipping...")
            continue

        try:
            with open(report_file, "r") as f:
                report = json.load(f)

            # Extract data
            fvdb_time = report.get("comparison", {}).get("fvdb_time", 0)
            gsplat_time = report.get("comparison", {}).get("gsplat_time", 0)
            fvdb_psnr = report.get("fvdb_training", {}).get("metrics", {}).get("psnr", 0)
            gsplat_psnr = report.get("gsplat_training", {}).get("metrics", {}).get("psnr", 0)
            fvdb_gaussians = report.get("fvdb_training", {}).get("metrics", {}).get("final_gaussian_count", 0)
            gsplat_gaussians = report.get("gsplat_training", {}).get("metrics", {}).get("final_gaussian_count", 0)

            # If Gaussian counts are missing from JSON, extract from logs
            if fvdb_gaussians == 0 or gsplat_gaussians == 0:
                fvdb_count, gsplat_count = extract_gaussian_count_from_logs(result_dir, scene)
                if fvdb_count is not None:
                    fvdb_gaussians = fvdb_count
                if gsplat_count is not None:
                    gsplat_gaussians = gsplat_count

            # Find checkpoint paths
            fvdb_checkpoint = None
            gsplat_checkpoint = None

            fvdb_result_dir = Path(result_dir) / f"{scene}_fvdb"
            if fvdb_result_dir.exists():
                checkpoint_dirs = list(fvdb_result_dir.glob("run_*/checkpoints"))
                if checkpoint_dirs:
                    latest_checkpoint_dir = max(checkpoint_dirs, key=lambda x: x.stat().st_mtime)
                    checkpoints = list(latest_checkpoint_dir.glob("ckpt_*.pt"))
                    if checkpoints:
                        checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
                        fvdb_checkpoint = str(checkpoints[-1])

            gsplat_result_dir = Path(result_dir) / f"{scene}_gsplat"
            if gsplat_result_dir.exists():
                checkpoint_dir = gsplat_result_dir / "ckpts"
                if checkpoint_dir.exists():
                    checkpoints = list(checkpoint_dir.glob("*"))
                    if checkpoints:
                        gsplat_checkpoint = str(max(checkpoints, key=lambda x: x.stat().st_mtime))

            detailed_data.append(
                {
                    "scene": scene,
                    "fvdb_time": fvdb_time,
                    "gsplat_time": gsplat_time,
                    "fvdb_psnr": fvdb_psnr,
                    "gsplat_psnr": gsplat_psnr,
                    "fvdb_gaussians": fvdb_gaussians,
                    "gsplat_gaussians": gsplat_gaussians,
                    "speedup": gsplat_time / fvdb_time if fvdb_time > 0 else float("inf"),
                    "psnr_diff": fvdb_psnr - gsplat_psnr if fvdb_psnr and gsplat_psnr else 0,
                    "gaussian_ratio": gsplat_gaussians / fvdb_gaussians if fvdb_gaussians > 0 else float("inf"),
                    "fvdb_checkpoint": fvdb_checkpoint,
                    "gsplat_checkpoint": gsplat_checkpoint,
                    "fvdb_result_dir": str(fvdb_result_dir),
                    "gsplat_result_dir": str(gsplat_result_dir),
                }
            )

        except Exception as e:
            logging.warning(f"Could not load report for {scene}: {e}")
            continue

    if not detailed_data:
        logging.warning("No valid data found for enhanced report")
        return

    # Create DataFrame
    df = pd.DataFrame(detailed_data)

    # Save detailed data
    df.to_csv(summary_dir / "detailed_comparative_data.csv", index=False)
    with open(summary_dir / "detailed_comparative_data.json", "w") as f:
        json.dump(detailed_data, f, indent=2)

    # Generate detailed report
    report_content = []
    report_content.append("# Enhanced Comparative Analysis Report")
    report_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("")

    # Summary statistics
    report_content.append("## Summary Statistics Across All Scenes")
    report_content.append("")
    report_content.append(f"**Average Training Time:**")
    report_content.append(f"- FVDB: {df['fvdb_time'].mean():.1f}s ± {df['fvdb_time'].std():.1f}s")
    report_content.append(f"- GSplat: {df['gsplat_time'].mean():.1f}s ± {df['gsplat_time'].std():.1f}s")
    report_content.append(f"- Average Speedup: {df['speedup'].mean():.2f}x")
    report_content.append("")

    report_content.append(f"**Average PSNR:**")
    report_content.append(f"- FVDB: {df['fvdb_psnr'].mean():.2f}dB ± {df['fvdb_psnr'].std():.2f}dB")
    report_content.append(f"- GSplat: {df['gsplat_psnr'].mean():.2f}dB ± {df['gsplat_psnr'].std():.2f}dB")
    report_content.append(f"- Average PSNR Difference: {df['psnr_diff'].mean():+.2f}dB")
    report_content.append("")

    report_content.append(f"**Average Final Gaussian Count:**")
    report_content.append(f"- FVDB: {df['fvdb_gaussians'].mean():,.0f} ± {df['fvdb_gaussians'].std():,.0f}")
    report_content.append(f"- GSplat: {df['gsplat_gaussians'].mean():,.0f} ± {df['gsplat_gaussians'].std():,.0f}")
    report_content.append(f"- Average Gaussian Ratio: {df['gaussian_ratio'].mean():.2f}x")
    report_content.append("")

    # Per-scene details
    report_content.append("## Per-Scene Analysis")
    report_content.append("")

    for _, row in df.iterrows():
        report_content.append(f"### {row['scene']}")
        report_content.append("")
        report_content.append(f"**Training Time:**")
        report_content.append(f"- FVDB: {row['fvdb_time']:.1f}s")
        report_content.append(f"- GSplat: {row['gsplat_time']:.1f}s")
        report_content.append(f"- Speedup: {row['speedup']:.2f}x")
        report_content.append("")

        report_content.append(f"**PSNR:**")
        report_content.append(f"- FVDB: {row['fvdb_psnr']:.2f}dB")
        report_content.append(f"- GSplat: {row['gsplat_psnr']:.2f}dB")
        report_content.append(f"- Difference: {row['psnr_diff']:+.2f}dB")
        report_content.append("")

        report_content.append(f"**Final Gaussian Count:**")
        report_content.append(f"- FVDB: {row['fvdb_gaussians']:,}")
        report_content.append(f"- GSplat: {row['gsplat_gaussians']:,}")
        report_content.append(f"- Ratio: {row['gaussian_ratio']:.2f}x")
        report_content.append("")

        report_content.append(f"**Checkpoints:**")
        if row["fvdb_checkpoint"]:
            report_content.append(f"- FVDB: [{row['fvdb_checkpoint']}]({row['fvdb_checkpoint']})")
        else:
            report_content.append(f"- FVDB: Not found")
        if row["gsplat_checkpoint"]:
            report_content.append(f"- GSplat: [{row['gsplat_checkpoint']}]({row['gsplat_checkpoint']})")
        else:
            report_content.append(f"- GSplat: Not found")
        report_content.append("")

        report_content.append(f"**Result Directories:**")
        report_content.append(f"- FVDB: [{row['fvdb_result_dir']}]({row['fvdb_result_dir']})")
        report_content.append(f"- GSplat: [{row['gsplat_result_dir']}]({row['gsplat_result_dir']})")
        report_content.append("")
        report_content.append("---")
        report_content.append("")

    # Save report
    with open(summary_dir / "enhanced_comparative_report.md", "w") as f:
        f.write("\n".join(report_content))

    print(f"\nEnhanced comparative report generated:")
    print(f"  Markdown: {summary_dir / 'enhanced_comparative_report.md'}")
    print(f"  CSV: {summary_dir / 'detailed_comparative_data.csv'}")
    print(f"  JSON: {summary_dir / 'detailed_comparative_data.json'}")


def generate_summary_charts(scenes: List[str], result_dir: str) -> None:
    """Generate summary charts comparing frameworks across multiple scenes."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Create summary directory
    summary_dir = Path(result_dir) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Collect data from all scenes
    summary_data = []

    for scene in scenes:
        # Load comparison report for this scene
        report_file = Path(result_dir) / f"{scene}_comparison_report.json"
        if not report_file.exists():
            logging.warning(f"No comparison report found for {scene}, skipping...")
            continue

        try:
            with open(report_file, "r") as f:
                report = json.load(f)

            # Extract data
            fvdb_time = report.get("comparison", {}).get("fvdb_time", 0)
            gsplat_time = report.get("comparison", {}).get("gsplat_time", 0)
            fvdb_psnr = report.get("fvdb_training", {}).get("metrics", {}).get("psnr", 0)
            gsplat_psnr = report.get("gsplat_training", {}).get("metrics", {}).get("psnr", 0)
            fvdb_gaussians = report.get("fvdb_training", {}).get("metrics", {}).get("final_gaussian_count", 0)
            gsplat_gaussians = report.get("gsplat_training", {}).get("metrics", {}).get("final_gaussian_count", 0)

            # If Gaussian counts are missing from JSON, extract from logs
            if fvdb_gaussians == 0 or gsplat_gaussians == 0:
                fvdb_count, gsplat_count = extract_gaussian_count_from_logs(result_dir, scene)
                if fvdb_count is not None:
                    fvdb_gaussians = fvdb_count
                if gsplat_count is not None:
                    gsplat_gaussians = gsplat_count

            summary_data.append(
                {
                    "scene": scene,
                    "fvdb_time": fvdb_time,
                    "gsplat_time": gsplat_time,
                    "fvdb_psnr": fvdb_psnr,
                    "gsplat_psnr": gsplat_psnr,
                    "fvdb_gaussians": fvdb_gaussians,
                    "gsplat_gaussians": gsplat_gaussians,
                    "speedup": gsplat_time / fvdb_time if fvdb_time > 0 else float("inf"),
                    "psnr_diff": fvdb_psnr - gsplat_psnr if fvdb_psnr and gsplat_psnr else 0,
                    "gaussian_ratio": gsplat_gaussians / fvdb_gaussians if fvdb_gaussians > 0 else float("inf"),
                }
            )

        except Exception as e:
            logging.warning(f"Could not load report for {scene}: {e}")
            continue

    if not summary_data:
        logging.warning("No valid data found for summary charts")
        return

    # Create DataFrame for easy manipulation
    df = pd.DataFrame(summary_data)

    # Save data to CSV and JSON
    df.to_csv(summary_dir / "summary_data.csv", index=False)
    with open(summary_dir / "summary_data.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Create side-by-side bar plots with more height for labels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    # Plot 1: Runtime Comparison
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, df["fvdb_time"], width, label="FVDB", color=fvdb_color)
    bars2 = ax1.bar(x + width / 2, df["gsplat_time"], width, label="GSplat", color=gsplat_color)

    # Add speedup labels
    max_time = 0
    for i, (fvdb_time, gsplat_time, speedup) in enumerate(zip(df["fvdb_time"], df["gsplat_time"], df["speedup"])):
        if speedup != float("inf"):
            label_y = max(fvdb_time, gsplat_time) + 50
            ax1.text(i, label_y, f"{speedup:.1f}x", ha="center", va="bottom", fontweight="bold")
            max_time = max(max_time, label_y)

    ax1.set_xlabel("Scene")
    ax1.set_ylabel("Training Time (seconds)")
    ax1.set_title("3D Gaussian Splatting Training Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["scene"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Adjust y-axis limits to accommodate labels
    if max_time > 0:
        ax1.set_ylim(0, max_time + 100)

    # Plot 2: PSNR Comparison
    bars3 = ax2.bar(x - width / 2, df["fvdb_psnr"], width, label="FVDB", color=fvdb_color)
    bars4 = ax2.bar(x + width / 2, df["gsplat_psnr"], width, label="GSplat", color=gsplat_color)

    # Add PSNR difference labels
    max_psnr = 0
    for i, (fvdb_psnr, gsplat_psnr, psnr_diff) in enumerate(zip(df["fvdb_psnr"], df["gsplat_psnr"], df["psnr_diff"])):
        if fvdb_psnr and gsplat_psnr:
            label_y = max(fvdb_psnr, gsplat_psnr) + 0.5
            ax2.text(i, label_y, f"{psnr_diff:+.1f}dB", ha="center", va="bottom", fontweight="bold")
            max_psnr = max(max_psnr, label_y)

    ax2.set_xlabel("Scene")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Peak Signal-to-Noise Ratio (PSNR)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["scene"], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Adjust y-axis limits to accommodate labels
    if max_psnr > 0:
        ax2.set_ylim(0, max_psnr + 2.0)

    # Plot 3: Gaussian Count Comparison
    bars5 = ax3.bar(x - width / 2, df["fvdb_gaussians"], width, label="FVDB", color=fvdb_color)
    bars6 = ax3.bar(x + width / 2, df["gsplat_gaussians"], width, label="GSplat", color=gsplat_color)

    # Add Gaussian ratio labels
    max_gaussians = 0
    for i, (fvdb_gaussians, gsplat_gaussians, gaussian_ratio) in enumerate(
        zip(df["fvdb_gaussians"], df["gsplat_gaussians"], df["gaussian_ratio"])
    ):
        if gaussian_ratio != float("inf"):
            label_y = max(fvdb_gaussians, gsplat_gaussians) + max(fvdb_gaussians, gsplat_gaussians) * 0.01
            ax3.text(
                i,
                label_y,
                f"{gaussian_ratio:.1f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
            max_gaussians = max(max_gaussians, label_y)

    ax3.set_xlabel("Scene")
    ax3.set_ylabel("Final Gaussian Count")
    ax3.set_title("Final Gaussian Splat Count")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df["scene"], rotation=45, ha="right")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Adjust y-axis limits to accommodate labels
    if max_gaussians > 0:
        ax3.set_ylim(0, max_gaussians + max_gaussians * 0.15)

    plt.tight_layout(pad=3.0)
    plt.savefig(summary_dir / "summary_comparison.png", dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS ACROSS ALL SCENES")
    print("=" * 80)

    print(f"Average Training Time:")
    print(f"  FVDB:   {df['fvdb_time'].mean():.1f}s ± {df['fvdb_time'].std():.1f}s")
    print(f"  GSplat: {df['gsplat_time'].mean():.1f}s ± {df['gsplat_time'].std():.1f}s")
    print(f"  Average Speedup: {df['speedup'].mean():.2f}x")

    print(f"\nAverage PSNR:")
    print(f"  FVDB:   {df['fvdb_psnr'].mean():.2f}dB ± {df['fvdb_psnr'].std():.2f}dB")
    print(f"  GSplat: {df['gsplat_psnr'].mean():.2f}dB ± {df['gsplat_psnr'].std():.2f}dB")
    print(f"  Average PSNR Difference: {df['psnr_diff'].mean():+.2f}dB")

    print(f"\nAverage Final Gaussian Count:")
    print(f"  FVDB:   {df['fvdb_gaussians'].mean():,.0f} ± {df['fvdb_gaussians'].std():,.0f}")
    print(f"  GSplat: {df['gsplat_gaussians'].mean():,.0f} ± {df['gsplat_gaussians'].std():,.0f}")
    print(f"  Average Gaussian Ratio: {df['gaussian_ratio'].mean():.2f}x")

    print(f"\nData exported to:")
    print(f"  CSV: {summary_dir / 'summary_data.csv'}")
    print(f"  JSON: {summary_dir / 'summary_data.json'}")
    print(f"  Plot: {summary_dir / 'summary_comparison.png'}")
    print("=" * 80)


def main():
    """
    fVDB Comparative Benchmark script.

    This script allows benchmarking and comparison of fVDB 3D Gaussian Splatting to GSplat on one or more scenes.
    It supports running training, evaluation, and generating summary plots from existing results.

    Scene Selection:
        - If --scenes is provided: Use only the specified scenes
        - If --scenes is not provided: Use all scenes defined in the config file
        - Use --list-scenes to see available scenes in the config

    Command-line Arguments:
        --config        Path to the benchmark configuration YAML file (required unless --plot-only).
        --scenes        Comma-separated list of scene names to benchmark (optional, defaults to all scenes in config).
        --result-dir    Directory to store results (default: results/benchmark).
        --train-only    Only run training (skip evaluation and plotting).
        --eval-only     Only run evaluation (skip training and plotting).
        --plot-only     Only generate plots from existing results (skip training and evaluation).
        --frameworks    Comma-separated list of frameworks to run (default: fvdb,gsplat).
        --log-level     Logging level (default: INFO).
        --list-scenes   List available scenes from config and exit.

    The script sets up signal handling for graceful interruption, parses arguments,
    loads configuration, and processes each scene as specified.

    Example usage:
        # Run all scenes from config
        python comparison_benchmark.py --config config.yaml

        # Run specific scenes
        python comparison_benchmark.py --config config.yaml --scenes garden,bicycle

        # List available scenes
        python comparison_benchmark.py --config config.yaml --list-scenes

        # Generate plots from existing results
        python comparison_benchmark.py --scenes garden,bicycle --plot-only

    Returns:
        None
    """
    # Set up signal handling for graceful interruption
    import signal

    def signal_handler(signum, frame):
        logging.info("Received interrupt signal, shutting down immediately...")

        # Force kill all tracked processes immediately
        for process_info in active_processes:
            try:
                if process_info["process"].poll() is None:  # Process is still running
                    logging.info(f"Force killing benchmark process: {process_info['name']}")
                    process_info["process"].kill()
            except Exception as e:
                logging.warning(f"Error killing process {process_info['name']}: {e}")

        # Exit immediately without waiting
        logging.info("Exiting immediately...")
        os._exit(1)  # Use os._exit to bypass cleanup handlers

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Simplified Comparative Benchmark")
    parser.add_argument(
        "--config", default="benchmark_config.yaml", help="Path to benchmark config YAML (required unless --plot-only)"
    )
    parser.add_argument(
        "--scenes", help="Comma-separated list of scene names to benchmark (default: all scenes from config)"
    )
    parser.add_argument("--result-dir", default="results/benchmark", help="Results directory")
    parser.add_argument("--train-only", action="store_true", help="Only run training")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing results")
    parser.add_argument("--frameworks", default="fvdb,gsplat", help="Comma-separated list of frameworks to run")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--list-scenes", action="store_true", help="List available scenes from config and exit")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load config (only needed if not plot-only)
    if not args.plot_only:
        if not args.config:
            parser.error("--config is required unless --plot-only is specified")
        config = load_config(args.config)

        # Handle --list-scenes option
        if args.list_scenes:
            available_scenes = get_available_scenes(config)
            print("Available scenes in config:")
            for scene in available_scenes:
                print(f"  - {scene}")
            sys.exit(0)
    else:
        config = None

    # Parse scenes
    if args.scenes:
        # Use scenes from command line
        scenes = [s.strip() for s in args.scenes.split(",")]
    elif not args.plot_only and config:
        # Use all scenes from config
        scenes = get_available_scenes(config)
        if not scenes:
            parser.error("No scenes found in config file")
        logging.info(f"Using all scenes from config: {', '.join(scenes)}")
    else:
        # For plot-only mode without config, scenes must be specified
        parser.error("--scenes is required for --plot-only mode when no config is provided")

    # Parse frameworks
    frameworks = [f.strip() for f in args.frameworks.split(",")]

    # Create results directory
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Process each scene
    for scene_name in scenes:
        logging.info(f"Processing scene: {scene_name}")

        if not args.plot_only:
            if config is None:
                parser.error("Config is required for training/evaluation")
            scene_info = get_scene_info(scene_name, config)
        else:
            scene_info = {"name": scene_name}

        fvdb_results = None
        gsplat_results = None
        eval_results = {}

        # Load existing results if plot-only mode
        if args.plot_only:
            logging.info(f"Loading existing results from: {result_dir}")
            fvdb_results, gsplat_results = load_existing_results(str(result_dir), scene_name)

            if fvdb_results and gsplat_results:
                logging.info("Found both FVDB and GSplat results, generating plots...")
                generate_comparative_plots(fvdb_results, gsplat_results, str(result_dir), scene_name)
            elif fvdb_results:
                logging.warning("Only FVDB results found, cannot generate comparative plots")
            elif gsplat_results:
                logging.warning("Only GSplat results found, cannot generate comparative plots")
            else:
                logging.error("No existing results found for plotting")

            continue

        # Run training
        if not args.eval_only:
            if "fvdb" in frameworks and config is not None:
                fvdb_results = run_fvdb_training(scene_info, str(result_dir), config)

            if "gsplat" in frameworks and config is not None:
                gsplat_results = run_gsplat_training(scene_info, str(result_dir), config)

        # Run evaluation
        if not args.train_only and config is not None:
            eval_results = run_evaluation(scene_info, str(result_dir), config)

        # Generate comparison report
        if fvdb_results or gsplat_results:
            generate_comparison_report(
                scene_name,
                fvdb_results or {"success": False, "total_time": 0, "metrics": {}},
                gsplat_results or {"success": False, "total_time": 0, "metrics": {}},
                eval_results,
                str(result_dir),
            )

        # Generate comparative plots
        if fvdb_results and gsplat_results:
            generate_comparative_plots(fvdb_results, gsplat_results, str(result_dir), scene_name)

        logging.info(f"Completed benchmark for {scene_name}")

    # Generate summary charts if multiple scenes were processed
    if len(scenes) > 1:
        generate_summary_charts(scenes, str(result_dir))
        generate_enhanced_comparative_report(scenes, str(result_dir))

    logging.info("All benchmarks completed!")


if __name__ == "__main__":
    main()
