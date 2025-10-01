# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
import pathlib
import signal
import subprocess
from typing import Any

import yaml

active_processes = []


def load_config(config_path: str | pathlib.Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_training_metrics(output: str, total_time: float) -> dict[str, Any]:
    """Extract training metrics from command output."""
    metrics: dict[str, Any] = {
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

    # Also capture steps from FVDB tqdm description lines like "... 41999/42000 [..] loss=..| ..."
    # We already parse steps from "(\d+)/\d+" above; keep as is.

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
    # New FVDB progress format example in pbar: "loss=0.021| sh degree=3| num gaussians=817,140"
    # Old FVDB summary debug format: "Num Gaussians: X (before: Y)"
    # GSplat format: "Now having X GSs"
    gaussian_patterns = [
        r"num gaussians=([\d,]+)",  # new FVDB
        r"Num Gaussians: ([\d,]+) \(before:",  # old FVDB
        r"Now having (\d+) GSs",  # GSplat
    ]
    for _pat in gaussian_patterns:
        _matches = re.findall(_pat, output)
        if _matches:
            count_str = _matches[-1].replace(",", "")
            try:
                metrics["final_gaussian_count"] = int(count_str)
                break
            except Exception:
                pass

    # Calculate final metrics
    if metrics["loss_values"]:
        metrics["final_loss"] = metrics["loss_values"][-1]
        metrics["min_loss"] = min(metrics["loss_values"])

    return metrics


def setup_signal_handlers():
    """
    Set up signal handlers to ensure all subprocesses are killed on interrupt.
    This ensures that if the main script is interrupted (e.g., via Ctrl+C),
    all child processes are also terminated to prevent orphaned processes.
    """

    def signal_handler(signum, frame):
        global active_processes
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


def run_command(
    cmd: list[str], cwd: str | None = None, env: dict[str, Any] | None = None, log_file: str | None = None
) -> tuple[int, str, str]:
    """
    Run a command in a subprocess and return exit code, stdout, and stderr.

    Args:
        cmd (list[str]): Command and arguments to run.
        cwd (str | None): Working directory to run the command in.
        env (dict[str, Any] | None): Environment variables to set for the command.
        log_file (str | None): If specified, path to a log file to capture output.

    Returns:
        exit_code (int): Exit code of the command.
        stdout (str): Captured standard output.
        stderr (str): Captured standard error.
    """
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
