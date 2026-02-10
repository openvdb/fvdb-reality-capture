# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
import pathlib
import signal
import subprocess
import sys
from typing import Any

import yaml
from git import InvalidGitRepositoryError, Repo
from git.exc import GitCommandError

active_processes = []


# =============================================================================
# Git Utilities
# =============================================================================


def get_git_info(repo_path: pathlib.Path) -> dict[str, Any]:
    """
    Get git repository information.

    Args:
        repo_path: Path to the git repository.

    Returns:
        Dictionary containing:
            - commit: Full commit SHA
            - short_commit: Short commit SHA (first 7 characters)
            - branch: Current branch name (or None if detached HEAD)
            - dirty: True if there are uncommitted changes
            - path: Path to the repository
    """
    repo_path = pathlib.Path(repo_path).resolve()

    if not repo_path.exists():
        return {
            "commit": None,
            "short_commit": None,
            "branch": None,
            "dirty": None,
            "path": str(repo_path),
            "error": f"Repository path does not exist: {repo_path}",
        }

    try:
        repo = Repo(repo_path)

        # Get full commit SHA
        commit = repo.head.commit.hexsha

        # Get short commit
        short_commit = commit[:7] if commit else None

        # Get current branch (None if detached HEAD)
        branch = None if repo.head.is_detached else repo.active_branch.name

        # Check if working directory is dirty
        dirty = repo.is_dirty()

        return {
            "commit": commit,
            "short_commit": short_commit,
            "branch": branch,
            "dirty": dirty,
            "path": str(repo_path),
        }

    except InvalidGitRepositoryError:
        return {
            "commit": None,
            "short_commit": None,
            "branch": None,
            "dirty": None,
            "path": str(repo_path),
            "error": f"Not a git repository: {repo_path}",
        }
    except Exception as e:
        return {
            "commit": None,
            "short_commit": None,
            "branch": None,
            "dirty": None,
            "path": str(repo_path),
            "error": f"Git error: {e}",
        }


def get_current_commit(repo_path: pathlib.Path) -> str | None:
    """
    Get the current HEAD commit SHA for a repository.

    Args:
        repo_path: Path to the git repository.

    Returns:
        Full commit SHA, or None if unable to determine.
    """
    info = get_git_info(repo_path)
    return info.get("commit")


def checkout_commit(repo_path: pathlib.Path, commit: str) -> bool:
    """
    Checkout a specific commit in the repository.

    Args:
        repo_path: Path to the git repository.
        commit: Commit SHA to checkout.

    Returns:
        True if checkout succeeded, False otherwise.

    Raises:
        RuntimeError: If the working directory has uncommitted changes,
            to prevent data loss.
    """
    repo_path = pathlib.Path(repo_path).resolve()

    try:
        repo = Repo(repo_path)

        # Check for uncommitted changes before checkout - fail to prevent data loss
        if repo.is_dirty():
            raise RuntimeError(
                f"Repository {repo_path} has uncommitted changes. "
                "Please commit or stash your changes before running benchmarks "
                "that require switching commits."
            )

        # First, fetch to ensure we have the commit
        logging.info(f"Fetching latest refs in {repo_path}...")
        try:
            for remote in repo.remotes:
                remote.fetch()
        except GitCommandError:
            # Don't fail if fetch fails (e.g., no network)
            logging.warning(f"Fetch failed for {repo_path}, continuing with local refs")

        # Checkout the commit
        short_commit = commit[:7] if len(commit) >= 7 else commit
        logging.info(f"Checking out commit {short_commit}... in {repo_path}")
        repo.git.checkout(commit)
        return True

    except InvalidGitRepositoryError:
        logging.error(f"Not a git repository: {repo_path}")
        return False
    except GitCommandError as e:
        logging.error(f"Failed to checkout commit {commit} in {repo_path}: {e}")
        return False
    except RuntimeError:
        # Re-raise RuntimeError for dirty repo (don't catch it here)
        raise


def build_fvdb_core(repo_path: pathlib.Path, verbose: bool = True) -> bool:
    """
    Build and install fvdb-core from the given repository path.

    Args:
        repo_path: Path to the fvdb-core repository.
        verbose: Whether to show verbose build output.

    Returns:
        True if build succeeded, False otherwise.
    """
    repo_path = pathlib.Path(repo_path).resolve()
    build_script = repo_path / "build.sh"

    if not build_script.exists():
        logging.error(f"build.sh not found in {repo_path}")
        return False

    try:
        logging.info(f"Building fvdb-core from {repo_path}...")
        cmd = ["./build.sh", "install"]
        if verbose:
            cmd.append("verbose")

        # Set up environment
        env = os.environ.copy()
        # Preserve important CUDA environment variables
        for var in ["TORCH_CUDA_ARCH_LIST", "CUDAARCHS", "CPM_SOURCE_CACHE"]:
            if var in os.environ:
                env[var] = os.environ[var]

        subprocess.run(
            cmd,
            cwd=repo_path,
            env=env,
            check=True,
            capture_output=not verbose,
        )
        logging.info("fvdb-core build completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to build fvdb-core: {e}")
        if e.stdout:
            logging.error(f"Build stdout: {e.stdout.decode('utf-8')}")
        if e.stderr:
            logging.error(f"Build stderr: {e.stderr.decode('utf-8')}")
        return False


def install_python_package(repo_path: pathlib.Path, editable: bool = False) -> bool:
    """
    Install a Python package from the given repository path.

    Args:
        repo_path: Path to the repository containing pyproject.toml or setup.py.
        editable: Whether to install in editable mode (-e).

    Returns:
        True if installation succeeded, False otherwise.
    """
    repo_path = pathlib.Path(repo_path).resolve()

    try:
        logging.info(f"Installing Python package from {repo_path}...")
        cmd = [sys.executable, "-m", "pip", "install"]
        if editable:
            cmd.extend(["-e", str(repo_path)])
        else:
            cmd.append(str(repo_path))

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Package installation from {repo_path} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install package from {repo_path}: {e}")
        if e.stdout:
            logging.error(f"Install stdout: {e.stdout}")
        if e.stderr:
            logging.error(f"Install stderr: {e.stderr}")
        return False


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


setup_signal_handlers()


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
        "psnr_values": [],  # Time-series PSNR values
        "psnr_steps": [],  # Steps corresponding to PSNR values
        "ssim_values": [],  # Time-series SSIM values
        "ssim_steps": [],  # Steps corresponding to SSIM values
        "lpips_values": [],  # Time-series LPIPS values
        "lpips_steps": [],  # Steps corresponding to LPIPS values
        "iteration_rates": [],  # Training throughput (steps/s or it/s)
        "rate_steps": [],  # Steps corresponding to iteration rates
        "gaussian_count_values": [],  # Time-series Gaussian count values
        "gaussian_count_steps": [],  # Steps corresponding to Gaussian counts
    }

    # Extract loss values and their corresponding steps from output
    import re

    # Extract loss and step together from each line to ensure proper alignment
    # Pattern: "loss=0.123| ... | 1234/42000 [" - captures both loss and step from the same line
    loss_step_pattern = r"loss=([0-9.]+).*?\|\s*(\d+)/\d+\s*\["

    loss_step_matches = re.findall(loss_step_pattern, output)
    if loss_step_matches:
        metrics["loss_values"] = [float(loss) for loss, step in loss_step_matches]
        metrics["loss_steps"] = [int(step) for loss, step in loss_step_matches]
    else:
        # Fallback to old method if pattern doesn't match
        loss_pattern = r"loss=([0-9.]+)"
        losses = re.findall(loss_pattern, output)
        metrics["loss_values"] = [float(loss) for loss in losses]

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

    # Extract evaluation metrics (PSNR, SSIM, LPIPS) - both time-series and final values
    # Split output into lines for more precise extraction
    output_lines = output.split("\n")

    # Track current step as we parse lines
    current_step = 0

    for line in output_lines:
        # Check if this line contains loss value (indicates actual training, not data loading)
        has_loss = "loss=" in line

        # Update current step if we find a step indicator in a training line (with loss)
        # Don't update from evaluation progress bars (they show image counts, not training steps)
        step_match = re.search(r"\|\s*(\d+)/\d+\s*\[", line)
        if step_match and has_loss:
            current_step = int(step_match.group(1))

        # Also check for fVDB checkpoint/evaluation step indicators
        # Pattern: "Saving checkpoint at global step 33200"
        checkpoint_match = re.search(r"global step\s+(\d+)", line)
        if checkpoint_match:
            current_step = int(checkpoint_match.group(1))

        # Extract iteration rate from progress bars
        # Pattern: "3.17steps/s" or "1.27it/s"
        # Only capture rates from lines with loss values to exclude data loading progress bars
        rate_match = re.search(r"(\d+\.\d+)(steps/s|it/s)", line)
        if rate_match and step_match and has_loss:
            rate_value = float(rate_match.group(1))
            metrics["iteration_rates"].append(rate_value)
            metrics["rate_steps"].append(current_step)

        # Extract PSNR/SSIM/LPIPS when they appear in evaluation lines
        psnr_match = re.search(r"PSNR:\s*([0-9.]+)", line)
        if psnr_match:
            psnr_value = float(psnr_match.group(1))
            metrics["psnr_values"].append(psnr_value)
            metrics["psnr_steps"].append(current_step)

        ssim_match = re.search(r"SSIM:\s*([0-9.]+)", line)
        if ssim_match:
            ssim_value = float(ssim_match.group(1))
            metrics["ssim_values"].append(ssim_value)
            metrics["ssim_steps"].append(current_step)

        lpips_match = re.search(r"LPIPS:\s*([0-9.]+)", line)
        if lpips_match:
            lpips_value = float(lpips_match.group(1))
            metrics["lpips_values"].append(lpips_value)
            metrics["lpips_steps"].append(current_step)

        # Extract Gaussian count from various formats
        # New FVDB format: "num gaussians=817,140"
        # Old FVDB format: "Num Gaussians: 817,140 (before:"
        # GSplat formats: "Now having 817140 GSs" or "Number of GS: 817140"
        gaussian_patterns = [
            r"num gaussians=([\d,]+)",
            r"Num Gaussians:\s*([\d,]+)\s*\(before:",
            r"Now having\s+([\d,]+)\s+GSs",
            r"Number of GS:\s+([\d,]+)",
        ]
        for pattern in gaussian_patterns:
            gaussian_match = re.search(pattern, line)
            if gaussian_match:
                count_str = gaussian_match.group(1).replace(",", "")
                try:
                    gaussian_count = int(count_str)
                    # Only add if we have a valid step and avoid duplicates
                    if current_step > 0 and (
                        not metrics["gaussian_count_steps"] or metrics["gaussian_count_steps"][-1] != current_step
                    ):
                        metrics["gaussian_count_values"].append(gaussian_count)
                        metrics["gaussian_count_steps"].append(current_step)
                    break  # Stop after first match
                except ValueError as e:
                    logging.warning(f"Failed to parse gaussian_count from '{count_str}' (pattern: {pattern}): {e}")

    # Store final values (last in the time series) for backward compatibility
    if metrics["psnr_values"]:
        metrics["psnr"] = metrics["psnr_values"][-1]
    if metrics["ssim_values"]:
        metrics["ssim"] = metrics["ssim_values"][-1]
    if metrics["lpips_values"]:
        metrics["lpips"] = metrics["lpips_values"][-1]

    # Extract training-only time from FVDB helper logs if available
    training_time_pattern = r"Training completed for .* in ([0-9.]+) seconds"
    import re as _re

    _m = _re.search(training_time_pattern, output)
    if _m:
        try:
            metrics["training_time"] = float(_m.group(1))
        except Exception as e:
            logging.warning(f"Failed to parse training_time from '{_m.group(1)}': {e}")

    # Extract final Gaussian count from time-series data if available, otherwise from full output
    if metrics["gaussian_count_values"]:
        metrics["final_gaussian_count"] = metrics["gaussian_count_values"][-1]
    else:
        # Fallback: extract from full output
        # New FVDB progress format example in pbar: "loss=0.021| sh degree=3| num gaussians=817,140"
        # Old FVDB summary debug format: "Num Gaussians: X (before: Y)"
        # GSplat format: "Now having X GSs"
        gaussian_patterns = [
            r"num gaussians=([\d,]+)",  # new FVDB
            r"Num Gaussians: ([\d,]+) \(before:",  # old FVDB
            r"Now having (\d+) GSs",  # GSplat
            r"Number of GS: (\d+)",  # GSplat
        ]
        for _pat in gaussian_patterns:
            _matches = re.findall(_pat, output)
            if _matches:
                count_str = _matches[-1].replace(",", "")
                try:
                    metrics["final_gaussian_count"] = int(count_str)
                    break
                except Exception as e:
                    logging.warning(f"Failed to parse final_gaussian_count from '{count_str}' (pattern: {_pat}): {e}")

    # Extract peak GPU memory
    # GSplat format: "Step:  99 {'mem': 0.19268178939819336, ...}"
    # FVDB format: "FVDB_PEAK_GPU_MEMORY_GB: 1.234"
    gsplat_mem_pattern = r"'mem':\s*([0-9.]+)"
    fvdb_mem_pattern = r"FVDB_PEAK_GPU_MEMORY_GB:\s*([0-9.]+)"

    gsplat_mem_matches = re.findall(gsplat_mem_pattern, output)
    if gsplat_mem_matches:
        # GSplat reports mem in GB, take the last value (final step)
        metrics["peak_gpu_memory_gb"] = float(gsplat_mem_matches[-1])

    fvdb_mem_matches = re.findall(fvdb_mem_pattern, output)
    if fvdb_mem_matches:
        # FVDB reports peak GPU memory in GB
        metrics["peak_gpu_memory_gb"] = float(fvdb_mem_matches[-1])

    # Calculate final metrics
    if metrics["loss_values"]:
        metrics["final_loss"] = metrics["loss_values"][-1]
        metrics["min_loss"] = min(metrics["loss_values"])

    return metrics


def run_command(
    cmd: list[str],
    cwd: str | None = None,
    env: dict[str, Any] | None = None,
    log_file: str | None = None,
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
                    return (
                        return_code,
                        "",
                        f"Process failed with exit code {return_code}",
                    )

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
