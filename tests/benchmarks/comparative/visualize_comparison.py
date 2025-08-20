#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Side-by-side visualization script for FVDB and GSplat checkpoints.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def find_project_root():
    """Find the project root directory (where the main visualize.py is located)."""
    current_dir = Path.cwd()

    # Look for visualize.py in current directory or parent directories
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "visualize.py").exists():
            return parent

    # Fallback: assume we're in the comparative benchmark directory
    # and go up to the main project directory
    return current_dir.parent.parent.parent


def find_data_directory(project_root: Path, scene: str) -> Path | None:
    """Find the data directory for a given scene."""
    # Common data directory patterns
    data_patterns = [
        project_root / "data" / "360_v2" / scene,
        project_root / "data" / scene,
        project_root / "datasets" / "360_v2" / scene,
        project_root / "datasets" / scene,
    ]

    for pattern in data_patterns:
        if pattern.exists():
            return pattern

    # If not found, return None and let the user specify
    return None


def run_fvdb_viewer(checkpoint_path: str, scene: str, project_root: Path, data_dir: Path | None = None):
    """Run FVDB visualizer."""
    # Build the command
    cmd = [
        "python3",
        str(project_root / "visualize.py"),
        "--checkpoint_path",
        str(checkpoint_path),
        "--device",
        "cuda",
    ]

    # Add dataset path if provided
    if data_dir:
        cmd.extend(["--dataset_path", str(data_dir)])

    print(f"Starting FVDB viewer on port 8080...")
    print(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    process = subprocess.Popen(cmd, env=env)
    return process


def run_gsplat_viewer(checkpoint_path: str, port: int = 8081, project_root: Path | None = None):
    """Run GSplat visualizer."""
    # Find GSplat directory
    if project_root:
        gsplat_dir = project_root / "benchmark" / "gsplat" / "examples"
    else:
        # Fallback: try to find it relative to current directory
        gsplat_dir = Path.cwd().parent.parent.parent / "benchmark" / "gsplat" / "examples"

    if not gsplat_dir.exists():
        print(f"Warning: GSplat directory not found at {gsplat_dir}")
        # Try absolute path as fallback
        gsplat_dir = Path("/home/mharris/github/openvdb/fvdb/projects/3d_gaussian_splatting/benchmark/gsplat/examples")

    cmd = [
        "python3",
        "simple_viewer.py",
        "--ckpt",
        str(checkpoint_path),
        "--output_dir",
        "temp_gsplat_output",
        "--port",
        str(port),
    ]

    print(f"Starting GSplat viewer on port {port}...")
    print(f"Command: cd {gsplat_dir} && {' '.join(cmd)}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    process = subprocess.Popen(cmd, env=env, cwd=gsplat_dir)
    return process


def find_checkpoint_paths(scene: str, project_root: Path):
    """Find checkpoint paths for both FVDB and GSplat."""
    # FVDB checkpoint patterns
    fvdb_patterns = [
        project_root / "results" / "benchmark" / scene / "run_*" / "checkpoints" / "ckpt_final.pt",
        project_root / "results" / scene / "run_*" / "checkpoints" / "ckpt_final.pt",
        project_root / "checkpoints" / scene / "ckpt_final.pt",
    ]

    # GSplat checkpoint patterns
    gsplat_patterns = [
        project_root
        / "benchmark"
        / "gsplat"
        / "examples"
        / "results"
        / "benchmark"
        / f"{scene}_gsplat"
        / "ckpts"
        / "ckpt_*_rank0.pt",
        project_root / "benchmark" / "gsplat" / "results" / f"{scene}_gsplat" / "ckpts" / "ckpt_*_rank0.pt",
    ]

    # Find FVDB checkpoint
    fvdb_checkpoint = None
    for pattern in fvdb_patterns:
        # Handle wildcards in the path
        if "run_*" in str(pattern):
            # For patterns with run_*, search in the scene directory
            scene_dir = pattern.parent.parent.parent
            if scene_dir.exists():
                for run_dir in scene_dir.glob("run_*"):
                    checkpoint_file = run_dir / "checkpoints" / "ckpt_final.pt"
                    if checkpoint_file.exists():
                        if fvdb_checkpoint is None or checkpoint_file.stat().st_mtime > fvdb_checkpoint.stat().st_mtime:
                            fvdb_checkpoint = checkpoint_file
        else:
            # Direct file check
            if pattern.exists():
                if fvdb_checkpoint is None or pattern.stat().st_mtime > fvdb_checkpoint.stat().st_mtime:
                    fvdb_checkpoint = pattern

    # Find GSplat checkpoint
    gsplat_checkpoint = None
    for pattern in gsplat_patterns:
        # Handle wildcards in the path
        if "ckpt_*_rank0.pt" in str(pattern):
            # For patterns with ckpt_*_rank0.pt, search in the ckpts directory
            ckpts_dir = pattern.parent
            if ckpts_dir.exists():
                for checkpoint_file in ckpts_dir.glob("ckpt_*_rank0.pt"):
                    if gsplat_checkpoint is None or checkpoint_file.stat().st_mtime > gsplat_checkpoint.stat().st_mtime:
                        gsplat_checkpoint = checkpoint_file
        else:
            # Direct file check
            if pattern.exists():
                if gsplat_checkpoint is None or pattern.stat().st_mtime > gsplat_checkpoint.stat().st_mtime:
                    gsplat_checkpoint = pattern

    return fvdb_checkpoint, gsplat_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Side-by-side FVDB vs GSplat visualization")
    parser.add_argument("--scene", type=str, required=True, help="Scene name (e.g., bicycle, garden)")
    parser.add_argument("--gsplat_port", type=int, default=8081, help="Port for GSplat viewer")
    parser.add_argument("--data_dir", type=str, help="Path to dataset directory (optional)")
    parser.add_argument("--fvdb_checkpoint", type=str, help="Path to FVDB checkpoint (optional)")
    parser.add_argument("--gsplat_checkpoint", type=str, help="Path to GSplat checkpoint (optional)")

    args = parser.parse_args()

    # Find project root
    project_root = find_project_root()
    print(f"Project root: {project_root}")

    # Find checkpoint paths
    if args.fvdb_checkpoint:
        fvdb_checkpoint = Path(args.fvdb_checkpoint)
    else:
        fvdb_checkpoint, gsplat_checkpoint = find_checkpoint_paths(args.scene, project_root)

    if args.gsplat_checkpoint:
        gsplat_checkpoint = Path(args.gsplat_checkpoint)
    elif not args.fvdb_checkpoint:
        # This was already set above
        pass

    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = find_data_directory(project_root, args.scene)

    # Check if checkpoints exist
    if not fvdb_checkpoint or not fvdb_checkpoint.exists():
        print(f"Error: FVDB checkpoint not found")
        if fvdb_checkpoint:
            print(f"  Expected: {fvdb_checkpoint}")
        print("  Use --fvdb_checkpoint to specify the path manually")
        sys.exit(1)

    if not gsplat_checkpoint or not gsplat_checkpoint.exists():
        print(f"Error: GSplat checkpoint not found")
        if gsplat_checkpoint:
            print(f"  Expected: {gsplat_checkpoint}")
        print("  Use --gsplat_checkpoint to specify the path manually")
        sys.exit(1)

    print(f"Scene: {args.scene}")
    print(f"FVDB checkpoint: {fvdb_checkpoint}")
    print(f"GSplat checkpoint: {gsplat_checkpoint}")
    if data_dir:
        print(f"Data directory: {data_dir}")
    else:
        print("Data directory: Not found (will try to load from checkpoint)")
    print(f"FVDB viewer will be available at: http://localhost:8080")
    print(f"GSplat viewer will be available at: http://localhost:{args.gsplat_port}")
    print()

    processes = []

    try:
        # Start FVDB viewer
        fvdb_process = run_fvdb_viewer(str(fvdb_checkpoint), args.scene, project_root, data_dir)
        processes.append(fvdb_process)

        # Wait a moment for FVDB to start
        time.sleep(5)

        # Start GSplat viewer
        gsplat_process = run_gsplat_viewer(str(gsplat_checkpoint), args.gsplat_port, project_root)
        processes.append(gsplat_process)

        print("\n" + "=" * 60)
        print("BOTH VIEWERS STARTED!")
        print("=" * 60)
        print(f"FVDB Viewer: http://localhost:8080")
        print(f"GSplat Viewer: http://localhost:{args.gsplat_port}")
        print("\nPress Ctrl+C to stop both viewers")
        print("=" * 60)

        # Keep running until interrupted
        while True:
            time.sleep(1)

            # Check if processes are still running
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    print(f"Warning: Process {i+1} has stopped")

    except KeyboardInterrupt:
        print("\nStopping viewers...")

    finally:
        # Clean up processes
        for process in processes:
            if process.poll() is None:  # Still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        print("Viewers stopped.")


if __name__ == "__main__":
    main()
