#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Profile both FVDB and GSplat inside the container.
"""
import os
import subprocess
import sys
import time


def run_fvdb_profile():
    """Profile FVDB"""
    print("=== PROFILING FVDB ===")

    # Create FVDB profiling script
    fvdb_script = """
import sys
import os
sys.path.insert(0, "/workspace/openvdb/fvdb/projects/3d_gaussian_splatting")
os.chdir("/workspace/openvdb/fvdb/projects/3d_gaussian_splatting")

from training.scene_optimization_runner import SceneOptimizationRunner, Config
import pathlib
import torch

# Create minimal config for profiling training - limit to exactly 10 steps
config = Config(
    max_epochs=1,  # Just one epoch (will be overridden by max_steps)
    max_steps=10,  # Exactly 10 training steps
    batch_size=1,
    save_at_percent=[],  # Don't save checkpoints
    eval_at_percent=[],  # Don't run evaluation
)

print("Creating FVDB runner...")
# Use the counter scene as requested - manually limit to 10 steps
runner = SceneOptimizationRunner.new_run(
    dataset_path=pathlib.Path("/workspace/data/360_v2/counter"),
    config=config,
    image_downsample_factor=4,
    normalization_type="pca",
    disable_viewer=True,
    results_path=pathlib.Path("/workspace/results/fvdb_step0_profile"),
    save_results=False,
)

print(f"Initial Gaussian count: {runner.model.num_gaussians}")
print("Starting FVDB training (limited to 10 steps)...")

# Add NVTX markers for better profiling
torch.cuda.nvtx.range_push("FVDB_Training")
checkpoint = runner.train()
torch.cuda.nvtx.range_pop()

print("FVDB training complete!")
"""

    # Write the script
    with open("/tmp/fvdb_profile_script.py", "w") as f:
        f.write(fvdb_script)

    # Run with nsys - capture full training run
    cmd = [
        "nsys",
        "profile",
        "--output=/workspace/results/fvdb_profile.nsys-rep",
        "--force-overwrite=true",
        "--trace=cuda,nvtx",
        "--cuda-memory-usage=true",
        "--capture-range=none",
        "--delay=2",
        "--duration=30",
        "--sample=cpu",
        "python3",
        "/tmp/fvdb_profile_script.py",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("FVDB STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("FVDB STDERR:")
        print(result.stderr)

    return result.returncode == 0


def run_gsplat_profile():
    """Profile GSplat"""
    print("\n=== PROFILING GSPLAT ===")

    # Change to GSplat directory and run with nsys
    cmd = [
        "nsys",
        "profile",
        "--output=/workspace/results/gsplat_profile.nsys-rep",
        "--force-overwrite=true",
        "--trace=cuda,nvtx",
        "--cuda-memory-usage=true",
        "--capture-range=none",
        "--delay=2",
        "--duration=30",
        "--sample=cpu",
        "python3",
        "simple_trainer.py",
        "default",
        "--data_dir",
        "/workspace/data/360_v2/counter",
        "--data_factor",
        "4",
        "--max_steps",
        "10",  # Exactly 10 steps for comparison
        "--disable_viewer",
        "--disable_video",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/workspace/gsplat/examples", capture_output=True, text=True)

    print("GSPLAT STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("GSPLAT STDERR:")
        print(result.stderr)

    return result.returncode == 0


def main():
    """Main profiling function."""
    print("Starting Profiling...")
    print(f"Working directory: {os.getcwd()}")

    # Create results directory
    os.makedirs("/workspace/results", exist_ok=True)

    # Profile FVDB
    fvdb_success = run_fvdb_profile()

    # Profile GSplat
    gsplat_success = run_gsplat_profile()

    print("\n=== PROFILING SUMMARY ===")
    print(f"FVDB profiling: {'SUCCESS' if fvdb_success else 'FAILED'}")
    print(f"GSplat profiling: {'SUCCESS' if gsplat_success else 'FAILED'}")

    if fvdb_success and gsplat_success:
        print("\n=== ANALYSIS COMMANDS ===")
        print("View profiles with:")
        print("  nsys-ui /workspace/results/fvdb_profile.nsys-rep")
        print("  nsys-ui /workspace/results/gsplat_profile.nsys-rep")
        print("\nExport data with:")
        print("  nsys export --type=sqlite /workspace/results/fvdb_profile.nsys-rep")
        print("  nsys export --type=sqlite /workspace/results/gsplat_profile.nsys-rep")

    return 0 if (fvdb_success and gsplat_success) else 1


if __name__ == "__main__":
    sys.exit(main())
