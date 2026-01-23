#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Regenerate training plots with fixed metric extraction.

This script re-extracts metrics from existing comparison report JSON files
using the corrected metric extraction code, then regenerates the training plots.
"""

import json
import pathlib
import sys

# Add the comparative benchmark utils to the path
sys.path.insert(0, str(pathlib.Path(__file__).parent / "tests/benchmarks/comparative/benchmark_utils"))
sys.path.insert(0, str(pathlib.Path(__file__).parent / "tests/benchmarks/comparative"))

from _common import extract_training_metrics
from comparison_benchmark import save_training_curves


def regenerate_scene_plot(scene_name: str, results_dir: pathlib.Path):
    """Regenerate training plot for a single scene."""
    report_path = results_dir / f"{scene_name}_comparison_report.json"

    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return False

    print(f"Processing {scene_name}...")

    # Load the existing report
    with open(report_path, "r") as f:
        report = json.load(f)

    # Re-extract metrics for each config from log files
    for config_name, config_data in report.items():
        if "training" not in config_data:
            continue

        training_data = config_data["training"]
        total_time = training_data.get("total_time", 0.0)

        # Find the training log file
        # Try different log file locations based on framework
        log_paths = []
        if "fvdb" in config_name:
            log_paths.append(results_dir / f"{scene_name}__{config_name}" / "training.log")
        elif "gsplat" in config_name:
            # Try both with and without the extra subdirectory
            log_paths.append(
                results_dir / f"{scene_name}__{config_name}" / f"{scene_name}_{config_name}" / "training.log"
            )
            log_paths.append(
                results_dir / f"{scene_name}_{config_name}" / f"{scene_name}_{config_name}" / "training.log"
            )
            # Also try the pattern like "bicycle_gsplat"
            simple_name = config_name.replace("_default", "").replace("_mcmc", "")
            log_paths.append(
                results_dir / f"{scene_name}__{config_name}" / f"{scene_name}_{simple_name}" / "training.log"
            )

        log_file = None
        for path in log_paths:
            if path.exists():
                log_file = path
                break

        if not log_file:
            print(f"  Warning: Log file not found for {config_name}")
            continue

        # Read the log file
        with open(log_file, "r") as f:
            stdout = f.read()

        # Re-extract metrics with fixed code
        print(f"  Re-extracting metrics for {config_name}...")
        new_metrics = extract_training_metrics(stdout, total_time)

        # Update the metrics in the report
        old_psnr_steps = training_data.get("metrics", {}).get("psnr_steps", [])
        new_psnr_steps = new_metrics.get("psnr_steps", [])

        if old_psnr_steps != new_psnr_steps:
            print(f"    PSNR steps: {old_psnr_steps} -> {new_psnr_steps}")

        training_data["metrics"] = new_metrics

    # Save the updated report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Updated report saved")

    # Extract colors and config order from the report
    colors = {}
    config_order = []
    for config_name in report.keys():
        config_order.append(config_name)
        # Use default colors if not specified
        if "fvdb_default" in config_name:
            colors[config_name] = "#1f77b4"  # Blue
        elif "fvdb_mcmc" in config_name:
            colors[config_name] = "#ff7f0e"  # Orange
        elif "gsplat_default" in config_name:
            colors[config_name] = "#2ca02c"  # Green
        elif "gsplat_mcmc" in config_name:
            colors[config_name] = "#d62728"  # Red
        else:
            colors[config_name] = "#999999"  # Gray

    # Regenerate the training plot
    print(f"  Regenerating plot...")
    save_training_curves(scene_name, results_dir, colors, config_order)
    print(f"  Plot saved: {results_dir / f'{scene_name}_training.png'}")

    return True


def main():
    results_dir = pathlib.Path("tests/benchmarks/comparative/results/example_matrix")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    # Process each scene that has a training plot
    scenes = ["garden", "bicycle", "bonsai", "counter", "kitchen", "room", "stump"]

    for scene in scenes:
        print()
        success = regenerate_scene_plot(scene, results_dir)
        if not success:
            print(f"Failed to regenerate plot for {scene}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
