#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Comparative Benchmark Script

This script runs training for both FVDB and GSplat with various optimization configurations
on one or more scenes, generates reports for each scene, and creates summary plots comparing results.
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from common import load_config, setup_signal_handlers
from run_fvdb_training import run_fvdb_training
from run_gsplat_training import run_gsplat_training


def save_report_for_run(scene_name: str, training_results: dict[str, Any], output_directory: pathlib.Path) -> None:
    """
    Generate a JSON report summarizing the training and evaluation results for a given scene.

    Args:
        scene_name (str): The name of the scene.
        training_results (Dict): A dictionary containing training results for each configuration.
        eval_results (Dict): A dictionary containing evaluation results.
        result_dir (str): The directory to save the report.

    Returns:
        None
    """
    report_file_path = output_directory / f"{scene_name}_comparison_report.json"

    reports = {}
    for config_name, result in training_results.items():
        report = {
            "config_name": config_name,
            "scene": scene_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training": result,
            "success": result["success"],
            "total_time": result["total_time"],
            "training_time": result.get("training_time", result["total_time"]),
            "final_loss": result["metrics"].get("final_loss", None),
        }
        reports[config_name] = report

    with open(report_file_path, "w") as f:
        json.dump(reports, f, indent=2)

    # Log summary to console
    logging.info("=== COMPARISON SUMMARY ===")
    for i, (config_name, report) in enumerate(reports.items()):
        logging.info(f"Config: {config_name}:")
        logging.info("----------------------------")
        logging.info(f" Scene: {scene_name}")

        total_time = report["total_time"]
        training_time = report.get("training_time", total_time)

        logging.info(
            f"  Training: {'SUCCESS' if report['success'] else 'FAILED'} "
            f"(Total: {total_time:.2f}s, Training: {training_time:.2f}s)"
        )

        if report["success"]:
            if "final_loss" in report:
                final_loss = report["final_loss"]
                logging.info(f"  Final Loss: {final_loss:.6f}")
        if i < len(reports) - 1:
            logging.info("----------------------------")

    logging.info(f"Detailed report saved to: {report_file_path}")


def save_summary_report(scenes: list[str], result_path: pathlib.Path) -> None:
    """
    Generate a summary report comparing different runs across multiple scenes.

    This function creates a summary directory, generates grouped bar charts for each metric,
    and CSV and JSON files containing statistics across all scenes and configurations.

    Args:
        scenes (list[str]): List of scene names to include in the summary.
        result_dir (str): Directory containing the individual scene reports.

    Returns:
        None
    """

    # Create summary directory
    summary_dir = result_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # A dictionary to hold data for plotting
    # This function is used to generate grouped bar charts for each metric
    # The dictionary has string keys for each metric with values
    # plot_dict[metric] = {config1: [scene1_value, scene2_value, ...], config2: [...], ...}
    plot_dict = {
        "total_time": {},
        "training_time": {},
        "PSNR": {},
        "SSIM": {},
        "num_gaussians": {},
    }

    # A dictionary to hold summary metrics and statistics for each scene/opt-config pair
    summary_data = {}

    for scene in scenes:
        # Load comparison report for this scene
        report_file = result_path / f"{scene}_comparison_report.json"
        if not report_file.exists():
            logging.warning(f"No comparison report found for {scene}, skipping...")
            continue

        try:
            with open(report_file, "r") as f:
                report = json.load(f)  # dict[str, Any] : config path -> report data
        except Exception as e:
            logging.warning(f"Could not load report for {scene}: {e}")
            continue

        if scene not in summary_data:
            summary_data[scene] = {}

        for opt_config_name, report in report.items():
            for metric in plot_dict.keys():
                if opt_config_name not in plot_dict[metric]:
                    plot_dict[metric][opt_config_name] = []

            total_time = report.get("total_time", 0)
            training_time = report.get("training_time", total_time)
            psnr = report.get("training", {}).get("metrics", {}).get("psnr", 0)
            ssim = report.get("training", {}).get("metrics", {}).get("ssim", 0)
            num_gaussians = report.get("training", {}).get("metrics", {}).get("final_gaussian_count", 0)

            plot_dict["total_time"][opt_config_name].append(total_time)
            plot_dict["training_time"][opt_config_name].append(training_time)
            plot_dict["PSNR"][opt_config_name].append(psnr)
            plot_dict["SSIM"][opt_config_name].append(ssim)
            plot_dict["num_gaussians"][opt_config_name].append(num_gaussians)

            assert opt_config_name not in summary_data[scene], f"Duplicate config {opt_config_name} for scene {scene}"
            summary_data[scene][opt_config_name] = {
                "total_time": total_time,
                "training_time": training_time,
                "PSNR": psnr,
                "SSIM": ssim,
                "num_gaussians": num_gaussians,
            }

    num_metrics = len(plot_dict)
    fig, axs = plt.subplots(num_metrics, figsize=(7, 6 * num_metrics), layout="constrained")
    # For each metric, create a grouped bar chart
    for i, (metric, metric_data) in enumerate(plot_dict.items()):
        ax = axs[i]
        ax.grid(True)
        x = np.arange(len(scenes))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0  # Used to offset bars within a group
        # For each optimizer config, we plot a bar for each scene (one bar per group)
        for i, (opt_config_name, measurement) in enumerate(metric_data.items()):
            offset = width * multiplier
            assert isinstance(measurement, list)
            rects = ax.bar(x + offset, measurement, width, label=opt_config_name)
            ax.bar_label(rects, padding=3)
            multiplier += 1
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(f"{metric}")
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_xticks(x + width, scenes)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center")
    plt.tight_layout(pad=3.0)
    plt.savefig(summary_dir / f"summary_comparison.png", dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close()

    statistics = {}

    # Compute and log summary statistics for each metric across all scenes and configs
    def _log_statistics(metric_name: str, title: str, unit: str):
        logging.info(f"{title}:")
        for config in plot_dict[metric_name].keys():
            _values = plot_dict[metric_name][config]
            _values_mean = np.mean(_values)
            _values_std = np.std(_values)
            _values_median = np.median(_values)
            _values_min = np.min(_values)
            _values_max = np.max(_values)
            logging.info(
                f"  {config}: Mean {_values_mean:.1f}{unit} ± {_values_std:.1f}{unit}, Median {_values_median:.1f}{unit}, Min {_values_min:.1f}{unit}, Max {_values_max:.1f}{unit}"
            )
            if metric_name not in statistics:
                statistics[metric_name] = {}
            statistics[metric_name][config] = {
                "mean": _values_mean,
                "std": _values_std,
                "median": _values_median,
                "min": _values_min,
                "max": _values_max,
            }

    logging.info("=" * 80)
    logging.info("SUMMARY STATISTICS ACROSS ALL SCENES")
    logging.info("=" * 80)

    _log_statistics("total_time", "Total Time", "s")
    _log_statistics("training_time", "Training Time", "s")
    _log_statistics("PSNR", "PSNR", "dB")
    _log_statistics("SSIM", "SSIM", "dB")
    _log_statistics("num_gaussians", "Final Gaussian Count", "")

    output_summary = {
        "per_scene": summary_data,
        "statistics": statistics,
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open(summary_dir / "summary_data.json", "w") as f:
        json.dump(output_summary, f, cls=NpEncoder, indent=2)

    logging.info(f"Data exported to:")
    logging.info(f"  JSON: {summary_dir / 'summary_data.json'}")
    logging.info(f"  Plot: {summary_dir / 'summary_comparison.png'}")
    logging.info("=" * 80)


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
        --benchmark-config Path to the benchmark configuration YAML file (required unless --plot-only).
        --opt-configs      Space separated list of optimization config YAML files to use.
        --scenes           Space-separated list of scene names to benchmark (optional, defaults to all scenes in config).
        --result-dir       Directory to store results (default: results/benchmark).
        --log-level        Logging level (default: INFO).
        --list-scenes      List available scenes from config and exit.

    The script sets up signal handling for graceful interruption, parses arguments,
    loads configuration, and processes each scene as specified.

    Example usage:
        # Run all scenes from config
        python comparison_benchmark.py --benchmark-config config.yaml --opt-configs opt1.yaml opt2.yaml

        # Run specific scenes
        python comparison_benchmark.py --benchmark-config config.yaml --scenes garden,bicycle --opt-configs opt1.yaml opt2.yaml

        # List available scenes
        python comparison_benchmark.py --benchmark-config config.yaml --list-scenes

        # Generate plots from existing results
        python comparison_benchmark.py --scenes garden,bicycle --plot-only

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Simplified Comparative Benchmark")
    parser.add_argument(
        "--benchmark-config",
        default="benchmark_config.yaml",
        help="Path to benchmark config YAML (required unless --plot-only)",
    )
    parser.add_argument(
        "--opt-configs",
        nargs="+",
        help="Space separated list of optimization config YAML files to use (required unless --plot-only)",
    )
    parser.add_argument(
        "--scenes", nargs="*", help="Space separated list of scene names to benchmark (default: all scenes from config)"
    )
    parser.add_argument("--result-dir", default="results/benchmark", help="Results directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--list-scenes", action="store_true", help="List available scenes from config and exit")

    args = parser.parse_args()

    setup_signal_handlers()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("benchmark.log")],
    )

    # Load config (only needed if not plot-only)
    if not args.benchmark_config:
        parser.error("--benchmark-config is required unless --plot-only is specified")
    benchmark_config = load_config(args.benchmark_config)

    available_scenes = [dataset.get("name") for dataset in benchmark_config.get("datasets", [])]
    if not available_scenes:
        parser.error("No scenes found in config file")

    # Handle --list-scenes option
    if args.list_scenes:
        print("Available scenes in config:")
        for scene in available_scenes:
            print(f"  - {scene}")
        sys.exit(0)

    # Parse scenes from command line if specified otherwise use all from config
    if args.scenes:
        # Use scenes from command line
        scenes = args.scenes
    else:
        # Use all scenes from config
        scenes = available_scenes
        logging.info(f"Using all scenes from config: {', '.join(scenes)}")

    # Create results directory
    results_path = pathlib.Path(args.result_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Process each scene
    for scene_name in scenes:
        logging.info(f"Processing scene: {scene_name}")

        # Run training

        training_results = {}
        if not args.opt_configs:
            parser.error("--opt-configs is required unless --plot-only or --eval-only is specified")

        # Validate that all optimization configs have unique names
        all_config_names = dict()
        for opt_config_path in args.opt_configs:
            opt_config = load_config(opt_config_path)
            if "framework" not in opt_config:
                raise RuntimeError(f"Framework not specified in opt config: {opt_config_path}")
            if "name" not in opt_config:
                raise RuntimeError(f"Name not specified in opt config: {opt_config_path}")
            config_name = opt_config["name"]
            if config_name in all_config_names:
                raise ValueError(
                    f"Duplicate config name detected: {config_name} in files {all_config_names[config_name]} and {opt_config_path}"
                )
            all_config_names[config_name] = opt_config_path

        # Run training for each optimization configuration
        for opt_config_path in args.opt_configs:
            opt_config = load_config(opt_config_path)
            framework = opt_config["framework"]
            config_name = opt_config["name"]
            if framework == "fvdb":
                fvdb_results = run_fvdb_training(
                    scene_name,
                    results_path,
                    pathlib.Path(args.benchmark_config),
                    pathlib.Path(opt_config_path),
                    config_name,
                )
                training_results[config_name] = fvdb_results

            elif framework == "gsplat":
                raise NotImplementedError("GSplat training not implemented in this script")
                # gsplat_results = run_gsplat_training(scene_name, result_dir, opt_config)

        # Generate summary report for each optimization config that was run
        if training_results:
            save_report_for_run(
                scene_name=scene_name,
                training_results=training_results,
                output_directory=results_path,
            )

        logging.info(f"Completed benchmark for {scene_name}")

    # Generate summary charts if multiple scenes were processed
    save_summary_report(scenes, results_path)

    logging.info("All benchmarks completed!")


if __name__ == "__main__":
    main()
