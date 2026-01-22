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
import yaml
from benchmark_utils import load_config, run_fvdb_training, run_gsplat_training

default_colors = ["#76B900", "#767676"]


def save_report_for_run(scene_name: str, training_results: dict[str, Any], output_directory: pathlib.Path) -> None:
    """
    Generate a JSON report summarizing the comparison of training results for multiple configurations for a given scene.

    Args:
        scene_name (str): The name of the scene.
        training_results (Dict): A dictionary containing training results for each configuration.
        output_directory (pathlib.Path): The directory to save the report.

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


def save_summary_report(
    scenes: list[str], result_path: pathlib.Path, colors: dict[str, str], config_order: list[str]
) -> None:
    """
    Generate a summary report comparing different runs across multiple scenes.

    This function creates a summary directory, generates grouped bar charts for each metric,
    and statistics files containing aggregated results across all scenes and configurations.

    Files saved:
        summary/summary_comparison.png
            Grouped bar chart visualization showing all metrics (training throughput, PSNR, SSIM,
            final Gaussian count, total time, training time, peak GPU memory) with one group per
            scene and one bar per configuration. Each bar is labeled with its value. Missing data
            shown as "NA".

        summary/summary_data.json
            Aggregated statistics in JSON format containing two top-level keys:
            - "per_scene": Dict mapping scene names to configuration results. For each scene and
              config combination, contains metrics (training_throughput, PSNR, SSIM, num_gaussians,
              total_time, training_time, peak_gpu_memory_gb).
            - "statistics": Dict mapping each metric to per-config statistics including mean,
              standard deviation, median, min, and max values computed across all scenes.

    Args:
        scenes (list[str]): List of scene names to include in the summary.
        result_path (pathlib.Path): Directory containing the individual scene reports
            (expects files like "{scene}_comparison_report.json").
        colors (dict[str, str]): Dictionary mapping configuration names to hex color codes for plotting.
        config_order (list[str]): List of configuration names in the order to display in plots.

    Returns:
        None
    """

    # Create summary directory
    summary_dir = result_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # plot_dict[metric][config] = [scene1_value, scene2_value, ...]
    plot_dict: dict[str, dict[str, list[float]]] = {
        "training_throughput": {c: [] for c in config_order},
        "PSNR": {c: [] for c in config_order},
        "SSIM": {c: [] for c in config_order},
        "num_gaussians": {c: [] for c in config_order},
        "total_time": {c: [] for c in config_order},
        "training_time": {c: [] for c in config_order},
        "peak_gpu_memory_gb": {c: [] for c in config_order},
    }

    # Labels with units (y-axis)
    plot_dict_labels = {
        "training_throughput": "Training Throughput (splats/s)",
        "PSNR": "PSNR (dB)",
        "SSIM": "SSIM (0-1)",
        "num_gaussians": "Final Gaussian Count",
        "total_time": "Total Time (s)",
        "training_time": "Training Time (s)",
        "peak_gpu_memory_gb": "Peak GPU Memory (GB)",
    }

    # Plot titles (properly capitalized)
    plot_dict_titles = {
        "training_throughput": "Training Throughput",
        "PSNR": "PSNR",
        "SSIM": "SSIM",
        "num_gaussians": "Num Gaussians",
        "total_time": "Total Time",
        "training_time": "Training Time",
        "peak_gpu_memory_gb": "Peak GPU Memory GB",
    }

    # A dictionary to hold summary metrics and statistics for each scene/opt-config pair
    summary_data = {}

    for scene in scenes:
        # Load comparison report for this scene
        report_file = result_path / f"{scene}_comparison_report.json"
        if not report_file.exists():
            logging.warning(f"No comparison report found for {scene}, skipping...")
            # Pad with NaNs so plots stay aligned
            for cfg in config_order:
                for metric in plot_dict.keys():
                    plot_dict[metric][cfg].append(float("nan"))
            continue

        try:
            with open(report_file, "r") as f:
                report = json.load(f)  # dict[str, Any] : config path -> report data
        except Exception as e:
            logging.warning(f"Could not load report for {scene}: {e}")
            continue

        if scene not in summary_data:
            summary_data[scene] = {}

        # Ensure we append values in the same config order for each scene
        for cfg in config_order:
            cfg_report = report.get(cfg)
            if cfg_report is None:
                for metric in plot_dict.keys():
                    plot_dict[metric][cfg].append(float("nan"))
                continue

            total_time = cfg_report.get("total_time", 0.0)
            training_time = cfg_report.get("training_time", total_time)
            psnr = cfg_report.get("training", {}).get("metrics", {}).get("psnr", float("nan"))
            ssim = cfg_report.get("training", {}).get("metrics", {}).get("ssim", float("nan"))
            num_gaussians = cfg_report.get("training", {}).get("metrics", {}).get("final_gaussian_count", float("nan"))
            peak_gpu_memory_gb = (
                cfg_report.get("training", {}).get("metrics", {}).get("peak_gpu_memory_gb", float("nan"))
            )
            training_throughput = num_gaussians / training_time if training_time and training_time > 0 else float("nan")

            plot_dict["training_throughput"][cfg].append(float(training_throughput))
            plot_dict["PSNR"][cfg].append(float(psnr))
            plot_dict["SSIM"][cfg].append(float(ssim))
            plot_dict["num_gaussians"][cfg].append(float(num_gaussians))
            plot_dict["total_time"][cfg].append(float(total_time))
            plot_dict["training_time"][cfg].append(float(training_time))
            plot_dict["peak_gpu_memory_gb"][cfg].append(float(peak_gpu_memory_gb))

            assert cfg not in summary_data[scene], f"Duplicate config {cfg} for scene {scene}"
            summary_data[scene][cfg] = {
                "training_throughput": training_throughput,
                "PSNR": psnr,
                "SSIM": ssim,
                "num_gaussians": num_gaussians,
                "total_time": total_time,
                "training_time": training_time,
                "peak_gpu_memory_gb": peak_gpu_memory_gb,
            }

    num_metrics = len(plot_dict)
    fig, axs = plt.subplots(num_metrics, figsize=(7, 6 * num_metrics))

    # For each metric, create a grouped bar chart
    for i, (metric, metric_data) in enumerate(plot_dict.items()):
        ax = axs[i]
        ax.grid(True)
        x = np.arange(len(scenes))  # the label locations
        gap = 0.2
        width = (1 - gap) / len(metric_data)  # the width of the bars
        multiplier = 0  # Used to offset bars within a group

        # For each optimizer config, we plot a bar for each scene (one bar per group)
        for _, (cfg_name, measurement) in enumerate(metric_data.items()):
            offset = width * multiplier
            assert isinstance(measurement, list)
            values = np.array(measurement, dtype=float)
            plot_values = np.nan_to_num(values, nan=0.0)

            rects = ax.bar(x + offset, plot_values, width, label=cfg_name, color=colors.get(cfg_name, "#999999"))

            # Per-bar labels: show NA for missing values
            if metric in ["num_gaussians"]:
                labels = ["NA" if np.isnan(v) else f"{int(v):d}" for v in values]
            else:
                labels = ["NA" if np.isnan(v) else f"{float(v):.2f}" for v in values]
            ax.bar_label(rects, labels=labels, rotation=45, padding=3)

            multiplier += 1
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(f"{plot_dict_labels[metric]}")
        ax.set_title(plot_dict_titles[metric])
        ax.set_xticks(x + width, scenes)
        # Make the xtick labels diagonal for better readability
        ax.set_xticklabels(scenes, rotation=45, ha="right")
        ax.margins(y=0.15)
        ax.grid(axis="x", visible=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center")
    plt.tight_layout(pad=3.0)
    # add space above heighest bars to avoid cutting off the labels
    plt.savefig(summary_dir / f"summary_comparison.png", dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close()

    statistics = {}

    # Compute and log summary statistics for each metric across all scenes and configs
    def _log_statistics(metric_name: str, title: str, unit: str):
        logging.info(f"{title}:")
        for config in plot_dict[metric_name].keys():
            _values = np.array(plot_dict[metric_name][config], dtype=float)
            _values = _values[~np.isnan(_values)]
            if _values.size == 0:
                logging.info(f"  {config}: (no data)")
                continue

            _values_mean = float(np.mean(_values))
            _values_std = float(np.std(_values))
            _values_median = float(np.median(_values))
            _values_min = float(np.min(_values))
            _values_max = float(np.max(_values))
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

    _log_statistics("training_throughput", "Training Throughput", "splats/s")
    _log_statistics("PSNR", "PSNR", "dB")
    _log_statistics("SSIM", "SSIM", "")
    _log_statistics("num_gaussians", "Final Gaussian Count", "")
    _log_statistics("total_time", "Total Time", "s")
    _log_statistics("training_time", "Training Time", "s")
    _log_statistics("peak_gpu_memory_gb", "Peak GPU Memory", "GB")

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

    This script benchmarks and compares fVDB 3D Gaussian Splatting to GSplat on multiple scenes
    defined in a matrix configuration YAML file. It runs training for each configured run,
    generates per-scene reports, and produces a summary report with comparative visualizations.

    Matrix Configuration Structure:
        The matrix YAML file defines:
        - datasets: List of scene definitions with paths and metadata
        - opt_configs: Mapping of optimizer config aliases to YAML file paths and colors
        - runs: List of (dataset, opt_config) pairs specifying which runs to execute

        Results are saved to results/<matrix_name>/ relative to the matrix file location.

    Command-line Arguments:
        --matrix       Path to matrix YAML file defining datasets, opt_configs, and runs (required).
        --log-level    Logging level (default: INFO). Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.
        --plot-only    Skip training and only generate plots from existing results.

    Workflow:
        1. Loads matrix configuration YAML file
        2. Creates results directory at results/<matrix_name>/
        3. For each run in the matrix:
           - Prepares framework-specific (fVDB or GSplat) configuration
           - Executes training and captures metrics
           - Saves per-run results to run_dir
        4. Generates per-scene comparison report (JSON) summarizing all runs for that scene
        5. Generates summary report across all scenes with plots and aggregated statistics
           Outputs to results/<matrix_name>/summary/

    Output Files:
        Per-scene reports (at results/<matrix_name>/):
            {scene_name}_comparison_report.json
                Comparison metrics for all configurations on a given scene

        Summary report (at results/<matrix_name>/summary/):
            summary_comparison.png
                Grouped bar charts comparing all metrics across scenes and configurations
            summary_data.json
                Per-scene results and aggregated statistics for all metrics

        Logs:
            benchmark.log (at results/<matrix_name>/)
                Full execution log with timestamps

    Example usage:
        # Run all scenes and configurations defined in matrix
        python comparison_benchmark.py --matrix garden_matrix.yml

        # Run with verbose logging
        python comparison_benchmark.py --matrix garden_matrix.yml --log-level DEBUG

        # Generate plots from existing results without re-training
        python comparison_benchmark.py --matrix garden_matrix.yml --plot-only

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Comparative Benchmark (matrix-driven)")
    parser.add_argument("--matrix", required=True, help="Path to matrix YAML file defining datasets, configs, and runs")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    parser.add_argument("--plot-only", action="store_true", help="Only plot results from an existing run and exit")

    args = parser.parse_args()

    matrix_path = pathlib.Path(args.matrix)
    matrix_dir = matrix_path.parent
    matrix_config = load_config(matrix_path)

    matrix_name = matrix_config.get("name")
    if not matrix_name:
        parser.error("matrix.yml must define a top-level 'name:' field")

    # Results live under results/<matrix_name>/ relative to the matrix file location
    results_path = (matrix_dir / "results" / str(matrix_name)).resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_path}")

    # Setup logging
    benchmark_log_path = results_path / "benchmark.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(benchmark_log_path)],
    )

    datasets = matrix_config.get("datasets", [])
    if not datasets:
        parser.error("matrix.yml must define a non-empty 'datasets:' list")
    dataset_by_name = {d.get("name"): d for d in datasets if isinstance(d, dict) and d.get("name")}

    opt_configs = matrix_config.get("opt_configs", {})
    if not isinstance(opt_configs, dict) or not opt_configs:
        parser.error("matrix.yml must define a non-empty 'opt_configs:' mapping")

    runs = matrix_config.get("runs", [])
    if not isinstance(runs, list) or not runs:
        parser.error("matrix.yml must define a non-empty 'runs:' list")

    def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge override into base and return a new dict."""
        out: dict[str, Any] = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
            else:
                out[k] = v
        return out

    # Build run plan grouped by scene
    runs_by_scene: dict[str, list[dict[str, Any]]] = {}
    config_order: list[str] = []
    all_config_colors: dict[str, str] = {}

    for run in runs:
        if not isinstance(run, dict):
            raise ValueError(f"Invalid run entry (expected mapping): {run}")
        scene_name = run.get("dataset")
        opt_alias = run.get("opt_config")
        if not scene_name or scene_name not in dataset_by_name:
            raise ValueError(f"Run references unknown dataset: {scene_name}")
        if not opt_alias or opt_alias not in opt_configs:
            logging.warning(f"Skipping run with unknown opt_config alias: {opt_alias}")
            continue

        variant = run.get("variant")
        run_key = f"{opt_alias}__{variant}" if variant else str(opt_alias)
        run_dir = results_path / f"{scene_name}__{run_key}"

        opt_entry = opt_configs[opt_alias]
        if not isinstance(opt_entry, dict) or "path" not in opt_entry:
            raise ValueError(f"opt_configs.{opt_alias} must be a mapping with a 'path' field")
        opt_config_path = (matrix_dir / opt_entry["path"]).resolve()
        opt_config = load_config(opt_config_path)
        if "framework" not in opt_config:
            raise RuntimeError(f"Framework not specified in opt config: {opt_config_path}")
        framework = opt_config["framework"]

        overrides = run.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"Run overrides must be a mapping, got: {type(overrides)}")
        framework_overrides = overrides.get(framework, {}) or {}
        if framework_overrides and not isinstance(framework_overrides, dict):
            raise ValueError(f"Run overrides for '{framework}' must be a mapping")

        if run_key not in config_order:
            config_order.append(run_key)
        color = opt_config.get("color")
        if isinstance(color, str):
            prev = all_config_colors.get(run_key)
            if prev is None:
                all_config_colors[run_key] = color
            elif prev != color:
                logging.warning(f"Color mismatch for {run_key}: {prev} vs {color}; keeping {prev}")

        runs_by_scene.setdefault(scene_name, []).append(
            {
                "scene_name": scene_name,
                "run_key": run_key,
                "run_dir": run_dir,
                "framework": framework,
                "opt_alias": opt_alias,
                "opt_config_path": opt_config_path,
                "opt_config": opt_config,
                "framework_overrides": framework_overrides,
            }
        )

    # Determine scenes to process (in datasets order, but only those with runs)
    scenes = [d["name"] for d in datasets if d.get("name") in runs_by_scene]
    if not scenes:
        parser.error("No runnable scenes found (check 'runs:' vs 'datasets:')")

    # Process each scene
    for scene_name in scenes:
        logging.info(f"Processing scene: {scene_name}")

        # Plot-only mode: do not run training, only summarize existing reports
        if args.plot_only:
            continue

        training_results: dict[str, Any] = {}

        # Run training for each configured run for this scene
        for run_def in runs_by_scene.get(scene_name, []):
            framework = run_def["framework"]
            run_key = run_def["run_key"]
            run_dir = run_def["run_dir"]
            opt_config_path = run_def["opt_config_path"]
            opt_config = run_def["opt_config"]
            framework_overrides = run_def["framework_overrides"]

            if framework == "fvdb":
                merged_opt = deep_merge(opt_config, framework_overrides)
                merged_opt_path = run_dir / "opt_config.yml"
                run_dir.mkdir(parents=True, exist_ok=True)
                with open(merged_opt_path, "w") as f:
                    yaml.safe_dump(merged_opt, f, default_flow_style=False, sort_keys=False)

                fvdb_results = run_fvdb_training(
                    scene_name=scene_name,
                    run_dir=run_dir,
                    matrix_config_path=matrix_path,
                    opt_config_path=merged_opt_path,
                    fvdb_results_base_path=run_dir / "fvdb_results",
                )
                training_results[run_key] = fvdb_results

            elif framework == "gsplat":
                # For GSplat, we support:
                # - deep-merge overrides into the opt-config for parameter extraction (e.g. max_epochs)
                # - append extra CLI args from opt-config + overrides
                gsplat_overrides_no_cli = dict(framework_overrides)
                gsplat_overrides_no_cli.pop("cli_args", None)
                merged_opt = deep_merge(opt_config, gsplat_overrides_no_cli)
                merged_opt_path = run_dir / "opt_config.yml"
                run_dir.mkdir(parents=True, exist_ok=True)
                with open(merged_opt_path, "w") as f:
                    yaml.safe_dump(merged_opt, f, default_flow_style=False, sort_keys=False)

                opt_cli_args = opt_config.get("cli_args", []) or []
                override_cli_args = framework_overrides.get("cli_args", []) or []
                if not isinstance(opt_cli_args, list) or not all(isinstance(x, str) for x in opt_cli_args):
                    raise ValueError(f"{opt_config_path} cli_args must be a list[str]")
                if not isinstance(override_cli_args, list) or not all(isinstance(x, str) for x in override_cli_args):
                    raise ValueError(f"Run overrides for gsplat.cli_args must be a list[str]")

                gsplat_results = run_gsplat_training(
                    scene_name=scene_name,
                    run_dir=run_dir,
                    matrix_config_path=matrix_path,
                    opt_config_path=merged_opt_path,
                    extra_cli_args=[*opt_cli_args, *override_cli_args],
                )
                training_results[run_key] = gsplat_results

            else:
                raise ValueError(f"Unsupported framework: {framework}")

        # Generate per-scene report
        if training_results:
            save_report_for_run(scene_name=scene_name, training_results=training_results, output_directory=results_path)

        logging.info(f"Completed benchmark for {scene_name}")

    # Generate summary charts if multiple scenes were processed
    if args.plot_only:
        # Warn about missing reports for expected scenes (behavior B)
        for scene_name in scenes:
            report_file = results_path / f"{scene_name}_comparison_report.json"
            if not report_file.exists():
                logging.warning(f"Missing comparison report for expected scene '{scene_name}': {report_file}")

    save_summary_report(scenes, results_path, all_config_colors, config_order)

    logging.info("All benchmarks completed!")


if __name__ == "__main__":
    main()
