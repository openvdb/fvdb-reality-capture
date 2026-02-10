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
from benchmark_utils import (
    build_fvdb_core,
    checkout_commit,
    get_current_commit,
    get_git_info,
    install_python_package,
    load_config,
    run_fvdb_training,
    run_gsplat_training,
)

default_colors = ["#76B900", "#767676"]


# =============================================================================
# Commit Management
# =============================================================================


def get_commits_from_opt_config(opt_config: dict[str, Any]) -> dict[str, str | None]:
    """
    Extract commit specifications from an opt_config.

    Args:
        opt_config: The loaded opt_config dictionary.

    Returns:
        Dictionary with keys 'fvdb_core', 'fvdb_reality_capture', 'gsplat'
        and values being commit SHAs or None if not specified.
    """
    commits = opt_config.get("commits", {}) or {}
    return {
        "fvdb_core": commits.get("fvdb_core"),
        "fvdb_reality_capture": commits.get("fvdb_reality_capture"),
        "gsplat": commits.get("gsplat"),
    }


def get_commit_key(
    opt_config: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """
    Get a hashable key representing the commit combination for an opt_config.

    Args:
        opt_config: The loaded opt_config dictionary.

    Returns:
        Tuple of (fvdb_core_commit, fvdb_reality_capture_commit, gsplat_commit).
    """
    commits = get_commits_from_opt_config(opt_config)
    return (commits["fvdb_core"], commits["fvdb_reality_capture"], commits["gsplat"])


def detect_repo_paths(matrix_config: dict[str, Any], matrix_dir: pathlib.Path) -> dict[str, pathlib.Path | None]:
    """
    Detect repository paths from matrix config or environment.

    Args:
        matrix_config: The loaded matrix configuration.
        matrix_dir: Directory containing the matrix config file.

    Returns:
        Dictionary with keys 'fvdb_core', 'fvdb_reality_capture', 'gsplat'
        and values being pathlib.Path objects or None if not found.
    """
    paths = matrix_config.get("paths", {}) or {}

    # Default paths based on typical container layout
    default_candidates = {
        "fvdb_core": [
            pathlib.Path("/workspace/openvdb/fvdb-core"),
            matrix_dir.parent.parent.parent.parent / "fvdb-core",
        ],
        "fvdb_reality_capture": [
            pathlib.Path("/workspace/openvdb/fvdb-reality-capture"),
            matrix_dir.parent.parent.parent,  # Relative to tests/benchmarks/comparative/
        ],
        "gsplat": [
            pathlib.Path("/workspace/gsplat"),
            (matrix_dir / paths.get("gsplat_base", "../gsplat/examples")).resolve(),
        ],
    }

    detected = {}
    for repo, candidates in default_candidates.items():
        # Check if explicitly specified in paths
        explicit_path = paths.get(f"{repo}_path")
        if explicit_path:
            p = pathlib.Path(explicit_path)
            if not p.is_absolute():
                p = (matrix_dir / p).resolve()
            if p.exists():
                detected[repo] = p.resolve()
                continue

        # Try default candidates
        for candidate in candidates:
            if candidate.exists():
                # For gsplat, we want the repo root, not examples dir
                if repo == "gsplat" and "examples" in str(candidate):
                    candidate = candidate.parent
                detected[repo] = candidate.resolve()
                break
        else:
            detected[repo] = None

    return detected


class CommitManager:
    """
    Manages checkout and build state for repositories.

    Tracks which commits are currently installed and handles checkout/rebuild
    when needed to minimize rebuilds.
    """

    def __init__(self, repo_paths: dict[str, pathlib.Path | None]):
        """
        Initialize the commit manager.

        Args:
            repo_paths: Dictionary mapping repo names to their paths.
        """
        self.repo_paths = repo_paths
        self.current_commits: dict[str, str | None] = {
            "fvdb_core": None,
            "fvdb_reality_capture": None,
            "gsplat": None,
        }
        self.initial_commits: dict[str, str | None] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize current commit state by reading from repositories."""
        if self._initialized:
            return

        for repo, path in self.repo_paths.items():
            if path and path.exists():
                self.current_commits[repo] = get_current_commit(path)
                logging.info(
                    f"Detected {repo} at commit {self.current_commits[repo][:7] if self.current_commits[repo] else 'unknown'}"
                )

        # Capture initial commits so we can restore them for groups that
        # don't specify a commit (required is None).
        self.initial_commits = dict(self.current_commits)
        self._initialized = True

    def ensure_commits(self, required_commits: dict[str, str | None], framework: str) -> dict[str, dict[str, Any]]:
        """
        Ensure the required commits are checked out and built.

        Args:
            required_commits: Dictionary of repo -> commit SHA (or None for current).
            framework: The framework being used ('fvdb' or 'gsplat').

        Returns:
            Dictionary of repo -> git_info for the current state after checkout/build.
        """
        self.initialize()

        git_info: dict[str, dict[str, Any]] = {}
        needs_fvdb_core_rebuild = False
        needs_fvdb_rc_reinstall = False
        needs_gsplat_rebuild = False

        # Check which repos need checkout
        for repo in ["fvdb_core", "fvdb_reality_capture", "gsplat"]:
            required = required_commits.get(repo)
            current = self.current_commits.get(repo)
            path = self.repo_paths.get(repo)

            # When no commit is specified, restore the initial HEAD so that
            # runs without explicit commits are not order-dependent on
            # whatever a previous commit-group left behind.
            if not required:
                initial = self.initial_commits.get(repo)
                if initial and path and initial != current:
                    logging.info(f"Restoring {repo} to initial commit {initial[:7]}")
                    if checkout_commit(path, initial):
                        self.current_commits[repo] = get_current_commit(path) or initial
                        if repo == "fvdb_core":
                            needs_fvdb_core_rebuild = True
                        elif repo == "fvdb_reality_capture":
                            needs_fvdb_rc_reinstall = True
                        elif repo == "gsplat":
                            needs_gsplat_rebuild = True
                    else:
                        logging.warning(f"Failed to restore {repo} to initial commit {initial[:7]}")
                continue

            # Warn if commit specified but repo path not found
            if not path:
                logging.warning(
                    f"Commit {required[:7] if len(required) >= 7 else required} specified for {repo}, "
                    f"but repository path was not found. Skipping checkout."
                )
                continue

            if required != current and path:
                logging.info(
                    f"Commit mismatch for {repo}: current={current[:7] if current else 'None'}, "
                    f"required={required[:7] if len(required) >= 7 else required}"
                )

                # Checkout the required commit
                if checkout_commit(path, required):
                    # Normalize to the full resolved HEAD SHA so that abbreviated
                    # SHAs or equivalent refs compare correctly on subsequent runs.
                    self.current_commits[repo] = get_current_commit(path) or required

                    # Mark for rebuild
                    if repo == "fvdb_core":
                        needs_fvdb_core_rebuild = True
                    elif repo == "fvdb_reality_capture":
                        needs_fvdb_rc_reinstall = True
                    elif repo == "gsplat":
                        needs_gsplat_rebuild = True
                else:
                    msg = f"Failed to checkout {required} in {repo}"
                    logging.error(msg)
                    raise RuntimeError(msg)

        # Perform rebuilds as needed
        if needs_fvdb_core_rebuild and self.repo_paths.get("fvdb_core"):
            logging.info("Rebuilding fvdb-core...")
            if not build_fvdb_core(self.repo_paths["fvdb_core"]):
                msg = "fvdb-core build failed!"
                logging.error(msg)
                raise RuntimeError(msg)

        if needs_fvdb_rc_reinstall and self.repo_paths.get("fvdb_reality_capture"):
            logging.info("Reinstalling fvdb-reality-capture...")
            if not install_python_package(self.repo_paths["fvdb_reality_capture"]):
                msg = "fvdb-reality-capture install failed!"
                logging.error(msg)
                raise RuntimeError(msg)

        if needs_gsplat_rebuild and self.repo_paths.get("gsplat"):
            logging.info("Rebuilding gsplat...")
            if not install_python_package(self.repo_paths["gsplat"]):
                msg = "gsplat build failed!"
                logging.error(msg)
                raise RuntimeError(msg)

        # Collect current git info for all relevant repos
        if framework == "fvdb":
            for repo in ["fvdb_core", "fvdb_reality_capture"]:
                path = self.repo_paths.get(repo)
                if path:
                    git_info[repo] = get_git_info(path)
        elif framework == "gsplat":
            path = self.repo_paths.get("gsplat")
            if path:
                git_info["gsplat"] = get_git_info(path)

        return git_info


def save_report_for_run(scene_name: str, training_results: dict[str, Any], output_directory: pathlib.Path) -> None:
    """
    Generate a JSON report summarizing the comparison of training results for multiple configurations for a given scene.

    Args:
        scene_name (str): The name of the scene.
        training_results (Dict): A dictionary containing training results for each configuration.
            Each result may contain a "repositories" key with git info.
        output_directory (pathlib.Path): The directory to save the report.

    Returns:
        None
    """
    report_file_path = output_directory / f"{scene_name}_comparison_report.json"

    reports = {}
    for config_name, result in training_results.items():
        # Extract repository info if present
        repositories = result.get("repositories", {})

        report = {
            "config_name": config_name,
            "scene": scene_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "repositories": repositories,
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

        # Log repository info if present
        repositories = report.get("repositories", {})
        if repositories:
            for repo_name, repo_info in repositories.items():
                if isinstance(repo_info, dict) and repo_info.get("commit"):
                    commit = repo_info.get("short_commit", repo_info.get("commit", "")[:7])
                    dirty = " (dirty)" if repo_info.get("dirty") else ""
                    logging.info(f"  {repo_name}: {commit}{dirty}")

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
    scenes: list[str],
    result_path: pathlib.Path,
    colors: dict[str, str],
    config_order: list[str],
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

            rects = ax.bar(
                x + offset,
                plot_values,
                width,
                label=cfg_name,
                color=colors.get(cfg_name, "#999999"),
            )

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
    plt.savefig(
        summary_dir / f"summary_comparison.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.5,
    )
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

    # Collect unique repository versions from reports
    repository_versions: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        report_file = result_path / f"{scene}_comparison_report.json"
        if report_file.exists():
            try:
                with open(report_file, "r") as f:
                    report = json.load(f)
                for cfg_name, cfg_report in report.items():
                    repos = cfg_report.get("repositories", {})
                    for repo_name, repo_info in repos.items():
                        if isinstance(repo_info, dict) and repo_info.get("commit"):
                            key = f"{cfg_name}:{repo_name}"
                            if key not in repository_versions:
                                repository_versions[key] = {
                                    "config": cfg_name,
                                    "repository": repo_name,
                                    "commit": repo_info.get("commit"),
                                    "short_commit": repo_info.get("short_commit"),
                                    "branch": repo_info.get("branch"),
                                    "dirty": repo_info.get("dirty"),
                                }
            except Exception as exc:
                logging.warning(
                    "Failed to load comparison report '%s': %s",
                    report_file,
                    exc,
                )

    output_summary = {
        "per_scene": summary_data,
        "statistics": statistics,
        "repository_versions": repository_versions,
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


def save_training_curves(
    scene_name: str,
    result_path: pathlib.Path,
    colors: dict[str, str],
    config_order: list[str],
) -> None:
    """
    Generate training curve plots for a single scene across multiple configurations.

    Creates a multi-subplot figure showing time-dependent metrics:
    - Iterations/s (training throughput over time)
    - Loss (training loss convergence)
    - Gaussian Count (number of Gaussians over time)
    - PSNR (if available from validation during training)
    - SSIM (if available from validation during training)

    Files saved:
        {scene_name}_training.png
            Line plot visualization with 2-5 subplots (depending on available metrics).
            Each subplot shows one metric over training steps with one line per configuration.

    Args:
        scene_name (str): Name of the scene to plot.
        result_path (pathlib.Path): Directory containing the scene comparison report.
        colors (dict[str, str]): Dictionary mapping configuration names to hex colors.
        config_order (list[str]): List of configuration names in display order.

    Returns:
        None
    """
    # Load comparison report for this scene
    report_file = result_path / f"{scene_name}_comparison_report.json"
    if not report_file.exists():
        logging.warning(f"Cannot generate training curves: missing report {report_file}")
        return

    try:
        with open(report_file, "r") as f:
            report = json.load(f)
    except Exception as e:
        logging.warning(f"Could not load report for training curves: {e}")
        return

    # Determine which metrics are available across all configs
    has_psnr = False
    has_ssim = False
    has_iterations = False
    has_loss = False
    has_gaussian_count = False

    for config_name in config_order:
        if config_name not in report:
            continue
        metrics = report[config_name].get("training", {}).get("metrics", {})
        if metrics.get("psnr_values") and len(metrics.get("psnr_values", [])) > 1:
            has_psnr = True
        if metrics.get("ssim_values") and len(metrics.get("ssim_values", [])) > 1:
            has_ssim = True
        if metrics.get("iteration_rates") and len(metrics.get("iteration_rates", [])) > 0:
            has_iterations = True
        if metrics.get("loss_values") and len(metrics.get("loss_values", [])) > 0:
            has_loss = True
        if metrics.get("gaussian_count_values") and len(metrics.get("gaussian_count_values", [])) > 0:
            has_gaussian_count = True

    # Determine subplot layout - separate iterations and loss for clarity
    num_plots = 0
    if has_iterations:
        num_plots += 1
    if has_loss:
        num_plots += 1
    if has_gaussian_count:
        num_plots += 1
    if has_psnr:
        num_plots += 1
    if has_ssim:
        num_plots += 1

    if num_plots == 0:
        logging.info(f"No training curve data available for {scene_name}, skipping training curves")
        return

    # Create figure with dynamic subplot layout
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 3.5 * num_plots))
    if num_plots == 1:
        axs = [axs]  # Make it iterable

    subplot_idx = 0

    # Subplot 1: Iterations/s (training throughput)
    if has_iterations:
        ax_iter = axs[subplot_idx]

        for config_name in config_order:
            if config_name not in report:
                continue

            metrics = report[config_name].get("training", {}).get("metrics", {})
            color = colors.get(config_name, "#999999")

            iteration_rates = metrics.get("iteration_rates", [])
            rate_steps = metrics.get("rate_steps", [])

            if iteration_rates and rate_steps and len(iteration_rates) > 0:
                ax_iter.plot(
                    rate_steps,
                    iteration_rates,
                    label=config_name,
                    color=color,
                    linewidth=1.2,
                    alpha=0.9,
                )

        ax_iter.set_ylabel("Training Throughput (it/s)", fontsize=11)
        ax_iter.set_xlabel("Training Step", fontsize=11)
        ax_iter.set_title("Training Throughput Over Time", fontsize=12, fontweight="bold")
        ax_iter.grid(True, alpha=0.3)
        ax_iter.legend(loc="best", framealpha=0.9)

        subplot_idx += 1

    # Subplot 2: Loss convergence
    if has_loss:
        ax_loss = axs[subplot_idx]

        for config_name in config_order:
            if config_name not in report:
                continue

            metrics = report[config_name].get("training", {}).get("metrics", {})
            color = colors.get(config_name, "#999999")

            loss_values = metrics.get("loss_values", [])
            loss_steps = metrics.get("loss_steps", [])

            if loss_values and loss_steps and len(loss_values) > 0:
                ax_loss.plot(
                    loss_steps,
                    loss_values,
                    label=config_name,
                    color=color,
                    linewidth=0.8,
                    alpha=0.75,
                )

        ax_loss.set_ylabel("Loss", fontsize=11)
        ax_loss.set_xlabel("Training Step", fontsize=11)
        ax_loss.set_title("Loss Convergence", fontsize=12, fontweight="bold")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc="best", framealpha=0.9)

        subplot_idx += 1

    # Subplot: Gaussian Count (if available)
    if has_gaussian_count:
        ax_gaussians = axs[subplot_idx]

        for config_name in config_order:
            if config_name not in report:
                continue

            metrics = report[config_name].get("training", {}).get("metrics", {})
            color = colors.get(config_name, "#999999")

            gaussian_count_values = metrics.get("gaussian_count_values", [])
            gaussian_count_steps = metrics.get("gaussian_count_steps", [])

            if gaussian_count_values and gaussian_count_steps and len(gaussian_count_values) > 0:
                ax_gaussians.plot(
                    gaussian_count_steps,
                    gaussian_count_values,
                    label=config_name,
                    color=color,
                    linewidth=1.2,
                    alpha=0.9,
                )

        ax_gaussians.set_ylabel("Number of Gaussians", fontsize=11)
        ax_gaussians.set_xlabel("Training Step", fontsize=11)
        ax_gaussians.set_title("Gaussian Count Over Time", fontsize=12, fontweight="bold")
        ax_gaussians.grid(True, alpha=0.3)
        ax_gaussians.legend(loc="best", framealpha=0.9)

        subplot_idx += 1

    # Subplot 2: PSNR (if available)
    if has_psnr:
        ax_psnr = axs[subplot_idx]

        for config_name in config_order:
            if config_name not in report:
                continue

            metrics = report[config_name].get("training", {}).get("metrics", {})
            psnr_values = metrics.get("psnr_values", [])
            psnr_steps = metrics.get("psnr_steps", [])

            if psnr_values and psnr_steps and len(psnr_values) > 1:
                color = colors.get(config_name, "#999999")
                ax_psnr.scatter(
                    psnr_steps,
                    psnr_values,
                    label=config_name,
                    color=color,
                    s=40,
                    alpha=0.8,
                )
                ax_psnr.plot(psnr_steps, psnr_values, color=color, alpha=0.3, linewidth=0.8)

        ax_psnr.set_ylabel("PSNR (dB)")
        ax_psnr.set_xlabel("Training Step")
        ax_psnr.set_title("PSNR Over Training")
        ax_psnr.grid(True, alpha=0.3)
        ax_psnr.legend(loc="best")

        subplot_idx += 1

    # Subplot 3: SSIM (if available)
    if has_ssim:
        ax_ssim = axs[subplot_idx]

        for config_name in config_order:
            if config_name not in report:
                continue

            metrics = report[config_name].get("training", {}).get("metrics", {})
            ssim_values = metrics.get("ssim_values", [])
            ssim_steps = metrics.get("ssim_steps", [])

            if ssim_values and ssim_steps and len(ssim_values) > 1:
                color = colors.get(config_name, "#999999")
                ax_ssim.scatter(
                    ssim_steps,
                    ssim_values,
                    label=config_name,
                    color=color,
                    s=40,
                    alpha=0.8,
                )
                ax_ssim.plot(ssim_steps, ssim_values, color=color, alpha=0.3, linewidth=0.8)

        ax_ssim.set_ylabel("SSIM")
        ax_ssim.set_xlabel("Training Step")
        ax_ssim.set_title("SSIM Over Training")
        ax_ssim.grid(True, alpha=0.3)
        ax_ssim.legend(loc="best")

        subplot_idx += 1

    # Add main title and subtitle
    scene_title = scene_name.replace("_", " ").title()
    fig.suptitle("Gaussian Splat Training", fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.97, f"Scene: {scene_title}", ha="center", fontsize=12, style="italic")

    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle and subtitle
    output_file = result_path / f"{scene_name}_training.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Training curves saved to: {output_file}")


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
    parser.add_argument(
        "--matrix",
        required=True,
        help="Path to matrix YAML file defining datasets, configs, and runs",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only plot results from an existing run and exit",
    )

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
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(benchmark_log_path),
        ],
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

    # Detect repository paths
    repo_paths = detect_repo_paths(matrix_config, matrix_dir)
    logging.info(f"Detected repository paths: {repo_paths}")

    # Initialize commit manager
    commit_manager = CommitManager(repo_paths)

    # Build run plan - collect all run definitions
    all_run_defs: list[dict[str, Any]] = []
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

        # Extract commit information from opt_config
        commits = get_commits_from_opt_config(opt_config)
        commit_key = get_commit_key(opt_config)

        run_def = {
            "scene_name": scene_name,
            "run_key": run_key,
            "run_dir": run_dir,
            "framework": framework,
            "opt_alias": opt_alias,
            "opt_config_path": opt_config_path,
            "opt_config": opt_config,
            "framework_overrides": framework_overrides,
            "commits": commits,
            "commit_key": commit_key,
        }

        all_run_defs.append(run_def)
        runs_by_scene.setdefault(scene_name, []).append(run_def)

    # Group runs by (framework, commit key) to minimize rebuilds while keeping frameworks separate
    runs_by_commit: dict[tuple, list[dict[str, Any]]] = {}
    for run_def in all_run_defs:
        framework = run_def["framework"]
        commit_key = run_def["commit_key"]
        group_key = (framework, commit_key)
        runs_by_commit.setdefault(group_key, []).append(run_def)

    # Log commit grouping
    logging.info(f"Found {len(runs_by_commit)} unique (framework, commit) combination(s)")
    for group_key, commit_runs in runs_by_commit.items():
        logging.info(f"  Group key {group_key}: {len(commit_runs)} run(s)")

    # Determine scenes to process (in datasets order, but only those with runs)
    scenes = [d["name"] for d in datasets if d.get("name") in runs_by_scene]
    if not scenes:
        parser.error("No runnable scenes found (check 'runs:' vs 'datasets:')")

    # Track training results by scene (accumulated across commit groups)
    all_training_results: dict[str, dict[str, Any]] = {scene: {} for scene in scenes}

    # Process runs grouped by (framework, commit key) to minimize rebuilds
    if not args.plot_only:
        for group_key, commit_runs in runs_by_commit.items():
            logging.info("=" * 60)
            logging.info(f"Processing commit group: {group_key}")
            logging.info("=" * 60)

            # Determine the framework for this commit group (they should all be the same)
            frameworks = {r["framework"] for r in commit_runs}
            if len(frameworks) > 1:
                logging.warning(f"Mixed frameworks in commit group: {frameworks}")

            # Get the first run's framework and commits for checkout
            first_run = commit_runs[0]
            framework = first_run["framework"]
            commits = first_run["commits"]

            # Ensure correct commits are checked out and built
            git_info = commit_manager.ensure_commits(commits, framework)
            logging.info(f"Repository state after checkout: {git_info}")

            # Now run all benchmarks for this commit group
            for run_def in commit_runs:
                scene_name = run_def["scene_name"]
                run_framework = run_def["framework"]
                run_key = run_def["run_key"]
                run_dir = run_def["run_dir"]
                opt_config_path = run_def["opt_config_path"]
                opt_config = run_def["opt_config"]
                framework_overrides = run_def["framework_overrides"]

                logging.info(f"Running benchmark: {scene_name} / {run_key}")

                if run_framework == "fvdb":
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
                    # Add git info to results
                    fvdb_results["repositories"] = git_info
                    all_training_results[scene_name][run_key] = fvdb_results

                elif run_framework == "gsplat":
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
                    if not isinstance(override_cli_args, list) or not all(
                        isinstance(x, str) for x in override_cli_args
                    ):
                        raise ValueError(f"Run overrides for gsplat.cli_args must be a list[str]")

                    gsplat_results = run_gsplat_training(
                        scene_name=scene_name,
                        run_dir=run_dir,
                        matrix_config_path=matrix_path,
                        opt_config_path=merged_opt_path,
                        extra_cli_args=[*opt_cli_args, *override_cli_args],
                    )
                    # Add git info to results
                    gsplat_results["repositories"] = git_info
                    all_training_results[scene_name][run_key] = gsplat_results

                else:
                    raise ValueError(f"Unsupported framework: {run_framework}")

        # Generate per-scene reports after all runs are complete
        for scene_name in scenes:
            training_results = all_training_results.get(scene_name, {})
            if training_results:
                save_report_for_run(
                    scene_name=scene_name,
                    training_results=training_results,
                    output_directory=results_path,
                )

                # Generate training curves for this scene
                save_training_curves(
                    scene_name=scene_name,
                    result_path=results_path,
                    colors=all_config_colors,
                    config_order=config_order,
                )

            logging.info(f"Completed reports for {scene_name}")

    # Generate summary charts if multiple scenes were processed
    if args.plot_only:
        # Warn about missing reports for expected scenes (behavior B)
        for scene_name in scenes:
            report_file = results_path / f"{scene_name}_comparison_report.json"
            if not report_file.exists():
                logging.warning(f"Missing comparison report for expected scene '{scene_name}': {report_file}")
            else:
                # Generate training curves from existing reports
                save_training_curves(
                    scene_name=scene_name,
                    result_path=results_path,
                    colors=all_config_colors,
                    config_order=config_order,
                )

    save_summary_report(scenes, results_path, all_config_colors, config_order)

    logging.info("All benchmarks completed!")


if __name__ == "__main__":
    main()
