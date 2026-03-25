#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Convert comparative benchmark summary_data.json into the JSON formats expected by
benchmark-action/github-action-benchmark.

Produces two files:
  benchmark_quality.json   -- customBiggerIsBetter  (PSNR, SSIM)
  benchmark_performance.json -- customSmallerIsBetter (training_time, peak_gpu_memory_gb)

Each entry is named "<scene>/<config> - <metric>" so it gets its own trend line
on the gh-pages dashboard.
"""

import argparse
import json
import math
import pathlib
import sys

QUALITY_METRICS = {
    "PSNR": "dB",
    "SSIM": "",
}

PERFORMANCE_METRICS = {
    "training_time": "seconds",
    "peak_gpu_memory_gb": "GB",
}

FVDB_CONFIGS = ("fvdb_default", "fvdb_mcmc")


def load_summary(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_entries(per_scene: dict, metric_map: dict, configs: tuple[str, ...]) -> list[dict]:
    entries: list[dict] = []
    for scene, scene_data in sorted(per_scene.items()):
        for config in configs:
            if config not in scene_data:
                continue
            metrics = scene_data[config]
            for metric_key, unit in metric_map.items():
                value = metrics.get(metric_key)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                entries.append(
                    {
                        "name": f"{scene}/{config} - {metric_key}",
                        "unit": unit,
                        "value": round(value, 4),
                    }
                )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "summary_json",
        type=pathlib.Path,
        help="Path to summary_data.json produced by comparison_benchmark.py",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory for output files (default: same as summary_json)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(FVDB_CONFIGS),
        help="Config names to extract (default: fvdb_default fvdb_mcmc)",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary_json)
    per_scene = summary.get("per_scene", {})
    if not per_scene:
        print("ERROR: summary_data.json contains no per_scene data", file=sys.stderr)
        return 1

    out_dir = args.output_dir or args.summary_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = tuple(args.configs)

    quality = build_entries(per_scene, QUALITY_METRICS, configs)
    performance = build_entries(per_scene, PERFORMANCE_METRICS, configs)

    quality_path = out_dir / "benchmark_quality.json"
    performance_path = out_dir / "benchmark_performance.json"

    with open(quality_path, "w") as f:
        json.dump(quality, f, indent=2)
    print(f"Wrote {len(quality)} quality entries to {quality_path}")

    with open(performance_path, "w") as f:
        json.dump(performance, f, indent=2)
    print(f"Wrote {len(performance)} performance entries to {performance_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
