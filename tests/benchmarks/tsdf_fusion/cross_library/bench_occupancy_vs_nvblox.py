# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Occupancy mapping cross-library comparison on Mai City LiDAR.

fvdb `Grid.integrate_occupancy_from_points_frames` vs nvblox
`ProjectiveIntegratorType.OCCUPANCY`. Both systems consume the same N
Mai City LiDAR sweeps and produce a per-voxel log-odds occupancy
representation; this driver measures end-to-end fusion time
(TSDF-equivalent role in nvblox's integrator) and scale ceiling.

This is the paper's fifth application of the topology-op vocabulary,
and closes the nvblox feature-parity gap. See session notes for the
primitive-matrix update.

Scope: Mai City LiDAR (the only sensor modality with a working
nvblox-side LiDAR occupancy path). 100 frames default, scale-sweep
mode matching `bench_esdf_vs_nvblox.py`.

Usage:

    cd fvdb-reality-capture/tests/benchmarks/tsdf_fusion/cross_library
    CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \\
    /home/fwilliams/bin/miniconda3/envs/fvdb/bin/python \\
        bench_occupancy_vs_nvblox.py \\
        --root ../../data/mai_city/mai_city \\
        --sequence 00 --n-frames 100 \\
        --voxel-sizes-m 0.2 0.1 0.05 0.03 0.02 \\
        --trunc-voxel-multiplier 3 \\
        --json-out ./results/scale_ceiling_mai_city.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

import fvdb

from mai_city_loader import load_mai_city_scene


NVBLOX_ENV_PYTHON = "/home/fwilliams/bin/miniconda3/envs/nvblox/bin/python"


def _run_fvdb(
    scene,
    voxel_size: float,
    truncation: float,
) -> dict[str, Any]:
    """Build a log-odds occupancy map from the scene via
    `integrate_occupancy_from_points_frames`. Measures fusion wall-
    clock (per-frame amortised) and peak GPU memory."""
    device = "cuda"

    points_torch = [
        torch.from_numpy(p).to(device=device, dtype=torch.float32)
        for p in scene.points_per_frame
    ]
    sensor_origins_t = torch.from_numpy(scene.sensor_origins).to(
        device=device, dtype=torch.float32)

    seed_grid = fvdb.Grid.from_dense(
        dense_dims=[1, 1, 1], ijk_min=[0, 0, 0],
        voxel_size=voxel_size, origin=[0, 0, 0], device=device,
    )
    log_odds_init = torch.zeros(
        seed_grid.num_voxels, device=device, dtype=torch.float32)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_grid, out_log_odds = seed_grid.integrate_occupancy_from_points_frames(
        truncation_distance=truncation,
        points_per_frame=points_torch,
        sensor_origins=sensor_origins_t,
        log_odds=log_odds_init,
    )
    torch.cuda.synchronize()
    fuse_s = time.perf_counter() - t0
    ms_per_f = fuse_s * 1000 / scene.n_frames

    peak_torch_gb = -1.0
    try:
        peak_torch_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass

    # A bit of distributional info on the final log-odds: how many
    # voxels are "confidently occupied" (log_odds > 2) vs
    # "confidently free" (log_odds < -2). Useful sanity for the
    # paper's "fvdb produces an actually-usable occupancy volume".
    lo = out_log_odds
    n_occupied = int((lo > 2.0).sum().item())
    n_free     = int((lo < -2.0).sum().item())
    n_unknown  = int((lo.abs() <= 2.0).sum().item())

    return {
        "system": "fvdb",
        "fuse_s": fuse_s,
        "ms_per_f": ms_per_f,
        "n_voxels": int(out_grid.num_voxels),
        "n_leaves": int(out_grid.num_leaf_nodes),
        "n_occupied_voxels": n_occupied,
        "n_free_voxels": n_free,
        "n_unknown_voxels": n_unknown,
        "log_odds_min": float(lo.min().item()),
        "log_odds_max": float(lo.max().item()),
        "peak_torch_gb": peak_torch_gb,
        "voxel_size_m": voxel_size,
        "truncation_m": truncation,
        "n_frames": scene.n_frames,
    }


def _run_nvblox(
    scene,
    voxel_size: float,
    truncation: float,
    num_azimuth: int = 1800,
    num_elevation: int = 64,
    vertical_fov_rad: float = 0.4712,
) -> dict[str, Any]:
    """Run nvblox occupancy integrator (LiDAR) via the dedicated-env
    subprocess."""
    runner = str(Path(__file__).resolve().parent / "nvblox_runner.py")

    with tempfile.TemporaryDirectory(prefix="nvblox_occ_") as tmp:
        pts_concat = np.concatenate(scene.points_per_frame, axis=0).astype(np.float32)
        offsets = np.zeros(scene.n_frames + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([p.shape[0] for p in scene.points_per_frame])

        npz_path = os.path.join(tmp, "mai_city_sweeps.npz")
        np.savez(
            npz_path,
            points_per_frame_concat=pts_concat,
            points_per_frame_offsets=offsets,
            sensor_origins=scene.sensor_origins.astype(np.float32),
            cam_to_world=scene.cam_to_world.astype(np.float32),
        )

        spec = {
            "workload": "lidar_occupancy",
            "voxel_size_m": voxel_size,
            "truncation_distance_m": truncation,
            "lidar_num_azimuth": num_azimuth,
            "lidar_num_elevation": num_elevation,
            "lidar_vertical_fov_rad": vertical_fov_rad,
            "lidar_min_valid_range_m": 1.0,
            "lidar_sweeps_npz": npz_path,
            "warmup_frames": 2,
        }
        spec_path = os.path.join(tmp, "spec.json")
        out_path  = os.path.join(tmp, "result.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "1")
        proc = subprocess.run(
            [NVBLOX_ENV_PYTHON, runner, "--spec", spec_path, "--output", out_path],
            env=env, capture_output=True, text=True, timeout=3600,
        )
        if not os.path.exists(out_path):
            return {"system": "nvblox", "ok": False,
                    "failure": f"no output. stderr tail: {proc.stderr[-500:]!r}"}
        with open(out_path, "r") as f:
            result = json.load(f)

    result["system"] = "nvblox"
    result["voxel_size_m"] = voxel_size
    result["truncation_m"] = truncation
    return result


def _run_one_config(
    scene, voxel_size: float, truncation: float, skip_nvblox: bool,
) -> dict[str, Any]:
    print(f"\n[config] voxel={voxel_size}  trunc={truncation}")
    result: dict[str, Any] = {
        "voxel_size_m": voxel_size,
        "truncation_m": truncation,
    }
    try:
        result["fvdb"] = _run_fvdb(scene, voxel_size, truncation)
        result["fvdb"]["ok"] = True
        r = result["fvdb"]
        print(f"  fvdb: fuse {r['fuse_s']:.2f} s  ({r['ms_per_f']:.1f} ms/f)  "
              f"voxels {r['n_voxels']:,}  leaves {r['n_leaves']:,}  "
              f"peak_gb {r['peak_torch_gb']:.2f}  "
              f"occ {r['n_occupied_voxels']:,} / "
              f"free {r['n_free_voxels']:,} / unknown {r['n_unknown_voxels']:,}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"  fvdb: OOM - {e}")
        result["fvdb"] = {"ok": False, "failure": f"OOM: {e}"}
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:  # noqa: BLE001
        print(f"  fvdb: failed ({type(e).__name__}: {e})")
        result["fvdb"] = {"ok": False, "failure": f"{type(e).__name__}: {e}"}

    torch.cuda.empty_cache()
    gc.collect()

    if skip_nvblox:
        result["nvblox"] = {"ok": False, "failure": "skipped"}
    else:
        try:
            nvblox_result = _run_nvblox(scene, voxel_size, truncation)
            result["nvblox"] = nvblox_result
            if nvblox_result.get("ok", False):
                print(f"  nvblox: fuse {nvblox_result['wall_s']:.2f} s  "
                      f"({nvblox_result['ms_per_f']:.1f} ms/f)  "
                      f"approx_vx {nvblox_result['n_voxels']:,}  "
                      f"blocks {nvblox_result['n_blocks']:,}  "
                      f"gpu_gb {nvblox_result.get('gpu_used_gb', -1):.2f}")
            else:
                print(f"  nvblox: FAILED {nvblox_result.get('failure', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  nvblox: driver error ({type(e).__name__}: {e})")
            result["nvblox"] = {"ok": False,
                                "failure": f"driver: {type(e).__name__}: {e}"}

    return result


def _format_scale_table(results: list[dict[str, Any]]) -> str:
    """Render the scale-ceiling summary table. `ratio` = fvdb_ms_per_f
    / nvblox_ms_per_f; >1 means nvblox wins, <1 means fvdb wins,
    `fvdb∞` means nvblox OOM/FAILed while fvdb succeeded."""
    lines = []
    lines.append("=== Occupancy Scale Ceiling (Mai City) ===")
    lines.append(f"{'voxel':>6}  {'fvdb_voxels':>12}  {'fvdb_ms/f':>10}  "
                 f"{'fvdb_gb':>7}  {'nvblox_ms/f':>12}  {'nvblox_gb':>9}  "
                 f"{'ratio':>8}")
    for r in results:
        vs = r["voxel_size_m"]
        fv = r.get("fvdb", {})
        nv = r.get("nvblox", {})
        if fv.get("ok"):
            fv_str = (f"{fv['n_voxels']:>12,}  "
                      f"{fv['ms_per_f']:>10.1f}  "
                      f"{fv['peak_torch_gb']:>7.2f}")
        else:
            fv_str = f"{'OOM':>12}  {'--':>10}  {'--':>7}"
        if nv.get("ok"):
            nv_ms = nv.get("ms_per_f", -1)
            nv_gb = nv.get("gpu_used_gb", -1)
            if fv.get("ok") and nv_ms > 0 and fv["ms_per_f"] > 0:
                ratio = f"{fv['ms_per_f'] / nv_ms:.2f}x"
            else:
                ratio = "--"
            nv_str = f"{nv_ms:>12.1f}  {nv_gb:>9.2f}  {ratio:>8}"
        else:
            fail = str(nv.get("failure", ""))
            looks_oom = ("OOM" in fail.upper()
                         or "OutOfMemory" in fail
                         or "cudaMalloc" in fail
                         or "out of memory" in fail.lower())
            if looks_oom:
                nv_str = f"{'OOM':>12}  {'--':>9}  {'fvdb∞':>8}"
            elif "skipped" in fail.lower():
                nv_str = f"{'skip':>12}  {'--':>9}  {'--':>8}"
            else:
                nv_str = f"{'FAIL':>12}  {'--':>9}  {'--':>8}"
        lines.append(f"{vs:>6.3f}  {fv_str}  {nv_str}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path to Mai City root (contains bin/sequences/.../velodyne)")
    ap.add_argument("--sequence", default="00")
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--voxel-sizes-m", type=float, nargs="+", required=True)
    ap.add_argument("--trunc-voxel-multiplier", type=float, default=3.0)
    ap.add_argument("--skip-nvblox", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    configs = [{
        "voxel_size": float(vs),
        "truncation":  float(vs) * args.trunc_voxel_multiplier,
    } for vs in args.voxel_sizes_m]

    print(f"[load] Mai City seq={args.sequence}  n_frames={args.n_frames}")
    scene = load_mai_city_scene(
        root_dir=args.root, sequence=args.sequence,
        max_frames=args.n_frames,
    )
    print(f"[load] done. {scene.n_frames} frames, "
          f"{scene.total_points / 1e6:.2f} M points total")

    results: list[dict[str, Any]] = []
    for cfg in configs:
        results.append(_run_one_config(
            scene, voxel_size=cfg["voxel_size"],
            truncation=cfg["truncation"], skip_nvblox=args.skip_nvblox,
        ))

    print("")
    print(_format_scale_table(results))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as f:
            json.dump({
                "config": {
                    "sequence": args.sequence,
                    "n_frames": args.n_frames,
                    "trunc_voxel_multiplier": args.trunc_voxel_multiplier,
                },
                "results": results,
            }, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
