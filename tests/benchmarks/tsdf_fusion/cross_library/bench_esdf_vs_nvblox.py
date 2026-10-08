# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
ESDF cross-library comparison: fvdb `Grid.compute_esdf` vs nvblox's
`Mapper.update_esdf`, both fed the same TSDF fused from Mai City LiDAR.

Setup. The paper needs a head-to-head number for the ESDF step itself
(post-TSDF wavefront / block-allocator update). The comparison is:

  1. Load N frames of Mai City seq00 LiDAR sweeps.
  2. In fvdb: call `Grid.integrate_tsdf_from_points_frames` to build
     the TSDF. Then time `Grid.compute_esdf(...)` for M warmup + K
     timed iterations, report median + min.
  3. In nvblox (subprocess, CUDA-12.4 env): feed the same LiDAR
     sweeps through `Mapper.add_depth_frame`, then time
     `Mapper.update_esdf(...)` for the same M + K schedule.
  4. Print a comparison table + write machine-readable JSON.

Notes:

  - The two ESDF implementations compute SAME-shape output (per-voxel
    world-unit signed distance with `|d| <= max_distance`), but on
    different underlying topologies (fvdb: narrow-band index grid;
    nvblox: block-hashed allocator with padded 8^3 blocks). Direct
    numerical comparison of per-voxel values isn't meaningful because
    the voxel sets differ. We only compare timings.
  - Mai City at 20 cm voxels is a sweet-spot: both systems complete
    the TSDF fusion well within GPU memory, and the ESDF step has
    enough voxel volume (~7 M fvdb, ~13 M nvblox per earlier bench
    numbers) for the timing delta to be meaningful.
  - We deliberately keep the frame count small (100 frames default)
    so TSDF fusion finishes in a few seconds and we're timing the
    ESDF step in isolation, not the fusion wall-clock.

Usage:

    cd fvdb-reality-capture/tests/benchmarks/tsdf_fusion/cross_library
    CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \\
    /home/fwilliams/bin/miniconda3/envs/fvdb/bin/python \\
        bench_esdf_vs_nvblox.py \\
        --root ../../data/mai_city/mai_city \\
        --sequence 00 --n-frames 100 \\
        --voxel-size 0.2 --truncation 0.6 --max-distance 2.0 \\
        --json-out ./results/esdf_vs_nvblox_mai_city.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
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
    max_distance: float,
    esdf_warm_calls: int,
) -> dict[str, Any]:
    """Build a TSDF from the scene via `integrate_tsdf_from_points_frames`,
    then time both one-shot `compute_esdf` (cold) and
    `compute_esdf_incremental` (warm) so we can compare apples-to-
    apples against nvblox's stateful `update_esdf`.

    Measurement protocol mirrors the nvblox runner:

      - `esdf_cold_ms`: single call to `compute_esdf` on the final
        TSDF (always "cold" in fvdb's stateless world).
      - `esdf_warm_ms_*`: `esdf_warm_calls` calls to
        `compute_esdf_incremental` feeding back the previous frame's
        output. This is fvdb's equivalent of nvblox's
        "no-dirty-blocks" fast path: idempotent warm-starts on an
        unchanged TSDF.
    """
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
    tsdf_init = torch.zeros(seed_grid.num_voxels, device=device, dtype=torch.float32)
    w_init = torch.zeros(seed_grid.num_voxels, device=device, dtype=torch.float32)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tsdf_grid, tsdf, w = seed_grid.integrate_tsdf_from_points_frames(
        truncation_distance=truncation,
        points_per_frame=points_torch,
        sensor_origins=sensor_origins_t,
        tsdf=tsdf_init,
        weights=w_init,
        carve_free_space=True,
    )
    torch.cuda.synchronize()
    tsdf_s = time.perf_counter() - t0

    n_voxels = int(tsdf_grid.num_voxels)
    n_leaves = int(tsdf_grid.num_leaf_nodes)

    # Cold: a single one-shot `compute_esdf` after TSDF fuse.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    esdf_grid, esdf = tsdf_grid.compute_esdf(
        tsdf, w, truncation_distance=truncation,
        max_distance=max_distance, use_vbm=True,
    )
    torch.cuda.synchronize()
    esdf_cold_ms = (time.perf_counter() - t0) * 1000.0

    esdf_n_voxels = int(esdf_grid.num_voxels)

    # Warm (no dirty mask): repeated `compute_esdf_incremental`
    # calls feeding the previous result back. Monotone-min is
    # idempotent at fixed point; cost is dominated by
    # dilate+merge+inject+seed+short-circuited-sweep = ~20-30 ms.
    warm_samples = []
    prev_grid = esdf_grid
    prev_esdf = esdf
    for _ in range(esdf_warm_calls):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        new_grid, new_esdf = tsdf_grid.compute_esdf_incremental(
            tsdf, w, prev_grid, prev_esdf,
            truncation_distance=truncation,
            max_distance=max_distance, use_vbm=True,
        )
        torch.cuda.synchronize()
        warm_samples.append((time.perf_counter() - t0) * 1000.0)
        prev_grid, prev_esdf = new_grid, new_esdf

    # Warm (dirty-mask short-circuit): an all-false dirty mask
    # triggers the Python-level cache-hit path in
    # `compute_esdf_incremental`, which returns the previous
    # (grid, esdf) directly without entering C++. This is the
    # apples-to-apples equivalent of nvblox's "no dirty blocks"
    # cache hit.
    dirty_all_false = torch.zeros(tsdf_grid.num_voxels, device=device,
                                   dtype=torch.bool)
    # Warmup.
    for _ in range(2):
        _ = tsdf_grid.compute_esdf_incremental(
            tsdf, w, esdf_grid, esdf,
            truncation_distance=truncation,
            max_distance=max_distance, use_vbm=True,
            dirty_mask=dirty_all_false,
        )
    warm_dirty_samples = []
    for _ in range(max(esdf_warm_calls, 10)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = tsdf_grid.compute_esdf_incremental(
            tsdf, w, esdf_grid, esdf,
            truncation_distance=truncation,
            max_distance=max_distance, use_vbm=True,
            dirty_mask=dirty_all_false,
        )
        torch.cuda.synchronize()
        warm_dirty_samples.append((time.perf_counter() - t0) * 1000.0)

    peak_torch_gb = -1.0
    try:
        peak_torch_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass

    return {
        "system": "fvdb",
        "tsdf_fuse_s": tsdf_s,
        "tsdf_n_voxels": n_voxels,
        "tsdf_n_leaves": n_leaves,
        "esdf_n_voxels": esdf_n_voxels,
        "esdf_cold_ms": esdf_cold_ms,
        "esdf_warm_ms_min": min(warm_samples) if warm_samples else -1.0,
        "esdf_warm_ms_median": (statistics.median(warm_samples)
                                 if warm_samples else -1.0),
        "esdf_warm_ms_max": max(warm_samples) if warm_samples else -1.0,
        "esdf_warm_calls": len(warm_samples),
        "esdf_warm_dirty_ms_min": (min(warm_dirty_samples)
                                    if warm_dirty_samples else -1.0),
        "esdf_warm_dirty_ms_median": (statistics.median(warm_dirty_samples)
                                       if warm_dirty_samples else -1.0),
        "peak_torch_gb": peak_torch_gb,
        "voxel_size_m": voxel_size,
        "truncation_m": truncation,
        "max_distance_m": max_distance,
        "n_frames": scene.n_frames,
    }


def _run_nvblox(
    scene,
    voxel_size: float,
    truncation: float,
    max_distance: float,
    esdf_warm_calls: int,
    num_azimuth: int = 1800,
    num_elevation: int = 64,
    vertical_fov_rad: float = 0.4712,
) -> dict[str, Any]:
    """Run nvblox ESDF via the dedicated-env subprocess."""
    runner = str(Path(__file__).resolve().parent / "nvblox_runner.py")

    with tempfile.TemporaryDirectory(prefix="nvblox_esdf_") as tmp:
        # Mirror bench_mai_city's npz packaging so the nvblox subprocess
        # can load the sweeps zero-copy.
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
            "workload": "lidar",
            "voxel_size_m": voxel_size,
            "truncation_distance_m": truncation,
            "lidar_num_azimuth": num_azimuth,
            "lidar_num_elevation": num_elevation,
            "lidar_vertical_fov_rad": vertical_fov_rad,
            "lidar_min_valid_range_m": 1.0,
            "lidar_sweeps_npz": npz_path,
            "warmup_frames": 2,
            "with_esdf": True,
            "esdf_warm_calls": esdf_warm_calls,
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
    result["max_distance_m"] = max_distance
    return result


def _run_one_config(
    scene,
    voxel_size: float,
    truncation: float,
    max_distance: float,
    esdf_warm_calls: int,
    skip_nvblox: bool,
) -> dict[str, Any]:
    """Run one (voxel_size, truncation, max_distance) config across
    both systems. Isolated per-config so OOM in one system doesn't
    cascade: wraps fvdb in try/except for
    `torch.cuda.OutOfMemoryError`, and frees intermediate state
    between configs so fine-voxel runs don't inherit fragmentation
    from coarse ones."""
    print(f"\n[config] voxel={voxel_size}  trunc={truncation}  "
          f"max_dist={max_distance}")
    result: dict[str, Any] = {
        "voxel_size_m": voxel_size,
        "truncation_m": truncation,
        "max_distance_m": max_distance,
    }

    # fvdb run (in-process). Catch OOM; reset allocator state.
    torch.cuda.reset_peak_memory_stats()
    try:
        result["fvdb"] = _run_fvdb(
            scene, voxel_size, truncation, max_distance, esdf_warm_calls,
        )
        result["fvdb"]["ok"] = True
        r = result["fvdb"]
        print(f"  fvdb: TSDF {r['tsdf_n_voxels']:,} vx, "
              f"ESDF {r['esdf_n_voxels']:,} vx, "
              f"cold {r['esdf_cold_ms']:.1f} ms, "
              f"warm {r['esdf_warm_ms_median']:.1f} ms, "
              f"peak_gb {r['peak_torch_gb']:.2f}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"  fvdb: OOM at voxel_size={voxel_size} - {e}")
        result["fvdb"] = {"ok": False, "failure": f"OOM: {e}"}
        # Drain torch's allocator so later configs don't inherit
        # fragmentation; gc.collect to release any stale Python refs.
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:  # noqa: BLE001
        print(f"  fvdb: failed ({type(e).__name__}: {e})")
        result["fvdb"] = {"ok": False,
                          "failure": f"{type(e).__name__}: {e}"}

    torch.cuda.empty_cache()
    gc.collect()

    # nvblox run (subprocess, so its own OOM doesn't affect us).
    if skip_nvblox:
        result["nvblox"] = {"ok": False, "failure": "skipped"}
    else:
        try:
            nvblox_result = _run_nvblox(
                scene, voxel_size, truncation, max_distance, esdf_warm_calls,
            )
            result["nvblox"] = nvblox_result
            if nvblox_result.get("ok", False):
                print(f"  nvblox: approx_vx {nvblox_result['n_voxels']:,}, "
                      f"cold {nvblox_result['esdf_cold_ms']:.1f} ms, "
                      f"warm {nvblox_result['esdf_warm_ms_median']:.2f} ms, "
                      f"gpu_gb {nvblox_result.get('gpu_used_gb', -1):.2f}")
            else:
                print(f"  nvblox: FAILED {nvblox_result.get('failure', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  nvblox: driver error ({type(e).__name__}: {e})")
            result["nvblox"] = {"ok": False,
                                "failure": f"driver: {type(e).__name__}: {e}"}

    return result


def _format_scale_table(results: list[dict[str, Any]]) -> str:
    """Render the scale-ceiling summary table as a printable string.

    `cold_x` column: how many times slower fvdb's cold ESDF is than
    nvblox's. Values > 1 mean nvblox wins; values < 1 mean fvdb
    wins; `∞` means nvblox OOM'd."""
    lines = []
    lines.append("=== ESDF Scale Ceiling (Mai City) ===")
    lines.append(f"{'voxel':>6}  {'fvdb_ESDF_vx':>12}  {'fvdb_cold_ms':>13}  "
                 f"{'fvdb_warm_ms':>13}  {'fvdb_dirty_ms':>14}  {'fvdb_gb':>7}  "
                 f"{'nvblox_cold_ms':>14}  {'nvblox_gb':>9}  {'cold_x':>8}")
    for r in results:
        vs = r["voxel_size_m"]
        fv = r.get("fvdb", {})
        nv = r.get("nvblox", {})
        if fv.get("ok"):
            dirty_ms = fv.get("esdf_warm_dirty_ms_median", -1.0)
            dirty_str = f"{dirty_ms * 1000:14.0f}μs" if dirty_ms > 0 and dirty_ms < 1.0 \
                else f"{dirty_ms:>14.2f}"
            fv_str = (f"{fv['esdf_n_voxels']:>12,}  "
                      f"{fv['esdf_cold_ms']:>13.1f}  "
                      f"{fv['esdf_warm_ms_median']:>13.1f}  "
                      f"{dirty_str}  "
                      f"{fv['peak_torch_gb']:>7.2f}")
        else:
            fv_str = f"{'OOM':>12}  {'--':>13}  {'--':>13}  {'--':>14}  {'--':>7}"
        if nv.get("ok"):
            nv_cold = nv.get("esdf_cold_ms", -1)
            nv_gb = nv.get("gpu_used_gb", -1)
            if fv.get("ok") and nv_cold > 0 and fv["esdf_cold_ms"] > 0:
                # cold_x = fvdb_cold / nvblox_cold; >1 means nvblox wins
                cold_x = f"{fv['esdf_cold_ms'] / nv_cold:.2f}x"
            else:
                cold_x = "--"
            nv_str = f"{nv_cold:>14.1f}  {nv_gb:>9.2f}  {cold_x:>8}"
        else:
            fail = nv.get("failure", "")
            fail_str = str(fail)
            looks_oom = ("OOM" in fail_str.upper()
                         or "OutOfMemory" in fail_str
                         or "cudaMalloc" in fail_str
                         or "bad_alloc" in fail_str
                         or "out of memory" in fail_str.lower())
            if looks_oom:
                nv_str = f"{'OOM':>14}  {'--':>9}  {'fvdb∞':>8}"
            elif "skipped" in fail_str.lower():
                nv_str = f"{'skip':>14}  {'--':>9}  {'--':>8}"
            else:
                nv_str = f"{'FAIL':>14}  {'--':>9}  {'--':>8}"
        lines.append(f"{vs:>6.3f}  {fv_str}  {nv_str}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path to Mai City root (contains bin/sequences/.../velodyne)")
    ap.add_argument("--sequence", default="00")
    ap.add_argument("--n-frames", type=int, default=100)
    # Single-config mode (backwards compatible):
    ap.add_argument("--voxel-size", type=float, default=None)
    ap.add_argument("--truncation", type=float, default=None)
    ap.add_argument("--max-distance", type=float, default=None)
    # Scale-sweep mode (preferred for the paper's scale-ceiling figure):
    ap.add_argument("--voxel-sizes-m", type=float, nargs="+", default=None,
                    help="Sweep over a list of voxel sizes. Truncation and "
                         "max_distance default to fixed multiples (3x and "
                         "10x) of each voxel size.")
    ap.add_argument("--trunc-voxel-multiplier", type=float, default=3.0,
                    help="In sweep mode: truncation = this × voxel_size")
    ap.add_argument("--max-distance-voxel-multiplier", type=float, default=10.0,
                    help="In sweep mode: max_distance = this × voxel_size")
    ap.add_argument("--esdf-warm-calls", type=int, default=5,
                    help="Number of warm (idempotent) ESDF calls to time "
                         "after the cold (first) call.")
    ap.add_argument("--skip-nvblox", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    # Build the list of configs to run.
    if args.voxel_sizes_m is not None:
        configs = []
        for vs in args.voxel_sizes_m:
            configs.append({
                "voxel_size": float(vs),
                "truncation":  float(vs) * args.trunc_voxel_multiplier,
                "max_distance": float(vs) * args.max_distance_voxel_multiplier,
            })
    else:
        # Single-config back-compat.
        if args.voxel_size is None or args.truncation is None or args.max_distance is None:
            raise SystemExit("Either --voxel-sizes-m or all of --voxel-size "
                             "/ --truncation / --max-distance must be set")
        configs = [{
            "voxel_size": args.voxel_size,
            "truncation": args.truncation,
            "max_distance": args.max_distance,
        }]

    print(f"[load] Mai City seq={args.sequence} n_frames={args.n_frames}")
    scene = load_mai_city_scene(
        root_dir=args.root, sequence=args.sequence,
        max_frames=args.n_frames,
    )
    print(f"[load] done. {scene.n_frames} frames, "
          f"{scene.total_points / 1e6:.2f} M points total")

    results: list[dict[str, Any]] = []
    for cfg in configs:
        results.append(_run_one_config(
            scene,
            voxel_size=cfg["voxel_size"],
            truncation=cfg["truncation"],
            max_distance=cfg["max_distance"],
            esdf_warm_calls=args.esdf_warm_calls,
            skip_nvblox=args.skip_nvblox,
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
                    "esdf_warm_calls": args.esdf_warm_calls,
                    "trunc_voxel_multiplier": args.trunc_voxel_multiplier,
                    "max_distance_voxel_multiplier": args.max_distance_voxel_multiplier,
                },
                "results": results,
            }, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
