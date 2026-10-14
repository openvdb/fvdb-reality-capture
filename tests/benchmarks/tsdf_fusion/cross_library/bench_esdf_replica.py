# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
ESDF cross-library comparison on Replica depth data (NICE-SLAM format).

Direct depth-TSDF counterpart to `bench_esdf_vs_nvblox.py` (which is
Mai City LiDAR-only). Both systems integrate the same N Replica depth
frames into a TSDF, then the ESDF compute step is timed in isolation:

  - fvdb: `Grid.integrate_tsdf_frames(..)` -> `Grid.compute_esdf(..)`
          (cold) / `Grid.compute_esdf_incremental(..)` (warm).
  - nvblox: subprocess `run_depth` in the CUDA-12.4 env -> time
            `Mapper.update_esdf(..)` cold + warm.

Scope: single Replica scene (default room0), 200-frame stride-10
subset matching the existing `bench_open3d_vs_fvdb.py` convention.
The timings are the ESDF step cost; TSDF fusion is reported but
not the headline number.

Typical invocation for the paper's Replica ESDF row:

    CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \\
    /home/fwilliams/bin/miniconda3/envs/fvdb/bin/python \\
        bench_esdf_replica.py \\
        --scene ../../data/Replica/room0 \\
        --n-frames 200 --stride 10 \\
        --voxel-sizes-m 0.02 0.01 0.005 0.003 \\
        --trunc-voxel-multiplier 3 --max-distance-voxel-multiplier 10 \\
        --json-out ./results/esdf_replica_room0.json
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

from replica_loader import load_replica_scene


NVBLOX_ENV_PYTHON = "/home/fwilliams/bin/miniconda3/envs/nvblox/bin/python"


def _build_fvdb_seed_grid(voxel_size: float, device: str = "cuda"):
    """Metadata-only seed grid for fvdb's `integrate_tsdf_frames` —
    1 voxel so it establishes the voxel_size / origin convention, and
    the integrator grows it via shell dilation per frame."""
    g = fvdb.Grid.from_dense(
        dense_dims=[1, 1, 1], ijk_min=[0, 0, 0],
        voxel_size=voxel_size, origin=[0, 0, 0], device=device,
    )
    tsdf = torch.zeros(g.num_voxels, device=device, dtype=torch.float32)
    w    = torch.zeros(g.num_voxels, device=device, dtype=torch.float32)
    return g, tsdf, w


def _run_fvdb(
    scene,
    voxel_size: float,
    truncation: float,
    max_distance: float,
    esdf_warm_calls: int,
) -> dict[str, Any]:
    """Build a TSDF from Replica depth frames, then time compute_esdf
    (cold) + compute_esdf_incremental (warm). Mirrors the Mai City
    driver's fvdb path."""
    device = "cuda"

    # Push the depth + pose arrays to GPU. Replica's depth_images at
    # 1200x680 x 200 frames = ~650 MB as fp32 — fits comfortably.
    K_t = torch.from_numpy(scene.K).to(device=device, dtype=torch.float32)
    cam_to_world = torch.from_numpy(scene.cam_to_world).to(
        device=device, dtype=torch.float32)
    depth_images = torch.from_numpy(scene.depth_images).to(
        device=device, dtype=torch.float32)
    N = scene.n_frames
    K_per_frame = K_t.unsqueeze(0).expand(N, 3, 3).contiguous()

    seed_grid, tsdf_init, w_init = _build_fvdb_seed_grid(voxel_size, device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tsdf_grid, tsdf, w = seed_grid.integrate_tsdf_frames(
        truncation_distance=truncation,
        projection_matrices=K_per_frame,
        cam_to_world_matrices=cam_to_world,
        tsdf=tsdf_init,
        weights=w_init,
        depth_images=depth_images,
    )
    torch.cuda.synchronize()
    tsdf_s = time.perf_counter() - t0

    n_voxels = int(tsdf_grid.num_voxels)
    n_leaves = int(tsdf_grid.num_leaf_nodes)

    # Cold one-shot.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    esdf_grid, esdf = tsdf_grid.compute_esdf(
        tsdf, w,
        truncation_distance=truncation,
        max_distance=max_distance,
        use_vbm=True,
    )
    torch.cuda.synchronize()
    esdf_cold_ms = (time.perf_counter() - t0) * 1000.0
    esdf_n_voxels = int(esdf_grid.num_voxels)

    # Warm incremental calls.
    warm_samples = []
    prev_grid, prev_esdf = esdf_grid, esdf
    for _ in range(esdf_warm_calls):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        new_grid, new_esdf = tsdf_grid.compute_esdf_incremental(
            tsdf, w, prev_grid, prev_esdf,
            truncation_distance=truncation,
            max_distance=max_distance,
            use_vbm=True,
        )
        torch.cuda.synchronize()
        warm_samples.append((time.perf_counter() - t0) * 1000.0)
        prev_grid, prev_esdf = new_grid, new_esdf

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
) -> dict[str, Any]:
    """Run nvblox depth-TSDF + ESDF via the subprocess runner."""
    runner = str(Path(__file__).resolve().parent / "nvblox_runner.py")

    with tempfile.TemporaryDirectory(prefix="nvblox_esdf_replica_") as tmp:
        depth_npz = os.path.join(tmp, "replica_depth.npz")
        np.savez(
            depth_npz,
            depth_images=scene.depth_images.astype(np.float32),
            cam_to_world=scene.cam_to_world.astype(np.float32),
            K=scene.K.astype(np.float32),
        )

        spec = {
            "workload": "depth",
            "voxel_size_m": voxel_size,
            "truncation_distance_m": truncation,
            "depth_npz": depth_npz,
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
    scene, voxel_size: float, truncation: float, max_distance: float,
    esdf_warm_calls: int, skip_nvblox: bool,
) -> dict[str, Any]:
    print(f"\n[config] voxel={voxel_size}  trunc={truncation}  max_dist={max_distance}")
    result: dict[str, Any] = {
        "voxel_size_m": voxel_size,
        "truncation_m": truncation,
        "max_distance_m": max_distance,
    }

    torch.cuda.reset_peak_memory_stats()
    try:
        result["fvdb"] = _run_fvdb(scene, voxel_size, truncation,
                                   max_distance, esdf_warm_calls)
        result["fvdb"]["ok"] = True
        r = result["fvdb"]
        print(f"  fvdb: TSDF {r['tsdf_n_voxels']:,} vx, "
              f"ESDF {r['esdf_n_voxels']:,} vx, "
              f"cold {r['esdf_cold_ms']:.1f} ms, "
              f"warm {r['esdf_warm_ms_median']:.1f} ms, "
              f"peak_gb {r['peak_torch_gb']:.2f}")
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
            nvblox_result = _run_nvblox(scene, voxel_size, truncation,
                                        max_distance, esdf_warm_calls)
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

    `cold_x` > 1 means nvblox wins; < 1 means fvdb wins; `fvdb∞`
    means nvblox OOM'd while fvdb ran."""
    lines = []
    lines.append("=== ESDF Scale Ceiling (Replica depth) ===")
    lines.append(f"{'voxel':>6}  {'fvdb_ESDF_vx':>12}  {'fvdb_cold_ms':>13}  "
                 f"{'fvdb_warm_ms':>13}  {'fvdb_gb':>7}  "
                 f"{'nvblox_cold_ms':>14}  {'nvblox_gb':>9}  {'cold_x':>8}")
    for r in results:
        vs = r["voxel_size_m"]
        fv = r.get("fvdb", {})
        nv = r.get("nvblox", {})
        if fv.get("ok"):
            fv_str = (f"{fv['esdf_n_voxels']:>12,}  "
                      f"{fv['esdf_cold_ms']:>13.1f}  "
                      f"{fv['esdf_warm_ms_median']:>13.1f}  "
                      f"{fv['peak_torch_gb']:>7.2f}")
        else:
            fv_str = f"{'OOM':>12}  {'--':>13}  {'--':>13}  {'--':>7}"
        if nv.get("ok"):
            nv_cold = nv.get("esdf_cold_ms", -1)
            nv_gb = nv.get("gpu_used_gb", -1)
            if fv.get("ok") and nv_cold > 0 and fv["esdf_cold_ms"] > 0:
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
    ap.add_argument("--scene", required=True,
                    help="Path to a Replica scene directory (e.g., .../Replica/room0)")
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--voxel-sizes-m", type=float, nargs="+", required=True)
    ap.add_argument("--trunc-voxel-multiplier", type=float, default=3.0)
    ap.add_argument("--max-distance-voxel-multiplier", type=float, default=10.0)
    ap.add_argument("--esdf-warm-calls", type=int, default=5)
    ap.add_argument("--skip-nvblox", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    configs = [{
        "voxel_size": float(vs),
        "truncation":  float(vs) * args.trunc_voxel_multiplier,
        "max_distance": float(vs) * args.max_distance_voxel_multiplier,
    } for vs in args.voxel_sizes_m]

    print(f"[load] Replica scene={args.scene}  n_frames={args.n_frames}  "
          f"stride={args.stride}")
    scene = load_replica_scene(
        scene_dir=args.scene,
        max_frames=args.n_frames,
        stride=args.stride,
    )
    print(f"[load] done. {scene.n_frames} frames @ "
          f"{scene.depth_images.shape[1]}x{scene.depth_images.shape[2]} depth")

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
                    "scene": args.scene,
                    "n_frames": args.n_frames,
                    "stride": args.stride,
                    "esdf_warm_calls": args.esdf_warm_calls,
                    "trunc_voxel_multiplier": args.trunc_voxel_multiplier,
                    "max_distance_voxel_multiplier":
                        args.max_distance_voxel_multiplier,
                },
                "results": results,
            }, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
