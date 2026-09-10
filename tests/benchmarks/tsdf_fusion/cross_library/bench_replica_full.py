# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Multi-scene Replica TSDF + ESDF benchmark at full frame rate.

Counterpart to `bench_esdf_kitti.py`: sweeps over multiple Replica
scenes (default office0/room0/room1) and multiple voxel sizes,
running fvdb's depth-TSDF + ESDF and nvblox's equivalents on each
config. Reuses `_run_nvblox` and `_run_one_config` from
`bench_esdf_replica.py` (which already returns TSDF + ESDF stats
per config), but replaces the in-process `_run_fvdb` with a
chunked variant that streams frames in batches to bound peak GPU
memory at 2000-frame full-rate (where the unchunked version's
~6.5 GB depth-image upload would already crowd the integrator).

Usage:

    cd fvdb-reality-capture/tests/benchmarks/tsdf_fusion/cross_library
    CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \\
    /home/fwilliams/bin/miniconda3/envs/fvdb/bin/python -u \\
        bench_replica_full.py \\
        --root .../data/Replica \\
        --scenes office0 room0 room1 \\
        --n-frames 2000 \\
        --voxel-sizes-m 0.02 0.01 0.005 0.003 \\
        --json-out ./results/replica_tsdf_esdf.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

import fvdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replica_loader import load_replica_scene  # noqa: E402
import bench_esdf_replica as upstream  # noqa: E402


def _build_seed_grid(voxel_size: float, device: str = "cuda"):
    g = fvdb.Grid.from_dense(
        dense_dims=[1, 1, 1], ijk_min=[0, 0, 0],
        voxel_size=voxel_size, origin=[0, 0, 0], device=device,
    )
    tsdf = torch.zeros(g.num_voxels, device=device, dtype=torch.float32)
    w    = torch.zeros(g.num_voxels, device=device, dtype=torch.float32)
    return g, tsdf, w


def _run_fvdb_chunked_replica(
    scene,
    voxel_size: float,
    truncation: float,
    max_distance: float,
    esdf_warm_calls: int,
    chunk_frames: int = 200,
) -> Dict[str, Any]:
    """Chunked depth-TSDF + ESDF on a Replica scene.

    Equivalent to upstream.bench_esdf_replica._run_fvdb but processes
    depth images in chunks so peak GPU memory does not include the
    full N x H x W x 4 B depth tensor (6.5 GB at 2000 frames x
    1200x680). Output schema matches the upstream function for
    drop-in compatibility with `_run_one_config`'s formatter.
    """
    device = "cuda"
    K_t = torch.from_numpy(scene.K).to(device=device, dtype=torch.float32)
    cam_to_world_full = torch.from_numpy(scene.cam_to_world).to(
        device=device, dtype=torch.float32)
    N = scene.n_frames

    g, tsdf, w = _build_seed_grid(voxel_size, device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for chunk_start in range(0, N, chunk_frames):
        chunk_end = min(chunk_start + chunk_frames, N)
        n_chunk = chunk_end - chunk_start
        chunk_depth = torch.from_numpy(
            scene.depth_images[chunk_start:chunk_end]
        ).to(device=device, dtype=torch.float32)
        chunk_c2w = cam_to_world_full[chunk_start:chunk_end]
        K_per_frame = K_t.unsqueeze(0).expand(n_chunk, 3, 3).contiguous()
        g, tsdf, w = g.integrate_tsdf_frames(
            truncation_distance=truncation,
            projection_matrices=K_per_frame,
            cam_to_world_matrices=chunk_c2w,
            tsdf=tsdf,
            weights=w,
            depth_images=chunk_depth,
        )
        del chunk_depth
    torch.cuda.synchronize()
    tsdf_s = time.perf_counter() - t0

    n_voxels = int(g.num_voxels)
    n_leaves = int(g.num_leaf_nodes)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    esdf_grid, esdf = g.compute_esdf(
        tsdf, w,
        truncation_distance=truncation,
        max_distance=max_distance,
        use_vbm=True,
    )
    torch.cuda.synchronize()
    esdf_cold_ms = (time.perf_counter() - t0) * 1000.0
    esdf_n_voxels = int(esdf_grid.num_voxels)

    warm_samples = []
    prev_grid, prev_esdf = esdf_grid, esdf
    for _ in range(esdf_warm_calls):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        new_grid, new_esdf = g.compute_esdf_incremental(
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
        "tsdf_ms_per_f": tsdf_s * 1000.0 / max(N, 1),
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
        "n_frames": N,
        "chunk_frames": chunk_frames,
    }


def _run_one_config_chunked(
    scene, voxel_size: float, truncation: float, max_distance: float,
    esdf_warm_calls: int, skip_nvblox: bool, chunk_frames: int = 200,
) -> Dict[str, Any]:
    """Chunked equivalent of bench_esdf_replica._run_one_config."""
    print(f"\n[config] voxel={voxel_size}  trunc={truncation}  "
          f"max_dist={max_distance}", flush=True)
    result: Dict[str, Any] = {
        "voxel_size_m": voxel_size,
        "truncation_m": truncation,
        "max_distance_m": max_distance,
    }

    torch.cuda.reset_peak_memory_stats()
    try:
        result["fvdb"] = _run_fvdb_chunked_replica(
            scene, voxel_size, truncation, max_distance, esdf_warm_calls,
            chunk_frames=chunk_frames,
        )
        result["fvdb"]["ok"] = True
        r = result["fvdb"]
        print(f"  fvdb: TSDF {r['tsdf_n_voxels']:,} vx ({r['tsdf_ms_per_f']:.2f} ms/f), "
              f"ESDF {r['esdf_n_voxels']:,} vx, "
              f"cold {r['esdf_cold_ms']:.1f} ms, "
              f"warm {r['esdf_warm_ms_median']:.1f} ms, "
              f"peak_gb {r['peak_torch_gb']:.2f}", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  fvdb: OOM - {str(e).splitlines()[0][:140]}", flush=True)
        result["fvdb"] = {"ok": False, "failure": f"torch OOM: {e}"}
        torch.cuda.empty_cache(); gc.collect()
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "CUDA error 2" in msg:
            print(f"  fvdb: nanoVDB OOM - {msg.splitlines()[0][:120]}",
                  flush=True)
            result["fvdb"] = {"ok": False, "failure": f"nanoVDB OOM: {msg}"}
        else:
            print(f"  fvdb: failed - {msg[:140]}", flush=True)
            result["fvdb"] = {"ok": False, "failure": f"runtime: {msg}"}
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:  # noqa: BLE001
        print(f"  fvdb: failed ({type(e).__name__}: {e})", flush=True)
        result["fvdb"] = {"ok": False, "failure": f"{type(e).__name__}: {e}"}

    torch.cuda.empty_cache()
    gc.collect()

    if skip_nvblox:
        result["nvblox"] = {"ok": False, "failure": "skipped"}
    else:
        try:
            nvblox_result = upstream._run_nvblox(
                scene, voxel_size, truncation, max_distance, esdf_warm_calls,
            )
            result["nvblox"] = nvblox_result
            if nvblox_result.get("ok", False):
                print(f"  nvblox: approx_vx {nvblox_result['n_voxels']:,}, "
                      f"cold {nvblox_result['esdf_cold_ms']:.1f} ms, "
                      f"warm {nvblox_result['esdf_warm_ms_median']:.2f} ms, "
                      f"gpu_gb {nvblox_result.get('gpu_used_gb', -1):.2f}",
                      flush=True)
            else:
                print(f"  nvblox: FAILED {nvblox_result.get('failure', '?')[:120]}",
                      flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  nvblox: driver error ({type(e).__name__}: {e})",
                  flush=True)
            result["nvblox"] = {"ok": False,
                                "failure": f"driver: {type(e).__name__}: {e}"}

    return result


# Mai City-style known-OOM rule for nvblox at fine voxels.
# Earlier Replica-stride10 evidence: nvblox OOMs at 5 mm.
# At full frame rate (10x more frames) we expect OOM at 10 mm.
_KNOWN_OOM_NVBLOX = lambda vs: vs <= 0.005   # 5 mm and below


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True,
                    help="path to extracted Replica/ (the dir containing "
                         "office0/ etc.)")
    ap.add_argument("--scenes", nargs="+",
                    default=["office0", "room0", "room1"])
    ap.add_argument("--n-frames", type=int, default=2000,
                    help="frames per scene (default 2000 = full Nice-SLAM rate)")
    ap.add_argument("--voxel-sizes-m", type=float, nargs="+",
                    default=[0.02, 0.01, 0.005, 0.003])
    ap.add_argument("--trunc-voxel-multiplier", type=float, default=3.0)
    ap.add_argument("--max-distance-voxel-multiplier", type=float, default=10.0)
    ap.add_argument("--esdf-warm-calls", type=int, default=5)
    ap.add_argument("--chunk-frames", type=int, default=200,
                    help="frames per fvdb integrate call")
    ap.add_argument("--skip-known-oom", action="store_true", default=True)
    ap.add_argument("--no-skip-known-oom", dest="skip_known_oom",
                    action="store_false")
    ap.add_argument("--skip-nvblox", action="store_true",
                    help="skip nvblox at every voxel")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    all_results: list[Dict[str, Any]] = []
    t_start = time.time()

    for scene_name in args.scenes:
        print(f"\n##### Replica scene='{scene_name}' #####", flush=True)
        scene_dir = os.path.join(args.root, scene_name)
        scene = load_replica_scene(
            scene_dir, max_frames=args.n_frames, stride=1,
        )
        print(f"[load] done. {scene.n_frames} frames, "
              f"{scene.depth_images.shape[1]}x{scene.depth_images.shape[2]} px",
              flush=True)

        for vs in args.voxel_sizes_m:
            cfg_skip_nvblox = args.skip_nvblox or (
                args.skip_known_oom and _KNOWN_OOM_NVBLOX(vs))
            r = _run_one_config_chunked(
                scene,
                voxel_size=float(vs),
                truncation=float(vs) * args.trunc_voxel_multiplier,
                max_distance=float(vs) * args.max_distance_voxel_multiplier,
                esdf_warm_calls=args.esdf_warm_calls,
                skip_nvblox=cfg_skip_nvblox,
                chunk_frames=args.chunk_frames,
            )
            r["scene"] = scene_name
            r["dataset"] = "replica"
            all_results.append(r)
            if args.json_out is not None:
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                with args.json_out.open("w") as f:
                    json.dump({
                        "config": {
                            "scenes": args.scenes,
                            "n_frames": args.n_frames,
                            "esdf_warm_calls": args.esdf_warm_calls,
                            "trunc_voxel_multiplier":
                                args.trunc_voxel_multiplier,
                            "max_distance_voxel_multiplier":
                                args.max_distance_voxel_multiplier,
                            "chunk_frames": args.chunk_frames,
                        },
                        "results": all_results,
                    }, f, indent=2)

        # Free the scene's host-side depth array between scenes.
        del scene
        gc.collect()

    print(f"\n##### Total wall time: {(time.time() - t_start) / 60:.1f} min #####",
          flush=True)


if __name__ == "__main__":
    main()
