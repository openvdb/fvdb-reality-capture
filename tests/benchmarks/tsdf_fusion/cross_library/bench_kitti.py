# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Long-trajectory LiDAR TSDF benchmark on KITTI Odometry (real-sensor
HDL-64).

Counterpart to `bench_mai_city.py`. KITTI replaces the synthetic
Mai City sweeps with real Velodyne data: longer trajectories
(seq 00 = 4541 frames over 3.7 km), real noise, real moving cars
in scene. The integrate path is unchanged
(`Grid.integrate_tsdf_from_points_frames`); the value of running
this is to validate that the Mai City speed/scale claims hold on
real-sensor data.

Reuses `run_fvdb`, `run_nvblox`, `run_vdbfusion` from
`bench_mai_city.py` since those functions are loader-agnostic
(they only depend on the duck-typed scene interface satisfied by
both `MaiCityScene` and `KittiScene`).

Usage:

    python bench_kitti.py \\
        --root .../data/KITTI \\
        --sequences 00 02 05 \\
        --voxel-sizes 0.4 0.2 0.1 0.05 0.03 0.02 \\
        --systems fvdb nvblox \\
        --skip-known-oom \\
        --json-out kitti_tsdf.json
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_loader import load_kitti_scene  # noqa: E402
from bench_mai_city import run_nvblox, run_vdbfusion  # noqa: E402


def run_fvdb_chunked(scene, voxel_size: float, truncation: float,
                     carve_free_space: bool = True,
                     chunk_frames: int = 200) -> Dict[str, Any]:
    """Chunked fvdb LiDAR TSDF for long trajectories.

    The unchunked `bench_mai_city.run_fvdb` pre-loads ALL frames'
    point clouds onto the GPU before the timed window. For KITTI's
    4541-frame sequence 00 (~549 M points x 12 B = 6.6 GB just
    for points), this triggers an OOM in nanoVDB's allocator before
    the integrate kernel even starts.

    This variant streams frames in chunks: at each step it loads
    `chunk_frames` worth of point clouds to GPU, runs one
    `integrate_tsdf_from_points_frames` over the chunk (which
    grows the persistent grid), then frees the chunk's CUDA
    tensors. Throughput is essentially identical to the one-shot
    path (the C++ kernel doesn't care about call boundaries) but
    peak memory is bounded by `chunk_frames * avg_points * 12 B`
    instead of `total_points * 12 B`.

    Memory accounting note: `peak_gb` is the post-fusion CUDA
    high-water mark (so it includes the final grid + sidecars),
    which is what the paper cares about. The transient point-
    upload spike per chunk is bounded.
    """
    import gc
    from fvdb import Grid

    torch.cuda.empty_cache()
    gc.collect()

    N = scene.n_frames
    g = Grid.from_dense(
        dense_dims=[10, 10, 10], ijk_min=[-5, -5, -5],
        voxel_size=voxel_size, origin=[0, 0, 0], device="cuda",
    )
    tsdf = torch.zeros(g.num_voxels, device="cuda", dtype=torch.float32)
    weights = torch.zeros(g.num_voxels, device="cuda", dtype=torch.float32)

    # Warmup on first 2 frames (matches bench_mai_city.run_fvdb).
    for i in range(min(2, N)):
        pts = torch.from_numpy(scene.points_per_frame[i]).cuda()
        origin = torch.from_numpy(scene.sensor_origins[i]).cuda()
        g, tsdf, weights = g.integrate_tsdf_from_points(
            truncation_distance=truncation,
            points=pts, sensor_origin=origin,
            tsdf=tsdf, weights=weights,
            carve_free_space=carve_free_space,
        )
    torch.cuda.synchronize()

    # Fresh state for the timed run.
    g = Grid.from_dense(
        dense_dims=[10, 10, 10], ijk_min=[-5, -5, -5],
        voxel_size=voxel_size, origin=[0, 0, 0], device="cuda",
    )
    tsdf = torch.zeros(g.num_voxels, device="cuda", dtype=torch.float32)
    weights = torch.zeros(g.num_voxels, device="cuda", dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()

    sensor_origins_full = torch.from_numpy(scene.sensor_origins).cuda()

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for chunk_start in range(0, N, chunk_frames):
            chunk_end = min(chunk_start + chunk_frames, N)
            chunk_pts = [
                torch.from_numpy(scene.points_per_frame[i]).cuda()
                for i in range(chunk_start, chunk_end)
            ]
            chunk_origins = sensor_origins_full[chunk_start:chunk_end]
            g, tsdf, weights = g.integrate_tsdf_from_points_frames(
                truncation_distance=truncation,
                points_per_frame=chunk_pts,
                sensor_origins=chunk_origins,
                tsdf=tsdf, weights=weights,
                carve_free_space=carve_free_space,
            )
            del chunk_pts
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        return {"system": "fvdb", "ok": False,
                "failure": f"torch OOM: {str(e).splitlines()[0][:140]}"}
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "CUDA error 2" in msg:
            return {"system": "fvdb", "ok": False,
                    "failure": f"runtime OOM: {msg.splitlines()[0][:140]}"}
        return {"system": "fvdb", "ok": False,
                "failure": f"runtime: {msg.splitlines()[0][:140]}"}

    ms_per_f = (time.perf_counter() - t0) * 1000 / N
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return {
        "system": "fvdb", "ok": True,
        "ms_per_f": ms_per_f, "peak_gb": peak_gb,
        "n_voxels": int(g.num_voxels), "n_frames": N,
        "chunk_frames": chunk_frames,
    }


# Mai City evidence (700-frame seq00) shows nvblox OOMs at 5 cm and
# below for LiDAR TSDF. KITTI has 6.5x more frames per sequence, so
# nvblox will OOM at LEAST as early. Skipping known-OOM cells saves
# ~30% wall time without losing paper-relevant data (we just record
# them as "skipped: known OOM from Mai City").
_KNOWN_OOM_NVBLOX = {
    "tsdf": lambda vs: vs <= 0.05,    # 5 cm and below
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True,
                   help="path to extracted KITTI/ (the dir containing dataset/)")
    p.add_argument("--sequences", nargs="+", default=["00", "02", "05"],
                   help="KITTI sequences (00..10 are training; 11..21 lack poses)")
    p.add_argument("--n-frames", type=int, default=0,
                   help="cap on frames per sequence (0 = all). "
                        "Useful for smoke tests.")
    p.add_argument("--voxel-sizes", nargs="+", type=float,
                   default=[0.4, 0.2, 0.1, 0.05, 0.03, 0.02],
                   help="voxel sizes in metres.")
    p.add_argument("--truncation-multiplier", type=float, default=3.0)
    p.add_argument("--carve-free-space", action="store_true", default=True)
    p.add_argument("--no-carve-free-space", dest="carve_free_space",
                   action="store_false")
    p.add_argument("--systems", nargs="+",
                   default=["fvdb", "nvblox"],
                   choices=["fvdb", "fvdb_per_frame", "vdbfusion", "nvblox"])
    p.add_argument("--skip-known-oom", action="store_true", default=True,
                   help="skip nvblox at voxels Mai City showed OOM (default on)")
    p.add_argument("--no-skip-known-oom", dest="skip_known_oom",
                   action="store_false")
    p.add_argument("--chunk-frames", type=int, default=200,
                   help="frames per fvdb integrate call. Bounds peak GPU "
                        "memory for the input point upload at ~chunk_frames "
                        "* avg_pts * 12 B. Default 200 -> ~290 MB transient "
                        "for KITTI (vs ~6.6 GB if all 4541 frames pre-loaded).")
    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()

    max_frames = None if args.n_frames <= 0 else args.n_frames

    all_results: List[Dict[str, Any]] = []
    t_start = time.time()

    for seq in args.sequences:
        print(f"\n##### KITTI seq={seq!r} #####", flush=True)
        scene = load_kitti_scene(args.root, sequence=seq, max_frames=max_frames)
        traj_len = float(np.linalg.norm(
            np.diff(scene.sensor_origins, axis=0), axis=1).sum())
        print(f"Loaded: {scene.n_frames} frames, "
              f"{scene.total_points:,} total points, "
              f"avg {scene.total_points // max(scene.n_frames, 1):,} pts/frame, "
              f"trajectory length {traj_len:.1f} m", flush=True)

        for vs in args.voxel_sizes:
            trunc = vs * args.truncation_multiplier
            print(f"\n=== seq={seq} voxel={vs*100:.1f}cm trunc={trunc*100:.1f}cm ===",
                  flush=True)
            for system in args.systems:
                t0 = time.time()
                if (args.skip_known_oom and system == "nvblox"
                        and _KNOWN_OOM_NVBLOX["tsdf"](vs)):
                    r = {"system": "nvblox", "ok": False,
                         "failure": "skipped: known OOM at this voxel from "
                                    "Mai City evidence",
                         "skipped": True}
                elif system == "fvdb":
                    r = run_fvdb_chunked(
                        scene, vs, trunc,
                        carve_free_space=args.carve_free_space,
                        chunk_frames=args.chunk_frames)
                elif system == "vdbfusion":
                    r = run_vdbfusion(scene, vs, trunc)
                elif system == "nvblox":
                    r = run_nvblox(scene, vs, trunc)
                else:
                    continue
                r["voxel_size"] = vs
                r["sequence"] = seq
                r["dataset"] = "kitti"
                r["wall_s"] = time.time() - t0
                all_results.append(r)
                if r.get("ok"):
                    extra = (f"  peak={r.get('peak_gb', 0):.2f} GB"
                             f"  voxels={r.get('n_voxels', 0):,}")
                    print(f"  {system:15s} OK  {r['ms_per_f']:7.2f} ms/f"
                          f"  ({r['wall_s']:.1f} s wall){extra}",
                          flush=True)
                elif r.get("skipped"):
                    print(f"  {system:15s} SKIP  {r.get('failure','')[:80]}",
                          flush=True)
                else:
                    print(f"  {system:15s} FAIL  ({r['wall_s']:.1f} s wall) "
                          f"{r.get('failure','?')[:120]}",
                          flush=True)
                # Drain torch's allocator between systems so a fail
                # doesn't poison the next attempt.
                torch.cuda.empty_cache()
            if args.json_out:
                with open(args.json_out, "w") as f:
                    json.dump({"results": all_results, "args": vars(args)},
                              f, indent=2)

    print(f"\n##### Total wall time: {(time.time() - t_start) / 60:.1f} min #####",
          flush=True)


if __name__ == "__main__":
    main()
