"""Decay-and-prune experiment on KITTI seq 00.

Single-number measurement of dynamic-scene decay's effect on persistent
voxel count, on a real-sensor LiDAR sequence with moving objects (cars,
cyclists, pedestrians on the road).

Two parallel runs:
  - "no_decay":  integrate first N frames in chunks; record voxel count
                  at each checkpoint.
  - "with_decay": integrate the same N frames in the same chunks; after
                  each chunk, apply decay-and-prune to the weight
                  sidecar (γ, threshold) and re-bind tsdf to the new
                  pruned grid.

Output: voxel count over time for both runs, plus the final reduction
ratio. The "single number" the paper cites is V_with_decay / V_no_decay
at the end of the integration window.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fvdb import Grid
from kitti_loader import load_kitti_scene


def integrate_chunk(g, tsdf, weights, scene, chunk_start, chunk_end,
                    truncation, sensor_origins_gpu):
    chunk_pts = [
        torch.from_numpy(scene.points_per_frame[i]).cuda()
        for i in range(chunk_start, chunk_end)
    ]
    chunk_origins = sensor_origins_gpu[chunk_start:chunk_end]
    g, tsdf, weights = g.integrate_tsdf_from_points_frames(
        truncation_distance=truncation,
        points_per_frame=chunk_pts,
        sensor_origins=chunk_origins,
        tsdf=tsdf, weights=weights,
        carve_free_space=True,
    )
    del chunk_pts
    return g, tsdf, weights


def run(scene, voxel_size, truncation, n_frames, chunk_frames,
        with_decay, decay_factor, prune_threshold):
    torch.cuda.empty_cache()
    g = Grid.from_dense(
        dense_dims=[10, 10, 10], ijk_min=[-5, -5, -5],
        voxel_size=voxel_size, origin=[0, 0, 0], device="cuda",
    )
    tsdf = torch.zeros(g.num_voxels, device="cuda", dtype=torch.float32)
    weights = torch.zeros(g.num_voxels, device="cuda", dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()

    sensor_origins_gpu = torch.from_numpy(scene.sensor_origins).cuda()

    timeline = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for chunk_start in range(0, n_frames, chunk_frames):
        chunk_end = min(chunk_start + chunk_frames, n_frames)
        g, tsdf, weights = integrate_chunk(
            g, tsdf, weights, scene, chunk_start, chunk_end,
            truncation, sensor_origins_gpu,
        )
        if with_decay:
            # Decay weights and prune the grid; tsdf rides along as an extra.
            g, weights, [tsdf] = g.decay_and_prune(
                weights,
                decay_factor=decay_factor,
                prune_threshold=prune_threshold,
                extra_sidecars=[tsdf],
            )
        torch.cuda.synchronize()
        timeline.append({
            "frames_integrated": chunk_end,
            "n_voxels": int(g.num_voxels),
        })

    wall_s = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return {
        "ok": True,
        "with_decay": with_decay,
        "decay_factor": decay_factor if with_decay else None,
        "prune_threshold": prune_threshold if with_decay else None,
        "n_frames_integrated": n_frames,
        "chunk_frames": chunk_frames,
        "final_voxels": int(g.num_voxels),
        "peak_gb": peak_gb,
        "wall_s": wall_s,
        "timeline": timeline,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sequence", default="00")
    ap.add_argument("--n-frames", type=int, default=500)
    ap.add_argument("--voxel-size", type=float, default=0.1)  # 10 cm
    ap.add_argument("--truncation", type=float, default=0.3)  # 3 * voxel
    ap.add_argument("--chunk-frames", type=int, default=25)
    ap.add_argument("--decay-factor", type=float, default=0.5)
    ap.add_argument("--prune-threshold", type=float, default=0.05)
    ap.add_argument("--json-out", type=Path, required=True)
    args = ap.parse_args()

    print(f"Loading KITTI seq {args.sequence}, first {args.n_frames} frames...",
          flush=True)
    scene = load_kitti_scene(args.root, sequence=args.sequence,
                              max_frames=args.n_frames)
    print(f"  loaded {scene.n_frames} frames, {scene.total_points:,} points",
          flush=True)

    print(f"\nRun 1/2: WITHOUT decay (baseline)", flush=True)
    no_decay = run(scene, args.voxel_size, args.truncation,
                    args.n_frames, args.chunk_frames,
                    with_decay=False,
                    decay_factor=1.0, prune_threshold=0.0)
    print(f"  final voxels: {no_decay['final_voxels']:,}", flush=True)

    print(f"\nRun 2/2: WITH decay (γ={args.decay_factor}, "
          f"threshold={args.prune_threshold} per chunk of "
          f"{args.chunk_frames} frames)", flush=True)
    with_decay = run(scene, args.voxel_size, args.truncation,
                      args.n_frames, args.chunk_frames,
                      with_decay=True,
                      decay_factor=args.decay_factor,
                      prune_threshold=args.prune_threshold)
    print(f"  final voxels: {with_decay['final_voxels']:,}", flush=True)

    reduction = (1 - with_decay["final_voxels"]
                  / no_decay["final_voxels"]) * 100
    print(f"\nReduction in persistent voxel count: {reduction:.1f}%",
          flush=True)

    out = {
        "args": vars(args) | {"json_out": str(args.json_out)},
        "scene": {
            "dataset": "kitti",
            "sequence": args.sequence,
            "n_frames": scene.n_frames,
            "total_points": int(scene.total_points),
        },
        "results": {
            "no_decay": no_decay,
            "with_decay": with_decay,
            "reduction_percent": reduction,
        },
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
