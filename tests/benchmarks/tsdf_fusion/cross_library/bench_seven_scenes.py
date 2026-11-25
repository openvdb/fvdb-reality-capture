# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Long-trajectory TSDF-fusion benchmark on Microsoft 7-Scenes.

Complements the Replica benchmarks (bounded room, 200 frames) with
a much longer-trajectory workload (~6000 frames, same physical room
revisited multiple times). Exercises the "accumGrid saturated early,
per-frame shell small, constant-factor matters" regime that Tree 4
(persistent growable grid) and Tree 5 (fused shell kernel) are
designed to attack. Uses the 7-Scenes `chess` dataset by default
(6 sequences x 1000 frames = 6000 total).

Run:

    python bench_seven_scenes.py --scene .../7-Scenes/chess \\
        --voxel-sizes 0.01 0.005 0.003 --n-frames 3000 \\
        --json-out results.json
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seven_scenes_loader import load_seven_scenes_scene  # noqa: E402


def run_fvdb(scene, voxel_size: float, truncation: float) -> Dict[str, Any]:
    """Run fvdb integrate_tsdf_frames on the whole scene, return stats."""
    from fvdb import Grid
    torch.cuda.empty_cache(); gc.collect()
    N = scene.n_frames
    depth = torch.from_numpy(scene.depth_images).cuda()
    c2w = torch.from_numpy(scene.cam_to_world).cuda()
    K = torch.from_numpy(scene.K).cuda().unsqueeze(0).expand(N, 3, 3).contiguous()
    g = Grid.from_dense(
        dense_dims=[1, 1, 1], ijk_min=[0, 0, 0],
        voxel_size=voxel_size, origin=[0, 0, 0], device="cuda",
    )
    tsdf = torch.zeros(g.num_voxels, device="cuda")
    w = torch.zeros(g.num_voxels, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    try:
        g, tsdf, w = g.integrate_tsdf_frames(
            truncation_distance=truncation,
            projection_matrices=K,
            cam_to_world_matrices=c2w,
            tsdf=tsdf, weights=w, depth_images=depth,
        )
        torch.cuda.synchronize()
        ms_per_f = (time.perf_counter() - t0) * 1000 / N
        peak = torch.cuda.max_memory_allocated() / 1e9
        n_voxels = int(g.num_voxels)
    except torch.cuda.OutOfMemoryError as e:
        return {"system": "fvdb", "ok": False,
                "failure": f"OOM: {str(e).splitlines()[0][:100]}"}
    return {
        "system": "fvdb", "ok": True,
        "ms_per_f": ms_per_f,
        "peak_gb": peak,
        "n_voxels": n_voxels,
        "n_frames": N,
    }


def run_open3d_cuda(scene, voxel_size: float, truncation: float,
                    block_count_base: int = 50000) -> Dict[str, Any]:
    """Run Open3D CUDA VBG integrate per-frame, return stats.

    Scales block_count with (base_voxel / voxel)^2 surface-area law
    starting from `block_count_base` at 1 cm, matching the pattern in
    `scale_scan.py`.
    """
    import open3d as o3d
    import open3d.core as o3c
    torch.cuda.empty_cache(); gc.collect()

    N = scene.n_frames
    d = o3c.Device("CUDA:0")
    trunc_mul = truncation / voxel_size
    # Scale block-count by (1cm/vs)^2 so fine voxels get larger hashmaps.
    bc = max(block_count_base,
             int(block_count_base * (0.01 / voxel_size) ** 2))

    try:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight"),
            attr_dtypes=(o3c.float32, o3c.float32),
            attr_channels=((1,), (1,)),
            voxel_size=voxel_size, block_resolution=16,
            block_count=bc, device=d,
        )
    except Exception as e:
        return {"system": "open3d_cuda", "ok": False,
                "failure": f"vbg_create: {str(e).splitlines()[0][:100]}",
                "block_count": bc}

    K_o3d = o3c.Tensor(np.asarray(scene.K, dtype=np.float64),
                       o3c.float64, device=o3c.Device("CPU:0"))
    # Stage depth on device + extrinsics on host once to keep per-frame
    # overhead honest (matches what scale_scan.py does).
    depths = [
        o3d.t.geometry.Image(
            o3c.Tensor(scene.depth_images[i].astype(np.float32),
                       o3c.float32, d))
        for i in range(N)
    ]
    exts = [
        o3c.Tensor(
            np.linalg.inv(scene.cam_to_world[i]).astype(np.float64),
            o3c.float64, device=o3c.Device("CPU:0"))
        for i in range(N)
    ]

    # Warmup.
    try:
        for i in range(min(2, N)):
            f = vbg.compute_unique_block_coordinates(
                depths[i], K_o3d, exts[i],
                depth_scale=1.0, depth_max=4.0,
                trunc_voxel_multiplier=trunc_mul)
            vbg.integrate(f, depths[i], K_o3d, exts[i],
                          depth_scale=1.0, depth_max=4.0,
                          trunc_voxel_multiplier=trunc_mul)
        o3c.cuda.synchronize()
    except Exception as e:
        return {"system": "open3d_cuda", "ok": False,
                "failure": f"warmup: {str(e).splitlines()[0][:100]}",
                "block_count": bc}

    # Fresh VBG for timed run so warmup doesn't skew sizes.
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"),
        attr_dtypes=(o3c.float32, o3c.float32),
        attr_channels=((1,), (1,)),
        voxel_size=voxel_size, block_resolution=16,
        block_count=bc, device=d,
    )
    t0 = time.perf_counter()
    try:
        for i in range(N):
            f = vbg.compute_unique_block_coordinates(
                depths[i], K_o3d, exts[i],
                depth_scale=1.0, depth_max=4.0,
                trunc_voxel_multiplier=trunc_mul)
            vbg.integrate(f, depths[i], K_o3d, exts[i],
                          depth_scale=1.0, depth_max=4.0,
                          trunc_voxel_multiplier=trunc_mul)
        o3c.cuda.synchronize()
    except Exception as e:
        return {"system": "open3d_cuda", "ok": False,
                "failure": f"integrate: {str(e).splitlines()[0][:100]}",
                "block_count": bc}
    ms_per_f = (time.perf_counter() - t0) * 1000 / N
    blocks = int(vbg.hashmap().size())
    return {
        "system": "open3d_cuda", "ok": True,
        "ms_per_f": ms_per_f,
        "block_count": bc,
        "blocks": blocks,
        "n_frames": N,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", required=True,
                   help="path to unzipped 7-Scenes scene, e.g. .../chess")
    p.add_argument("--n-frames", type=int, default=3000,
                   help="cap total frames (0 = use all)")
    p.add_argument("--voxel-sizes", nargs="+", type=float,
                   default=[0.02, 0.01, 0.005, 0.003])
    p.add_argument("--truncation-multiplier", type=float, default=3.0)
    p.add_argument("--systems", nargs="+",
                   default=["fvdb", "open3d_cuda"],
                   choices=["fvdb", "open3d_cuda"])
    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()

    max_frames = None if args.n_frames <= 0 else args.n_frames
    scene = load_seven_scenes_scene(args.scene, max_frames=max_frames)
    print(f"Loaded {scene.n_frames} frames from {args.scene!r}")

    results: List[Dict[str, Any]] = []
    for vs in args.voxel_sizes:
        trunc = vs * args.truncation_multiplier
        print(f"\n=== voxel_size={vs*1000:.2f}mm  trunc={trunc*1000:.2f}mm ===")
        for system in args.systems:
            if system == "fvdb":
                r = run_fvdb(scene, vs, trunc)
            elif system == "open3d_cuda":
                r = run_open3d_cuda(scene, vs, trunc)
            else:
                continue
            r["voxel_size"] = vs
            results.append(r)
            if r.get("ok"):
                extra = ""
                if "peak_gb" in r:
                    extra = f"  peak={r['peak_gb']:.2f} GB  voxels={r['n_voxels']:,}"
                elif "blocks" in r:
                    extra = f"  blocks={r['blocks']:,}/{r['block_count']}"
                print(f"  {system:15s} OK  {r['ms_per_f']:7.2f} ms/f{extra}")
            else:
                print(f"  {system:15s} FAIL  {r.get('failure','?')[:120]}")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"results": results, "args": vars(args)},
                          f, indent=2)


if __name__ == "__main__":
    main()
