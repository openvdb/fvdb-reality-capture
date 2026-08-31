# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Long-trajectory LiDAR TSDF-fusion benchmark on Mai City (synthetic
Velodyne HDL-64).

Mai City sequence 00 is a 700 m synthetic drive at 10 m/s with 700
Velodyne-style sweeps (~130 K points each after the opening few
frames). Complements the depth-TSDF benches
(`bench_open3d_vs_fvdb.py` on Replica, `bench_seven_scenes.py` on
7-Scenes) by exercising the LiDAR integrate path
(`Grid.integrate_tsdf_from_points`) on an unbounded-surface
trajectory.

Runs the following systems (each optional via --systems):
  - `fvdb`:           current per-frame-looped `integrate_tsdf_from_points`
                       (the implementation that ships today).
  - `fvdb_composed`:  explicit `FVDB_TSDF_COMPOSED=1` ablation (same
                       path, but opts out of any future persistent-state
                       changes so the baseline is reproducible).
  - `vdbfusion`:      PRBonn VDBFusion CPU/OpenVDB baseline. Disabled
                       by default because `pip install vdbfusion`
                       currently fails due to the vendored
                       blosc-1.x + zlib-1.2.8 not compiling with
                       modern gcc (missing `#include <unistd.h>` for
                       `lseek`/`read`/`write`/`close`); the fix
                       requires either a Docker image or patched
                       third-party sources. See `install_vdbfusion.sh`
                       in this directory for the working recipe.

Usage:

    python bench_mai_city.py \\
        --root .../data/mai_city/mai_city \\
        --sequence 00 \\
        --n-frames 700 \\
        --voxel-sizes 0.2 0.1 0.05 \\
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
from mai_city_loader import load_mai_city_scene  # noqa: E402


def run_fvdb(scene, voxel_size: float, truncation: float,
             carve_free_space: bool = True,
             batched: bool = True) -> Dict[str, Any]:
    """Run fvdb's LiDAR TSDF on a Mai City sweep sequence.

    Two paths:
      - `batched=True` (default): `Grid.integrate_tsdf_from_points_frames`
        runs the whole N-sweep loop in C++, avoiding the per-frame
        Python <-> C++ dispatch + JaggedTensor rewrap cost. Shipped
        in the batched N-frame path. Output matches the per-frame loop
        within a 10-ULP tolerance (atomic-add reorder in the ray-walk
        kernel is NOT bit-deterministic even for two back-to-back
        sequential runs, so the batched path's agreement floor is
        just that atomic-noise floor).
      - `batched=False`: Python `for` loop over `integrate_tsdf_from_points`,
        matching the per-frame loop. Useful as
        an ablation baseline to quantify the Python-dispatch overhead.
    """
    from fvdb import Grid
    torch.cuda.empty_cache()
    gc.collect()

    N = scene.n_frames
    # Start from a tiny dense seed grid rather than truly empty --
    # `integrate_tsdf_from_points` requires a non-empty initial
    # GridBatch (there's an "empty grid handle" runtime check in the
    # C++ path). 10^3 voxels at any origin costs essentially nothing.
    g = Grid.from_dense(
        dense_dims=[10, 10, 10], ijk_min=[-5, -5, -5],
        voxel_size=voxel_size, origin=[0, 0, 0], device="cuda",
    )
    tsdf = torch.zeros(g.num_voxels, device="cuda", dtype=torch.float32)
    weights = torch.zeros(g.num_voxels, device="cuda", dtype=torch.float32)

    # Warmup 2 frames (matches the pattern in bench_seven_scenes).
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

    # Fresh state for timed run.
    g = Grid.from_dense(
        dense_dims=[10, 10, 10], ijk_min=[-5, -5, -5],
        voxel_size=voxel_size, origin=[0, 0, 0], device="cuda",
    )
    tsdf = torch.zeros(g.num_voxels, device="cuda", dtype=torch.float32)
    weights = torch.zeros(g.num_voxels, device="cuda", dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()

    try:
        if batched:
            # Batched N-frame path: one C++ call for all N frames.
            # Pre-load all points to GPU OUTSIDE the timed window --
            # the point-transfer step isn't the thing the batched path is
            # optimizing, and a real streaming pipeline would overlap
            # it with the integrate kernels anyway. We measure the
            # integrate-only cost for a fair comparison against
            # `fvdb_per_frame`.
            pts_per_frame = [
                torch.from_numpy(p).cuda() for p in scene.points_per_frame
            ]
            origins = torch.from_numpy(scene.sensor_origins).cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g, tsdf, weights = g.integrate_tsdf_from_points_frames(
                truncation_distance=truncation,
                points_per_frame=pts_per_frame,
                sensor_origins=origins,
                tsdf=tsdf, weights=weights,
                carve_free_space=carve_free_space,
            )
            torch.cuda.synchronize()
        else:
            # Ablation baseline: Python per-frame loop. For symmetry,
            # also pre-load all points outside the timing window --
            # otherwise the per-frame path would get "credit" for the
            # H->D copy being interleaved with compute.
            pts_per_frame_cu = [
                torch.from_numpy(p).cuda() for p in scene.points_per_frame
            ]
            origins = torch.from_numpy(scene.sensor_origins).cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(N):
                g, tsdf, weights = g.integrate_tsdf_from_points(
                    truncation_distance=truncation,
                    points=pts_per_frame_cu[i],
                    sensor_origin=origins[i],
                    tsdf=tsdf, weights=weights,
                    carve_free_space=carve_free_space,
                )
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        return {"system": "fvdb", "ok": False,
                "failure": f"OOM: {str(e).splitlines()[0][:100]}"}
    except RuntimeError as e:
        return {"system": "fvdb", "ok": False,
                "failure": f"runtime: {str(e).splitlines()[0][:140]}"}

    ms_per_f = (time.perf_counter() - t0) * 1000 / N
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return {
        "system": "fvdb" if batched else "fvdb_per_frame",
        "ok": True,
        "ms_per_f": ms_per_f,
        "peak_gb": peak_gb,
        "n_voxels": int(g.num_voxels),
        "n_frames": N,
    }


def run_nvblox(scene, voxel_size: float, truncation: float,
               nvblox_env_python: str = "/home/fwilliams/bin/miniconda3/envs/nvblox/bin/python",
               num_azimuth: int = 1800, num_elevation: int = 64,
               vertical_fov_rad: float = 0.4712,  # ~27 deg for HDL-64
               **_ignored) -> Dict[str, Any]:
    """Run NVIDIA nvblox's LiDAR TSDF integrator on a Mai City sweep
    sequence by spawning a subprocess into the `nvblox` conda env.

    nvblox needs CUDA 12 + torch<=2.9.1 (its wheel targets those); our
    fvdb env has CUDA 13 + torch 2.10. Rather than downgrading our
    env, we build a dedicated `nvblox` conda env (see
    `install_nvblox.sh`, to be written) and call it out-of-process.
    The subprocess pattern also keeps nvblox's CUDA context fully
    isolated from our process -- no risk of the two libraries fighting
    over GPU memory or CUDA streams.

    nvblox represents LiDAR input as a (num_elevation, num_azimuth)
    spherical range image. The runner script converts our raw 3D
    Mai City points -> sensor frame -> range image internally. We
    use HDL-64 defaults matching the Mai City dataset.
    """
    import os, subprocess, tempfile

    if not os.path.exists(nvblox_env_python):
        return {"system": "nvblox", "ok": False,
                "failure": f"nvblox env python not found: {nvblox_env_python}"}

    runner = str(Path(__file__).resolve().parent / "nvblox_runner.py")
    if not os.path.exists(runner):
        return {"system": "nvblox", "ok": False,
                "failure": f"nvblox_runner.py missing: {runner}"}

    # Stash points + poses into an npz so the subprocess can load
    # them zero-copy (JSON-encoding N x ~100k points is too slow).
    with tempfile.TemporaryDirectory(prefix="nvblox_bench_") as tmp:
        # Concatenate all frames' points + build offsets for O(1)
        # slicing on the nvblox side.
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
        }
        spec_path = os.path.join(tmp, "spec.json")
        out_path = os.path.join(tmp, "result.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "1")
        try:
            proc = subprocess.run(
                [nvblox_env_python, runner, "--spec", spec_path,
                 "--output", out_path],
                env=env, capture_output=True, text=True,
                timeout=3600,  # 1h cap; way above expected runtime
            )
        except subprocess.TimeoutExpired as e:
            return {"system": "nvblox", "ok": False,
                    "failure": f"subprocess timeout: {e}"}

        if not os.path.exists(out_path):
            return {"system": "nvblox", "ok": False,
                    "failure": f"nvblox runner did not produce output. "
                               f"stderr tail: {proc.stderr[-500:]!r}"}
        with open(out_path, "r") as f:
            result = json.load(f)

    # Normalise keys to match fvdb/vdbfusion output schema. Prefer
    # `gpu_used_gb` (CUDA driver-reported aggregate used memory)
    # over the RSS delta for a GPU library -- RSS only captures
    # pinned-host allocations which vastly under-represent nvblox's
    # actual footprint.
    if result.get("gpu_used_gb", -1) > 0:
        result.setdefault("peak_gb", result["gpu_used_gb"])
    else:
        result.setdefault("peak_gb", result.get("peak_rss_delta_gb", 0.0))
    result["system"] = "nvblox"
    if not result.get("ok"):
        # Surface a short failure string to the driver's print loop.
        return {"system": "nvblox", "ok": False,
                "failure": result.get("failure", "unknown") +
                           (f" (stderr tail: {proc.stderr[-300:]!r})"
                            if proc.stderr else "")}
    return result


def run_vdbfusion(scene, voxel_size: float, truncation: float,
                  carve_free_space: bool = True,
                  **_ignored) -> Dict[str, Any]:
    """Run PRBonn VDBFusion CPU TSDF integrator (multi-threaded OpenVDB).

    VDBFusion's API: construct a `VDBVolume(voxel_size, sdf_trunc,
    space_carving)`, call `integrate(points, origin)` per sweep,
    then `extract_triangle_mesh()` at the end. Runs on CPU with
    TBB parallelism; this is exactly the baseline the paper wants
    to anchor against for LiDAR TSDF.

    Install notes: `pip install vdbfusion` fails on modern gcc
    because of vendored blosc-1.x + zlib-1.2.8 toolchain issues.
    A working recipe is to clone OpenVDB 12 from source, build it
    against the fvdb conda env's TBB / Blosc / Boost, and rebuild
    the VDBFusion wheel with `CMAKE_MODULE_PATH` pointing at the
    OpenVDB prefix. See the helper script `install_vdbfusion.sh`
    next to this file for the steps.
    """
    try:
        import vdbfusion
    except ImportError as e:
        return {
            "system": "vdbfusion", "ok": False,
            "failure": f"vdbfusion not installed: {e}",
        }
    import gc
    gc.collect()

    # Warmup on first two frames so the OpenVDB thread pool + grid
    # allocator are hot (matches the fvdb harness's 2-frame warmup).
    vol = vdbfusion.VDBVolume(
        voxel_size=voxel_size, sdf_trunc=truncation,
        space_carving=carve_free_space,
    )
    for i in range(min(2, scene.n_frames)):
        vol.integrate(
            scene.points_per_frame[i].astype(np.float64, copy=False),
            scene.sensor_origins[i].astype(np.float64, copy=False),
        )
    del vol
    gc.collect()

    # Fresh instance for the timed run.
    vol = vdbfusion.VDBVolume(
        voxel_size=voxel_size, sdf_trunc=truncation,
        space_carving=carve_free_space,
    )
    # CPU workload — use wall-clock via perf_counter. We don't have
    # GPU memory stats for CPU libraries; report peak RSS via
    # `resource` instead so the comparison table has *some* memory
    # number for both sides.
    import resource
    base_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t0 = time.perf_counter()
    try:
        for i in range(scene.n_frames):
            vol.integrate(
                scene.points_per_frame[i].astype(np.float64, copy=False),
                scene.sensor_origins[i].astype(np.float64, copy=False),
            )
    except Exception as e:  # noqa: BLE001
        return {"system": "vdbfusion", "ok": False,
                "failure": f"runtime: {str(e).splitlines()[0][:140]}"}
    wall_s = time.perf_counter() - t0
    ms_per_f = wall_s * 1000 / scene.n_frames

    # Voxel count: `vol.tsdf.activeVoxelCount()` would be the natural
    # API but the Python binding doesn't expose it directly; instead
    # run a mesh extract (which iterates the active set) and report
    # the vertex count as a proxy for surface complexity. Timing is
    # already captured above so this doesn't skew ms/f.
    try:
        verts, tris = vol.extract_triangle_mesh(min_weight=0.1)
        n_verts = int(verts.shape[0])
        n_tris = int(tris.shape[0])
    except Exception as e:  # noqa: BLE001
        n_verts = n_tris = -1
        print(f"    warning: VDBFusion mesh extract failed: {e}")

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "system": "vdbfusion", "ok": True,
        "ms_per_f": ms_per_f,
        "wall_s": wall_s,
        # CPU RSS delta in "GB" so the column visually lines up with
        # fvdb's GPU peak. Apples-to-oranges (host vs device mem), but
        # order-of-magnitude is what the paper table cares about.
        "peak_gb": max(0.0, (peak_rss_kb - base_rss_kb) / 1e6),
        "peak_rss_gb": peak_rss_kb / 1e6,
        "n_frames": scene.n_frames,
        "n_voxels": n_verts,  # approximation via mesh vertex count
        "n_mesh_verts": n_verts,
        "n_mesh_tris": n_tris,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True,
                   help="path to extracted mai_city/ (the dir containing bin/)")
    p.add_argument("--sequence", default="00",
                   help="Mai City sequence: 00 (700m drive), 01, or 02")
    p.add_argument("--n-frames", type=int, default=700,
                   help="frame cap (0 or negative uses all frames)")
    p.add_argument("--voxel-sizes", nargs="+", type=float,
                   default=[0.2, 0.1, 0.05],
                   help="voxel sizes in metres. Outdoor LiDAR TSDF "
                        "defaults: 20 cm (fast), 10 cm (balanced), "
                        "5 cm (detailed).")
    p.add_argument("--truncation-multiplier", type=float, default=3.0)
    p.add_argument("--carve-free-space", action="store_true", default=True,
                   help="VDBFusion-compatible free-space carving. Default on.")
    p.add_argument("--no-carve-free-space", dest="carve_free_space",
                   action="store_false")
    p.add_argument("--systems", nargs="+",
                   default=["fvdb", "vdbfusion"],
                   choices=["fvdb", "fvdb_per_frame", "vdbfusion", "nvblox"])
    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()

    max_frames = None if args.n_frames <= 0 else args.n_frames
    scene = load_mai_city_scene(
        args.root, sequence=args.sequence, max_frames=max_frames,
    )
    traj_len = float(np.linalg.norm(
        np.diff(scene.sensor_origins, axis=0), axis=1).sum())
    print(f"Loaded Mai City seq={args.sequence!r}: "
          f"{scene.n_frames} frames, {scene.total_points:,} total points, "
          f"trajectory length {traj_len:.1f} m")

    results: List[Dict[str, Any]] = []
    for vs in args.voxel_sizes:
        trunc = vs * args.truncation_multiplier
        print(f"\n=== voxel_size={vs*100:.1f}cm  trunc={trunc*100:.1f}cm ===")
        for system in args.systems:
            if system == "fvdb":
                r = run_fvdb(scene, vs, trunc,
                             carve_free_space=args.carve_free_space,
                             batched=True)
            elif system == "fvdb_per_frame":
                r = run_fvdb(scene, vs, trunc,
                             carve_free_space=args.carve_free_space,
                             batched=False)
            elif system == "vdbfusion":
                r = run_vdbfusion(scene, vs, trunc)
            elif system == "nvblox":
                r = run_nvblox(scene, vs, trunc)
            else:
                continue
            r["voxel_size"] = vs
            results.append(r)
            if r.get("ok"):
                extra = (f"  peak={r.get('peak_gb', 0):.2f} GB  "
                         f"voxels={r.get('n_voxels', 0):,}")
                print(f"  {system:15s} OK  {r['ms_per_f']:7.2f} ms/f{extra}")
            else:
                print(f"  {system:15s} FAIL  {r.get('failure','?')[:140]}")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"results": results, "args": vars(args)},
                          f, indent=2)


if __name__ == "__main__":
    main()
