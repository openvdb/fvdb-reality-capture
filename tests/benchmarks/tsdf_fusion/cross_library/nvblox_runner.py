# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Standalone nvblox TSDF-fusion runner for cross-library benchmarks.

Designed to be invoked as a subprocess from a benchmark driver that
runs in a different conda env. This script runs in the `nvblox`
conda env (torch==2.6.0+cu124, CUDA toolkit 12.4, the nvblox_torch
wheel we built from source in install_nvblox.sh). It reads a
dataset-specific JSON spec from stdin or a file, integrates the
sweeps/frames via nvblox's `Mapper`, and writes the timing + voxel
stats JSON to stdout or an output file.

Supported workloads:
  - LiDAR sweeps (Mai City, KITTI): the spec provides a list of
    3D point clouds (in sensor or world frame; we always send
    sensor-frame to nvblox and pass the world pose alongside) plus
    HDL-64-style LiDAR intrinsics (num_azimuth, num_elevation,
    vertical_fov_rad). The runner reprojects each sweep to a
    (num_elevation, num_azimuth) depth image and hands it to
    `Mapper.add_depth_frame`. Nvblox's LiDAR integrator expects
    exactly this representation.
  - Depth frames (Replica, 7-Scenes): the spec provides depth
    image paths + camera intrinsics. The runner loads each depth
    image and hands it to `Mapper.add_depth_frame`.

Spec format (JSON):
    {
        "workload": "lidar" | "depth",
        "voxel_size_m": 0.2,
        "truncation_distance_m": 0.6,
        # LiDAR only:
        "lidar_num_azimuth": 1800,
        "lidar_num_elevation": 64,
        "lidar_vertical_fov_rad": 0.4712,  # ~27 deg for HDL-64
        "lidar_min_valid_range_m": 1.0,
        "lidar_sweeps_npz": "/path/to/mai_city_sweeps.npz",  # holds (points_per_frame, sensor_origins, cam_to_world)
        # Depth only:
        "depth_image_paths": [...],
        "depth_intrinsics": {"fu": ..., "fv": ..., "cu": ..., "cv": ..., "width": ..., "height": ...},
        "depth_poses_npy": "/path/to/poses_Nx4x4.npy",
        "depth_scale": 1000.0,  # for uint16 mm depth
        "depth_max_m": 8.0,
        "warmup_frames": 2,
    }

Output (JSON to stdout or `--output`):
    {
        "ok": true,
        "ms_per_f": 123.45,
        "wall_s": 78.9,
        "peak_rss_gb": 3.14,
        "n_frames": 700,
        "n_voxels": 123456,      # nvblox's activeBlockCount * 512 (approximate)
        "n_mesh_verts": 10000,
        "n_mesh_tris": 20000,
        "voxel_size_m": 0.2,
        "failure": null,
    }
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor


def _points_to_spherical_range_image(
    points_sensor: np.ndarray,
    num_azimuth: int,
    num_elevation: int,
    vertical_fov_rad: float,
) -> np.ndarray:
    """Reproject an [N, 3] sensor-frame point cloud to an
    (H, W) = (num_elevation, num_azimuth) range image matching
    nvblox's `Sensor.from_lidar` parameterisation.

    Reference: nvblox's internal `Lidar::project` uses
        azimuth = atan2(y, x)                              # (-pi, pi]
        elevation = atan2(z, sqrt(x**2 + y**2))            # (-v_fov/2, +v_fov/2)
        u = (azimuth + pi) / (2*pi) * num_azimuth
        v = (elevation + v_fov/2) / v_fov * num_elevation

    Points outside the vertical FoV or mapped to the same pixel as a
    closer point are discarded (kept: min range per pixel).

    Returns an fp32 (num_elevation, num_azimuth) array; pixels with
    no hit are 0.0 (nvblox skips depth==0 automatically).
    """
    if points_sensor.shape[0] == 0:
        return np.zeros((num_elevation, num_azimuth), dtype=np.float32)

    x = points_sensor[:, 0]
    y = points_sensor[:, 1]
    z = points_sensor[:, 2]
    radial_xy = np.sqrt(x * x + y * y)
    ranges = np.sqrt(radial_xy * radial_xy + z * z)

    # Filter zero-range and out-of-FoV points. Keep epsilon for
    # numerical stability at the pole (radial_xy == 0 would NaN
    # atan2(z, 0) to +/-pi/2 which is fine but let's be defensive).
    valid = (ranges > 1e-4) & (radial_xy > 1e-6)
    x, y, z, radial_xy, ranges = x[valid], y[valid], z[valid], radial_xy[valid], ranges[valid]

    azimuth = np.arctan2(y, x)                   # (-pi, pi]
    elevation = np.arctan2(z, radial_xy)         # roughly (-v_fov/2, +v_fov/2) for LiDAR

    # Clamp into the valid range the sensor accepts.
    el_min = -vertical_fov_rad / 2.0
    el_max = +vertical_fov_rad / 2.0
    in_fov = (elevation >= el_min) & (elevation < el_max)
    azimuth, elevation, ranges = azimuth[in_fov], elevation[in_fov], ranges[in_fov]

    # Map to pixel indices.
    u = ((azimuth + math.pi) / (2.0 * math.pi) * num_azimuth).astype(np.int64)
    v = ((elevation - el_min) / vertical_fov_rad * num_elevation).astype(np.int64)
    # Clip to valid range (guards against +pi azimuth rounding to num_azimuth).
    u = np.clip(u, 0, num_azimuth - 1)
    v = np.clip(v, 0, num_elevation - 1)

    # Keep min range per pixel. Vectorised via numpy: combine (v, u)
    # into a single flat index, then sort by range ascending and use
    # np.unique on the flat index with `return_index=True` to pick
    # the first (smallest-range) entry per pixel.
    flat_idx = v * num_azimuth + u
    order = np.argsort(ranges)              # ascending
    flat_sorted = flat_idx[order]
    range_sorted = ranges[order]
    _, first_idx = np.unique(flat_sorted, return_index=True)
    pick_flat = flat_sorted[first_idx]
    pick_range = range_sorted[first_idx]

    depth = np.zeros(num_elevation * num_azimuth, dtype=np.float32)
    depth[pick_flat] = pick_range.astype(np.float32)
    return depth.reshape(num_elevation, num_azimuth)


def _world_points_to_sensor_frame(
    points_world: np.ndarray,
    sensor_to_world: np.ndarray,
) -> np.ndarray:
    """Invert the sensor pose to bring world-frame points back into
    the sensor-local frame nvblox expects.
    `sensor_to_world` is the 4x4 camera-to-world transform.
    """
    # p_s = R_ws^T (p_w - t_ws)
    R = sensor_to_world[:3, :3]
    t = sensor_to_world[:3, 3]
    return (points_world - t[None, :]) @ R


def run_lidar(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run nvblox LiDAR TSDF on a Mai City-style sweep sequence.

    The npz at `lidar_sweeps_npz` is produced by `nvblox_bench_setup_mai_city`
    in the driver -- it's a dict with:
      - `points_per_frame_concat`: [total_points, 3] fp32, all frames concatenated
      - `points_per_frame_offsets`: [n_frames + 1] int64 (csr-style offsets)
      - `sensor_origins`: [n_frames, 3] fp32
      - `cam_to_world`: [n_frames, 4, 4] fp32
    This indirection is much faster than JSON-encoding 700 x 100k points.
    """
    voxel_size_m = float(spec["voxel_size_m"])
    trunc_m = float(spec["truncation_distance_m"])
    num_azimuth = int(spec["lidar_num_azimuth"])
    num_elevation = int(spec["lidar_num_elevation"])
    vertical_fov_rad = float(spec["lidar_vertical_fov_rad"])
    min_valid_range_m = float(spec.get("lidar_min_valid_range_m", 1.0))
    warmup_frames = int(spec.get("warmup_frames", 2))

    data = np.load(spec["lidar_sweeps_npz"])
    pts_concat = data["points_per_frame_concat"]          # [N_total, 3]
    pts_offsets = data["points_per_frame_offsets"]        # [n_frames + 1]
    sensor_origins = data["sensor_origins"]               # [n_frames, 3]
    cam_to_world_np = data["cam_to_world"]                # [n_frames, 4, 4]
    n_frames = len(pts_offsets) - 1

    # Build the nvblox mapper up-front.
    params = MapperParams()
    # Set truncation distance if the API exposes it. nvblox's default is
    # 4 * voxel_size which may or may not match our desired 3x multiplier;
    # configure explicitly so the comparison is fair.
    try:
        # The param name is `truncation_distance_vox` (in voxels) in
        # recent nvblox. Compute ratio.
        vox_ratio = trunc_m / voxel_size_m
        params.tsdf_integrator_truncation_distance_vox = float(vox_ratio)
    except AttributeError:
        # Older API -- skip and accept default.
        pass

    mapper = Mapper(
        voxel_sizes_m=[voxel_size_m],
        integrator_types=[ProjectiveIntegratorType.TSDF],
        mapper_parameters=params,
    )

    sensor = Sensor.from_lidar(
        num_azimuth_divisions=num_azimuth,
        num_elevation_divisions=num_elevation,
        vertical_fov_rad=vertical_fov_rad,
        min_valid_range_m=min_valid_range_m,
    )

    # Warmup on the first two frames to pre-allocate nvblox's block pool.
    def one_frame(i: int) -> None:
        start, end = int(pts_offsets[i]), int(pts_offsets[i + 1])
        pts_w = pts_concat[start:end]
        pose = cam_to_world_np[i]
        pts_s = _world_points_to_sensor_frame(pts_w, pose)
        depth_img = _points_to_spherical_range_image(
            pts_s, num_azimuth, num_elevation, vertical_fov_rad)
        depth_t = torch.from_numpy(depth_img).cuda()
        pose_t = torch.from_numpy(pose).float()
        mapper.add_depth_frame(
            depth_frame=depth_t,
            t_w_c=pose_t,
            sensor=sensor,
            mapper_id=0,
        )

    for i in range(min(warmup_frames, n_frames)):
        one_frame(i)
    mapper.clear(mapper_id=0)
    torch.cuda.synchronize()
    gc.collect()

    base_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_frames):
        one_frame(i)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    ms_per_f = wall_s * 1000 / n_frames

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Voxel count: use the TSDF layer's active-block count * 512 as a
    # conservative upper bound on active voxels. nvblox stores an 8^3
    # block as a unit; voxels inside a block with zero weight still
    # "count" toward the allocation, so this inflates vs fvdb's
    # strict surface-only narrow-band number. We report it but call
    # out the caveat in the bench driver.
    tsdf_view = mapper.tsdf_layer_view(mapper_id=0)
    n_blocks = None
    try:
        n_blocks = int(tsdf_view.num_allocated_blocks())
    except Exception:
        try:
            n_blocks = int(tsdf_view.num_blocks())
        except Exception:
            n_blocks = None
    # nvblox uses 8^3 voxels per block by default; let the layer
    # report the actual block-dim-in-voxels so we do the right math
    # even if someone changes the block size in the future.
    block_dim_vox = 8
    try:
        block_dim_vox = int(tsdf_view.block_dim_in_voxels())
    except Exception:
        pass
    n_voxels_upper = (
        n_blocks * (block_dim_vox ** 3) if n_blocks is not None else -1)

    # Approximate GPU-mem peak. `num_allocated_bytes()` reports the
    # live size of the TSDF layer only; we want the total nvblox
    # allocator footprint. Torch's cuda memory API works in-process
    # even though nvblox uses its own allocator pool, because nvblox
    # goes through the same CUDA driver -- `mem_get_info()` sees the
    # aggregate used memory.
    gpu_used_gb = -1.0
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        gpu_used_gb = (total_b - free_b) / 1e9
    except Exception:
        pass
    peak_torch_gb = -1.0
    try:
        peak_torch_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass

    # Mesh extraction for quality reference.
    n_mesh_verts = n_mesh_tris = -1
    try:
        mapper.update_color_mesh(mapper_id=0)
        mesh = mapper.get_color_mesh(mapper_id=0)
        n_mesh_verts = int(mesh.vertices.shape[0])
        n_mesh_tris = int(mesh.triangles.shape[0])
    except Exception:
        pass

    # Optional: time nvblox's ESDF-from-TSDF update step.
    #
    # Important caveat about `Mapper.update_esdf`: nvblox's ESDF is
    # INCREMENTAL BY DEFAULT. The first call after a new TSDF state
    # does the real work (building the ESDF across all dirty blocks);
    # subsequent calls on the same unchanged TSDF state hit an
    # internal "no dirty blocks" fast-path and take ~0.05 ms (just
    # the dirty-block check). So we time two things separately:
    #
    #   - `esdf_cold_ms`: the very first `update_esdf` call after
    #     TSDF fusion. This is the cost of "build the whole ESDF
    #     from scratch" — directly comparable to fvdb's stateless
    #     `compute_esdf`.
    #   - `esdf_warm_ms_*`: subsequent calls on the same TSDF state.
    #     These are the dirty-block-check no-op cost — directly
    #     comparable to fvdb's `compute_esdf_incremental` on a static
    #     scene (idempotent warm-start).
    esdf_cold_ms = -1.0
    esdf_warm_ms_min = -1.0
    esdf_warm_ms_median = -1.0
    esdf_warm_calls  = int(spec.get("esdf_warm_calls", 5))
    if bool(spec.get("with_esdf", False)):
        try:
            # Cold call: the only call where nvblox actually builds
            # the ESDF across all blocks. This is the cost we want
            # to compare against fvdb's one-shot `compute_esdf`.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mapper.update_esdf(mapper_id=0)
            torch.cuda.synchronize()
            esdf_cold_ms = (time.perf_counter() - t0) * 1000.0

            # Warm calls: same TSDF state, no dirty blocks. Should
            # all be near-zero because nvblox short-circuits on the
            # dirty-block check.
            warm_samples = []
            for _ in range(esdf_warm_calls):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                mapper.update_esdf(mapper_id=0)
                torch.cuda.synchronize()
                warm_samples.append((time.perf_counter() - t0) * 1000.0)
            if warm_samples:
                warm_samples.sort()
                esdf_warm_ms_min = warm_samples[0]
                esdf_warm_ms_median = warm_samples[len(warm_samples) // 2]
        except Exception:
            # ESDF update may not be available on every mapper config.
            esdf_cold_ms = -2.0
            esdf_warm_ms_min = -2.0
            esdf_warm_ms_median = -2.0

    return {
        "ok": True,
        "ms_per_f": ms_per_f,
        "wall_s": wall_s,
        "peak_rss_gb": peak_rss_kb / 1e6,
        "peak_rss_delta_gb": max(0.0, (peak_rss_kb - base_rss_kb) / 1e6),
        "gpu_used_gb": gpu_used_gb,
        "peak_torch_gb": peak_torch_gb,
        "n_frames": n_frames,
        "n_voxels": n_voxels_upper,
        "n_blocks": n_blocks if n_blocks is not None else -1,
        "n_mesh_verts": n_mesh_verts,
        "n_mesh_tris": n_mesh_tris,
        "voxel_size_m": voxel_size_m,
        "esdf_cold_ms": esdf_cold_ms,
        "esdf_warm_ms_min": esdf_warm_ms_min,
        "esdf_warm_ms_median": esdf_warm_ms_median,
        "esdf_warm_calls": esdf_warm_calls,
    }


def run_lidar_occupancy(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run nvblox LiDAR occupancy (log-odds) integration on a Mai
    City-style sweep sequence. Simpler than `run_lidar` (which does
    TSDF + optional ESDF + optional mesh) — occupancy is a single
    integrator type with a single per-voxel quantity, so we just:

      1. Build a Mapper with ProjectiveIntegratorType.OCCUPANCY.
      2. Feed each sweep through add_depth_frame (same spherical-
         range-image proxy as run_lidar).
      3. Report per-frame wall-clock, final block count, GPU memory.

    No ESDF, no mesh. For the paper's scale-ceiling comparison
    against fvdb's `integrate_occupancy_from_points_frames`.
    """
    voxel_size_m = float(spec["voxel_size_m"])
    trunc_m = float(spec["truncation_distance_m"])
    num_azimuth = int(spec["lidar_num_azimuth"])
    num_elevation = int(spec["lidar_num_elevation"])
    vertical_fov_rad = float(spec["lidar_vertical_fov_rad"])
    min_valid_range_m = float(spec.get("lidar_min_valid_range_m", 1.0))
    warmup_frames = int(spec.get("warmup_frames", 2))

    data = np.load(spec["lidar_sweeps_npz"])
    pts_concat = data["points_per_frame_concat"]
    pts_offsets = data["points_per_frame_offsets"]
    cam_to_world_np = data["cam_to_world"]
    n_frames = len(pts_offsets) - 1

    params = MapperParams()
    # nvblox's occupancy integrator has its own truncation param
    # (distinct from the TSDF one). Try the common spellings;
    # fall back to default if neither exists on this build.
    for attr in (
        "occupancy_integrator_truncation_distance_vox",
        "occupancy_integrator_max_integration_distance_m",
    ):
        try:
            if "vox" in attr:
                setattr(params, attr, float(trunc_m / voxel_size_m))
            # max_integration_distance is different semantic; skip
        except AttributeError:
            pass

    mapper = Mapper(
        voxel_sizes_m=[voxel_size_m],
        integrator_types=[ProjectiveIntegratorType.OCCUPANCY],
        mapper_parameters=params,
    )

    sensor = Sensor.from_lidar(
        num_azimuth_divisions=num_azimuth,
        num_elevation_divisions=num_elevation,
        vertical_fov_rad=vertical_fov_rad,
        min_valid_range_m=min_valid_range_m,
    )

    def one_frame(i: int) -> None:
        start, end = int(pts_offsets[i]), int(pts_offsets[i + 1])
        pts_w = pts_concat[start:end]
        pose = cam_to_world_np[i]
        pts_s = _world_points_to_sensor_frame(pts_w, pose)
        depth_img = _points_to_spherical_range_image(
            pts_s, num_azimuth, num_elevation, vertical_fov_rad)
        depth_t = torch.from_numpy(depth_img).cuda()
        pose_t = torch.from_numpy(pose).float()
        mapper.add_depth_frame(
            depth_frame=depth_t, t_w_c=pose_t,
            sensor=sensor, mapper_id=0,
        )

    for i in range(min(warmup_frames, n_frames)):
        one_frame(i)
    mapper.clear(mapper_id=0)
    torch.cuda.synchronize()
    gc.collect()

    base_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_frames):
        one_frame(i)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    ms_per_f = wall_s * 1000 / n_frames

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Layer-size reporting. Occupancy uses a different layer view
    # than TSDF. Try the likely method names; tolerate absence.
    n_blocks = None
    block_dim_vox = 8
    for view_getter in ("occupancy_layer_view", "tsdf_layer_view"):
        try:
            v = getattr(mapper, view_getter)(mapper_id=0)
            try:
                n_blocks = int(v.num_allocated_blocks())
            except Exception:
                try:
                    n_blocks = int(v.num_blocks())
                except Exception:
                    continue
            try:
                block_dim_vox = int(v.block_dim_in_voxels())
            except Exception:
                pass
            break
        except Exception:
            continue
    n_voxels_upper = (
        n_blocks * (block_dim_vox ** 3) if n_blocks is not None else -1)

    gpu_used_gb = -1.0
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        gpu_used_gb = (total_b - free_b) / 1e9
    except Exception:
        pass
    peak_torch_gb = -1.0
    try:
        peak_torch_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass

    return {
        "ok": True,
        "ms_per_f": ms_per_f,
        "wall_s": wall_s,
        "peak_rss_gb": peak_rss_kb / 1e6,
        "peak_rss_delta_gb": max(0.0, (peak_rss_kb - base_rss_kb) / 1e6),
        "gpu_used_gb": gpu_used_gb,
        "peak_torch_gb": peak_torch_gb,
        "n_frames": n_frames,
        "n_voxels": n_voxels_upper,
        "n_blocks": n_blocks if n_blocks is not None else -1,
        "voxel_size_m": voxel_size_m,
    }


def run_depth(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run nvblox depth-TSDF on a Replica / 7-Scenes-style sequence.

    The npz at `depth_npz` is produced by the driver and holds:
      - `depth_images`: [N, H, W] fp32 in METRES (0 = no measurement)
      - `cam_to_world`: [N, 4, 4] fp32 camera-to-world poses
      - `K`:            [3, 3] fp32 intrinsics matrix
            K[0,0]=fx, K[1,1]=fy, K[0,2]=cx, K[1,2]=cy
    Mirrors the indirection used by `run_lidar` — JSON-encoding large
    depth arrays is too slow.

    nvblox input format:
      - `Sensor.from_camera_matrix(K, W, H)`.
      - `Mapper.add_depth_frame(depth_frame, t_w_c, sensor)`, where
        `depth_frame` is a CUDA fp32 tensor and `t_w_c` is a 4x4
        host tensor (matching the test_mapper_add_frames example).

    Timing protocol and ESDF handling are identical to `run_lidar`.
    """
    voxel_size_m = float(spec["voxel_size_m"])
    trunc_m = float(spec["truncation_distance_m"])
    warmup_frames = int(spec.get("warmup_frames", 2))

    data = np.load(spec["depth_npz"])
    depth_images = data["depth_images"]   # [N, H, W] fp32 metres
    cam_to_world = data["cam_to_world"]   # [N, 4, 4] fp32
    K_np         = data["K"]              # [3, 3] fp32
    n_frames, height, width = depth_images.shape

    params = MapperParams()
    try:
        vox_ratio = trunc_m / voxel_size_m
        params.tsdf_integrator_truncation_distance_vox = float(vox_ratio)
    except AttributeError:
        pass

    mapper = Mapper(
        voxel_sizes_m=[voxel_size_m],
        integrator_types=[ProjectiveIntegratorType.TSDF],
        mapper_parameters=params,
    )

    K_t = torch.from_numpy(K_np).to(dtype=torch.float32)
    sensor = Sensor.from_camera_matrix(K_t, width, height)

    def one_frame(i: int) -> None:
        depth_t = torch.from_numpy(depth_images[i]).cuda()
        pose_t  = torch.from_numpy(cam_to_world[i]).float()
        mapper.add_depth_frame(
            depth_frame=depth_t, t_w_c=pose_t,
            sensor=sensor, mapper_id=0,
        )

    for i in range(min(warmup_frames, n_frames)):
        one_frame(i)
    mapper.clear(mapper_id=0)
    torch.cuda.synchronize()
    gc.collect()

    base_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_frames):
        one_frame(i)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    ms_per_f = wall_s * 1000 / n_frames

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    tsdf_view = mapper.tsdf_layer_view(mapper_id=0)
    n_blocks = None
    try:
        n_blocks = int(tsdf_view.num_allocated_blocks())
    except Exception:
        try:
            n_blocks = int(tsdf_view.num_blocks())
        except Exception:
            n_blocks = None
    block_dim_vox = 8
    try:
        block_dim_vox = int(tsdf_view.block_dim_in_voxels())
    except Exception:
        pass
    n_voxels_upper = (
        n_blocks * (block_dim_vox ** 3) if n_blocks is not None else -1)

    gpu_used_gb = -1.0
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        gpu_used_gb = (total_b - free_b) / 1e9
    except Exception:
        pass
    peak_torch_gb = -1.0
    try:
        peak_torch_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass

    n_mesh_verts = n_mesh_tris = -1
    try:
        mapper.update_color_mesh(mapper_id=0)
        mesh = mapper.get_color_mesh(mapper_id=0)
        n_mesh_verts = int(mesh.vertices.shape[0])
        n_mesh_tris  = int(mesh.triangles.shape[0])
    except Exception:
        pass

    # ESDF timing (same cold/warm split as run_lidar).
    esdf_cold_ms = -1.0
    esdf_warm_ms_min = -1.0
    esdf_warm_ms_median = -1.0
    esdf_warm_calls = int(spec.get("esdf_warm_calls", 5))
    if bool(spec.get("with_esdf", False)):
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mapper.update_esdf(mapper_id=0)
            torch.cuda.synchronize()
            esdf_cold_ms = (time.perf_counter() - t0) * 1000.0

            warm_samples = []
            for _ in range(esdf_warm_calls):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                mapper.update_esdf(mapper_id=0)
                torch.cuda.synchronize()
                warm_samples.append((time.perf_counter() - t0) * 1000.0)
            if warm_samples:
                warm_samples.sort()
                esdf_warm_ms_min = warm_samples[0]
                esdf_warm_ms_median = warm_samples[len(warm_samples) // 2]
        except Exception:
            esdf_cold_ms = -2.0
            esdf_warm_ms_min = -2.0
            esdf_warm_ms_median = -2.0

    return {
        "ok": True,
        "ms_per_f": ms_per_f,
        "wall_s": wall_s,
        "peak_rss_gb": peak_rss_kb / 1e6,
        "peak_rss_delta_gb": max(0.0, (peak_rss_kb - base_rss_kb) / 1e6),
        "gpu_used_gb": gpu_used_gb,
        "peak_torch_gb": peak_torch_gb,
        "n_frames": n_frames,
        "n_voxels": n_voxels_upper,
        "n_blocks": n_blocks if n_blocks is not None else -1,
        "n_mesh_verts": n_mesh_verts,
        "n_mesh_tris": n_mesh_tris,
        "voxel_size_m": voxel_size_m,
        "esdf_cold_ms": esdf_cold_ms,
        "esdf_warm_ms_min": esdf_warm_ms_min,
        "esdf_warm_ms_median": esdf_warm_ms_median,
        "esdf_warm_calls": esdf_warm_calls,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", required=True, help="Path to JSON spec file")
    p.add_argument("--output", required=True, help="Path to JSON output file")
    args = p.parse_args()

    with open(args.spec, "r") as f:
        spec = json.load(f)

    try:
        if spec["workload"] == "lidar":
            result = run_lidar(spec)
        elif spec["workload"] == "lidar_occupancy":
            result = run_lidar_occupancy(spec)
        elif spec["workload"] == "depth":
            result = run_depth(spec)
        else:
            raise NotImplementedError(
                f"workload {spec['workload']!r} not implemented (supported: "
                "'lidar', 'lidar_occupancy', 'depth')")
    except Exception as e:  # noqa: BLE001
        result = {
            "ok": False,
            "failure": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "voxel_size_m": spec.get("voxel_size_m", -1),
        }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
