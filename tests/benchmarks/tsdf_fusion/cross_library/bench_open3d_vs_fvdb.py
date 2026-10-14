# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
fvdb vs Open3D smoke comparison on a synthetic RGB-D sphere scene.

Workload: N views of a unit-radius-ish sphere rendered from random
camera poses via analytical ray-sphere intersection. No external
dataset download; runs in a few seconds.

Measures for each library:
  - Wall-clock: per-frame integration time, mesh extraction time.
  - Peak memory: GPU (fvdb) or host (Open3D, CPU).
  - Mesh size: vertex count, triangle count.
  - Reconstruction quality: symmetric Chamfer-L1 vs GT sphere surface
    (uniform-sampled), F-score at tau = voxel_size.

Useful as a head-to-head smoke test that doesn't require the nvblox
or VDBFusion Docker infrastructure.
"""

from __future__ import annotations

import os

# Default to PyTorch's expandable-segments allocator *before* importing
# torch, since the allocator config is read at first CUDA initialization
# and setting it later has no effect. At Replica-scale (N=200 frames,
# 1200x680) the TSDF truncation-shell build fragments enough of the
# default caching allocator that fvdb can OOM on office0 / room1 even
# though <10 GB is genuinely live; `expandable_segments=True` lets torch
# grow existing blocks instead of reserving new ones, reclaiming the
# ~10-15 GB of fragmented "reserved but unallocated" space and unblocking
# N=200 on every Replica scene we benchmark. Overridable via the env var.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import dataclasses
import gc
import math
import resource
import time
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch

import fvdb
from fvdb import Grid, JaggedTensor

try:
    from .replica_loader import load_replica_scene, ReplicaScene
except ImportError:
    # Allow running as a script from the parent dir.
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from replica_loader import load_replica_scene, ReplicaScene  # type: ignore


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------


def _look_at(eye: np.ndarray, target: np.ndarray, up_world: np.ndarray) -> np.ndarray:
    """Return a cam->world 4x4 matrix placing the camera at `eye` looking at `target`."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_world)
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, forward)
    # OpenCV camera convention: +x right, +y down, +z forward.
    R = np.stack([right, -up, forward], axis=1)
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = R
    cam_to_world[:3, 3] = eye
    return cam_to_world


def _render_sphere_depth(
    sphere_center: np.ndarray,
    sphere_radius: float,
    cam_to_world: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
    max_depth: float,
) -> np.ndarray:
    """Analytical ray-sphere depth renderer.

    Returns a (H, W) float32 depth image. Pixels that miss the sphere
    get 0 (no-measurement convention used by both libraries).
    """
    # Build per-pixel ray directions in camera space (OpenCV: +z forward).
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x)
    dirs_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1, keepdims=True)

    # Transform to world.
    R = cam_to_world[:3, :3]
    origin = cam_to_world[:3, 3]
    dirs_world = dirs_cam @ R.T

    # Ray-sphere intersection. (O + t*d - c) · (O + t*d - c) = r^2
    oc = origin - sphere_center
    b = 2.0 * (dirs_world @ oc)
    c = oc @ oc - sphere_radius * sphere_radius
    disc = b * b - 4.0 * c
    hit = disc >= 0.0
    disc_clamped = np.where(hit, disc, 0.0)
    t = (-b - np.sqrt(disc_clamped)) / 2.0
    depth_cam = t * dirs_cam[:, 2]  # depth (z) along camera forward
    depth_cam = np.where((t > 0) & hit & (depth_cam < max_depth), depth_cam, 0.0)
    return depth_cam.reshape(H, W).astype(np.float32)


@dataclasses.dataclass
class SyntheticScene:
    depth_images: np.ndarray  # [N, H, W] float32, 0 = miss
    cam_to_world: np.ndarray  # [N, 4, 4] float32
    K: np.ndarray  # [3, 3] float32 (shared intrinsics)
    sphere_center: np.ndarray  # [3] float32
    sphere_radius: float


def make_sphere_scene(
    n_frames: int = 16,
    W: int = 320,
    H: int = 240,
    sphere_radius: float = 0.5,
    sphere_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cam_radius: float = 1.5,
    max_depth: float = 5.0,
    fov_deg: float = 60.0,
    seed: int = 0,
) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    sphere_center_np = np.asarray(sphere_center, dtype=np.float32)

    # Fibonacci-lattice cameras on a sphere around the target.
    idx = np.arange(n_frames) + 0.5
    phi = idx * math.pi * (3.0 - math.sqrt(5.0))
    cos_theta = 1.0 - 2.0 * idx / n_frames
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))
    cams = np.stack(
        [
            cam_radius * np.cos(phi) * sin_theta,
            cam_radius * np.sin(phi) * sin_theta,
            cam_radius * cos_theta,
        ],
        axis=1,
    ).astype(np.float32)
    # Jitter the cameras slightly to avoid perfect symmetry masking bugs.
    cams += 0.02 * rng.standard_normal(cams.shape).astype(np.float32)

    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cam_to_worlds = np.stack(
        [_look_at(eye, sphere_center_np, up_world) for eye in cams], axis=0
    ).astype(np.float32)

    # Shared intrinsics.
    fx = fy = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1]], dtype=np.float32)

    depth_images = np.stack(
        [
            _render_sphere_depth(
                sphere_center_np, sphere_radius, c2w, K, W, H, max_depth
            )
            for c2w in cam_to_worlds
        ],
        axis=0,
    )

    return SyntheticScene(
        depth_images=depth_images,
        cam_to_world=cam_to_worlds,
        K=K,
        sphere_center=sphere_center_np,
        sphere_radius=sphere_radius,
    )


# ---------------------------------------------------------------------------
# Per-library integration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunResult:
    library: str
    integrate_s: float
    mesh_extract_s: float
    mesh_V: int
    mesh_F: int
    mesh_verts: np.ndarray  # [V, 3] float32 in world space
    peak_gpu_mb: float
    peak_host_mb: float


def _peak_host_mb() -> float:
    """Coarse peak host-memory sample via RUSAGE_SELF.ru_maxrss (KiB on Linux).

    Only meaningful as a *delta* across a library call — the process
    baseline includes libtorch + CUDA runtime, which is constant and
    dominates the absolute value.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _estimate_scene_bbox(scene) -> tuple[np.ndarray, np.ndarray]:
    """Return world-space ``(min_xyz, max_xyz)`` enclosing all
    sensor positions + a simple depth-extent margin.

    This sizes the initial dense bounding-box grid for fvdb without
    requiring the caller to hand-tune `grid_extent` per scene. For
    the synthetic sphere we'd happily use a fixed `[-1.5, 1.5]^3`;
    for Replica rooms we need something scene-dependent.
    """
    centres = scene.cam_to_world[:, :3, 3]
    bbox_min = centres.min(axis=0).astype(np.float32)
    bbox_max = centres.max(axis=0).astype(np.float32)
    # Expand by the p95 of observed depth so any ray endpoint lands
    # inside the initial grid. 99th percentile would be more
    # conservative but spikes from stray pixels blow up the extent.
    valid = scene.depth_images[scene.depth_images > 0]
    margin = float(np.percentile(valid, 95)) if valid.size else 1.0
    bbox_min = bbox_min - margin
    bbox_max = bbox_max + margin
    return bbox_min, bbox_max


def integrate_fvdb(
    scene,
    voxel_size: float,
    truncation: float,
    device: str = "cuda",
    grid_extent: float | None = None,
    mode: str = "per_frame",
    start_empty: bool = False,
) -> RunResult:
    """Integrate all frames into a TSDF volume.

    `mode`:
      - "per_frame": loop N calls to `Grid.integrate_tsdf` (rebuilds
        topology per frame). Legacy / naive path.
      - "frames":    one call to `Grid.integrate_tsdf_frames` (builds
        topology once over all frames). (one-shot N-frame topology).

    `grid_extent` / `start_empty`:
      - start_empty=True: initial grid is a minimal 1x1x1 (same as
        the LiDAR integrator's convention) — the integrate call's
        union-topology build grows it to cover the scene. Required
        for room-scale scenes where a dense bbox would OOM.
      - grid_extent=float: half-extent of a unit-box dense grid at
        origin (legacy, kept for the synthetic sphere smoke).
      - grid_extent=None: auto-estimate dense bbox from scene sensor
        positions + depth p95.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)
    gc.collect()
    host_before = _peak_host_mb()

    if start_empty:
        # 1x1x1 minimal grid at origin — metadata (voxel_size, origin)
        # carries through to the union grid; no up-front dense allocation.
        grid = Grid.from_dense(
            dense_dims=[1, 1, 1],
            ijk_min=[0, 0, 0],
            voxel_size=voxel_size,
            origin=[0, 0, 0],
            device=device,
        )
    else:
        if grid_extent is None:
            bbox_min_np, bbox_max_np = _estimate_scene_bbox(scene)
        else:
            bbox_min_np = np.array([-grid_extent] * 3, np.float32)
            bbox_max_np = np.array([grid_extent] * 3, np.float32)

        side_xyz = np.ceil((bbox_max_np - bbox_min_np) / voxel_size).astype(np.int64)
        side_xyz = np.maximum(side_xyz, 1)
        ijk_min = np.floor(bbox_min_np / voxel_size).astype(np.int64)
        grid = Grid.from_dense(
            dense_dims=[int(side_xyz[0]), int(side_xyz[1]), int(side_xyz[2])],
            ijk_min=[int(ijk_min[0]), int(ijk_min[1]), int(ijk_min[2])],
            voxel_size=voxel_size,
            origin=[0, 0, 0],
            device=device,
        )
    tsdf = torch.zeros(grid.num_voxels, device=device, dtype=torch.float32)
    weights = torch.zeros(grid.num_voxels, device=device, dtype=torch.float32)

    # Pre-upload camera matrices + depth (warm the caches before timing).
    K = torch.from_numpy(scene.K).to(device)
    cam_to_world = torch.from_numpy(scene.cam_to_world).to(device)
    depth_images = torch.from_numpy(scene.depth_images).to(device)
    torch.cuda.synchronize()

    N = scene.depth_images.shape[0]
    K_per_frame = K.unsqueeze(0).expand(N, 3, 3).contiguous()

    t0 = time.perf_counter()
    if mode == "frames":
        grid, tsdf, weights = grid.integrate_tsdf_frames(
            truncation_distance=truncation,
            projection_matrices=K_per_frame,
            cam_to_world_matrices=cam_to_world,
            tsdf=tsdf,
            weights=weights,
            depth_images=depth_images,
        )
    elif mode == "per_frame":
        for f in range(N):
            grid, tsdf, weights = grid.integrate_tsdf(
                truncation_distance=truncation,
                projection_matrices=K.unsqueeze(0),
                cam_to_world_matrices=cam_to_world[f : f + 1],
                tsdf=tsdf,
                weights=weights,
                depth_images=depth_images[f : f + 1],
            )
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'per_frame' or 'frames'")
    torch.cuda.synchronize()
    integrate_s = time.perf_counter() - t0

    # Mesh extraction should only run on observed voxels — otherwise MC
    # finds a spurious zero-crossing at the boundary between the
    # truncation band's `-1` values and the unobserved-voxel default of
    # 0. Prune the grid to `weights > 0` (the same pattern as the LiDAR
    # integrator's sphere-reconstruction test).
    observed_mask = weights > 0
    pruned_grid = grid.pruned_grid(observed_mask)
    pruned_tsdf = tsdf[observed_mask]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    v, f, _ = pruned_grid.marching_cubes(pruned_tsdf, 0.0)
    torch.cuda.synchronize()
    mesh_extract_s = time.perf_counter() - t0

    peak_gpu_mb = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)

    verts_np = v.detach().cpu().numpy()
    return RunResult(
        library=f"fvdb_{mode}",
        integrate_s=integrate_s,
        mesh_extract_s=mesh_extract_s,
        mesh_V=v.shape[0],
        mesh_F=f.shape[0],
        mesh_verts=verts_np,
        peak_gpu_mb=peak_gpu_mb,
        peak_host_mb=_peak_host_mb() - host_before,
    )


def integrate_open3d_cuda(
    scene,
    voxel_size: float,
    truncation: float,
    device: str = "cuda",
    block_resolution: int = 16,
    block_count: int = 100_000,
) -> RunResult:
    """Integrate via Open3D's `t.geometry.VoxelBlockGrid` (GPU).

    This is Open3D's "tensor" TSDF backend — CUDA-native, the
    canonical comparison for fvdb's GPU path (vs `ScalableTSDFVolume`
    which is CPU-only and therefore an unfair column on its own).
    Matches NICE-SLAM / Co-SLAM / ESLAM's Open3D-based baselines.
    """
    import open3d as o3d
    import open3d.core as o3c

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)
    gc.collect()
    host_before = _peak_host_mb()

    o3d_device = o3c.Device("CUDA:0")

    # trunc_voxel_multiplier = truncation / voxel_size; Open3D expects
    # it as an integer-ish float (default 8 -> 8 * voxel_size truncation).
    trunc_mul = float(truncation / voxel_size)

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"),
        attr_dtypes=(o3c.float32, o3c.float32),
        attr_channels=((1,), (1,)),
        voxel_size=voxel_size,
        block_resolution=block_resolution,
        block_count=block_count,
        device=o3d_device,
    )

    K_o3d = o3c.Tensor(
        np.asarray(scene.K, dtype=np.float64),
        o3c.float64, device=o3c.Device("CPU:0"),
    )

    # Warm-up: Open3D's first CUDA invocation eats ~hundreds of ms of
    # JIT; amortize it outside the timed region so per-frame numbers
    # are steady-state.
    warm_depth = o3d.t.geometry.Image(
        o3c.Tensor(scene.depth_images[0:1].astype(np.float32).squeeze(0),
                   o3c.float32, o3d_device)
    )
    warm_ext = o3c.Tensor(
        np.linalg.inv(scene.cam_to_world[0]).astype(np.float64),
        o3c.float64, device=o3c.Device("CPU:0"),
    )
    _ = vbg.compute_unique_block_coordinates(
        warm_depth, K_o3d, warm_ext, depth_scale=1.0, depth_max=10.0)
    # Reset the vbg after warmup (empty it so the timed run starts
    # from the same state as fvdb: no pre-allocated data).
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"),
        attr_dtypes=(o3c.float32, o3c.float32),
        attr_channels=((1,), (1,)),
        voxel_size=voxel_size,
        block_resolution=block_resolution,
        block_count=block_count,
        device=o3d_device,
    )

    N = scene.depth_images.shape[0]
    # Precompute extrinsics (world-to-cam inverse of cam_to_world).
    extrinsics = np.linalg.inv(scene.cam_to_world)

    # Stage depth onto device once so we're timing TSDF integration,
    # not host->device DMA. (fvdb does the same staging before its
    # timed region.)
    depth_staging = []
    for i in range(N):
        d = o3c.Tensor(scene.depth_images[i].astype(np.float32),
                       o3c.float32, o3d_device)
        depth_staging.append(o3d.t.geometry.Image(d))

    extrinsic_staging = [
        o3c.Tensor(extrinsics[i].astype(np.float64),
                   o3c.float64, device=o3c.Device("CPU:0"))
        for i in range(N)
    ]

    o3c.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N):
        depth_i = depth_staging[i]
        ext_i = extrinsic_staging[i]
        frustum = vbg.compute_unique_block_coordinates(
            depth_i, K_o3d, ext_i,
            depth_scale=1.0, depth_max=10.0,
            trunc_voxel_multiplier=trunc_mul,
        )
        vbg.integrate(
            frustum, depth_i, K_o3d, ext_i,
            depth_scale=1.0, depth_max=10.0,
            trunc_voxel_multiplier=trunc_mul,
        )
    o3c.cuda.synchronize()
    integrate_s = time.perf_counter() - t0

    o3c.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        mesh = vbg.extract_triangle_mesh()
        o3c.cuda.synchronize()
        mesh_extract_s = time.perf_counter() - t0
        verts = mesh.vertex.positions.cpu().numpy().astype(np.float32)
        tris = mesh.triangle.indices.cpu().numpy().astype(np.int64)
    except RuntimeError as e:
        # Open3D's VBG.extract_triangle_mesh asserts on SetVertexColors
        # shape when the grid is empty / too-tiny to have extracted any
        # verts. Tolerate that in the warmup / degenerate-scene case.
        o3c.cuda.synchronize()
        mesh_extract_s = time.perf_counter() - t0
        verts = np.empty((0, 3), np.float32)
        tris = np.empty((0, 3), np.int64)

    peak_gpu_mb = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
    return RunResult(
        library="open3d_cuda",
        integrate_s=integrate_s,
        mesh_extract_s=mesh_extract_s,
        mesh_V=verts.shape[0],
        mesh_F=tris.shape[0],
        mesh_verts=verts,
        peak_gpu_mb=peak_gpu_mb,
        peak_host_mb=_peak_host_mb() - host_before,
    )


def integrate_open3d(
    scene: SyntheticScene,
    voxel_size: float,
    truncation: float,
) -> RunResult:
    """Integrate via Open3D's ScalableTSDFVolume (CPU)."""
    gc.collect()
    host_before = _peak_host_mb()

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=truncation,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )

    W = scene.depth_images.shape[2]
    H = scene.depth_images.shape[1]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        W, H,
        fx=float(scene.K[0, 0]),
        fy=float(scene.K[1, 1]),
        cx=float(scene.K[0, 2]),
        cy=float(scene.K[1, 2]),
    )

    t0 = time.perf_counter()
    for f in range(scene.depth_images.shape[0]):
        depth = o3d.geometry.Image(scene.depth_images[f])
        # Dummy colour so the RGBDImage constructor is happy.
        dummy_color = o3d.geometry.Image(
            np.zeros((H, W, 3), dtype=np.uint8)
        )
        # `depth_trunc` is the "beyond this range, treat as no measurement"
        # cap — not the TSDF truncation-band width. Set it well past our
        # scene extent so no pixels are silently dropped (fvdb's
        # integrate_tsdf doesn't have an equivalent parameter, so it
        # integrates every non-zero pixel).
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            dummy_color,
            depth,
            depth_scale=1.0,
            depth_trunc=1e6,
            convert_rgb_to_intensity=False,
        )
        # Open3D wants world->cam (extrinsic), not cam->world.
        extrinsic = np.linalg.inv(scene.cam_to_world[f])
        volume.integrate(rgbd, intrinsics, extrinsic)
    integrate_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh = volume.extract_triangle_mesh()
    mesh_extract_s = time.perf_counter() - t0

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris = np.asarray(mesh.triangles, dtype=np.int64)

    return RunResult(
        library="open3d",
        integrate_s=integrate_s,
        mesh_extract_s=mesh_extract_s,
        mesh_V=verts.shape[0],
        mesh_F=tris.shape[0],
        mesh_verts=verts,
        peak_gpu_mb=0.0,
        peak_host_mb=_peak_host_mb() - host_before,
    )


# ---------------------------------------------------------------------------
# Quality metrics vs GT sphere
# ---------------------------------------------------------------------------


def _sample_sphere_surface(
    n_points: int, center: np.ndarray, radius: float, rng: np.random.Generator
) -> np.ndarray:
    """Uniform-area sample n_points on a sphere surface."""
    phi = rng.uniform(0, 2 * math.pi, n_points)
    cos_theta = rng.uniform(-1, 1, n_points)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    pts = np.stack(
        [
            radius * np.cos(phi) * sin_theta,
            radius * np.sin(phi) * sin_theta,
            radius * cos_theta,
        ],
        axis=1,
    ).astype(np.float32)
    pts += center
    return pts


def _chunked_nn_distances(
    points_a: torch.Tensor, points_b: torch.Tensor, chunk: int = 4096
) -> torch.Tensor:
    """For each point in `points_a`, return min L2 distance to `points_b`.

    Chunked to avoid an `|A|*|B|` pairwise matrix blowing up memory on
    Replica-scale meshes (hundreds-of-thousands of verts x sampled-GT).
    """
    out = torch.empty(points_a.shape[0], device=points_a.device)
    for s in range(0, points_a.shape[0], chunk):
        e = min(s + chunk, points_a.shape[0])
        dmat = torch.cdist(points_a[s:e], points_b)
        out[s:e] = dmat.min(dim=1).values
    return out


def _compute_chamfer_f_score(
    mesh_verts_np: np.ndarray,
    gt_np: np.ndarray,
    f_score_tau: float,
    device: str,
) -> Dict[str, float]:
    if mesh_verts_np.shape[0] == 0 or gt_np.shape[0] == 0:
        return {
            "chamfer_l1": float("nan"),
            "f_score": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }
    mesh_t = torch.from_numpy(mesh_verts_np).to(device)
    gt_t = torch.from_numpy(gt_np).to(device)
    d_m_to_gt = _chunked_nn_distances(mesh_t, gt_t)
    d_gt_to_m = _chunked_nn_distances(gt_t, mesh_t)
    chamfer_l1 = (d_m_to_gt.mean().item() + d_gt_to_m.mean().item()) * 0.5
    prec = (d_m_to_gt < f_score_tau).float().mean().item()
    recall = (d_gt_to_m < f_score_tau).float().mean().item()
    f_score = 2 * prec * recall / (prec + recall + 1e-12)
    return {
        "chamfer_l1": chamfer_l1,
        "f_score": f_score,
        "precision": prec,
        "recall": recall,
    }


def mesh_quality_vs_gt_mesh(
    mesh_verts: np.ndarray,
    gt_mesh_path: str,
    n_gt_samples: int = 200_000,
    f_score_tau: float = 0.02,
    device: str = "cuda",
) -> Dict[str, float]:
    """Symmetric Chamfer-L1 + F-score vs a GT mesh `.ply`, point-sampled."""
    import open3d as o3d

    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    if len(gt_mesh.triangles) == 0:
        raise ValueError(f"GT mesh at {gt_mesh_path!r} has no triangles")
    # Uniform surface-area sampling for GT — this is the canonical
    # protocol used by NICE-SLAM / Co-SLAM / ESLAM / GO-Surf etc.
    gt_pcd = gt_mesh.sample_points_uniformly(n_gt_samples, seed=0)
    gt_np = np.asarray(gt_pcd.points, dtype=np.float32)
    return _compute_chamfer_f_score(mesh_verts, gt_np, f_score_tau, device)


def mesh_quality_vs_gt(
    mesh_verts: np.ndarray,
    scene: SyntheticScene,
    n_gt_samples: int = 20_000,
    f_score_tau: float = 0.02,
    device: str = "cuda",
) -> Dict[str, float]:
    """Symmetric Chamfer-L1 + F-score vs uniformly-sampled GT sphere points.

    Operates on GPU via torch.cdist for speed.
    """
    if mesh_verts.shape[0] == 0:
        return {"chamfer_l1": float("nan"), "f_score_tau": float("nan")}

    rng = np.random.default_rng(0)
    gt = _sample_sphere_surface(n_gt_samples, scene.sphere_center, scene.sphere_radius, rng)

    mesh_t = torch.from_numpy(mesh_verts).to(device)
    gt_t = torch.from_numpy(gt).to(device)

    # Chunked cdist to avoid V * G memory blow-up.
    chunk = 4096
    d_m_to_gt = torch.empty(mesh_t.shape[0], device=device)
    d_gt_to_m = torch.empty(gt_t.shape[0], device=device)
    for s in range(0, mesh_t.shape[0], chunk):
        e = min(s + chunk, mesh_t.shape[0])
        dmat = torch.cdist(mesh_t[s:e], gt_t)
        d_m_to_gt[s:e] = dmat.min(dim=1).values
    for s in range(0, gt_t.shape[0], chunk):
        e = min(s + chunk, gt_t.shape[0])
        dmat = torch.cdist(gt_t[s:e], mesh_t)
        d_gt_to_m[s:e] = dmat.min(dim=1).values

    chamfer_l1 = (d_m_to_gt.mean().item() + d_gt_to_m.mean().item()) * 0.5

    prec = (d_m_to_gt < f_score_tau).float().mean().item()
    recall = (d_gt_to_m < f_score_tau).float().mean().item()
    f_score = 2 * prec * recall / (prec + recall + 1e-12)

    return {
        "chamfer_l1": chamfer_l1,
        "f_score": f_score,
        "precision": prec,
        "recall": recall,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    # Scene selection.
    p.add_argument(
        "--scene",
        type=str,
        default=None,
        help=(
            "Path to a NICE-SLAM Replica scene root (containing `results/` "
            "and `traj.txt`). If omitted, a synthetic sphere scene is used."
        ),
    )
    p.add_argument(
        "--gt-mesh",
        type=str,
        default=None,
        help=(
            "Path to the scene's GT mesh `.ply` (Replica ships these under "
            "the original Replica-Dataset tree, e.g. `room_0/mesh.ply`). "
            "If provided, mesh quality vs this mesh is reported. If omitted "
            "on a Replica scene, quality metrics are skipped."
        ),
    )
    # Synthetic-only args (ignored when --scene is given).
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument("--W", type=int, default=320)
    p.add_argument("--H", type=int, default=240)
    # Replica-only args.
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Replica scene: keep every Nth frame (1 = all).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Replica scene: cap the number of frames after stride.",
    )
    p.add_argument(
        "--depth-downsample",
        type=int,
        default=1,
        help="Replica scene: spatially downsample depth by this factor (1200/680 "
             "-> 600/340 at 2, 300/170 at 4). Intrinsics are scaled accordingly. "
             "Useful for avoiding OOM in fvdb's one-shot topology build at high "
             "frame counts (the N-frames union ingests frames*H*W points).",
    )
    # Reconstruction config.
    p.add_argument("--voxel-size", type=float, default=0.02)
    p.add_argument("--truncation", type=float, default=0.06)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--f-score-tau",
        type=float,
        default=None,
        help="F-score threshold; defaults to voxel_size.",
    )
    # Which columns to run (set to skip slow / broken paths while iterating).
    p.add_argument("--skip-per-frame", action="store_true")
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="If set, also dump the results table as JSON to this path (for "
             "the multi-scene suite runner).",
    )
    # fvdb grid init strategy. Default auto: dense bounding-box for
    # synthetic (small, matches the smoke), empty 1x1x1 for Replica
    # (dense-bbox at 2 cm for a room blows GPU memory).
    p.add_argument(
        "--fvdb-start",
        choices=("auto", "empty", "dense"),
        default="auto",
        help="fvdb initial grid: auto (empty for --scene, dense for synthetic), "
             "empty (1x1x1, union grows), or dense (auto-bbox from scene).",
    )
    args = p.parse_args()

    if args.f_score_tau is None:
        args.f_score_tau = args.voxel_size

    if args.scene:
        scene = load_replica_scene(
            args.scene, max_frames=args.max_frames, stride=args.stride
        )
        if args.depth_downsample > 1:
            # Spatially stride the depth + rescale intrinsics so the
            # world-space rays are identical (just sampled sparsely).
            ds = int(args.depth_downsample)
            scene.depth_images = scene.depth_images[:, ::ds, ::ds].copy()
            # Shift + scale principal point consistent with np strided slicing:
            # pixel i in downsampled corresponds to pixel i * ds in original.
            # Pinhole projection: x_px = fx * X/Z + cx. After downsample,
            # new_cx = cx / ds (since a continuous pixel coord scales by 1/ds
            # when we re-index i -> i*ds).
            scene.K = scene.K.copy()
            scene.K[0, 0] /= ds  # fx
            scene.K[1, 1] /= ds  # fy
            scene.K[0, 2] /= ds  # cx
            scene.K[1, 2] /= ds  # cy
        hit = (scene.depth_images > 0).sum() / scene.depth_images.size
        print(
            f"scene: Replica {scene.name} — {scene.n_frames} frames of "
            f"{scene.depth_images.shape[2]}x{scene.depth_images.shape[1]}"
            + (f" (downsampled {args.depth_downsample}x)" if args.depth_downsample > 1 else "")
            + f", hit-ratio {hit:.2%}"
        )
        n_frames_eff = scene.n_frames
    else:
        scene = make_sphere_scene(
            n_frames=args.n_frames,
            W=args.W,
            H=args.H,
        )
        hit = (scene.depth_images > 0).sum() / scene.depth_images.size
        print(
            f"scene: synthetic sphere — {args.n_frames} frames of {args.W}x{args.H}, "
            f"sphere R={scene.sphere_radius}, hit-ratio {hit:.2%}"
        )
        n_frames_eff = args.n_frames
    # Shadow the n_frames var used below — both synthetic and Replica
    # paths set `n_frames_eff` above.
    args.n_frames = n_frames_eff
    print(
        f"config: voxel_size={args.voxel_size}, truncation={args.truncation}, "
        f"f_score_tau={args.f_score_tau}"
    )

    # Warmup: run a tiny synthetic scene through each path first to prime
    # the CUDA kernel cache (first MC invocation otherwise eats ~70 ms of
    # JIT overhead in whichever path runs first, skewing the comparison).
    tiny_scene = make_sphere_scene(n_frames=2, W=64, H=48)
    # Use start_empty for the warmup so fine-voxel configs (<= 5 mm)
    # don't trip a 3-m dense allocation that OOMs before the real run
    # even starts. Metadata (voxel_size, origin) still warms the kernel
    # cache from the tiny scene's first integrate call.
    if not args.skip_per_frame:
        _ = integrate_fvdb(tiny_scene, args.voxel_size, args.truncation,
                           device=args.device, mode="per_frame",
                           start_empty=True)
    _ = integrate_fvdb(tiny_scene, args.voxel_size, args.truncation,
                       device=args.device, mode="frames",
                       start_empty=True)
    _ = integrate_open3d(tiny_scene, args.voxel_size, args.truncation)
    _ = integrate_open3d_cuda(tiny_scene, args.voxel_size, args.truncation,
                              device=args.device)

    # grid_extent=None auto-estimates from scene sensor + depth for
    # synthetic; 1.5 keeps the 1.5-m hand-tuned bounding box the smoke
    # was calibrated on. For Replica we default to start_empty=True
    # because a 2-cm dense bbox on a room-scale scene blows GPU memory.
    fvdb_start = args.fvdb_start
    if fvdb_start == "auto":
        fvdb_start = "empty" if args.scene else "dense"
    start_empty = fvdb_start == "empty"
    real_grid_extent = None if args.scene else 1.5

    # Three columns: fvdb per-frame (legacy), fvdb frames (one-shot N-frame topology), Open3D.
    # `--skip-per-frame` elides the slow legacy path when iterating on a
    # large real scene where we already know it's ~15x slower.
    fvdb_per_frame: RunResult | None = None
    if not args.skip_per_frame:
        fvdb_per_frame = integrate_fvdb(
            scene, args.voxel_size, args.truncation,
            device=args.device, mode="per_frame", grid_extent=real_grid_extent,
            start_empty=start_empty,
        )
    fvdb_frames = integrate_fvdb(
        scene, args.voxel_size, args.truncation,
        device=args.device, mode="frames", grid_extent=real_grid_extent,
        start_empty=start_empty,
    )
    open3d_result = integrate_open3d(scene, args.voxel_size, args.truncation)
    open3d_cuda_result = integrate_open3d_cuda(
        scene, args.voxel_size, args.truncation, device=args.device,
    )

    def _quality(mesh_verts: np.ndarray) -> Dict[str, float]:
        if args.gt_mesh:
            return mesh_quality_vs_gt_mesh(
                mesh_verts, args.gt_mesh,
                f_score_tau=args.f_score_tau, device=args.device,
            )
        if args.scene:
            # Replica scene without a GT mesh: skip quality.
            return {}
        return mesh_quality_vs_gt(
            mesh_verts, scene, f_score_tau=args.f_score_tau, device=args.device,
        )

    fvdb_pf_q = _quality(fvdb_per_frame.mesh_verts) if fvdb_per_frame else {}
    fvdb_fr_q = _quality(fvdb_frames.mesh_verts)
    open3d_q = _quality(open3d_result.mesh_verts)
    open3d_cuda_q = _quality(open3d_cuda_result.mesh_verts)

    def row(name, a, b, c, d):
        name_w = 26
        col_w = 15
        return f"| {name:<{name_w}} | {a:>{col_w}} | {b:>{col_w}} | {c:>{col_w}} | {d:>{col_w}} |"

    print()
    print(row("metric", "fvdb/per_frame", "fvdb/frames", "open3d(CPU)", "open3d(CUDA)"))
    print("|" + "-" * 28 + "|" + ("-" * 16 + ":|") * 4)
    print(row("device", args.device, args.device, "cpu", args.device))
    print(row(
        "integrate ms/frame",
        f"{fvdb_per_frame.integrate_s * 1000 / args.n_frames:.2f}" if fvdb_per_frame else "-",
        f"{fvdb_frames.integrate_s * 1000 / args.n_frames:.2f}",
        f"{open3d_result.integrate_s * 1000 / args.n_frames:.2f}",
        f"{open3d_cuda_result.integrate_s * 1000 / args.n_frames:.2f}",
    ))
    print(row(
        "integrate total (ms)",
        f"{fvdb_per_frame.integrate_s * 1000:.1f}" if fvdb_per_frame else "-",
        f"{fvdb_frames.integrate_s * 1000:.1f}",
        f"{open3d_result.integrate_s * 1000:.1f}",
        f"{open3d_cuda_result.integrate_s * 1000:.1f}",
    ))
    print(row(
        "mesh extract (ms)",
        f"{fvdb_per_frame.mesh_extract_s * 1000:.1f}" if fvdb_per_frame else "-",
        f"{fvdb_frames.mesh_extract_s * 1000:.1f}",
        f"{open3d_result.mesh_extract_s * 1000:.1f}",
        f"{open3d_cuda_result.mesh_extract_s * 1000:.1f}",
    ))
    print(row(
        "mesh verts",
        str(fvdb_per_frame.mesh_V) if fvdb_per_frame else "-",
        str(fvdb_frames.mesh_V),
        str(open3d_result.mesh_V),
        str(open3d_cuda_result.mesh_V),
    ))
    print(row(
        "mesh tris",
        str(fvdb_per_frame.mesh_F) if fvdb_per_frame else "-",
        str(fvdb_frames.mesh_F),
        str(open3d_result.mesh_F),
        str(open3d_cuda_result.mesh_F),
    ))
    print(row(
        "peak GPU MB (torch)",
        f"{fvdb_per_frame.peak_gpu_mb:.1f}" if fvdb_per_frame else "-",
        f"{fvdb_frames.peak_gpu_mb:.1f}",
        "-",
        # torch tracker does NOT see Open3D's cudaMalloc allocations;
        # paper should measure via nvidia-smi before/after for this column.
        "(n/a via torch)",
    ))
    print(row(
        "peak host MB (delta)",
        f"{fvdb_per_frame.peak_host_mb:.1f}" if fvdb_per_frame else "-",
        f"{fvdb_frames.peak_host_mb:.1f}",
        f"{open3d_result.peak_host_mb:.1f}",
        f"{open3d_cuda_result.peak_host_mb:.1f}",
    ))
    if fvdb_fr_q:  # quality metrics available
        print(row(
            f"F-score @ tau={args.f_score_tau:g}",
            f"{fvdb_pf_q.get('f_score', 0):.4f}" if fvdb_pf_q else "-",
            f"{fvdb_fr_q.get('f_score', 0):.4f}",
            f"{open3d_q.get('f_score', 0):.4f}",
            f"{open3d_cuda_q.get('f_score', 0):.4f}",
        ))
        print(row(
            "Chamfer-L1 (m)",
            f"{fvdb_pf_q.get('chamfer_l1', float('nan')):.4f}" if fvdb_pf_q else "-",
            f"{fvdb_fr_q.get('chamfer_l1', float('nan')):.4f}",
            f"{open3d_q.get('chamfer_l1', float('nan')):.4f}",
            f"{open3d_cuda_q.get('chamfer_l1', float('nan')):.4f}",
        ))
    else:
        print(row("quality vs GT", "(skipped — pass --gt-mesh to enable)", "", "", ""))

    # Summary: speedup of fvdb/frames over fvdb/per_frame, and ratio vs Open3D.
    frames_ms = fvdb_frames.integrate_s * 1000 / args.n_frames
    open3d_ms = open3d_result.integrate_s * 1000 / args.n_frames
    print()
    if fvdb_per_frame is not None:
        per_frame_ms = fvdb_per_frame.integrate_s * 1000 / args.n_frames
        print(
            f"fvdb/frames vs fvdb/per_frame: {per_frame_ms / frames_ms:.2f}x faster  "
            f"({per_frame_ms:.1f} -> {frames_ms:.1f} ms/frame)"
        )
    def _ratio_str(ref_ms, label):
        ratio = frames_ms / ref_ms
        relation = "slower" if ratio >= 1 else "faster"
        return (
            f"fvdb/frames vs {label}: {ratio:.2f}x {relation}  "
            f"({ref_ms:.2f} vs {frames_ms:.2f} ms/frame)"
        )

    print(_ratio_str(open3d_ms, "open3d(CPU) "))
    open3d_cuda_ms = open3d_cuda_result.integrate_s * 1000 / args.n_frames
    print(_ratio_str(open3d_cuda_ms, "open3d(CUDA)"))

    if args.json_out:
        import json

        def result_to_dict(r: "RunResult | None"):
            if r is None:
                return None
            return {
                "library": r.library,
                "integrate_s": r.integrate_s,
                "integrate_ms_per_frame": r.integrate_s * 1000 / args.n_frames,
                "mesh_extract_s": r.mesh_extract_s,
                "mesh_V": r.mesh_V,
                "mesh_F": r.mesh_F,
                "peak_gpu_mb": r.peak_gpu_mb,
                "peak_host_mb": r.peak_host_mb,
            }

        out = {
            "scene": {
                "name": getattr(scene, "name", "synthetic_sphere"),
                "n_frames": args.n_frames,
                "W": int(scene.depth_images.shape[2]),
                "H": int(scene.depth_images.shape[1]),
            },
            "config": {
                "voxel_size": args.voxel_size,
                "truncation": args.truncation,
                "f_score_tau": args.f_score_tau,
                "depth_downsample": getattr(args, "depth_downsample", 1),
                "stride": args.stride,
                "max_frames": args.max_frames,
                "fvdb_start": fvdb_start,
            },
            "fvdb_per_frame": result_to_dict(fvdb_per_frame),
            "fvdb_frames": result_to_dict(fvdb_frames),
            "open3d": result_to_dict(open3d_result),
            "open3d_cuda": result_to_dict(open3d_cuda_result),
            "quality": {
                "fvdb_per_frame": fvdb_pf_q,
                "fvdb_frames": fvdb_fr_q,
                "open3d": open3d_q,
                "open3d_cuda": open3d_cuda_q,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
