# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Loader for the Mai City synthetic LiDAR dataset.

Mai City (StachnissLab, Uni Bonn) provides synthetic Velodyne HDL-64
sweeps generated from a 3D CAD model of an urban block. The dataset
uses the KITTI odometry file format:

    <root>/bin/sequences/XX/velodyne/NNNNN.bin    float32[N, 4]: (x, y, z, reflectance)
    <root>/bin/sequences/XX/times.txt             per-frame timestamps
    <root>/bin/sequences/XX/calib.txt             P0..P3, Tr calibration (identity for Mai City)
    <root>/bin/poses/XX.txt                        per-frame cam0 pose as 3x4 flattened

Three sequences are provided:
    00 — 700 m drive at 10 m/s, 700 frames (main benchmark sequence)
    01 — 100 m block loop, Velodyne HDL-64
    02 — 100 m block loop, 320-beam Velodyne-like sensor

This is the LiDAR complement to `seven_scenes_loader.py` and
`replica_loader.py`. It matches the same `Scene`-style dataclass
interface that the TSDF benchmark drivers consume.

Download instructions:
    wget https://www.ipb.uni-bonn.de/html/projects/mai_city/mai_city.tar.gz
    tar -xf mai_city.tar.gz  # creates `mai_city/` with bin/, ply/, bags/
    # we only need bin/ — ply/ is the same data in a slower format and
    # bags/ is for ROS users; safe to delete both after extraction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class MaiCityScene:
    """Single Mai City sequence parsed and ready for TSDF fusion."""

    # Per-frame data — LiDAR workloads have variable points-per-sweep,
    # so we don't stack into a single [N_frames, N_points, 3] tensor.
    # Instead we keep a list of [N_i, 3] fp32 point clouds.
    points_per_frame: List[np.ndarray]      # length = n_frames, each [N_i, 3] fp32 world-frame
    sensor_origins: np.ndarray               # [n_frames, 3] fp32 world-frame sensor translations
    cam_to_world: np.ndarray                 # [n_frames, 4, 4] fp32 full pose (handy for
                                             # users who want orientation too)

    @property
    def n_frames(self) -> int:
        return len(self.points_per_frame)

    @property
    def total_points(self) -> int:
        return sum(p.shape[0] for p in self.points_per_frame)


def _read_kitti_poses(path: str) -> np.ndarray:
    """Read a KITTI-format pose file.

    Each non-empty line is 12 floats = a row-major 3x4 matrix. Returns
    a [K, 4, 4] float32 array with the bottom row [0, 0, 0, 1] appended.
    """
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 12:
        raise ValueError(
            f"expected 12 floats per line in {path!r}, got {raw.shape[1]}"
        )
    poses = np.eye(4, dtype=np.float32)[None].repeat(raw.shape[0], axis=0)
    poses[:, :3, :] = raw.reshape(-1, 3, 4)
    return poses


def _read_velodyne_bin(path: str, max_points: Optional[int] = None) -> np.ndarray:
    """Read a KITTI-format Velodyne `.bin` file.

    Format: consecutive float32 quadruples (x, y, z, intensity). Returns
    an [N, 3] float32 point cloud (intensity dropped). If `max_points`
    is set, randomly subsample the cloud (used for throughput sanity
    checks; real benches use all points).
    """
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    pts = raw[:, :3].copy()  # drop intensity; .copy() so downstream
                              # `.astype(torch.float32)` doesn't warn
                              # about non-contiguous stride
    if max_points is not None and pts.shape[0] > max_points:
        # Deterministic stride so reruns produce identical benchmarks.
        step = pts.shape[0] // max_points
        pts = pts[::step][:max_points].copy()
    return pts


def load_mai_city_scene(
    root_dir: str,
    sequence: str = "00",
    *,
    max_frames: Optional[int] = None,
    stride: int = 1,
    max_points_per_frame: Optional[int] = None,
) -> MaiCityScene:
    """Load a Mai City sequence and return points in WORLD frame.

    Arguments
    ---------
    root_dir:
        Path to the extracted Mai City root, e.g. `.../data/mai_city/mai_city`.
        Must contain `bin/sequences/<sequence>/velodyne/` and
        `bin/poses/<sequence>.txt`.
    sequence:
        Sequence name ("00", "01", "02"). Default "00" (the 700 m drive).
    max_frames:
        Cap on frames after stride.
    stride:
        Keep every `stride`-th frame. `stride=1` loads all 700 frames
        of sequence 00.
    max_points_per_frame:
        If set, deterministically stride-subsample each sweep to this
        many points. Useful for quick smoke tests at ~10 K pts/sweep;
        leave `None` for real benches (uses all ~70 K points/sweep).

    Returns
    -------
    MaiCityScene:
        `points_per_frame[i]` is an [N_i, 3] float32 world-frame point
        cloud for frame `i`. `sensor_origins[i]` is the [3] float32
        world-frame sensor position (= `cam_to_world[i, :3, 3]`).
        `cam_to_world[i]` is the full 4x4 pose.
    """
    seq_dir = os.path.join(root_dir, "bin", "sequences", sequence, "velodyne")
    poses_path = os.path.join(root_dir, "bin", "poses", f"{sequence}.txt")
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(
            f"Mai City sequence velodyne dir missing: {seq_dir!r}\n"
            f"Extract the dataset so that {root_dir!r}/bin/sequences/{sequence}/"
            f" exists. See this file's docstring for download instructions."
        )
    if not os.path.isfile(poses_path):
        raise FileNotFoundError(f"Mai City poses missing: {poses_path!r}")

    # KITTI format: 5-digit numeric filenames (00000.bin, 00001.bin, ...).
    bin_files = sorted(
        f for f in os.listdir(seq_dir)
        if f.endswith(".bin") and f[:-4].isdigit()
    )
    if not bin_files:
        raise FileNotFoundError(
            f"No `.bin` velodyne files found in {seq_dir!r}"
        )

    # Read all poses up front. The poses file has (N_frames + 1) lines
    # for some sequences (last line is final state past last sweep); we
    # just use the first N_frames matching the velodyne files.
    all_poses = _read_kitti_poses(poses_path)

    # Select frames per stride + cap.
    keep_idxs = list(range(0, len(bin_files), stride))
    if max_frames is not None:
        keep_idxs = keep_idxs[:max_frames]

    points_per_frame: List[np.ndarray] = []
    sensor_origins = np.empty((len(keep_idxs), 3), dtype=np.float32)
    cam_to_world = np.empty((len(keep_idxs), 4, 4), dtype=np.float32)
    for out_i, frame_i in enumerate(keep_idxs):
        # Load LiDAR points in SENSOR frame.
        pts_sensor = _read_velodyne_bin(
            os.path.join(seq_dir, bin_files[frame_i]),
            max_points=max_points_per_frame,
        )
        # Transform to world via pose. Homogeneous multiplication:
        # p_world = R * p_sensor + t
        pose = all_poses[frame_i]
        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_world = (pts_sensor @ R.T) + t[None, :]
        points_per_frame.append(pts_world.astype(np.float32, copy=False))
        sensor_origins[out_i] = t
        cam_to_world[out_i] = pose

    return MaiCityScene(
        points_per_frame=points_per_frame,
        sensor_origins=sensor_origins,
        cam_to_world=cam_to_world,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("root_dir", help="path to extracted mai_city/ (the one with bin/ inside)")
    p.add_argument("--sequence", default="00")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-points-per-frame", type=int, default=None)
    args = p.parse_args()
    scene = load_mai_city_scene(
        args.root_dir, sequence=args.sequence,
        max_frames=args.max_frames, stride=args.stride,
        max_points_per_frame=args.max_points_per_frame,
    )
    print(f"sequence={args.sequence!r}")
    print(f"  n_frames: {scene.n_frames}")
    print(f"  total points: {scene.total_points:,}")
    print(f"  avg points/frame: {scene.total_points / scene.n_frames:,.0f}")
    print(f"  trajectory extent: {np.ptp(scene.sensor_origins, axis=0)} m")
    print(f"  trajectory length (approx): "
          f"{np.linalg.norm(np.diff(scene.sensor_origins, axis=0), axis=1).sum():.1f} m")
