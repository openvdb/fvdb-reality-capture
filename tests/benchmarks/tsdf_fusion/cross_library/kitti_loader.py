# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Loader for the KITTI Odometry dataset.

KITTI Odometry (Geiger et al., 2012) provides 22 Velodyne HDL-64 sweeps
through Karlsruhe with ground-truth cam0 poses for sequences 00-10
(11-21 are the test set with no published poses). The on-disk layout
matches `mai_city_loader.py`'s with two differences:

    <root>/dataset/sequences/XX/velodyne/NNNNNN.bin   float32[N, 4]: (x, y, z, reflectance)
    <root>/dataset/sequences/XX/calib.txt              P0..P3 (camera intrinsics) + Tr (velo->cam0)
    <root>/dataset/sequences/XX/times.txt              per-frame timestamps
    <root>/dataset/poses/XX.txt                        cam0 poses (00-10 only)

The two differences vs Mai City:

1. **Filename width is 6 digits** (`000000.bin`), not 5.
2. **Poses are in cam0 frame, points are in Velodyne frame.** We must
   compose `T_velo_world = T_cam_world @ Tr` to put each sweep into
   the same world frame as the trajectory. Mai City's `Tr` is identity
   so its loader skips this step; KITTI's `Tr` is a meaningful 0.3 m
   translation + ~90 degree rotation between sensors.

The standard SLAM evaluation triples are 00 (4541 frames, urban),
02 (4661 frames, residential), 05 (2761 frames, suburban). Sequences
01 (highway), 06-10 are smaller/specialized.

Download: see `download_kitti.py` (parallel resumable downloader).
Extract:  unzip data_odometry_velodyne.zip + data_odometry_calib.zip
          + data_odometry_poses.zip into the same root; the three
          archives merge into `<root>/dataset/`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class KittiScene:
    """Single KITTI Odometry sequence parsed and ready for TSDF fusion.

    Mirrors `MaiCityScene` exactly, so all the existing benchmark
    drivers can swap one for the other.
    """

    points_per_frame: List[np.ndarray]      # length n_frames, each [N_i, 3] fp32 world-frame
    sensor_origins: np.ndarray              # [n_frames, 3] fp32 world-frame Velodyne positions
    cam_to_world: np.ndarray                # [n_frames, 4, 4] fp32 Velodyne->world poses
                                            # (NOT cam0->world; named for compatibility
                                            # with the bench drivers that expect this attr)

    @property
    def n_frames(self) -> int:
        return len(self.points_per_frame)

    @property
    def total_points(self) -> int:
        return sum(p.shape[0] for p in self.points_per_frame)


def _read_kitti_poses(path: str) -> np.ndarray:
    """Read a KITTI-format pose file (12 floats per line = row-major 3x4
    matrix). Returns [K, 4, 4] float32 with the bottom row [0,0,0,1]."""
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 12:
        raise ValueError(
            f"expected 12 floats per line in {path!r}, got {raw.shape[1]}")
    poses = np.eye(4, dtype=np.float32)[None].repeat(raw.shape[0], axis=0)
    poses[:, :3, :] = raw.reshape(-1, 3, 4)
    return poses


def _read_kitti_calib_tr(calib_path: str) -> np.ndarray:
    """Read the `Tr:` line of a KITTI calib.txt and return a 4x4 float32
    homogeneous transform from Velodyne to cam0 frame.

    The line has 12 floats (row-major 3x4); we append [0,0,0,1].
    """
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("Tr"):
                continue
            # `Tr: <12 floats>` — split off the colon then parse.
            after_colon = line.split(":", 1)[1].strip()
            vals = np.fromstring(after_colon, dtype=np.float32, sep=" ")
            if vals.size != 12:
                raise ValueError(
                    f"calib Tr in {calib_path!r}: expected 12 floats, got {vals.size}")
            T = np.eye(4, dtype=np.float32)
            T[:3, :] = vals.reshape(3, 4)
            return T
    raise ValueError(f"no `Tr:` line found in {calib_path!r}")


def _read_velodyne_bin(path: str, max_points: Optional[int] = None) -> np.ndarray:
    """Read a KITTI-format Velodyne `.bin` file. Returns [N, 3] float32
    (intensity dropped). If `max_points` is set, deterministic stride."""
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    pts = raw[:, :3].copy()
    if max_points is not None and pts.shape[0] > max_points:
        step = pts.shape[0] // max_points
        pts = pts[::step][:max_points].copy()
    return pts


def load_kitti_scene(
    root_dir: str,
    sequence: str = "00",
    *,
    max_frames: Optional[int] = None,
    stride: int = 1,
    max_points_per_frame: Optional[int] = None,
) -> KittiScene:
    """Load a KITTI Odometry sequence and return points in WORLD frame.

    Arguments
    ---------
    root_dir:
        Path to the extracted KITTI Odometry root, e.g.
        `.../data/KITTI`. Must contain `dataset/sequences/<seq>/`
        and `dataset/poses/<seq>.txt`.
    sequence:
        Sequence name, "00".."10" (training only; 11-21 lack poses).
    max_frames, stride, max_points_per_frame:
        Same semantics as `load_mai_city_scene`.

    Returns
    -------
    KittiScene with points in world frame (Velodyne sensor origins
    derived from `T_cam_world @ Tr`).
    """
    seq_dir = os.path.join(root_dir, "dataset", "sequences", sequence, "velodyne")
    calib_path = os.path.join(root_dir, "dataset", "sequences", sequence, "calib.txt")
    poses_path = os.path.join(root_dir, "dataset", "poses", f"{sequence}.txt")
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(
            f"KITTI sequence velodyne dir missing: {seq_dir!r}\n"
            f"Extract data_odometry_velodyne.zip into {root_dir!r} so that "
            f"{root_dir!r}/dataset/sequences/{sequence}/velodyne/ exists.")
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(
            f"KITTI calib missing: {calib_path!r}\n"
            f"Extract data_odometry_calib.zip into the same root.")
    if not os.path.isfile(poses_path):
        raise FileNotFoundError(
            f"KITTI poses missing: {poses_path!r}\n"
            f"Sequences 11-21 are the test set with no published poses; "
            f"only 00-10 are usable for benchmarking. Extract "
            f"data_odometry_poses.zip into the same root.")

    bin_files = sorted(
        f for f in os.listdir(seq_dir)
        if f.endswith(".bin") and f[:-4].isdigit())
    if not bin_files:
        raise FileNotFoundError(f"No `.bin` velodyne files found in {seq_dir!r}")

    Tr = _read_kitti_calib_tr(calib_path)              # [4, 4] velo->cam0
    cam_poses = _read_kitti_poses(poses_path)          # [N, 4, 4] cam0->world

    if cam_poses.shape[0] != len(bin_files):
        # Some KITTI mirrors include a final trailing pose; trim if so.
        if cam_poses.shape[0] == len(bin_files) + 1:
            cam_poses = cam_poses[:-1]
        else:
            raise ValueError(
                f"frame/pose count mismatch in seq {sequence!r}: "
                f"{len(bin_files)} velodyne files vs {cam_poses.shape[0]} poses")

    keep_idxs = list(range(0, len(bin_files), stride))
    if max_frames is not None:
        keep_idxs = keep_idxs[:max_frames]

    points_per_frame: List[np.ndarray] = []
    sensor_origins = np.empty((len(keep_idxs), 3), dtype=np.float32)
    velo_to_world = np.empty((len(keep_idxs), 4, 4), dtype=np.float32)

    for out_i, frame_i in enumerate(keep_idxs):
        pts_velo = _read_velodyne_bin(
            os.path.join(seq_dir, bin_files[frame_i]),
            max_points=max_points_per_frame)

        # Compose Velodyne -> world transform:
        #   p_world = T_cam_world @ Tr @ p_velo
        T_v2w = cam_poses[frame_i] @ Tr     # [4, 4]
        R = T_v2w[:3, :3]
        t = T_v2w[:3, 3]
        pts_world = (pts_velo @ R.T) + t[None, :]
        points_per_frame.append(pts_world.astype(np.float32, copy=False))
        sensor_origins[out_i] = t
        velo_to_world[out_i] = T_v2w

    return KittiScene(
        points_per_frame=points_per_frame,
        sensor_origins=sensor_origins,
        cam_to_world=velo_to_world,    # named for bench-driver compat
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("root_dir", help="path to extracted KITTI/ (the one with dataset/ inside)")
    p.add_argument("--sequence", default="00")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-points-per-frame", type=int, default=None)
    args = p.parse_args()
    scene = load_kitti_scene(
        args.root_dir, sequence=args.sequence,
        max_frames=args.max_frames, stride=args.stride,
        max_points_per_frame=args.max_points_per_frame)
    print(f"sequence={args.sequence!r}")
    print(f"  n_frames: {scene.n_frames}")
    print(f"  total points: {scene.total_points:,}")
    print(f"  avg points/frame: {scene.total_points / scene.n_frames:,.0f}")
    print(f"  trajectory extent: {np.ptp(scene.sensor_origins, axis=0)} m")
    print(f"  trajectory length (approx): "
          f"{np.linalg.norm(np.diff(scene.sensor_origins, axis=0), axis=1).sum():.1f} m")
