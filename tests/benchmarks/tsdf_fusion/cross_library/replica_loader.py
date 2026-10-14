# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Replica NICE-SLAM format loader.

Reads the RGB-D + pose sequence released by NICE-SLAM (and used by
iMAP / Co-SLAM / ESLAM and many follow-ups), so numbers produced
here are directly comparable against those papers' tables.

Directory layout the loader expects (one scene root):

    <scene>/
        results/
            frame000000.jpg
            depth000000.png      (uint16 depth, scale 1/6553.5 -> metres)
            frame000001.jpg
            depth000001.png
            ...
        traj.txt                 (one 4x4 cam->world matrix per frame,
                                  row-major, 16 floats per line)

Intrinsics are fixed for the whole Replica set in the NICE-SLAM
rendering — fx = fy = 600, cx = 599.5, cy = 339.5, W = 1200,
H = 680 — but we expose overrides for the occasional variant.

Minimal API:

    scene = load_replica_scene("path/to/room_0", max_frames=200, stride=2)
    # scene.depth_images: np.ndarray [N, H, W] float32 metres
    # scene.cam_to_world: np.ndarray [N, 4, 4] float32
    # scene.K: np.ndarray [3, 3] float32
    # scene.name: str

Downstream benchmark code treats the result shape-for-shape the same
as the synthetic `SyntheticScene` returned by `make_sphere_scene`.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np


@dataclasses.dataclass
class ReplicaScene:
    depth_images: np.ndarray  # [N, H, W] float32, metres, 0 = no measurement
    cam_to_world: np.ndarray  # [N, 4, 4] float32
    K: np.ndarray             # [3, 3] float32
    name: str

    # Mimic the SyntheticScene attributes the benchmark script expects.
    sphere_center: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3, np.float32))
    sphere_radius: float = 0.0

    @property
    def n_frames(self) -> int:
        return self.depth_images.shape[0]


# NICE-SLAM's rendering convention for Replica (standard across their
# entire 8-scene release and the iMAP / Co-SLAM / ESLAM re-uses).
DEFAULT_REPLICA_INTRINSICS = {
    "W": 1200,
    "H": 680,
    "fx": 600.0,
    "fy": 600.0,
    "cx": 599.5,
    "cy": 339.5,
    "png_depth_scale": 6553.5,  # depth_m = depth_u16 / png_depth_scale
}


def _read_traj(path: str) -> np.ndarray:
    """Parse `traj.txt`: one 4x4 row-major cam->world matrix per line."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    mats = []
    for ln in lines:
        vals = [float(x) for x in ln.split()]
        if len(vals) != 16:
            raise ValueError(
                f"{path}: expected 16 floats per line (4x4 matrix), got {len(vals)}"
            )
        mats.append(np.asarray(vals, np.float32).reshape(4, 4))
    return np.stack(mats, axis=0)


def load_replica_scene(
    scene_dir: str,
    max_frames: Optional[int] = None,
    stride: int = 1,
    png_depth_scale: float = DEFAULT_REPLICA_INTRINSICS["png_depth_scale"],
    intrinsics: Optional[dict] = None,
) -> ReplicaScene:
    """Load a NICE-SLAM rendered Replica scene from disk.

    Args:
        scene_dir: path to the scene root (contains `results/` and `traj.txt`).
        max_frames: if set, load only the first `max_frames` frames (after stride).
        stride: take every `stride`-th frame from the trajectory (1 = all frames).
        png_depth_scale: divisor to turn uint16 PNG depth into metres.
        intrinsics: dict overriding `fx, fy, cx, cy, W, H`. Defaults to the
            standard NICE-SLAM Replica intrinsics.

    Returns:
        ReplicaScene with depth_images, cam_to_world, K populated.
    """
    # Lazy-imports so `replica_loader` can be imported in a minimal env that
    # doesn't have PIL / opencv — we only need them when actually loading.
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "load_replica_scene needs pillow installed — pip install pillow"
        ) from e

    intrinsics = {**DEFAULT_REPLICA_INTRINSICS, **(intrinsics or {})}
    W = int(intrinsics["W"])
    H = int(intrinsics["H"])
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    results_dir = os.path.join(scene_dir, "results")
    traj_path = os.path.join(scene_dir, "traj.txt")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            f"Replica scene at {scene_dir!r} is missing `results/` — "
            "expected NICE-SLAM rendering layout"
        )
    if not os.path.isfile(traj_path):
        raise FileNotFoundError(
            f"Replica scene at {scene_dir!r} is missing `traj.txt`"
        )

    cam_to_world_all = _read_traj(traj_path)
    n_total = cam_to_world_all.shape[0]

    frame_indices = list(range(0, n_total, stride))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    depth_images = np.zeros((len(frame_indices), H, W), dtype=np.float32)
    cam_to_world = np.empty((len(frame_indices), 4, 4), dtype=np.float32)
    for out_i, src_i in enumerate(frame_indices):
        depth_path = os.path.join(results_dir, f"depth{src_i:06d}.png")
        if not os.path.isfile(depth_path):
            raise FileNotFoundError(
                f"missing {depth_path} — dataset may be incomplete"
            )
        depth_u16 = np.asarray(Image.open(depth_path), dtype=np.uint16)
        if depth_u16.shape != (H, W):
            raise ValueError(
                f"{depth_path}: shape {depth_u16.shape} != expected ({H}, {W}); "
                "override intrinsics via the `intrinsics=` argument"
            )
        depth_images[out_i] = depth_u16.astype(np.float32) / png_depth_scale
        cam_to_world[out_i] = cam_to_world_all[src_i]

    return ReplicaScene(
        depth_images=depth_images,
        cam_to_world=cam_to_world,
        K=K,
        name=os.path.basename(os.path.normpath(scene_dir)),
    )
