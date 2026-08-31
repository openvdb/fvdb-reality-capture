# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Loader for the Microsoft 7-Scenes RGB-D dataset.

7-Scenes is a published Kinect-v1 dataset where each "scene" (chess,
fire, heads, office, pumpkin, redkitchen, stairs) contains 2-7
independent recording sequences of ~1000 frames each, all of the
same physical room. All sequences share a common global coordinate
frame from the dataset's camera-relocalisation ground truth, so
sequences can be concatenated into a single long trajectory for
TSDF-fusion benchmarking.

This is the long-trajectory complement to `replica_loader.py`:
- Replica room0 at our current stride=10 gives 200 frames per
  trajectory, and the scene is a small bedroom -- ideal for the
  "fine voxel size at moderate frame count" scale-ceiling story.
- 7-Scenes concatenated gives up to 6000 frames with lots of
  trajectory revisit, which exercises the "bounded surface +
  growing frame count + accumGrid mostly saturated" regime that
  Tree 4 (persistent grid) is supposed to help.

Typical use in the benchmarks here:

    scene = load_seven_scenes_scene(
        'path/to/7-Scenes/chess',
        sequences=['seq-01', 'seq-02', ...],  # or None for all
        max_frames=None,
        stride=1,
    )

Layout on disk (after `unzip seq-0i.zip`):
    chess/seq-01/frame-000000.color.png
    chess/seq-01/frame-000000.depth.png
    chess/seq-01/frame-000000.pose.txt
    chess/seq-01/frame-000001.color.png
    ...
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# Microsoft 7-Scenes Kinect v1 depth intrinsics (these are used in
# all published re-localisation / SLAM baselines on this dataset).
# The color image has slightly different intrinsics; we only use
# depth here so we ignore that.
DEFAULT_SEVEN_SCENES_INTRINSICS = {
    "W": 640,
    "H": 480,
    "fx": 585.0,
    "fy": 585.0,
    "cx": 320.0,
    "cy": 240.0,
    # 7-Scenes depth PNGs store depth in millimetres, with 65535
    # reserved for invalid / too-far pixels. Default threshold for
    # valid data: anything <= 4 m (4000 mm) -- past that the Kinect
    # v1 sensor's noise dominates the measurement.
    "depth_scale_mm_to_m": 1.0 / 1000.0,
    "depth_max_m": 4.0,
    "depth_invalid_marker": 65535,
}


@dataclass
class SevenScenesScene:
    """A parsed 7-Scenes scene ready to feed to `integrate_tsdf_frames`."""

    depth_images: np.ndarray  # [N, H, W] float32 metres (invalid -> 0)
    cam_to_world: np.ndarray  # [N, 4, 4] float32
    K: np.ndarray             # [3, 3] float32 (intrinsic, shared)
    sequence_ids: np.ndarray  # [N] int32: which sequence each frame came from

    @property
    def n_frames(self) -> int:
        return self.depth_images.shape[0]


_FRAME_RE = re.compile(r"frame-(?P<idx>\d{6})\.depth\.png$")


def _frame_index(path: str) -> int:
    m = _FRAME_RE.search(path)
    if m is None:
        raise ValueError(f"path does not match frame-XXXXXX.depth.png: {path!r}")
    return int(m.group("idx"))


def load_seven_scenes_scene(
    scene_dir: str,
    *,
    sequences: Optional[Sequence[str]] = None,
    max_frames: Optional[int] = None,
    stride: int = 1,
    intrinsics: Optional[dict] = None,
) -> SevenScenesScene:
    """Load one or more 7-Scenes sequences and concatenate into a single trajectory.

    Arguments
    ---------
    scene_dir:
        Path to the unzipped scene, e.g. `.../7-Scenes/chess`. Must
        contain `seq-01/`, `seq-02/`, ... subdirectories.
    sequences:
        Explicit list of sequence names to load in order. If None,
        loads all sequences present, sorted lexicographically.
    max_frames:
        Cap on total frames after stride (applied across the
        concatenated trajectory).
    stride:
        Keep every `stride`-th frame (applied per-sequence before
        concatenation so the first frame of each sequence is always
        included -- matches the convention `replica_loader` uses).
    intrinsics:
        Override defaults. Only `W`, `H`, `fx`, `fy`, `cx`, `cy` are
        read out for the camera matrix; depth scaling is always
        `/1000` and the invalid marker is always 65535 (both are
        fixed by the dataset).
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "load_seven_scenes_scene needs pillow: pip install pillow"
        ) from e

    intr = {**DEFAULT_SEVEN_SCENES_INTRINSICS, **(intrinsics or {})}
    W, H = int(intr["W"]), int(intr["H"])
    K = np.array(
        [[intr["fx"], 0, intr["cx"]],
         [0, intr["fy"], intr["cy"]],
         [0, 0, 1]],
        dtype=np.float32,
    )

    if sequences is None:
        sequences = sorted(
            name for name in os.listdir(scene_dir)
            if name.startswith("seq-") and os.path.isdir(os.path.join(scene_dir, name))
        )
    if not sequences:
        raise FileNotFoundError(
            f"No `seq-*/` subdirectories found in {scene_dir!r}"
        )

    depth_list: List[np.ndarray] = []
    c2w_list: List[np.ndarray] = []
    seq_id_list: List[int] = []

    for sid, seq_name in enumerate(sequences):
        seq_dir = os.path.join(scene_dir, seq_name)
        depth_files = sorted(glob.glob(os.path.join(seq_dir, "frame-*.depth.png")),
                              key=_frame_index)
        if stride > 1:
            depth_files = depth_files[::stride]
        for dpath in depth_files:
            idx = _frame_index(dpath)
            ppath = os.path.join(seq_dir, f"frame-{idx:06d}.pose.txt")
            if not os.path.isfile(ppath):
                # Dataset has occasional missing frames; skip.
                continue

            # Depth in mm uint16; invalid = 65535 -> 0 metres.
            d = np.array(Image.open(dpath), dtype=np.uint16)
            if d.shape != (H, W):
                raise ValueError(
                    f"{dpath} depth is {d.shape}, expected ({H}, {W})"
                )
            d_m = d.astype(np.float32) * intr["depth_scale_mm_to_m"]
            d_m[d == intr["depth_invalid_marker"]] = 0.0
            # Also zero out implausibly-far pixels (sensor noise).
            d_m[d_m > intr["depth_max_m"]] = 0.0

            # Pose: space-separated 4x4.
            pose = np.loadtxt(ppath, dtype=np.float32)
            if pose.shape != (4, 4):
                raise ValueError(
                    f"{ppath} pose is {pose.shape}, expected (4, 4)"
                )

            depth_list.append(d_m)
            c2w_list.append(pose)
            seq_id_list.append(sid)

            if max_frames is not None and len(depth_list) >= max_frames:
                break
        if max_frames is not None and len(depth_list) >= max_frames:
            break

    if not depth_list:
        raise FileNotFoundError(
            f"No frames loaded from {scene_dir!r} "
            f"(sequences={sequences}, stride={stride})"
        )

    return SevenScenesScene(
        depth_images=np.stack(depth_list, axis=0),
        cam_to_world=np.stack(c2w_list, axis=0),
        K=K,
        sequence_ids=np.array(seq_id_list, dtype=np.int32),
    )


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("scene_dir", help="path to unzipped 7-Scenes scene, e.g. .../7-Scenes/chess")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    scene = load_seven_scenes_scene(
        args.scene_dir, max_frames=args.max_frames, stride=args.stride
    )
    print(f"Loaded {scene.n_frames} frames from {args.scene_dir!r}")
    print(f"  depth_images: {scene.depth_images.shape} {scene.depth_images.dtype}")
    print(f"  cam_to_world: {scene.cam_to_world.shape} {scene.cam_to_world.dtype}")
    print(f"  K:\n{scene.K}")
    seqs, counts = np.unique(scene.sequence_ids, return_counts=True)
    print(f"  per-sequence counts: " + ", ".join(
        f"seq{sid}={c}" for sid, c in zip(seqs.tolist(), counts.tolist())
    ))
    valid_frac = (scene.depth_images > 0).mean()
    print(f"  valid-depth fraction (mean over frames): {valid_frac*100:.1f}%")
    print(f"  depth range (valid pixels): "
          f"{scene.depth_images[scene.depth_images > 0].min():.3f} -> "
          f"{scene.depth_images.max():.3f} m")
