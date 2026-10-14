# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Partial downloader for the NICE-SLAM Replica rendering.

The published Replica.zip is 12.4 GB (all 8 scenes). Via HTTP range
requests we can extract just the scenes / strided frames we need,
which is several orders of magnitude faster on bandwidth-limited
links (the ETH cvg-data mirror rate-limits to ~180 KB/s).

Usage:

    python download_replica.py room_0 \\
        --out /path/to/data/Replica \\
        --stride 10              # keep 1/10th of frames
        --max-frames 200         # cap total frames per scene

Default: stride=1, no cap (full scene, ~1.6 GB per scene).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import List, Optional

try:
    from remotezip import RemoteZip
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "download_replica needs `remotezip` installed: pip install remotezip"
    ) from e


REPLICA_ZIP_URL = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"

# NICE-SLAM's zip layout: `Replica/<scene>/results/...` + `Replica/<scene>/traj.txt`.
# The archive root directory is `Replica/`.

# Archive uses `frame000000.jpg` / `depth000000.png` naming.
_FRAME_RE = re.compile(r"Replica/(?P<scene>[^/]+)/results/(?:frame|depth)(?P<idx>\d{6})\.(?:jpg|png)$")


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def download_scene(
    scene: str,
    out_dir: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
    url: str = REPLICA_ZIP_URL,
) -> None:
    """Download a subset of `scene` from the remote Replica.zip via HTTP ranges.

    Writes into `<out_dir>/<scene>/results/` and `<out_dir>/<scene>/traj.txt`.
    """
    os.makedirs(out_dir, exist_ok=True)
    scene_root = os.path.join(out_dir, scene)
    results_dir = os.path.join(scene_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"opening remote zip @ {url} (central-directory fetch only)...")
    with RemoteZip(url) as rz:
        names = rz.namelist()

        # Find frame indices available for this scene.
        scene_prefix = f"Replica/{scene}/"
        traj_name = f"{scene_prefix}traj.txt"
        if traj_name not in names:
            avail = sorted({n.split("/")[1] for n in names if n.startswith("Replica/") and "/" in n[len("Replica/"):]})
            raise SystemExit(
                f"scene {scene!r} not found in archive. Available scenes:\n  "
                + "\n  ".join(avail)
            )

        # Enumerate (frame_idx, jpg_name, png_name) tuples for this scene.
        frames_idx: List[int] = []
        for n in names:
            m = _FRAME_RE.match(n)
            if m and m.group("scene") == scene:
                frames_idx.append(int(m.group("idx")))
        frames_idx = sorted(set(frames_idx))
        n_total = len(frames_idx)
        kept = frames_idx[::stride]
        if max_frames is not None:
            kept = kept[:max_frames]

        members: List[str] = [traj_name]
        for i in kept:
            members.append(f"{scene_prefix}results/frame{i:06d}.jpg")
            members.append(f"{scene_prefix}results/depth{i:06d}.png")

        # Sum compressed sizes so we know the download bytes budget.
        total_compressed = 0
        for name in members:
            try:
                info = rz.getinfo(name)
                total_compressed += info.compress_size
            except KeyError:
                print(f"  warn: missing member {name}", file=sys.stderr)
        print(
            f"scene={scene!r}: {n_total} frames in archive, keeping "
            f"{len(kept)} after stride={stride} + max_frames={max_frames}; "
            f"fetching ~{_format_bytes(total_compressed)} of compressed data"
        )

        # Extract each member to `out_dir`. remotezip translates this to
        # HTTP Range requests under the hood.
        t0 = time.perf_counter()
        for i, name in enumerate(members):
            rz.extract(name, path=out_dir)
            if i % 20 == 0 or i == len(members) - 1:
                dt = time.perf_counter() - t0
                kbps = (rz._session_fetched_bytes / 1024.0 / max(dt, 1e-3)) if hasattr(rz, "_session_fetched_bytes") else None
                rate = f"  ({kbps:.0f} KB/s)" if kbps else ""
                print(f"  [{i + 1:>5d} / {len(members):>5d}]  {name}{rate}")

        # remotezip extracts to `<out_dir>/Replica/<scene>/...`; move to
        # `<out_dir>/<scene>/...` so `replica_loader` sees the expected
        # layout ("scene-root has results/ and traj.txt at top level").
        tmp_root = os.path.join(out_dir, "Replica")
        if os.path.isdir(tmp_root):
            extracted_scene = os.path.join(tmp_root, scene)
            if os.path.isdir(extracted_scene):
                # Move contents of extracted_scene into scene_root (which
                # already exists).
                for entry in os.listdir(extracted_scene):
                    src = os.path.join(extracted_scene, entry)
                    dst = os.path.join(scene_root, entry)
                    if os.path.exists(dst):
                        # Directory already exists: merge contents.
                        if os.path.isdir(src) and os.path.isdir(dst):
                            for sub in os.listdir(src):
                                os.rename(
                                    os.path.join(src, sub),
                                    os.path.join(dst, sub),
                                )
                            os.rmdir(src)
                        else:
                            raise RuntimeError(
                                f"conflict moving {src!r} -> {dst!r}"
                            )
                    else:
                        os.rename(src, dst)
                os.rmdir(extracted_scene)
            if os.path.isdir(tmp_root) and not os.listdir(tmp_root):
                os.rmdir(tmp_root)

    print(f"done. scene written to {scene_root!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("scene", help="scene name (e.g. room_0, office_0)")
    p.add_argument("--out", default=None,
                   help="output dir (default: <this_dir>/../data/Replica)")
    p.add_argument("--stride", type=int, default=1,
                   help="keep 1 / stride frames")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap total frames after stride")
    p.add_argument("--url", default=REPLICA_ZIP_URL)
    args = p.parse_args()

    out = args.out
    if out is None:
        here = os.path.dirname(os.path.abspath(__file__))
        out = os.path.normpath(os.path.join(here, "..", "data", "Replica"))

    download_scene(args.scene, out, stride=args.stride,
                   max_frames=args.max_frames, url=args.url)


if __name__ == "__main__":
    main()
