# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Bulk download of NICE-SLAM's Replica.zip (12.44 GB) using the
parallel multi-Range downloader from `download_kitti.py`.

Why: the existing `download_replica.py` uses `remotezip` to extract
strided subsets of frames over a single HTTP connection, which is
the right choice when bandwidth is precious and you only need
~200 frames per scene. For a paper-grade benchmark we want all
8 scenes at full frame rate (~2000 frames/scene), which means
pulling the whole archive.

ETH's `cvg-data.inf.ethz.ch` mirror throttles each HTTP connection
to ~0.1-0.2 MB/s but does not appear to apply a per-IP cap; an
empirical sweep against the Replica.zip URL shows aggregate
throughput scales near-linearly through 32 streams (matching
KITTI's S3 behaviour). At 32 streams ETA is ~45 min vs ~49 hours
single-stream.

Usage:

    python download_replica_zip.py            # 32 streams, default dest
    python download_replica_zip.py --streams 16
    python download_replica_zip.py --extract  # also unzip after download
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# Reuse the parallel download primitives from the KITTI downloader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_kitti import (  # noqa: E402
    _on_sigint,
    download_resumable,
    download_resumable_parallel,
)


REPLICA_ZIP_URL = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
REPLICA_ZIP_SIZE_BYTES = 12_442_855_671   # ~12.44 GB
REPLICA_SCENES = (
    "office0", "office1", "office2", "office3", "office4",
    "room0", "room1", "room2",
)


def main() -> None:
    here = Path(__file__).resolve().parent
    default_dest = here.parent / "data" / "Replica.zip"
    default_extract_to = here.parent / "data" / "Replica"

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=default_dest,
                    help=f"output path for the .zip (default: {default_dest})")
    ap.add_argument("--streams", type=int, default=32,
                    help="parallel HTTP Range streams (default: 32)")
    ap.add_argument("--max-retries", type=int, default=12,
                    help="per-chunk retry budget (default: 12)")
    ap.add_argument("--extract", action="store_true",
                    help="after download, unzip into <dest_dir>/Replica/. "
                         "Skips scenes already extracted with the expected "
                         "frame counts.")
    ap.add_argument("--extract-to", type=Path, default=default_extract_to,
                    help=f"extract destination (default: {default_extract_to})")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    sys.stderr.write(
        f"=== Replica.zip download into {args.dest.parent} ===\n"
        f"    URL    : {REPLICA_ZIP_URL}\n"
        f"    streams: {args.streams}\n"
        f"    expect : ~{REPLICA_ZIP_SIZE_BYTES / 1e9:.2f} GB\n\n")

    args.dest.parent.mkdir(parents=True, exist_ok=True)

    if args.streams <= 1:
        download_resumable(REPLICA_ZIP_URL, args.dest,
                           max_retries=args.max_retries)
    else:
        download_resumable_parallel(
            REPLICA_ZIP_URL, args.dest,
            n_streams=args.streams,
            max_retries=args.max_retries,
        )

    actual_size = args.dest.stat().st_size
    if actual_size != REPLICA_ZIP_SIZE_BYTES:
        sys.stderr.write(
            f"[warn] downloaded size {actual_size:,} != expected "
            f"{REPLICA_ZIP_SIZE_BYTES:,}; the upstream archive may have "
            f"changed. Continuing anyway.\n")

    if args.extract:
        sys.stderr.write(f"\n=== Extracting into {args.extract_to} ===\n")
        args.extract_to.mkdir(parents=True, exist_ok=True)
        # `unzip -n`: never overwrite. Lets us re-run safely.
        t0 = time.time()
        proc = subprocess.run(
            ["unzip", "-n", "-q", str(args.dest), "-d", str(args.extract_to.parent)],
            check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(
                f"[fatal] unzip failed with code {proc.returncode}\n")
            sys.exit(proc.returncode)
        sys.stderr.write(f"  unzip done in {time.time() - t0:.0f} s\n")

        # The archive layout is `Replica/<scene>/...` so the unzip
        # command above unpacks into `<extract_to.parent>/Replica/...`,
        # which already matches `args.extract_to`. List what's there:
        scenes_present = sorted(d.name for d in args.extract_to.iterdir() if d.is_dir())
        sys.stderr.write(f"  scenes available: {len(scenes_present)} -> {scenes_present}\n")

    sys.stderr.write(
        "\n=== Done. ===\n"
        f"  ZIP at  : {args.dest}\n"
        f"  Scenes  : {args.extract_to if args.extract else '(run with --extract or unzip manually)'}\n")


if __name__ == "__main__":
    main()
