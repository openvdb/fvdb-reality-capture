# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Resumable downloader for the KITTI Odometry dataset.

Use case: bringing KITTI sequence 00 (and friends) into
`tests/benchmarks/tsdf_fusion/data/KITTI/` for the LiDAR TSDF /
ESDF / occupancy benchmarks. The full Velodyne archive is ~80 GB
so we want a downloader that:

  - Resumes from any previously-downloaded prefix (HTTP Range).
  - Survives connection drops, server hiccups, and Ctrl-C with
    bounded exponential backoff retries.
  - Reports per-second progress (so monitoring in tmux is useful).
  - Verifies final byte count against the server's Content-Length.
  - Is idempotent: a completed download is detected and skipped on
    re-run; an over-full partial is treated as complete (server-
    Content-Length mismatch within fp safety).

The KITTI Odometry archives we care about for LiDAR-only benches:

    data_odometry_velodyne.zip   ~80 GB  Velodyne .bin files (all 22 sequences)
    data_odometry_calib.zip       ~1 MB  sensor calibration matrices
    data_odometry_poses.zip       ~4 MB  ground truth poses for seq 00-10

Color, grayscale, and IMU archives are NOT downloaded (LiDAR TSDF
doesn't need them; saves ~85 GB).

Hosted on https://s3.eu-central-1.amazonaws.com/avg-kitti/.
S3 fully supports HTTP Range requests, so our resume logic works
against the canonical source.

Typical use:

    cd tests/benchmarks/tsdf_fusion/cross_library
    python3 download_kitti.py \\
        --dest ../../data/KITTI/zips \\
        --files calib poses velodyne

Run inside tmux for the velodyne archive (~30-90 minutes depending
on link). The script prints progress to stdout once per second; if
tmux loses your SSH session, the download keeps going. Re-running
the same command picks up where the partial left off.

Exit codes:
    0  all requested files completed successfully
    1  one or more files exhausted retry budget
    2  CLI / configuration error
    130 SIGINT (Ctrl-C); partial files are preserved so a re-run resumes
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import os
import shutil
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Deque, List, Optional, Tuple


KITTI_S3_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"

KITTI_FILES: dict[str, dict[str, Any]] = {
    "velodyne": {
        "filename": "data_odometry_velodyne.zip",
        "approx_size_gb": 80.0,
        "description": ("Velodyne LiDAR .bin files for all 22 odometry "
                        "sequences. THE main download."),
    },
    "calib": {
        "filename": "data_odometry_calib.zip",
        "approx_size_gb": 0.001,
        "description": "Sensor-to-vehicle calibration matrices.",
    },
    "poses": {
        "filename": "data_odometry_poses.zip",
        "approx_size_gb": 0.004,
        "description": "Ground-truth poses for sequences 00-10.",
    },
}


# Globals used by the SIGINT handler so we can flush + exit cleanly.
_active_partial: Optional[Path] = None
_active_handle = None


def _on_sigint(signum, frame):  # noqa: ARG001
    """Flush + close all in-progress partials before exiting so the
    on-disk byte counts match what we wrote. Doesn't delete
    anything -- a re-run resumes from the existing parts."""
    global _active_partial, _active_handle, _active_progress, _active_parts_dir

    # Single-stream path bookkeeping.
    if _active_handle is not None:
        try:
            _active_handle.flush()
            os.fsync(_active_handle.fileno())
            _active_handle.close()
        except Exception:
            pass
    if _active_partial is not None and _active_partial.exists():
        sys.stderr.write(
            f"\n[sigint] preserved partial: {_active_partial}\n"
            f"         re-run the same command to resume.\n")

    # Parallel path: signal worker threads to flush + return; main
    # thread can't wait for them here (signal handler context), so
    # we set the stop_event and trust the threads to fsync and exit
    # before our sys.exit(130) tears down the process.
    if _active_progress is not None:
        _active_progress.stop_event.set()
        # Brief pause to let workers fsync their part files; we run
        # in the main thread's signal context so workers continue
        # executing meanwhile.
        time.sleep(0.5)
        if _active_parts_dir is not None and _active_parts_dir.exists():
            n = sum(1 for _ in _active_parts_dir.iterdir())
            sys.stderr.write(
                f"\n[sigint] preserved {n} part files in: {_active_parts_dir}\n"
                f"         re-run the same command to resume.\n")

    sys.exit(130)


def get_remote_size(url: str, timeout: float = 30.0,
                    max_retries: int = 5) -> int:
    """HEAD request to read Content-Length. Retries on transient
    network failures with exponential backoff."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                cl = resp.headers.get("Content-Length")
                if cl is None:
                    raise RuntimeError(
                        f"server did not send Content-Length for {url}")
                return int(cl)
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            last_exc = e
            backoff = min(60.0, 2.0 ** attempt)
            sys.stderr.write(
                f"  [head] error ({e}); retrying in {backoff:.0f}s\n")
            time.sleep(backoff)
    assert last_exc is not None
    raise last_exc


def _format_eta(remaining_bytes: float, mbps: float) -> str:
    if mbps <= 1e-3:
        return "??"
    seconds = remaining_bytes / (mbps * 1e6)
    if seconds < 60:
        return f"{seconds:5.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:5.1f}m"
    if seconds < 24 * 3600:
        return f"{seconds / 3600:5.1f}h"
    return f"{seconds / 86400:5.1f}d"


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:5.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:5.1f}m"
    return f"{seconds / 3600:5.1f}h"


class _RollingRate:
    """Track download rate via a sliding window over the last
    `window_seconds` of (timestamp, total_bytes) samples. Gives a
    responsive instantaneous-ish rate that catches actual speed
    changes much better than a session-average."""

    def __init__(self, window_seconds: float = 30.0):
        self.window_seconds = window_seconds
        self.samples: Deque[Tuple[float, int]] = collections.deque()

    def update(self, t: float, total_bytes: int) -> None:
        self.samples.append((t, total_bytes))
        cutoff = t - self.window_seconds
        while len(self.samples) > 1 and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def mbps(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        t0, b0 = self.samples[0]
        t1, b1 = self.samples[-1]
        dt = t1 - t0
        if dt <= 0.0:
            return 0.0
        return (b1 - b0) / dt / 1e6


# -------------------------------------------------------------------
# Parallel multi-Range download.
#
# KITTI's S3 bucket throttles each HTTP connection to ~0.18 MB/s
# (verified empirically; ping is fine at 99 ms RTT, BDP analysis
# shows TCP-window-limit is far above what we observe). Splitting
# the download into N parallel HTTP Range requests scales near-
# linearly up to ~8 streams, less so beyond. Default to 8.
# -------------------------------------------------------------------


@dataclasses.dataclass
class _ChunkPlan:
    """Plan for one chunk of a multi-stream download."""
    idx: int           # chunk index (0..N-1)
    byte_start: int    # absolute byte offset in the full file
    byte_end: int      # absolute byte offset (inclusive)
    part_path: Path    # per-chunk file in <dest>.parts/

    @property
    def expected_size(self) -> int:
        return self.byte_end - self.byte_start + 1

    @property
    def have_size(self) -> int:
        return self.part_path.stat().st_size if self.part_path.exists() else 0

    @property
    def is_complete(self) -> bool:
        return self.have_size == self.expected_size


# Tracker shared across all download threads. Each thread
# atomically updates its own counter; main thread polls and
# aggregates for the progress display. Python's GIL makes
# integer-load-and-store on a list element atomic; we don't
# need a lock for the reader's snapshot.
class _ChunkProgress:
    def __init__(self, n_chunks: int):
        self.bytes_per_chunk: List[int] = [0] * n_chunks
        # Per-chunk completion flag for clean shutdown reporting.
        self.completed: List[bool] = [False] * n_chunks
        # Set by main thread on SIGINT to ask workers to stop.
        self.stop_event: threading.Event = threading.Event()
        # Sticky exception from any worker.
        self.first_error: Optional[BaseException] = None
        self.error_lock: threading.Lock = threading.Lock()

    def total_bytes(self) -> int:
        return sum(self.bytes_per_chunk)

    def record_error(self, exc: BaseException) -> None:
        with self.error_lock:
            if self.first_error is None:
                self.first_error = exc


def _download_chunk(
    plan: _ChunkPlan,
    url: str,
    progress: _ChunkProgress,
    chunk_io_size: int = 1 << 20,
    max_retries: int = 12,
    connect_timeout: float = 30.0,
) -> None:
    """Download bytes [plan.byte_start, plan.byte_end] of `url` to
    `plan.part_path`, resuming from the existing partial size.
    Per-chunk retry loop with exponential backoff. Honors
    progress.stop_event for cooperative cancellation."""
    plan.part_path.parent.mkdir(parents=True, exist_ok=True)
    have = plan.have_size
    progress.bytes_per_chunk[plan.idx] = have

    if have >= plan.expected_size:
        if have > plan.expected_size:
            with open(plan.part_path, "rb+") as f:
                f.truncate(plan.expected_size)
            progress.bytes_per_chunk[plan.idx] = plan.expected_size
        progress.completed[plan.idx] = True
        return

    retries = 0
    while plan.have_size < plan.expected_size and not progress.stop_event.is_set():
        absolute_start = plan.byte_start + plan.have_size
        absolute_end   = plan.byte_end
        try:
            req = urllib.request.Request(url)
            req.add_header("Range",
                           f"bytes={absolute_start}-{absolute_end}")
            with urllib.request.urlopen(req, timeout=connect_timeout) as resp:
                if resp.status != 206:
                    raise RuntimeError(
                        f"chunk {plan.idx}: requested Range bytes="
                        f"{absolute_start}-{absolute_end} but server "
                        f"returned status {resp.status} (expected 206)")
                with open(plan.part_path, "ab") as f:
                    while True:
                        if progress.stop_event.is_set():
                            f.flush()
                            try:
                                os.fsync(f.fileno())
                            except OSError:
                                pass
                            return
                        chunk = resp.read(chunk_io_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        progress.bytes_per_chunk[plan.idx] += len(chunk)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
        except (urllib.error.URLError, socket.timeout,
                ConnectionError, OSError) as e:
            retries += 1
            if retries > max_retries:
                err = RuntimeError(
                    f"chunk {plan.idx}: exhausted {max_retries} retries "
                    f"(last: {type(e).__name__}: {e}); part preserved at "
                    f"{plan.part_path}")
                progress.record_error(err)
                return
            backoff = min(60.0, 2.0 ** min(retries, 6))
            time.sleep(backoff)

    if not progress.stop_event.is_set():
        progress.completed[plan.idx] = True


def _migrate_legacy_partial(dest: Path, parts_dir: Path,
                            chunk_0_path: Path) -> None:
    """If a previous single-stream run left behind `<dest>.partial`,
    treat it as chunk 0's existing prefix. Salvages any progress
    rather than re-downloading from byte 0."""
    legacy = dest.with_name(dest.name + ".partial")
    if not legacy.exists():
        return
    parts_dir.mkdir(parents=True, exist_ok=True)
    if chunk_0_path.exists():
        # New-format part already there; pick the larger of the two.
        if legacy.stat().st_size > chunk_0_path.stat().st_size:
            chunk_0_path.unlink()
            shutil.move(str(legacy), str(chunk_0_path))
            sys.stderr.write(
                f"  [migrate] {legacy.name} ({chunk_0_path.stat().st_size:,} B)"
                f" -> {chunk_0_path.name}\n")
        else:
            legacy.unlink()
    else:
        shutil.move(str(legacy), str(chunk_0_path))
        sys.stderr.write(
            f"  [migrate] {legacy.name} ({chunk_0_path.stat().st_size:,} B)"
            f" -> {chunk_0_path.name}\n")


def _concatenate_parts(plans: List[_ChunkPlan], dest: Path) -> None:
    """Stream-concatenate completed part files into `dest`. Uses an
    intermediate `<dest>.combining` so an interrupted concat doesn't
    leave a half-written file at the final name."""
    combining = dest.with_name(dest.name + ".combining")
    if combining.exists():
        combining.unlink()
    with open(combining, "wb") as out:
        for plan in plans:
            if not plan.is_complete:
                raise RuntimeError(
                    f"_concatenate_parts: chunk {plan.idx} incomplete "
                    f"({plan.have_size}/{plan.expected_size})")
            with open(plan.part_path, "rb") as src:
                shutil.copyfileobj(src, out, length=4 << 20)  # 4 MiB
        out.flush()
        os.fsync(out.fileno())
    combining.rename(dest)


# Track active parts dir so SIGINT handler can flag clean shutdown.
_active_parts_dir: Optional[Path] = None
_active_progress: Optional[_ChunkProgress] = None


def download_resumable_parallel(
    url: str,
    dest_path: Path,
    n_streams: int = 32,
    min_chunk_bytes: int = 16 << 20,    # 16 MiB
    max_retries: int = 12,
    progress_interval_s: float = 1.0,
    connect_timeout: float = 30.0,
) -> Path:
    """Multi-Range parallel resumable downloader.

    Splits the file into up to `n_streams` chunks, downloads each in
    its own thread via an HTTP Range request, then concatenates the
    parts to the final destination. Each chunk is independently
    resumable (its part file persists across runs). Falls back to
    a single stream when the total size is small enough that
    splitting would make each chunk smaller than `min_chunk_bytes`.

    Public behaviour matches the old `download_resumable`: on entry,
    any prior `<dest>.partial` (legacy single-stream) is migrated
    into chunk 0's part file; on success, the final file lives at
    `<dest>` and parts dir is removed; on SIGINT, parts are
    preserved and a re-run picks up where each chunk left off.
    """
    global _active_parts_dir, _active_progress

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = dest.with_name(dest.name + ".parts")

    sys.stderr.write(f"[head] {url}\n")
    final_size = get_remote_size(url, timeout=connect_timeout)
    sys.stderr.write(f"  Content-Length: {final_size:,} bytes "
                     f"({final_size / 1e9:.2f} GB)\n")

    if dest.exists():
        if dest.stat().st_size == final_size:
            sys.stderr.write(f"[skip] {dest.name} already complete "
                             f"({final_size:,} bytes)\n")
            return dest
        sys.stderr.write(f"[warn] {dest.name} exists at "
                         f"{dest.stat().st_size}, expected {final_size}; "
                         f"removing\n")
        dest.unlink()

    # Decide on stream count. Keep each chunk above min_chunk_bytes so
    # tiny files don't fan out 8 threads to download <4 MB total.
    desired_chunks = max(1, n_streams)
    chunk_size = max(min_chunk_bytes, (final_size + desired_chunks - 1) // desired_chunks)
    actual_chunks = (final_size + chunk_size - 1) // chunk_size
    if actual_chunks <= 1:
        sys.stderr.write(f"[plan] small file (~{final_size / 1e6:.1f} MB); "
                         f"using 1 stream\n")
    else:
        sys.stderr.write(f"[plan] {actual_chunks} streams x ~"
                         f"{chunk_size / 1e9:.2f} GB each\n")

    plans: List[_ChunkPlan] = []
    for i in range(actual_chunks):
        start = i * chunk_size
        end = min(start + chunk_size - 1, final_size - 1)
        plans.append(_ChunkPlan(
            idx=i, byte_start=start, byte_end=end,
            part_path=parts_dir / f"chunk_{i:04d}",
        ))

    # Salvage any legacy single-stream partial -> chunk 0.
    if actual_chunks > 0:
        _migrate_legacy_partial(dest, parts_dir, plans[0].part_path)

    progress = _ChunkProgress(len(plans))
    for plan in plans:
        progress.bytes_per_chunk[plan.idx] = plan.have_size
        if plan.is_complete:
            progress.completed[plan.idx] = True

    initial_total = progress.total_bytes()
    if initial_total > 0:
        sys.stderr.write(f"[resume] {initial_total:,} bytes already "
                         f"downloaded across {sum(1 for p in plans if p.have_size > 0)} "
                         f"existing parts\n")

    _active_parts_dir = parts_dir
    _active_progress = progress

    # Launch workers.
    threads: List[threading.Thread] = []
    for plan in plans:
        if plan.is_complete:
            continue
        t = threading.Thread(
            target=_download_chunk,
            args=(plan, url, progress),
            kwargs={"max_retries": max_retries,
                    "connect_timeout": connect_timeout},
            name=f"chunk-{plan.idx}",
            daemon=False,
        )
        t.start()
        threads.append(t)

    # Main thread: aggregate progress, print once per second.
    rate = _RollingRate(window_seconds=30.0)
    session_start = time.time()
    rate.update(session_start, initial_total)
    last_log = session_start

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.1)
            now = time.time()
            cur_total = progress.total_bytes()
            rate.update(now, cur_total)
            if now - last_log >= progress_interval_s:
                mbps = rate.mbps()
                pct = cur_total / final_size * 100 if final_size > 0 else 0.0
                eta = _format_eta(final_size - cur_total, mbps)
                elapsed = _format_elapsed(now - session_start)
                session_dl_gb = (cur_total - initial_total) / 1e9
                done_chunks = sum(1 for c in progress.completed if c)
                sys.stderr.write(
                    f"  {pct:6.2f}%  "
                    f"{cur_total / 1e9:7.3f} / {final_size / 1e9:6.2f} GB  "
                    f"{mbps:6.1f} MB/s  "
                    f"elapsed {elapsed}  ETA {eta}  "
                    f"({done_chunks}/{len(plans)} chunks, "
                    f"+{session_dl_gb:.2f} GB this session)\n")
                sys.stderr.flush()
                last_log = now
    except KeyboardInterrupt:
        # Bubbled up from main thread; the SIGINT handler also fires.
        progress.stop_event.set()
        for t in threads:
            t.join(timeout=5.0)
        raise

    for t in threads:
        t.join()

    if progress.first_error is not None:
        raise progress.first_error

    # All chunks must be complete to concatenate.
    incomplete = [p for p in plans if not p.is_complete]
    if incomplete:
        raise RuntimeError(
            f"download_resumable_parallel: {len(incomplete)} chunks "
            f"incomplete: " +
            ", ".join(f"{p.idx}({p.have_size}/{p.expected_size})"
                      for p in incomplete[:5]) +
            ("..." if len(incomplete) > 5 else ""))

    # Verify total size.
    actual_total = sum(p.have_size for p in plans)
    if actual_total != final_size:
        raise RuntimeError(
            f"size mismatch after parallel download: parts total "
            f"{actual_total}, expected {final_size}")

    sys.stderr.write(f"[concat] joining {len(plans)} parts -> {dest.name}\n")
    _concatenate_parts(plans, dest)
    # Remove the parts dir on success.
    try:
        shutil.rmtree(parts_dir)
    except OSError:
        pass

    elapsed_total = time.time() - session_start
    session_dl = actual_total - initial_total
    avg_mbps = (session_dl / elapsed_total / 1e6
                if elapsed_total > 0 else 0.0)
    sys.stderr.write(
        f"[done] {dest.name}: {actual_total:,} bytes "
        f"({actual_total / 1e9:.2f} GB total, {session_dl / 1e9:.2f} GB this session, "
        f"{_format_elapsed(elapsed_total)} session avg {avg_mbps:.1f} MB/s)\n")
    return dest


def download_resumable(
    url: str,
    dest_path: Path,
    chunk_size: int = 1 << 20,         # 1 MiB
    max_retries: int = 12,
    progress_interval_s: float = 1.0,
    connect_timeout: float = 30.0,
) -> Path:
    """Download ``url`` to ``dest_path`` with HTTP Range resume.

    On entry: existing ``dest_path`` files are checked against the
    server's Content-Length and skipped if complete; existing
    ``.partial`` files are resumed from. On normal exit: the
    completed file is renamed from ``<dest>.partial`` to ``<dest>``.
    On SIGINT: the partial is preserved so a re-run picks up.

    Raises if max_retries are exhausted on a single segment, or if
    the final byte count doesn't match Content-Length (i.e. server
    truncated the response).
    """
    global _active_partial, _active_handle

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")

    sys.stderr.write(f"[head] {url}\n")
    final_size = get_remote_size(url, timeout=connect_timeout)
    sys.stderr.write(f"  Content-Length: {final_size:,} bytes "
                     f"({final_size / 1e9:.2f} GB)\n")

    # Idempotent skip: same-name file already exists at expected size.
    if dest.exists():
        if dest.stat().st_size == final_size:
            sys.stderr.write(f"[skip] {dest.name} already complete "
                             f"({final_size:,} bytes)\n")
            return dest
        sys.stderr.write(f"[warn] {dest.name} exists but size mismatch "
                         f"({dest.stat().st_size} vs {final_size}); "
                         f"removing and re-downloading\n")
        dest.unlink()

    have_bytes = partial.stat().st_size if partial.exists() else 0

    if have_bytes >= final_size:
        # Over- or exactly full partial -- assume server Content-Length
        # didn't drift; promote to final.
        if have_bytes > final_size:
            sys.stderr.write(f"[warn] partial ({have_bytes}) exceeds "
                             f"expected ({final_size}); truncating\n")
            with open(partial, "rb+") as f:
                f.truncate(final_size)
        partial.rename(dest)
        return dest

    if have_bytes > 0:
        sys.stderr.write(f"[resume] {partial.name} from byte "
                         f"{have_bytes:,} ({have_bytes / final_size * 100:.1f}%)\n")

    retries = 0
    start_session_bytes = have_bytes
    session_start = time.time()
    rate = _RollingRate(window_seconds=30.0)
    rate.update(session_start, have_bytes)

    while have_bytes < final_size:
        try:
            req = urllib.request.Request(url)
            if have_bytes > 0:
                req.add_header("Range", f"bytes={have_bytes}-")

            with urllib.request.urlopen(req, timeout=connect_timeout) as resp:
                # When we send a Range request, the server SHOULD respond 206
                # Partial Content. If it returns 200, the server doesn't honor
                # ranges and we'd silently re-download from byte 0 — fail loud
                # instead.
                status = resp.status
                if have_bytes > 0 and status != 206:
                    raise RuntimeError(
                        f"resume failed: requested Range bytes={have_bytes}- "
                        f"but server returned status {status} (expected 206). "
                        f"Server doesn't support resume on this URL; re-run "
                        f"after `rm {partial}` to start over.")

                _active_partial = partial
                with open(partial, "ab") as f:
                    _active_handle = f
                    last_log = time.time()
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        have_bytes += len(chunk)
                        now = time.time()
                        rate.update(now, have_bytes)
                        if now - last_log >= progress_interval_s:
                            mbps = rate.mbps()  # rolling 30s window
                            pct = have_bytes / final_size * 100
                            eta = _format_eta(final_size - have_bytes, mbps)
                            elapsed = _format_elapsed(now - session_start)
                            session_dl_gb = (have_bytes - start_session_bytes) / 1e9
                            sys.stderr.write(
                                f"  {pct:6.2f}%  "
                                f"{have_bytes / 1e9:7.3f} / "
                                f"{final_size / 1e9:6.2f} GB  "
                                f"{mbps:6.1f} MB/s  "
                                f"elapsed {elapsed}  ETA {eta}  "
                                f"(+{session_dl_gb:.2f} GB this session)\n")
                            sys.stderr.flush()
                            last_log = now
                    f.flush()
                    os.fsync(f.fileno())
                _active_handle = None
                _active_partial = None

        except (urllib.error.URLError, socket.timeout,
                ConnectionError, OSError) as e:
            retries += 1
            if retries > max_retries:
                sys.stderr.write(
                    f"[fatal] {dest.name}: exhausted {max_retries} retries "
                    f"(last error: {e}). Resume from partial on next run.\n")
                raise
            backoff = min(60.0, 2.0 ** min(retries, 6))
            # Re-stat the partial in case the connection dropped mid-write.
            actual = partial.stat().st_size if partial.exists() else 0
            sys.stderr.write(
                f"  [retry {retries}/{max_retries}] {type(e).__name__}: {e}\n"
                f"  partial is now {actual:,} bytes; retrying in "
                f"{backoff:.0f}s\n")
            sys.stderr.flush()
            time.sleep(backoff)
            have_bytes = partial.stat().st_size if partial.exists() else 0

    actual = partial.stat().st_size
    if actual != final_size:
        raise RuntimeError(
            f"size mismatch: partial has {actual} bytes, expected {final_size}")
    partial.rename(dest)

    elapsed_total = time.time() - session_start
    session_dl = actual - start_session_bytes
    avg_mbps = (session_dl / elapsed_total / 1e6
                if elapsed_total > 0 else 0.0)
    sys.stderr.write(
        f"[done] {dest.name}: {actual:,} bytes "
        f"({actual / 1e9:.2f} GB total, {session_dl / 1e9:.2f} GB this session, "
        f"{_format_elapsed(elapsed_total)} session avg {avg_mbps:.1f} MB/s)\n")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Default destination mirrors the existing Mai City / Replica
    # layout: `tests/benchmarks/tsdf_fusion/data/<DATASET>/`.
    # __file__ is .../cross_library/download_kitti.py, so its
    # parent.parent is .../tsdf_fusion.
    ap.add_argument(
        "--dest", type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data" / "KITTI" / "zips",
        help="Destination directory for the .zip files.")
    ap.add_argument(
        "--files", nargs="+", default=["calib", "poses", "velodyne"],
        choices=sorted(KITTI_FILES.keys()),
        help="Which archives to download. "
             "Order: small-first by default so you get something usable "
             "even if the velodyne download is interrupted.")
    ap.add_argument(
        "--max-retries", type=int, default=12,
        help="Per-file retry budget on connection drops.")
    ap.add_argument(
        "--streams", type=int, default=32,
        help="Number of parallel HTTP Range streams. KITTI's S3 "
             "bucket throttles each connection to ~0.15-0.20 MB/s. "
             "Empirical sweep (3 trials each, 30 s) on this link: "
             "1->0.12, 8->1.22, 16->3.01, 24->3.75, 32->6.07, "
             "48->8.55 MB/s aggregate. 32 is the cleanest knee: "
             "lowest variance and a super-linear jump from 24. "
             "Above 48 the slow-outlier tail starts to dominate. "
             "Set to 1 for single-stream behaviour.")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    args.dest.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(f"=== KITTI download into {args.dest} ===\n")

    failed: list[str] = []
    for name in args.files:
        spec = KITTI_FILES[name]
        url = f"{KITTI_S3_BASE}/{spec['filename']}"
        dest = args.dest / spec["filename"]
        sys.stderr.write(
            f"\n--- {name}: {spec['filename']} "
            f"(~{spec['approx_size_gb']} GB)\n"
            f"    {spec['description']}\n")
        try:
            if args.streams <= 1:
                download_resumable(url, dest, max_retries=args.max_retries)
            else:
                download_resumable_parallel(
                    url, dest,
                    n_streams=args.streams,
                    max_retries=args.max_retries,
                )
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[fatal] {name}: {type(e).__name__}: {e}\n")
            failed.append(name)

    if failed:
        sys.stderr.write(f"\nFAILED: {failed}\n")
        return 1

    sys.stderr.write("\nAll downloads complete.\n")
    sys.stderr.write(f"Next steps:\n")
    sys.stderr.write(f"  cd {args.dest.parent}\n")
    sys.stderr.write(f"  for z in zips/*.zip; do unzip -n \"$z\"; done\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
