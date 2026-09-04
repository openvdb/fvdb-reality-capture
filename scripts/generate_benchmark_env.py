#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Derive the benchmark conda environment's pins from fvdb-core.

The nightly benchmarks build the fvdb-core wheel using fvdb-core's
``env/build_environment.yml`` and install it into the environment described by
``tests/benchmarks/comparative/docker/benchmark_environment.yml``. If the two
disagree the wheel is compiled against one libtorch and loaded against another,
and the run dies at ``import fvdb`` with an undefined-symbol ImportError.

Rather than keeping the two in sync by hand, this script rewrites the pinned
lines in the benchmark environment from fvdb-core. Only those lines are
touched; the file stays a normal, readable, committed conda environment so the
Dockerfile and local users keep working unchanged.

Examples::

    # match fvdb-core main
    scripts/generate_benchmark_env.py

    # match a specific fvdb-core commit (what CI does)
    scripts/generate_benchmark_env.py --ref 419fcb67e82d63c53bcbf7410c4c825969ac288a

    # match a local checkout, e.g. the one you just built a wheel from
    scripts/generate_benchmark_env.py --from-local ../fvdb-core

    # fail if the committed file is stale, without modifying it
    scripts/generate_benchmark_env.py --check
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# fvdb-core's build environment is the source of truth: it is the environment
# the nightly compiles the wheel in, so it determines the resulting ABI. The
# other env files (dev/test/learn) usually agree but are not what gets built.
CORE_REPO = "openvdb/fvdb-core"
CORE_ENV_RELPATH = "env/build_environment.yml"
BENCH_ENV_RELPATH = "tests/benchmarks/comparative/docker/benchmark_environment.yml"

# Pins that affect the compiled ABI or the interpreter the wheel is built for.
PINNED_KEYS = ("pytorch-gpu", "cuda-version", "python")

TIMEOUT_SECONDS = 30


def _pin_pattern(key: str) -> re.Pattern[str]:
    return re.compile(rf"^(?P<prefix>\s*-\s*{re.escape(key)}=)(?P<value>\S+)\s*$", re.MULTILINE)


def read_pins(text: str, source: str) -> dict[str, str]:
    """Return the pinned value for each key, failing if any is unreadable."""
    pins: dict[str, str] = {}
    missing: list[str] = []
    for key in PINNED_KEYS:
        match = _pin_pattern(key).search(text)
        if match is None:
            missing.append(key)
        else:
            pins[key] = match.group("value")
    if missing:
        raise SystemExit(
            f"error: could not read {', '.join(missing)} from {source}.\n"
            f"       Expected lines of the form '  - <key>=<value>'.\n"
            f"       If the pin format changed, update PINNED_KEYS/_pin_pattern in this script."
        )
    return pins


def fetch_core_env(ref: str) -> str:
    url = f"https://raw.githubusercontent.com/{CORE_REPO}/{ref}/{CORE_ENV_RELPATH}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"error: fetching {url} failed with HTTP {exc.code}. Is '{ref}' a valid ref?") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"error: could not reach {url}: {exc.reason}") from exc


def load_local_core_env(checkout: Path) -> str:
    path = checkout / CORE_ENV_RELPATH
    if not path.is_file():
        raise SystemExit(f"error: {path} not found. Is '{checkout}' an fvdb-core checkout?")
    return path.read_text(encoding="utf-8")


def apply_pins(text: str, pins: dict[str, str]) -> str:
    for key, value in pins.items():
        pattern = _pin_pattern(key)
        text, count = pattern.subn(lambda m: f"{m.group('prefix')}{value}", text)
        if count != 1:
            raise SystemExit(f"error: expected exactly one '{key}=' line in the benchmark env, found {count}.")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--ref", default="main", help="fvdb-core git ref to read pins from (default: main)")
    source.add_argument("--from-local", type=Path, metavar="PATH", help="read pins from a local fvdb-core checkout")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 and print a diff if the committed file is stale",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    bench_path = repo_root / BENCH_ENV_RELPATH
    if not bench_path.is_file():
        raise SystemExit(f"error: {bench_path} not found.")

    if args.from_local is not None:
        core_text = load_local_core_env(args.from_local)
        origin = f"{args.from_local}/{CORE_ENV_RELPATH}"
    else:
        core_text = fetch_core_env(args.ref)
        origin = f"{CORE_REPO}@{args.ref}:{CORE_ENV_RELPATH}"

    core_pins = read_pins(core_text, origin)
    before = bench_path.read_text(encoding="utf-8")
    bench_pins = read_pins(before, str(bench_path))
    after = apply_pins(before, core_pins)

    print(f"source: {origin}")
    for key in PINNED_KEYS:
        arrow = "" if bench_pins[key] == core_pins[key] else f"  (was {bench_pins[key]})"
        print(f"  {key:<14} {core_pins[key]}{arrow}")

    if before == after:
        print(f"{bench_path.relative_to(repo_root)} is already up to date.")
        return 0

    if args.check:
        diff = difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{BENCH_ENV_RELPATH}",
            tofile=f"b/{BENCH_ENV_RELPATH}",
        )
        sys.stdout.writelines(diff)
        print(
            f"\nerror: {BENCH_ENV_RELPATH} is out of date with {origin}.\n"
            f"       Run scripts/generate_benchmark_env.py and commit the result.",
            file=sys.stderr,
        )
        return 1

    bench_path.write_text(after, encoding="utf-8")
    print(f"updated {bench_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
