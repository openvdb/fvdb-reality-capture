# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Minimal repro for gsplat cold-start rendering bug on NVIDIA L4 GPUs.

On a freshly launched L4 instance, the first gsplat process produces
different (incorrect) output compared to subsequent processes, even
with identical deterministic inputs. The bug affects the very first
forward pass and persists for the lifetime of that process.

This script runs two approaches:

  1. Synthetic: rasterizes deterministic Gaussians in two subprocesses,
     compares rendered image checksums.

  2. Training: runs simple_trainer.py for 50 steps twice, compares loss
     at step 0. (Requires mipnerf360 bonsai dataset.)

On L4 (fresh instance):
  - Process 1 (cold): loss ~0.27, rendered image differs
  - Process 2 (warm): loss ~0.20, correct output

On other GPUs (RTX 6000 Ada, A100, etc.):
  - Both processes produce identical output

Evidence from CI (5 repetitions on fresh L4):
  - DefaultStrategy rep1 (first process): PSNR 14.2, loss 0.181
  - DefaultStrategy rep2-5: PSNR 27.2-27.3, loss 0.042-0.044
  - When MCMC runs first instead: MCMC gets PSNR 14.3, Default gets 27.2
  - Bug affects whichever strategy runs first on a fresh GPU

Usage:
  # Synthetic test (no dataset needed):
  python gsplat_l4_coldstart_repro.py --mode synthetic

  # Training test (needs mipnerf360/bonsai):
  python gsplat_l4_coldstart_repro.py --mode training --data-dir /path/to/360_v2/bonsai

  # Both:
  python gsplat_l4_coldstart_repro.py --mode both --data-dir /path/to/360_v2/bonsai

Requires: gsplat (pip install gsplat), torch with CUDA
"""
import argparse
import json
import os
import re
import subprocess
import sys
import textwrap


SYNTHETIC_WORKER = textwrap.dedent(
    r"""
import torch
import json

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = "cuda"
N = 200000
H, W = 512, 512

means = torch.randn(N, 3, device=device) * 2.0
quats = torch.randn(N, 4, device=device)
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.rand(N, 3, device=device) * 0.05
opacities = torch.sigmoid(torch.randn(N, device=device))
sh0 = torch.rand(N, 1, 3, device=device)

viewmat = torch.eye(4, device=device).unsqueeze(0)
viewmat[0, 2, 3] = 5.0

K = torch.zeros(1, 3, 3, device=device)
K[0, 0, 0] = 500.0
K[0, 1, 1] = 500.0
K[0, 0, 2] = W / 2
K[0, 1, 2] = H / 2
K[0, 2, 2] = 1.0

from gsplat import rasterization

with torch.no_grad():
    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh0,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        sh_degree=0,
        render_mode="RGB",
        packed=False,
    )

img = renders[0]
result = {
    "mean_r": float(img[:, :, 0].mean()),
    "mean_g": float(img[:, :, 1].mean()),
    "mean_b": float(img[:, :, 2].mean()),
    "std": float(img.std()),
    "checksum": float(img.double().sum()),
    "alpha_mean": float(alphas[0].mean()),
    "gpu": torch.cuda.get_device_name(0),
    "compute_cap": list(torch.cuda.get_device_capability(0)),
}
print("RESULT:" + json.dumps(result))
"""
)


def run_subprocess(label, script):
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [{label}] FAILED (exit code {result.returncode})")
        stderr = result.stderr[-2000:] if result.stderr else "(no stderr)"
        print(f"  {stderr}")
        return None

    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])

    print(f"  [{label}] No RESULT line in output")
    return None


def test_synthetic():
    print("Test 1: Synthetic rasterization (200K Gaussians, deterministic)")
    print("-" * 60)

    r1 = run_subprocess("run1-cold", SYNTHETIC_WORKER)
    r2 = run_subprocess("run2-warm", SYNTHETIC_WORKER)

    if r1 is None or r2 is None:
        print("  One or both runs failed.\n")
        return False

    print(f"  GPU: {r1['gpu']} (sm_{r1['compute_cap'][0]}{r1['compute_cap'][1]})")
    print()
    print(f"  {'':>12s}  {'Run 1 (cold)':>14s}  {'Run 2 (warm)':>14s}  {'Match':>6s}")
    print(f"  {'-' * 53}")

    all_match = True
    for key in ["mean_r", "mean_g", "mean_b", "std", "checksum", "alpha_mean"]:
        v1, v2 = r1[key], r2[key]
        match = abs(v1 - v2) < 1e-4
        if not match:
            all_match = False
        print(f"  {key:>12s}  {v1:>14.6f}  {v2:>14.6f}  {'OK' if match else 'DIFF':>6s}")

    print()
    if all_match:
        print("  PASS: Both runs produced identical output.")
    else:
        print("  FAIL: Run 1 (cold) produced different output from Run 2 (warm).")
    return all_match


def test_training(data_dir, gsplat_dir):
    print("Test 2: Training (simple_trainer.py, 50 steps, bonsai)")
    print("-" * 60)

    trainer_path = os.path.join(gsplat_dir, "examples", "simple_trainer.py")
    if not os.path.exists(trainer_path):
        print(f"  simple_trainer.py not found at: {trainer_path}")
        print(f"  Skipping training test.\n")
        return None

    if not os.path.isdir(data_dir):
        print(f"  Dataset not found at: {data_dir}")
        print(f"  Skipping training test.\n")
        return None

    losses = []
    for i, label in enumerate(["run1-cold", "run2-warm"], 1):
        result_dir = f"/tmp/gsplat_coldstart_repro_{label}"
        cmd = [
            sys.executable,
            trainer_path,
            "default",
            "--data_dir",
            data_dir,
            "--result_dir",
            result_dir,
            "--data_factor",
            "8",
            "--max_steps",
            "50",
            "--disable_viewer",
            "--disable_video",
            "--batch_size",
            "1",
        ]
        print(f"  Running {label}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        loss_match = re.search(r"loss=([\d.]+)\|", result.stderr or result.stdout or "")
        if loss_match:
            loss = float(loss_match.group(1))
            losses.append(loss)
            print(f"  [{label}] First loss: {loss:.4f}")
        else:
            print(f"  [{label}] Could not parse loss from output")
            if result.returncode != 0:
                print(f"  Exit code: {result.returncode}")
                print(f"  {(result.stderr or '')[-500:]}")
            losses.append(None)

        subprocess.run(["rm", "-rf", result_dir], capture_output=True)

    if losses[0] is not None and losses[1] is not None:
        diff = abs(losses[0] - losses[1])
        match = diff < 0.01
        print()
        print(f"  Loss difference: {diff:.4f} ({'OK' if match else 'SIGNIFICANT'})")
        if match:
            print("  PASS: Both runs started with similar loss.")
        else:
            print("  FAIL: First run started with significantly different loss.")
            print(f"  Cold: {losses[0]:.4f}, Warm: {losses[1]:.4f}")
        return match
    return None


def main():
    parser = argparse.ArgumentParser(description="gsplat L4 cold-start repro")
    parser.add_argument("--mode", choices=["synthetic", "training", "both"], default="synthetic")
    parser.add_argument("--data-dir", default="/workspace/data/360_v2/bonsai", help="Path to mipnerf360 bonsai dataset")
    parser.add_argument(
        "--gsplat-dir", default="/workspace/gsplat", help="Path to gsplat source (for simple_trainer.py)"
    )
    args = parser.parse_args()

    print()
    print("gsplat L4 cold-start rendering bug repro")
    print("=" * 60)
    print()

    results = {}

    if args.mode in ("synthetic", "both"):
        results["synthetic"] = test_synthetic()
        print()

    if args.mode in ("training", "both"):
        results["training"] = test_training(args.data_dir, args.gsplat_dir)
        print()

    print("=" * 60)
    any_fail = any(v is False for v in results.values())
    if any_fail:
        print("COLD-START BUG DETECTED on this GPU.")
        sys.exit(1)
    else:
        all_pass = all(v is True for v in results.values())
        if all_pass:
            print("No cold-start bug detected on this GPU.")
        else:
            print("Some tests were skipped. See above for details.")


if __name__ == "__main__":
    main()
