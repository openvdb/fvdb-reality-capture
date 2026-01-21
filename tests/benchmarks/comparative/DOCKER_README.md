# Docker Environment for FVDB vs GSplat Benchmark

Standalone Docker environment for running comparative benchmarks.

## Prerequisites

- Docker with NVIDIA runtime (`nvidia-docker`)
- NVIDIA drivers on host

## Quick Start

```bash
# Build
docker compose -f docker/docker-compose.yml build

# Run (detached) - auto-detects UID/GID from bind mounts
docker compose -f docker/docker-compose.yml up -d

# If auto-detection fails, export manually:
# export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"

# Check build status
docker logs fvdb-benchmark

# Open shell
docker compose -f docker/docker-compose.yml exec --user "$(id -u):$(id -g)" benchmark bash
```

## Running Benchmarks

Inside the container:

```bash
# Run all benchmarks from matrix.yml
python comparison_benchmark.py --matrix matrix.yml

# Plot existing results
python comparison_benchmark.py --matrix matrix.yml --plot-only
```

## Data Setup

The container mounts:
- `./data` → `/workspace/data`
- `./results` → `/workspace/results`

Download datasets with `frgs download all` (requires fvdb-reality-capture installed).

## Troubleshooting

```bash
# Check GPU
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Check mounts
docker compose -f docker/docker-compose.yml exec benchmark ls -la /workspace/
```

## Cleanup

```bash
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml down --rmi all  # also remove image
```
