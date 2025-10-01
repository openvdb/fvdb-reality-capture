# Docker Environment for FVDB vs GSplat Benchmark

This Docker setup provides a standalone environment for running the comparative benchmark between FVDB and GSplat.

## Prerequisites

1. **Docker**: Install Docker and Docker Compose
2. **NVIDIA Docker**: Install NVIDIA Docker runtime for GPU support
3. **NVIDIA Drivers**: Ensure NVIDIA drivers are installed on the host

## Architecture

At the time of writing, some packages were not available pre-built for Blackwell GPUs (e.g. 5090 series)
and the required CUDA toolkit 12.8 or higher. Therefore we provide a custom Docker setup for Blackwell
(compute 12.0) as well as for earlier GPUs. This Docker setup uses a parameterized configuration that
supports both through environment files:

- **`docker/docker-compose.yml`**: Single, parameterized compose file
- **`docker/env.standard`**: Environment variables for pre-Blackwell GPUs
- **`docker/env.blackwell`**: Environment variables for Blackwell GPUs (RTX 50xx series)

The key differences are:
- **Dockerfile**: `Dockerfile` vs `Dockerfile.blackwell`
- **Container name**: `fvdb-benchmark` vs `fvdb-benchmark-blackwell`
- **CUDA architecture**: Auto-detection vs `TORCH_CUDA_ARCH_LIST=12.0`
- **Build parallelism**: Both use `CMAKE_BUILD_PARALLEL_LEVEL=18`

## Quick Start

### 1. Build the Docker Image

Note: the following `docker` commands use `docker-compose` v2 "plugin" syntax. For older
`docker-compose` standalone syntax, change `docker compose` to `docker-compose`.

For pre-Blackwell GPUs:
```bash
docker compose -f docker/docker-compose.yml --env-file docker/env.standard build
```

For Blackwell GPUs (RTX 50xx series):
```bash
docker compose -f docker/docker-compose.yml --env-file docker/env.blackwell build
```

### 2. Run the Container

#### Pre-Blackwell GPUs:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/env.standard up -d
```

After starting, the fvdb build will continue in the background. Check its status with:

```bash
docker logs fvdb-benchmark
```

Open an interactive bash shell in the container:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/env.standard exec benchmark bash
```

#### Blackwell GPUs:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/env.blackwell up -d
```

After starting, the fvdb build will continue in the background. Check its status with:

```bash
docker logs fvdb-benchmark-blackwell
```

Open an interactive bash shell in the container:
```
docker compose -f docker/docker-compose.yml --env-file docker/env.blackwell exec benchmark bash
```

### 3. Run Benchmarks

Once inside the container:

```bash
By default the script looks for the configuration in `benchmark_config.yaml` in the current
directory. You can override this with `--config=<path/to/config_file.yaml>`.

# List available scenes
python3 comparison_benchmark.py --list-scenes

# Run training benchmark for specific scenes
python3 comparison_benchmark.py --scenes bicycle,garden

# Run training benchmark for all scenes on only the gsplat framework
python3 comparison_benchmark.py --frameworks gsplat

# Generate plots from existing results for the bicycle and garden scenes
python3 comparison_benchmark.py --scenes bicycle,garden --plot-only

# Launch visualization
python3 visualize_comparison.py --scene bicycle
```

## Data Setup

### Mounting Data

The Docker setup mounts the following directories:

- `./data` → `/workspace/data` (for Mip-NeRF 360 dataset)
- `./results` → `/workspace/results` (for benchmark results)
- `./benchmark_config.yaml` → `/workspace/benchmark_config.yaml`

### Preparing Data

1. **Place your Mip-NeRF 360 dataset** in the `./data` directory:
   ```
   ./data/
   └── 360_v2/
       ├── bicycle/
       ├── garden/
       ├── bonsai/
       └── ...
   ```

2. **Ensure benchmark_config.yaml** is in the current directory

3. **Create results directory**:
   ```bash
   mkdir -p results
   ```

## Visualization

The container exposes ports for visualization:

- **Port 8080**: FVDB viewer (http://localhost:8080)
- **Port 8081**: GSplat viewer (http://localhost:8081)

### Accessing Viewers

When running visualization inside the container, you can access the viewers from your host machine:

```bash
# Inside container
python3 visualize_comparison.py --scene bicycle

# On host machine - open browser to:
# http://localhost:8080 (FVDB viewer)
# http://localhost:8081 (GSplat viewer)
```

## Environment Details

### Conda Environment

The container uses a custom conda environment called `benchmark` with:

- **Python 3.11**
- **PyTorch with CUDA 12.1**
- **All benchmark dependencies**: tyro, pyyaml, matplotlib, pandas, etc.
- **Visualization tools**: viser, nerfview

### GPU Support

The container is configured for NVIDIA GPU support:

- Uses `nvidia/cuda:12.1-devel-ubuntu22.04` base image
- Includes CUDA toolkit and PyTorch with CUDA support
- Exposes all NVIDIA devices to the container

## Troubleshooting

### GPU Issues

If GPU is not detected:

```bash
# Check if NVIDIA Docker is working
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Check GPU inside container
docker-compose exec benchmark nvidia-smi
```

### Port Conflicts

If ports 8080 or 8081 are already in use, modify `docker/docker-compose.yml`:

```yaml
ports:
  - "8082:8080"  # Map host port 8082 to container port 8080
  - "8083:8081"  # Map host port 8083 to container port 8081
```

### Data Access Issues

Ensure data directories are properly mounted:

```bash
# Check mounted volumes (use appropriate env file)
docker-compose -f docker/docker-compose.yml --env-file docker/env.standard exec benchmark ls -la /workspace/

# Check data directory
docker-compose -f docker/docker-compose.yml --env-file docker/env.standard exec benchmark ls -la /workspace/data/
```

## Development

### Rebuilding Environment

To rebuild the Docker image after changes:

```bash
# For pre-Blackwell GPUs
docker compose -f docker/docker-compose.yml --env-file docker/env.standard down
docker compose -f docker/docker-compose.yml --env-file docker/env.standard build --no-cache
docker compose -f docker/docker-compose.yml --env-file docker/env.standard up -d

# For Blackwell GPUs
docker compose -f docker/docker-compose.yml --env-file docker/env.blackwell down
docker compose -f docker/docker-compose.yml --env-file docker/env.blackwell build --no-cache
docker compose -f docker/docker-compose.yml --env-file docker/env.blackwell up -d
```

### Adding Dependencies

To add new dependencies:

1. **Edit `docker/benchmark_environment.yml`** (standard) or `docker/benchmark_environment_blackwell.yml` (Blackwell) to add conda/pip packages
2. **Rebuild the image**:
   ```bash
   # For pre-Blackwell GPUs
   docker-compose -f docker/docker-compose.yml --env-file docker/env.standard build --no-cache

   # For Blackwell GPUs
   docker-compose -f docker/docker-compose.yml --env-file docker/env.blackwell build --no-cache
   ```

### Persistent Data

All results are saved to the mounted `./results` directory and persist between container restarts.

## Cleanup

To stop and remove the container:

```bash
# For pre-Blackwell GPUs
docker-compose -f docker/docker-compose.yml --env-file docker/env.standard down

# For Blackwell GPUs
docker-compose -f docker/docker-compose.yml --env-file docker/env.blackwell down
```

To remove the image as well:

```bash
# For pre-Blackwell GPUs
docker-compose -f docker/docker-compose.yml --env-file docker/env.standard down --rmi all

# For Blackwell GPUs
docker-compose -f docker/docker-compose.yml --env-file docker/env.blackwell down --rmi all
```
