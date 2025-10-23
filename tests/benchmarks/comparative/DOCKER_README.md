# Docker Environment for FVDB vs GSplat Benchmark

This Docker setup provides a standalone environment for running the comparative benchmark between FVDB and GSplat.

## Prerequisites

1. **Docker**: Install Docker and Docker Compose
2. **NVIDIA Docker**: Install NVIDIA Docker runtime for GPU support
3. **NVIDIA Drivers**: Ensure NVIDIA drivers are installed on the host

## Quick Start

Note: the following `docker` commands use `docker-compose` v2 "plugin" syntax. For older
`docker-compose` standalone syntax, change `docker compose` to `docker-compose`.

### 1. Build the Docker Image

```bash
docker compose -f docker/docker-compose.yml build
```

### 2. Run the Container

```bash
docker compose -f docker/docker-compose.yml up -d
```

After starting, the fvdb build will continue in the background. Check its status with:

```bash
docker logs fvdb-benchmark
```

Open an interactive bash shell in the container:

```bash
docker compose -f docker/docker-compose.yml exec benchmark bash
```

### 3. Run Benchmarks

Benchmark configuration is controlled by two files.

 - The benchmark config file specifies paths and datasets. By default the benchmark script looks for this in
   `benchmark_config.yaml` in the current directory. You can override this with
   `--benchmark-config=<path/to/config_file.yml>`.
 - The optimization config file specifies settings for the gaussian splatting optimizer (e.g. fvdb or gsplat).
   We provide some default configs in opt_configs/ that you can customize, and some fast-running (single epoch) configs
   for debugging in opt_configs/debug.

Once inside the container:

```bash

# Run training benchmark for all scenes on fvdb and gsplat with specified default options
python3 comparison_benchmark.py --opt-configs ./opt_configs/fvdb_default.yml ./opt_configs/gsplat_default.yaml

# List available scenes
python3 comparison_benchmark.py --list-scenes

# Run training benchmark for specific scenes with fvdb default options
python3 comparison_benchmark.py --opt-configs ./opt_configs/fvdb_default.yml --scenes bicycle,garden

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

With `fvdb-reality-capture` installed, use `frgs download all` to download the fvdb benchmark datasets

   ```
   ./data/
   └── 360_v2/
       ├── bicycle/
       ├── garden/
       ├── bonsai/
       └── ...
   └── safety_park/
   └── gettysburg/
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
docker-compose -f docker/docker-compose.yml exec benchmark ls -la /workspace/

# Check data directory
docker-compose -f docker/docker-compose.yml benchmark ls -la /workspace/data/
```

## Development

### Rebuilding Environment

To rebuild the Docker image after changes:

```bash
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml build --no-cache
docker compose -f docker/docker-compose.yml up -d
```

### Adding Dependencies

To add new dependencies:

1. **Edit `docker/benchmark_environment.yml`** to add conda/pip packages
2. **Rebuild the image**:
   ```bash
   docker-compose -f docker/docker-compose.yml build --no-cache
   ```

### Persistent Data

All results are saved to the mounted `./results` directory and persist between container restarts.

## Cleanup

To stop and remove the container:

```bash
docker-compose -f docker/docker-compose.yml down
```

To remove the image as well:

```bash
docker-compose -f docker/docker-compose.yml down --rmi all
```
