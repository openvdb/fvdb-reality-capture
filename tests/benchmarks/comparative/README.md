# FVDB vs GSplat Comparative Benchmark

Compares 3D Gaussian Splatting implementations: FVDB and GSplat.

## Quick Start

```bash
# Run all benchmarks defined in matrix.yml
python comparison_benchmark.py --matrix matrix.yml

# Generate plots from existing results
python comparison_benchmark.py --matrix matrix.yml --plot-only
```

## Configuration

Benchmarks are configured via a **matrix YAML file** (e.g., `matrix.yml`):

```yaml
name: example_matrix

paths:
  gsplat_base: ../gsplat/examples
  data_base: /workspace/data

datasets:
  - name: garden
    path: 360_v2/garden
  - name: bicycle
    path: 360_v2/bicycle

opt_configs:
  fvdb_default:
    path: opt_configs/fvdb_default.yml
  gsplat_default:
    path: opt_configs/gsplat_default.yml

runs:
  - dataset: garden
    opt_config: fvdb_default
  - dataset: garden
    opt_config: gsplat_default
    overrides:
      gsplat:
        cli_args:
          - --strategy.cap_max
          - "5000000"
```

### Optimization Configs

Stored in `opt_configs/`. Each defines framework-specific training parameters:

- `fvdb_default.yml` / `fvdb_mcmc_default.yml` - FVDB configurations
- `gsplat_default.yml` / `gsplat_mcmc_default.yml` - GSplat configurations

## Command Line Options

```
--matrix PATH      Path to matrix YAML file (required)
--plot-only        Only generate plots from existing results
--log-level LEVEL  Logging level (default: INFO)
```

## Output

Results are saved to `results/<matrix_name>/`:

```
results/example_matrix/
├── garden__fvdb_default/
│   └── training.log
├── garden__gsplat_default/
│   └── training.log
├── garden_comparison_report.json
└── summary/
    ├── summary_comparison.png
    └── summary_data.json
```

### Metrics

- **Training Time** / **Total Time**
- **PSNR**, **SSIM** (quality)
- **Gaussian Count**
- **Peak GPU Memory**
- **Training Throughput** (splats/s)

## Docker

See [DOCKER_README.md](DOCKER_README.md) for containerized execution.
