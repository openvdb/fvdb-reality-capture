# FVDB vs GSplat Comparative Benchmark

This benchmark system provides comprehensive comparison between two 3D Gaussian Splatting implementations: FVDB and GSplat. It includes training, evaluation, analysis, and visualization capabilities.

## Features

- **Complete Training Pipeline**: Runs both FVDB and GSplat training with identical parameters
- **Comprehensive Metrics**: Tracks training time, PSNR, SSIM, LPIPS, and Gaussian count
- **Advanced Analysis**: Generates detailed comparative reports and summary statistics
- **Enhanced Visualization**: Creates professional plots with proper color schemes and labels
- **Flexible Execution**: Supports training, evaluation, plotting, and analysis modes
- **Robust Configuration**: Automatic path discovery and flexible parameter management
- **Side-by-Side Visualization**: Launch both viewers simultaneously for direct comparison

## Prerequisites

### Required Dependencies

1. **Python Dependencies**:
   ```bash
   pip install pyyaml matplotlib pandas numpy tyro
   ```

2. **System Dependencies**:
   - CUDA-compatible GPU for training
   - Sufficient GPU memory for 3D Gaussian Splatting

3. **Data Requirements**:
   - Mip-NeRF 360 dataset in `data/360_v2/` directory
   - COLMAP sparse reconstruction data for each scene

### Environment Setup

1. **FVDB Environment**: Ensure FVDB is properly installed and configured
2. **GSplat Environment**: Ensure GSplat is installed and accessible
3. **Data Preparation**: Ensure COLMAP data is in the correct format

## Benchmark Script (`comparison_benchmark.py`)

The main benchmark script provides comprehensive training, evaluation, and analysis capabilities.

### Usage

```bash
python3 comparison_benchmark.py --scenes bicycle garden --plot-only
```

### Command Line Options

- `--config PATH`: Path to benchmark config YAML (required unless --plot-only)
- `--scenes LIST`: Comma-separated list of scene names to benchmark (default: all scenes from config)
- `--result-dir PATH`: Directory to store results (default: results/benchmark)
- `--train-only`: Only run training (skip evaluation and plotting)
- `--eval-only`: Only run evaluation (skip training and plotting)
- `--plot-only`: Only generate plots from existing results (skip training and evaluation)
- `--frameworks LIST`: Comma-separated list of frameworks to run (default: fvdb,gsplat)
- `--log-level LEVEL`: Logging level (default: INFO)
- `--list-scenes`: List available scenes from config and exit
- `--help`: Show help message

### Examples

1. **Run all scenes from config**:
   ```bash
   python3 comparison_benchmark.py --config benchmark_config.yaml
   ```

2. **Run specific scenes**:
   ```bash
   python3 comparison_benchmark.py --config benchmark_config.yaml --scenes garden,bicycle
   ```

3. **List available scenes**:
   ```bash
   python3 comparison_benchmark.py --config benchmark_config.yaml --list-scenes
   ```

4. **Generate plots from existing results**:
   ```bash
   python3 comparison_benchmark.py --scenes garden,bicycle --plot-only
   ```

5. **Run training only**:
   ```bash
   python3 comparison_benchmark.py --config benchmark_config.yaml --scenes bicycle --train-only
   ```

### Output Structure

The benchmark creates a comprehensive output structure:

```
results/benchmark/
├── bicycle_fvdb/
│   └── *.log
├── bicycle_gsplat/
│   └── *.log
└── summary/
    ├── enhanced_comparative_report.md
    ├── detailed_comparative_data.csv
    ├── summary_comparison.png
    ├── summary_data.csv
    └── summary_data.json
```

### Metrics Collected

#### Training Metrics
- **Total Training Time**: Wall-clock time for complete training
- **Loss vs Time**: Loss values recorded at regular intervals
- **Final Loss**: Loss value at the end of training
- **Gaussian Count**: Number of Gaussians at final iteration

#### Evaluation Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **Evaluation Time**: Time per evaluation image

#### Performance Metrics
- **Speedup Ratio**: GSplat training time / FVDB training time
- **Gaussian Ratio**: FVDB Gaussian count / GSplat Gaussian count
- **Quality Comparison**: PSNR, SSIM, and LPIPS differences

### Summary Charts
- **Training Time Comparison**: Bar chart comparing training times across scenes
- **PSNR Comparison**: Bar chart comparing PSNR values across scenes
- **Gaussian Count Comparison**: Bar chart comparing final Gaussian counts

### Configuration File

The `benchmark_config.yaml` file contains:
- **Dataset Configuration**: Scene paths and settings
- **Training Parameters**: Batch size, epochs, learning rates
- **Evaluation Settings**: Metrics and thresholds

### Analysis Features

#### Reporting
- **Per-Scene Analysis**: Detailed metrics for each scene
- **Cross-Scene Comparison**: Summary statistics across all scenes
- **Performance Analysis**: Speedup and quality comparisons
- **Statistical Summary**: Mean, median, and standard deviation

#### Data Export
- **CSV Format**: For spreadsheet analysis
- **JSON Format**: For programmatic analysis
- **Markdown Reports**: For documentation and sharing

## Visualization Script (`visualize_comparison.py`)

A separate script for launching side-by-side viewers to directly compare FVDB and GSplat results.

### Usage

```bash
python3 visualize_comparison.py --scene bicycle
```

### Features

- **FVDB Viewer**: Runs on port 8080 (http://localhost:8080)
- **GSplat Viewer**: Runs on port 8081 (http://localhost:8081)
- **Automatic Path Discovery**: Attempts to find checkpoints and data automatically

### Command Line Options

- `--scene SCENE`: Scene name (e.g., bicycle, garden) (required)
- `--gsplat_port PORT`: Port for GSplat viewer (default: 8081)
- `--data_dir PATH`: Path to dataset directory (optional)
- `--fvdb_checkpoint PATH`: Path to FVDB checkpoint (optional)
- `--gsplat_checkpoint PATH`: Path to GSplat checkpoint (optional)

### Examples

1. **Basic usage**:
   ```bash
   python3 visualize_comparison.py --scene bicycle
   ```

2. **Custom ports**:
   ```bash
   python3 visualize_comparison.py --scene garden --gsplat_port 8082
   ```

3. **Manual checkpoint paths**:
   ```bash
   python3 visualize_comparison.py --scene bicycle --fvdb_checkpoint /path/to/fvdb.pt --gsplat_checkpoint /path/to/gsplat.pt
   ```

### Remote Visualization

The visualization script works well for remote viewing:
- **FVDB Viewer**: Accessible via browser at http://localhost:8080
- **GSplat Viewer**: Accessible via browser at http://localhost:8081

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install pyyaml matplotlib pandas numpy tyro
   ```

2. **Config File Not Found** (Benchmark Script):
   - Ensure `benchmark_config.yaml` exists in the current directory
   - Use `--config` to specify a different config file path

3. **Checkpoint Not Found** (Visualization Script):
   - Use `--fvdb_checkpoint` or `--gsplat_checkpoint` to specify manually
   - Check that checkpoints exist in the expected locations

4. **Data Directory Not Found**:
   - Use `--data_dir` to specify the dataset path manually
   - Check that the dataset is in the expected location
