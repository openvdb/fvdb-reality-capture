<!--
  Copyright Contributors to the OpenVDB Project
  SPDX-License-Identifier: Apache-2.0
-->

# TSDF / ESDF / Occupancy cross-library benchmarks

Benchmark drivers comparing fvdb against external sparse-voxel
libraries (nvblox, VDBFusion, Open3D) on real and synthetic RGB-D /
LiDAR workloads.

The drivers live in [`cross_library/`](cross_library/) and are
organised by dataset and workload:

| Driver                          | Dataset            | Workload                                        |
|---------------------------------|--------------------|-------------------------------------------------|
| `bench_open3d_vs_fvdb.py`       | Synthetic sphere   | TSDF, fvdb vs Open3D                            |
| `bench_replica_full.py`         | Replica            | TSDF + ESDF, multi-scene, full frame rate       |
| `bench_esdf_replica.py`         | Replica            | ESDF scale sweep                                |
| `bench_kitti.py`                | KITTI Odometry     | LiDAR TSDF                                      |
| `bench_esdf_kitti.py`           | KITTI Odometry     | LiDAR ESDF                                      |
| `bench_occupancy_kitti.py`      | KITTI Odometry     | LiDAR occupancy                                 |
| `bench_decay_kitti.py`          | KITTI Odometry     | Dynamic-scene decay-and-prune                   |
| `bench_mai_city.py`             | Mai City (LiDAR)   | LiDAR TSDF, fvdb vs VDBFusion vs nvblox         |
| `bench_seven_scenes.py`         | 7-Scenes (RGB-D)   | TSDF, long-trajectory                           |
| `bench_esdf_vs_nvblox.py`       | Mai City           | ESDF, fvdb vs nvblox                            |
| `bench_occupancy_vs_nvblox.py`  | Mai City           | Occupancy, fvdb vs nvblox                       |

Supporting modules in the same directory:

- `kitti_loader.py`, `replica_loader.py`, `mai_city_loader.py`,
  `seven_scenes_loader.py` — minimal dataset loaders that yield
  per-frame `(intrinsics, cam-to-world, depth_or_points)` tuples.
- `download_kitti.py`, `download_replica.py`,
  `download_replica_zip.py` — resumable, parallel-stream downloaders
  for the public dataset archives.
- `nvblox_runner.py` — Python wrapper around the nvblox CLI that
  matches the cross-library interface used by the bench drivers.
- `install_nvblox.sh`, `install_vdbfusion.sh` — reproducible install
  recipes for the two C++ comparison libraries; each builds against
  the fvdb conda env's TBB / Blosc / Boost.

Each bench script accepts `--help` for its full argument surface and
typically writes a JSON results file alongside printed per-frame
summaries.

## Data

Datasets are not vendored. Place them under
[`data/`](data/) (gitignored) and point each bench driver at the
appropriate subdirectory via its `--data-root` (or equivalent) flag.