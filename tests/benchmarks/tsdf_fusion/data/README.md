# TSDF fusion benchmark datasets

This directory is git-ignored except for this README and `.gitignore`.
Populate it by downloading the NICE-SLAM Replica rendering (and,
optionally, per-scene GT meshes) — see below.

## Replica (NICE-SLAM rendering)

- **What**: 8 rendered RGB-D sequences over the Replica rooms /
  offices. Each scene is ~1.6 GB on disk, full bundle 12.4 GB.
- **Where**:
  ```bash
  cd $THIS_DIR
  curl -L -O https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
  unzip -q Replica.zip
  ```
- **Layout expected by `replica_loader.py`**:
  ```
  data/Replica/
    room_0/
      results/
        frame000000.jpg
        depth000000.png
        ...
      traj.txt
    office_0/
      ...
  ```

## GT meshes (Replica, for F-score / Chamfer-L1)

The NICE-SLAM rendering zip above does NOT include GT meshes. They
come from Meta's original Replica-Dataset release, which has a
larger footprint and a license-acceptance gate. For initial
benchmark runs the script prints perf numbers without quality
metrics if no GT mesh is provided.

To enable quality metrics, obtain the GT mesh for a scene (e.g.
`room_0/mesh.ply`) from the Replica-Dataset release and pass its
path via `--gt-mesh` to `bench_open3d_vs_fvdb.py`.

A smaller "just the meshes" drop has occasionally circulated in the
community (e.g. via the NICE-SLAM GitHub issues); if we find a
stable URL we'll add a direct-download step here.

## Disk budget

| item              | size |
|-------------------|------|
| Replica.zip       | 12.4 GB (full) |
| Replica/ extracted | ~13 GB (8 scenes)|
| room_0 alone      | ~1.6 GB |
