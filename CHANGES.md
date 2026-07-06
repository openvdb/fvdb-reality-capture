fVDB-Reality-Capture Version History
====================================

## Version 0.5.0 - July 1, 2026

*23 commits, 100+ files changed, 7 contributors.*

This release tracks fVDB 0.5.0. It catches up to fVDB-core's new composable camera and batched-image APIs, adds dense depth supervision for Gaussian splat reconstruction, reworks Gaussian splat USD export onto the OpenUSD ParticleField3DGaussianSplat standard, moves documentation to a versioned Read the Docs site, switches to the upstream PyCOLMAP, and hardens the release/CI pipeline and repository governance across the fVDB repositories.

**Highlights:**
- Reworked Gaussian splat reconstruction on top of fVDB-core's new camera/render API (fVDB-core #518), cleaning up how reality-capture represents cameras, distortion, and world-space render behavior.
- Added a `DepthMapAttribute` for per-image depth rasters and wired optional dense depth supervision into `GaussianSplatReconstruction`.
- Reworked Gaussian splat USD export onto the OpenUSD `ParticleField3DGaussianSplat` schema (Isaac Sim 6.0+), defaulting to single-file `.usdc` output with opt-in `.usdz`, and expanded `frgs convert` with mesh embedding, upright rotation, and prim-naming options for Isaac Sim.
- Fixed a scene-normalization bug that inverted camera matrices, and a TSDF meshing crash caused by fVDB-core's move to a batched depth-image API.
- Switched COLMAP dependency management to the official PyCOLMAP repository and synced the benchmark environment to PyTorch 2.11.
- Migrated documentation to a versioned Read the Docs site.
- Split `CODEOWNERS` into two review tiers (NVIDIA sign-off for governance/CI files) and fixed the CI change-detection gate for docs-only PRs, kept consistent with fvdb-core and fvdb-examples.

**Contributors:** @harrism, @matthewdcong, @swahtz, @fwilliams, @dylan-eustice, @phapalova, @zlalena

---

### Reconstruction & Gaussian Splatting

**New Features:**
- Reworked Gaussian splat reconstruction to support world-space rendering on top of fVDB-core's new composable camera/render API (fVDB-core #518), cleaning up how reality-capture represents cameras, distortion models, and render-time camera behavior (#253 - @fwilliams).
- Added `DepthMapAttribute`, a per-image depth-raster attribute with scale-aware (metric vs. relative) semantics, and wired optional dense depth supervision into `GaussianSplatReconstruction` (#288 - @fwilliams).

**Bug Fixes:**
- Fixed a crash when `accumulated_gradient_step_counts` is `None` during Gaussian splat refinement (#281 - @harrism).
- Fixed performance regressions with pinhole camera models introduced by the camera-model rework (#280 - @matthewdcong).

---

### Structure-from-Motion & Scene Handling

- Fixed scene normalization storing a reference to `camera_to_world_matrices` instead of `world_to_camera_matrices`, which inverted the transform passed to similarity normalization and produced an incorrect scene scale (#285 - @matthewdcong).

---

### Mesh Reconstruction (TSDF Fusion)

- Fixed a crash in `mesh_from_splats` / `tsdf_from_splats` / `tsdf_from_splats_dlnr` caused by fVDB-core's move to a batched depth-image API, realigning reality-capture with fVDB-core main (#292 - @dylan-eustice).

---

### USD Export & Isaac Sim Integration

**New Features:**
- Reworked Gaussian splat USD export onto the OpenUSD `UsdVol.ParticleField3DGaussianSplat` schema (Isaac Sim 6.0+), and defaulted exports to a single self-contained `.usdc` file with `.usdz` archive packaging available opt-in. The legacy Omniverse NuRec format (`UsdVol.Volume` + `.nurec`) is retained behind a `legacy` / `--legacy` flag for Isaac Sim versions prior to 6.0. This renames the export entry points from `export_splats_to_usdz` / `GaussianSplatReconstruction.save_usdz` to `export_splats_to_usd` / `save_usd` (taking a `usdz` flag), a breaking API change (#294 - @zlalena, @swahtz).
- Extended `frgs convert` to export USD (`.usdc`/`.usdz`) with optional collision-mesh embedding, ecef2enu upright rotation for Isaac Sim, a customizable asset prim name (`--prim-path`), and legacy-format selection (`--legacy`) (#294 - @zlalena, @swahtz).
- Overhauled `scripts/create_isaac_ready_files.py` to produce a single aligned mesh + splat Isaac-ready asset, with optional bounding-box cropping, origin centering, and watertight mesh conversion, and added `tests/unit/test_export_splats_to_usd.py` covering `.usdc`/`.usdz` write/read round-trips (#294 - @zlalena, @swahtz).

**Bug Fixes:**
- Hardened degenerate cases in USD export: zero-norm quaternions no longer produce NaN/Inf orientations, empty gaussian/mesh sets fail fast with clear errors (falling back to mesh-only export when a crop removes all splats), and shN/SH-degree coefficient-count mismatches now emit a warning instead of silently padding or truncating (#294 - @zlalena, @swahtz).

---

### PyTorch & Dependency Compatibility

- Switched from a fork to the official PyCOLMAP repository for the COLMAP dependency, and fixed a PyCOLMAP version mismatch that was failing nightly tests (#293, #297 - @matthewdcong).
- Synced the benchmark environment to PyTorch 2.11 to match the fVDB-core build (#296 - @harrism).
- Raised the minimum `usd-core` to `>=26.3` for `ParticleField3DGaussianSplat` schema support (#294 - @zlalena, @swahtz).

---

### Documentation

- Migrated documentation to a versioned Read the Docs site (#284 - @swahtz).
- Added a notebook showing how to create a COLMAP dataset for use in fVDB-Reality-Capture (#268 - @zlalena).
- Fixed installation instructions: removed the outdated `editor_force` flag and updated the referenced fVDB-core version (#290, #275 - @phapalova, @harrism).

---

### Benchmarks & Nightly CI

- Updated the nightly benchmark to PyTorch 2.10 / CUDA 13.0 (later synced to 2.11) (#272 - @harrism).
- Fixed a nightly benchmark artifact-download `JSONDecodeError` (#278 - @harrism).

---

### CI / DevOps / Governance

- Decoupled the PyPI and S3 publish targets so both run on every release, replacing the mutually-exclusive `s3` flag with independent routing flags (#267 - @swahtz).
- Added an event-driven issue-triage labels workflow and hardened its team-membership check (#274, #276 - @harrism).
- Split `CODEOWNERS` into two review tiers — general code reviewable by any maintainer, while governance, legal, and CI/CD files require an NVIDIA maintainer — and added the governance docs (`MAINTAINERS.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`). Kept identical across `fvdb-core`, `fvdb-reality-capture`, and `fvdb-examples` (#298 - @harrism).
- Fixed required status checks being skipped (and permanently blocking) on docs-only PRs, and corrected the change-detection gate so docs-only PRs skip cleanly while code and mixed PRs still run tests (#299, #300 - @harrism).

## Version 0.4.0 - March 14, 2026

*55 commits, 92 files changed, 7 contributors.*

This release focuses on Gaussian splatting quality and performance, foundation-model integrations for segmentation and open-vocabulary workflows, a major overhaul of the comparative benchmarking system, and the project's first automated publish/release pipeline.

**Highlights:**
- Gaussian splatting gains a new MCMC-based optimizer and sparse depth regularization, alongside an extensible custom-attribute system for `SfmScene` that lets downstream projects (GARfVDB, LangSplatV2) attach typed per-point/per-image/per-camera data that propagates automatically through the transform pipeline.
- New foundation-model wrappers — OpenCLIP, SAM1, and multi-scale SAM2 mask generation — bring open-vocabulary and segmentation support to the reconstruction pipeline.
- Gaussian splat training gets faster: Morton-ordered Gaussian storage improves spatial locality for an 8-10% speedup, and broadcasting appended optimizer state saves ~200ms per iteration on multi-GPU runs.
- The comparative benchmark system was overhauled with matrix-based configuration, GPU memory tracking, time-series metric plots, and a new nightly job that trains fVDB against GSplat side-by-side on MipNeRF360 scenes.
- Shipped the first automated `publish.yml` workflow: wheel builds, S3 staging, and PyPI publication with GPU validation smoke tests, plus numerous SfM/COLMAP loader correctness fixes and NVIDIA-branded documentation.

**Contributors:** @diz-vara, @eh-dub, @fwilliams, @harrism, @matthewdcong, @NotMorven, @swahtz

---

### Gaussian Splatting & Optimization

**New Features:**
- Added a Markov Chain Monte Carlo (MCMC) optimizer for Gaussian splat radiance field reconstruction, along with a refactored optimizer registration and deserialization system for extensibility (#214 - @fwilliams, @harrism).
- Added sparse depth regularization, applying a simple L1 loss against sparse depth supervision during Gaussian splat training (#188 - @fwilliams).

**Optimizations:**
- Reordered Gaussian storage using Morton (Z-order) codes to preserve spatial locality as Gaussians are refined and appended, yielding an 8-10% training speedup (#233 - @matthewdcong).
- Reduced Gaussian splat optimizer overhead by broadcasting appended parameter tensors instead of allocating and filling large zero tensors, saving roughly 200ms per optimizer iteration on multi-GPU reconstructions (#248 - @matthewdcong).

**Bug Fixes:**
- Fixed dataloader overhead, progress-bar accuracy, and cache-path handling when restarting Gaussian splat training from a checkpoint (#159, #166, #202 - @matthewdcong).
- Fixed GSplat config parameters not being passed through to `simple_trainer.py` (#236 - @harrism).
- Fixed a `NameError` for `training_images` in `run_gsplat_training.py` (#247 - @harrism).

---

### SfM / COLMAP Dataset

- Added an extensible, pluggable custom-attribute system for `SfmScene`, letting downstream projects register typed per-point, per-image, and per-camera data that automatically propagates through the transform pipeline — filtering, spatial transforms, image downsampling, and cropping (#245 - @fwilliams).
- Fixed a `SfmCameraMetadata`/downsample interaction bug where undistorted (rather than original) image dimensions and intrinsics were serialized, risking double application of undistortion on deserialization (#228 - @swahtz).
- Added downsampling and caching of SfM masks, reusing cached masks on subsequent loads (#219 - @diz-vara).
- Fixed handling of images with empty `point_indices`, which previously failed to load correctly (#220 - @diz-vara).
- Fixed COLMAP `images.txt` loading to handle images with empty feature points (#177 - @NotMorven).

---

### Foundation Models

- Added an `OpenCLIPModel` wrapper for encoding images and text into a shared embedding space via OpenCLIP (#231 - @swahtz).
- Added a `SAM1Model` wrapper providing an interface consistent with `SAM2Model`, for parity with LangSplatV2-style experiments (#242 - @swahtz).
- Extended `SAM2Model` with multi-scale mask generation, supporting both the original flat mask mode and SAM2's native small/medium/large mask semantics used by LangSplatV2 (#239 - @swahtz).

---

### Benchmarking

- Added a nightly comparative benchmark job that trains fVDB and GSplat side-by-side on MipNeRF360 scenes (garden, bonsai, bicycle), tracking PSNR/SSIM and training time/peak memory for regression detection (#240 - @harrism).
- Overhauled the comparative benchmark system with a matrix-based `matrix.yml` configuration (replacing separate `--config`/`--opt-configs` flags), GPU memory tracking, and support for comparing specific commits (#226, #235 - @harrism).
- Added nightly Gaussian splatting unit benchmarks and accompanying CI tests, plus throughput and time-series training-metric plots for comparison benchmarks (#164, #168, #225, #227 - @harrism).
- Fixed numerous nightly-benchmark CI reliability issues: workflow triggering in forked repos, skipping runs with no new commits, stale local files, checkpoint/config layout drift, and worker-count sizing via `os.cpu_count()` (#169, #171, #173, #209, #223, #232, #234 - @harrism).
- Fixed 3DGS unit benchmarks to match the updated checkpoint API (#162 - @harrism).

---

### CI / Release Infrastructure

- Added the project's first automated publish workflow: builds a pure-Python wheel, stages it to S3 on `release/v*` pushes, and publishes to PyPI/TestPyPI, with GPU validation smoke tests against fvdb-core and unit/benchmark contract tests (#258 - @harrism).
- Iterated on the publish workflow to fix wheel URL resolution, glob patterns, AWS credentials, and Rocky Linux 8 validation (#260, #262 - @harrism), and switched it to use `uv` for Python installation (#263, #264, #265 - @swahtz).
- Fixed CI runner action tokens and a unit-test job that wasn't merging in fork-branch changes (#212, #229 - @swahtz).
- Added an `aarch64` workaround using the `usd-exchange` package in place of `usd-core`, which lacks aarch64 binaries (#190 - @matthewdcong).

---

### Documentation

- Applied NVIDIA branding to the documentation site (#217 - @fwilliams).
- Added Google Analytics to the documentation site and removed a stale `_Cpp` reference from the docs configuration (#160 - @fwilliams; #161 - @harrism).
- Fixed a typo in the sensor data loading tutorial and updated the demo notebook for the viewer's scene-reset behavior (#174 - @eh-dub; #158 - @swahtz).
- Added `AGENTS.md` with guidance for AI coding agents working in the repository (#238 - @harrism).

## Version 0.3.0 - October 24, 2025

*163 commits, ~150 files changed, 8 contributors.*

This is the initial public release of fVDB-Reality-Capture, a toolbox built on top of fVDB for turning multi-view captures into 3D Gaussian splat reconstructions, meshes, and other derived assets. The release establishes the core pipeline end-to-end: loading COLMAP/e57/simple-directory SfM captures into a common `SfmScene` representation, training and refining 3D Gaussian splats on fVDB's `GaussianSplat3d`, extracting meshes via TSDF fusion, a `frgs` command-line tool, benchmarking utilities, foundation-model-assisted masking, and a full documentation and CI setup.

**Highlights:**
- New `SfmScene`/`ColmapDataset` data model for loading and transforming COLMAP, e57, and plain-directory captures, with a composable torchvision-style transform pipeline and an on-disk caching layer.
- A Gaussian splatting reconstruction pipeline built on fVDB's `GaussianSplat3d`, with a documented and optimized `GaussianSplatOptimizer` (refinement, pose optimization, spatial chunking for large scenes) and checkpointing.
- Mesh reconstruction from trained splats via TSDF fusion, including a DLNR-based stereo-depth path and SAM2-based foundation-model masking for cleaner reconstructions.
- PLY/USDZ export and S3 upload/download utilities, unified behind a single pip-installable `frgs` CLI (download, reconstruct, convert, show-data, show, evaluate, mesh, mesh-dlnr).
- A benchmarking suite (end-to-end benchmark, Nsight profiling scripts, Dockerized CI runs) plus a full Sphinx/GitHub Pages documentation site with tutorials and notebooks.

**Contributors:** @fwilliams, @swahtz, @harrism, @matthewdcong, @bbartlett-nv, @zlalena, @vinegh4, @phapalova

---

### COLMAP/SfM Dataset Loading

- Rewrote the COLMAP dataset loader and introduced the `SfmScene` data model as the common representation for captures (@fwilliams).
- Added loading of SfM scenes from **e57** scanner data, with follow-up fixes for robustness (#39, #41 - @fwilliams).
- Added a loader for a simple directory of images, JSON camera poses, and a PLY point cloud (`SfmScene.from_simple_directory`), including handling for datasets without image-to-point visibility mappings (#82 - @fwilliams).
- Added a torchvision-style composable transform pipeline for the reality-capture "battery" of transforms, plus a dedicated `TransformScene` transform for applying an arbitrary transform matrix (e.g. one saved in a checkpoint) to an `SfmScene` (@fwilliams; #142 - @swahtz).
- Added a better/self-contained caching API (`Cache`, SQLite-backed) for derived dataset artifacts, replacing the earlier `DatasetCache` (@fwilliams).
- Fixed non-deterministic PCA in the COLMAP dataset loader caused by OpenBLAS, switching to MKL (@swahtz).
- Fixed bugs found running against real-world (nvrobotics) and text-format COLMAP data, including correct step counting for batch size > 1 (#134, #135 - @fwilliams).
- Unit tests added for `SfmScene` and transforms (#27 - @fwilliams), and a tutorial notebook for `SfmScene` (#139 - @fwilliams).
- Added support for e57/PLY safety-park and other example datasets, and an `miris_factory` example dataset (#40, #82 - @fwilliams).

### Gaussian Splatting Training & Optimization

**New Features:**
- Migrated the reconstruction pipeline onto fVDB's C++ `GaussianSplat3d` class (`from fvdb import GaussianSplat3d`), replacing the project's earlier Python-side Gaussian splat representation (@fwilliams).
- Refactored pose optimization and rewrote `GaussianSplatOptimizer` with documented, configurable refinement (splitting/duplication/deletion), percentile-based gradient pruning thresholds, and deferred pose optimization until after refinement completes (#66, #84 - @fwilliams).
- Added spatial chunking to `GaussianSplatReconstruction` (`nchunks`/`chunk_overlap_pct`), splitting large scenes into overlapping crops that are reconstructed independently and merged back together (@fwilliams).
- Added functions to filter `GaussianSplat3d` results by mean, opacity, or scale (#98 - @swahtz).
- Rolled a self-contained PSNR and LPIPS implementation to remove the `torchmetrics` dependency, then switched to SSIM/PSNR from `fvdb.utils.metrics` (@fwilliams; #29 - @harrism).
- Renamed `SceneOptimizationRunner` to `GaussianSplatReconstruction`, reworked checkpointing to serialize `SfmScene`s directly, and promoted USDZ export to its own tool as part of a broader API/notebook overhaul (#96 - @fwilliams).
- Added a `frgs evaluate` script and an `frgs mesh`/`frgs mesh-dlnr` extraction path built on the new checkpoint API (@fwilliams).

**Optimizations:**
- Fixed a performance regression in the Gaussian splat loss computation (#137 - @matthewdcong).
- Fused the L1/SSIM loss interpolation into a single `torch.lerp` call to cut memory bandwidth and temporary tensors (#85 - @matthewdcong).
- Deferred `loss.item()` synchronization to a single point per iteration, improving training throughput (#144 - @matthewdcong).
- Reduced extra parameter copies during Gaussian refinement (duplication/splitting/deletion) to cut memory usage (#84 - @fwilliams).

**Bug Fixes:**
- Fixed model checkpointing and restarting training from a checkpoint, including a `weights_only=False` load failure under newer PyTorch (@fwilliams; #147 - @matthewdcong).
- Fixed a bug in Gaussian splat refinement shared with the gsplat/INRIA reference implementations that improves reconstruction quality (#84 - @fwilliams).
- Fixed I/O and small numerical bugs found while running on Puerto Rico scene data and other real captures (@fwilliams; #134 - @fwilliams).
- Fixed the training metric label in TensorBoard logging (@matthewdcong) and fixed `tensorboard add_images` (#138 - @swahtz).

### Mesh Reconstruction (TSDF Fusion)

- Added mesh reconstruction from trained Gaussian splats via TSDF fusion (`_tsdf_from_splats.py`/`_mesh_from_splats.py`, exposed as `frgs mesh`) (@fwilliams).
- Added support for per-image weighting in TSDF fusion (@fwilliams) and extra meshing parameters for thresholding low-opacity background pixels and downsampling large images (#95 - @fwilliams).
- Added a DLNR-based stereo-depth meshing path (`_tsdf_from_splats_dlnr.py`, `frgs mesh-dlnr`) and made its depth baseline scale with per-image rendered depth rather than overall scene scale, making meshing robust across capture types (e.g. orbit vs. robot navigation vs. multi-scale captures) (#86 - @fwilliams).
- Added a foundation-models module with a SAM2 wrapper for mask-assisted meshing (@fwilliams, #87 - @swahtz).
- Documented the TSDF/meshing algorithms in detail with attribution to the underlying papers (#90 - @fwilliams).

### Command-Line Tools & I/O

- Unified all Gaussian-splatting command-line utilities into a single pip-installable `frgs` CLI (download, reconstruct, convert, show-data, show, evaluate, mesh, mesh-dlnr, points, resume) (#97 - @fwilliams).
- Added C++ PLY saving and extra metadata fields in exported PLY files (#37 - @fwilliams).
- Added USDZ export from PLY and fixed its SH-coefficient data layout/reordering (#71, #143 - @swahtz).
- Added S3 upload/download utilities and tests, and moved the S3 module to its proper package location (#69, #81, #33 - @harrism, @fwilliams).
- Renamed the package from `fvdb_3dgs`/`fvdb_gs3d` to `fvdb_reality_capture` and cleaned up the public import API into clearly scoped submodules (#45, #126 - @fwilliams).
- Added scripts for extracting geo-referenced orthomosaics and geotagged video frames (@bbartlett-nv).

### Benchmarking

- Added an end-to-end Gaussian splatting benchmark and a comparative 3D Gaussian Splatting benchmark suite (@harrism).
- Added Nsight profiling scripts for comparing 3DGS performance (@harrism) and support for benchmarking across multiple configurations (#72 - @fwilliams).
- Updated the benchmark Docker setup (CPM cache, paths) and fixed Docker benchmark path issues (#73, #78 - @harrism, @fwilliams).
- Updated the comparison benchmark to track the latest fvdb-reality-capture API and repo layout (#34, #140 - @harrism).

### Isaac Sim Integration

- Added files/scripts for using fvdb-reality-capture data with Isaac Sim (#47 - @zlalena).

### Documentation

- Added a full Sphinx documentation site with GitHub Pages deployment, including numerous workflow iterations to get the docs build and custom-domain (CNAME) deployment working (#102, #105, #107-#111, #113-#120, #123, #150-#151 - @fwilliams).
- Added tutorials and notebooks, including a "reconstruct Gaussian splats" walkthrough notebook and an `SfmScene` tutorial (#96, #139 - @fwilliams).
- Wrote detailed documentation for `GaussianSplatOptimizer`, TSDF mesh extraction, transforms, and the `frgs` tools (#84, #90, #121, #125, #131 - @fwilliams).
- Rewrote the top-level and Gaussian-splatting READMEs, and aligned install instructions with fvdb-core (#127, #148, #152, #155 - @fwilliams, @harrism, @swahtz), including a fix pointing the Gaussian splatting README at the correct conda environment (@vinegh4).

### CI / DevOps / Packaging

- Added the OpenVDB license and fVDB's code-style GitHub Action (#1 - @swahtz), and a CODEOWNERS file (#32 - @harrism).
- Added a CI unit-test GitHub Actions workflow and switched tests to the `pull_request_target` trigger (#80, #83 - @harrism, @swahtz).
- Converted the project to a `pyproject.toml`-based package and dropped an unneeded dependency (#28 - @fwilliams).
- Added missing `requests` and editor dependencies (#154, #153 - @matthewdcong, @phapalova).
- Fixed CI build issues and bumped the release version to 0.3.0 (#100, #156 - @fwilliams).
