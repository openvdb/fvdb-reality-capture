# fVDB Reality Capture Version History

## Unreleased

*21 commits, 60+ files changed, 7 contributors so far.*

Work since v0.4.0 is tracking towards the next release. It focuses on catching up to breaking changes in `fvdb-core`'s camera and batched-image APIs, adding dense depth supervision to Gaussian splat training, and hardening the release/CI pipeline.

**Highlights:**
- Reworked Gaussian splat reconstruction on top of `fvdb-core`'s new camera/render API (`fvdb-core#518`), cleaning up how reality-capture represents cameras, distortion, and render-time behavior for world-space rendering.
- Added a `DepthMapAttribute` for per-image depth rasters and wired optional dense depth supervision into `GaussianSplatReconstruction`.
- Fixed a scene-normalization bug that inverted camera matrices, and a TSDF meshing crash caused by `fvdb-core`'s move to a batched depth-image API.
- Decoupled PyPI and S3 publishing so both run on every release, and switched COLMAP dependency management to the official PyCOLMAP repository.
- Split `CODEOWNERS` into two review tiers, requiring NVIDIA sign-off on governance and CI/CD infrastructure files, and fixed the CI change-detection gate for docs-only PRs.

**Contributors:** @dylan-eustice, @fwilliams, @swahtz, @harrism, @matthewdcong, @phapalova, @zlalena

---

### Gaussian Splatting & Reconstruction

- Reworked Gaussian splat reconstruction to support world-space rendering on top of `fvdb-core`'s new camera/render API, cleaning up how reality-capture represents cameras, distortion models, and render-time camera behavior (#253 - @fwilliams).
- Fixed performance regressions with pinhole camera models introduced by the camera model rework (#280 - @fwilliams).
- Fixed a crash when `accumulated_gradient_step_counts` is `None` during Gaussian splat refinement (#281 - @fwilliams).
- Added `DepthMapAttribute`, a per-image depth raster attribute with scale-aware (metric vs. relative) semantics, and wired optional dense depth supervision into `GaussianSplatReconstruction` (#288 - @fwilliams).

---

### SfM / Scene Processing

- Fixed scene normalization storing a reference to `camera_to_world_matrices` instead of `world_to_camera_matrices`, which inverted the transform passed to similarity normalization and produced an incorrect scene scale (#285 - @fwilliams).

---

### Mesh Reconstruction (TSDF Fusion)

- Fixed a crash in `mesh_from_splats`/`tsdf_from_splats`/`tsdf_from_splats_dlnr` caused by `fvdb-core`'s move to a batched depth-image API, realigning reality-capture with `fvdb-core` main (#292 - @fwilliams).

---

### Build & Dependencies

- Switched from a fork to the official PyCOLMAP repository for the COLMAP dependency (#293 - @fwilliams).
- Fixed a PyCOLMAP version mismatch that was failing nightly tests (#297 - @fwilliams).
- Fixed the fVDB installation instructions by removing a stale `editor_force` flag (#290 - @swahtz).

---

### Benchmarking

- Synced the benchmark environment to PyTorch 2.11 to match the `fvdb-core` build (#296 - @harrism).
- Updated the nightly benchmark to PyTorch 2.10 and CUDA 13.0 (#272 - @harrism).
- Fixed a `JSONDecodeError` in nightly benchmark artifact downloads (#278 - @harrism).

---

### CI / Release Infrastructure

- Decoupled PyPI and S3 publish targets so both run on release: releases now upload sdist and wheel to production PyPI and additionally upload the wheel to the S3 production index, with independent routing flags replacing the old mutually-exclusive `s3` flag (#267 - @harrism).
- Added an event-driven issue-triage labels workflow, then hardened its membership check (#274, #276 - @harrism).
- Fixed the CI change-detection gate so it correctly skips required checks on docs-only PRs (#299, #300 - @swahtz).
- Split `CODEOWNERS` into two review tiers — general code review vs. required NVIDIA sign-off on governance, legal, and CI/CD infrastructure files (`.github/`, `LICENSE`, `MAINTAINERS.md`, `SECURITY.md`, etc.) — and added accompanying governance docs, kept consistent with `fvdb-core` and `fvdb-examples` (#298 - @harrism).

---

### Documentation

- Added a COLMAP tutorial notebook (#268 - @zlalena).
- Added Read the Docs configuration (#284 - @swahtz).
- Fixed an outdated `fvdb-core` version reference in the installation docs (#275 - @dylan-eustice).

## Version 0.4.0 - March 2026

*52 commits, 100+ files changed, 7 contributors.*

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
- Added a Markov Chain Monte Carlo (MCMC) optimizer for Gaussian splat radiance field reconstruction, along with a refactored optimizer registration and deserialization system for extensibility (#214 - @fwilliams).
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

*163 commits, ~150 files changed, 9 contributors.*

This is the initial public release of fVDB Reality Capture, a toolbox built on top of fVDB for turning multi-view captures into 3D Gaussian splat reconstructions, meshes, and other derived assets. The release establishes the core pipeline end-to-end: loading COLMAP/e57/simple-directory SfM captures into a common `SfmScene` representation, training and refining 3D Gaussian splats (including multi-GPU support), extracting meshes via TSDF fusion, an interactive Viser-based viewer, a `frgs` command-line tool, benchmarking utilities, foundation-model-assisted masking, and a full documentation and CI setup.

**Highlights:**
- New `SfmScene`/`ColmapDataset` data model for loading and transforming COLMAP, e57, and plain-directory captures, with a composable torchvision-style transform pipeline and an on-disk caching layer.
- A rewritten Gaussian splatting stack centered on a new C++ `GaussianSplat3d` class, a documented and optimized `GaussianSplatOptimizer` (refinement, pose optimization, MCMC-style densification controls), checkpointing, and multi-GPU training support.
- Mesh reconstruction from trained splats via TSDF fusion (`splat2mesh`), including DLNR-based stereo depth estimation and SAM2-based foundation-model masking for cleaner reconstructions.
- A Python Viser viewer with a registrable custom-renderer API, camera image overlays, and up-axis controls, plus PLY/USDZ export and a unified `frgs` CLI (download, reconstruct, convert, show-data, show, evaluate, mesh).
- `GARfVDB`, a GARField-style radiance-field segmentation model built on Gaussian splats, with a contrastive-loss training pipeline, SAM2 mask generation, and a paper-faithful evaluation pass.
- A benchmarking suite (end-to-end benchmark, Nsight profiling scripts, Dockerized CI runs) plus a full Sphinx/GitHub Pages documentation site with tutorials and notebooks.

**Contributors:** @fwilliams, @swahtz, @harrism, @matthewdcong, @bbartlett-nv, @zlalena, @blackencino, @vinegh4, @phapalova

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
- Added a new C++ `GaussianSplat3d` class and switched the Python training/inference path to the new C++ Gaussian Splat API (@fwilliams).
- Added multi-GPU Gaussian splat training (@bbartlett-nv) and multi-GPU camera pose optimization (@zlalena).
- Refactored pose optimization and rewrote `GaussianSplatOptimizer` with documented, configurable refinement (splitting/duplication/deletion), percentile-based gradient pruning thresholds, and deferred pose optimization until after refinement completes (#66, #84 - @fwilliams).
- Added `__getitem__`/`__setitem__`, chunking, and concatenation support to `GaussianSplat3d` (@fwilliams).
- Added functions to filter `GaussianSplat3d` results by mean, opacity, or scale (#98 - @swahtz).
- Rolled a self-contained PSNR and LPIPS implementation to remove the `torchmetrics` dependency, then switched to SSIM/PSNR from `fvdb.utils.metrics` (@fwilliams; #29 - @harrism).
- Renamed `SceneOptimizationRunner` to `GaussianSplatReconstruction`, reworked checkpointing to serialize `SfmScene`s directly, and promoted USDZ export to its own tool as part of a broader API/notebook overhaul (#96 - @fwilliams).
- Added a `frgs evaluate` script and an `frgs mesh` extraction path built on the new checkpoint API (@fwilliams).
- Cleaned up the underlying fVDB `GridBatch`/`Grid` Python interface (immutable construction via `from_*` classmethods, tightened tensor/`JaggedTensor` conversion rules) that the reconstruction pipeline builds on (@blackencino).

**Optimizations:**
- Fixed a performance regression in the Gaussian splat loss computation (#137 - @matthewdcong).
- Fused the L1/SSIM loss interpolation into a single `torch.lerp` call to cut memory bandwidth and temporary tensors (#85 - @matthewdcong).
- Deferred `loss.item()` synchronization to a single point per iteration, improving multi-GPU throughput by ~10% and single-GPU by ~5-7% (#144 - @matthewdcong).
- Reduced extra parameter copies during Gaussian refinement (duplication/splitting/deletion) to cut memory usage (#84 - @fwilliams).

**Bug Fixes:**
- Fixed model checkpointing and restarting training from a checkpoint, including a `weights_only=False` load failure under newer PyTorch (@fwilliams; #147 - @matthewdcong).
- Fixed a bug in Gaussian splat refinement shared with the gsplat/INRIA reference implementations that improves reconstruction quality (#84 - @fwilliams).
- Fixed I/O and small numerical bugs found while running on Puerto Rico scene data and other real captures (@fwilliams; #134 - @fwilliams).
- Fixed the training metric label in TensorBoard logging (@matthewdcong) and fixed `tensorboard add_images` (#138 - @swahtz).

### Mesh Reconstruction (TSDF Fusion)

- Added mesh reconstruction from trained Gaussian splats via TSDF fusion, along with the `splat2mesh` script (@fwilliams).
- Added support for per-image weighting in TSDF fusion (@fwilliams) and extra meshing parameters for thresholding low-opacity background pixels and downsampling large images (#95 - @fwilliams).
- Made the DLNR stereo-depth baseline scale with per-image rendered depth rather than overall scene scale, making meshing robust across capture types (e.g. orbit vs. robot navigation vs. multi-scale captures) (#86 - @fwilliams).
- Added a foundation-models module and GS2Mesh integration for mask-assisted meshing (@fwilliams), followed by a SAM2 foundation model for mask prediction (#87 - @swahtz).
- Documented the TSDF/meshing algorithms in detail with attribution to the underlying papers (#90 - @fwilliams).

### Visualization & Viewer

- Added a Python Viewer API with support for registering custom renderers (@fwilliams).
- Added camera image overlays and up-axis controls to the Viser-based 3DGS viewer (@swahtz).
- Fixed the viewer to use the `ProjectionType` enum consistently (#70, #77 - @matthewdcong).
- Added a new visualization API (#136 - @fwilliams) and fixed the visualization server's IP address handling (#145 - @swahtz).
- Added attribution for the turbo colormap used in visualizations (@fwilliams).

### Command-Line Tools & I/O

- Unified all Gaussian-splatting command-line utilities into a single pip-installable `frgs` CLI (download, reconstruct, convert, show-data, show, evaluate, mesh) (#97 - @fwilliams).
- Added C++ PLY saving and extra metadata fields in exported PLY files (#37 - @fwilliams).
- Added USDZ export from PLY (`ply_to_usdz`) and fixed its SH-coefficient data layout/reordering (#71, #143 - @swahtz).
- Added S3 upload/download utilities and tests, and moved the S3 module to its proper package location (#69, #81, #33 - @harrism, @fwilliams).
- Renamed the package from `fvdb_3dgs`/`fvdb_gs3d` to `fvdb_reality_capture` and cleaned up the public import API into clearly scoped submodules (#45, #126 - @fwilliams).

### Multi-GPU Training

- Added multi-GPU Gaussian splat training support (@bbartlett-nv) and multi-GPU camera pose optimization (@zlalena).
- Fixed stride handling for splat filtering utilities after the multi-GPU refactor (@bbartlett-nv).
- Improved multi-GPU training throughput by deferring loss-logging synchronization (#144 - @matthewdcong).

### Benchmarking

- Added an end-to-end Gaussian splatting benchmark and a comparative 3D Gaussian Splatting benchmark suite (@harrism).
- Added Nsight profiling scripts for comparing 3DGS performance (@harrism) and support for benchmarking across multiple configurations (#72 - @fwilliams).
- Updated the benchmark Docker setup (CPM cache, paths) and fixed Docker benchmark path issues (#73, #78 - @harrism, @fwilliams).
- Updated the comparison benchmark to track the latest fvdb-reality-capture API and repo layout (#34, #140 - @harrism).

### Scene Segmentation (GARfVDB)

- Implemented GARField-style radiance-field segmentation on top of Gaussian splat scenes (`GARfVDB`), including a configurable grid-based feature model, a contrastive-loss training pipeline, SAM2-based mask generation, and a core `render_top_contributing_gaussian_ids` fVDB kernel for per-pixel Gaussian attribution (@swahtz).
- Added a GARField evaluation phase: conversion from Nerfstudio scenes to COLMAP scenes, reuse of GARField-generated masks for training, and the "completeness" experiment from the GARField paper, plus checkpoint and PCA robustness fixes (@swahtz).

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
