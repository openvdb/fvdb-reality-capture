fVDB-Reality-Capture Version History
====================================

## Version 0.5.0 - July 1, 2026

*23 commits, 100+ files changed, 7 contributors.*

This release tracks fVDB 0.5.0. It adopts fVDB's new composable camera model and batched image API, adds dense depth supervision for Gaussian splat reconstruction, and moves documentation to a fully versioned Read the Docs site. It also switches to the upstream PyCOLMAP, upgrades the benchmark environment to PyTorch 2.11, and hardens CI, nightly benchmarks, issue triage, and repository governance across the fVDB repositories.

**Highlights:**
- Added composable camera-model support to Gaussian splat reconstruction, matching fVDB's new `CameraModel`/`ProjectionMethod` API.
- Added a `DepthMapAttribute` and optional dense depth supervision for reconstruction.
- Adapted TSDF integration to fVDB's new batched image API and synced the benchmark environment to PyTorch 2.11.
- Switched to the official PyCOLMAP repository.
- Migrated documentation to a versioned Read the Docs site.
- Hardened CI, nightly benchmarks, event-driven issue triage, and repository governance (CODEOWNERS), shared across the fVDB repos.

**Contributors:** @harrism, @matthewdcong, @swahtz, @fwilliams, @dylan-eustice, @phapalova, @zlalena

---

### Reconstruction & Gaussian Splatting

**New Features:**
- Added camera-model support to Gaussian splat reconstruction, adopting fVDB's new composable camera API (fVDB #518) that separates camera semantics from projection implementation (#253 - @fwilliams).
- Added a `DepthMapAttribute` and optional dense depth supervision to the reconstruction pipeline (#288 - @fwilliams).

**Bug Fixes:**
- Fixed a crash when `accumulated_gradient_step_counts` is `None` during Gaussian splat refinement (#281 - @harrism).
- Fixed performance regressions with pinhole camera models (#280 - @matthewdcong).

---

### Structure-from-Motion & Scene Handling

- Fixed incorrect camera matrices produced during scene normalization (#285 - @matthewdcong).
- Switched to the official PyCOLMAP repository for Structure-from-Motion (#293 - @matthewdcong).

---

### TSDF Integration

- Fixed TSDF integration to work with fVDB's new batched image API (#292 - @dylan-eustice).

---

### PyTorch & Dependency Compatibility

- Synced the benchmark environment to PyTorch 2.11 to match the fVDB-core build (#296 - @harrism).
- Fixed a PyCOLMAP version mismatch that was failing nightly tests (#297 - @matthewdcong).

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

- Added an event-driven issue-triage labels workflow and hardened its team-membership check (#274, #276 - @harrism).
- Split `CODEOWNERS` into two review tiers — general code reviewable by any maintainer, while governance, legal, and CI/CD files require an NVIDIA maintainer — and added the governance docs (`MAINTAINERS.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`). Kept identical across `fvdb-core`, `fvdb-reality-capture`, and `fvdb-examples` (#298 - @harrism).
- Fixed required status checks being skipped (and permanently blocking) on docs-only PRs, and corrected the change-detection gate so docs-only PRs skip cleanly while code and mixed PRs still run tests (#299, #300 - @harrism).

## Version 0.4.0 - March 14, 2026

*55 commits, 92 files changed, 7 contributors.*

This release adds Markov Chain Monte Carlo (MCMC) Gaussian splat optimization, integrates several foundation models (SAM1, SAM2, OpenCLIP) for mask and feature generation, introduces an extensible custom-attribute system for `SfmScene`, and ships the first automated publish/release pipeline (PyPI + S3) alongside nightly comparative benchmarks against gsplat.

**Highlights:**
- Added an MCMC Gaussian splat optimizer and sparse depth regularization for reconstruction.
- Integrated SAM1, SAM2, and OpenCLIP foundation models for segmentation/feature generation.
- Added an extensible custom-attribute system for `SfmScene`, plus SfM mask caching and COLMAP/point-index loading fixes.
- Added an automated publish workflow (PyPI + S3, GPU-validated) and a formal release process.
- Added nightly comparative benchmarks (fVDB vs. gsplat) with time-series metric plots.
- Added NVIDIA branding and initial `__version__` support.

**Contributors:** @harrism, @swahtz, @fwilliams, @matthewdcong, @diz-vara, @eh-dub, @NotMorven

---

### Gaussian Splatting & Optimization

- Added a Markov Chain Monte Carlo (MCMC) Gaussian splat optimizer (#214 - @fwilliams).
- Added sparse depth regularization (#188 - @fwilliams).
- Optimized the Gaussian splat optimizer by broadcasting appended parameters, and used Morton ordering to improve spatial locality (#248, #233 - @matthewdcong).

---

### Foundation Models

- Added SAM2 multi-scale mask generation, SAM1, and OpenCLIP foundation-model support (#239, #242, #231 - @swahtz).

---

### Structure-from-Motion & Data Loading

- Added an extensible custom-attribute system for `SfmScene` (#245 - @fwilliams).
- Fixed `SfmCameraMetadata` distortion downsampling, added SfM mask downsampling/caching, and handled empty `point_indices` (#228, #219, #220 - @swahtz, @diz-vara).
- Fixed bugs in COLMAP images text loading (#177 - @NotMorven).
- Fixed dataloader overhead, progress-bar accuracy, and cache paths when restarting from a checkpoint (#159, #166, #202 - @matthewdcong).

---

### Benchmarks & Nightly CI

- Added nightly comparative benchmarks (fVDB vs. gsplat) with throughput and time-series training-metric plots, per-commit comparison, and CI coverage for the 3DGS benchmark interfaces (#240, #168, #227, #235, #225, #226 - @harrism).
- Numerous nightly-benchmark reliability fixes (#164, #169, #173, #175, #232, #234, #162 - @harrism).

---

### Build, Packaging & Release

- Added an automated publish workflow (PyPI + S3) with GPU-validated builds on Rocky Linux 8 / manylinux, plus a formal release process (#258, #259, #260, #262, #263, #264, #265 - @harrism, @swahtz).
- Added a `__version__` attribute, a `matplotlib` benchmark dependency, and an aarch64 workaround; bumped versions to 0.3.1 and 0.4.0 (#256, #257, #190, #157, #250 - @harrism, @matthewdcong, @swahtz, @fwilliams).

---

### CI, Docs & Branding

- Fixed CI runner tokens, forked-repo nightly triggers, and fork-branch merging in unit tests (#212, #209, #229 - @swahtz).
- Added NVIDIA branding (#217 - @fwilliams).
- Documentation and tutorial fixes: viewer-reset demo notebook, analytics, `_Cpp` reference removal, notebook-checkpoint gitignore, and a tutorial typo fix (#158, #160, #161, #163, #174 - @swahtz, @fwilliams, @harrism, @eh-dub).

## Version 0.3.0 - October 24, 2025

*164 commits, 177 files changed, 9 contributors.*

First public release of fVDB-Reality-Capture, a reality-capture toolbox built on fVDB. It establishes the Structure-from-Motion scene representation, the Gaussian splat reconstruction pipeline and optimizer, TSDF mesh extraction from splats, foundation-model integrations, a viewer, CLI tooling, and documentation.

**Highlights:**
- `SfmScene` scene representation with COLMAP, E57, and simple-directory loaders and scene transforms.
- Gaussian splat reconstruction with a documented, configurable `GaussianSplatOptimizer` and TSDF mesh extraction from splats.
- SAM2 foundation-model integration and a new visualization API.
- USDZ export, PLY metadata, and Isaac Sim support files.
- Benchmarking harness, CI unit tests, and a documentation site with tutorials.

**Contributors:** @fwilliams, @swahtz, @harrism, @matthewdcong, @zlalena, @phapalova, @blackencino, @bbartlett-nv

---

### Scene Representation & Data Loading

- Added the `SfmScene` representation with unit-tested transforms, COLMAP / E57 / simple-directory loaders, and a `TransformScene` transform (#27, #39, #41, #82, #86, #142 - @fwilliams, @swahtz).
- Added a projection-type enum used across the reconstruction and viewer, and saved extra metadata in exported PLYs (#70, #77, #37 - @matthewdcong, @fwilliams).
- Fixed COLMAP text-loading and batch-optimization bugs (#135 - @fwilliams).

---

### Reconstruction & Optimization

- Refactored and documented `GaussianSplatOptimizer` and reworked its parameters (#84, #66 - @fwilliams).
- Optimized L1/SSIM loss interpolation and fixed a loss-computation performance regression (#85, #137 - @matthewdcong).
- Added TSDF mesh extraction from splats with meshing parameters and documentation (#90, #95 - @fwilliams).

---

### Foundation Models, Viewer & Export

- Added SAM2 foundation-model support (#87 - @swahtz).
- Added a new visualization API and helpers to filter `GaussianSplat3d` results (#136, #98 - @fwilliams, @swahtz).
- Added USDZ export (`ply_to_usdz`) and Isaac Sim support files (#71, #143, #47 - @swahtz, @zlalena).

---

### Benchmarks, CI & Docs

- Added a benchmarking harness (Docker/CPM cache, multiple configs) and a comparison benchmark (#72, #73, #34, #140 - @fwilliams, @harrism).
- Added the CI unit-test workflow (#80, #83 - @harrism, @swahtz).
- Established the documentation site with SfmScene/transforms/tools tutorials and install instructions, renamed the package `fvdb_3dgs` → `fvdb_reality_capture`, and set up `pyproject.toml` dependencies (#139, #45, #46, #28, #152, #153, #154 - @fwilliams, @swahtz, @matthewdcong, @phapalova).
