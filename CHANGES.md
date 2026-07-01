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
