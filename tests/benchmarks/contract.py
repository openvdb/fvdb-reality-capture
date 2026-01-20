# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Benchmark contract checks to prevent nightly benchmark breakage.

This module defines a small, explicit compatibility contract used by
`tests/benchmarks/test_3dgs.py` and `tests/benchmarks/generate_benchmark_checkpoints.py`.
"""

from __future__ import annotations

from typing import Any

import yaml

from fvdb_reality_capture.radiance_fields import (
    GaussianSplatOptimizerConfig,
    GaussianSplatOptimizerMCMCConfig,
    GaussianSplatReconstruction,
    GaussianSplatReconstructionConfig,
)

CONTRACT_VERSION = 1
"""
Semantic version of the benchmark contract.

Increment this when the expected checkpoint/config schemas change in a
backwards-incompatible way for the benchmark suite.
"""

TOP_LEVEL_CHECKPOINT_KEYS = {
    "magic",
    "version",
    "step",
    "config",
    "sfm_scene",
    "model",
    "optimizer",
    "train_indices",
    "val_indices",
    "num_training_poses",
    "pose_adjust_model",
    "pose_adjust_optimizer",
    "pose_adjust_scheduler",
}
"""
Required top-level keys in benchmark checkpoints.
"""

RECONSTRUCTION_CONFIG_KEYS = {
    "seed",
    "max_epochs",
    "max_steps",
    "eval_at_percent",
    "save_at_percent",
    "batch_size",
    "crops_per_image",
    "sh_degree",
    "increase_sh_degree_every_epoch",
    "initial_opacity",
    "initial_covariance_scale",
    "ssim_lambda",
    "lpips_net",
    "sparse_depth_reg",
    "random_bkgd",
    "refine_start_epoch",
    "refine_stop_epoch",
    "refine_every_epoch",
    "ignore_masks",
    "remove_gaussians_outside_scene_bbox",
    "optimize_camera_poses",
    "pose_opt_lr",
    "pose_opt_reg",
    "pose_opt_lr_decay",
    "pose_opt_start_epoch",
    "pose_opt_stop_epoch",
    "pose_opt_init_std",
    "near_plane",
    "far_plane",
    "min_radius_2d",
    "eps_2d",
    "antialias",
    "tile_size",
}
"""
Allowed field names for `GaussianSplatReconstructionConfig` in benchmark configs
and checkpoint `config` payloads. Defaults mean values may be omitted elsewhere,
but the contract constrains the *set* of valid keys.
"""

OPTIMIZER_CONFIG_KEYS = {
    "max_gaussians",
    "insertion_grad_2d_threshold_mode",
    "deletion_opacity_threshold",
    "deletion_scale_3d_threshold",
    "deletion_scale_2d_threshold",
    "insertion_grad_2d_threshold",
    "insertion_scale_3d_threshold",
    "insertion_scale_2d_threshold",
    "opacity_updates_use_revised_formulation",
    "insertion_split_factor",
    "insertion_duplication_factor",
    "reset_opacities_every_n_refinements",
    "use_scales_for_deletion_after_n_refinements",
    "use_screen_space_scales_for_refinement_until",
    "spatial_scale_mode",
    "spatial_scale_multiplier",
    "means_lr",
    "log_scales_lr",
    "quats_lr",
    "logit_opacities_lr",
    "sh0_lr",
    "shN_lr",
}
"""
Allowed fields in `GaussianSplatOptimizerConfig` in benchmark configs and checkpoint `optimizer.config` payloads.
"""

MCMC_OPTIMIZER_EXTRA_KEYS = {
    "noise_lr",
    "insertion_rate",
    "binomial_coeffs_n_max",
    "opacity_regularization",
    "scale_regularization",
}
"""
Extra fields allowed only when using `GaussianSplatOptimizerMCMCConfig`.
"""


def load_benchmark_yaml(path: str) -> dict[str, Any]:
    """
    Load a YAML file as a dict.

    Used by benchmark config validation and comparison benchmark tests.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _raise_contract_error(message: str, *, details: dict[str, Any] | None = None) -> None:
    """
    Raise a ValueError with optional structured details appended.
    """
    if details:
        detail_lines = [f"{k}={v!r}" for k, v in details.items()]
        message = message + " (" + ", ".join(detail_lines) + ")"
    raise ValueError(message)


def validate_checkpoint_contract(state: dict[str, Any]) -> None:
    """
    Validate the on-disk checkpoint schema used by benchmark tests.

    A valid checkpoint must include:
    - Top-level keys in `TOP_LEVEL_CHECKPOINT_KEYS`
    - `magic == "GaussianSplattingCheckpoint"`
    - `version == GaussianSplatReconstruction.version`
    - `config` keys constrained by `RECONSTRUCTION_CONFIG_KEYS`
    - `optimizer.config` keys constrained by `OPTIMIZER_CONFIG_KEYS` (plus
      `MCMC_OPTIMIZER_EXTRA_KEYS` when using the MCMC optimizer)

    This is intentionally strict: missing or extra keys are treated as errors.
    """
    if not isinstance(state, dict):
        _raise_contract_error("Checkpoint state must be a dict", details={"type": type(state).__name__})

    missing = TOP_LEVEL_CHECKPOINT_KEYS - set(state.keys())
    if missing:
        _raise_contract_error("Checkpoint missing required keys", details={"missing": sorted(missing)})

    if state.get("magic") != "GaussianSplattingCheckpoint":
        _raise_contract_error(
            "Checkpoint magic mismatch",
            details={"magic": state.get("magic"), "expected": "GaussianSplattingCheckpoint"},
        )
    if state.get("version") != GaussianSplatReconstruction.version:
        _raise_contract_error(
            "Checkpoint version mismatch",
            details={"version": state.get("version"), "expected": GaussianSplatReconstruction.version},
        )

    cfg = state.get("config")
    if not isinstance(cfg, dict):
        _raise_contract_error("Checkpoint config must be a dict", details={"type": type(cfg).__name__})

    cfg_keys = set(cfg.keys())
    missing_cfg = RECONSTRUCTION_CONFIG_KEYS - cfg_keys
    extra_cfg = cfg_keys - RECONSTRUCTION_CONFIG_KEYS
    if missing_cfg or extra_cfg:
        _raise_contract_error(
            "Checkpoint config key mismatch",
            details={"missing": sorted(missing_cfg), "extra": sorted(extra_cfg)},
        )

    optimizer = state.get("optimizer")
    if not isinstance(optimizer, dict):
        _raise_contract_error("Checkpoint optimizer must be a dict", details={"type": type(optimizer).__name__})
    opt_cfg = optimizer.get("config")
    if not isinstance(opt_cfg, dict):
        _raise_contract_error("Checkpoint optimizer config must be a dict", details={"type": type(opt_cfg).__name__})

    opt_cfg_keys = set(opt_cfg.keys())
    opt_base = OPTIMIZER_CONFIG_KEYS
    opt_mcmc = OPTIMIZER_CONFIG_KEYS | MCMC_OPTIMIZER_EXTRA_KEYS
    if opt_cfg_keys != opt_base and opt_cfg_keys != opt_mcmc:
        _raise_contract_error(
            "Checkpoint optimizer config key mismatch",
            details={
                "keys": sorted(opt_cfg_keys),
                "expected": "GaussianSplatOptimizerConfig or GaussianSplatOptimizerMCMCConfig",
            },
        )


def validate_benchmark_yaml(config: dict[str, Any], *, require_run_paths: bool = True) -> None:
    """
    Validate the benchmark YAML schema used by `generate_benchmark_checkpoints.py`.

    A valid benchmark YAML must include:
    - `paths.data_base`
    - `datasets[]` entries with `name`, `path` and (when `require_run_paths` is True)
      `run_directory`, `checkpoint_paths`
    - `optimization_config.splat_optimizer` in {"GaussianSplatOptimizer","GaussianSplatOptimizerMCMC"}
    - `optimization_config.reconstruction_config` keys constrained by `RECONSTRUCTION_CONFIG_KEYS`
    - `optimization_config.optimization_config` keys constrained by `OPTIMIZER_CONFIG_KEYS` (+ MCMC extras)
    - `optimization_config.training_arguments` with `image_downsample_factor`, `use_every_n_as_val`, `device`
    """
    if not isinstance(config, dict):
        _raise_contract_error("Benchmark config must be a dict", details={"type": type(config).__name__})

    paths = config.get("paths", {})
    if not isinstance(paths, dict) or "data_base" not in paths:
        _raise_contract_error("Benchmark config missing paths.data_base")

    datasets = config.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        _raise_contract_error("Benchmark config datasets must be a non-empty list")
    for d in datasets:
        if not isinstance(d, dict):
            _raise_contract_error("Each dataset entry must be a dict")
        required_keys = ["name", "path"]
        if require_run_paths:
            required_keys.extend(["run_directory", "checkpoint_paths"])
        for key in required_keys:
            if key not in d:
                _raise_contract_error("Dataset missing required key", details={"missing": key})
        if require_run_paths and not isinstance(d.get("checkpoint_paths"), list):
            _raise_contract_error("Dataset checkpoint_paths must be a list")

    opt_section = config.get("optimization_config", {})
    if not isinstance(opt_section, dict):
        _raise_contract_error("optimization_config must be a dict")
    splat_optimizer = opt_section.get("splat_optimizer")
    if splat_optimizer not in ("GaussianSplatOptimizer", "GaussianSplatOptimizerMCMC"):
        _raise_contract_error(
            "optimization_config.splat_optimizer must be GaussianSplatOptimizer or GaussianSplatOptimizerMCMC",
            details={"splat_optimizer": splat_optimizer},
        )

    recon_cfg = opt_section.get("reconstruction_config", {})
    if not isinstance(recon_cfg, dict):
        _raise_contract_error("optimization_config.reconstruction_config must be a dict")
    recon_extra = set(recon_cfg.keys()) - RECONSTRUCTION_CONFIG_KEYS
    if recon_extra:
        _raise_contract_error("Unknown reconstruction_config keys", details={"extra": sorted(recon_extra)})

    opt_cfg = opt_section.get("optimization_config", {})
    if not isinstance(opt_cfg, dict):
        _raise_contract_error("optimization_config.optimization_config must be a dict")
    allowed_opt_keys = OPTIMIZER_CONFIG_KEYS | (
        MCMC_OPTIMIZER_EXTRA_KEYS if splat_optimizer == "GaussianSplatOptimizerMCMC" else set()
    )
    opt_extra = set(opt_cfg.keys()) - allowed_opt_keys
    if opt_extra:
        _raise_contract_error("Unknown optimization_config keys", details={"extra": sorted(opt_extra)})

    training_args = opt_section.get("training_arguments", {})
    if not isinstance(training_args, dict):
        _raise_contract_error("optimization_config.training_arguments must be a dict")
    for key in ("image_downsample_factor", "use_every_n_as_val", "device"):
        if key not in training_args:
            _raise_contract_error("training_arguments missing required key", details={"missing": key})


def validate_comparative_benchmark_yaml(config: dict[str, Any]) -> None:
    """
    Validate the comparative benchmark matrix YAML schema.

    A valid matrix YAML must include:
    - top-level `name`
    - `paths` with `fvdb_base`, `gsplat_base`, `data_base`
    - `datasets[]` entries with `name`, `path`
    - `opt_configs` mapping with entries containing `path`
    - `runs[]` entries with `dataset` and `opt_config`
    """
    if not isinstance(config, dict):
        _raise_contract_error("Comparative benchmark config must be a dict", details={"type": type(config).__name__})

    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        _raise_contract_error("Comparative benchmark config missing paths section")
    for key in ("fvdb_base", "gsplat_base", "data_base"):
        if key not in paths:
            _raise_contract_error("Comparative benchmark config missing paths key", details={"missing": key})

    datasets = config.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        _raise_contract_error("Comparative benchmark datasets must be a non-empty list")
    for d in datasets:
        if not isinstance(d, dict):
            _raise_contract_error("Each dataset entry must be a dict")
        for key in ("name", "path"):
            if key not in d:
                _raise_contract_error("Dataset missing required key", details={"missing": key})


def validate_comparative_opt_config(config: dict[str, Any]) -> None:
    """
    Validate a single comparative opt-config file.

    For FVDB configs, the schema is strict against current dataclasses.
    For GSplat configs, we keep this permissive (the config is passed through).

    A valid FVDB opt-config must include:
    - `framework: fvdb`
    - `name`
    - `reconstruction_config` keys constrained by `RECONSTRUCTION_CONFIG_KEYS`
    - `optimization_config` keys constrained by `OPTIMIZER_CONFIG_KEYS` (+ MCMC extras)
    - `training_arguments` with `image_downsample_factor`, `use_every_n_as_val`, `device`

    A valid GSplat opt-config must include:
    - `framework: gsplat`
    - `name`
    - either `training` or `preset`
    """
    if not isinstance(config, dict):
        _raise_contract_error("Opt config must be a dict", details={"type": type(config).__name__})
    framework = config.get("framework")
    if framework not in ("fvdb", "gsplat"):
        _raise_contract_error("Opt config framework must be fvdb or gsplat", details={"framework": framework})
    if "name" not in config:
        _raise_contract_error("Opt config missing name")

    if framework == "fvdb":
        recon_cfg = config.get("reconstruction_config", {})
        if not isinstance(recon_cfg, dict):
            _raise_contract_error("FVDB opt config reconstruction_config must be a dict")
        recon_extra = set(recon_cfg.keys()) - RECONSTRUCTION_CONFIG_KEYS
        if recon_extra:
            _raise_contract_error("Unknown FVDB reconstruction_config keys", details={"extra": sorted(recon_extra)})

        opt_cfg = config.get("optimization_config", {})
        if not isinstance(opt_cfg, dict):
            _raise_contract_error("FVDB opt config optimization_config must be a dict")
        splat_optimizer = config.get("splat_optimizer", "GaussianSplatOptimizer")
        allowed_opt_keys = OPTIMIZER_CONFIG_KEYS | (
            MCMC_OPTIMIZER_EXTRA_KEYS if splat_optimizer == "GaussianSplatOptimizerMCMC" else set()
        )
        opt_extra = set(opt_cfg.keys()) - allowed_opt_keys
        if opt_extra:
            _raise_contract_error("Unknown FVDB optimization_config keys", details={"extra": sorted(opt_extra)})

        training_args = config.get("training_arguments", {})
        if not isinstance(training_args, dict):
            _raise_contract_error("FVDB opt config training_arguments must be a dict")
        for key in ("image_downsample_factor", "use_every_n_as_val", "device"):
            if key not in training_args:
                _raise_contract_error("training_arguments missing required key", details={"missing": key})

    else:
        # gsplat: keep this permissive since the config is passed through to the external trainer.
        if "training" not in config and "preset" not in config:
            _raise_contract_error("GSplat opt config must define training or preset")


def _assert_contract_matches_dataclasses() -> None:
    """
    Internal check to keep the explicit contract in sync with the dataclasses.

    Used by tests to ensure the contract stays aligned with code changes.
    """
    recon_keys = set(vars(GaussianSplatReconstructionConfig()).keys())
    if recon_keys != RECONSTRUCTION_CONFIG_KEYS:
        _raise_contract_error(
            "RECONSTRUCTION_CONFIG_KEYS out of sync with GaussianSplatReconstructionConfig",
            details={
                "missing": sorted(recon_keys - RECONSTRUCTION_CONFIG_KEYS),
                "extra": sorted(RECONSTRUCTION_CONFIG_KEYS - recon_keys),
            },
        )
    opt_keys = set(vars(GaussianSplatOptimizerConfig()).keys())
    if opt_keys != OPTIMIZER_CONFIG_KEYS:
        _raise_contract_error(
            "OPTIMIZER_CONFIG_KEYS out of sync with GaussianSplatOptimizerConfig",
            details={
                "missing": sorted(opt_keys - OPTIMIZER_CONFIG_KEYS),
                "extra": sorted(OPTIMIZER_CONFIG_KEYS - opt_keys),
            },
        )
    mcmc_keys = set(vars(GaussianSplatOptimizerMCMCConfig()).keys())
    expected_mcmc = OPTIMIZER_CONFIG_KEYS | MCMC_OPTIMIZER_EXTRA_KEYS
    if mcmc_keys != expected_mcmc:
        _raise_contract_error(
            "MCMC_OPTIMIZER_EXTRA_KEYS out of sync with GaussianSplatOptimizerMCMCConfig",
            details={"missing": sorted(mcmc_keys - expected_mcmc), "extra": sorted(expected_mcmc - mcmc_keys)},
        )
