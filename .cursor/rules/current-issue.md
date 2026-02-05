# Current Task: Issue #230

> **ACTION REQUIRED**: When you receive a message referencing this file, immediately start by exploring the codebase to understand the issue, then create a detailed implementation plan.

## Comparative benchmark: GSplat config values are not passed to simple_trainer.py

**Repository:** openvdb/fvdb-reality-capture
**Issue:** https://github.com/openvdb/fvdb-reality-capture/issues/230
**Labels:** none
**Assignees:** unassigned

## Description

## Problem

Many config values in `opt_configs/gsplat_mcmc_default.yml` are **dead config** - they are defined but never passed to GSplat's `simple_trainer.py` CLI. This means:

1. The YAML appears to offer control over GSplat behavior but doesn't actually work
2. Benchmarks may not be running with intended configurations
3. Config mismatches between FVDB and GSplat go undetected

### Affected Parameters

In `gsplat_mcmc_default.yml`, the following are **never passed** to GSplat:

- `opacity_reg`, `scale_reg` - regularization (happens to match GSplat defaults by accident)
- `initial_opacity`, `initial_covariance_scale` - initialization parameters
- Learning rates (`means_lr`, `scales_lr`, etc.)
- Many other training config values

### Current State

`run_gsplat_training.py` only passes a subset of parameters via CLI:
- `--strategy.refine_start_iter`, `--strategy.refine_stop_iter`, `--strategy.refine_every`
- `--max_steps`, `--data_factor`, `--global_scale`
- `--eval_steps`

Everything else uses GSplat's hardcoded defaults.

## Proposed Solution

Add a **parameter mapping table** in `run_gsplat_training.py` that maps FVDB/benchmark config names to GSplat CLI flags:

```python
GSPLAT_PARAM_MAPPING = {
    # Config key -> GSplat CLI flag
    "opacity_reg": "--opacity_reg",
    "scale_reg": "--scale_reg",
    "initial_opacity": "--init_opa",
    "initial_covariance_scale": "--init_scale",
    "means_lr": "--means_lr",
    "log_scales_lr": "--scales_lr",
    # ... etc
}
```

This approach:
1. Makes parameter correspondence explicit and documented
2. Makes dead config obvious (no mapping = not passed)
3. Allows validation that all config values are being used
4. Enables parity testing between FVDB and GSplat with identical configs

## Related Finding

During investigation, we found significant **initialization differences** between GSplat MCMC defaults and FVDB defaults:

| Parameter | GSplat MCMC | FVDB |
|-----------|-------------|------|
| init_opacity | 0.5 | 0.1 |
| init_scale | 0.1 | 1.0 |

This may explain observed early PSNR differences in comparative benchmarks.

## Files Affected

- `tests/benchmarks/comparative/benchmark_utils/run_gsplat_training.py`
- `tests/benchmarks/comparative/opt_configs/gsplat_mcmc_default.yml`
- `tests/benchmarks/comparative/opt_configs/gsplat_default.yml`

## Your Task

You are assigned to implement this issue. Follow these steps:

1. **Explore** - Search the codebase to understand the relevant code and architecture
2. **Plan** - Create a detailed implementation plan with specific files and changes
3. **Implement** - Make the necessary code changes
4. **Test** - If existing tests don't cover new/changed functionality, add or update tests
5. **Verify** - Ensure all tests pass
6. **Format** - Run appropriate formatters and style checks

Start by exploring the codebase to understand where changes need to be made.

---

## Implementation Plan

### Problem Summary

The benchmark runner `run_gsplat_training.py` only passes a subset of config parameters to GSplat. Parameters like `opacity_reg`, `scale_reg`, `initial_opacity`, `initial_covariance_scale`, and learning rates defined in `gsplat_mcmc_default.yml` are never passed, causing GSplat to use its hardcoded defaults instead.

### Key Files

- `tests/benchmarks/comparative/benchmark_utils/run_gsplat_training.py` - Main file to modify
- `tests/benchmarks/comparative/opt_configs/gsplat_mcmc_default.yml` - MCMC config
- `tests/benchmarks/comparative/opt_configs/gsplat_default.yml` - Default config

### GSplat CLI Parameter Reference

From `~/github/gsplat/examples/simple_trainer.py`, the available CLI flags include:

**Initialization:**
- `--init_opa` (default: 0.1, MCMC mode: 0.5)
- `--init_scale` (default: 1.0, MCMC mode: 0.1)

**Regularization:**
- `--opacity_reg` (default: 0.0, MCMC mode: 0.01)
- `--scale_reg` (default: 0.0, MCMC mode: 0.01)

**Learning Rates:**
- `--means_lr`, `--scales_lr`, `--opacities_lr`, `--quats_lr`, `--sh0_lr`, `--shN_lr`

**Rendering:**
- `--near_plane`, `--far_plane`, `--antialiased`, `--random_bkgd`, `--ssim_lambda`, `--sh_degree`

**Strategy (both modes):**
- `--strategy.refine_start_iter`, `--strategy.refine_stop_iter`, `--strategy.refine_every`, `--strategy.verbose`

**Strategy (DefaultStrategy only):**
- `--strategy.reset_every`, `--strategy.prune_opa`, `--strategy.grow_grad2d`

**Strategy (MCMCStrategy only):**
- `--strategy.cap_max`, `--strategy.noise_lr`, `--strategy.min_opacity`

### Implementation Steps

#### 1. Add Parameter Mapping Dictionary

Add a mapping dictionary at module level in `run_gsplat_training.py`:

```python
# Maps FVDB/benchmark config names -> GSplat CLI flags
GSPLAT_PARAM_MAPPING = {
    # Initialization
    "initial_opacity": "--init_opa",
    "initial_covariance_scale": "--init_scale",

    # Regularization
    "opacity_reg": "--opacity_reg",
    "scale_reg": "--scale_reg",

    # Rendering parameters
    "near_plane": "--near_plane",
    "far_plane": "--far_plane",
    "antialias": "--antialiased",  # Note: different naming
    "random_bkgd": "--random_bkgd",
    "ssim_lambda": "--ssim_lambda",
    "sh_degree": "--sh_degree",

    # Learning rates (if exposed in YAML)
    "means_lr": "--means_lr",
    "log_scales_lr": "--scales_lr",
    "opacities_lr": "--opacities_lr",
    "quats_lr": "--quats_lr",
    "sh0_lr": "--sh0_lr",
    "shN_lr": "--shN_lr",
}
```

#### 2. Add Helper Function to Build CLI Args

Create a helper function that extracts config values and builds CLI arguments:

```python
def build_gsplat_cli_args(opt_config: dict) -> list[str]:
    """Build CLI arguments from opt_config using the parameter mapping."""
    args = []
    training_config = opt_config.get("training", {}).get("config", {})

    for config_key, cli_flag in GSPLAT_PARAM_MAPPING.items():
        if config_key in training_config:
            value = training_config[config_key]
            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    args.append(cli_flag)
            else:
                args.extend([cli_flag, str(value)])

    return args
```

#### 3. Integrate into Command Building

Modify the command building section (around line 138) to include mapped parameters:

```python
# After building the base command, add mapped parameters
mapped_args = build_gsplat_cli_args(opt_config)
cmd.extend(mapped_args)
```

#### 4. Add Logging for Transparency

Log which parameters are being passed for debugging and verification:

```python
logging.info(f"GSplat mapped parameters: {mapped_args}")
```

### Testing Strategy

1. Run the benchmark with logging to verify CLI command construction
2. Compare GSplat training output with and without the fix to verify parameters take effect
3. Check that initialization parameters (`init_opa`, `init_scale`) produce expected early training behavior

### Notes

- GSplat MCMC mode hardcodes `init_opa=0.5` and `init_scale=0.1`, while FVDB defaults use `0.1` and `1.0` respectively
- The current YAML configs set `initial_opacity: 0.1` which matches FVDB but differs from GSplat MCMC defaults
- This fix will ensure the YAML values override GSplat's mode-specific defaults
