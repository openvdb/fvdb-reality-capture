# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for build_gsplat_cli_args() and GSPLAT_PARAM_MAPPING.

Note: The SH interval epoch-to-step conversion (increase_sh_degree_every_epoch ->
sh_degree_interval) happens in run_gsplat_training() via extract_training_params,
not in build_gsplat_cli_args(), so it is not tested here.
"""
import importlib.util
import sys
from pathlib import Path


def _load_run_gsplat_module():
    """
    Load the run_gsplat_training module, temporarily registering the parent
    package so that relative imports resolve.

    Returns the module *and* the keys that were added to sys.path / sys.modules
    so callers can clean up.
    """
    benchmark_utils_dir = Path(__file__).resolve().parent / "comparative" / "benchmark_utils"
    module_path = benchmark_utils_dir / "run_gsplat_training.py"

    comparative_dir = str(benchmark_utils_dir.parent)
    added_to_path = comparative_dir not in sys.path
    if added_to_path:
        sys.path.insert(0, comparative_dir)

    added_modules: list[str] = []

    # Ensure the parent package exists so relative imports resolve.
    if "benchmark_utils" not in sys.modules:
        parent_init = benchmark_utils_dir / "__init__.py"
        parent_spec = importlib.util.spec_from_file_location(
            "benchmark_utils",
            str(parent_init),
            submodule_search_locations=[str(benchmark_utils_dir)],
        )
        if parent_spec and parent_spec.loader:
            parent_mod = importlib.util.module_from_spec(parent_spec)
            sys.modules["benchmark_utils"] = parent_mod
            added_modules.append("benchmark_utils")
            parent_spec.loader.exec_module(parent_mod)

    mod_name = "benchmark_utils.run_gsplat_training"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        module_path,
        submodule_search_locations=[],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    added_modules.append(mod_name)
    spec.loader.exec_module(module)

    return module, comparative_dir if added_to_path else None, added_modules


# Load once at module level and clean up sys.path / sys.modules so we don't
# leak synthetic packages into other test modules.
_mod, _added_path, _added_modules = _load_run_gsplat_module()
build_gsplat_cli_args = _mod.build_gsplat_cli_args
GSPLAT_PARAM_MAPPING = _mod.GSPLAT_PARAM_MAPPING

for _m in _added_modules:
    sys.modules.pop(_m, None)
if _added_path is not None:
    try:
        sys.path.remove(_added_path)
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_opt_config(**training_config_kwargs) -> dict:
    """Build a minimal opt_config dict wrapping the given training config keys."""
    return {"training": {"config": training_config_kwargs}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildGsplatCliArgs:
    """Tests for build_gsplat_cli_args()."""

    def test_empty_config_returns_empty_list(self):
        assert build_gsplat_cli_args({}) == []

    def test_empty_training_config_returns_empty_list(self):
        assert build_gsplat_cli_args({"training": {"config": {}}}) == []

    def test_scalar_float_parameter(self):
        opt = _make_opt_config(initial_opacity=0.5)
        args = build_gsplat_cli_args(opt)
        assert args == ["--init_opa", "0.5"]

    def test_scalar_int_parameter(self):
        opt = _make_opt_config(batch_size=4)
        args = build_gsplat_cli_args(opt)
        assert args == ["--batch_size", "4"]

    def test_scalar_int_sh_degree_parameter(self):
        opt = _make_opt_config(sh_degree=3)
        args = build_gsplat_cli_args(opt)
        assert args == ["--sh_degree", "3"]

    def test_boolean_true_emits_flag_only(self):
        opt = _make_opt_config(antialias=True)
        args = build_gsplat_cli_args(opt)
        assert args == ["--antialiased"]

    def test_boolean_false_emits_nothing(self):
        opt = _make_opt_config(antialias=False)
        args = build_gsplat_cli_args(opt)
        assert args == []

    def test_missing_keys_are_skipped(self):
        # Keys not in GSPLAT_PARAM_MAPPING are silently ignored.
        opt = _make_opt_config(not_a_real_key=42, another_fake=True)
        args = build_gsplat_cli_args(opt)
        assert args == []

    def test_multiple_parameters_all_emitted(self):
        opt = _make_opt_config(
            initial_opacity=0.5,
            initial_covariance_scale=0.1,
            opacity_regularization=0.01,
            scale_regularization=0.01,
        )
        args = build_gsplat_cli_args(opt)
        assert "--init_opa" in args
        assert "--init_scale" in args
        assert "--opacity_reg" in args
        assert "--scale_reg" in args
        # Each scalar flag should be followed by its value.
        assert args[args.index("--init_opa") + 1] == "0.5"
        assert args[args.index("--init_scale") + 1] == "0.1"
        assert args[args.index("--opacity_reg") + 1] == "0.01"
        assert args[args.index("--scale_reg") + 1] == "0.01"

    def test_mixed_boolean_and_scalar(self):
        opt = _make_opt_config(
            antialias=True,
            random_bkgd=False,
            near_plane=0.01,
        )
        args = build_gsplat_cli_args(opt)
        assert "--antialiased" in args
        assert "--random_bkgd" not in args
        assert "--near_plane" in args
        assert args[args.index("--near_plane") + 1] == "0.01"

    def test_fvdb_regularization_naming(self):
        """Ensure the FVDB naming convention (opacity_regularization) maps correctly."""
        opt = _make_opt_config(opacity_regularization=0.02, scale_regularization=0.03)
        args = build_gsplat_cli_args(opt)
        assert args[args.index("--opacity_reg") + 1] == "0.02"
        assert args[args.index("--scale_reg") + 1] == "0.03"

    def test_pose_optimization_boolean(self):
        opt_on = _make_opt_config(optimize_camera_poses=True)
        assert "--pose_opt" in build_gsplat_cli_args(opt_on)

        opt_off = _make_opt_config(optimize_camera_poses=False)
        assert "--pose_opt" not in build_gsplat_cli_args(opt_off)

    def test_learning_rates(self):
        opt = _make_opt_config(means_lr=1.6e-4, scales_lr=5e-3, opacities_lr=5e-2)
        args = build_gsplat_cli_args(opt)
        assert "--means_lr" in args
        assert "--scales_lr" in args
        assert "--opacities_lr" in args

    def test_all_mapping_keys_are_strings(self):
        """Sanity check: every key and value in GSPLAT_PARAM_MAPPING is a string."""
        for key, value in GSPLAT_PARAM_MAPPING.items():
            assert isinstance(key, str), f"Key {key!r} is not a string"
            assert isinstance(value, str), f"Value {value!r} is not a string"
            assert value.startswith("--"), f"CLI flag {value!r} doesn't start with '--'"
