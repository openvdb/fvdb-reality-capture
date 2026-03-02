# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


def _load_comparison_benchmark_module():
    comparative_dir = Path(__file__).resolve().parent / "comparative"
    if str(comparative_dir) not in sys.path:
        sys.path.insert(0, str(comparative_dir))
    module_path = comparative_dir / "comparison_benchmark.py"
    spec = importlib.util.spec_from_file_location("comparison_benchmark", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, comparative_dir


def _get_build_gsplat_cli_args():
    _load_comparison_benchmark_module()  # ensures sys.path includes comparative dir
    from benchmark_utils.run_gsplat_training import build_gsplat_cli_args

    return build_gsplat_cli_args


def _write_minimal_report(report_path: Path, include_time_series: bool = False) -> None:
    metrics = {"psnr": 1.0, "ssim": 1.0, "final_gaussian_count": 1}

    if include_time_series:
        # Add time-series data for training curve plots
        metrics.update(
            {
                "loss_steps": [0, 1000, 2000, 3000],
                "loss_values": [0.1, 0.08, 0.06, 0.05],
                "psnr_steps": [0, 1000, 2000, 3000],
                "psnr_values": [20.0, 22.5, 24.0, 25.0],
                "ssim_steps": [0, 1000, 2000, 3000],
                "ssim_values": [0.7, 0.75, 0.8, 0.85],
                "gaussian_count_steps": [0, 1000, 2000, 3000],
                "gaussian_count_values": [100000, 150000, 180000, 200000],
                "iterations_per_s_steps": [0, 1000, 2000, 3000],
                "iterations_per_s_values": [50.0, 48.0, 47.0, 46.0],
            }
        )

    report = {
        "fvdb_default": {
            "config_name": "fvdb_default",
            "scene": "garden",
            "timestamp": "1970-01-01 00:00:00",
            "training": {
                "success": True,
                "total_time": 1.0,
                "training_time": 1.0,
                "exit_code": 0,
                "metrics": metrics,
                "result_dir": "results/benchmark",
            },
            "success": True,
            "total_time": 1.0,
            "training_time": 1.0,
            "final_loss": 0.0,
        }
    }
    report_path.write_text(json.dumps(report))


def _write_minimal_matrix(matrix_path: Path, comparative_dir: Path, include_gsplat: bool) -> None:
    opt_configs = {
        "fvdb_default": {
            "path": str((comparative_dir / "opt_configs" / "fvdb_default.yml").resolve()),
        }
    }
    runs = [{"dataset": "garden", "opt_config": "fvdb_default"}]
    if include_gsplat:
        opt_configs["gsplat_default"] = {
            "path": str((comparative_dir / "opt_configs" / "gsplat_default.yml").resolve()),
        }
        runs.append({"dataset": "garden", "opt_config": "gsplat_default"})

    matrix = {
        "name": "test_matrix",
        "paths": {
            "gsplat_base": "unused",
            "data_base": "unused",
        },
        "datasets": [{"name": "garden", "path": "360_v2/garden"}],
        "opt_configs": opt_configs,
        "runs": runs,
    }
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False))


def test_comparison_benchmark_plot_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module, comparative_dir = _load_comparison_benchmark_module()

    matrix_path = tmp_path / "matrix.yml"
    _write_minimal_matrix(matrix_path, comparative_dir, include_gsplat=False)

    results_path = tmp_path / "results" / "test_matrix"
    results_path.mkdir(parents=True, exist_ok=True)
    report_path = results_path / "garden_comparison_report.json"
    _write_minimal_report(report_path)

    args = [
        "comparison_benchmark.py",
        "--matrix",
        str(matrix_path),
        "--plot-only",
    ]
    monkeypatch.setattr(sys, "argv", args)
    module.main()

    assert (results_path / "summary" / "summary_data.json").exists()
    assert (results_path / "summary" / "summary_comparison.png").exists()


def test_comparison_benchmark_with_stubbed_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module, comparative_dir = _load_comparison_benchmark_module()

    def _stub_train(*_args, **_kwargs):
        return {
            "success": True,
            "total_time": 1.0,
            "training_time": 1.0,
            "exit_code": 0,
            "metrics": {
                "psnr": 1.0,
                "ssim": 1.0,
                "final_gaussian_count": 1,
                "final_loss": 0.0,
            },
            "result_dir": str(tmp_path),
        }

    monkeypatch.setattr(module, "run_fvdb_training", _stub_train)
    monkeypatch.setattr(module, "run_gsplat_training", _stub_train)

    matrix_path = tmp_path / "matrix.yml"
    _write_minimal_matrix(matrix_path, comparative_dir, include_gsplat=True)
    args = [
        "comparison_benchmark.py",
        "--matrix",
        str(matrix_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    module.main()

    report_path = tmp_path / "results" / "test_matrix" / "garden_comparison_report.json"
    assert report_path.exists()
    assert (tmp_path / "results" / "test_matrix" / "summary" / "summary_data.json").exists()


def test_comparison_benchmark_gsplat_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module, comparative_dir = _load_comparison_benchmark_module()

    def _stub_gsplat_training(
        *,
        scene_name: str,
        run_dir: Path,
        matrix_config_path: Path,
        opt_config_path: Path,
        extra_cli_args: list[str],
    ):
        assert scene_name == "garden"
        assert extra_cli_args == ["--strategy.cap_max", "123"]
        merged_opt = yaml.safe_load(Path(opt_config_path).read_text())
        assert merged_opt["mode"] == "mcmc"
        assert merged_opt["training"]["config"]["max_epochs"] == 123
        return {
            "success": True,
            "total_time": 1.0,
            "training_time": 1.0,
            "exit_code": 0,
            "metrics": {
                "psnr": 1.0,
                "ssim": 1.0,
                "final_gaussian_count": 1,
                "final_loss": 0.0,
            },
            "result_dir": str(run_dir),
        }

    monkeypatch.setattr(module, "run_gsplat_training", _stub_gsplat_training)

    matrix = {
        "name": "test_matrix",
        "paths": {"gsplat_base": "unused", "data_base": "unused"},
        "datasets": [{"name": "garden", "path": "360_v2/garden"}],
        "opt_configs": {
            "gsplat_mcmc": {"path": str((comparative_dir / "opt_configs" / "gsplat_mcmc_default.yml").resolve())}
        },
        "runs": [
            {
                "dataset": "garden",
                "opt_config": "gsplat_mcmc",
                "overrides": {
                    "gsplat": {
                        "training": {"config": {"max_epochs": 123}},
                        "cli_args": ["--strategy.cap_max", "123"],
                    }
                },
            }
        ],
    }
    matrix_path = tmp_path / "matrix.yml"
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False))

    args = [
        "comparison_benchmark.py",
        "--matrix",
        str(matrix_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    module.main()

    report_path = tmp_path / "results" / "test_matrix" / "garden_comparison_report.json"
    assert report_path.exists()


def test_training_curves_plot_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that training curve plots are generated when time-series data is present."""
    module, comparative_dir = _load_comparison_benchmark_module()

    matrix_path = tmp_path / "matrix.yml"
    _write_minimal_matrix(matrix_path, comparative_dir, include_gsplat=False)

    results_path = tmp_path / "results" / "test_matrix"
    results_path.mkdir(parents=True, exist_ok=True)
    report_path = results_path / "garden_comparison_report.json"

    # Write report with time-series data
    _write_minimal_report(report_path, include_time_series=True)

    args = [
        "comparison_benchmark.py",
        "--matrix",
        str(matrix_path),
        "--plot-only",
    ]
    monkeypatch.setattr(sys, "argv", args)
    module.main()

    # Assert summary plots exist
    assert (results_path / "summary" / "summary_data.json").exists()
    assert (results_path / "summary" / "summary_comparison.png").exists()

    # Assert training curves plot exists
    training_curves_plot = results_path / "garden_training.png"
    assert training_curves_plot.exists(), f"Training curves plot not found at {training_curves_plot}"


def test_comparative_configs_match_contract():
    from . import contract

    comparative_dir = Path(__file__).resolve().parent / "comparative"
    matrix_config = comparative_dir / "matrix.yml"
    config = contract.load_benchmark_yaml(str(matrix_config))
    contract.validate_comparative_benchmark_yaml(config)

    opt_configs_dir = comparative_dir / "opt_configs"
    opt_config_paths = list(opt_configs_dir.rglob("*.yml"))
    assert opt_config_paths, "No opt_configs found to validate"
    for path in opt_config_paths:
        with path.open("r") as f:
            opt_cfg = yaml.safe_load(f)
        contract.validate_comparative_opt_config(opt_cfg)


def test_get_commits_from_opt_config():
    """Test extraction of commits from opt_config."""
    module, _ = _load_comparison_benchmark_module()

    # Test with no commits section
    opt_config = {"framework": "fvdb", "name": "test"}
    commits = module.get_commits_from_opt_config(opt_config)
    assert commits["fvdb_core"] is None
    assert commits["fvdb_reality_capture"] is None
    assert commits["gsplat"] is None

    # Test with commits section
    opt_config = {
        "framework": "fvdb",
        "name": "test",
        "commits": {
            "fvdb_core": "abc123",
            "fvdb_reality_capture": "def456",
        },
    }
    commits = module.get_commits_from_opt_config(opt_config)
    assert commits["fvdb_core"] == "abc123"
    assert commits["fvdb_reality_capture"] == "def456"
    assert commits["gsplat"] is None


def test_get_commit_key():
    """Test commit key generation for grouping runs."""
    module, _ = _load_comparison_benchmark_module()

    # Test with no commits
    opt_config = {"framework": "fvdb"}
    key = module.get_commit_key(opt_config)
    assert key == (None, None, None)

    # Test with commits
    opt_config = {
        "framework": "fvdb",
        "commits": {"fvdb_core": "abc123", "fvdb_reality_capture": "def456"},
    }
    key = module.get_commit_key(opt_config)
    assert key == ("abc123", "def456", None)

    # Test key is hashable (can be used as dict key)
    d = {key: "test_value"}
    assert d[key] == "test_value"


def test_comparison_benchmark_with_commits_in_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that repository info is included in reports when commits are specified."""
    module, comparative_dir = _load_comparison_benchmark_module()

    def _stub_train(*_args, **_kwargs):
        return {
            "success": True,
            "total_time": 1.0,
            "training_time": 1.0,
            "exit_code": 0,
            "metrics": {
                "psnr": 1.0,
                "ssim": 1.0,
                "final_gaussian_count": 1,
                "final_loss": 0.0,
            },
            "result_dir": str(tmp_path),
        }

    monkeypatch.setattr(module, "run_fvdb_training", _stub_train)

    # Mock CommitManager to avoid actual git operations
    class MockCommitManager:
        def __init__(self, *args, **kwargs):
            pass

        def initialize(self):
            pass

        def ensure_commits(self, required_commits, framework):
            return {
                "fvdb_core": {
                    "commit": "abc123def456",
                    "short_commit": "abc123",
                    "branch": "main",
                    "dirty": False,
                }
            }

    monkeypatch.setattr(module, "CommitManager", MockCommitManager)

    # Create opt_config with commits
    opt_config_content = """
framework: fvdb
name: test_with_commits
color: "#1f77b4"
commits:
  fvdb_core: abc123def456
splat_optimizer: GaussianSplatOptimizer
reconstruction_config:
  max_epochs: 1
training_arguments:
  image_downsample_factor: 4
  use_every_n_as_val: 10
  device: cuda
"""
    opt_config_path = tmp_path / "opt_config_with_commits.yml"
    opt_config_path.write_text(opt_config_content)

    matrix = {
        "name": "test_commits_matrix",
        "paths": {"gsplat_base": "unused", "data_base": "unused"},
        "datasets": [{"name": "garden", "path": "360_v2/garden"}],
        "opt_configs": {"test_commits": {"path": str(opt_config_path)}},
        "runs": [{"dataset": "garden", "opt_config": "test_commits"}],
    }
    matrix_path = tmp_path / "matrix.yml"
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False))

    args = ["comparison_benchmark.py", "--matrix", str(matrix_path)]
    monkeypatch.setattr(sys, "argv", args)
    module.main()

    # Check that report contains repository info
    report_path = tmp_path / "results" / "test_commits_matrix" / "garden_comparison_report.json"
    assert report_path.exists()
    with report_path.open() as f:
        report = json.load(f)
    assert "test_commits" in report
    assert "repositories" in report["test_commits"]
    assert "fvdb_core" in report["test_commits"]["repositories"]


# ---------------------------------------------------------------------------
# build_gsplat_cli_args tests
# ---------------------------------------------------------------------------


class TestBuildGsplatCliArgs:
    @pytest.fixture(autouse=True)
    def _load_func(self):
        self.build_args = _get_build_gsplat_cli_args()

    def test_empty_config_returns_empty_list(self):
        assert self.build_args({}) == []
        assert self.build_args({"training": {}}) == []
        assert self.build_args({"training": {"config": {}}}) == []

    def test_scalar_params_mapped(self):
        opt_config = {"training": {"config": {"near_plane": 0.01, "far_plane": 1e10}}}
        args = self.build_args(opt_config)
        assert "--near_plane" in args
        assert args[args.index("--near_plane") + 1] == "0.01"
        assert "--far_plane" in args
        assert args[args.index("--far_plane") + 1] == "10000000000.0"

    def test_boolean_true_adds_flag(self):
        opt_config = {"training": {"config": {"antialias": True}}}
        args = self.build_args(opt_config)
        assert "--antialiased" in args

    def test_boolean_false_omits_flag(self):
        opt_config = {"training": {"config": {"antialias": False, "random_bkgd": False}}}
        args = self.build_args(opt_config)
        assert args == []

    def test_unmapped_params_ignored(self):
        opt_config = {"training": {"config": {"unknown_param": 42, "near_plane": 0.5}}}
        args = self.build_args(opt_config)
        assert "--near_plane" in args
        assert len(args) == 2  # flag + value, nothing for unknown_param

    def test_mixed_types(self):
        opt_config = {
            "training": {
                "config": {
                    "near_plane": 0.01,
                    "antialias": True,
                    "random_bkgd": False,
                    "sh_degree": 3,
                    "optimize_camera_poses": True,
                }
            }
        }
        args = self.build_args(opt_config)
        assert "--near_plane" in args
        assert "--antialiased" in args
        assert "--random_bkgd" not in args
        assert "--sh_degree" in args
        assert args[args.index("--sh_degree") + 1] == "3"
        assert "--pose_opt" in args
