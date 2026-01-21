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


def _write_minimal_report(report_path: Path) -> None:
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
                "metrics": {"psnr": 1.0, "ssim": 1.0, "final_gaussian_count": 1},
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
            "fvdb_base": "unused",
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
            "metrics": {"psnr": 1.0, "ssim": 1.0, "final_gaussian_count": 1, "final_loss": 0.0},
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
        *, scene_name: str, run_dir: Path, matrix_config_path: Path, opt_config_path: Path, extra_cli_args: list[str]
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
            "metrics": {"psnr": 1.0, "ssim": 1.0, "final_gaussian_count": 1, "final_loss": 0.0},
            "result_dir": str(run_dir),
        }

    monkeypatch.setattr(module, "run_gsplat_training", _stub_gsplat_training)

    matrix = {
        "name": "test_matrix",
        "paths": {"fvdb_base": "unused", "gsplat_base": "unused", "data_base": "unused"},
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
