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


def test_comparison_benchmark_plot_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module, comparative_dir = _load_comparison_benchmark_module()

    report_path = tmp_path / "garden_comparison_report.json"
    _write_minimal_report(report_path)

    benchmark_config = comparative_dir / "benchmark_config.yaml"
    args = [
        "comparison_benchmark.py",
        "--benchmark-config",
        str(benchmark_config),
        "--plot-only",
        "--result-dir",
        str(tmp_path),
        "--scenes",
        "garden",
    ]
    monkeypatch.setattr(sys, "argv", args)
    module.main()

    assert (tmp_path / "summary" / "summary_data.json").exists()
    assert (tmp_path / "summary" / "summary_comparison.png").exists()


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

    benchmark_config = comparative_dir / "benchmark_config.yaml"
    fvdb_config = comparative_dir / "opt_configs" / "fvdb_default.yml"
    gsplat_config = comparative_dir / "opt_configs" / "gsplat_default.yml"
    args = [
        "comparison_benchmark.py",
        "--benchmark-config",
        str(benchmark_config),
        "--opt-configs",
        str(fvdb_config),
        str(gsplat_config),
        "--scenes",
        "garden",
        "--result-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    module.main()

    report_path = tmp_path / "garden_comparison_report.json"
    assert report_path.exists()
    assert (tmp_path / "summary" / "summary_data.json").exists()


def test_comparative_configs_match_contract():
    from . import contract

    comparative_dir = Path(__file__).resolve().parent / "comparative"
    benchmark_config = comparative_dir / "benchmark_config.yaml"
    config = contract.load_benchmark_yaml(str(benchmark_config))
    contract.validate_comparative_benchmark_yaml(config)

    opt_configs_dir = comparative_dir / "opt_configs"
    opt_config_paths = list(opt_configs_dir.rglob("*.yml"))
    assert opt_config_paths, "No opt_configs found to validate"
    for path in opt_config_paths:
        with path.open("r") as f:
            opt_cfg = yaml.safe_load(f)
        contract.validate_comparative_opt_config(opt_cfg)
