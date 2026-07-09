# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import pathlib

import tyro

from fvdb_reality_capture.cli.frgs._instance_segmentation import InstanceSegmentation
from fvdb_reality_capture.cli.frgs._resume import Resume
from fvdb_reality_capture.cli.frgs._show import Show


def test_instance_segmentation_cli_contract():
    command = tyro.cli(
        InstanceSegmentation,
        args=[
            "dataset",
            "--reconstruction-path",
            "scene.ply",
            "--out-path",
            "scene.garfvdb",
            "--cfg.max-epochs",
            "2",
        ],
    )
    assert command.dataset_path == pathlib.Path("dataset")
    assert command.reconstruction_path == pathlib.Path("scene.ply")
    assert command.out_path == pathlib.Path("scene.garfvdb")
    assert command.cfg.max_epochs == 2
    assert command.cfg.model.use_grid


def test_show_accepts_garfvdb_controls():
    command = tyro.cli(
        Show,
        args=["scene.garfvdb", "--scale-fraction", "0.25", "--mask-blend", "0.75"],
    )
    assert command.input_path == pathlib.Path("scene.garfvdb")
    assert command.scale_fraction == 0.25
    assert command.mask_blend == 0.75


def test_resume_uses_product_specific_default_at_execution_time():
    command = tyro.cli(Resume, args=["checkpoint.pt"])
    assert command.checkpoint_path == pathlib.Path("checkpoint.pt")
    assert command.out_path is None
