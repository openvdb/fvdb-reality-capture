# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import pathlib
import tempfile

import pytest
import torch

from fvdb_reality_capture.checkpoints import (
    TRAINING_CHECKPOINT_SCHEMA,
    TRAINING_CHECKPOINT_SCHEMA_VERSION,
    TrainingCheckpointError,
    TrainingCheckpointVersionError,
    create_training_checkpoint,
    load_training_checkpoint,
    parse_training_checkpoint,
)
from fvdb_reality_capture.cli.frgs._resume import Resume
from fvdb_reality_capture.cli.frgs._resume_registry import (
    ResumeHandler,
    UnknownCheckpointMethodError,
    get_resume_handler,
    register_resume_handler,
)
from fvdb_reality_capture.instance_segmentation.checkpoint import (
    GARFVDB_TRAINING_METHOD,
)
from fvdb_reality_capture.instance_segmentation.training.segmentation_writer import (
    GARfVDBWriter,
    GARfVDBWriterConfig,
)
from fvdb_reality_capture.radiance_fields import (
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)
from fvdb_reality_capture.radiance_fields.checkpoint import (
    GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD,
)


def _state(version: str = "1.2.3") -> dict:
    return {"version": version, "step": 7, "payload": torch.tensor([1, 2, 3])}


def test_training_checkpoint_container_round_trip():
    checkpoint = create_training_checkpoint("test.method", _state(), method_version="method-v7")
    serialized = checkpoint.to_dict()
    assert serialized["schema"] == TRAINING_CHECKPOINT_SCHEMA
    assert serialized["schema_version"] == TRAINING_CHECKPOINT_SCHEMA_VERSION
    assert serialized["method"] == "test.method"
    assert serialized["method_version"] == "method-v7"
    restored = parse_training_checkpoint(serialized)
    assert restored.method == checkpoint.method
    assert restored.method_version == checkpoint.method_version
    assert torch.equal(restored.state["payload"], checkpoint.state["payload"])


def test_load_training_checkpoint_rejects_future_schema():
    root = create_training_checkpoint("test.future", _state(), method_version="1.2.3").to_dict()
    root["schema_version"] = TRAINING_CHECKPOINT_SCHEMA_VERSION + 1
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "future.pt"
        torch.save(root, path)
        with pytest.raises(TrainingCheckpointVersionError, match="newer"):
            load_training_checkpoint(path)


def test_load_training_checkpoint_rejects_directories_without_product_extension_knowledge():
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "scene.garfvdb"
        path.mkdir()
        with pytest.raises(TrainingCheckpointError, match="directory"):
            load_training_checkpoint(path)


def test_checkpoint_dispatch_does_not_depend_on_file_extension():
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "intentionally-unusual.garfvdb"
        torch.save(
            create_training_checkpoint(
                "test.extension_independent",
                _state(),
                method_version="1.2.3",
            ).to_dict(),
            path,
        )
        checkpoint = load_training_checkpoint(path)
    assert checkpoint.method == "test.extension_independent"


def test_released_flat_gaussian_checkpoint_is_adapted():
    state = {
        "magic": "GaussianSplattingCheckpoint",
        "version": "0.1.0",
    }
    checkpoint = parse_training_checkpoint(state)
    assert checkpoint.method == GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD
    assert checkpoint.state == state


def test_unknown_resume_method_has_no_gaussian_fallback():
    with pytest.raises(UnknownCheckpointMethodError, match="unknown.method"):
        get_resume_handler("unknown.method")

    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "unknown.pt"
        torch.save(
            create_training_checkpoint("unknown.method", _state(), method_version="1.2.3").to_dict(),
            path,
        )
        with pytest.raises(UnknownCheckpointMethodError, match="unknown.method"):
            Resume(checkpoint_path=path, device="cpu").execute()


def test_resume_dispatches_registered_method_and_uses_handler_default():
    method = "tests.synthetic_resume"
    calls = []

    def callback(checkpoint, command, out_path):
        calls.append((checkpoint, command, out_path))

    register_resume_handler(
        ResumeHandler(
            method=method,
            default_output_name="synthetic.product",
            callback=callback,
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "synthetic.pt"
        torch.save(
            create_training_checkpoint(method, _state(), method_version="1.2.3").to_dict(),
            path,
        )
        command = Resume(checkpoint_path=path, device="cpu")
        command.execute()

    assert len(calls) == 1
    checkpoint, passed_command, out_path = calls[0]
    assert checkpoint.method == method
    assert passed_command is command
    assert out_path == pathlib.Path("synthetic.product")


@pytest.mark.parametrize(
    ("writer_factory", "expected_method"),
    [
        (
            lambda root: GaussianSplatReconstructionWriter(
                run_name="checkpoint_container",
                save_path=root,
                config=GaussianSplatReconstructionWriterConfig(
                    save_images=False,
                    save_checkpoints=True,
                    save_plys=False,
                    save_metrics=False,
                ),
            ),
            GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD,
        ),
        (
            lambda root: GARfVDBWriter(
                run_name="checkpoint_container",
                save_path=root,
                config=GARfVDBWriterConfig(
                    save_images=False,
                    save_checkpoints=True,
                    save_metrics=False,
                ),
            ),
            GARFVDB_TRAINING_METHOD,
        ),
    ],
)
def test_disk_writers_save_versioned_containers(writer_factory, expected_method):
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        writer = writer_factory(root)
        writer.save_checkpoint(4, "train_ckpt.pt", _state())
        paths = list(root.rglob("train_ckpt.pt"))
        assert len(paths) == 1
        checkpoint = load_training_checkpoint(paths[0], expected_method=expected_method)
        assert checkpoint.method == expected_method
        assert checkpoint.state["step"] == 7
