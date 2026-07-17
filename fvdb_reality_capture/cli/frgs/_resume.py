# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Annotated

import fvdb.viz as fviz
import torch
import tyro
from tyro.conf import arg

from fvdb_reality_capture.checkpoints import (
    TrainingCheckpoint,
    load_training_checkpoint,
)
from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.instance_segmentation import (
    GARFVDB_TRAINING_METHOD,
    GARfVDBTrainer,
)
from fvdb_reality_capture.instance_segmentation.training.segmentation_writer import (
    GARfVDBWriter,
    GARfVDBWriterConfig,
)
from fvdb_reality_capture.radiance_fields import (
    GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD,
    GaussianSplatReconstruction,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)

from ._common import save_model_from_runner
from ._resume_registry import (
    ResumeContext,
    ResumeHandler,
    get_resume_handler,
    register_resume_handler,
)


@dataclass
class WriterConfig(GaussianSplatReconstructionWriterConfig):
    """
    Configuration for saving and logging metrics, images, and checkpoints.
    """

    # Path to save logs, checkpoints, and other output to.
    # Defaults to `frgs_logs` in the current working directory.
    log_path: pathlib.Path | None = pathlib.Path("frgs_logs")

    # How frequently to log metrics during reconstruction.
    log_every: int = 10


@dataclass
class Resume(BaseCommand):
    """
    Resume a Reality Capture training run. The versioned checkpoint envelope dispatches to the registered method
    handler. Source dataset paths recorded by the checkpoint must remain available at the same path as when the
    checkpoint was created.

    Example usage:

        # Resume reconstruction from a checkpoint and save the final model to out_resumed.ply
        frgs resume checkpoint.pt -o out_resumed.ply
    """

    # Path to a versioned Reality Capture training checkpoint.
    checkpoint_path: tyro.conf.Positional[pathlib.Path]

    # Configure saving and logging metrics, images, and checkpoints.
    io: WriterConfig = field(default_factory=WriterConfig)

    # Name of the run. If None, a name will be generated based on the current date and time.
    run_name: Annotated[str | None, arg(aliases=["-n"])] = None

    # How frequently (in epochs) to update the viewer during reconstruction.
    # An epoch is one full pass through the dataset. If -1, do not visualize.
    update_viz_every: Annotated[float, arg(aliases=["-uv"])] = -1.0

    # The port to expose the viewer server on if update_viz_every > 0.
    viewer_port: Annotated[int, arg(aliases=["-p"])] = 8080

    # The IP address to expose the viewer server on if update_viz_every > 0.
    viewer_ip_address: Annotated[str, arg(aliases=["-ip"])] = "127.0.0.1"

    # Which device to use for reconstruction. Must be a cuda device. You can pass in a specific device index via
    # cuda:N where N is the device index, or "cuda" to use the default cuda device.
    # CPU is not supported. Default is "cuda:0".
    device: Annotated[str | torch.device, arg(aliases=["-d"])] = "cuda:0"

    # If set, show verbose debug messages.
    verbose: Annotated[bool, arg(aliases=["-v"])] = False

    # Output path. Defaults to out_resumed.ply for reconstruction checkpoints and
    # out_resumed.garfvdb for GARfVDB checkpoints.
    out_path: Annotated[pathlib.Path | None, arg(aliases=["-o"])] = None

    reconstruction_path: Annotated[pathlib.Path | None, arg(aliases=["-r"])] = None
    """Override the reconstruction referenced by a GARfVDB checkpoint if it moved."""

    def execute(self) -> None:
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
        logger = logging.getLogger(__name__)

        logger.info(f"Loading checkpoint at {self.checkpoint_path}")
        checkpoint = load_training_checkpoint(self.checkpoint_path, map_location=self.device)
        handler = get_resume_handler(checkpoint.method)
        out_path = self.out_path or pathlib.Path(handler.default_output_name)
        logger.info("Dispatching checkpoint method %s", checkpoint.method)
        handler.callback(checkpoint, self, out_path)


def _resume_garfvdb(checkpoint: TrainingCheckpoint, command: ResumeContext, out_path: pathlib.Path) -> None:
    if command.update_viz_every > 0:
        raise ValueError("Live GARfVDB resume visualization is unsupported; resume first, then use frgs show.")
    writer_config = GARfVDBWriterConfig(
        save_images=command.io.save_images,
        save_checkpoints=command.io.save_checkpoints,
        save_metrics=command.io.save_metrics,
        metrics_file_buffer_size=command.io.metrics_file_buffer_size,
        use_tensorboard=command.io.use_tensorboard,
        save_images_to_tensorboard=command.io.save_images_to_tensorboard,
    )
    writer = GARfVDBWriter(
        run_name=command.run_name,
        save_path=command.io.log_path,
        config=writer_config,
        exist_ok=False,
    )
    trainer = GARfVDBTrainer.from_checkpoint_state(
        checkpoint.state,
        writer=writer,
        device=command.device,
        reconstruction_path=command.reconstruction_path,
    )
    trainer.train()
    logging.getLogger(__name__).info("Saving resumed GARfVDB product to %s", out_path)
    trainer.to_product().save(out_path)


def _resume_gaussian_splat(checkpoint: TrainingCheckpoint, command: ResumeContext, out_path: pathlib.Path) -> None:
    writer_config = GaussianSplatReconstructionWriterConfig(
        save_images=command.io.save_images,
        save_checkpoints=command.io.save_checkpoints,
        save_plys=command.io.save_plys,
        save_metrics=command.io.save_metrics,
        metrics_file_buffer_size=command.io.metrics_file_buffer_size,
        use_tensorboard=command.io.use_tensorboard,
        save_images_to_tensorboard=command.io.save_images_to_tensorboard,
    )
    writer = GaussianSplatReconstructionWriter(
        run_name=command.run_name,
        save_path=command.io.log_path,
        config=writer_config,
        exist_ok=False,
    )
    if command.update_viz_every > 0:
        logging.getLogger(__name__).info(
            "Starting viewer server on %s:%d",
            command.viewer_ip_address,
            command.viewer_port,
        )
        fviz.init(
            ip_address=command.viewer_ip_address,
            port=command.viewer_port,
            verbose=command.verbose,
        )
        viz_scene = fviz.get_scene("Gaussian Splat Reconstruction Visualization")
    else:
        viz_scene = None

    runner = GaussianSplatReconstruction.from_state_dict(
        checkpoint.state,
        device=command.device,
        writer=writer,
        viz_scene=viz_scene,
        log_interval_steps=command.io.log_every,
        viz_update_interval_epochs=command.update_viz_every,
    )
    runner.optimize()
    logging.getLogger(__name__).info("Saving final model to %s", out_path)
    save_model_from_runner(out_path, runner)


register_resume_handler(
    ResumeHandler(
        method=GARFVDB_TRAINING_METHOD,
        default_output_name="out_resumed.garfvdb",
        callback=_resume_garfvdb,
    )
)
register_resume_handler(
    ResumeHandler(
        method=GAUSSIAN_SPLAT_RECONSTRUCTION_METHOD,
        default_output_name="out_resumed.ply",
        callback=_resume_gaussian_splat,
    )
)
