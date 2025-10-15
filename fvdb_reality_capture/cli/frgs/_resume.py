# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Annotated

import torch
import tyro
from fvdb.viz import Viewer
from tyro.conf import arg

from fvdb_reality_capture.training import (
    GaussianSplatReconstruction,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)

from ._common import BaseCommand


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
    Resume reconstructing a 3D Gaussian Splatting model from a checkpoint. This command loads a model
    checkpoint and continues reconstruction from that point. The dataset used to create the checkpoint
    must be at the same path as when the checkpoint was created.
    """

    # Path to the checkpoint file containing the Gaussian Splat model.
    checkpoint_path: tyro.conf.Positional[pathlib.Path]

    # Configuration for saving metrics and checkpoints.
    io: WriterConfig = WriterConfig()

    # Name of the run. If None, a name will be generated based on the current date and time.
    run_name: Annotated[str | None, arg(aliases=["-n"])] = None

    # How frequently (in epochs) to update the viewer during reconstruction.
    # An epoch is one full pass through the training images. If -1, do not visualize.
    update_viz_every: Annotated[float, arg(aliases=["-uv"])] = -1.0

    # Which device to use for reconstruction. Must be a cuda device. You can pass in a specific device index via
    # cuda:N where N is the device index, or "cuda" to use the default cuda device.
    # CPU is not supported. Default is "cuda".
    device: Annotated[str | torch.device, arg(aliases=["-d"])] = "cuda"

    # If set, show verbose debug messages.
    verbose: Annotated[bool, arg(aliases=["-v"])] = False

    # Path to save the output PLY file.
    # Defaults to `out.ply` in the current working directory.
    # Path must end in .ply or .usdz.
    out_path: Annotated[pathlib.Path, arg(aliases=["-o"])] = pathlib.Path("out_resumed.ply")

    def execute(self) -> None:
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
        logger = logging.getLogger(__name__)

        logger.info(f"Loading checkpoint at {self.checkpoint_path}")
        checkpoint_state = torch.load(self.checkpoint_path, map_location=self.device)

        writer = GaussianSplatReconstructionWriter(
            run_name=self.run_name, save_path=self.io.log_path, config=self.io, exist_ok=False
        )
        if self.update_viz_every > 0:
            viewer = Viewer()
        else:
            viewer = None

        runner = GaussianSplatReconstruction.from_state_dict(
            checkpoint_state,
            device=self.device,
            writer=writer,
            viewer=viewer,
            log_interval_steps=self.io.log_every,
            viewer_update_interval_epochs=self.update_viz_every,
        )

        runner.train()

        runner.model.save_ply(self.out_path, metadata=runner.optimization_metadata)


def main(
    checkpoint_path: pathlib.Path,
    io: GaussianSplatReconstructionWriterConfig = GaussianSplatReconstructionWriterConfig(),
    run_name: str | None = None,
    log_path: pathlib.Path | None = pathlib.Path("fvdb_gslogs"),
    device: str | torch.device = "cuda",
    visualize_every: int = -1,
    log_every: int = 10,
    verbose: bool = False,
    out_file_name: str = "resumed.ply",
):
    """
    Resume training a 3D Gaussian Splatting model from a checkpoint. This function loads a model
    checkpoint and continues training from that point. The dataset used to create the checkpoint
    must be at the same path as when the checkpoint was created.

    Args:
        checkpoint_path (pathlib.Path): Path to the checkpoint file.
        io (GaussianSplatReconstructionWriterConfig): Configuration for saving metrics and checkpoints.
        run_name (str | None): Name of the training run.
        log_path (pathlib.Path | None): Path to log metrics, and checkpoints. If None, no metrics or checkpoints will be saved.
        device (str | torch.device): Device to use for training.
        visualize_every (int): Update the viewer every n epochs. If -1, do not visualize.
        log_every (int): Log training metrics every n steps.
        verbose (bool): Whether to log debug messages.
        out_file_name (str): Name of the output PLY file to save the model. Default is "resumed.ply".
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")

    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")

    writer = GaussianSplatReconstructionWriter(run_name=run_name, save_path=log_path, config=io, exist_ok=False)

    if visualize_every > 0:
        viewer = Viewer()
    else:
        viewer = None

    runner = GaussianSplatReconstruction.from_state_dict(
        checkpoint_state,
        device=device,
        writer=writer,
        viewer=viewer,
        log_interval_steps=log_every,
        viewer_update_interval_epochs=visualize_every,
    )

    runner.train()

    runner.model.save_ply(out_file_name, metadata=runner.optimization_metadata)

    logger = logging.getLogger(__name__)

    if viewer is not None:
        logger.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
