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

from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.radiance_fields import (
    GaussianSplatReconstruction,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)

from ._common import save_model_from_runner


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
    Resume reconstructing a 3D Gaussian Splat radiance field from a checkpoint. This command loads a model
    checkpoint and continues reconstruction from that point. The dataset used to create the checkpoint
    must be at the same path as when the checkpoint was created.

    Example usage:

        # Resume reconstruction from a checkpoint and save the final model to out_resumed.ply
        frgs resume checkpoint.pt -o out_resumed.ply
    """

    # Path to the checkpoint file containing the Gaussian Splat radiance field.
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
        checkpoint_state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        writer = GaussianSplatReconstructionWriter(
            run_name=self.run_name, save_path=self.io.log_path, config=self.io, exist_ok=False
        )
        if self.update_viz_every > 0:
            logger.info(f"Starting viewer server on {self.viewer_ip_address}:{self.viewer_port}")
            fviz.init(ip_address=self.viewer_ip_address, port=self.viewer_port, verbose=self.verbose)
            viz_scene = fviz.get_scene("Gaussian Splat Reconstruction Visualization")
        else:
            viz_scene = None

        runner = GaussianSplatReconstruction.from_state_dict(
            checkpoint_state,
            device=self.device,
            writer=writer,
            viz_scene=viz_scene,
            log_interval_steps=self.io.log_every,
            viz_update_interval_epochs=self.update_viz_every,
        )

        runner.optimize()

        logger.info(f"Saving final model to {self.out_path}")
        save_model_from_runner(self.out_path, runner)
