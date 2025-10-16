# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time

import torch
import tyro
from fvdb.viz import Viewer

from fvdb_reality_capture.training import (
    GaussianSplatReconstruction,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)


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
