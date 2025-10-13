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
    results_path: pathlib.Path | None = None,
    device: str | torch.device = "cuda",
    visualize_every: int = -1,
    log_every: int = 10,
    verbose: bool = False,
):
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")

    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")

    writer = GaussianSplatReconstructionWriter(run_name=run_name, save_path=results_path, config=io, exist_ok=False)

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

    logger = logging.getLogger(__name__)
    if viewer is not None:
        logger.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
