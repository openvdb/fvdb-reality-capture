# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time

import torch
import tyro
from fvdb.viz import Viewer

from fvdb_reality_capture.training import GaussianSplatReconstruction


def main(
    checkpoint_path: pathlib.Path,
    override_results_path: pathlib.Path | None = None,
    run_name_suffix: str = "_resumed",
    device: str | torch.device = "cuda",
    visualize_every_epoch: int = -1,
    log_tensorboard_every_step: int = 100,
    tensorboard_path: pathlib.Path | None = None,
    save_eval_images: bool = False,
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    if visualize_every_epoch > 0:
        viewer = Viewer()
    else:
        viewer = None

    runner = GaussianSplatReconstruction.from_checkpoint(
        checkpoint_path=checkpoint_path,
        override_results_path=override_results_path,
        tensorboard_path=tensorboard_path,
        tensorboard_log_interval_steps=log_tensorboard_every_step,
        save_eval_images=save_eval_images,
        device=device,
        viewer=viewer,
        run_name_suffix=run_name_suffix,
        viewer_update_interval_epochs=visualize_every_epoch,
    )

    runner.train()

    logger = logging.getLogger(__name__)
    if not visualize_every_epoch > 0:
        logger.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
