# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time

import torch
import tyro
from fvdb_3dgs.training import Checkpoint, SceneOptimizationRunner


def main(
    checkpoint_path: pathlib.Path,
    dataset_path: pathlib.Path | None = None,
    results_path: pathlib.Path = pathlib.Path("results"),
    device: str | torch.device = "cuda",
    disable_viewer: bool = False,
    log_tensorboard_every: int = 100,
    log_images_to_tensorboard: bool = False,
    save_results: bool = True,
    save_eval_images: bool = False,
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    checkpoint: Checkpoint = Checkpoint.load(
        checkpoint_path,
        device=device,
        dataset_path=dataset_path,
    )
    runner = SceneOptimizationRunner.from_checkpoint(
        checkpoint=checkpoint,
        results_path=results_path,
        disable_viewer=disable_viewer,
        log_tensorboard_every=log_tensorboard_every,
        log_images_to_tensorboard=log_images_to_tensorboard,
        save_eval_images=save_eval_images,
        save_results=save_results,
    )

    runner.train()

    logger = logging.getLogger(__name__)
    if not disable_viewer:
        logger.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
