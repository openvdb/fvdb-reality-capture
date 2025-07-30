# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time

import torch
import tyro
from training import SceneOptimizationRunner
from training.checkpoint import Checkpoint


def main(
    checkpoint_path: pathlib.Path,
    dataset_path: pathlib.Path | None = None,
    device: str | torch.device = "cuda",
):
    """
    Visualize a scene in a saved checkpoint file.

    Args:
        checkpoint_path (pathlib.Path): Path to the checkpoint file containing the Gaussian splat model.
        dataset_path (pathlib.Path | None): Path to the dataset used for training or None to use the dataset
            in the checkpoint if it is available (default is None).
        device (str | torch.device): Device to use for computation (default is "cuda").
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    checkpoint: Checkpoint = Checkpoint.load(
        checkpoint_path,
        device=device,
        dataset_path=dataset_path,
    )

    # The runner will create a visualization for us so we'll just create one pause while the
    # viewer is running.
    runner = SceneOptimizationRunner.from_checkpoint(
        checkpoint=checkpoint,
        results_path=pathlib.Path("results"),
        disable_viewer=False,
        log_tensorboard_every=100,
        log_images_to_tensorboard=False,
        save_eval_images=False,
        save_results=False,
    )

    logger = logging.getLogger("visualize")
    logger.info("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
