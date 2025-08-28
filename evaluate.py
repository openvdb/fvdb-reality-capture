# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib

import torch
import tyro
from fvdb_3dgs.training import Checkpoint, SceneOptimizationRunner


def main(
    checkpoint_path: pathlib.Path,
    dataset_path: pathlib.Path | None = None,
    device: str | torch.device = "cuda",
):
    """
    Run evaluation on a Gaussian splat scene. This will render each image in the validation set,
    compute statistics (PSNR, SSIM, LPIPS), and save the rendered images and ground truth validation
    images to disk.

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
        disable_viewer=True,
        log_tensorboard_every=100,
        log_images_to_tensorboard=False,
        save_eval_images=True,
        save_results=True,
    )

    logger = logging.getLogger("evaluate")
    logger.info("Running eval on checkpoint.")
    runner.eval()


if __name__ == "__main__":
    tyro.cli(main)
