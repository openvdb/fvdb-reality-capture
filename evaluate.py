# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib

import torch
import tyro

from fvdb_reality_capture.training import GaussianSplatReconstruction


def main(
    checkpoint_path: pathlib.Path,
    results_path: pathlib.Path = pathlib.Path("results"),
    save_images: bool = True,
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

    runner = GaussianSplatReconstruction.from_checkpoint(
        checkpoint_path=checkpoint_path, save_eval_images=save_images, device=device
    )

    logger = logging.getLogger("evaluate")
    logger.info("Running eval on checkpoint.")
    runner.eval()


if __name__ == "__main__":
    tyro.cli(main)
