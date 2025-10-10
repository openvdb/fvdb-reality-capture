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
    save_results: bool = True,
    save_images: bool = True,
    device: str | torch.device = "cuda",
):
    """
    Run evaluation on a Gaussian splat scene. This will render each image in the validation set,
    compute statistics (PSNR, SSIM, LPIPS), and save the rendered images and ground truth validation
    images to disk.

    Args:
        checkpoint_path (pathlib.Path): Path to the checkpoint file containing the Gaussian splat model.
        save_results (bool): Whether to save the evaluation results (default is True).
            Results will be saved in a subdirectory of the checkpoint directory.
        save_images (bool): Whether to save the rendered images (default is True).
        device (str | torch.device): Device to use for computation (default is "cuda").
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    runner = GaussianSplatReconstruction.from_checkpoint(
        checkpoint_path=checkpoint_path, save_eval_images=save_images, device=device, save_results=save_results
    )

    logger = logging.getLogger("evaluate")
    logger.info("Running eval on checkpoint.")
    runner.eval()


if __name__ == "__main__":
    tyro.cli(main)
