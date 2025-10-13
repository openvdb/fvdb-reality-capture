# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib

import torch
import tyro

from fvdb_reality_capture.training import (
    GaussianSplatReconstruction,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)


def main(
    checkpoint_path: pathlib.Path,
    save_path: pathlib.Path | None = None,
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

    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")

    writer_config = GaussianSplatReconstructionWriterConfig(
        save_images=save_images, save_metrics=True, save_plys=False, save_checkpoints=False, use_tensorboard=False
    )
    writer = GaussianSplatReconstructionWriter(
        run_name=None,
        save_path=save_path if save_path is not None else checkpoint_path.parent / "eval",
        config=writer_config,
        exist_ok=True,
    )

    runner = GaussianSplatReconstruction.from_state_dict(checkpoint_state, device=device, writer=writer)

    logger = logging.getLogger("evaluate")
    logger.info("Running eval on checkpoint.")
    runner.eval()


if __name__ == "__main__":
    tyro.cli(main)
