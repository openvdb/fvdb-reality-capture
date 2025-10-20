# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib
from dataclasses import dataclass
from typing import Annotated

import torch
import tyro
from tyro.conf import arg

from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.radiance_fields import (
    GaussianSplatReconstruction,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)


@dataclass
class Evaluate(BaseCommand):
    """
    Evaluate a Gaussian Splat reconstruction on the validation set.

    You can change the validation split used for evaluation using `--use-every-n-as-val`
    argument. For example, `--use-every-n-as-val 10` will use every 10th image in the dataset
    as a validation image. If you do not provide this argument, the validation split
    provided in the dataset (if any) will be used. If the dataset does not provide a validation
    split, all images will be used for evaluation.

    This will render each image in the validation set, compute statistics (PSNR, SSIM, LPIPS),
    and save the rendered images and ground truth validation images to disk.

    By default results will be saved to a directory named "eval" in the same directory as the checkpoint.

    Example usage:

        # Evaluate a checkpoint and save results to the default log path
        frgs evaluate checkpoint.pt

        # Evaluate a checkpoint on a new dataset split
        frgs evaluate checkpoint.pt --use-every-n-as-val 10

        # Evaluate a checkpoint and save results to a custom log path
        frgs evaluate checkpoint.pt --log-path ./eval_results

        # Evaluate a checkpoint but don't write out rendered images
        frgs evaluate checkpoint.pt --save-images False

    """

    # Path to the checkpoint file containing the Gaussian Splat model.
    checkpoint_path: tyro.conf.Positional[pathlib.Path]

    # Path to save the evaluation results. If not provided, results will be saved in a subdirectory
    # of the checkpoint directory named "eval".
    log_path: Annotated[pathlib.Path | None, arg(aliases=["-l"])] = None

    # Use every n-th image as a validation image. If not set, will use the validation split
    # provided in the dataset. If the dataset does not provide a validation split, will use
    # all images for evaluation.
    use_every_n_as_val: Annotated[int | None, arg(aliases=["-vn"])] = None

    # Whether to save the rendered images. Defaults to True.
    save_images: Annotated[bool, arg(aliases=["-s"])] = True

    # Device to use for computation. Defaults to "cuda".
    device: str | torch.device = "cuda"

    def execute(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
        logger = logging.getLogger(__name__)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file {self.checkpoint_path} does not exist.")

        if self.log_path is None:
            self.log_path = self.checkpoint_path.parent / "eval"

        logger.info(f"Evaluating checkpoint: {self.checkpoint_path}")
        checkpoint_state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        writer_config = GaussianSplatReconstructionWriterConfig(
            save_images=self.save_images,
            save_metrics=True,
            save_plys=False,
            save_checkpoints=False,
            use_tensorboard=False,
        )

        writer = GaussianSplatReconstructionWriter(
            run_name=None,
            save_path=self.log_path,
            config=writer_config,
            exist_ok=True,
        )

        runner = GaussianSplatReconstruction.from_state_dict(
            checkpoint_state, device=self.device, writer=writer, override_use_every_n_as_val=self.use_every_n_as_val
        )

        if runner.validation_dataset is None or len(runner.validation_dataset) == 0:
            if self.use_every_n_as_val is None:
                logger.info("No validation split found in dataset. Using all images for evaluation.")
                use_every_n_as_val = 1
            else:
                logger.info(
                    f"No validation split found in dataset. Using every {self.use_every_n_as_val} image for evaluation."
                )
                use_every_n_as_val = self.use_every_n_as_val
            runner = GaussianSplatReconstruction.from_state_dict(
                checkpoint_state,
                device=self.device,
                writer=writer,
                override_use_every_n_as_val=use_every_n_as_val,
            )
        logger = logging.getLogger("evaluate")
        logger.info("Running evaluation on checkpoint.")
        runner.eval()
