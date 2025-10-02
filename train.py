# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time
from typing import Literal

import torch
import tyro

from fvdb_reality_capture.training import Config, SceneOptimizationRunner


def main(
    dataset_path: pathlib.Path,
    cfg: Config = Config(),
    run_name: str | None = None,
    image_downsample_factor: int = 4,
    points_percentile_filter: float = 0.0,
    normalization_type: Literal["none", "pca", "ecef2enu", "similarity"] = "pca",
    crop_bbox: tuple[float, float, float, float, float, float] | None = None,
    crop_to_points: bool = False,
    min_points_per_image: int = 5,
    results_path: pathlib.Path = pathlib.Path("results"),
    device: str | torch.device = "cuda",
    use_every_n_as_val: int = 8,
    disable_viewer: bool = False,
    log_tensorboard_every: int = 100,
    log_images_to_tensorboard: bool = False,
    save_results: bool = True,
    save_eval_images: bool = False,
    verbose: bool = False,
    dataset_type: Literal["colmap", "simple_directory"] = "colmap",
):
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")

    runner = SceneOptimizationRunner.new_run(
        config=cfg,
        dataset_path=dataset_path,
        run_name=run_name,
        image_downsample_factor=image_downsample_factor,
        points_percentile_filter=points_percentile_filter,
        normalization_type=normalization_type,
        crop_bbox=crop_bbox,
        crop_to_points=crop_to_points,
        min_points_per_image=min_points_per_image,
        results_path=results_path,
        device=device,
        use_every_n_as_val=use_every_n_as_val,
        disable_viewer=disable_viewer,
        log_tensorboard_every=log_tensorboard_every,
        log_images_to_tensorboard=log_images_to_tensorboard,
        save_results=save_results,
        save_eval_images=save_eval_images,
        dataset_type=dataset_type,
    )

    runner.train()

    logger = logging.getLogger("train")
    if not disable_viewer:
        logger.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
