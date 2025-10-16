# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import tyro
from fvdb.viz import Viewer

from fvdb_reality_capture import SfmScene
from fvdb_reality_capture.training import (
    GaussianSplatOptimizerConfig,
    GaussianSplatReconstruction,
    GaussianSplatReconstructionConfig,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)
from fvdb_reality_capture.transforms import (
    Compose,
    CropScene,
    CropSceneToPoints,
    DownsampleImages,
    FilterImagesWithLowPoints,
    NormalizeScene,
    PercentileFilterPoints,
)


@dataclass
class SceneTransformConfig:
    """
    Configuration for the transforms to apply to the SfmScene before training.
    """

    # Downsample images by this factor
    image_downsample_factor: int = 4
    # JPEG quality to use when resaving images after downsampling
    rescale_jpeg_quality: int = 95
    # Percentile of points to filter out based on their distance from the median point
    points_percentile_filter: float = 0.0
    # Type of normalization to apply to the scene
    normalization_type: Literal["none", "pca", "ecef2enu", "similarity"] = "pca"
    # Optional bounding box (in the normalized space) to crop the scene to (xmin, xmax, ymin, ymax, zmin, zmax)
    crop_bbox: tuple[float, float, float, float, float, float] | None = None
    # Whether to crop the scene to the bounding box or not
    crop_to_points: bool = False
    # Minimum number of 3D points that must be visible in an image for it to be included in training
    min_points_per_image: int = 5
    # Bounding box to which we crop the scene (in the original space) (xmin, xmax, ymin, ymax, zmin, zmax)
    crop_bbox: tuple[float, float, float, float, float, float] | None = None

    @property
    def scene_transform(self):
        # Dataset transform
        transforms = [
            NormalizeScene(normalization_type=self.normalization_type),
            PercentileFilterPoints(
                percentile_min=np.full((3,), self.points_percentile_filter),
                percentile_max=np.full((3,), 100.0 - self.points_percentile_filter),
            ),
            DownsampleImages(
                image_downsample_factor=self.image_downsample_factor,
                rescaled_jpeg_quality=self.rescale_jpeg_quality,
            ),
            FilterImagesWithLowPoints(min_num_points=self.min_points_per_image),
        ]
        if self.crop_bbox is not None:
            transforms.append(CropScene(self.crop_bbox))
        if self.crop_to_points:
            transforms.append(CropSceneToPoints(margin=0.0))
        return Compose(*transforms)


def main(
    dataset_path: pathlib.Path,
    cfg: GaussianSplatReconstructionConfig = GaussianSplatReconstructionConfig(),
    tx: SceneTransformConfig = SceneTransformConfig(),
    opt: GaussianSplatOptimizerConfig = GaussianSplatOptimizerConfig(),
    io: GaussianSplatReconstructionWriterConfig = GaussianSplatReconstructionWriterConfig(),
    dataset_type: Literal["colmap", "simple_directory", "e57"] = "colmap",
    run_name: str | None = None,
    log_path: pathlib.Path | None = pathlib.Path("fvdb_gslogs"),
    use_every_n_as_val: int = -1,
    device: str | torch.device = "cuda",
    log_every: int = 10,
    visualize_every: int = -1,
    verbose: bool = False,
    out_file_name: str = "result.ply",
):
    """
    Optimize a Gaussian splat radiance field from a set of images and camera poses.

    Args:
        dataset_path (pathlib.Path): Path to the dataset. For "colmap" datasets, this should be the
            directory containing the `images` and `sparse` subdirectories. For "simple_directory" datasets,
            this should be the directory containing the images and a `cameras.txt` file.
        cfg (GaussianSplatReconstructionConfig): Configuration for the Gaussian splat reconstruction.
        tx (SceneTransformConfig): Configuration for the transforms to apply to the SfmScene before training.
        opt (GaussianSplatOptimizerConfig): Configuration for the optimizer.
        io (GaussianSplatReconstructionWriterConfig): Configuration for saving metrics and checkpoints.
        dataset_type (Literal["colmap", "simple_directory", "e57"]): Type of dataset to load.
        run_name (str | None): Name of the run. If None, a name will be generated based on the current date and time.
        log_path (pathlib.Path | None): Path to log metrics, and checkpoints. If None, no metrics or checkpoints will be saved.
        use_every_n_as_val (int): Use every n-th image as a validation image.
        device (str | torch.device): Device to use for training.
        log_every (int): Log training metrics every n steps.
        visualize_every (int): Update the viewer every n epochs. If -1, do not visualize.
        verbose (bool): Whether to log debug messages.
        out_file_name (str): Name of the output PLY file to save the model. Default is "result.ply".
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
    logger = logging.getLogger(__name__)

    if dataset_type == "colmap":
        sfm_scene = SfmScene.from_colmap(dataset_path)
    elif dataset_type == "simple_directory":
        sfm_scene = SfmScene.from_simple_directory(dataset_path)
    elif dataset_type == "e57":
        sfm_scene = SfmScene.from_e57(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset_type {dataset_type}")

    if visualize_every > 0:
        viewer = Viewer()
    else:
        viewer = None

    writer = GaussianSplatReconstructionWriter(run_name=run_name, save_path=log_path, config=io, exist_ok=False)
    runner = GaussianSplatReconstruction.from_sfm_scene(
        tx.scene_transform(sfm_scene),
        config=cfg,
        optimizer_config=opt,
        writer=writer,
        viewer=viewer,
        use_every_n_as_val=use_every_n_as_val,
        log_interval_steps=log_every,
        viewer_update_interval_epochs=visualize_every,
        device=device,
    )

    runner.train()

    runner.model.save_ply(out_file_name, metadata=runner.optimization_metadata)

    if viewer is not None:
        logger.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
