# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import logging
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import tqdm
import tyro
from fvdb import GaussianSplat3d
from fvdb.viz import Viewer

from fvdb_reality_capture.sfm_scene import SfmScene
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
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    overlap_percent: float = 0.1,
    cfg: GaussianSplatReconstructionConfig = GaussianSplatReconstructionConfig(
        remove_gaussians_outside_scene_bbox=True
    ),
    tx: SceneTransformConfig = SceneTransformConfig(),
    opt: GaussianSplatOptimizerConfig = GaussianSplatOptimizerConfig(),
    io: GaussianSplatReconstructionWriterConfig = GaussianSplatReconstructionWriterConfig(),
    dataset_type: Literal["colmap", "simple_directory", "e57"] = "colmap",
    run_name: str | None = None,
    log_path: pathlib.Path | None = pathlib.Path("fvdb_gslogs"),
    use_every_n_as_val: int = 8,
    device: str | torch.device = "cuda",
    log_every: int = 10,
    visualize_every: int = -1,
    verbose: bool = False,
    out_file_name: str = "chunked.ply",
):
    """
    Train a 3D Gaussian Splatting model on a dataset in chunks.
    This function divides the scene into chunks based on the specified number of divisions
    along each axis (nx, ny, nz) and trains the model on each chunk separately.

    The chunks are defined by bounding boxes that overlap by a specified percentage
    (overlap_percent) in [0, 1] to ensure continuity between chunks.

    The chunks are subsequently merged into a single model checkpoint.

    Args:
        dataset_path (pathlib.Path): Path to the dataset.
        nx (int): Number of chunks along the x-axis (after normalization).
        ny (int): Number of chunks along the y-axis (after normalization).
        nz (int): Number of chunks along the z-axis (after normalization).
        overlap_percent (float): Percentage of overlap between chunks in [0, 1].
        cfg (GaussianSplatReconstructionConfig): Configuration for the Gaussian Splat Reconstruction.
        tx (SceneTransformConfig): Configuration for the scene transforms.
        opt (GaussianSplatOptimizerConfig): Configuration for the optimizer.
        io (GaussianSplatReconstructionWriterConfig): Configuration for saving metrics and checkpoints.
        dataset_type (Literal["colmap", "simple_directory", "e57"]): Type of dataset.
        run_name (str | None): Name of the training run.
        log_path (pathlib.Path | None): Path to log metrics, and checkpoints. If None, no metrics or
            checkpoints will be saved. Default is "fvdb_gslogs".
        use_every_n_as_val (int): Use every n-th image as a validation image.
        device (str | torch.device): Device to use for training.
        log_every (int): Log training metrics every n steps.
        visualize_every (int): Update the viewer every n epochs. If -1, do not visualize.
        verbose (bool): Whether to log debug messages.
        out_file_name (str): Name of the output PLY file to save the merged model. Default is "chunked.ply".
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

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

    transform = tx.scene_transform
    sfm_scene: SfmScene = transform(sfm_scene)

    if visualize_every > 0:
        viewer = Viewer()
    else:
        viewer = None

    scene_points = sfm_scene.points

    # Compute a list of crop bounding boxes for each chunk
    # Each bounding box is a tuple of the form (xmin, ymin, zmin, xmax, ymax, zmax)
    crops_bboxes = []
    xmin, ymin, zmin = scene_points.min(axis=0)
    xmax, ymax, zmax = scene_points.max(axis=0)
    chunk_size_x = (xmax - xmin) / nx
    chunk_size_y = (ymax - ymin) / ny
    chunk_size_z = (zmax - zmin) / nz
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        # Calculate the bounding box for the current chunk
        # with overlap based on the specified percentage
        crop_bbox = (
            xmin + i * chunk_size_x - 0.5 * chunk_size_x * overlap_percent,
            ymin + j * chunk_size_y - 0.5 * chunk_size_y * overlap_percent,
            zmin + k * chunk_size_z - 0.5 * chunk_size_z * overlap_percent,
            xmin + (i + 1) * chunk_size_x + 0.5 * chunk_size_x * overlap_percent,
            ymin + (j + 1) * chunk_size_y + 0.5 * chunk_size_y * overlap_percent,
            zmin + (k + 1) * chunk_size_z + 0.5 * chunk_size_z * overlap_percent,
        )
        crops_bboxes.append(crop_bbox)

    num_chunks = len(crops_bboxes)
    logger.info(f"Total number of chunks: {num_chunks}")

    writer = GaussianSplatReconstructionWriter(run_name=run_name, save_path=log_path, config=io, exist_ok=False)

    with tempfile.TemporaryDirectory(delete=True) as ply_temp_dir:
        # TODO: Stripe accross GPUs
        chunk_ply_paths: list[pathlib.Path] = []
        for i, bbox in enumerate(crops_bboxes):
            logger.info(f"Optimizing chunk {i+1}/{num_chunks}: bbox {bbox}")
            chunk_transform = CropScene(bbox=bbox)

            scene_chunk = chunk_transform(sfm_scene)

            runner = GaussianSplatReconstruction.from_sfm_scene(
                sfm_scene=scene_chunk,
                config=cfg,
                optimizer_config=opt,
                writer=writer,
                viewer=viewer,
                use_every_n_as_val=use_every_n_as_val,
                log_interval_steps=log_every,
                viewer_update_interval_epochs=visualize_every,
                device=device,
            )
            runner.train(True, f"train_chunk_{i:04d}")

            if runner.model.num_gaussians == 0:
                logger.warning(
                    f"Chunk {i} resulted in a model with 0 gaussians. This chunk will be skipped during merging."
                )
            chunk_ply_path = pathlib.Path(ply_temp_dir) / f"chunk_{i:04d}.ply"
            runner.model.save_ply(chunk_ply_path, {})
            chunk_ply_paths.append(chunk_ply_path)

        logger.info("All chunks have been processed. Merging splats...")

        splats = []
        for ply_path in tqdm.tqdm(chunk_ply_paths):
            splat_chunk, _ = GaussianSplat3d.from_ply(ply_path, device=device)
            splats.append(splat_chunk)

        logger.info("All PLY files loaded. Merging...")
        merged_splats = GaussianSplat3d.cat(splats)
        logger.info(f"Merging completed. Saving merged checkpoint to file {out_file_name}.")

        merged_splats.save_ply(out_file_name, runner.optimization_metadata)


if __name__ == "__main__":
    tyro.cli(main)
