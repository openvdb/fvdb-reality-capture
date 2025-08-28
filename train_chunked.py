# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import logging
import pathlib
import time
from functools import partial
from multiprocessing import Pool
from typing import Literal, Sequence

import numpy as np
import torch
import tqdm
import tyro
from fvdb_3dgs.io import DatasetCache, load_colmap_dataset
from fvdb_3dgs.sfm_scene import SfmScene
from fvdb_3dgs.training import Config, SceneOptimizationRunner, SfmDataset
from fvdb_3dgs.transforms import (
    Compose,
    DownsampleImages,
    NormalizeScene,
    PercentileFilterPoints,
)
from fvdb_3dgs.viewer import Viewer

from fvdb import GaussianSplat3d


def _make_unique_name_directory_based_on_time(results_base_path: pathlib.Path, prefix: str) -> tuple[str, pathlib.Path]:
    """
    Generate a unique name and directory based on the current time.

    The run directory will be created under `results_base_path` with a name in the format
    `prefix_YYYY-MM-DD-HH-MM-SS`. If a directory with the same name already exists,
    it will attempt to create a new one by appending an incremented number to

    Returns:
        run_name: A unique run name in the format "run_YYYY-MM-DD-HH-MM-SS".
        run_path: A pathlib.Path object pointing to the created directory.
    """
    attempts = 0
    max_attempts = 50
    run_name = f"{prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    logger = logging.getLogger(__name__)
    while attempts < 50:
        results_path = results_base_path / run_name
        try:
            results_path.mkdir(exist_ok=False, parents=True)
            break
        except FileExistsError:
            attempts += 1
            logger.debug(f"Directory {results_path} already exists. Attempting to create a new one.")
            # Generate a new run name with an incremented attempt number
            run_name = f"{prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}_{attempts+1:02d}"
            continue
    if attempts >= max_attempts:
        raise FileExistsError(f"Failed to generate a unique results directory name after {max_attempts} attempts.")

    logger.info(f"Creating unique directory with name {run_name} after {attempts} attempts.")

    return run_name, results_path


def _run_on_chunk(
    chunk_id: int,
    chunk_bboxes: Sequence[tuple[float, float, float, float, float, float]],
    dataset_path: pathlib.Path,
    cfg: Config,
    chunk_run_name_prefix: str,
    image_downsample_factor: int,
    points_percentile_filter: float,
    normalization_type: Literal["none", "pca", "ecef2enu", "similarity"],
    results_path: pathlib.Path,
    device: str | torch.device,
    use_every_n_as_val: int,
    log_tensorboard_every: int,
    log_images_to_tensorboard: bool,
    save_results: bool,
    save_eval_images: bool,
):
    """
    Train a 3D Gaussian Splatting model on a specific chunk of the dataset.
    This function initializes a SceneOptimizationRunner for the specified chunk,
    sets up the training configuration, and starts the training process.

    Args:
        chunk_id (int): The ordinal of the chunk to train on in [0, len(chunk_bboxes) - 1].
        chunk_bboxes (Sequence[tuple[float, float, float, float, float, float]]): A sequence of bounding boxes for each chunk.
            Each bounding box is defined as a tuple of the form (xmin, ymin, zmin, xmax, ymax, zmax).
        dataset_path (pathlib.Path): Path to the dataset directory containing the SFM data.
        cfg (Config): Configuration for the training process.
        chunk_run_name_prefix (str): Prefix for the run name of the chunk.
        image_downsample_factor (int): Factor by which to downsample images for training.
        points_percentile_filter (float): Percentile filter to apply to the points in the scene.
        normalization_type (Literal["none", "pca", "ecef2enu", "similarity"]): Type of normalization to apply to the scene.
            Options are "none", "pca", "ecef2enu", or "similarity".
        results_path (pathlib.Path): Path to save the results of the training.
        device (str | torch.device): Device to run the training on, e.g., "cuda" or "cpu".
        use_every_n_as_val (int): Use every n-th image as validation data.
        log_tensorboard_every (int): Log to TensorBoard every n iterations.
        log_images_to_tensorboard (bool): Whether to log images to TensorBoard.
        save_results (bool): Whether to save the results of the training.
        save_eval_images (bool): Whether to save evaluation images during training.
    """
    runner = SceneOptimizationRunner.new_run(
        config=cfg,
        dataset_path=dataset_path,
        run_name=chunk_run_name_prefix + f"_chunk_{chunk_id:04d}",
        image_downsample_factor=image_downsample_factor,
        points_percentile_filter=points_percentile_filter,
        normalization_type=normalization_type,
        crop_bbox=chunk_bboxes[chunk_id],
        results_path=results_path,
        device=device,
        use_every_n_as_val=use_every_n_as_val,
        disable_viewer=False,
        log_tensorboard_every=log_tensorboard_every,
        log_images_to_tensorboard=log_images_to_tensorboard,
        save_results=save_results,
        save_eval_images=save_eval_images,
    )

    runner.train()


def plot_chunk_bboxes(
    chunk_bboxes: Sequence[tuple[float, float, float, float, float, float]],
    scene_points: np.ndarray,
    full_train_dataset: SfmDataset,
):
    """
    Debug utility to visualize the chunk bounding boxes and scene points in a viewer.
    This function will open a viewer and display the scene points and chunk bounding boxes.

    Args:
        chunk_bboxes: A sequence of bounding boxes for each chunk of the form
            (xmin, ymin, zmin, xmax, ymax, zmax).
        scene_points: The points in the scene to visualize.
        full_train_dataset: The full training dataset containing the points and their RGB colors.
    """

    logger = logging.getLogger("train_chunked")

    viewer = Viewer()
    server = viewer.viser_server

    scene_points = full_train_dataset.points
    scene_points_rgb = full_train_dataset.points_rgb

    xmin, ymin, zmin = scene_points.min(axis=0)
    xmax, ymax, zmax = scene_points.max(axis=0)

    points = server.scene.add_point_cloud("scene points", points=scene_points, colors=scene_points_rgb)
    points.point_size = 0.001
    box_origin = np.array([xmin, ymin, zmin])
    box_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    box_center = box_origin + 0.5 * box_size
    box = server.scene.add_box(
        f"bbox",
        position=box_center,
        dimensions=box_size,
        color=(1.0, 0.0, 0.0),
    )
    box.wireframe = True

    for i, chunk_bbox in enumerate(chunk_bboxes):
        crop_origin = np.array(chunk_bbox[:3])
        crop_size = np.array(chunk_bbox[3:]) - np.array(chunk_bbox[:3])
        crop_center = crop_origin + 0.5 * crop_size
        color = np.random.rand(3) / 2.0 + 0.5
        crop_box = server.scene.add_box(
            f"chunk_{i}",
            position=crop_center,
            dimensions=crop_size,
            color=color,
        )
        crop_box.wireframe = True

    logger.info("Viewer is running. Press Ctrl+C to exit.")
    time.sleep(10000000)  # Keep the viewer open for a while


def main(
    dataset_path: pathlib.Path,
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    overlap_percent: float = 0.1,
    cfg: Config = Config(remove_gaussians_outside_scene_bbox=True),
    run_name: str | None = None,
    image_downsample_factor: int = 4,
    points_percentile_filter: float = 0.0,
    normalization_type: Literal["none", "pca", "ecef2enu", "similarity"] = "pca",
    results_path: pathlib.Path = pathlib.Path("results"),
    device: str | torch.device = "cuda",
    use_every_n_as_val: int = 8,
    log_tensorboard_every: int = 100,
    log_images_to_tensorboard: bool = False,
    save_results: bool = True,
    save_eval_images: bool = False,
):
    """
    Train a 3D Gaussian Splatting model on a dataset in chunks.
    This function divides the scene into chunks based on the specified number of divisions
    along each axis (nx, ny, nz) and trains the model on each chunk separately.

    The chunks are defined by bounding boxes that overlap by a specified percentage
    (overlap_percent) in [0, 1] to ensure continuity between chunks.

    The chunks are subsequently merged into a single model checkpoint.

    Args:
        dataset_path (pathlib.Path): Path to the dataset directory containing the SFM data.
        nx (int): Number of divisions along the x-axis.
        ny (int): Number of divisions along the y-axis.
        nz (int): Number of divisions along the z-axis.
        overlap_percent (float): Percentage of overlap between chunks in [0, 1].
        cfg (Config): Configuration for the training process.
        run_name (str | None): Name of the run. If None, a unique name will be generated based on the current time.
        image_downsample_factor (int): Factor by which to downsample images for training.
        points_percentile_filter (float): Percentile filter to apply to the points in the scene.
        normalization_type (Literal["none", "pca", "ecef2enu", "similarity"]): Type of normalization to apply to the scene.
            Options are "none", "pca", "ecef2enu", or "similarity".
        results_path (pathlib.Path): Path to save the results of the training.
        device (str | torch.device): Device to run the training on, e.g., "cuda" or "cpu".
        use_every_n_as_val (int): Use every n-th image as validation data.
        log_tensorboard_every (int): Log to TensorBoard every n iterations.
        log_images_to_tensorboard (bool): Whether to log images to TensorBoard.
        save_results (bool): Whether to save the results of the training.
        save_eval_images (bool): Whether to save evaluation images during training.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    logger = logging.getLogger("train_chunked")

    transform = Compose(
        NormalizeScene(normalization_type=normalization_type),
        PercentileFilterPoints(
            percentile_min=np.full((3,), points_percentile_filter),
            percentile_max=np.full((3,), 100.0 - points_percentile_filter),
        ),
        DownsampleImages(
            image_downsample_factor=image_downsample_factor,
        ),
    )

    sfm_scene: SfmScene
    cache: DatasetCache
    sfm_scene, cache = load_colmap_dataset(dataset_path)
    sfm_scene, cache = transform(sfm_scene, cache)

    indices = np.arange(sfm_scene.num_images)
    mask = np.ones(len(indices), dtype=bool)
    mask[::use_every_n_as_val] = False
    train_indices = indices[mask]

    full_train_dataset = SfmDataset(sfm_scene, train_indices)

    scene_points = full_train_dataset.points

    xmin, ymin, zmin = scene_points.min(axis=0)
    xmax, ymax, zmax = scene_points.max(axis=0)

    crops_bboxes = []

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

    if run_name is None:
        chunk_run_name, chunk_results_path = _make_unique_name_directory_based_on_time(
            results_path, prefix=run_name or "chunked_run"
        )
    else:
        chunk_run_name = run_name
        chunk_results_path = results_path / run_name
        if chunk_results_path.exists():
            raise FileExistsError(
                f"Results path {chunk_results_path} already exists. Please choose a different run name."
            )

    # TODO: Stripe accross GPUs
    # ngpus = len(devices)
    # dev_cycle = itertools.cycle(devices)
    # devices_per_cluster = [next(dev_cycle) for _ in range(num_clusters)]
    #  pool = Pool(ngpus)
    # for _ in pool.imap_unordered(partial_function, clusters):
    #    pass

    chunk_runner_partial = partial(
        _run_on_chunk,
        chunk_bboxes=crops_bboxes,
        dataset_path=dataset_path,
        cfg=cfg,
        chunk_run_name_prefix=chunk_run_name,
        image_downsample_factor=image_downsample_factor,
        points_percentile_filter=points_percentile_filter,
        normalization_type=normalization_type,
        results_path=chunk_results_path,
        device=device,
        use_every_n_as_val=use_every_n_as_val,
        log_tensorboard_every=log_tensorboard_every,
        log_images_to_tensorboard=log_images_to_tensorboard,
        save_results=save_results,
        save_eval_images=save_eval_images,
    )

    for chunk_id in range(num_chunks):
        logger.info(f"Starting training for chunk {chunk_id + 1}/{num_chunks}")
        chunk_runner_partial(chunk_id)
        logger.info(f"Finished training for chunk {chunk_id + 1}/{num_chunks}")

    logger.info("All chunks have been processed. Merging splats...")
    run_names = [f"{chunk_run_name}_chunk_{i:04d}" for i in range(num_chunks)]
    splats = []
    for chunk_id in tqdm.tqdm(range(num_chunks)):
        ply_path = chunk_results_path / run_names[chunk_id] / "checkpoints" / "ckpt_final.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file {ply_path} does not exist.")
        logger.info(f"Loading PLY for chunk {chunk_id + 1}/{num_chunks} from {ply_path}")
        splat_chunk, _ = GaussianSplat3d.from_ply(ply_path, device=device)
        splats.append(splat_chunk)

    logger.info("All PLY files loaded. Merging...")
    merged_splats = GaussianSplat3d.cat(splats)
    logger.info(f"Merging completed. Saving merged checkpoint to {chunk_results_path / 'merged.ply'}")

    out_ply_path = chunk_results_path / "merged.ply"
    normalization_transform = torch.from_numpy(full_train_dataset.sfm_scene.transformation_matrix).to(torch.float32)
    training_camera_to_world_matrices = torch.from_numpy(full_train_dataset.camera_to_world_matrices).to(torch.float32)
    training_projection_matrices = torch.from_numpy(full_train_dataset.projection_matrices).to(torch.float32)
    image_sizes = torch.from_numpy(full_train_dataset.image_sizes).to(torch.int32)

    training_metadata = {
        "normalization_transform": normalization_transform,
        "camera_to_world_matrices": training_camera_to_world_matrices,
        "projection_matrices": training_projection_matrices,
        "image_sizes": image_sizes,
    }
    merged_splats.save_ply(out_ply_path, training_metadata)


if __name__ == "__main__":
    tyro.cli(main)
