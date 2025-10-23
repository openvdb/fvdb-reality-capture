# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import itertools
import logging
import pathlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import fvdb.viz as fviz
import numpy as np
import torch
import tqdm
from fvdb import GaussianSplat3d
from tyro.conf import Positional, arg

from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.radiance_fields import (
    GaussianSplatOptimizerConfig,
    GaussianSplatReconstruction,
    GaussianSplatReconstructionConfig,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)
from fvdb_reality_capture.sfm_scene import SfmScene
from fvdb_reality_capture.transforms import (
    BaseTransform,
    Compose,
    CropScene,
    CropSceneToPoints,
    DownsampleImages,
    FilterImagesWithLowPoints,
    NormalizeScene,
    PercentileFilterPoints,
)

from ._common import (
    DatasetType,
    load_sfm_scene,
    save_model_from_runner,
    save_model_from_splats,
)


@dataclass
class SceneTransformConfig:
    """
    Configure how an SfmScene is transformed before optimization.
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
    # Minimum number of 3D points that must be visible in an image for it to be included in the optimization
    min_points_per_image: int = 5
    # Bounding box to which we crop the scene (in the original space) (xmin, xmax, ymin, ymax, zmin, zmax)
    crop_bbox: tuple[float, float, float, float, float, float] | None = None

    @property
    def scene_transform(self) -> BaseTransform:
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


@dataclass
class WriterConfig(GaussianSplatReconstructionWriterConfig):
    """
    Configuration for saving and logging metrics, images, and checkpoints.
    """

    # Path to save logs, checkpoints, and other output to.
    # Defaults to `frgs_logs` in the current working directory.
    log_path: pathlib.Path | None = pathlib.Path("frgs_logs")

    # How frequently to log metrics during reconstruction.
    log_every: int = 10


@dataclass
class Reconstruct(BaseCommand):
    """
    Reconstruct a Gaussian Splat Radiance Field from a dataset of posed images, and save the result as a PLY or USDZ file.


    Example usage:

        # Reconstruct a Gaussian splat radiance field from a Colmap dataset
        frgs reconstruct ./colmap_dataset -o ./output.ply

        # Reconstruct a Gaussian splat radiance field from a dataset of e57 files
        frgs reconstruct ./simple_directory_dataset --dataset-type e57 --out-path ./output.usdz
    """

    # Path to the dataset. For "colmap" datasets, this should be the
    # directory containing the `images` and `sparse` subdirectories. For "simple_directory" datasets,
    # this should be the directory containing the images and a `cameras.txt` file.
    dataset_path: Positional[Path]

    # Path to save the output PLY file.
    # Defaults to `out.ply` in the current working directory.
    # Path must end in .ply or .usdz.
    out_path: Annotated[Path, arg(aliases=["-o"])] = Path("out.ply")

    # Name of the run. If None, a name will be generated based on the current date and time.
    run_name: Annotated[str | None, arg(aliases=["-n"])] = None

    # Type of dataset to load.
    dataset_type: Annotated[DatasetType, arg(aliases=["-dt"])] = "colmap"

    # Use every n-th image as a validation image. If -1, do not use a validation set.
    use_every_n_as_val: Annotated[int, arg(aliases=["-vn"])] = -1

    # How frequently (in epochs) to update the viewer during reconstruction.
    # An epoch is one full pass through the dataset. If -1, do not visualize.
    update_viz_every: Annotated[float, arg(aliases=["-uv"])] = -1.0

    # The port to expose the viewer server on if update_viz_every > 0.
    viewer_port: Annotated[int, arg(aliases=["-p"])] = 8080

    # The IP address to expose the viewer server on if update_viz_every > 0.
    viewer_ip_address: Annotated[str, arg(aliases=["-ip"])] = "127.0.0.1"

    # Which device to use for reconstruction. Must be a cuda device. You can pass in a specific device index via
    # cuda:N where N is the device index, or "cuda" to use the default cuda device.
    # CPU is not supported. Default is "cuda".
    device: Annotated[str | torch.device, arg(aliases=["-d"])] = "cuda"

    # If set, show verbose debug messages.
    verbose: Annotated[bool, arg(aliases=["-v"])] = False

    # Configuration parameters for the Gaussian splat reconstruction.
    cfg: GaussianSplatReconstructionConfig = field(default_factory=GaussianSplatReconstructionConfig)

    # Configuration for the transforms to apply to the scene before reconstruction.
    tx: SceneTransformConfig = field(default_factory=SceneTransformConfig)

    # Configuration for the optimizer used to reconstruct the Gaussian splat radiance field.
    opt: GaussianSplatOptimizerConfig = field(default_factory=GaussianSplatOptimizerConfig)

    # Configure saving and logging metrics, images, and checkpoints.
    io: WriterConfig = field(default_factory=WriterConfig)

    # Configuration to split the dataset into chunks for reconstruction.
    # If set to (1, 1, 1), the dataset will not be chunked.
    nchunks: Annotated[tuple[int, int, int], arg(aliases=["-nc"])] = (1, 1, 1)

    # Percentage of overlap between chunks if reconstructing in chunks. Must be in [0, 1].
    # Only used if nchunks is not (1, 1, 1).
    # Default is 0.1 (10% overlap).
    chunk_overlap_pct: Annotated[float, arg(aliases=["-nco"])] = 0.1

    def get_crop_bboxes(self, sfm_scene: SfmScene) -> list[tuple[float, float, float, float, float, float]]:
        """
        Compute a list of crop bounding boxes for each chunk.

        Each bounding box is a tuple of the form (xmin, ymin, zmin, xmax, ymax, zmax).

        Args:
            sfm_scene (SfmScene): The SfM scene to compute the bounding boxes for.
        Returns:
            list[tuple[float, float, float, float, float, float]]: List of crop bounding boxes for each chunk.
        """
        scene_points = sfm_scene.points
        nx, ny, nz = self.nchunks
        overlap_percent = self.chunk_overlap_pct

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
                float(xmin + i * chunk_size_x - 0.5 * chunk_size_x * overlap_percent),
                float(ymin + j * chunk_size_y - 0.5 * chunk_size_y * overlap_percent),
                float(zmin + k * chunk_size_z - 0.5 * chunk_size_z * overlap_percent),
                float(xmin + (i + 1) * chunk_size_x + 0.5 * chunk_size_x * overlap_percent),
                float(ymin + (j + 1) * chunk_size_y + 0.5 * chunk_size_y * overlap_percent),
                float(zmin + (k + 1) * chunk_size_z + 0.5 * chunk_size_z * overlap_percent),
            )
            crops_bboxes.append(crop_bbox)
        return crops_bboxes

    def _run_chunked_reconstruction(
        self,
        sfm_scene: SfmScene,
        writer: GaussianSplatReconstructionWriter,
        viz_scene: fviz.Scene | None,
    ):
        """
        Reconstruct the scene in chunks and merge the results.
        The chunks are defined by self.nchunks and self.chunk_overlap_pct, and
        saved as intermediate PLY files in a temporary directory prior to merging.

        Args:
            sfm_scene (SfmScene): The SfM scene to reconstruct.
            writer (GaussianSplatReconstructionWriter): Writer to use for logging and saving metrics.
            viz_scene (fviz.Scene | None): :class:`fviz.Scene` to use for visualization. If ``None``, no visualization will be done.

        """
        crop_bboxes = self.get_crop_bboxes(sfm_scene)
        num_chunks = len(crop_bboxes)

        with tempfile.TemporaryDirectory(delete=True) as ply_temp_dir:
            # TODO: Stripe accross GPUs
            chunk_ply_paths: list[pathlib.Path] = []
            for i, bbox in enumerate(crop_bboxes):
                self.logger.info(f"Reconstructing chunk {i+1}/{num_chunks}: bbox {bbox}")
                chunk_transform = CropScene(bbox=bbox)

                scene_chunk = chunk_transform(sfm_scene)
                # This is important to ensure that gaussians are not
                # initialized outside the chunk bbox
                self.cfg.remove_gaussians_outside_scene_bbox = True

                runner = GaussianSplatReconstruction.from_sfm_scene(
                    sfm_scene=scene_chunk,
                    config=self.cfg,
                    optimizer_config=self.opt,
                    writer=writer,
                    viz_scene=viz_scene,
                    use_every_n_as_val=self.use_every_n_as_val,
                    log_interval_steps=self.io.log_every,
                    viz_update_interval_epochs=self.update_viz_every,
                    device=self.device,
                )
                runner.optimize(True, f"recon_chunk_{i:04d}")

                if runner.model.num_gaussians == 0:
                    self.logger.warning(
                        f"Chunk {i} resulted in a model with 0 gaussians. This chunk will be skipped during merging."
                    )
                chunk_ply_path = pathlib.Path(ply_temp_dir) / f"chunk_{i:04d}.ply"
                runner.model.save_ply(chunk_ply_path, {})
                chunk_ply_paths.append(chunk_ply_path)

            self.logger.info("All chunks have been processed. Merging splats...")

            splats = []
            for ply_path in tqdm.tqdm(chunk_ply_paths):
                # TODO: Do we want to do this on the CPU instead?
                splat_chunk, _ = GaussianSplat3d.from_ply(ply_path, device=self.device)
                splats.append(splat_chunk)

            self.logger.info("All chunks files loaded. Merging...")
            merged_splats = GaussianSplat3d.cat(splats)

            self.logger.info(f"Saving merged model to {self.out_path}")
            save_model_from_splats(self.out_path, merged_splats, runner.reconstruction_metadata)

    def _run_single_reconstruction(
        self,
        sfm_scene: SfmScene,
        writer: GaussianSplatReconstructionWriter,
        viz_scene: fviz.Scene | None,
    ):
        """
        Reconstruct a single scene and save as a PLY or USDZ file.

        Args:
            sfm_scene (SfmScene): The SfM scene to reconstruct.
            writer (GaussianSplatReconstructionWriter): Writer to use for logging and saving metrics.
            viz_scene (fviz.Scene | None): :class:`fviz.Scene` to use for visualization. If ``None``, no visualization will be done.
        """
        runner = GaussianSplatReconstruction.from_sfm_scene(
            sfm_scene,
            config=self.cfg,
            optimizer_config=self.opt,
            writer=writer,
            viz_scene=viz_scene,
            use_every_n_as_val=self.use_every_n_as_val,
            log_interval_steps=self.io.log_every,
            viz_update_interval_epochs=self.update_viz_every,
            device=self.device,
        )

        runner.optimize()

        self.logger.info(f"Saving final model to {self.out_path}")
        save_model_from_runner(self.out_path, runner)

    def execute(self) -> None:
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
        self.logger = logging.getLogger(__name__)

        if self.out_path.suffix.lower() not in [".ply", ".usdz"]:
            raise ValueError("Output path must end in .ply or .usdz")
        if self.out_path.exists():
            raise ValueError(f"Output path {self.out_path} already exists")

        self.logger.info(f"Loading dataset from {self.dataset_path}")
        sfm_scene = load_sfm_scene(self.dataset_path, self.dataset_type)

        if self.update_viz_every > 0:
            self.logger.info(f"Starting viewer server on {self.viewer_ip_address}:{self.viewer_port}")
            fviz.init(ip_address=self.viewer_ip_address, port=self.viewer_port, verbose=self.verbose)
            viz_scene = fviz.get_scene("Gaussian Splat Reconstruction Visualization")
        else:
            viz_scene = None

        scene_transform: BaseTransform = self.tx.scene_transform
        sfm_scene: SfmScene = scene_transform(sfm_scene)

        writer = GaussianSplatReconstructionWriter(
            run_name=self.run_name,
            save_path=self.io.log_path,
            config=self.io,
            exist_ok=False,
        )

        if self.nchunks[0] < 1 or self.nchunks[1] < 1 or self.nchunks[2] < 1:
            raise ValueError("nchunks must be a tuple of 3 positive integers")
        if self.chunk_overlap_pct < 0.0 or self.chunk_overlap_pct >= 1.0:
            raise ValueError("chunk_overlap_pct must be in the range [0, 1)")

        if self.nchunks == (1, 1, 1):
            self._run_single_reconstruction(sfm_scene, writer, viz_scene)
        else:
            self._run_chunked_reconstruction(sfm_scene, writer, viz_scene)
