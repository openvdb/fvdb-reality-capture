# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Annotated

import torch
from tyro.conf import Positional, arg

from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.instance_segmentation import (
    GARfVDBTrainer,
    GARfVDBTrainingConfig,
    GARfVDBTransformConfig,
)
from fvdb_reality_capture.instance_segmentation.training.segmentation_writer import (
    GARfVDBWriter,
    GARfVDBWriterConfig,
)

from ._common import DatasetType, load_sfm_scene, load_splats_from_file


@dataclass
class InstanceSegmentationWriterConfig(GARfVDBWriterConfig):
    """Configure GARfVDB metrics, images, and training checkpoints."""

    log_path: pathlib.Path | None = pathlib.Path("frgs_logs")
    """Base directory for run logs. Set to ``None`` to disable disk logging."""

    log_every: int = 10
    """Log metrics every N training steps."""


@dataclass
class InstanceSegmentation(BaseCommand):
    """Train a GARfVDB scale-conditioned instance feature field from an existing reconstruction.

    Example:
        frgs instance-segmentation ./colmap_dataset \
            --reconstruction-path scene.ply \
            --out-path scene.garfvdb
    """

    dataset_path: Positional[pathlib.Path]
    """Dataset containing the images and camera poses used by the reconstruction."""

    reconstruction_path: Annotated[pathlib.Path, arg(aliases=["-r"])]
    """Existing Reality Capture PLY or reconstruction checkpoint."""

    out_path: Annotated[pathlib.Path, arg(aliases=["-o"])] = pathlib.Path("out.garfvdb")
    """Portable output bundle. The path must end in ``.garfvdb``."""

    run_name: Annotated[str | None, arg(aliases=["-n"])] = None
    dataset_type: Annotated[DatasetType, arg(aliases=["-dt"])] = "colmap"
    use_every_n_as_val: Annotated[int, arg(aliases=["-vn"])] = -1
    device: Annotated[str | torch.device, arg(aliases=["-d"])] = "cuda"
    verbose: Annotated[bool, arg(aliases=["-v"])] = False
    cache_dataset: bool = True

    cfg: GARfVDBTrainingConfig = field(default_factory=GARfVDBTrainingConfig)
    tx: GARfVDBTransformConfig = field(default_factory=GARfVDBTransformConfig)
    io: InstanceSegmentationWriterConfig = field(default_factory=InstanceSegmentationWriterConfig)

    def execute(self) -> None:
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format="%(levelname)s : %(message)s",
        )
        logger = logging.getLogger(__name__)
        if self.out_path.suffix != ".garfvdb":
            raise ValueError("GARfVDB output path must end in .garfvdb")
        if self.out_path.exists():
            raise FileExistsError(f"Output path already exists: {self.out_path}")
        if not self.reconstruction_path.exists():
            raise FileNotFoundError(f"Reconstruction does not exist: {self.reconstruction_path}")
        if not self.cfg.model.use_grid:
            raise ValueError("frgs instance-segmentation requires --cfg.model.use-grid")

        logger.info("Loading dataset from %s", self.dataset_path)
        sfm_scene = load_sfm_scene(self.dataset_path, self.dataset_type)
        logger.info("Loading Gaussian carrier from %s", self.reconstruction_path)
        carrier, metadata = load_splats_from_file(self.reconstruction_path, self.device)
        normalization_transform = metadata.get("normalization_transform")
        if normalization_transform is None:
            raise ValueError(
                "Reconstruction metadata does not contain normalization_transform. "
                "Use a PLY or checkpoint produced by fvdb-reality-capture."
            )

        self.tx.device = self.device
        transformed_scene = self.tx.build_scene_transforms(carrier, normalization_transform)(sfm_scene)
        writer = GARfVDBWriter(
            run_name=self.run_name,
            save_path=self.io.log_path,
            config=self.io,
            exist_ok=False,
        )
        trainer = GARfVDBTrainer.new(
            sfm_scene=transformed_scene,
            gs_model=carrier,
            gs_model_path=self.reconstruction_path,
            writer=writer,
            config=self.cfg,
            device=self.device,
            use_every_n_as_val=self.use_every_n_as_val,
            viewer_update_interval_epochs=-1,
            log_interval_steps=self.io.log_every,
            cache_dataset=self.cache_dataset,
            reconstruction_metadata=metadata,
        )
        trainer.train()
        logger.info("Saving portable GARfVDB bundle to %s", self.out_path)
        trainer.to_product().save(self.out_path)


__all__ = ["InstanceSegmentation", "InstanceSegmentationWriterConfig"]
