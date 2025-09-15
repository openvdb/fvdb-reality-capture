# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
from typing import Sequence

import numpy as np
import torch

from fvdb import GaussianSplat3d

from ..transforms import BaseTransform
from .camera_pose_adjust import CameraPoseAdjustment
from .gaussian_splat_optimizer import GaussianSplatOptimizer


class Checkpoint:
    """
    Class representing a checkpoint for scene optimization.

    A checkpoint contains data about the model, the training configuration, optimizer, and dataset metadata.

    At the very least, a checkpoint will ALWAYS contain:
    - splats (GaussianSplat3d): The Gaussian Splatting model.
    - config (dict): The configuration used for training.
    - run_name (str): The name of the run associated with this checkpoint.
    - optimizer (GaussianSplatOptimizer): The optimizer used for training.
    - num_training_images (int): The number of images used in training
    - dataset_path (pathlib.Path): The path to the dataset used for training.
    - dataset_transform (BaseTransform): The transform applied to the dataset to normalize/rescale/etc.
    - dataset_splits (dict[str, np.ndarray]): The indices of images used for train/test/val/etc. splits (e.g. train/val/test).

    The checkpoint can also OPTIONALLY contain:
    - pose_adjust_model (CameraPoseAdjustment | None): The camera pose adjustment model, if used.
    - pose_adjust_optimizer (torch.optim.Adam | None): The optimizer for the camera pose adjustment model, if used.
    - pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None): The learning rate scheduler for the camera pose adjustment optimizer, if used.
    """

    version = "1.1.0"

    def __init__(
        self,
        step: int,
        run_name: str,
        model: GaussianSplat3d,
        config: dict,
        dataset_path: str | pathlib.Path,
        dataset_transform: BaseTransform,
        dataset_splits: dict[str, Sequence[int] | torch.Tensor | np.ndarray],
        optimizer: GaussianSplatOptimizer,
        pose_adjust_model: CameraPoseAdjustment | None,
        pose_adjust_optimizer: torch.optim.Adam | None,
        pose_adjust_scheduler: torch.optim.lr_scheduler.ExponentialLR | None,
    ):
        self._step = step
        self._run_name = run_name
        self._model = model
        self._config = config
        self._optimizer = optimizer
        self._pose_adjust_model = pose_adjust_model
        self._pose_adjust_optimizer = pose_adjust_optimizer
        self._pose_adjust_scheduler = pose_adjust_scheduler
        self._dataset_path = dataset_path
        self._dataset_transform = dataset_transform

        self._dataset_splits: dict[str, np.ndarray] = {}
        for k, v in dataset_splits.items():
            if isinstance(v, torch.Tensor):
                self._dataset_splits[k] = v.cpu().numpy().astype(np.int32)
            elif isinstance(v, np.ndarray):
                self._dataset_splits[k] = v.astype(np.int32)
            else:
                self._dataset_splits[k] = np.asarray(v).astype(np.int32)

    @torch.no_grad()
    def save(self, path: pathlib.Path):
        """
        Save the checkpoint to a file.

        Args:
            path (pathlib.Path): The path to save the checkpoint file.
        """
        checkpoint_data = {
            "version": self.version,
            "step": self._step,
            "run_name": self._run_name,
            "splats": self._model.state_dict(),
            "config": self._config,
            "dataset_path": str(self._dataset_path),
            "dataset_transform": self._dataset_transform.state_dict(),
            "dataset_splits": self._dataset_splits,
            "optimizer": self._optimizer.state_dict(),
            "num_training_poses": self._pose_adjust_model.num_poses if self._pose_adjust_model else None,
            "pose_adjust_model": self._pose_adjust_model.state_dict() if self._pose_adjust_model else None,
            "pose_adjust_optimizer": self._pose_adjust_optimizer.state_dict() if self._pose_adjust_optimizer else None,
            "pose_adjust_scheduler": self._pose_adjust_scheduler.state_dict() if self._pose_adjust_scheduler else None,
        }

        torch.save(checkpoint_data, path)

    @torch.no_grad()
    @staticmethod
    def load(
        path: pathlib.Path,
        device: torch.device | str = "cpu",
        dataset_path: pathlib.Path | None = None,
    ):
        logger = logging.getLogger(f"{Checkpoint.__class__.__module__}.{Checkpoint.__class__}.load")
        logger.info(f"Loading checkpoint from {path} on device {device}...")
        checkpoint_data = torch.load(path, map_location=device, weights_only=False)

        assert "version" in checkpoint_data, "Version information is missing in the checkpoint."
        assert checkpoint_data["version"] == Checkpoint.version, (
            f"Checkpoint version {checkpoint_data['version']} is not compatible with "
            f"SceneOptimizationCheckpoint version {Checkpoint.version}."
        )

        assert "step" in checkpoint_data, "Step is missing in the checkpoint."
        step = checkpoint_data["step"]

        assert "run_name" in checkpoint_data, "Run name is missing in the checkpoint."
        run_name = checkpoint_data["run_name"]

        assert "splats" in checkpoint_data, "Model state is missing in the checkpoint."
        model = GaussianSplat3d.from_state_dict(checkpoint_data["splats"]).to(device)

        assert "config" in checkpoint_data, "Configuration is missing in the checkpoint."
        config = checkpoint_data["config"]

        if dataset_path is None:
            assert "dataset_path" in checkpoint_data, "Dataset path is missing in the checkpoint."
            dataset_path = pathlib.Path(checkpoint_data["dataset_path"])

        assert "dataset_transform" in checkpoint_data, "Dataset transform is missing in the checkpoint."
        dataset_transform = BaseTransform.from_state_dict(checkpoint_data["dataset_transform"])

        assert "dataset_splits" in checkpoint_data, "Dataset splits are missing in the checkpoint."
        dataset_splits = checkpoint_data["dataset_splits"]

        assert "optimizer" in checkpoint_data, "Optimizer state is missing in the checkpoint."
        optimizer_state = checkpoint_data["optimizer"]
        optimizer = GaussianSplatOptimizer(model)
        optimizer.load_state_dict(optimizer_state)

        assert "num_training_poses" in checkpoint_data, "Number of training poses is missing in the checkpoint."
        num_training_poses = checkpoint_data["num_training_poses"]

        assert "pose_adjust_model" in checkpoint_data, "Pose adjustment model state is missing in the checkpoint."
        pose_adjust_model_state = checkpoint_data["pose_adjust_model"]

        assert (
            "pose_adjust_optimizer" in checkpoint_data
        ), "Pose adjustment optimizer state is missing in the checkpoint."
        pose_adjust_optimizer_state = checkpoint_data["pose_adjust_optimizer"]

        assert (
            "pose_adjust_scheduler" in checkpoint_data
        ), "Pose adjustment scheduler state is missing in the checkpoint."
        pose_adjust_scheduler_state = checkpoint_data["pose_adjust_scheduler"]

        pose_adjust_model = None
        pose_adjust_optimizer = None
        pose_adjust_scheduler = None
        if num_training_poses is not None and pose_adjust_model_state is not None:
            assert pose_adjust_optimizer_state is not None, "Pose adjustment optimizer state is missing."
            assert pose_adjust_scheduler_state is not None, "Pose adjustment scheduler state is missing."

            pose_adjust_model = CameraPoseAdjustment(num_poses=num_training_poses).to(device)
            pose_adjust_model.load_state_dict(pose_adjust_model_state)

            pose_adjust_optimizer = torch.optim.Adam(
                pose_adjust_model.parameters(),
                lr=config["pose_opt_lr"] * 100.0,  # Scale the learning rate for pose optimization
                weight_decay=config["pose_opt_reg"],
            )
            pose_adjust_optimizer.load_state_dict(pose_adjust_optimizer_state)
            pose_adjust_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                pose_adjust_optimizer, gamma=config["pose_opt_lr_decay"]
            )
            pose_adjust_scheduler.load_state_dict(pose_adjust_scheduler_state)

        return Checkpoint(
            step=step,
            run_name=run_name,
            model=model,
            config=config,
            dataset_path=dataset_path,
            dataset_transform=dataset_transform,
            dataset_splits=dataset_splits,
            optimizer=optimizer,
            pose_adjust_model=pose_adjust_model,
            pose_adjust_optimizer=pose_adjust_optimizer,
            pose_adjust_scheduler=pose_adjust_scheduler,
        )

    @property
    def splats(self) -> GaussianSplat3d:
        """
        Get the Gaussian Splatting model from the checkpoint.

        Returns:
            GaussianSplat3d: The Gaussian Splatting model.
        """
        return self._model

    @property
    def config(self) -> dict:
        """
        Get the configuration used for training from the checkpoint.

        Returns:
            Config: The configuration used for training.
        """
        return self._config

    @property
    def run_name(self) -> str:
        """
        Get the name of the run associated with this checkpoint.

        Returns:
            str: The name of the run.
        """
        return self._run_name

    @property
    def step(self) -> int:
        """
        Get the training step at which the checkpoint was created.

        Returns:
            int: The training step.
        """
        return self._step

    @property
    def optimizer(self) -> GaussianSplatOptimizer:
        """
        Get the optimizer used for training from the checkpoint.

        Returns:
            GaussianSplatOptimizer: The optimizer used for training.
        """
        return self._optimizer

    @property
    def pose_adjust_model(self) -> CameraPoseAdjustment | None:
        """
        Get the camera pose adjustment model from the checkpoint.

        Returns:
            CameraPoseAdjustment | None: The camera pose adjustment model, or None if not present.
        """
        return self._pose_adjust_model if self._pose_adjust_model is not None else None

    @property
    def pose_adjust_optimizer(self) -> torch.optim.Adam | None:
        """
        Get the camera pose adjustment optimizer from the checkpoint.

        Returns:
            torch.optim.Adam | None: The camera pose adjustment optimizer, or None if not present.
        """
        return self._pose_adjust_optimizer if self._pose_adjust_optimizer is not None else None

    @property
    def pose_adjust_scheduler(self) -> torch.optim.lr_scheduler.ExponentialLR | None:
        """
        Get the camera pose adjustment optimizer from the checkpoint if present.

        Returns:
            torch.optim.Adam | None: The camera pose adjustment optimizer, or None if not present.
        """
        return self._pose_adjust_scheduler if self._pose_adjust_scheduler is not None else None

    @property
    def dataset_path(self) -> pathlib.Path:
        """
        Get the path to the dataset.

        Returns:
            pathlib.Path: The path to the data used to train this checkpoint
        """
        return pathlib.Path(self._dataset_path)

    @property
    def dataset_transform(self) -> BaseTransform:
        """
        Get the transform for this dataset.

        Returns:
            BaseTransform: The transform used with this dataset.
        """
        return self._dataset_transform

    @property
    def dataset_splits(self) -> dict[str, np.ndarray]:
        """
        Get the dataset splits used for training from the checkpoint.

        Returns:
            dict[str, np.ndarray]: The dataset splits used for training.
        """
        return self._dataset_splits
