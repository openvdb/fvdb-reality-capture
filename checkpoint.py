# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
from typing import Any

import torch
from camera_pose_adjust import CameraPoseAdjustment
from datasets import SfmDataset
from fvdb.optim import GaussianSplatOptimizer

from fvdb import GaussianSplat3d


class Checkpoint:
    __PRIVATE__ = object()

    def __init__(
        self,
        step: int,
        run_name: str | None,
        model: GaussianSplat3d,
        optimizer: GaussianSplatOptimizer,
        config: dict,
        train_dataset: SfmDataset | None,
        eval_dataset: SfmDataset | None,
        initial_training_poses: torch.Tensor | None,
        final_training_poses: torch.Tensor | None,
        training_projection_matrices: torch.Tensor | None,
        training_image_sizes: torch.Tensor | None,
        eval_poses: torch.Tensor | None,
        eval_projection_matrices: torch.Tensor | None,
        eval_image_sizes: torch.Tensor | None,
        pose_adjust_model: CameraPoseAdjustment | None,
        pose_adjust_optimizer: torch.optim.Adam | None,
        pose_adjust_scheduler: torch.optim.lr_scheduler.ExponentialLR | None,
        _private: Any = None,
    ):
        """
        Create a checkpoint for the model, optimizer, and configuration.

        Note: Do not call this constructor directly. Use the `make_checkpoint` or `load` methods instead.

        Args:
            step (int): The training step at which the checkpoint is created.
            model (GaussianSplat3d): The Gaussian Splatting model.
            optimizer (GaussianSplatOptimizer): The optimizer used for training.
            config (Config): The configuration used for training.
        """
        if _private is not Checkpoint.__PRIVATE__:
            raise ValueError("Checkpoint can only be initialized through the `load` or `make_checkpoint` fucntions.")

        self._step = step
        self._model = model
        self._optimizer = optimizer
        self._config = config

        self._pose_adjust_model = pose_adjust_model
        self._pose_adjust_optimizer = pose_adjust_optimizer
        self._pose_adjust_scheduler = pose_adjust_scheduler

        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset

        self._initial_training_poses = initial_training_poses
        self._final_training_poses = final_training_poses
        self._training_projection_matrices = training_projection_matrices
        self._training_image_sizes = training_image_sizes

        self._eval_poses = eval_poses
        self._eval_projection_matrices = eval_projection_matrices
        self._eval_image_sizes = eval_image_sizes
        self._run_name = run_name

    @property
    def step(self) -> int:
        """
        Get the training step at which the checkpoint was created.

        Returns:
            int: The training step.
        """
        return self._step

    @property
    def splats(self) -> GaussianSplat3d:
        """
        Get the Gaussian Splatting model from the checkpoint.

        Returns:
            GaussianSplat3d: The Gaussian Splatting model.
        """
        return self._model

    @property
    def optimizer(self) -> GaussianSplatOptimizer:
        """
        Get the optimizer used for training from the checkpoint.

        Returns:
            GaussianSplatOptimizer: The optimizer used for training.
        """
        return self._optimizer

    @property
    def config(self) -> dict:
        """
        Get the configuration used for training from the checkpoint.

        Returns:
            Config: The configuration used for training.
        """
        return self._config

    @property
    def pose_adjust_model(self) -> CameraPoseAdjustment | None:
        """
        Get the camera pose adjustment model from the checkpoint.

        Returns:
            CameraPoseAdjustment | None: The camera pose adjustment model, or None if not present.
        """
        return self._pose_adjust_model

    @property
    def pose_adjust_optimizer(self) -> torch.optim.Adam | None:
        """
        Get the camera pose adjustment optimizer from the checkpoint.

        Returns:
            torch.optim.Adam | None: The camera pose adjustment optimizer, or None if not present.
        """
        return self._pose_adjust_optimizer

    @property
    def pose_adjust_scheduler(self) -> torch.optim.lr_scheduler.ExponentialLR | None:
        """
        Get the camera pose adjustment learning rate scheduler from the checkpoint.

        Returns:
            torch.optim.lr_scheduler.ExponentialLR | None: The camera pose adjustment learning rate scheduler, or None if not present.
        """
        return self._pose_adjust_scheduler

    @property
    def train_dataset(self) -> SfmDataset | None:
        """
        Get the training dataset from the checkpoint.

        Returns:
            SfmDataset | None: The training dataset, or None if not present.
        """
        return self._train_dataset

    @property
    def eval_dataset(self) -> SfmDataset | None:
        """
        Get the evaluation dataset from the checkpoint.

        Returns:
            SfmDataset | None: The evaluation dataset, or None if not present.
        """
        return self._eval_dataset

    @property
    def initial_training_poses(self) -> torch.Tensor | None:
        """
        Get the initial camera-to-world poses used for training from the checkpoint.

        Returns:
            torch.Tensor | None: The initial camera-to-world poses, or None if not present.
        """
        return self._initial_training_poses

    @property
    def final_training_poses(self) -> torch.Tensor | None:
        """
        Get the final camera-to-world poses used for training from the checkpoint.

        Returns:
            torch.Tensor | None: The final camera-to-world poses, or None if not present.
        """
        return self._final_training_poses

    @property
    def training_projection_matrices(self) -> torch.Tensor | None:
        """
        Get the projection matrices used for training from the checkpoint.

        Returns:
            torch.Tensor | None: The projection matrices, or None if not present.
        """
        return self._training_projection_matrices

    @property
    def training_image_sizes(self) -> torch.Tensor | None:
        """
        Get the image sizes used for training from the checkpoint.

        Returns:
            torch.Tensor | None: The image sizes, or None if not present.
        """
        return self._training_image_sizes

    @property
    def eval_poses(self) -> torch.Tensor | None:
        """
        Get the camera-to-world poses used for evaluation from the checkpoint.

        Returns:
            torch.Tensor | None: The camera-to-world poses used for evaluation, or None if not present.
        """
        return self._eval_poses

    @property
    def eval_projection_matrices(self) -> torch.Tensor | None:
        """
        Get the projection matrices used for evaluation from the checkpoint.

        Returns:
            torch.Tensor | None: The projection matrices used for evaluation, or None if not present.
        """
        return self._eval_projection_matrices

    @property
    def eval_image_sizes(self) -> torch.Tensor | None:
        """
        Get the image sizes used for evaluation from the checkpoint.

        Returns:
            torch.Tensor | None: The image sizes used for evaluation, or None if not present.
        """
        return self._eval_image_sizes

    @property
    def run_name(self) -> str | None:
        """
        Get the name of the run associated with this checkpoint.

        Returns:
            str | None: The name of the run, or None if the run is un-named.
        """
        return self._run_name

    @torch.no_grad()
    @staticmethod
    def make_checkpoint(
        step: int,
        run_name: str | None,
        model: GaussianSplat3d,
        optimizer: GaussianSplatOptimizer,
        train_dataset: SfmDataset,
        eval_dataset: SfmDataset,
        config: dict,
        pose_adjust_model: CameraPoseAdjustment | None,
        pose_adjust_optimizer: torch.optim.Adam | None,
        pose_adjust_scheduler: torch.optim.lr_scheduler.ExponentialLR | None,
    ) -> "Checkpoint":

        if pose_adjust_model is not None and pose_adjust_optimizer is None:
            raise ValueError("Pose optimizer must be provided if pose adjust model is provided.")
        if pose_adjust_optimizer is not None and pose_adjust_model is None:
            raise ValueError("Pose adjust model must be provided if pose optimizer is provided.")

        dtype = model.dtype
        device = model.device
        initial_training_poses = torch.from_numpy(train_dataset.camera_to_world_matrices).to(device=device, dtype=dtype)
        final_training_poses = initial_training_poses
        if pose_adjust_model is not None:
            image_ids = torch.arange(len(train_dataset), device=device, dtype=torch.long)
            final_training_poses = pose_adjust_model(initial_training_poses, image_ids=image_ids)

        training_projection_matrices = torch.from_numpy(train_dataset.projection_matrices).to(
            device=device, dtype=dtype
        )
        training_image_sizes = torch.from_numpy(train_dataset.image_sizes).to(device=device, dtype=dtype)

        eval_poses = torch.from_numpy(eval_dataset.camera_to_world_matrices).to(device=device, dtype=dtype)
        eval_projection_matrices = torch.from_numpy(eval_dataset.projection_matrices).to(device=device, dtype=dtype)
        eval_image_sizes = torch.from_numpy(eval_dataset.image_sizes).to(device=device, dtype=dtype)

        return Checkpoint(
            step=step,
            run_name=run_name,
            model=model,
            optimizer=optimizer,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            initial_training_poses=initial_training_poses,
            final_training_poses=final_training_poses,
            training_projection_matrices=training_projection_matrices,
            training_image_sizes=training_image_sizes,
            eval_poses=eval_poses,
            eval_projection_matrices=eval_projection_matrices,
            eval_image_sizes=eval_image_sizes,
            pose_adjust_model=pose_adjust_model,
            pose_adjust_optimizer=pose_adjust_optimizer,
            pose_adjust_scheduler=pose_adjust_scheduler,
            _private=Checkpoint.__PRIVATE__,
        )

    @torch.no_grad()
    def save(self, path: pathlib.Path):
        """
        Save the checkpoint to a file.

        Args:
            path (pathlib.Path): The path to save the checkpoint file.
        """
        checkpoint_data = {
            "step": self._step,
            "config": self._config,
            "splats": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "run_name": self._run_name,
        }

        if self._pose_adjust_model is not None:
            assert self._pose_adjust_optimizer is not None, "Pose optimizer state is missing in the checkpoint."
            assert self._pose_adjust_scheduler is not None, "Pose scheduler state is missing in the checkpoint."
            checkpoint_data["pose_adjust_model"] = self._pose_adjust_model.state_dict()
            checkpoint_data["num_training_images"] = self._pose_adjust_model.num_poses

        if self._pose_adjust_optimizer is not None:
            assert "pose_adjust_model" in checkpoint_data, "Pose model state is missing in the checkpoint."
            assert self._pose_adjust_scheduler is not None, "Pose scheduler state is missing in the checkpoint."
            checkpoint_data["pose_adjust_optimizer"] = self._pose_adjust_optimizer.state_dict()

        if self._pose_adjust_scheduler is not None:
            assert "pose_adjust_optimizer" in checkpoint_data, "Pose optimizer state is missing in the checkpoint."
            checkpoint_data["pose_adjust_scheduler"] = self._pose_adjust_scheduler.state_dict()

        if self._train_dataset is not None:
            checkpoint_data["train_dataset"] = self._train_dataset.state_dict()
        if self._eval_dataset is not None:
            checkpoint_data["eval_dataset"] = self._eval_dataset.state_dict()

        if self._initial_training_poses is not None:
            checkpoint_data["initial_training_poses"] = self._initial_training_poses
        if self._final_training_poses is not None:
            checkpoint_data["final_training_poses"] = self._final_training_poses
        if self._training_projection_matrices is not None:
            checkpoint_data["training_projection_matrices"] = self._training_projection_matrices
        if self._training_image_sizes is not None:
            checkpoint_data["training_image_sizes"] = self._training_image_sizes

        if self._eval_poses is not None:
            checkpoint_data["eval_poses"] = self._eval_poses
        if self._eval_projection_matrices is not None:
            checkpoint_data["eval_projection_matrices"] = self._eval_projection_matrices
        if self._eval_image_sizes is not None:
            checkpoint_data["eval_image_sizes"] = self._eval_image_sizes

        torch.save(checkpoint_data, path)

    @torch.no_grad()
    @staticmethod
    def load(
        path: pathlib.Path,
        device: torch.device | str = "cpu",
        dataset_path: pathlib.Path | None = None,
        load_datasets: bool = True,
    ) -> "Checkpoint":
        checkpoint_data = torch.load(path, map_location=device, weights_only=False)

        step = checkpoint_data["step"]

        model = GaussianSplat3d.from_state_dict(checkpoint_data["splats"])
        config = checkpoint_data["config"]

        optimizer = GaussianSplatOptimizer(model)
        optimizer.load_state_dict(checkpoint_data["optimizer"])

        pose_adjust_model = None
        pose_adjust_optimizer = None
        pose_adjust_scheduler = None
        if "pose_adjust_model" in checkpoint_data:
            assert "pose_adjust_optimizer" in checkpoint_data, "Pose optimizer state is missing in the checkpoint."
            assert "num_training_images" in checkpoint_data, "Number of training images is missing in the checkpoint."
            assert "pose_adjust_scheduler" in checkpoint_data, "Pose scheduler state is missing in the checkpoint."
            num_poses = checkpoint_data["num_training_images"]
            assert isinstance(num_poses, int), "Number of training images should be an integer."
            pose_adjust_model = CameraPoseAdjustment(num_poses=num_poses).to(device=device)
            pose_adjust_model.load_state_dict(checkpoint_data["pose_adjust_model"])
            pose_adjust_optimizer = torch.optim.Adam(
                pose_adjust_model.parameters(),
                lr=config["pose_opt_lr"] * 100.0,  # Scale the learning rate for pose optimization
                weight_decay=config["pose_opt_reg"],
            )
            pose_adjust_optimizer.load_state_dict(checkpoint_data["pose_adjust_optimizer"])
            pose_adjust_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                pose_adjust_optimizer, gamma=config["pose_opt_lr_decay"]
            )
            pose_adjust_scheduler.load_state_dict(checkpoint_data["pose_adjust_scheduler"])

        train_dataset = None
        eval_dataset = None
        if "train_dataset" in checkpoint_data and load_datasets:
            train_dataset = SfmDataset.from_state_dict(checkpoint_data["train_dataset"], map_path=dataset_path)
        if "eval_dataset" in checkpoint_data and load_datasets:
            eval_dataset = SfmDataset.from_state_dict(checkpoint_data["eval_dataset"], map_path=dataset_path)

        initial_training_poses = None
        final_training_poses = None
        training_projection_matrices = None
        training_image_sizes = None
        if "initial_training_poses" in checkpoint_data:
            initial_training_poses = checkpoint_data["initial_training_poses"].to(device=device)
        if "final_training_poses" in checkpoint_data:
            final_training_poses = checkpoint_data["final_training_poses"].to(device=device)
        if "training_projection_matrices" in checkpoint_data:
            training_projection_matrices = checkpoint_data["training_projection_matrices"].to(device=device)
        if "training_image_sizes" in checkpoint_data:
            training_image_sizes = checkpoint_data["training_image_sizes"].to(device=device)

        eval_poses = None
        eval_projection_matrices = None
        eval_image_sizes = None
        if "eval_poses" in checkpoint_data:
            eval_poses = checkpoint_data["eval_poses"].to(device=device)
        if "eval_projection_matrices" in checkpoint_data:
            eval_projection_matrices = checkpoint_data["eval_projection_matrices"].to(device=device)
        if "eval_image_sizes" in checkpoint_data:
            eval_image_sizes = checkpoint_data["eval_image_sizes"].to(device=device)

        run_name = checkpoint_data.get("run_name", "default_run")

        return Checkpoint(
            step=step,
            run_name=run_name,
            model=model,
            optimizer=optimizer,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            initial_training_poses=initial_training_poses,
            final_training_poses=final_training_poses,
            training_projection_matrices=training_projection_matrices,
            training_image_sizes=training_image_sizes,
            eval_poses=eval_poses,
            eval_projection_matrices=eval_projection_matrices,
            eval_image_sizes=eval_image_sizes,
            pose_adjust_model=pose_adjust_model,
            pose_adjust_optimizer=pose_adjust_optimizer,
            pose_adjust_scheduler=pose_adjust_scheduler,
            _private=Checkpoint.__PRIVATE__,
        )
