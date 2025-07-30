# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
from typing import Any

import torch
from datasets import SfmDataset
from fvdb.optim import GaussianSplatOptimizer

from fvdb import GaussianSplat3d

from .camera_pose_adjust import CameraPoseAdjustment


class _CheckpoinPoseOptimizerState:
    """
    Internal class to represent the state of an optimizer in a checkpoint.

    This is used to store the state of the optimizer when saving and loading checkpoints.
    """

    version = "1.0.0"

    def __init__(
        self,
        num_training_poses: int,
        pose_adjust_model: CameraPoseAdjustment,
        pose_adjust_optimizer: torch.optim.Adam,
        pose_adjust_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    ):
        self.num_training_poses = num_training_poses
        self.pose_adjust_model = pose_adjust_model
        self.pose_adjust_optimizer = pose_adjust_optimizer
        self.pose_adjust_scheduler = pose_adjust_scheduler

    def state_dict(self) -> dict:
        """
        Get the state dictionary of the pose optimizer.

        Returns:
            dict: The state dictionary containing the pose adjust model, optimizer, and scheduler states.
        """
        return {
            "version": self.version,
            "num_training_poses": self.num_training_poses,
            "pose_adjust_model": self.pose_adjust_model.state_dict(),
            "pose_adjust_optimizer": self.pose_adjust_optimizer.state_dict(),
            "pose_adjust_scheduler": self.pose_adjust_scheduler.state_dict(),
        }

    @staticmethod
    def from_state_dict(checkpoint_data: dict, config: dict, device: torch.device) -> "_CheckpoinPoseOptimizerState":
        assert "version" in checkpoint_data, "Version information is missing in the checkpoint."
        if checkpoint_data["version"] != _CheckpoinPoseOptimizerState.version:
            raise ValueError(
                f"Checkpoint version {checkpoint_data['version']} does not match expected version { _CheckpoinPoseOptimizerState.version}."
            )
        assert "pose_adjust_optimizer" in checkpoint_data, "Pose optimizer state is missing in the checkpoint."
        assert "num_training_poses" in checkpoint_data, "Number of training images is missing in the checkpoint."
        assert "pose_adjust_scheduler" in checkpoint_data, "Pose scheduler state is missing in the checkpoint."
        assert "num_training_poses" in checkpoint_data, "Number of training poses is missing in the checkpoint."
        num_poses = checkpoint_data["num_training_poses"]
        assert isinstance(num_poses, int), "Number of training poses should be an integer."
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

        return _CheckpoinPoseOptimizerState(
            num_training_poses=num_poses,
            pose_adjust_model=pose_adjust_model,
            pose_adjust_optimizer=pose_adjust_optimizer,
            pose_adjust_scheduler=pose_adjust_scheduler,
        )


class _CheckpointOptimizerState:
    """
    Internal class to represent the state of an optimizer in a checkpoint.
    This is used to store the state of the optimizer when saving and loading checkpoints.
    """

    version = "1.0.0"

    def __init__(
        self, step: int, optimizer: GaussianSplatOptimizer, pose_optimizer: _CheckpoinPoseOptimizerState | None = None
    ):
        self.step = step
        self.optimizer = optimizer
        self.pose_optimizer = pose_optimizer

    @property
    def has_pose_optimizer(self) -> bool:
        """
        Check if the optimizer state contains a pose optimizer.

        Returns:
            bool: True if the pose optimizer is present, False otherwise.
        """
        return self.pose_optimizer is not None

    @property
    def pose_adjust_model(self) -> CameraPoseAdjustment | None:
        """
        Get the camera pose adjustment model from the pose optimizer if present.

        Returns:
            CameraPoseAdjustment | None: The camera pose adjustment model, or None if not present.
        """
        return self.pose_optimizer.pose_adjust_model if self.pose_optimizer is not None else None

    @property
    def pose_adjust_optimizer(self) -> torch.optim.Adam | None:
        """
        Get the camera pose adjustment optimizer from the pose optimizer if present.

        Returns:
            torch.optim.Adam | None: The camera pose adjustment optimizer, or None if not present.
        """
        return self.pose_optimizer.pose_adjust_optimizer if self.pose_optimizer is not None else None

    @property
    def pose_adjust_scheduler(self) -> torch.optim.lr_scheduler.ExponentialLR | None:
        """
        Get the camera pose adjustment scheduler from the pose optimizer if present.

        Returns:
            torch.optim.lr_scheduler.ExponentialLR | None: The camera pose adjustment scheduler, or None if not present.
        """
        return self.pose_optimizer.pose_adjust_scheduler if self.pose_optimizer is not None else None

    def state_dict(self) -> dict:
        """
        Get the state dictionary of the optimizer.

        Returns:
            dict: The state dictionary containing the optimizer state.
        """
        ret = {
            "version": self.version,
            "step": self.step,
            "optimizer": self.optimizer.state_dict(),
        }
        if self.pose_optimizer is not None:
            ret["pose_optimizer"] = self.pose_optimizer.state_dict()
        return ret

    @staticmethod
    def from_state_dict(checkpoint_data: dict, model: GaussianSplat3d, config: dict) -> "_CheckpointOptimizerState":
        assert "version" in checkpoint_data, "Version information is missing in the checkpoint."
        if checkpoint_data["version"] != _CheckpointOptimizerState.version:
            raise ValueError(
                f"Checkpoint version {checkpoint_data['version']} does not match expected version {_CheckpointOptimizerState.version}."
            )
        assert "optimizer" in checkpoint_data, "Optimizer state is missing in the checkpoint."
        assert "step" in checkpoint_data, "Step information is missing in the checkpoint."
        optimizer = GaussianSplatOptimizer(model)
        optimizer.load_state_dict(checkpoint_data["optimizer"])

        pose_optimizer = None
        if "pose_optimizer" in checkpoint_data:
            pose_optimizer = _CheckpoinPoseOptimizerState.from_state_dict(
                checkpoint_data["pose_optimizer"], config, model.device
            )

        return _CheckpointOptimizerState(
            step=checkpoint_data["step"], optimizer=optimizer, pose_optimizer=pose_optimizer
        )


class _CheckpointDatasetState:
    """
    Internal class to represent the state of a dataset in a checkpoint.
    This is used to store the state of the dataset when saving and loading checkpoints.
    """

    version = "1.0.0"

    def __init__(self, training_dataset: SfmDataset, eval_dataset: SfmDataset):
        self.training_dataset = training_dataset
        self.eval_dataset = eval_dataset

    def state_dict(self) -> dict:
        """
        Get the state dictionary of the dataset.

        Returns:
            dict: The state dictionary containing the training and evaluation dataset states.
        """
        return {
            "version": self.version,
            "train_dataset": self.training_dataset.state_dict(),
            "eval_dataset": self.eval_dataset.state_dict(),
        }

    @staticmethod
    def from_state_dict(checkpoint_data: dict, map_path: pathlib.Path | None = None) -> "_CheckpointDatasetState":
        """
        Create a dataset state from the checkpoint data.
        Args:
            checkpoint_data (dict): The checkpoint data containing the dataset states.
            map_path (pathlib.Path | None): The path to the dataset, if the checkpoint contains datasets,
                they will be updated to use this path instead of the one they were saved with.

        Returns:
            _CheckpointDatasetState: The dataset state created from the checkpoint data.
        """
        assert "train_dataset" in checkpoint_data, "Training dataset state is missing in the checkpoint."
        assert "eval_dataset" in checkpoint_data, "Evaluation dataset state is missing in the checkpoint."
        if "version" in checkpoint_data and checkpoint_data["version"] != _CheckpointDatasetState.version:
            raise ValueError(
                f"Checkpoint version {checkpoint_data['version']} does not match expected version {_CheckpointDatasetState.version}."
            )

        train_dataset = SfmDataset.from_state_dict(checkpoint_data["train_dataset"], map_path=map_path)
        eval_dataset = SfmDataset.from_state_dict(checkpoint_data["eval_dataset"], map_path=map_path)
        return _CheckpointDatasetState(
            training_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )


class _CheckpointTrainingMetadataState:
    """
    Internal class to represent a minimal state of a checkpoint.
    This is used to store the minimal state of the checkpoint when saving and loading checkpoints.
    """

    version = "1.0.0"

    def __init__(
        self,
        run_name: str,
        initial_training_poses: torch.Tensor,
        final_training_poses: torch.Tensor,
        training_projection_matrices: torch.Tensor,
        training_image_sizes: torch.Tensor,
        eval_poses: torch.Tensor,
        eval_projection_matrices: torch.Tensor,
        eval_image_sizes: torch.Tensor,
        scene_scale: float,
    ):
        self.run_name = run_name
        self.initial_training_poses = initial_training_poses
        self.final_training_poses = final_training_poses
        self.training_projection_matrices = training_projection_matrices
        self.training_image_sizes = training_image_sizes
        self.eval_poses = eval_poses
        self.eval_projection_matrices = eval_projection_matrices
        self.eval_image_sizes = eval_image_sizes
        self.scene_scale = scene_scale

    def state_dict(self) -> dict:
        """
        Get the state dictionary for the checkpoint.
        Returns:
            dict: The state dictionary containing the checkpoint states.
        """
        return {
            "version": self.version,
            "run_name": self.run_name,
            "initial_training_poses": self.initial_training_poses,
            "final_training_poses": self.final_training_poses,
            "training_projection_matrices": self.training_projection_matrices,
            "training_image_sizes": self.training_image_sizes,
            "eval_poses": self.eval_poses,
            "eval_projection_matrices": self.eval_projection_matrices,
            "eval_image_sizes": self.eval_image_sizes,
            "scene_scale": self.scene_scale,
        }

    @staticmethod
    def from_state_dict(state_dict: dict) -> "_CheckpointTrainingMetadataState":
        """
        Create a checkpoint state from the state dictionary.
        Args:
            state_dict (dict): The state dictionary containing the checkpoint states.
        Returns:
            _CheckpointBaseState: The checkpoint state created from the state dictionary.
        """
        assert "version" in state_dict, "Version information is missing in the checkpoint."
        version = state_dict["version"]
        if version != _CheckpointTrainingMetadataState.version:
            raise ValueError(
                f"Checkpoint version {version} does not match expected version {_CheckpointTrainingMetadataState.version}."
            )

        assert "run_name" in state_dict, "Run name is missing in the checkpoint."
        assert "initial_training_poses" in state_dict, "Initial training poses are missing in the checkpoint."
        assert "final_training_poses" in state_dict, "Final training poses are missing in the checkpoint."
        assert (
            "training_projection_matrices" in state_dict
        ), "Training projection matrices are missing in the checkpoint."
        assert "training_image_sizes" in state_dict, "Training image sizes are missing in the checkpoint."
        assert "eval_poses" in state_dict, "Evaluation poses are missing in the checkpoint."
        assert "eval_projection_matrices" in state_dict, "Evaluation projection matrices are missing in the checkpoint."
        assert "eval_image_sizes" in state_dict, "Evaluation image sizes are missing in the checkpoint."
        assert "scene_scale" in state_dict, "Scene scale is missing in the checkpoint."

        return _CheckpointTrainingMetadataState(
            run_name=state_dict["run_name"],
            initial_training_poses=state_dict["initial_training_poses"],
            final_training_poses=state_dict["final_training_poses"],
            training_projection_matrices=state_dict["training_projection_matrices"],
            training_image_sizes=state_dict["training_image_sizes"],
            eval_poses=state_dict["eval_poses"],
            eval_projection_matrices=state_dict["eval_projection_matrices"],
            eval_image_sizes=state_dict["eval_image_sizes"],
            scene_scale=state_dict["scene_scale"],
        )


class Checkpoint:
    """
    Class representing a checkpoint for scene optimization.

    A checkpoint contains data about the model, the training configuration, and basic metadata
    about the training dataset.

    In addition, it can also contain the optimizer and dataset used to train the model.

    At the very least, a checkpoint will ALWAYS contain:
    - splats (GaussianSplat3d): The Gaussian Splatting model.
    - config (dict): The configuration used for training.
    - initial_training_poses (torch.Tensor): The initial camera-to-world poses used for training.
    - final_training_poses (torch.Tensor): The final camera-to-world poses used for training.
    - training_projection_matrices (torch.Tensor): The projection matrices used for training.
    - training_image_sizes (torch.Tensor): The image sizes used for training.
    - eval_poses (torch.Tensor): The camera-to-world poses used for evaluation.
    - eval_projection_matrices (torch.Tensor): The projection matrices used for evaluation.
    - eval_image_sizes (torch.Tensor): The image sizes used for evaluation.
    - run_name (str): The name of the run associated with this checkpoint.
    - scene_scale (float): The scale of the scene.
    - camera_to_world_matrices (torch.Tensor): The camera-to-world matrices used for both training and evaluation.
    - projection_matrices (torch.Tensor): The projection matrices used for both training and evaluation.
    - image_sizes (torch.Tensor): The image sizes used for both training and evaluation.

    The checkpoint can also OPTIONALLY contain:
    - optimizer (GaussianSplatOptimizer | None): The optimizer used for training.
    - pose_adjust_model (CameraPoseAdjustment | None): The camera pose adjustment model, if used.
    - pose_adjust_optimizer (torch.optim.Adam | None): The optimizer for the camera pose adjustment model, if used.
    - pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None): The learning rate scheduler for the camera pose adjustment optimizer, if used.
    - training_dataset (SfmDataset | None): The training dataset used for training if it was saved and can be located.
    - eval_dataset (SfmDataset | None): The evaluation dataset used for evaluation if it was saved and can be located.
    """

    __PRIVATE__ = object()

    version = "1.0.0"

    def __init__(
        self,
        model: GaussianSplat3d,
        config: dict | None,
        training_metadata_state: _CheckpointTrainingMetadataState,
        optimizer_state: _CheckpointOptimizerState | None,
        dataset_state: _CheckpointDatasetState | None,
        _private: Any = None,
    ):
        if _private is not Checkpoint.__PRIVATE__:
            raise ValueError(
                "SceneOptimizationCheckpoint can only be initialized through the `load` or `make_checkpoint` functions."
            )
        self._model = model
        self._config = config
        self._training_metadata_state = training_metadata_state
        self._optimizer_state = optimizer_state
        self._dataset_state = dataset_state

    @torch.no_grad()
    @staticmethod
    def make_minimal_checkpoint(
        run_name: str,
        model: GaussianSplat3d,
        scene_scale: float,
        training_camera_to_world_matrices: torch.Tensor,
        training_projection_matrices: torch.Tensor,
        training_image_sizes: torch.Tensor,
        eval_camera_to_world_matrices: torch.Tensor,
        eval_projection_matrices: torch.Tensor,
        eval_image_sizes: torch.Tensor,
    ) -> "Checkpoint":
        """
        Create a minimal checkpoint for the model and configuration without optimizer or dataset states.

        This is useful if you want to postprocess the checkpoint without needing the optimizer or dataset states.

        Args:
            run_name (str): The name of the run associated with this checkpoint.
            model (GaussianSplat3d): The Gaussian Splatting model.
            config (dict): The configuration used for training.
            training_camera_to_world_matrices (torch.Tensor): The initial camera-to-world poses used for training.
            training_projection_matrices (torch.Tensor): The projection matrices used for training.
            training_image_sizes (torch.Tensor): The image sizes used for training.
            eval_camera_to_world_matrices (torch.Tensor): The camera-to-world poses used for evaluation.
            eval_projection_matrices (torch.Tensor): The projection matrices used for evaluation.
            eval_image_sizes (torch.Tensor): The image sizes used for evaluation.
        """

        # Training metadata state
        dtype = model.dtype
        device = model.device
        training_metadata_state = _CheckpointTrainingMetadataState(
            run_name=run_name,
            initial_training_poses=training_camera_to_world_matrices.to(device=device, dtype=dtype),
            final_training_poses=training_camera_to_world_matrices.to(device=device, dtype=dtype),
            training_projection_matrices=training_projection_matrices.to(device=device, dtype=dtype),
            training_image_sizes=training_image_sizes.to(device=device, dtype=dtype),
            eval_poses=eval_camera_to_world_matrices.to(device=device, dtype=dtype),
            eval_projection_matrices=eval_projection_matrices.to(device=device, dtype=dtype),
            eval_image_sizes=eval_image_sizes.to(device=device, dtype=dtype),
            scene_scale=scene_scale,
        )

        return Checkpoint(
            model=model,
            config=None,
            training_metadata_state=training_metadata_state,
            optimizer_state=None,
            dataset_state=None,
            _private=Checkpoint.__PRIVATE__,
        )

    @torch.no_grad()
    @staticmethod
    def make_checkpoint(
        step: int,
        run_name: str,
        model: GaussianSplat3d,
        config: dict,
        train_dataset: SfmDataset,
        eval_dataset: SfmDataset,
        optimizer: GaussianSplatOptimizer | None,
        pose_adjust_model: CameraPoseAdjustment | None,
        pose_adjust_optimizer: torch.optim.Adam | None,
        pose_adjust_scheduler: torch.optim.lr_scheduler.ExponentialLR | None,
    ) -> "Checkpoint":
        """
        Create a checkpoint for the model, optimizer, and configuration.

        Args:
            step (int): The training step at which the checkpoint is created.
            run_name (str): The name of the run associated with this checkpoint.
            model (GaussianSplat3d): The Gaussian Splatting model.
            config (dict): The configuration used for training.
            train_dataset (SfmDataset): The training dataset
            eval_dataset (SfmDataset): The evaluation dataset.
            optimizer (GaussianSplatOptimizer | None): The optimizer used for training or None to not save the optimizer.
            pose_adjust_model (CameraPoseAdjustment | None): The camera pose adjustment model, if used.
            pose_adjust_optimizer (torch.optim.Adam | None): The optimizer for the camera pose adjustment model, if used.
            pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None):
                The learning rate scheduler for the camera pose adjustment optimizer, if used.

        Returns:
            Checkpoint: A new checkpoint instance containing the model, optimizer, datasets, and camera poses
                used during training and evaluation.
        """

        # Dataset state
        dataset_state = _CheckpointDatasetState(
            training_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Training metadata state
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
        training_metadata_state = _CheckpointTrainingMetadataState(
            run_name=run_name,
            initial_training_poses=initial_training_poses,
            final_training_poses=final_training_poses,
            training_projection_matrices=training_projection_matrices,
            training_image_sizes=training_image_sizes,
            eval_poses=eval_poses,
            eval_projection_matrices=eval_projection_matrices,
            eval_image_sizes=eval_image_sizes,
            scene_scale=train_dataset.scene_scale,
        )

        # Optimizer state
        optimizer_state = None
        if optimizer is not None and pose_adjust_model is not None:
            if pose_adjust_optimizer is None:
                raise ValueError("Pose optimizer must be provided if pose adjust model is provided.")
            if pose_adjust_scheduler is None:
                raise ValueError("Pose adjust scheduler must be provided if pose adjust model is provided.")

            pose_optimizer_state = _CheckpoinPoseOptimizerState(
                num_training_poses=len(train_dataset),
                pose_adjust_model=pose_adjust_model,
                pose_adjust_optimizer=pose_adjust_optimizer,
                pose_adjust_scheduler=pose_adjust_scheduler,
            )

            optimizer_state = _CheckpointOptimizerState(
                step=step, optimizer=optimizer, pose_optimizer=pose_optimizer_state
            )
        elif optimizer is not None and pose_adjust_model is None:
            optimizer_state = _CheckpointOptimizerState(step=step, optimizer=optimizer, pose_optimizer=None)

        return Checkpoint(
            model=model,
            config=config,
            training_metadata_state=training_metadata_state,
            optimizer_state=optimizer_state,
            dataset_state=dataset_state,
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
            "version": self.version,
            "splats": self._model.state_dict(),
            "config": self._config,
            "training_metadata_state": self._training_metadata_state.state_dict(),
            "optimizer_state": self._optimizer_state.state_dict() if self._optimizer_state is not None else None,
            "dataset_state": self._dataset_state.state_dict() if self._dataset_state is not None else None,
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

        # Load the model
        assert "splats" in checkpoint_data, "Model state is missing in the checkpoint."
        model = GaussianSplat3d.from_state_dict(checkpoint_data["splats"]).to(device)
        training_metadata_state = _CheckpointTrainingMetadataState.from_state_dict(
            checkpoint_data["training_metadata_state"]
        )

        # Load the training config
        assert "config" in checkpoint_data, "Configuration is missing in the checkpoint."
        config = checkpoint_data["config"]

        # Load the optimizer if present
        optimizer_state = None
        assert "optimizer_state" in checkpoint_data, "Optimizer state is missing in the checkpoint."
        if checkpoint_data["optimizer_state"] is not None:
            optimizer_state = _CheckpointOptimizerState.from_state_dict(
                checkpoint_data["optimizer_state"], model, config
            )

        # Try and load the dataset if present
        dataset_state = None
        if "dataset_state" in checkpoint_data:
            try:
                dataset_state = _CheckpointDatasetState.from_state_dict(
                    checkpoint_data["dataset_state"], map_path=dataset_path
                )
            except FileNotFoundError as e:
                logger.warning(
                    f"Dataset state could not be loaded from checkpoint because the dataset was not found."
                    f"You can specify the dataset_path if you know where the data is located or you can ignore "
                    f"this warning if you do not need the datasets for this checkpoint. Exception: {e}"
                )

        return Checkpoint(
            model=model,
            config=config,
            training_metadata_state=training_metadata_state,
            optimizer_state=optimizer_state,
            dataset_state=dataset_state,
            _private=Checkpoint.__PRIVATE__,
        )

    @property
    def has_optimizer(self) -> bool:
        """
        Check if the checkpoint contains an optimizer state.

        Returns:
            bool: True if the checkpoint contains an optimizer state, False otherwise.
        """
        return self._optimizer_state is not None

    @property
    def has_datasets(self) -> bool:
        """
        Check if the checkpoint contains a dataset state.

        Returns:
            bool: True if the checkpoint contains a dataset state, False otherwise.
        """
        return self._dataset_state is not None

    @property
    def has_config(self) -> bool:
        """
        Check if the checkpoint contains a training configuration.

        Returns:
            bool: True if the checkpoint contains a configuration, False otherwise.
        """
        return self._config is not None

    @property
    def splats(self) -> GaussianSplat3d:
        """
        Get the Gaussian Splatting model from the checkpoint.

        Returns:
            GaussianSplat3d: The Gaussian Splatting model.
        """
        return self._model

    @property
    def config(self) -> dict | None:
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
        return self._training_metadata_state.run_name

    @property
    def scene_scale(self) -> float:
        """
        Get the scale of the scene used for depth calculations.

        Returns:
            float: The scale of the scene.
        """
        return self._training_metadata_state.scene_scale

    @property
    def camera_to_world_matrices(self) -> torch.Tensor:
        """
        Get the camera-to-world matrices used for both training and validation from the checkpoint.

        Returns:
            torch.Tensor: The camera-to-world matrices for both training and evaluation.
        """
        final_train_c2w = self._training_metadata_state.final_training_poses
        eval_c2w = self._training_metadata_state.eval_poses
        return torch.cat([final_train_c2w, eval_c2w], dim=0)

    @property
    def projection_matrices(self) -> torch.Tensor:
        """
        Get the projection matrices used for both training and evaluation from the checkpoint.

        Returns:
            torch.Tensor: The projection matrices for both training and evaluation.
        """
        final_train_proj = self._training_metadata_state.training_projection_matrices
        val_proj = self._training_metadata_state.eval_projection_matrices
        return torch.cat([final_train_proj, val_proj], dim=0)

    @property
    def image_sizes(self) -> torch.Tensor:
        """
        Get the image sizes used for both training and validation from the checkpoint.

        Returns:
            torch.Tensor: The image sizes for both training and evaluation.
        """
        final_train_sizes = self._training_metadata_state.training_image_sizes
        val_sizes = self._training_metadata_state.eval_image_sizes
        return torch.cat([final_train_sizes, val_sizes], dim=0)

    @property
    def initial_training_poses(self) -> torch.Tensor:
        """
        Get the initial camera-to-world poses used for training from the checkpoint.

        Returns:
            torch.Tensor: The initial camera-to-world poses of each image in the training dataset.
        """
        return self._training_metadata_state.initial_training_poses

    @property
    def final_training_poses(self) -> torch.Tensor:
        """
        Get the final camera-to-world poses used for training from the checkpoint.

        Returns:
            torch.Tensor: The final camera-to-world poses of each image in the training dataset.
        """
        return self._training_metadata_state.final_training_poses

    @property
    def training_projection_matrices(self) -> torch.Tensor:
        """
        Get the projection matrices used for training from the checkpoint.

        Returns:
            torch.Tensor: The projection matrices of each image in the training dataset.
        """
        return self._training_metadata_state.training_projection_matrices

    @property
    def training_image_sizes(self) -> torch.Tensor:
        """
        Get the image sizes used for training from the checkpoint.

        Returns:
            torch.Tensor: The image sizes of each image in the training dataset.
        """
        return self._training_metadata_state.training_image_sizes

    @property
    def eval_poses(self) -> torch.Tensor:
        """
        Get the camera-to-world poses used for evaluation from the checkpoint.

        Returns:
            torch.Tensor: The camera-to-world poses of each image in the evaluation dataset.
        """
        return self._training_metadata_state.eval_poses

    @property
    def eval_projection_matrices(self) -> torch.Tensor:
        """
        Get the projection matrices used for evaluation from the checkpoint.

        Returns:
            torch.Tensor: The projection matrices of each image in the evaluation dataset.
        """
        return self._training_metadata_state.eval_projection_matrices

    @property
    def eval_image_sizes(self) -> torch.Tensor:
        """
        Get the image sizes used for evaluation from the checkpoint.

        Returns:
            torch.Tensor: The image size of each image in the evaluation dataset.
        """
        return self._training_metadata_state.eval_image_sizes

    #
    # Properties to access the optimizer and pose adjustment model, if present.
    #
    @property
    def step(self) -> int | None:
        """
        Get the training step at which the checkpoint was created.

        Returns:
            int | None: The training step, or None if not present.
        """
        return self._optimizer_state.step if self._optimizer_state is not None else None

    @property
    def optimizer(self) -> GaussianSplatOptimizer | None:
        """
        Get the optimizer used for training from the checkpoint.

        Returns:
            GaussianSplatOptimizer | None: The optimizer used for training, or None if not present.
        """
        return self._optimizer_state.optimizer if self._optimizer_state is not None else None

    @property
    def pose_adjust_model(self) -> CameraPoseAdjustment | None:
        """
        Get the camera pose adjustment model from the checkpoint.

        Returns:
            CameraPoseAdjustment | None: The camera pose adjustment model, or None if not present.
        """
        return self._optimizer_state.pose_adjust_model if self._optimizer_state is not None else None

    @property
    def pose_adjust_optimizer(self) -> torch.optim.Adam | None:
        """
        Get the camera pose adjustment optimizer from the checkpoint.

        Returns:
            torch.optim.Adam | None: The camera pose adjustment optimizer, or None if not present.
        """
        return self._optimizer_state.pose_adjust_optimizer if self._optimizer_state is not None else None

    @property
    def pose_adjust_scheduler(self) -> torch.optim.lr_scheduler.ExponentialLR | None:
        """
        Get the camera pose adjustment optimizer from the checkpoint if present.

        Returns:
            torch.optim.Adam | None: The camera pose adjustment optimizer, or None if not present.
        """
        return self._optimizer_state.pose_adjust_scheduler if self._optimizer_state is not None else None

    #
    # Dataset properties to access the training and evaluation datasets, if present.
    #
    @property
    def train_dataset(self) -> SfmDataset | None:
        """
        Get the training dataset from the checkpoint if present.

        Returns:
            SfmDataset | None: The training dataset, or None if not present.
        """
        return self._dataset_state.training_dataset if self._dataset_state is not None else None

    @property
    def eval_dataset(self) -> SfmDataset | None:
        """
        Get the evaluation dataset from the checkpoint.

        Returns:
            SfmDataset | None: The evaluation dataset, or None if not present.
        """
        return self._dataset_state.eval_dataset if self._dataset_state is not None else None
