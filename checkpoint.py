# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
from typing import Any

import numpy as np
import torch
import tqdm
from camera_pose_adjust import CameraPoseAdjustment
from datasets import SfmDataset
from fvdb.optim import GaussianSplatOptimizer
from skimage import feature, morphology

from fvdb import GaussianSplat3d, Grid


class Checkpoint:
    """
    Class representing a checkpoint for the Gaussian Splatting model.

    The checkpoint contains the model, optimizer, configuration, datasets, and camera poses
    used during training and evaluation. It can be saved to and loaded from a file.

    Note: Do not instantiate this class directly. Use the `make_checkpoint` or `load` methods instead.

    A checkpoint is a snapshot of the training state at a specific step, including:
    - `splats`: The Gaussian Splatting model.
    - `optimizer`: The optimizer used for training.
    - `config`: The configuration used for training.
    - `train_dataset`: The training dataset.
    - `eval_dataset`: The evaluation dataset.
    - `initial_training_poses`: The initial camera-to-world poses used for training.
    - `final_training_poses`: The final camera-to-world poses after potential optimization.
    - `training_projection_matrices`: The projection matrices for each image used in training.
    - `training_image_sizes`: The image sizes for each image used in training.
    - `eval_poses`: The camera-to-world poses used for validation.
    - `eval_projection_matrices`: The projection matrices for each image used in validation.
    - `eval_image_sizes`: The image sizes for each image used in validation.
    - `pose_adjust_model`: The camera pose adjustment model, if used.
    - `pose_adjust_optimizer`: The optimizer for the camera pose adjustment model, if used.
    - `pose_adjust_scheduler`: The learning rate scheduler for the camera pose adjustment optimizer, if used.
    - `run_name`: The name of the run associated with this checkpoint, if any.
    """

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
            train_dataset (SfmDataset | None): The training dataset, if available.
            eval_dataset (SfmDataset | None): The evaluation dataset, if available.
            initial_training_poses (torch.Tensor | None): The initial camera-to-world poses used for training, if available.
            final_training_poses (torch.Tensor | None): The final camera-to-world poses after potential optimization, if available.
            training_projection_matrices (torch.Tensor | None): The projection matrices used for training, if available.
            training_image_sizes (torch.Tensor | None): The image sizes used for training, if available.
            eval_poses (torch.Tensor | None):
                The camera-to-world poses used for evaluation, if available.
            eval_projection_matrices (torch.Tensor | None):
                The projection matrices used for evaluation, if available.
            eval_image_sizes (torch.Tensor | None): The image sizes used for evaluation, if available.
            pose_adjust_model (CameraPoseAdjustment | None): The camera pose adjustment model, if used.
            pose_adjust_optimizer (torch.optim.Adam | None): The optimizer for the camera pose adjustment model, if used.
            pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None):
                The learning rate scheduler for the camera pose adjustment optimizer, if used.
            _private (Any): Internal use only. Should be set to `Checkpoint.__PRIVATE__` to allow instantiation.
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
    def camera_to_world_matrices(self) -> torch.Tensor | None:
        """
        Get the camera-to-world matrices used for both training and validation from the checkpoint.

        Returns:
            torch.Tensor | None: The camera-to-world matrices, or None if not present.
        """
        final_train_c2w = self._final_training_poses
        eval_c2w = self._eval_poses
        if final_train_c2w is not None and eval_c2w is not None:
            return torch.cat([final_train_c2w, eval_c2w], dim=0)
        elif final_train_c2w is not None:
            return final_train_c2w
        elif eval_c2w is not None:
            return eval_c2w
        else:
            return None

    @property
    def projection_matrices(self) -> torch.Tensor | None:
        """
        Get the projection matrices used for both training and validation from the checkpoint.

        Returns:
            torch.Tensor | None: The projection matrices, or None if not present.
        """
        final_train_proj = self._training_projection_matrices
        val_proj = self._eval_projection_matrices
        if final_train_proj is not None and val_proj is not None:
            return torch.cat([final_train_proj, val_proj], dim=0)
        elif final_train_proj is not None:
            return final_train_proj
        elif val_proj is not None:
            return val_proj
        else:
            return None

    @property
    def image_sizes(self) -> torch.Tensor | None:
        """
        Get the image sizes used for both training and validation from the checkpoint.

        Returns:
            torch.Tensor | None: The image sizes, or None if not present.
        """
        final_train_sizes = self._training_image_sizes
        val_sizes = self._eval_image_sizes
        if final_train_sizes is not None and val_sizes is not None:
            return torch.cat([final_train_sizes, val_sizes], dim=0)
        elif final_train_sizes is not None:
            return final_train_sizes
        elif val_sizes is not None:
            return val_sizes
        else:
            return None

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
        """
        Create a checkpoint for the model, optimizer, and configuration.

        Args:
            step (int): The training step at which the checkpoint is created.
            run_name (str | None): The name of the run associated with this checkpoint, if any.
            model (GaussianSplat3d): The Gaussian Splatting model.
            optimizer (GaussianSplatOptimizer): The optimizer used for training.
            train_dataset (SfmDataset): The training dataset.
            eval_dataset (SfmDataset): The evaluation dataset.
            config (dict): The configuration used for training.
            pose_adjust_model (CameraPoseAdjustment | None): The camera pose adjustment model, if used.
            pose_adjust_optimizer (torch.optim.Adam | None): The optimizer for the camera pose adjustment model, if used.
            pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None):
                The learning rate scheduler for the camera pose adjustment optimizer, if used.

        Returns:
            Checkpoint: A new checkpoint instance containing the model, optimizer, datasets, and camera poses
                used during training and evaluation.
        """

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
        """
        Load a checkpoint from a file.

        Args:
            path (pathlib.Path): The path to the checkpoint file.
            device (torch.device | str): The device to load the checkpoint onto. Defaults to "cpu".
            dataset_path (pathlib.Path | None): The path to the dataset, if the checkpoint contains datasets,
                they will be updated to use this path instead of the one they were saved with.
            load_datasets (bool): Whether to load the training and evaluation datasets from the checkpoint. Defaults to True.
                Setting this to false is useful if you don't have access to the datasets the checkpoint was trained on,
                but still want to load the model and optimizer state.
        Returns:
            Checkpoint: A new checkpoint instance loaded from the file.
        """
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


@torch.no_grad()
def extract_tsdf_from_checkpoint(
    checkpoint: Checkpoint,
    trunctation_margin: float,
    near: float = 0.1,
    far: float = 1e10,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    show_progress: bool = True,
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """
    Extract a TSDF using TSDF fusion from depth maps rendered from a Gaussian splat.
    The TSDF fusion algorithm is based on the paper:
    "KinectFusion: Real-Time Dense Surface Mapping and Tracking"
    (https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/)

    Args:
        checkpoint (Checkpoint): A checkpoint containing the Gaussian splat model and camera parameters.
        truncation_margin (float): Margin for truncating the TSDF, in world units.
        near (float): Near plane distance below which we'll ignore depth samples (default is 0.0).
        far (float): Far plane distance above which we'll ignore depth samples (default is 1e10).
        device (torch.device | str): Device to use (default is "cuda").
        dtype: Data type for the TSDF and weights. Default is torch.float16.
        feature_dtype: Data type for the features (default is torch.uint8 which is good for RGB colors).
        show_progress (bool): Whether to show a progress bar (default is True).

    Returns:
        grid (Grid): A Grid object encoding the topology of the TSDF.
        tsdf (torch.Tensor): A tensor of TSDF values indexed by the grid.
        features (torch.Tensor): A tensor of features (e.g., colors) indexed by the grid.
    """

    model: GaussianSplat3d = checkpoint.splats.to(device)

    camera_to_world_matrices = checkpoint.camera_to_world_matrices
    projection_matrices = checkpoint.projection_matrices
    image_sizes = checkpoint.image_sizes

    if camera_to_world_matrices is None:
        raise ValueError("Camera to world matrices are not available in the checkpoint.")
    if projection_matrices is None:
        raise ValueError("Projection matrices are not available in the checkpoint.")
    if image_sizes is None:
        raise ValueError("Image sizes are not available in the checkpoint.")

    voxel_size = trunctation_margin / 2.0
    accum_grid = Grid.from_dense(dense_dims=1, ijk_min=0, voxel_size=voxel_size, origin=0.0, device=model.device)
    tsdf = torch.zeros(accum_grid.num_voxels, device=model.device, dtype=dtype)
    weights = torch.zeros(accum_grid.num_voxels, device=model.device, dtype=dtype)
    features = torch.zeros((accum_grid.num_voxels, model.num_channels), device=model.device, dtype=feature_dtype)

    enumerator = (
        tqdm.tqdm(range(len(camera_to_world_matrices)), unit="imgs", desc="Extracting TSDF")
        if show_progress
        else range(len(camera_to_world_matrices))
    )

    for i in enumerator:
        cam_to_world_matrix = camera_to_world_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        world_to_cam_matrix = torch.linalg.inv(cam_to_world_matrix).contiguous().to(dtype=torch.float32, device=device)
        projection_matrix = projection_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        image_size = image_sizes[i]

        # We set near and far planes to 0.0 and 1e10 respectively to avoid clipping
        # in the rendering process. Instead, we will use the provided near and far planes
        # to filter the depth images after rendering so pixels out of range will not be integrated
        # into the TSDF.
        feature_and_depth, alpha = model.render_images_and_depths(
            world_to_camera_matrices=world_to_cam_matrix.unsqueeze(0),
            projection_matrices=projection_matrix.unsqueeze(0),
            image_width=int(image_size[1].item()),
            image_height=int(image_size[0].item()),
            near=0.0,
            far=1e10,
        )

        if feature_dtype == torch.uint8:
            feature_images = (feature_and_depth[..., : model.num_channels].clip_(min=0.0, max=1.0) * 255.0).to(
                feature_dtype
            )
        else:
            feature_images = feature_and_depth[..., : model.num_channels].to(feature_dtype)
        feature_images = feature_images.squeeze(0)
        depth_images = (feature_and_depth[..., -1].unsqueeze(-1) / alpha.clamp(min=1e-10)).to(dtype).squeeze(0)
        weight_images = ((depth_images > near) & (depth_images < far)).to(dtype).squeeze(0)

        accum_grid, tsdf, weights, features = accum_grid.integrate_tsdf_with_features(
            trunctation_margin,
            projection_matrix.to(dtype),
            cam_to_world_matrix.to(dtype),
            tsdf,
            features,
            weights,
            depth_images,
            feature_images,
            weight_images,
        )

        if show_progress:
            assert isinstance(enumerator, tqdm.tqdm)
            enumerator.set_postfix({"accumulated_voxels": accum_grid.num_voxels})

        # TSDF fusion is a bit of a torture case for the PyTorch memory allocator since
        # it progressively allocates bigger tensors which don't fit in the memory pool,
        # causing the pool to grow larger and larger.
        # To avoid this, we synchronize the CUDA device and empty the cache after each image.
        del feature_images, depth_images, weight_images
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # After integrating all the images, we prune the grid to remove empty voxels which have no weights.
    # This is done to reduce the size of the grid and speed up the marching cubes algorithm
    # which will be used to extract the mesh.
    new_grid = accum_grid.pruned_grid(weights > 0.0)
    filter_tsdf = new_grid.inject_from(accum_grid, tsdf)
    filter_colors = new_grid.inject_from(accum_grid, features)

    return new_grid, filter_tsdf, filter_colors


@torch.no_grad()
def extract_mesh_from_checkpoint(
    checkpoint: Checkpoint,
    truncation_margin: float,
    near: float = 0.1,
    far: float = 1e10,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract a mesh from a Gaussian splat using TSDF fusion and marching cubes.
    The TSDF fusion algorithm is based on the paper:
    "KinectFusion: Real-Time Dense Surface Mapping and Tracking"
    (https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/)

    Args:
        checkpoint (Checkpoint): A checkpoint containing the Gaussian splat model and camera parameters.
        truncation_margin (float): Margin for truncating the TSDF, in world units.
        near (float): Near plane distance below which we'll ignore depth samples (default is 0.0).
        far (float): Far plane distance above which we'll ignore depth samples (default is 1e10).
        device (torch.device | str): Device to use (default is "cuda").
        dtype: Data type for the TSDF and weights. Default is torch.float16.
        feature_dtype: Data type for the features (default is torch.uint8 which is good for RGB colors).
        show_progress (bool): Whether to show a progress bar (default is True).

    Returns:
        mesh_vertices (torch.Tensor): Vertices of the extracted mesh.
        mesh_faces (torch.Tensor): Faces of the extracted mesh.
        mesh_colors (torch.Tensor): Colors of the extracted mesh vertices.
    """

    accum_grid, tsdf, colors = extract_tsdf_from_checkpoint(
        checkpoint,
        truncation_margin,
        near=near,
        far=far,
        device=device,
        dtype=dtype,
        feature_dtype=feature_dtype,
        show_progress=show_progress,
    )

    mesh_vertices, mesh_faces, _ = accum_grid.marching_cubes(tsdf, 0.0)
    mesh_colors = accum_grid.sample_trilinear(mesh_vertices, colors.to(dtype)) / 255.0
    mesh_colors.clip_(min=0.0, max=1.0)

    return mesh_vertices, mesh_faces, mesh_colors


@torch.no_grad()
def extract_point_cloud_from_checkpoint(
    checkpoint: Checkpoint,
    near: float = 0.1,
    far: float = 1e10,
    depth_image_downsample_factor: int = 1,
    canny_edge_std: float = 1.0,
    canny_mask_dilation: int = 5,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cuda",
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a point cloud from a Gaussian splat using depth rendering, possibly filtering points
    using Canny edge detection on the depth images.

    Args:
        checkpoint (Checkpoint): A checkpoint containing the Gaussian splat model and camera parameters.
        near (float): Near plane distance below which we'll ignore depth samples (default is 0.1).
        far (float): Far plane distance above which we'll ignore depth samples (default is 1e10).
        depth_image_downsample_factor (int): Factor by which to downsample the depth images before extracting points
            (default is 1, no downsampling). This is useful to reduce the number of points extracted from the point cloud
            and speed up the extraction process. A value of 2 will downsample the depth images by a factor of 2 in both dimensions,
            resulting in a point cloud with approximately 1/4 the number of points compared to the original depth images.
        quantization (float): Quantization step for the point cloud (default is 0.0, no quantization).
        canny_edge_std (float): Standard deviation for the Gaussian filter applied to the depth image
            before Canny edge detection (default is 1.0). Set to 0.0 to disable canny edge filtering.
        canny_mask_dilation (int): Dilation size for the Canny edge mask (default is 5).
        dtype (torch.dtype): Data type for the point cloud and colors (default is torch.float16).
        device (torch.device | str): Device to use (default is "cuda").
        show_progress (bool): Whether to show a progress bar (default is True).

    Returns:
        points (torch.Tensor): A [num_points, 3] shaped tensor of points in camera space.
        colors (torch.Tensor): A [num_points, 3] shaped tensor of RGB colors for the points.
    """

    model = checkpoint.splats.to(device)
    camera_to_world_matrices = checkpoint.camera_to_world_matrices
    projection_matrices = checkpoint.projection_matrices
    image_sizes = checkpoint.image_sizes

    if camera_to_world_matrices is None:
        raise ValueError("Camera to world matrices are not available in the checkpoint.")
    if projection_matrices is None:
        raise ValueError("Projection matrices are not available in the checkpoint.")
    if image_sizes is None:
        raise ValueError("Image sizes are not available in the checkpoint.")

    points_list = []
    colors_list = []

    enumerator = (
        tqdm.tqdm(range(len(camera_to_world_matrices)), unit="imgs", desc="Extracting Point Cloud")
        if show_progress
        else range(len(camera_to_world_matrices))
    )

    total_points = 0
    for i in enumerator:
        cam_to_world_matrix = camera_to_world_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        world_to_cam_matrix = torch.linalg.inv(cam_to_world_matrix).contiguous()
        projection_matrix = projection_matrices[i].to(model.device).to(dtype=torch.float32, device=device)
        inv_projection_matrix = torch.linalg.inv(projection_matrix).contiguous()

        image_size = image_sizes[i]
        image_width = int(image_size[1].item())
        image_height = int(image_size[0].item())

        # We set near and far planes to 0.0 and 1e10 respectively to avoid clipping
        # in the rendering process. Instead, we will use the provided near and far planes
        # to filter the depth images after rendering so pixels out of range will not be accumulated
        feature_and_depth, alpha = model.render_images_and_depths(
            world_to_camera_matrices=world_to_cam_matrix.unsqueeze(0),
            projection_matrices=projection_matrix.unsqueeze(0),
            image_width=image_width,
            image_height=image_height,
            near=0.0,
            far=1e10,
        )

        feature_image = feature_and_depth[..., : model.num_channels].squeeze(0)
        depth_image = (feature_and_depth[..., -1].unsqueeze(-1) / alpha.clamp(min=1e-10)).squeeze()  # [H, W]

        assert feature_image.shape == (image_height, image_width, model.num_channels)
        assert depth_image.shape == (image_height, image_width)

        mask = ((depth_image > near) & (depth_image < far)).squeeze(-1)  # [H, W]
        # TODO: Add GPU Canny edge detection
        if canny_edge_std > 0.0:
            canny_mask = torch.tensor(
                morphology.dilation(
                    feature.canny(depth_image.squeeze(-1).cpu().numpy(), sigma=canny_edge_std),
                    footprint=np.ones((canny_mask_dilation, canny_mask_dilation)),
                )
                == 0,
                device=device,
            )
            mask = mask & canny_mask

        # Unproject depth image to camera space coordinates
        row, col = torch.meshgrid(
            torch.arange(0, image_height, device=device, dtype=torch.float32),
            torch.arange(0, image_width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        cam_pts = torch.stack([col, row, torch.ones_like(row)])  # [3, H, W]
        cam_pts = inv_projection_matrix @ cam_pts.view(3, -1)  # [3, H, W]
        cam_pts = cam_pts.view(3, image_height, image_width) * depth_image.unsqueeze(0)  # [3, H, W]

        # Transform camera space coordinates to world coordinates
        world_pts = torch.cat(
            [cam_pts, torch.ones(1, cam_pts.shape[1], cam_pts.shape[2]).to(cam_pts)], dim=0
        )  # [4, H, W]
        world_pts = cam_to_world_matrix @ world_pts.view(4, -1)  # [4, H, W]
        world_pts = world_pts[:3] / world_pts[3].unsqueeze(0)  # [3, H * W]
        world_pts = world_pts.view(3, image_height, image_width).permute(1, 2, 0)  # [H, W, 3]

        # Optionally downsample the world points and feature image
        world_pts = world_pts[::depth_image_downsample_factor, ::depth_image_downsample_factor, :]
        feature_image = feature_image[::depth_image_downsample_factor, ::depth_image_downsample_factor, :]
        mask = mask[::depth_image_downsample_factor, ::depth_image_downsample_factor]

        world_pts = world_pts[mask].view(-1, 3)  # [num_points, 3]
        features = feature_image[mask]  # [num_points, C]

        if world_pts.numel() == 0:
            continue

        assert world_pts.shape[0] == features.shape[0], "Number of points and features must match."

        if show_progress:
            assert isinstance(enumerator, tqdm.tqdm)
            enumerator.set_postfix({"total_points": total_points})

        points_list.append(world_pts.to(dtype))
        colors_list.append(features.to(dtype))
        total_points += points_list[-1].shape[0]

    return torch.cat(points_list, dim=0).to(dtype), torch.cat(colors_list, dim=0).to(dtype).clip_(min=0.0, max=1.0)
