# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TextIO

import cv2
import numpy as np
import torch
from fvdb import GaussianSplat3d
from fvdb.types import NumericScalar, to_FloatingScalar


class GaussianSplatReconstructionBaseWriter(ABC):
    """
    Base class for logging and saving data during Gaussian splat reconstruction.

    This class defines the interface for logging metrics, saving images, checkpoints, and PLY files during
    Gaussian splat reconstruction. Concrete implementations must implement all abstract methods.

    To implement custom logging/saving behavior, subclass this class and implement the abstract methods.
    """

    @abstractmethod
    def log_metric(self, global_step: int, metric_name: str, metric_value: NumericScalar) -> None:
        """
        Abstract method to log a scalar metric value. This function is called during reconstruction to log metrics such as loss, PSNR, etc.

        Args:
            global_step (int): The global step at which the metric is being logged.
            metric_name (str): The name of the metric being logged.
            metric_value (NumericScalar): The value of the metric being logged. Must be a scalar type (int, float, np.number, torch.number, etc.).
        """
        pass

    @abstractmethod
    def save_image(self, global_step: int, image_name: str, image: torch.Tensor) -> None:
        """
        Abstract method to save an image. This function is called during reconstruction to save images such as rendered outputs or intermediate results.

        Args:
            global_step (int): The global step at which the image is being saved.
            image_name (str): The name of the image being saved.
            image (torch.Tensor): The image tensor to be saved.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, global_step: int, checkpoint_name: str, checkpoint: dict[str, Any]) -> None:
        """
        Abstract method to save a checkpoint. This function is called during reconstruction to save model checkpoints.

        Args:
            global_step (int): The global step at which the checkpoint is being saved.
            checkpoint_name (str): The name of the checkpoint being saved.
            checkpoint (dict[str, Any]): The checkpoint data to be saved.
        """
        pass

    @abstractmethod
    def save_ply(
        self, global_step: int, ply_name: str, model: GaussianSplat3d, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Abstract method to save a Gaussian splat model to a PLY file. This function is called during reconstruction to save the current state of the model.

        Args:
            global_step (int): The global step at which the PLY file is being saved.
            ply_name (str): The name of the PLY file being saved.
            model (GaussianSplat3d): The Gaussian splat model to be saved.
            metadata (dict[str, Any] | None): Optional metadata to be saved with the PLY file.
        """
        pass


@dataclass
class GaussianSplatReconstructionWriterConfig:
    """
    Parameters for configuring the behavior of a  :class:`GaussianSplatReconstructionWriter`.
    Controls what data gets saved to disk, how much buffering to use, and whether to use TensorBoard.
    """

    # Whether to save images to disk
    save_images: bool = False
    """
    Whether to save images to disk. If ``False``, images will not be saved to disk.

    Default is ``False``.
    """

    # Whether to save checkpoints to disk
    save_checkpoints: bool = True
    """
    Whether to save checkpoints to disk. If ``False``, checkpoints will not be saved to disk.

    Default is ``True``.
    """

    # Whether to save PLY files to disk
    save_plys: bool = True
    """
    Whether to save PLY files to disk. If ``False``, PLY files will not be saved to disk.

    Default is ``True``.
    """

    # Whether to save metrics to a CSV file
    save_metrics: bool = True
    """
    Whether to save metrics to a CSV file. If ``False``, metrics will not be saved to a CSV file.

    Default is ``True``.
    """

    # How much buffering to use for metrics file logging
    metrics_file_buffer_size: int = 8 * 1024 * 1024  # 8 MB
    """
    How much buffering (in bytes) to use for metrics file logging. Larger values can improve performance when logging many metrics.

    Default is 8 MiB.
    """

    # Whether to use TensorBoard for logging metrics and images
    use_tensorboard: bool = False
    """
    Whether to use TensorBoard for logging metrics and images. If ``True``, metrics and images will be logged to TensorBoard.

    Default is ``False``.
    """

    # Whether to also save images to TensorBoard if use_tensorboard is True
    save_images_to_tensorboard: bool = False
    """
    Whether to also save images to TensorBoard if :obj:`use_tensorboard` is ``True``. If ``True``, images will be saved to TensorBoard.

    Default is ``False``.
    """


class GaussianSplatReconstructionWriter(GaussianSplatReconstructionBaseWriter):
    """
    Class to handle logging and saving data during Gaussian splat reconstruction.
    This class is responsible for saving, checkpoints, PLY files, images, and metrics.
    It can also log metrics and images to TensorBoard if requested.

    .. code-block:: text

        save_path/run_name/
            checkpoints/
                <step>/
                    <first_checkpoint>.pth
                    <second_checkpoint>.pth
                    ...
                <step>/
                    ...
            ply/
                <step>/
                    <first_ply>.ply
                    <second_ply>.ply
                    ...
                <step>/
                    ...
            images/
                <step>/
                    <first_image>.png
                    <second_image>.png
                    ...
                <step>/
                    ...
            tensorboard/
                events.out.tfevents...
            metrics_log.csv

    """

    def __init__(
        self,
        run_name: str | None,
        save_path: pathlib.Path | None,
        exist_ok: bool = False,
        config: GaussianSplatReconstructionWriterConfig = GaussianSplatReconstructionWriterConfig(),
    ) -> None:
        """
        Create a new :class:`GaussianSplatReconstructionWriter` instance, which can log and save data during reconstruction.

        Args:
            run_name (str | None): Name of this reconstruction run. If None, a unique name will be generated.
            save_path (pathlib.Path | None): Path to the directory where results should be saved.
                If None, no data will be saved to disk.
            exist_ok (bool): Whether to keep existing data in the save_path/run_name directory if it exists. If ``False``,
                an error will be raised if the directory already exists. Default is ``False``.
            config (GaussianSplatReconstructionWriterConfig): Configuration parameters for what data to save and how.

        """
        super().__init__()
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._config = config

        # There are several modes of operation:
        # 1. You provide a save_path, but no run_name -> we create a unique run_name and directory
        # 2. You provide neither save_path nor run_name -> we create a unique run_name, but do not save anything
        # 3. You provide both save_path and run_name -> we use them, and create the directory if it does not exist.
        #    If it exists, we raise an error unless exist_ok is True (then we overwrite).
        # 4. You provide a run_name, but no save_path -> we use the run_name, but do not save anything
        if run_name is None and save_path is not None:
            # You are saving data, but did not provide a run name
            save_path = save_path.resolve()
            run_name, save_path = self._make_unique_results_directory_based_on_time(save_path, prefix="run")
        elif run_name is None and save_path is None:
            # You are not saving data, and did not provide a run name
            save_path = None
            run_name = None
        elif run_name is not None and save_path is not None:
            # You are saving data, and provided a run name, so use it
            save_path = (save_path / run_name).resolve()
            if not exist_ok and save_path.exists():
                raise FileExistsError(f"Directory {save_path} already exists. Use exist_ok=True to overwrite.")
            save_path.mkdir(parents=True, exist_ok=exist_ok)
        else:
            # You are not saving data, but provided a run name, so use it as a tag only
            run_name = run_name
            save_path = None

        self._run_name = run_name
        self._save_path = save_path

        # Create directory to save PLY files if requested
        self._ply_path: pathlib.Path | None = None
        if self._config.save_plys and self._save_path is not None:
            self._ply_path = self._save_path / "ply"
            self._ply_path.mkdir(parents=True, exist_ok=True)

        # Create directory to save checkpoints if requested
        self._checkpoints_path: pathlib.Path | None = None
        if self._config.save_checkpoints and self._save_path is not None:
            self._checkpoints_path = self._save_path / "checkpoints"
            self._checkpoints_path.mkdir(parents=True, exist_ok=True)

        # Create directory to save images if requested
        self._images_path: pathlib.Path | None = None
        if self._config.save_images and self._save_path is not None:
            self._images_path = self._save_path / "images"
            self._images_path.mkdir(parents=True, exist_ok=True)

        # Create buffered, append-only file to log metrics if requested
        self._metrics_log_file_handle: TextIO | None = None
        if self._config.save_metrics and self._save_path is not None:
            self._metrics_path = self._save_path / "metrics_log.csv"
            self._metrics_log_file_handle = open(
                self._metrics_path, "a", buffering=self._config.metrics_file_buffer_size
            )

        # Create directory for tensorboard logs and SummaryWriter if requested
        self._tensorboard_path: pathlib.Path | None = None
        if self._config.use_tensorboard and self._save_path is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tensorboard_path = self._save_path / "tensorboard"
                self._tensorboard_path.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=str(self._tensorboard_path))
            except ImportError as e:
                self._logger.warning(
                    "TensorBoard logging is enabled, but torch.utils.tensorboard is not available. "
                    "Disabling TensorBoard logging."
                )
                self._tensorboard_path = None
                self._tb_writer = None

    @staticmethod
    def _to_batched_uint8_image(image: torch.Tensor):
        """
        Convert a torch.Tensor image of shape   ``(H, W)``, ``(H, W, C)`` or ``(B, H, W, C)`` to a batched uint8 image with shape ``(B, H, W, C)``.

        Handles error checking and conversion from floating point to uint8 if necessary.

        Args:
            image (torch.Tensor): Input image tensor. Must have shape ``(H, W)``, ``(H, W, C)`` or ``(B, H, W, C)`` and a floating point or uint8 dtype.
                If the image is 2D ``(H, W)``, it is treated as a grayscale image with 1 channel.
                If the image is 3D ``(H, W, C)``, it is treated as a single image with ``C`` channels.
                If the image is 4D ``(B, H, W, C)``, it is treated as a batch of ``B`` images with ``C`` channels.
        Returns:
            torch.Tensor: Batched uint8 image tensor with shape ``(B, H, W, C)`` and dtype torch.uint8.

        """
        # Ensure image is a torch.Tensor with 2 (H, W), 3 (H, W, C) or 4 (B, H, W, C) dimensions
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be a torch.Tensor")
        if image.ndim not in (2, 3, 4):
            raise ValueError("Image must be 2D (H, W), 3D (H, W, C) or 4D (B, H, W, C)")

        # Check that image has valid number of channels (1, 3 or 4)
        num_channels: int = image.shape[-1] if image.ndim in (3, 4) else 1
        if num_channels not in (1, 3, 4):
            raise ValueError(
                f"Image must have shape (H, W), (H, W, C) or (B, H, W, C) with C=1, C=3 or C=4. Got Invalid shape {image.shape} resulting in C={num_channels}."
            )
        # Add channel dimension if image is 2D grayscale
        if image.ndim == 2:
            # Add channel dimension
            image = image.unsqueeze(-1)

        # Add batch dimension if image is 3D (H, W, C)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Now image is guaranteed to be 4D (B, H, W, C), but check again
        assert image.ndim == 4, "Image must be 4D (B, H, W, C) after adding batch dimension"

        # If you pass in a floating point image, convert to uint8 since we currently only support saving PNG and JPEG
        if image.is_floating_point():
            # Convert floating point images to uint8
            image = (image * 255.0).clip(0, 255).to(torch.uint8)

        # Ensure image is of type uint8 now (either originally or converted from floating point)
        if image.dtype != torch.uint8:
            raise ValueError(f"Image must be of type torch.uint8 or floating point. Got {image.dtype}.")

        return image

    @staticmethod
    def _resolve_saved_file(
        base_path: pathlib.Path,
        global_step: int,
        file_name: str,
        file_type: str,
        allowed_sufixes: tuple[str, ...],
        default_suffix: str,
    ) -> pathlib.Path:
        """
        Resolve the full path for a file to be saved at a given global step, and create necessary directories needed to save it.
        Ensures the file name is safe and valid, and that the file has an allowed suffix/extension.

        Args:
            base_path (pathlib.Path): Base directory where the file should be saved.
            global_step (int): Global step at which the file is being saved. A subdirectory with this name will be created under base_path.
            file_name (str): Name of the file to be saved. Can include subdirectories (*e.g.* ``"reconstruction/image_0001.png"``).
            file_type (str): Type of the file used for error messages (*e.g.* ``"Image"``, ``"Checkpoint"``).
            allowed_sufixes (tuple[str, ...]): Allowed file suffixes/extensions.
            default_suffix (str): Default file suffix/extension to use if none is provided.

        Returns:
            pathlib.Path: Full path where the file should be saved. All necessary directories will be created.
        """
        # Resolve the path for the file to be saved at this global step
        step_file_path = (base_path / f"{global_step:08d}" / file_name).resolve()

        # Ensure the file path is within the base directory
        # This prevents directory traversal attacks (e.g. the user passes in "../../etc/passwd")
        if not step_file_path.is_relative_to(base_path):
            raise ValueError(f"{file_type} name {file_name} results in a path outside of {base_path.name} directory.")

        # Create parent directory if it does not exist. If the user passes in a nested path, we need to create it.
        # e.g. file_name = "reconstruction/image_0001.png" will create "images/reconstruction/" directory
        if not step_file_path.parent.exists():
            step_file_path.parent.mkdir(parents=True, exist_ok=True)

        # If no suffix is provided, add the default suffix
        if step_file_path.suffix == "":
            step_file_path = step_file_path.with_suffix(default_suffix)

        # Ensure the file has a valid suffix
        if step_file_path.suffix not in allowed_sufixes:
            raise ValueError(
                f"{file_type} name {file_name} must have one of the following extensions: {allowed_sufixes}. Got {step_file_path.suffix}."
            )

        # Return the resolved file path
        return step_file_path

    def _make_unique_results_directory_based_on_time(
        self, base_path: pathlib.Path, prefix: str
    ) -> tuple[str, pathlib.Path]:
        """
        Generate a unique results directory based on the current time, and return its name and path.

        The results directory will be created under ``base_path`` with a name in the format
        ``prefix_YYYY-MM-DD-HH-MM-SS``. If a directory with the same name already exists,
        it will attempt to create a new one by appending an incremented number to the name

        Returns:
            run_name: The name of a unique log directory for a specific run in the format ``run_YYYY-MM-DD-HH-MM-SS``.
            log_path: A pathlib.Path object pointing to the created log directory.
        """
        attempts = 0
        max_attempts = 50
        run_name = f"{prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        while attempts < 50:
            log_path = base_path / run_name
            try:
                log_path.mkdir(exist_ok=False, parents=True)
                break
            except FileExistsError:
                attempts += 1
                self._logger.debug(f"Results directory {log_path} already exists. Attempting to create a new one.")
                # Generate a new run name with an incremented attempt number
                run_name = f"{prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}_{attempts+1:02d}"
                continue
        if attempts >= max_attempts:
            raise FileExistsError(f"Failed to generate a unique log directory name after {max_attempts} attempts.")

        self._logger.info(f"Created unique log directory with name {run_name} after {attempts} attempts.")

        return run_name, log_path

    @property
    def run_name(self) -> str | None:
        """
        Return the name of this reconstruction run, or ``None`` if the writer is not saving any data. The name of the run matches
        the name of the directory where logged results are being saved.

        Returns:
            str | None: The name of this reconstruction run, or ``None`` if the writer is not saving any data.
        """
        return self._run_name

    @property
    def log_path(self) -> pathlib.Path | None:
        """
        Return the path where logged results are being saved, or ``None`` if no results are being saved.

        Returns:
            log_path (pathlib.Path | None): The path where logged results are being saved, or ``None`` if
                no logged results are being saved.
        """
        return self._save_path

    @torch.no_grad()
    def log_metric(self, global_step: int, metric_name: str, metric_value: NumericScalar) -> None:
        """
        Log a scalar metric value. This function is called during reconstruction to log metrics such as loss, PSNR, etc.

        Args:
            global_step (int): The global step at which the metric is being logged.
            metric_name (str): The name of the metric being logged.
            metric_value (NumericScalar): The value of the metric being logged.
        """
        # Append metric to CSV file if requested
        if self._config.save_metrics and self._metrics_log_file_handle is not None:
            metric_value = to_FloatingScalar(metric_value).item()
            self._metrics_log_file_handle.write(f"{global_step},{metric_name},{metric_value}\n")

        # Log metric to TensorBoard if requested
        if self._config.use_tensorboard and hasattr(self, "_tb_writer") and self._tb_writer is not None:
            metric_value = to_FloatingScalar(metric_value).item()
            self._tb_writer.add_scalar(metric_name, metric_value, global_step)

    @torch.no_grad()
    def save_image(self, global_step: int, image_name: str, image: torch.Tensor, jpeg_quality: int = 98):
        """
        Save an image to disk and/or TensorBoard. This function is called during reconstruction to save
        rendered images, error maps, etc.

        Args:
            global_step (int): The global step at which the image is being saved.
            image_name (str): The name of the image being saved. This will be used as the file name. Must have a ``.png``
                or ``.jpg``/``.jpeg`` suffix.
            image (torch.Tensor): The image tensor to be saved. Must have shape ``(H, W)``, ``(H, W, C)`` or ``(B, H, W, C)`` and
                have a floating point or uint8 dtype.
            jpeg_quality (int): Quality of JPEG images if saving as JPEG. Must be between 0 and 100. Default is 98.
        """
        # This function only does something if you are saving images to disk or TensorBoard
        if not self._config.save_images and not (
            self._config.use_tensorboard and self._config.save_images_to_tensorboard
        ):
            return

        image = self._to_batched_uint8_image(image)  # (B, H, W, C)

        # If you are saving images to disk, do so now
        if self._config.save_images and self._images_path is not None:
            image_path = self._resolve_saved_file(
                base_path=self._images_path,
                global_step=global_step,
                file_name=image_name,
                file_type="Image",
                allowed_sufixes=(".png", ".jpg", ".jpeg"),
                default_suffix=".png",
            )

            # Ensure the image path is within the images directory
            # This prevents directory traversal attacks (e.g. the user passes in "../../etc/passwd")
            if not image_path.is_relative_to(self._images_path):
                raise ValueError(f"Image name {image_name} results in path outside of images directory.")

            # Create parent directory if it does not exist. If the user passes in a nested path, we need to create it.
            # e.g. image_name = "reconstruction/image_0001.png" will create "images/reconstruction/" directory
            if not image_path.parent.exists():
                image_path.parent.mkdir(parents=True, exist_ok=True)

            # Image is always 4D here (B, H, W, C)
            batch_size: int = image.shape[0]
            num_channels: int = image.shape[-1]
            for b in range(batch_size):
                image_batch_path = (
                    image_path.parent / f"{image_path.stem}_{b:04d}{image_path.suffix}"
                    if batch_size > 1
                    else image_path
                )

                image_np = image[b].cpu().numpy()
                if num_channels == 1:
                    image_np = image_np[:, :, 0]  # Remove channel dimension for grayscale
                if num_channels == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

                if image_batch_path.suffix.lower() == ".png":
                    # Save as PNG
                    cv2.imwrite(str(image_batch_path), image_np)
                elif image_batch_path.suffix.lower() == ".jpg" or image_batch_path.suffix.lower() == ".jpeg":
                    # Save as JPEG
                    cv2.imwrite(str(image_batch_path), image_np, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                else:
                    raise ValueError(f"Unsupported image format {image_batch_path.suffix} for saving. Use .png or .jpg")

        if self._config.use_tensorboard and self._config.save_images_to_tensorboard and hasattr(self, "_tb_writer"):
            if hasattr(self, "_tb_writer") and self._tb_writer is not None:
                self._tb_writer.add_images(image_name, image.permute(0, 3, 1, 2).contiguous(), global_step)

    @torch.no_grad()
    def save_checkpoint(self, global_step: int, checkpoint_name: str, checkpoint: dict[str, Any]) -> None:
        """
        Save a reconstruction checkpoint to disk. This function is called during reconstruction to save model and optimizer state.

        Args:
            global_step (int): The global step at which the checkpoint is being saved.
            checkpoint_name (str): The name of the checkpoint file. This will be used as the file name. Must have a ``.pth`` or ``.pt`` suffix.
            checkpoint (dict[str, Any]): The checkpoint dictionary to be saved. Typically contains model state, optimizer state, etc.
        """
        if self._config.save_checkpoints and self._checkpoints_path is not None:
            ckpt_path = self._resolve_saved_file(
                base_path=self._checkpoints_path,
                global_step=global_step,
                file_name=checkpoint_name,
                file_type="Checkpoint",
                allowed_sufixes=(".pth", ".pt"),
                default_suffix=".pt",
            )

            # Save checkpoint using torch.save
            torch.save(checkpoint, ckpt_path)

    @torch.no_grad()
    def save_ply(
        self, global_step: int, ply_name: str, model: GaussianSplat3d, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Save the current Gaussian splat model to a PLY file. This function is called during reconstruction to save
        the reconstructed model at various stages.

        Args:
            global_step (int): The global step at which the PLY file is being saved.
            ply_name (str): The name of the PLY file. This will be used as the file name. Must have a ``.ply`` suffix.
            model (GaussianSplat3d): The Gaussian splat model to be saved.
            metadata (dict[str, Any] | None): Optional metadata to include in the PLY file (e.g. camera parameters, reconstruction config, etc.).
        """
        if self._config.save_plys and self._ply_path is not None:
            ply_path = self._resolve_saved_file(
                base_path=self._ply_path,
                global_step=global_step,
                file_name=ply_name,
                file_type="PLY",
                allowed_sufixes=(".ply",),
                default_suffix=".ply",
            )

            # Save PLY using model's built-in save function
            model.save_ply(str(ply_path), metadata=metadata)
