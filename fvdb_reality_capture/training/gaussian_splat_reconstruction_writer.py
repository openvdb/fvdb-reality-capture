import logging
import pathlib
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TextIO

import cv2
import numpy as np
import torch
from fvdb import GaussianSplat3d
from fvdb.types import NumericScalar, to_FloatingScalar


class GaussianReconstructionBaseWriter(ABC):
    """
    Base class for logging and saving data during Gaussian splat reconstruction.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def log_metric(self, global_step: int, metric_name: str, metric_value: NumericScalar):
        pass

    @abstractmethod
    def save_image(self, global_step: int, image_name: str, image: torch.Tensor):
        pass

    @abstractmethod
    def save_checkpoint(self, global_step: int, checkpoint_name: str, checkpoint: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def save_ply(
        self, global_step: int, ply_name: str, model: GaussianSplat3d, metadata: dict[str, Any] | None = None
    ) -> None:
        pass


@dataclass
class GaussianReconstructionWriterConfig:
    # Configure what kind of data gets saved to disk
    save_images: bool = False
    save_checkpoints: bool = True
    save_plys: bool = True
    save_metrics: bool = True

    # How much buffering to use for metrics file logging
    metrics_file_buffer_size: int = 8 * 1024 * 1024  # 8 MB

    use_tensorboard: bool = False
    save_images_to_tensorboard: bool = False


class GaussianReconstructionWriter(GaussianReconstructionBaseWriter):
    """
    Class to handle logging and saving data during Gaussian splat reconstruction.
    This class is responsible for saving, checkpoints, PLY files, images, and metrics.
    It can also log metrics and images to TensorBoard if requested.
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
        config: GaussianReconstructionWriterConfig = GaussianReconstructionWriterConfig(),
    ) -> None:
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
            # You are not saving data, and did not provide a run name, so generate one
            run_name = str(uuid.uuid4())
        elif run_name is not None and save_path is not None:
            # You are saving data, and provided a run name, so use it
            save_path = (save_path / run_name).resolve()
            if not exist_ok and save_path.exists():
                raise FileExistsError(f"Directory {save_path} already exists. Use exist_ok=True to overwrite.")
            save_path.mkdir(parents=True, exist_ok=exist_ok)
        else:
            # You are not saving data, but provided a run name, so just use it
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
        # Resolve the path for the file to be saved at this global step
        step_file_path = (base_path / f"{global_step:08d}" / file_name).resolve()

        # Ensure the file path is within the base directory
        # This prevents directory traversal attacks (e.g. the user passes in "../../etc/passwd")
        if not step_file_path.is_relative_to(base_path):
            raise ValueError(f"{file_type} name {file_name} results in a path outside of {base_path.name} directory.")

        # Create parent directory if it does not exist. If the user passes in a nested path, we need to create it.
        # e.g. file_name = "train/image_0001.png" will create "images/train/" directory
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

        The results directory will be created under `base_path` with a name in the format
        `prefix_YYYY-MM-DD-HH-MM-SS`. If a directory with the same name already exists,
        it will attempt to create a new one by appending an incremented number to the name

        Returns:
            run_name: The name of a unique results directory in the format "run_YYYY-MM-DD-HH-MM-SS".
            run_path: A pathlib.Path object pointing to the created results directory.
        """
        attempts = 0
        max_attempts = 50
        run_name = f"{prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        while attempts < 50:
            results_path = base_path / run_name
            try:
                results_path.mkdir(exist_ok=False, parents=True)
                break
            except FileExistsError:
                attempts += 1
                self._logger.debug(f"Results directory {results_path} already exists. Attempting to create a new one.")
                # Generate a new run name with an incremented attempt number
                run_name = f"{prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}_{attempts+1:02d}"
                continue
        if attempts >= max_attempts:
            raise FileExistsError(f"Failed to generate a unique results directory name after {max_attempts} attempts.")

        self._logger.info(f"Created unique results directory with name {run_name} after {attempts} attempts.")

        return run_name, results_path

    @property
    def run_name(self) -> str | None:
        return self._run_name

    @property
    def results_path(self) -> pathlib.Path | None:
        return self._save_path

    @torch.no_grad()
    def log_metric(self, global_step: int, metric_name: str, metric_value: NumericScalar):
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
            # e.g. image_name = "train/image_0001.png" will create "images/train/" directory
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

                if image_batch_path.suffix.lower() == ".png":
                    # Save as PNG
                    image_np = image[b].cpu().numpy()
                    if num_channels == 1:
                        image_np = image_np[:, :, 0]  # Remove channel dimension for grayscale
                    cv2.imwrite(str(image_batch_path), image_np)
                elif image_batch_path.suffix.lower() == ".jpg" or image_batch_path.suffix.lower() == ".jpeg":
                    # Save as JPEG
                    image_np = (image[b].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    if num_channels == 1:
                        image_np = image_np[:, :, 0]  # Remove channel dimension for grayscale
                    cv2.imwrite(str(image_batch_path), image_np, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                else:
                    raise ValueError(f"Unsupported image format {image_batch_path.suffix} for saving. Use .png or .jpg")

        if self._config.use_tensorboard and self._config.save_images_to_tensorboard and hasattr(self, "_tb_writer"):
            if hasattr(self, "_tb_writer") and self._tb_writer is not None:
                self._tb_writer.add_images(image_name, image, global_step)

    @torch.no_grad()
    def save_checkpoint(self, global_step: int, checkpoint_name: str, checkpoint: dict[str, Any]) -> None:
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
