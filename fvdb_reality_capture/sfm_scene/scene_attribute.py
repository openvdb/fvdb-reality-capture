# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypeVar

import numpy as np

REGISTERED_SCENE_ATTRIBUTES: dict[str, type["SceneAttribute"]] = {}

DerivedAttribute = TypeVar("DerivedAttribute", bound=type)


def scene_attribute(cls: DerivedAttribute) -> DerivedAttribute:
    """
    Decorator to register a scene attribute class for serialization.

    Mirrors the ``@transform`` decorator pattern used by
    :class:`~fvdb_reality_capture.transforms.base_transform.BaseTransform`.

    Args:
        cls: The attribute class to register. Must be a subclass of :class:`SceneAttribute`.

    Returns:
        cls: The registered attribute class.
    """
    if not issubclass(cls, SceneAttribute):
        raise TypeError(f"Scene attribute {cls} must inherit from SceneAttribute.")

    REGISTERED_SCENE_ATTRIBUTES[cls.type_name()] = cls
    return cls


class InterpolationMode(str, Enum):
    """Interpolation modes for resizing raster data."""

    AREA = "area"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    NEAREST = "nearest"
    LINEAR = "linear"


class SceneAttribute(ABC):
    """
    Abstract base class for custom attributes attached to an :class:`SfmScene`.

    Subclasses define how the attribute responds to scene operations via hook
    methods.  The base class provides no-op defaults for every hook so that
    new operations can be added without breaking existing attribute types.

    All hook methods (``on_filter_points``, ``on_downsample_images``, etc.)
    are designed to be overridden by subclasses that need custom behavior.
    For example, subclass :class:`PerImageRasterAttribute` and override
    :meth:`on_downsample_images` to implement a domain-specific
    downsampling strategy for raster data that cannot use standard
    interpolation.
    """

    @staticmethod
    @abstractmethod
    def type_name() -> str:
        """Return a unique string identifier used for serialization."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """Serialize the attribute to a dictionary compatible with the scene's
        serialization mechanism (e.g. pickle / ``torch.save``).

        The returned dict does not need to be JSON-serializable; it may
        contain NumPy arrays or other objects supported by ``SfmScene``
        serialization.
        """
        ...

    @staticmethod
    @abstractmethod
    def from_state_dict(state_dict: dict) -> "SceneAttribute":
        """Reconstruct the attribute from a serialization-compatible state dictionary."""
        ...

    # -- Validation ----------------------------------------------------------

    def validate(self, attr_name: str, num_points: int, num_images: int, camera_ids: set[int]) -> None:
        """Validate sizes against the owning scene. No-op by default."""

    # -- Core hooks (dispatched by SfmScene methods) -------------------------

    def on_filter_points(self, mask: np.ndarray) -> "SceneAttribute":
        return self

    def on_filter_images(self, mask: np.ndarray) -> "SceneAttribute":
        return self

    def on_select_images(self, indices: np.ndarray) -> "SceneAttribute":
        return self

    def on_spatial_transform(self, matrix: np.ndarray) -> "SceneAttribute":
        return self

    # -- Transform-specific hooks --------------------------------------------
    # Override these in subclasses to implement custom behavior (e.g. a
    # domain-specific downsampling strategy).

    def on_downsample_images(self, attr_name: str, downsample_factor: int, output_cache: Any) -> "SceneAttribute":
        """Called when images are downsampled.  Override to implement custom
        resizing logic for attribute data that cannot use standard interpolation.

        Args:
            attr_name: Name under which this attribute is registered on the scene.
            downsample_factor: Integer factor by which images are being reduced.
            output_cache: The :class:`SfmCache` of the output scene, available
                for writing downsampled files.

        Returns:
            A new (or the same) attribute instance with appropriately resized data.
        """
        return self

    def on_crop_scene(self, attr_name: str, bbox: np.ndarray, output_cache: Any) -> "SceneAttribute":
        """Called when the scene is spatially cropped.  Override to implement
        custom crop behavior.

        Args:
            attr_name: Name under which this attribute is registered on the scene.
            bbox: The crop bounding box as a ``(6,)`` array ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
            output_cache: The :class:`SfmCache` of the output scene.

        Returns:
            A new (or the same) attribute instance with appropriately cropped data.
        """
        return self


# ---------------------------------------------------------------------------
# Concrete attribute types
# ---------------------------------------------------------------------------


@scene_attribute
class PerPointAttribute(SceneAttribute):
    """Per-point data that varies across the scene's 3D point cloud."""

    def __init__(self, data: np.ndarray, transform_mode: str = "none"):
        if transform_mode not in ("none", "rotate", "rigid"):
            raise ValueError(f"transform_mode must be 'none', 'rotate', or 'rigid', got '{transform_mode}'")
        self._data = data
        self._transform_mode = transform_mode

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def transform_mode(self) -> str:
        return self._transform_mode

    @staticmethod
    def type_name() -> str:
        return "PerPointAttribute"

    def validate(self, attr_name: str, num_points: int, num_images: int, camera_ids: set[int]) -> None:
        if self._data.shape[0] != num_points:
            raise ValueError(
                f"Attribute '{attr_name}': expected data.shape[0] == {num_points} (num_points), "
                f"got {self._data.shape[0]}"
            )

    def on_filter_points(self, mask: np.ndarray) -> "PerPointAttribute":
        return PerPointAttribute(self._data[mask], transform_mode=self._transform_mode)

    def on_spatial_transform(self, matrix: np.ndarray) -> "PerPointAttribute":
        if self._transform_mode == "none":
            return self

        linear = matrix[:3, :3]

        if self._transform_mode == "rotate":
            # Extract pure rotation by factoring out scale via polar decomposition.
            U, _, Vt = np.linalg.svd(linear)
            R_pure = U @ Vt
            # Ensure proper rotation (det = +1)
            if np.linalg.det(R_pure) < 0:
                U[:, -1] *= -1
                R_pure = U @ Vt
            new_data = self._data @ R_pure.T
            return PerPointAttribute(new_data, transform_mode=self._transform_mode)

        # "rigid" – full affine
        translation = matrix[:3, 3]
        new_data = self._data @ linear.T + translation
        return PerPointAttribute(new_data, transform_mode=self._transform_mode)

    def state_dict(self) -> dict:
        return {
            "data": self._data.tolist(),
            "transform_mode": self._transform_mode,
        }

    @staticmethod
    def from_state_dict(state_dict: dict) -> "PerPointAttribute":
        return PerPointAttribute(
            data=np.array(state_dict["data"]),
            transform_mode=state_dict.get("transform_mode", "none"),
        )


@scene_attribute
class PerImageValueAttribute(SceneAttribute):
    """Lightweight per-image values stored in memory."""

    def __init__(self, values: list):
        self._values = list(values)

    @property
    def values(self) -> list:
        return self._values

    @staticmethod
    def type_name() -> str:
        return "PerImageValueAttribute"

    def validate(self, attr_name: str, num_points: int, num_images: int, camera_ids: set[int]) -> None:
        if len(self._values) != num_images:
            raise ValueError(
                f"Attribute '{attr_name}': expected len(values) == {num_images} (num_images), "
                f"got {len(self._values)}"
            )

    def on_filter_images(self, mask: np.ndarray) -> "PerImageValueAttribute":
        return PerImageValueAttribute([v for v, keep in zip(self._values, mask) if keep])

    def on_select_images(self, indices: np.ndarray) -> "PerImageValueAttribute":
        return PerImageValueAttribute([self._values[i] for i in indices])

    def state_dict(self) -> dict:
        return {"values": self._values}

    @staticmethod
    def from_state_dict(state_dict: dict) -> "PerImageValueAttribute":
        return PerImageValueAttribute(values=state_dict["values"])


@scene_attribute
class PerImageRasterAttribute(SceneAttribute):
    """Per-image raster data stored as files on disk, spatially aligned to images."""

    def __init__(
        self,
        paths: list[str],
        resize_interpolation: InterpolationMode = InterpolationMode.AREA,
        file_type: str = "png",
    ):
        self._paths = list(paths)
        self._resize_interpolation = (
            InterpolationMode(resize_interpolation)
            if not isinstance(resize_interpolation, InterpolationMode)
            else resize_interpolation
        )
        self._file_type = file_type

    @property
    def paths(self) -> list[str]:
        return self._paths

    @property
    def resize_interpolation(self) -> InterpolationMode:
        return self._resize_interpolation

    @property
    def file_type(self) -> str:
        return self._file_type

    @staticmethod
    def type_name() -> str:
        return "PerImageRasterAttribute"

    def validate(self, attr_name: str, num_points: int, num_images: int, camera_ids: set[int]) -> None:
        if len(self._paths) != num_images:
            raise ValueError(
                f"Attribute '{attr_name}': expected len(paths) == {num_images} (num_images), " f"got {len(self._paths)}"
            )

    def on_filter_images(self, mask: np.ndarray) -> "PerImageRasterAttribute":
        return PerImageRasterAttribute(
            paths=[p for p, keep in zip(self._paths, mask) if keep],
            resize_interpolation=self._resize_interpolation,
            file_type=self._file_type,
        )

    def on_select_images(self, indices: np.ndarray) -> "PerImageRasterAttribute":
        return PerImageRasterAttribute(
            paths=[self._paths[i] for i in indices],
            resize_interpolation=self._resize_interpolation,
            file_type=self._file_type,
        )

    def on_downsample_images(
        self, attr_name: str, downsample_factor: int, output_cache: Any
    ) -> "PerImageRasterAttribute":
        import pathlib

        import cv2
        import torch

        from .sfm_cache import SfmCache

        cache: SfmCache = output_cache

        cache_folder_name = f"attr_{attr_name}_downsample_{downsample_factor}x"
        attr_cache = cache.make_folder(cache_folder_name, description=f"Downsampled raster attribute '{attr_name}'")

        # Check if cache is valid
        if attr_cache.num_files == len(self._paths):
            # Assume cache is valid – return paths from cache
            new_paths = []
            all_cached = True
            num_zeropad = len(str(len(self._paths))) + 2
            for i in range(len(self._paths)):
                file_name = f"raster_{i:0{num_zeropad}}"
                if not attr_cache.has_file(file_name):
                    all_cached = False
                    break
                meta = attr_cache.get_file_metadata(file_name)
                new_paths.append(str(meta["path"]))
            if all_cached:
                return PerImageRasterAttribute(
                    paths=new_paths,
                    resize_interpolation=self._resize_interpolation,
                    file_type=self._file_type,
                )

        # Regenerate
        attr_cache.clear_current_folder()
        new_paths = []
        num_zeropad = len(str(len(self._paths))) + 2

        _INTERP_TO_CV2 = {
            InterpolationMode.AREA: cv2.INTER_AREA,
            InterpolationMode.BILINEAR: cv2.INTER_LINEAR,
            InterpolationMode.BICUBIC: cv2.INTER_CUBIC,
            InterpolationMode.NEAREST: cv2.INTER_NEAREST,
            InterpolationMode.LINEAR: cv2.INTER_LINEAR,
        }
        _INTERP_TO_TORCH = {
            InterpolationMode.AREA: "area",
            InterpolationMode.BILINEAR: "bilinear",
            InterpolationMode.BICUBIC: "bicubic",
            InterpolationMode.NEAREST: "nearest",
            InterpolationMode.LINEAR: "bilinear",
        }

        for i, path in enumerate(self._paths):
            ext = pathlib.Path(path).suffix.lower()
            file_name = f"raster_{i:0{num_zeropad}}"

            if ext in (".png", ".jpg", ".jpeg"):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"Failed to load raster attribute '{attr_name}' from {path}")
                h, w = img.shape[:2]
                new_h = int(h / downsample_factor)
                new_w = int(w / downsample_factor)
                resized = cv2.resize(
                    img,
                    (new_w, new_h),
                    interpolation=_INTERP_TO_CV2[self._resize_interpolation],
                )
                meta = attr_cache.write_file(file_name, resized, data_type=self._file_type)
                new_paths.append(str(meta["path"]))

            elif ext == ".npy":
                arr = np.load(path)
                resized = self._resize_array(arr, downsample_factor, attr_name, _INTERP_TO_TORCH)
                meta = attr_cache.write_file(file_name, resized, data_type="npy")
                new_paths.append(str(meta["path"]))

            elif ext == ".pt":
                data = torch.load(path, weights_only=False)
                if isinstance(data, torch.Tensor):
                    resized = self._resize_tensor(data, downsample_factor, attr_name, _INTERP_TO_TORCH)
                elif isinstance(data, np.ndarray):
                    resized = self._resize_array(data, downsample_factor, attr_name, _INTERP_TO_TORCH)
                else:
                    raise TypeError(
                        f"Cannot resize attribute '{attr_name}': loaded data is {type(data).__name__}, "
                        f"expected torch.Tensor or numpy.ndarray. Subclass PerImageRasterAttribute to "
                        f"handle custom data formats."
                    )
                meta = attr_cache.write_file(file_name, resized, data_type="pt")
                new_paths.append(str(meta["path"]))

            else:
                raise ValueError(f"Unsupported file extension '{ext}' for attribute '{attr_name}'")

        return PerImageRasterAttribute(
            paths=new_paths,
            resize_interpolation=self._resize_interpolation,
            file_type=self._file_type,
        )

    def _resize_array(self, arr: np.ndarray, factor: int, attr_name: str, interp_map: dict) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(arr)
        resized_tensor = self._resize_tensor(tensor, factor, attr_name, interp_map)
        return resized_tensor.numpy()

    def _resize_tensor(self, tensor: "torch.Tensor", factor: int, attr_name: str, interp_map: dict) -> "torch.Tensor":
        import torch
        import torch.nn.functional as F

        if tensor.ndim < 2:
            raise ValueError(
                f"Cannot resize attribute '{attr_name}': loaded tensor has shape {tuple(tensor.shape)} "
                f"with < 2 spatial dimensions. PerImageRasterAttribute expects a spatial raster (H, W, ...)."
            )

        is_integer = not tensor.is_floating_point() and not tensor.is_complex()
        if is_integer and self._resize_interpolation != InterpolationMode.NEAREST:
            raise TypeError(
                f"Cannot resize attribute '{attr_name}': integer tensor (dtype={tensor.dtype}) "
                f"requires InterpolationMode.NEAREST, but this attribute uses "
                f"InterpolationMode.{self._resize_interpolation.name}."
            )

        # Determine spatial layout: assume (H, W, ...) unless first dim is small and last two are large
        # Heuristic: if ndim >= 3 and shape[0] <= 16 and shape[-2] > 16, assume (C, H, W)
        chw_layout = False
        if tensor.ndim >= 3 and tensor.shape[0] <= 16 and tensor.shape[-2] > 16 and tensor.shape[-1] > 16:
            chw_layout = True

        original_dtype = tensor.dtype
        if is_integer:
            work_tensor = tensor.float()
        else:
            work_tensor = tensor.float() if tensor.dtype != torch.float32 else tensor

        if chw_layout:
            # (C, H, W) layout
            if work_tensor.ndim == 3:
                work_tensor = work_tensor.unsqueeze(0)  # (1, C, H, W)
                h, w = work_tensor.shape[2], work_tensor.shape[3]
                new_h, new_w = int(h / factor), int(w / factor)
                resized = F.interpolate(
                    work_tensor,
                    size=(new_h, new_w),
                    mode=interp_map[self._resize_interpolation],
                )
                resized = resized.squeeze(0)  # (C, H', W')
            else:
                raise ValueError(
                    f"Cannot resize attribute '{attr_name}': unsupported tensor shape {tuple(tensor.shape)} "
                    f"in CHW layout."
                )
        else:
            # (H, W, ...) layout
            h, w = work_tensor.shape[0], work_tensor.shape[1]
            new_h, new_w = int(h / factor), int(w / factor)
            trailing = work_tensor.shape[2:]
            if len(trailing) == 0:
                # (H, W) -> (1, 1, H, W)
                work_tensor = work_tensor.unsqueeze(0).unsqueeze(0)
                resized = F.interpolate(
                    work_tensor,
                    size=(new_h, new_w),
                    mode=interp_map[self._resize_interpolation],
                )
                resized = resized.squeeze(0).squeeze(0)
            else:
                # (H, W, C, ...) -> reshape to (1, prod(trailing), H, W), resize, reshape back
                flat_trailing = int(np.prod(trailing))
                work_tensor = work_tensor.reshape(h, w, flat_trailing)
                work_tensor = work_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                resized = F.interpolate(
                    work_tensor,
                    size=(new_h, new_w),
                    mode=interp_map[self._resize_interpolation],
                )
                resized = resized.squeeze(0).permute(1, 2, 0)  # (H', W', C)
                resized = resized.reshape(new_h, new_w, *trailing)

        if is_integer:
            resized = resized.round().to(original_dtype)
        elif original_dtype != torch.float32:
            resized = resized.to(original_dtype)

        return resized

    def state_dict(self) -> dict:
        return {
            "paths": self._paths,
            "resize_interpolation": self._resize_interpolation.value,
            "file_type": self._file_type,
        }

    @staticmethod
    def from_state_dict(state_dict: dict) -> "PerImageRasterAttribute":
        return PerImageRasterAttribute(
            paths=state_dict["paths"],
            resize_interpolation=InterpolationMode(state_dict["resize_interpolation"]),
            file_type=state_dict.get("file_type", "png"),
        )


@scene_attribute
class PerCameraAttribute(SceneAttribute):
    """Per-camera-sensor metadata keyed by camera ID."""

    def __init__(self, values: dict[int, Any]):
        self._values = dict(values)

    @property
    def values(self) -> dict[int, Any]:
        return self._values

    @staticmethod
    def type_name() -> str:
        return "PerCameraAttribute"

    def validate(self, attr_name: str, num_points: int, num_images: int, camera_ids: set[int]) -> None:
        invalid_keys = set(self._values.keys()) - camera_ids
        if invalid_keys:
            raise ValueError(
                f"Attribute '{attr_name}': keys {invalid_keys} are not valid camera IDs. " f"Valid IDs: {camera_ids}"
            )

    def state_dict(self) -> dict:
        return {"values": self._values}

    @staticmethod
    def from_state_dict(state_dict: dict) -> "PerCameraAttribute":
        values = {int(k): v for k, v in state_dict["values"].items()}
        return PerCameraAttribute(values=values)
