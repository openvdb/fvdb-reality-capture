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


class TransformMode(str, Enum):
    """How per-point data responds to spatial transforms on the scene.

    Modes
    -----
    NONE
        The attribute is not affected by spatial transforms.
    ROTATE
        Only the rotational component of the transform is applied (scale and
        shear are factored out via polar decomposition). Useful for normals.
    AFFINE
        The full 3x3 linear part (rotation, scale, shear) plus translation
        is applied. Use for positional data that should follow the scene
        geometry exactly.
    """

    NONE = "none"
    ROTATE = "rotate"
    AFFINE = "affine"


class InterpolationMode(str, Enum):
    """Interpolation modes for resizing raster data."""

    AREA = "area"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    NEAREST = "nearest"

    def to_cv2(self) -> int:
        """Return the corresponding ``cv2.INTER_*`` constant."""
        import cv2

        _map = {
            InterpolationMode.AREA: cv2.INTER_AREA,
            InterpolationMode.BILINEAR: cv2.INTER_LINEAR,
            InterpolationMode.BICUBIC: cv2.INTER_CUBIC,
            InterpolationMode.NEAREST: cv2.INTER_NEAREST,
        }
        return _map[self]

    def to_torch_str(self) -> str:
        """Return the mode string accepted by :func:`torch.nn.functional.interpolate`."""
        _map = {
            InterpolationMode.AREA: "area",
            InterpolationMode.BILINEAR: "bilinear",
            InterpolationMode.BICUBIC: "bicubic",
            InterpolationMode.NEAREST: "nearest",
        }
        return _map[self]


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
        """Called when points are filtered.  ``mask`` is a boolean array of
        shape ``(N,)`` where ``True`` keeps the point.  No-op by default."""
        return self

    def on_filter_images(self, mask: np.ndarray) -> "SceneAttribute":
        """Called when images are filtered.  ``mask`` is a boolean array of
        shape ``(I,)`` where ``True`` keeps the image.  No-op by default."""
        return self

    def on_select_images(self, indices: np.ndarray) -> "SceneAttribute":
        """Called when a subset of images is selected by index.  No-op by default."""
        return self

    def on_spatial_transform(self, matrix: np.ndarray) -> "SceneAttribute":
        """Called when a 4x4 spatial transform is applied to the scene.  No-op by default."""
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

    def __init__(self, data: np.ndarray, transform_mode: TransformMode | str = TransformMode.NONE):
        self._data = data
        self._transform_mode = (
            TransformMode(transform_mode) if not isinstance(transform_mode, TransformMode) else transform_mode
        )

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def transform_mode(self) -> TransformMode:
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
        if self._transform_mode == TransformMode.NONE:
            return self

        linear = matrix[:3, :3]

        if self._transform_mode == TransformMode.ROTATE:
            # Extract pure rotation by factoring out scale via polar decomposition.
            U, _, Vt = np.linalg.svd(linear)
            R_pure = U @ Vt
            # Ensure proper rotation (det = +1)
            if np.linalg.det(R_pure) < 0:
                U[:, -1] *= -1
                R_pure = U @ Vt
            new_data = self._data @ R_pure.T
            return PerPointAttribute(new_data, transform_mode=self._transform_mode)

        # "affine" – full linear + translation
        translation = matrix[:3, 3]
        new_data = self._data @ linear.T + translation
        return PerPointAttribute(new_data, transform_mode=self._transform_mode)

    def state_dict(self) -> dict:
        return {
            "data": self._data,
            "dtype": str(self._data.dtype),
            "transform_mode": self._transform_mode.value,
        }

    @staticmethod
    def from_state_dict(state_dict: dict) -> "PerPointAttribute":
        raw = state_dict["data"]
        dtype = state_dict.get("dtype", None)
        if isinstance(raw, np.ndarray):
            data = raw
        else:
            data = np.array(raw, dtype=dtype)
        return PerPointAttribute(
            data=data,
            transform_mode=TransformMode(state_dict.get("transform_mode", "none")),
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
    """Per-image raster data stored as files on disk, spatially aligned to images.

    All rasters **must** use ``(H, W)`` or ``(H, W, C, ...)`` layout, matching
    the convention used by OpenCV and the rest of the dataset pipeline.
    ``(C, H, W)`` tensors are **not** supported -- callers must permute to
    ``(H, W, C)`` before registering the attribute.
    """

    def __init__(
        self,
        paths: list[str],
        resize_interpolation: InterpolationMode = InterpolationMode.AREA,
    ):
        """
        Args:
            paths: One file path per image, in the same order as the scene's image list.
            resize_interpolation: Interpolation mode used when downsampling rasters.
        """
        self._paths = list(paths)
        self._resize_interpolation = (
            InterpolationMode(resize_interpolation)
            if not isinstance(resize_interpolation, InterpolationMode)
            else resize_interpolation
        )

    @property
    def paths(self) -> list[str]:
        return self._paths

    @property
    def resize_interpolation(self) -> InterpolationMode:
        return self._resize_interpolation

    @staticmethod
    def type_name() -> str:
        return "PerImageRasterAttribute"

    def _with_paths(self, new_paths: list[str]) -> "PerImageRasterAttribute":
        """Return a copy of this attribute with replaced paths."""
        return PerImageRasterAttribute(paths=new_paths, resize_interpolation=self._resize_interpolation)

    def validate(self, attr_name: str, num_points: int, num_images: int, camera_ids: set[int]) -> None:
        if len(self._paths) != num_images:
            raise ValueError(
                f"Attribute '{attr_name}': expected len(paths) == {num_images} (num_images), " f"got {len(self._paths)}"
            )

    def on_filter_images(self, mask: np.ndarray) -> "PerImageRasterAttribute":
        return self._with_paths([p for p, keep in zip(self._paths, mask) if keep])

    def on_select_images(self, indices: np.ndarray) -> "PerImageRasterAttribute":
        return self._with_paths([self._paths[i] for i in indices])

    def on_downsample_images(
        self, attr_name: str, downsample_factor: int, output_cache: Any
    ) -> "PerImageRasterAttribute":
        import pathlib

        import cv2
        import torch

        from .sfm_cache import SfmCache

        cache: SfmCache = output_cache

        cache_folder_name = f"attr_{attr_name}_downsample_{downsample_factor}x_{self._resize_interpolation.value}"
        attr_cache = cache.make_folder(cache_folder_name, description=f"Downsampled raster attribute '{attr_name}'")
        num_zeropad = len(str(len(self._paths))) + 2

        # Check if cache is valid
        if attr_cache.num_files == len(self._paths):
            new_paths = []
            all_cached = True
            for i in range(len(self._paths)):
                file_name = f"raster_{i:0{num_zeropad}}"
                if not attr_cache.has_file(file_name):
                    all_cached = False
                    break
                meta = attr_cache.get_file_metadata(file_name)
                new_paths.append(str(meta["path"]))
            if all_cached:
                return self._with_paths(new_paths)

        # Regenerate
        attr_cache.clear_current_folder()
        new_paths = []

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
                if new_h < 1 or new_w < 1:
                    raise ValueError(
                        f"Cannot downsample raster attribute '{attr_name}': downsample_factor={downsample_factor} "
                        f"produces output size ({new_h}, {new_w}) from input ({h}, {w}). "
                        f"Output height and width must each be at least 1."
                    )
                resized = cv2.resize(
                    img,
                    (new_w, new_h),
                    interpolation=self._resize_interpolation.to_cv2(),
                )
                out_type = "jpg" if ext in (".jpg", ".jpeg") else "png"
                meta = attr_cache.write_file(file_name, resized, data_type=out_type)
                new_paths.append(str(meta["path"]))

            elif ext == ".npy":
                arr = np.load(path)
                resized = self._resize_array(arr, downsample_factor, attr_name)
                meta = attr_cache.write_file(file_name, resized, data_type="npy")
                new_paths.append(str(meta["path"]))

            elif ext == ".pt":
                data = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(data, torch.Tensor):
                    resized = self._resize_tensor(data, downsample_factor, attr_name)
                elif isinstance(data, np.ndarray):
                    resized = self._resize_array(data, downsample_factor, attr_name)
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

        return self._with_paths(new_paths)

    def _resize_array(self, arr: np.ndarray, factor: int, attr_name: str) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(arr)
        resized_tensor = self._resize_tensor(tensor, factor, attr_name)
        return resized_tensor.numpy()

    def _resize_tensor(self, tensor: "torch.Tensor", factor: int, attr_name: str) -> "torch.Tensor":
        """Resize a tensor raster by the given downsample ``factor``.

        The tensor must be in ``(H, W)`` or ``(H, W, C, ...)`` layout.
        ``(C, H, W)`` layout is **not** supported; see the class docstring.

        Non-float32 floating-point tensors (e.g. float64) are temporarily
        cast to float32 for interpolation and cast back afterward -- minor
        precision loss may occur.

        Integer tensors are also cast to float32 for nearest-neighbor
        interpolation.  Values larger than 2^24 (~16.7 M) are not exactly
        representable in float32, so very large label IDs may be corrupted.
        If this is a concern, subclass and override with a float64 path.
        """
        import torch
        import torch.nn.functional as F

        if tensor.ndim < 2:
            raise ValueError(
                f"Cannot resize attribute '{attr_name}': loaded tensor has shape {tuple(tensor.shape)} "
                f"with < 2 spatial dimensions. PerImageRasterAttribute expects (H, W) or (H, W, C, ...)."
            )

        if tensor.is_complex():
            raise TypeError(
                f"Cannot resize attribute '{attr_name}': complex tensors (dtype={tensor.dtype}) are not supported."
            )

        is_integer = not tensor.is_floating_point()
        if is_integer and self._resize_interpolation != InterpolationMode.NEAREST:
            raise TypeError(
                f"Cannot resize attribute '{attr_name}': integer tensor (dtype={tensor.dtype}) "
                f"requires InterpolationMode.NEAREST, but this attribute uses "
                f"InterpolationMode.{self._resize_interpolation.name}."
            )

        original_dtype = tensor.dtype
        if is_integer:
            work_tensor = tensor.float()
        else:
            work_tensor = tensor.float() if tensor.dtype != torch.float32 else tensor

        h, w = work_tensor.shape[0], work_tensor.shape[1]
        new_h, new_w = int(h / factor), int(w / factor)
        if new_h < 1 or new_w < 1:
            raise ValueError(
                f"Cannot resize attribute '{attr_name}': factor={factor} produces output size ({new_h}, {new_w}) "
                f"from input shape {tuple(tensor.shape)}. Output height and width must each be at least 1."
            )
        trailing = work_tensor.shape[2:]
        mode = self._resize_interpolation.to_torch_str()
        # align_corners=False is required for bilinear/bicubic to avoid
        # deprecation warnings and ensure consistent pixel-edge alignment
        # across torch versions. nearest/area do not accept this parameter.
        interp_kwargs: dict = {"size": (new_h, new_w), "mode": mode}
        if mode in ("bilinear", "bicubic"):
            interp_kwargs["align_corners"] = False

        if len(trailing) == 0:
            work_tensor = work_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            resized = F.interpolate(work_tensor, **interp_kwargs)
            resized = resized.squeeze(0).squeeze(0)  # (H', W')
        else:
            flat_trailing = int(np.prod(trailing))
            work_tensor = work_tensor.reshape(h, w, flat_trailing)
            work_tensor = work_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            resized = F.interpolate(work_tensor, **interp_kwargs)
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
        }

    @staticmethod
    def from_state_dict(state_dict: dict) -> "PerImageRasterAttribute":
        return PerImageRasterAttribute(
            paths=state_dict["paths"],
            resize_interpolation=InterpolationMode(state_dict["resize_interpolation"]),
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
