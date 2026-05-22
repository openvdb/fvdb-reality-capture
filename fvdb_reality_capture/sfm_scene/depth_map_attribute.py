# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import pathlib
from enum import Enum
from typing import Any

import numpy as np

from .scene_attribute import (
    InterpolationMode,
    PerImageRasterAttribute,
    scene_attribute,
)


class DepthScale(str, Enum):
    """How depth values respond to spatial transforms of the owning scene.

    METRIC
        Depth values are in scene units (multiplied by :attr:`DepthMapAttribute.unit_scale`).
        A similarity transform applied to the scene must rescale these values; the attribute
        absorbs the scale factor into ``unit_scale`` via :meth:`DepthMapAttribute.on_spatial_transform`
        rather than rewriting the rasters on disk.

    RELATIVE
        Depth values are dimensionless (e.g. predictions from a monocular relative-depth
        estimator that are only meaningful up to a per-image scale). Spatial transforms
        leave the attribute untouched.
    """

    METRIC = "metric"
    RELATIVE = "relative"


class DepthMissingPolicy(str, Enum):
    """How invalid pixels are encoded in the on-disk rasters.

    NAN
        Invalid pixels are stored as NaN (only valid for floating-point rasters).
    ZERO
        Invalid pixels are stored as 0 (typical for 16-bit PNG depth maps).
    SENTINEL
        Invalid pixels are equal to :attr:`DepthMapAttribute.invalid_value`.
    """

    NAN = "nan"
    ZERO = "zero"
    SENTINEL = "sentinel"


@scene_attribute
class DepthMapAttribute(PerImageRasterAttribute):
    """A per-image depth map raster with scale-aware semantics.

    Extends :class:`PerImageRasterAttribute` with three pieces of information that
    a generic raster cannot model:

    1. ``unit_scale`` -- a multiplier from raw on-disk values to scene-unit depth,
       which is updated automatically when the scene is rescaled (see :meth:`on_spatial_transform`).
    2. ``scale`` -- :class:`DepthScale.METRIC` (depths track the scene) or
       :class:`DepthScale.RELATIVE` (dimensionless; immune to spatial transforms).
    3. A validity convention (:class:`DepthMissingPolicy`) so the dataset loader can
       emit a per-pixel ``valid`` mask alongside the depth tensor.

    The default ``resize_interpolation`` is :class:`InterpolationMode.NEAREST`, which
    avoids bilerping across depth discontinuities. Averaging modes (AREA/BILINEAR/BICUBIC)
    are only permitted with ``missing_policy=NAN``: with a ZERO or SENTINEL policy they
    would blend the invalid fill value into neighboring valid pixels when downsampling,
    silently corrupting them, so that combination raises in the constructor.

    The class is registered with the scene-attribute registry under the type name
    ``"DepthMapAttribute"``, so an :class:`SfmScene` carrying one round-trips through
    :meth:`SfmScene.state_dict` / :meth:`SfmScene.from_state_dict`.
    """

    def __init__(
        self,
        paths: list[str],
        unit_scale: float = 1.0,
        scale: DepthScale | str = DepthScale.METRIC,
        missing_policy: DepthMissingPolicy | str = DepthMissingPolicy.NAN,
        invalid_value: float | None = None,
        resize_interpolation: InterpolationMode | str = InterpolationMode.NEAREST,
    ):
        """
        Args:
            paths: One file path per image, in the same order as the scene's image list.
                Supported formats are the same as :class:`PerImageRasterAttribute`
                (``.png``, ``.jpg``/``.jpeg``, ``.npy``, ``.pt``). 16-bit single-channel
                PNGs are read with :func:`cv2.imread(..., cv2.IMREAD_UNCHANGED)`.

                .. warning::
                    ``.pt`` files are deserialized with :func:`torch.load`, which executes
                    arbitrary code via pickle. Only load ``.pt`` depth maps produced by a
                    trusted pipeline. For depth maps from untrusted sources prefer the
                    non-executable ``.npy`` or PNG formats.
            unit_scale: Multiplier from raw stored values to scene-unit depth. For depth maps
                stored as 16-bit PNGs in millimeters with the scene in meters, use ``0.001``.
                For float rasters already in scene units, leave at ``1.0``. This value is
                composed with the linear scale of any subsequent spatial transform when
                ``scale == METRIC``.
            scale: :class:`DepthScale.METRIC` (default) for depths that track the scene's
                coordinate system, or :class:`DepthScale.RELATIVE` for scale-invariant
                depth predictions (e.g. monocular relative depth).
            missing_policy: Encoding of invalid pixels (see :class:`DepthMissingPolicy`).
            invalid_value: The sentinel value used when ``missing_policy == SENTINEL``.
                Must be ``None`` for the other policies.
            resize_interpolation: Interpolation mode for downsampling. Defaults to
                :class:`InterpolationMode.NEAREST` to avoid mixing depths across
                discontinuities.
        """
        super().__init__(paths=paths, resize_interpolation=resize_interpolation)

        self._unit_scale = float(unit_scale)
        self._scale = DepthScale(scale) if not isinstance(scale, DepthScale) else scale
        self._missing_policy = (
            DepthMissingPolicy(missing_policy) if not isinstance(missing_policy, DepthMissingPolicy) else missing_policy
        )

        if self._missing_policy == DepthMissingPolicy.SENTINEL:
            if invalid_value is None:
                raise ValueError("missing_policy=SENTINEL requires an explicit invalid_value.")
            self._invalid_value = float(invalid_value)
        else:
            if invalid_value is not None:
                raise ValueError(
                    f"invalid_value must be None when missing_policy={self._missing_policy.value!r}; "
                    f"got {invalid_value!r}."
                )
            self._invalid_value = None

        # Averaging interpolation (AREA/BILINEAR/BICUBIC) mixes neighboring pixels when the
        # raster is downsampled. With a ZERO or SENTINEL missing policy the invalid pixels
        # hold a real finite value (0 or the sentinel), so averaging silently bleeds that
        # value into adjacent valid pixels and the result still looks "valid". Only NEAREST
        # avoids this. NaN-encoded invalids instead propagate to NaN under averaging, which
        # is then correctly flagged invalid on reload -- so require NaN for non-NEAREST modes.
        if self._resize_interpolation != InterpolationMode.NEAREST and self._missing_policy != DepthMissingPolicy.NAN:
            raise ValueError(
                f"DepthMapAttribute with resize_interpolation="
                f"{self._resize_interpolation.value!r} and missing_policy="
                f"{self._missing_policy.value!r} is unsafe: averaging interpolation would blend "
                f"invalid {self._missing_policy.value!r} values into neighboring valid pixels when "
                f"downsampling, silently corrupting them. Use resize_interpolation=NEAREST, or switch "
                f"to missing_policy=NAN (NaNs propagate under interpolation and are re-flagged invalid)."
            )

    @property
    def unit_scale(self) -> float:
        """Multiplier from raw stored values to scene-unit depth."""
        return self._unit_scale

    @property
    def scale(self) -> DepthScale:
        """Whether the depths track scene transforms (METRIC) or are scale-invariant (RELATIVE)."""
        return self._scale

    @property
    def missing_policy(self) -> DepthMissingPolicy:
        """How invalid pixels are encoded in the on-disk rasters."""
        return self._missing_policy

    @property
    def invalid_value(self) -> float | None:
        """Sentinel value used when ``missing_policy == SENTINEL``; ``None`` otherwise."""
        return self._invalid_value

    @staticmethod
    def type_name() -> str:
        return "DepthMapAttribute"

    def _replace_paths(self, new_paths: list[str]) -> "DepthMapAttribute":
        return DepthMapAttribute(
            paths=new_paths,
            unit_scale=self._unit_scale,
            scale=self._scale,
            missing_policy=self._missing_policy,
            invalid_value=self._invalid_value,
            resize_interpolation=self._resize_interpolation,
        )

    # -- Hook overrides ------------------------------------------------------

    def on_filter_images(self, mask: np.ndarray) -> "DepthMapAttribute":
        return self._replace_paths([p for p, keep in zip(self._paths, mask) if keep])

    def on_select_images(self, indices: np.ndarray) -> "DepthMapAttribute":
        return self._replace_paths([self._paths[i] for i in indices])

    def on_downsample_images(self, attr_name: str, downsample_factor: int, output_cache: Any) -> "DepthMapAttribute":
        downsampled = super().on_downsample_images(attr_name, downsample_factor, output_cache)
        return self._replace_paths(downsampled.paths)

    def on_spatial_transform(self, matrix: np.ndarray) -> "DepthMapAttribute":
        """Fold the linear scale of ``matrix`` into ``unit_scale`` (METRIC only).

        For ``scale == RELATIVE`` this is a no-op. For ``scale == METRIC`` the 3x3
        linear part of ``matrix`` must be a similarity (uniform scale, no reflection);
        any other transform raises :class:`ValueError` rather than silently producing
        anisotropically deformed depth.
        """
        if self._scale != DepthScale.METRIC:
            return self

        linear = matrix[:3, :3]
        svals = np.linalg.svd(linear, compute_uv=False)  # sorted descending
        # Reject degenerate (rank-deficient / near-zero scale) transforms first: these
        # would otherwise fold s~=0 into unit_scale and silently zero out every depth.
        if svals[0] <= 1e-8:
            raise ValueError(
                f"DepthMapAttribute(scale=METRIC) got a degenerate spatial transform with "
                f"near-zero scale (singular values {svals}); this would collapse all depths to zero."
            )
        # Allow a small relative tolerance because normalize_scene builds the transform
        # in float64 from accumulated outer products.
        if not np.allclose(svals, svals[0], rtol=1e-4, atol=1e-6):
            raise ValueError(
                f"DepthMapAttribute(scale=METRIC) requires a similarity (uniform-scale) "
                f"spatial transform; got singular values {svals}. A non-uniform 3x3 would "
                f"deform depth anisotropically, which cannot be represented by a single "
                f"scalar multiplier. Convert the attribute to scale=RELATIVE if you "
                f"intentionally want it to be invariant to such transforms."
            )
        if np.linalg.det(linear) < 0:
            raise ValueError("DepthMapAttribute(scale=METRIC) does not support reflective (det<0) spatial transforms.")

        s = float(svals.mean())
        return DepthMapAttribute(
            paths=self._paths,
            unit_scale=self._unit_scale * s,
            scale=self._scale,
            missing_policy=self._missing_policy,
            invalid_value=self._invalid_value,
            resize_interpolation=self._resize_interpolation,
        )

    # -- Loader (called by SfmDataset) ---------------------------------------

    def load_depth(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load the raw depth raster at ``index`` and apply ``unit_scale`` + validity mask.

        The raster must be single-channel: a ``(H, W)`` array, or ``(H, W, 1)`` which
        is squeezed to ``(H, W)``. Any other shape (e.g. a 3-channel RGB image) raises
        :class:`ValueError`.

        Returns:
            depth (np.ndarray): float32 ``(H, W)`` array. Invalid pixels are set to 0
                in the returned array; consult ``valid`` to distinguish them from real
                zero depth.
            valid (np.ndarray): bool ``(H, W)`` array, ``True`` where the depth is
                valid under :attr:`missing_policy`.
        """
        import cv2
        import torch

        path = self._paths[index]
        ext = pathlib.Path(path).suffix.lower()

        if ext in (".png", ".jpg", ".jpeg"):
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(f"Failed to load depth raster from {path}")
            arr = np.asarray(raw)
        elif ext == ".npy":
            arr = np.load(path)
        elif ext == ".pt":
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(loaded, torch.Tensor):
                # detach()+cpu() so a tensor saved with requires_grad or on a non-CPU
                # device still converts cleanly; contiguous() guards against numpy()
                # rejecting a non-contiguous view.
                arr = loaded.detach().cpu().contiguous().numpy()
            elif isinstance(loaded, np.ndarray):
                arr = loaded
            else:
                raise TypeError(
                    f"DepthMapAttribute: expected torch.Tensor or numpy.ndarray in "
                    f"{path}, got {type(loaded).__name__}."
                )
        else:
            raise ValueError(f"DepthMapAttribute: unsupported file extension '{ext}' for {path}.")

        # Depth maps must be single-channel. Squeeze a trailing singleton channel
        # (H, W, 1) -> (H, W); reject genuinely multi-channel rasters such as an RGB
        # image supplied by mistake, which would otherwise broadcast silently in the
        # dense-depth loss.
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError(
                f"DepthMapAttribute: expected a single-channel (H, W) depth raster at {path}, "
                f"got shape {tuple(arr.shape)}. Multi-channel images (e.g. RGB) are not valid depth maps."
            )

        # Compute the validity mask on the *raw* values (before unit_scale) so that
        # SENTINEL values are matched in the same units the user specified.
        if self._missing_policy == DepthMissingPolicy.NAN:
            if not np.issubdtype(arr.dtype, np.floating):
                raise TypeError(
                    f"DepthMapAttribute(missing_policy=NAN): raster at {path} has dtype "
                    f"{arr.dtype}, which cannot represent NaN. Use ZERO or SENTINEL."
                )
            valid = ~np.isnan(arr)
        elif self._missing_policy == DepthMissingPolicy.ZERO:
            valid = arr > 0
        else:  # SENTINEL
            assert self._invalid_value is not None
            valid = arr != self._invalid_value

        depth = arr.astype(np.float32, copy=False) * np.float32(self._unit_scale)
        depth = np.where(valid, depth, np.float32(0.0))
        return depth, valid

    # -- Serialization -------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "paths": self._paths,
            "unit_scale": self._unit_scale,
            "scale": self._scale.value,
            "missing_policy": self._missing_policy.value,
            "invalid_value": self._invalid_value,
            "resize_interpolation": self._resize_interpolation.value,
        }

    @staticmethod
    def from_state_dict(state_dict: dict) -> "DepthMapAttribute":
        return DepthMapAttribute(
            paths=list(state_dict["paths"]),
            unit_scale=float(state_dict.get("unit_scale", 1.0)),
            scale=DepthScale(state_dict.get("scale", DepthScale.METRIC.value)),
            missing_policy=DepthMissingPolicy(state_dict.get("missing_policy", DepthMissingPolicy.NAN.value)),
            invalid_value=state_dict.get("invalid_value"),
            resize_interpolation=InterpolationMode(
                state_dict.get("resize_interpolation", InterpolationMode.NEAREST.value)
            ),
        )

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def warn_if_scene_already_transformed(scene: Any, attr_name: str = "depth") -> None:
        """Emit a warning if ``scene`` already has a non-identity transformation matrix.

        Call this from user code at the moment a METRIC depth attribute is attached to
        a scene to surface the most common footgun (attaching metric depths after
        :class:`NormalizeScene` has run). The check is best-effort: it emits a Python
        :class:`warning <warnings.warn>` rather than raising, and is a no-op if the
        scene exposes no ``transformation_matrix``.
        """
        import warnings

        raw_transform = getattr(scene, "transformation_matrix", None)
        if raw_transform is None:
            return
        transform = np.asarray(raw_transform)
        if transform.shape != (4, 4):
            # Unexpected shape -- treat as "can't tell", don't crash on a best-effort check.
            return
        if not np.allclose(transform, np.eye(4), atol=1e-6):
            warnings.warn(
                f"Attaching DepthMapAttribute(scale=METRIC) '{attr_name}' to a scene whose "
                f"transformation_matrix is not identity. The on-disk depths must already be "
                f"in the *transformed* scene's units, or unit_scale must compensate. "
                f"Apply DepthMapAttribute before transforms (e.g. NormalizeScene) so that "
                f"on_spatial_transform can fold the scale automatically.",
                stacklevel=2,
            )
