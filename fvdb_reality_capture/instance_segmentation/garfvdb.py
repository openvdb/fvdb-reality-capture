# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Public GARfVDB inference product."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import fvdb
import torch
from fvdb import GaussianSplat3d

from .model import GARfVDBModel
from .training.dataset import GARfVDBInput


class GARfVDB:
    """A portable scale-conditioned instance feature field and its Gaussian carrier."""

    def __init__(self, model: GARfVDBModel, reconstruction_metadata: Mapping[str, Any] | None = None) -> None:
        if not model.model_config.use_grid:
            raise ValueError("The first-class GARfVDB product requires GARfVDBConfig.use_grid=True")
        self._model = model
        self._reconstruction_metadata = dict(reconstruction_metadata or {})

    @property
    def model(self) -> GARfVDBModel:
        """The underlying PyTorch GARfVDB model."""
        return self._model

    @property
    def carrier(self) -> GaussianSplat3d:
        """The exact filtered Gaussian carrier used by the feature field."""
        return self._model.gs_model

    @property
    def encoder_grids(self) -> fvdb.GridBatch:
        """The ordered multiresolution encoder grid batch."""
        return self._model.encoder_gridbatch

    @property
    def reconstruction_metadata(self) -> dict[str, Any]:
        """A shallow copy of camera and reconstruction metadata bundled with the carrier."""
        return dict(self._reconstruction_metadata)

    @property
    def max_grouping_scale(self) -> float:
        """Maximum supported grouping scale in scene units."""
        return float(self._model.max_grouping_scale.item())

    def _validate_scale(self, scale: float) -> None:
        if scale < 0.0 or scale > self.max_grouping_scale:
            raise ValueError(f"scale must be in [0, {self.max_grouping_scale}], got {scale}")

    @torch.no_grad()
    def render_features(
        self,
        camera_to_world: torch.Tensor,
        projection: torch.Tensor,
        image_size: tuple[int, int],
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render scale-conditioned features and alpha.

        Args:
            camera_to_world: Camera-to-world matrix with shape ``(4, 4)`` or ``(B, 4, 4)``.
            projection: Camera intrinsic matrix with shape ``(3, 3)`` or ``(B, 3, 3)``.
            image_size: ``(width, height)`` shared by the batch.
            scale: Grouping scale in scene units.
        """
        self._validate_scale(scale)
        device = self._model.device
        camera_to_world = torch.as_tensor(camera_to_world, dtype=torch.float32, device=device)
        projection = torch.as_tensor(projection, dtype=torch.float32, device=device)
        if camera_to_world.ndim == 2:
            camera_to_world = camera_to_world.unsqueeze(0)
        if projection.ndim == 2:
            projection = projection.unsqueeze(0)
        if camera_to_world.ndim != 3 or camera_to_world.shape[-2:] != (4, 4):
            raise ValueError("camera_to_world must have shape (4, 4) or (B, 4, 4)")
        if projection.ndim != 3 or projection.shape[-2:] != (3, 3):
            raise ValueError("projection must have shape (3, 3) or (B, 3, 3)")
        if projection.shape[0] != camera_to_world.shape[0]:
            raise ValueError("camera_to_world and projection batch sizes must match")
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        world_to_camera = torch.linalg.inv(camera_to_world).contiguous()
        model_input = GARfVDBInput(
            {
                "projection": projection.contiguous(),
                "camera_to_world": camera_to_world.contiguous(),
                "world_to_camera": world_to_camera,
                "image_w": [width] * camera_to_world.shape[0],
                "image_h": [height] * camera_to_world.shape[0],
            }
        )
        return self._model.get_mask_output(model_input, scale)

    @torch.no_grad()
    def gaussian_affinities(self, scale: float) -> torch.Tensor:
        """Return per-Gaussian affinities aligned with :attr:`carrier`."""
        self._validate_scale(scale)
        return self._model.get_gaussian_affinity_output(scale)

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Save this product as a portable ``.garfvdb`` directory."""
        from .artifact import save_garfvdb_bundle

        return save_garfvdb_bundle(self, path)

    @classmethod
    def load(cls, path: str | pathlib.Path, device: str | torch.device = "cuda") -> "GARfVDB":
        """Load and validate a portable ``.garfvdb`` directory."""
        from .artifact import load_garfvdb_bundle

        return load_garfvdb_bundle(path, device=device)


__all__ = ["GARfVDB"]
