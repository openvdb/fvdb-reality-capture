# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Apply camera poses saved by a Gaussian reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fvdb_reality_capture.sfm_scene import SfmPosedImageMetadata, SfmScene
from fvdb_reality_capture.transforms import BaseTransform, transform


@transform
class ApplyReconstructionCameraPoses(BaseTransform):
    """Replace scene camera poses with the final poses used by a reconstruction."""

    version = "1.0.0"

    def __init__(
        self,
        camera_to_world_matrices: np.ndarray | torch.Tensor,
        image_ids: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        if isinstance(camera_to_world_matrices, torch.Tensor):
            camera_to_world_matrices = camera_to_world_matrices.detach().cpu().numpy()
        matrices = np.asarray(camera_to_world_matrices)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
            raise ValueError(
                "Reconstruction camera_to_world_matrices must have shape " f"(num_images, 4, 4), got {matrices.shape}."
            )
        if not np.isfinite(matrices).all():
            raise ValueError("Reconstruction camera_to_world_matrices must contain only finite values.")
        try:
            np.linalg.inv(matrices)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Reconstruction camera_to_world_matrices must be invertible.") from exc
        self._camera_to_world_matrices = matrices.copy()

        if isinstance(image_ids, torch.Tensor):
            image_ids = image_ids.detach().cpu().numpy()
        if image_ids is None:
            self._image_ids = None
        else:
            image_ids_array = np.asarray(image_ids)
            if image_ids_array.ndim != 1 or len(image_ids_array) != len(matrices):
                raise ValueError(
                    "Reconstruction image_ids must have shape " f"({len(matrices)},), got {image_ids_array.shape}."
                )
            image_ids_array = image_ids_array.astype(np.int64, copy=False)
            if len(np.unique(image_ids_array)) != len(image_ids_array):
                raise ValueError("Reconstruction image_ids must be unique.")
            self._image_ids = image_ids_array.copy()

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        if self._image_ids is None:
            if len(self._camera_to_world_matrices) != input_scene.num_images:
                raise ValueError(
                    "Reconstruction metadata does not contain image_ids, so camera poses can only be matched "
                    "positionally when the reconstruction and transformed scene have the same number of images. "
                    f"Reconstruction has {len(self._camera_to_world_matrices)} poses; "
                    f"scene has {input_scene.num_images} images."
                )
            poses_by_scene_index = dict(enumerate(self._camera_to_world_matrices))
            poses_by_image_id = None
        else:
            poses_by_scene_index = None
            poses_by_image_id = dict(zip(self._image_ids.tolist(), self._camera_to_world_matrices))

        updated_images = []
        num_matched = 0
        for scene_index, image in enumerate(input_scene.images):
            if poses_by_scene_index is not None:
                camera_to_world = poses_by_scene_index[scene_index]
            else:
                assert poses_by_image_id is not None
                camera_to_world = poses_by_image_id.get(image.image_id)
                if camera_to_world is None:
                    updated_images.append(image)
                    continue

            num_matched += 1
            updated_images.append(
                SfmPosedImageMetadata(
                    world_to_camera_matrix=np.linalg.inv(camera_to_world),
                    camera_to_world_matrix=camera_to_world.copy(),
                    camera_metadata=image.camera_metadata,
                    camera_id=image.camera_id,
                    image_path=image.image_path,
                    mask_path=image.mask_path,
                    point_indices=image.point_indices,
                    image_id=image.image_id,
                )
            )

        if num_matched == 0 and input_scene.num_images > 0:
            raise ValueError("None of the reconstruction camera poses match the transformed scene image IDs.")
        return input_scene.replace(images=updated_images)

    @staticmethod
    def name() -> str:
        return "ApplyReconstructionCameraPoses"

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
            "version": self.version,
            "camera_to_world_matrices": self._camera_to_world_matrices,
            "image_ids": self._image_ids,
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "ApplyReconstructionCameraPoses":
        if state_dict.get("name") != ApplyReconstructionCameraPoses.name():
            raise ValueError(
                f"Expected state_dict with name {ApplyReconstructionCameraPoses.name()!r}, "
                f"got {state_dict.get('name')!r}."
            )
        if state_dict.get("version") != ApplyReconstructionCameraPoses.version:
            raise ValueError(
                f"Unsupported {ApplyReconstructionCameraPoses.name()} version {state_dict.get('version')!r}; "
                f"expected {ApplyReconstructionCameraPoses.version!r}."
            )
        return ApplyReconstructionCameraPoses(
            camera_to_world_matrices=state_dict["camera_to_world_matrices"],
            image_ids=state_dict.get("image_ids"),
        )


__all__ = ["ApplyReconstructionCameraPoses"]
