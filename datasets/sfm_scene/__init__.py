# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib

from ..image_dataset_cache import ImageDatasetCache
from .sfm_metadata import SfmCameraMetadata, SfmCameraType, SfmImageMetadata
from .sfm_scene import SfmScene

__all__ = [
    "SfmCameraMetadata",
    "SfmImageMetadata",
    "SfmCameraType",
    "SfmScene",
]


def load_colmap_scene(
    dataset_path: pathlib.Path,
    normalization_type: str = "pca",
) -> tuple[SfmScene, ImageDatasetCache]:
    from .colmap_dataset_reader import ColmapDatasetReader

    """
    Load a COLMAP scene from the specified path.

    Args:
        dataset_path: Path to the COLMAP dataset directory.
        normalization_type: Type of normalization to apply to the scene.

    Returns:
        An instance of SfmScene containing the loaded data.
    """
    from .colmap_dataset_reader import ColmapDatasetReader

    dataset_reader = ColmapDatasetReader(colmap_path=dataset_path)
    return SfmScene.from_dataset_reader(dataset_reader=dataset_reader), dataset_reader.cache
