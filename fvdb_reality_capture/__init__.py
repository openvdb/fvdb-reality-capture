# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from importlib.metadata import PackageNotFoundError, version

from .enums import CameraModel, ProjectionMethod, RollingShutterType
from . import radiance_fields
from .radiance_fields import (
    GaussianSplat3d,
    ProjectedGaussianSplats,
    evaluate_spherical_harmonics,
    gaussian_render_jagged,
    gaussian_splat_to_view_data,
)
from . import dev, foundation_models, sfm_scene, tools, transforms
from .tools import download_example_data

try:
    __version__ = version("fvdb_reality_capture")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "GaussianSplat3d",
    "ProjectedGaussianSplats",
    "gaussian_render_jagged",
    "evaluate_spherical_harmonics",
    "gaussian_splat_to_view_data",
    "RollingShutterType",
    "CameraModel",
    "ProjectionMethod",
    "dev",
    "foundation_models",
    "radiance_fields",
    "sfm_scene",
    "tools",
    "transforms",
    "download_example_data",
]
