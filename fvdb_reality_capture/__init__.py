# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from importlib.metadata import PackageNotFoundError, version

from . import dev, foundation_models, radiance_fields, sfm_scene, tools, transforms
from .tools import download_example_data

try:
    __version__ = version("fvdb_reality_capture")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "dev",
    "foundation_models",
    "radiance_fields",
    "sfm_scene",
    "tools",
    "transforms",
    "download_example_data",
]
