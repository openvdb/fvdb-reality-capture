# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tyro.conf import Positional

from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.tools import download_example_data

# dataset names
DatasetName = Literal[
    "all",
    "mipnerf360",
    "gettysburg",
    "safety_park",
    "miris_factory",
]


@dataclass
class Download(BaseCommand):
    """
    Download example datasets for fvdb-reality-capture.
    """

    # The name of the dataset to download. Use "all" to download all datasets.
    name: Positional[DatasetName] = "all"

    # The path to download the dataset to. Data will be downloaded to a
    # subdirectory of this path with the name of the dataset.
    # _e.g._ `--download-path ./data` will download the Gettysburg dataset to `./data/gettysburg`.
    # Defaults to ${CWD}/data.
    download_path: str | Path = Path.cwd() / "data"

    def execute(self) -> None:
        download_example_data(self.name, self.download_path)
