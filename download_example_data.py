# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from typing import Literal

import tyro

# dataset names
dataset_names = Literal[
    "all",
    "mipnerf360",
    "gettysburg",
    "safety_park",
    "miris_factory",
]


def main(
    dataset: dataset_names = "all",
    download_path: str | Path = Path.cwd() / "data",
):
    """
    Download example datasets used in FVDB-RealityCapture.

    Args:
        dataset (str): Name of the dataset to download. Options are "mipnerf360" or "gettysburg".
        download_path (str | Path): Path to the directory where the dataset will be downloaded.
            Default is the current working directory + "data".
    """

    from fvdb_reality_capture.tools._download_example_data import download_example_data

    download_example_data(dataset, download_path)


if __name__ == "__main__":
    tyro.cli(main)
