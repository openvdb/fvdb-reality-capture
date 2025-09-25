# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib

import requests
import tqdm

_PRETRAINED_WEIGHTS_PATH = pathlib.Path(__file__).parent / "_weights"


def get_weights_path_for_model(filename: str | pathlib.Path, weights_url: str, model_name: str = "") -> pathlib.Path:
    """
    Get the path to a weights file named `filename` downloading the file from `weights_url` if
    it does not exist in the directory defined by `get_pretrained_weights_path()`.

    Args:
        filename (str | pathlib.Path): The name of the weights file.
        weights_url (str): The URL to download the weights from.
        model_name (str, optional): The name of the model using these weights (for logging purposes).

    Returns:
        pathlib.Path: A valid path to the weights file.
    """
    path_to_weights = get_pretrained_weights_path() / filename
    if not path_to_weights.exists():
        _logger = logging.getLogger(f"fvdb_gs.io.get_pretrained_weights")
        _logger.info(f"Weights not found at {path_to_weights}. Downloading from {weights_url}.")
        response = requests.get(weights_url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            assert total_size > 0, "Downloaded file is empty."
            with open(path_to_weights, "wb") as f:
                with tqdm.tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading weights {'(' + model_name + ')' if model_name else ''}",
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            _logger.info("Weights downloaded successfully.")
        else:
            raise RuntimeError(f"Failed to download weights from {weights_url}. Status code: {response.status_code}")

    return path_to_weights


def get_pretrained_weights_path() -> pathlib.Path:
    """
    Returns the path to the pretrained weights.
    """
    if not _PRETRAINED_WEIGHTS_PATH.exists():
        _PRETRAINED_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
    return _PRETRAINED_WEIGHTS_PATH


def set_pretrained_weights_path(path: pathlib.Path | str):
    """
    Sets the path to the pretrained weights.
    """
    global _PRETRAINED_WEIGHTS_PATH
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not path.is_absolute():
        path = path.absolute()
    _PRETRAINED_WEIGHTS_PATH = pathlib.Path(path)
    if not _PRETRAINED_WEIGHTS_PATH.exists():
        _PRETRAINED_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
