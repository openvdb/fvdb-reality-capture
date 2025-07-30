# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib

_PRETRAINED_WEIGHTS_PATH = pathlib.Path(__file__).parent / "_weights"


def get_pretrained_weights_path():
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
