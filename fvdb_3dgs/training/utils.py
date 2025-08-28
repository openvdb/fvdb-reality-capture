# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time


def make_unique_name_directory_based_on_time(results_base_path: pathlib.Path, prefix: str) -> tuple[str, pathlib.Path]:
    """
    Generate a unique name and directory based on the current time.

    The run directory will be created under `results_base_path` with a name in the format
    `prefix_YYYY-MM-DD-HH-MM-SS`. If a directory with the same name already exists,
    it will attempt to create a new one by appending an incremented number to

    Returns:
        run_name: A unique run name in the format "run_YYYY-MM-DD-HH-MM-SS".
        run_path: A pathlib.Path object pointing to the created directory.
    """
    attempts = 0
    max_attempts = 50
    run_name = f"{prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    logger = logging.getLogger(__name__)
    while attempts < 50:
        results_path = results_base_path / run_name
        try:
            results_path.mkdir(exist_ok=False, parents=True)
            break
        except FileExistsError:
            attempts += 1
            logger.debug(f"Directory {results_path} already exists. Attempting to create a new one.")
            # Generate a new run name with an incremented attempt number
            run_name = f"{prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}_{attempts+1:02d}"
            continue
    if attempts >= max_attempts:
        raise FileExistsError(f"Failed to generate a unique results directory name after {max_attempts} attempts.")

    logger.info(f"Creating unique directory with name {run_name} after {attempts} attempts.")

    return run_name, results_path
