# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
import pathlib
import sys
import threading

import tyro

from fvdb_3dgs.utils import s3


def main(source_file_path: pathlib.Path, destination_file_path: pathlib.Path):
    """
    Upload a file to the fvdb-data S3 bucket. This only works for developers with write access to the bucket.

    Args:
        source_file_path (pathlib.Path): Path to the file to upload.
        destination_file_path (pathlib.Path): Path to the file in the S3 bucket. Will be prefixed with "fvdb-reality-capture".
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    fvdb_prefix = "fvdb-reality-capture"
    bucket = "fvdb-data"

    uri = s3.upload(source_file_path, bucket, fvdb_prefix / destination_file_path)


if __name__ == "__main__":
    args = tyro.cli(main)
