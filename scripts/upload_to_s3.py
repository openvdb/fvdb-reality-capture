# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
import pathlib
import sys
import threading

import boto3
import tyro


class ProgressPercentage(object):
    """
    Helper class to report progress of S3 uploads.
    Taken from: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
    """

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write("\r%s %s / %s (%.2f%%)" % (self._filename, self._seen_so_far, self._size, percentage))
            sys.stdout.flush()


def main(source_file_path: pathlib.Path, destination_file_path: pathlib.Path):
    """
    Upload a file to the fvdb-data S3 bucket. This only works for developers with write access to the bucket.

    Args:
        source_file_path (pathlib.Path): Path to the file to upload.
        destination_file_path (pathlib.Path): Path to the file in the S3 bucket. Will be prefixed with "fvdb-reality-capture".
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not source_file_path.exists():
        raise FileNotFoundError(f"File {source_file_path} does not exist.")
    if not source_file_path.is_file():
        raise ValueError(f"Path {source_file_path} is not a file.")

    fvdb_prefix = "fvdb-reality-capture"

    logger.info(f"Uploading file {source_file_path} to S3 bucket fvdb-data as {fvdb_prefix}/{destination_file_path}...")
    s3 = boto3.client("s3")

    local_file_path = str(source_file_path)
    bucket_name = "fvdb-data"
    s3_object_key = str(pathlib.Path(fvdb_prefix) / str(destination_file_path))

    try:
        s3.upload_file(local_file_path, bucket_name, s3_object_key, Callback=ProgressPercentage(local_file_path))
        logger.info(f"File '{local_file_path}' uploaded to S3 bucket '{bucket_name}' as '{s3_object_key}'")
    except Exception as e:
        logger.error(f"Error uploading file: {e}")


if __name__ == "__main__":
    args = tyro.cli(main)
