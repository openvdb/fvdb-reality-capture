# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
from pathlib import Path

from tyro.conf import Positional

from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.dev import s3


@dataclass
class S3Upload(BaseCommand):
    """
    Upload a file to the fvdb-data S3 bucket. This only works for developers with write access to the bucket.
    """

    # The path to the file to upload.
    source_file_path: Positional[Path]

    # The path to the file in the S3 bucket. Will be prefixed with "fvdb-reality-capture".
    destination_file_path: Positional[Path]

    def execute(self) -> None:
        fvdb_prefix = "fvdb-reality-capture"
        bucket = "fvdb-data"

        s3.upload(self.source_file_path, bucket, str(Path(fvdb_prefix) / self.destination_file_path))


@dataclass
class S3Download(BaseCommand):
    """
    Download a file from the fvdb-data S3 bucket. This only works for developers with read access to the bucket.
    """

    # The path to the file to download.
    source_file_path: Positional[Path]

    # The path to the file in the S3 bucket. Will be prefixed with "fvdb-reality-capture".
    destination_file_path: Positional[Path]

    def execute(self) -> None:
        fvdb_prefix = "fvdb-reality-capture"
        bucket = "fvdb-data"
        src_uri = f"s3://{bucket}/{fvdb_prefix}/{self.source_file_path}"

        dst_path = self.destination_file_path.resolve()
        s3.download(src_uri, dst_path.parent)


@dataclass
class S3Rm(BaseCommand):
    """
    Delete a file from the fvdb-data S3 bucket. This only works for developers with write access to the bucket.
    """

    # The path to the file to delete.
    file_path: Positional[Path]

    def execute(self) -> None:
        fvdb_prefix = "fvdb-reality-capture"
        bucket = "fvdb-data"
        src_uri = f"s3://{bucket}/{fvdb_prefix}/{self.file_path}"

        s3.delete(src_uri)


@dataclass
class S3Exists(BaseCommand):
    """
    Check if a file exists in the fvdb-data S3 bucket. This only works for developers with read access to the bucket.
    """

    # The path to the file to check.
    file_path: Positional[Path]

    def execute(self) -> None:
        fvdb_prefix = "fvdb-reality-capture"
        bucket = "fvdb-data"
        src_uri = f"s3://{bucket}/{fvdb_prefix}/{self.file_path}"

        if s3.exists(src_uri):
            print(f"File {src_uri} exists.")
        else:
            print(f"File {src_uri} does not exist.")
