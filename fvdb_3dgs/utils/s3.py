# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
import pathlib
import sys
import threading
from urllib.parse import urlparse

import boto3
from botocore.client import BaseClient

logger = logging.getLogger(__name__)


def default_cache_dir() -> pathlib.Path:
    """
    Get the cache directory for a given bucket.
    """
    return pathlib.Path.home() / ".cache"


def _check_cache_hit(
    bucket: str, key: str, cache_dir: pathlib.Path | None = None, client: BaseClient | None = None
) -> bool:
    """
    Check if a file is already cached in the local cache directory.

    Args:
        bucket (str): The name of the S3 bucket.
        key (str): The key of the S3 object.
        cache_dir (pathlib.Path | None): The directory to cache the file in. Defaults to the user's home directory.
        client (boto3.client.S3 | None): The S3 client to use. Defaults to a new client.

    Returns:
        bool: True if the file is already cached, False otherwise.
    """
    base_cache = (cache_dir or default_cache_dir()) / bucket
    local_path = base_cache / key
    if not local_path.exists():
        return False

    # Compare modified date to upstream S3 object
    s3 = client if client is not None else boto3.client("s3")
    try:
        s3_head = s3.head_object(Bucket=bucket, Key=key)
        s3_last_modified = s3_head["LastModified"].timestamp()
        local_mtime = local_path.stat().st_mtime
        # Allow a small time difference due to possible clock skew
        if abs(local_mtime - s3_last_modified) < 2:
            return True
        else:
            return False
    except Exception as e:
        logger.warning(f"Could not check S3 object for {bucket}/{key}: {e}")
        return False


class ProgressPercentage(object):
    """
    Helper class to report progress of S3 uploads and downloads.
    Modified from: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
    """

    def __init__(self, filename):
        self._filename = filename
        if isinstance(filename, str) and filename.startswith("s3://"):
            # Download: get size from S3
            parsed = urlparse(filename)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            s3 = boto3.client("s3")
            try:
                s3_head = s3.head_object(Bucket=bucket, Key=key)
                self._size = float(s3_head["ContentLength"])
            except Exception as e:
                logger.warning(f"Could not get S3 object size for {bucket}/{key}: {e}")
                self._size = 1.0  # Avoid division by zero
        else:
            # Upload: get size from local file
            self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._finalized = False

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write("\r%s %s / %s (%.2f%%)" % (self._filename, self._seen_so_far, self._size, percentage))
            sys.stdout.flush()
            if not self._finalized and self._seen_so_far >= self._size:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._finalized = True


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parse an S3 URI into a bucket and key.

    Args:
        s3_uri (str): The S3 URI to parse.

    Returns:
        tuple[str, str]: The S3 bucket and key.
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def download(s3_uri: str, cache_dir: pathlib.Path | None = None, client: BaseClient | None = None) -> pathlib.Path:
    """
    Download a file from S3 to a local cache directory.

    Args:
        s3_uri (str): The S3 URI of the file to download.
        cache_dir (pathlib.Path | None): The directory to cache the file in. Defaults to the user's home directory.
        client (boto3.client | None): The S3 client to use. Defaults to a new client.

    Returns:
        pathlib.Path: The path to the downloaded file.
    """
    bucket, key = parse_s3_uri(s3_uri)
    s3 = client or boto3.client("s3")
    base_cache = (cache_dir or default_cache_dir()) / bucket
    local_path = base_cache / key
    if _check_cache_hit(bucket, key, cache_dir, s3):
        logger.info(f"File already exists in cache at {local_path}, skipping download.")
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading checkpoint from S3 to {local_path}...")
    s3.download_file(bucket, key, str(local_path), Callback=ProgressPercentage(s3_uri))
    logger.info(f"Downloaded to {local_path}")
    return local_path


def upload(source_path: pathlib.Path, bucket: str, key: str, client: BaseClient | None = None) -> str:
    """
    Upload a file to S3.

    Args:
        source_path (pathlib.Path): The path to the file to upload.
        bucket (str): The name of the S3 bucket.
        key (str): The key of the S3 object.
        client (boto3.client.S3 | None): The S3 client to use. Defaults to a new client.

    Returns:
        str: The S3 URI of the uploaded file.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"File {source_path} does not exist.")
    if not source_path.is_file():
        raise ValueError(f"Path {source_path} is not a file.")

    logger.info(f"Uploading file {source_path} to S3 {bucket} as {key}...")

    s3 = client or boto3.client("s3")
    try:
        s3.upload_file(source_path, bucket, key, Callback=ProgressPercentage(source_path))
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise e

    uri = f"s3://{bucket}/{key}"
    logger.info(f"File uploaded to {uri}")
    return uri


def delete(s3_uri: str, client: BaseClient | None = None) -> None:
    """
    Delete a file from S3.

    Args:
        s3_uri (str): The S3 URI of the file to delete.
        client (boto3.client.S3 | None): The S3 client to use. Defaults to a new client.
    """
    bucket, key = parse_s3_uri(s3_uri)
    s3 = client or boto3.client("s3")
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        logger.info(f"File deleted from {bucket}/{key}")
    except Exception as e:
        logger.error(f"Error deleting {bucket}/{key}: {e}")
        raise e


def exists(s3_uri: str, client: BaseClient | None = None) -> bool:
    """
    Check if a file exists in S3.

    Args:
        s3_uri (str): The S3 URI of the file to check.
        client (boto3.client.S3 | None): The S3 client to use. Defaults to a new client.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    bucket, key = parse_s3_uri(s3_uri)
    s3 = client or boto3.client("s3")
    try:
        return s3.head_object(Bucket=bucket, Key=key) is not None
    except Exception:
        return False
