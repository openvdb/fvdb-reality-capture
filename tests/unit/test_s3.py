# tests for fvdb_3dgs.utils.s3

import pathlib
import tempfile

import boto3
import pytest

from fvdb_3dgs.utils import s3


@pytest.fixture(scope="module")
def s3_client():
    return boto3.client("s3")


def test_upload_download_delete(s3_client):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir) / "test.txt"
        tmp_path.write_text("test")
        uri = s3.upload(tmp_path, "fvdb-data", "fvdb-reality-capture/test.txt", client=s3_client)
        downloaded_path = s3.download(uri, client=s3_client)
        assert downloaded_path.read_bytes() == tmp_path.read_bytes()
        assert s3.download(uri, client=s3_client) == downloaded_path
        s3.delete(uri, client=s3_client)
        assert not s3.exists(uri, client=s3_client)


def test_cacheing(s3_client):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir) / "test.txt"
        tmp_path.write_text("test")
        uri = s3.upload(tmp_path, "fvdb-data", "fvdb-reality-capture/test.txt", client=s3_client)
        downloaded_path = s3.download(uri, cache_dir=pathlib.Path(tmp_dir), client=s3_client)
        assert downloaded_path == pathlib.Path(tmp_dir) / "fvdb-data" / "fvdb-reality-capture" / "test.txt"
        assert downloaded_path.read_bytes() == tmp_path.read_bytes()
        assert s3.download(uri, cache_dir=pathlib.Path(tmp_dir), client=s3_client) == downloaded_path
        s3.delete(uri, client=s3_client)
        assert not s3.exists(uri, client=s3_client)


def test_default_s3_client():
    # test that there are no issues using a default-constructed client
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir) / "test.txt"
        tmp_path.write_text("test")
        uri = s3.upload(tmp_path, "fvdb-data", "fvdb-reality-capture/test.txt")
        s3.delete(uri)
