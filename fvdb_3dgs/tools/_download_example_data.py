# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib
import shutil

import requests
import tqdm


def _download_one_dataset(dataset_name: str, dataset_url: str, dataset_download_path: pathlib.Path):
    logger = logging.getLogger(f"{__name__}.download_example_data")
    dataset_filename = pathlib.Path(dataset_url).name
    dataset_file_path = dataset_download_path / dataset_filename

    if dataset_download_path.exists():
        logger.warning(f"Dataset directory {dataset_download_path} already exists. Skipping download.")
        return

    dataset_download_path.mkdir(parents=True, exist_ok=True)

    response = requests.get(dataset_url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        assert total_size > 0, "Downloaded file is empty."
        logger.info(f"Downloading dataset {dataset_name} from {dataset_url} to {dataset_file_path}")
        with open(dataset_file_path, "wb") as f:
            with tqdm.tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading dataset {dataset_name}",
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        logger.info("Dataset downloaded successfully.")
    else:
        raise RuntimeError(f"Failed to download weights from {dataset_url}. Status code: {response.status_code}")

    logger.info(f"Extracting archive {dataset_filename} to {dataset_download_path}.")
    shutil.unpack_archive(dataset_file_path, extract_dir=dataset_download_path)


def download_example_data(dataset="all", download_path: str | pathlib.Path = pathlib.Path.cwd() / "data"):

    # dataset urls

    dataset_urls = {
        "mipnerf360": "https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/360_v2.zip",
        "gettysburg": "https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/gettysburg.zip",
        "safety_park": "https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/safety_park.zip",
    }

    # where each dataset goes
    dataset_directories = {
        "mipnerf360": "360_v2",
        "gettysburg": "gettysburg",
        "safety_park": "safety_park",
    }

    if isinstance(download_path, str):
        download_path = pathlib.Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)

    if dataset == "all":
        for dataset_name in dataset_urls:
            dataset_download_path = download_path / dataset_directories[dataset_name]
            _download_one_dataset(dataset_name, dataset_urls[dataset_name], dataset_download_path)
    else:
        dataset_download_path = download_path / dataset_directories[dataset]
        _download_one_dataset(dataset, dataset_urls[dataset], dataset_download_path)
