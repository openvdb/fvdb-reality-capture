# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

"""
Download checkpoints from S3.
"""

import logging
import pathlib
import sys

import boto3
import yaml

from fvdb_reality_capture.dev import s3

logger = logging.getLogger("download_checkpoints_from_s3")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

s3_client = boto3.client("s3")

# download benchmark_config.yaml from S3
config_path = pathlib.Path.cwd() / "benchmark_config.yaml"
s3.download_uncached("s3://fvdb-data/fvdb-reality-capture/benchmark/benchmark_config.yaml", config_path, s3_client)

# load benchmark_config.yaml
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# get the datasets
datasets = config["datasets"]

# download the checkpoints from S3
# if the path is
# s3://fvdb-data/fvdb-reality-capture/benchmark/garden/run_676e4e49eac3402e025d286381816433219481f9_01/checkpoints/00000664/reconstruct_ckpt.pt
# then the local path should be
# {run_directory}/checkpoints/00000664/reconstruct_ckpt.pt
for dataset_index, dataset in enumerate(datasets):
    run_directory = dataset["run_directory"]
    checkpoint_paths = dataset["checkpoint_paths"]
    for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths):
        # local path is s3 uri with repository, bucket and "benchmark" removed from the start
        local_path = pathlib.Path(checkpoint_path.replace("s3://fvdb-data/fvdb-reality-capture/benchmark/", ""))
        local_path = pathlib.Path("results") / local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading checkpoint {checkpoint_path} to {local_path}")
        if not local_path.exists():
            s3.download_uncached(checkpoint_path, local_path, s3_client)
        else:
            logger.info(f"Checkpoint {checkpoint_path} already exists, skipping download")
        config["datasets"][dataset_index]["checkpoint_paths"][checkpoint_index] = str(local_path)

# save the updated configuration to a new file
with open(config_path, "w") as f:
    yaml.dump(config, f, sort_keys=False)
