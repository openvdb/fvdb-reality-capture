# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

"""
Upload checkpoints to S3.
"""

import logging
import pathlib
import sys

import boto3
import yaml

from fvdb_reality_capture.dev import s3

logger = logging.getLogger("upload_checkpoints_to_s3")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# load benchmark_config.yaml
with open("benchmark_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# get the datasets
datasets = config["datasets"]

s3_client = boto3.client("s3")

# upload the checkpoints to S3
for dataset_index, dataset in enumerate(datasets):
    name = dataset["name"]
    path = dataset["path"]
    run_directory = dataset["run_directory"]
    checkpoint_paths = dataset["checkpoint_paths"]

    logger.info(f"Uploading checkpoints for dataset {name} to S3")

    # example input path
    # results/garden/run_676e4e49eac3402e025d286381816433219481f9_01/checkpoints/00000664/reconstruct_ckpt.pt
    # should be uploaded to
    # s3://fvdb-data/fvdb-reality-capture/benchmark/garden/run_676e4e49eac3402e025d286381816433219481f9_01/checkpoints/00000664/reconstruct_ckpt.pt

    for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths):
        logger.info(f"Uploading checkpoint {checkpoint_path} to S3")
        # strip results/ prefix from the checkpoint path if it exists
        if checkpoint_path.startswith("results/"):
            checkpoint_s3_path = checkpoint_path[len("results/") :]
        else:
            checkpoint_s3_path = checkpoint_path
        s3_path = f"fvdb-reality-capture/benchmark/{checkpoint_s3_path}"
        # only upload the file if it doesn't already exist in S3
        if not s3.exists(f"s3://fvdb-data/{s3_path}"):
            s3_uri = s3.upload(pathlib.Path(checkpoint_path), f"fvdb-data", s3_path, client=s3_client)
        else:
            s3_uri = f"s3://fvdb-data/{s3_path}"
            logger.info(f"Checkpoint {checkpoint_path} already exists in S3, skipping upload")

        # replace the checkpoint path with the S3 URI
        checkpoint_paths[checkpoint_index] = s3_uri

    # save the updated configuration to a new file
    config["datasets"][dataset_index]["checkpoint_paths"] = checkpoint_paths
    with open("benchmark_config_temp.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # upload the new config file to S3
    s3.upload(
        pathlib.Path("benchmark_config_temp.yaml"),
        "fvdb-data",
        "fvdb-reality-capture/benchmark/benchmark_config.yaml",
        client=s3_client,
    )

    # delete the temporary config file
    pathlib.Path("benchmark_config_temp.yaml").unlink()
