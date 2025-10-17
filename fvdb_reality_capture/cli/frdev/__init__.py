# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import tyro

from fvdb_reality_capture.cli import BaseCommand

from ._s3 import S3Download, S3Exists, S3Rm, S3Upload


def frdev():
    cmd: BaseCommand = tyro.cli(S3Download | S3Rm | S3Upload | S3Exists)
    cmd.execute()
