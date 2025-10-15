# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import tyro

from fvdb_reality_capture.cli import BaseCommand

from ._convert import Convert
from ._download import Download
from ._evaluate import Evaluate
from ._mesh_basic import MeshBasic
from ._mesh_dlnr import MeshDLNR
from ._points import Points
from ._reconstruct import Reconstruct
from ._resume import Resume
from ._show import Show
from ._show_data import ShowData


def frgs():
    cmd: BaseCommand = tyro.cli(
        Download | Reconstruct | Convert | ShowData | Show | Resume | Evaluate | MeshBasic | MeshDLNR | Points
    )
    cmd.execute()
