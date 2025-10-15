# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from typing import Annotated

import tyro

from .base_command import BaseCommand
from .convert import Convert
from .download import Download
from .evaluate import Evaluate
from .mesh import Mesh
from .reconstruct import Reconstruct
from .show import Show
from .show_data import ShowData


def frgs():
    cmd: BaseCommand = tyro.cli(Download | Reconstruct | Convert | ShowData | Show | Evaluate | Mesh)
    cmd.execute()
