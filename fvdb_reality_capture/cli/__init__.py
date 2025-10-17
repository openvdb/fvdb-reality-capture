# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from abc import ABC, abstractmethod


class BaseCommand(ABC):
    """
    Base class for CLI commands.

    Commands should implement the `execute` method, which will be called when the command is run.
    """

    @abstractmethod
    def execute(self) -> None:
        """
        Execute the command.
        """
        pass
