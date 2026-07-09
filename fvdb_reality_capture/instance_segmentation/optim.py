# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Literal

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class ExponentialLRWithRampUpScheduler(LRScheduler):
    """Exponential decay scheduler with linear or cosine warmup ramp.

    The scheduler first ramps up to ``lr_init`` over ``warmup_steps`` steps,
    then exponentially decays to ``lr_final`` over the remaining steps until
    ``max_steps``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_init: float,
        lr_pre_warmup: float = 1e-8,
        lr_final: float | None = None,
        warmup_steps: int = 0,
        max_steps: int = 100000,
        ramp: Literal["linear", "cosine"] = "cosine",
        last_epoch: int = -1,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: The optimizer to schedule.
            lr_init: Initial learning rate after warmup.
            lr_pre_warmup: Learning rate before warmup.
            lr_final: Final learning rate. If not provided, it will be set to lr_init.
            warmup_steps: Number of warmup steps.
            max_steps: The maximum number of steps.
            ramp: The ramp function to use during the warmup.
            last_epoch: The index of last epoch.
        """
        self._lr_init = lr_init
        self._lr_pre_warmup = lr_pre_warmup
        self._lr_final = lr_final if lr_final is not None else lr_init
        self._warmup_steps = warmup_steps
        self._max_steps = max_steps
        self._ramp = ramp
        super().__init__(optimizer, last_epoch)

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, value: int):
        self._max_steps = value

    def get_lr(self) -> list[float]:
        step = self.last_epoch + 1
        if step < self._warmup_steps:
            if self._ramp == "cosine":
                lr = self._lr_pre_warmup + (self._lr_init - self._lr_pre_warmup) * np.sin(
                    0.5 * np.pi * np.clip(step / self._warmup_steps, 0, 1)
                )
            else:
                lr = self._lr_pre_warmup + (self._lr_init - self._lr_pre_warmup) * step / self._warmup_steps
        else:
            t = np.clip((step - self._warmup_steps) / (self._max_steps - self._warmup_steps), 0, 1)
            lr = np.exp(np.log(self._lr_init) * (1 - t) + np.log(self._lr_final) * t)

        return [lr] * len(self.base_lrs)
