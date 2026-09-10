# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""CLI-owned dispatch registry for resumable training methods."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from fvdb_reality_capture.checkpoints import TrainingCheckpoint


class ResumeWriterOptions(Protocol):
    save_images: bool
    save_checkpoints: bool
    save_plys: bool
    save_metrics: bool
    metrics_file_buffer_size: int
    use_tensorboard: bool
    save_images_to_tensorboard: bool
    log_path: pathlib.Path | None
    log_every: int


class ResumeContext(Protocol):
    """Options available to method-specific resume callbacks."""

    io: ResumeWriterOptions
    run_name: str | None
    update_viz_every: float
    viewer_port: int
    viewer_ip_address: str
    device: str | torch.device
    verbose: bool
    reconstruction_path: pathlib.Path | None


ResumeCallback = Callable[[TrainingCheckpoint, ResumeContext, pathlib.Path], None]


class UnknownCheckpointMethodError(ValueError):
    """Raised when no CLI resume handler is registered for a checkpoint method."""


@dataclass(frozen=True)
class ResumeHandler:
    """Resume implementation registered for one stable checkpoint method ID."""

    method: str
    default_output_name: str
    callback: ResumeCallback


_RESUME_HANDLERS: dict[str, ResumeHandler] = {}


def register_resume_handler(handler: ResumeHandler) -> None:
    """Register a method-specific resume implementation."""
    if not handler.method:
        raise ValueError("Resume handler method must be non-empty")
    if handler.method in _RESUME_HANDLERS:
        raise ValueError(f"A resume handler is already registered for {handler.method!r}")
    _RESUME_HANDLERS[handler.method] = handler


def get_resume_handler(method: str) -> ResumeHandler:
    """Return the handler for *method* or raise a diagnostic error."""
    handler = _RESUME_HANDLERS.get(method)
    if handler is None:
        registered = ", ".join(sorted(_RESUME_HANDLERS)) or "none"
        raise UnknownCheckpointMethodError(
            f"No resume handler is registered for checkpoint method {method!r}. Registered methods: {registered}"
        )
    return handler


def registered_resume_methods() -> tuple[str, ...]:
    """Return stable method IDs with registered resume handlers."""
    return tuple(sorted(_RESUME_HANDLERS))


__all__ = [
    "ResumeContext",
    "ResumeHandler",
    "UnknownCheckpointMethodError",
    "get_resume_handler",
    "register_resume_handler",
    "registered_resume_methods",
]
