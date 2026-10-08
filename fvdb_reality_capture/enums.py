# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Enums used by the Gaussian splatting API.

The camera enums that fvdb kernels accept are owned by :mod:`fvdb` and re-exported here unchanged,
so :class:`fvdb_reality_capture.CameraModel` is the same object as :class:`fvdb.CameraModel`.
:class:`ProjectionMethod` and :class:`GaussianRenderMode` select stages of the composable rendering
pipeline in :mod:`fvdb_reality_capture.functional` and are defined here.
"""

from enum import IntEnum

from fvdb import CameraModel, RollingShutterType

__all__ = ["RollingShutterType", "CameraModel", "ProjectionMethod", "GaussianRenderMode"]


class ProjectionMethod(IntEnum):
    """
    Which fvdb projection kernel :func:`fvdb_reality_capture.functional.project_gaussians` calls.
    """

    AUTO = 0
    """Choose the default implementation for the selected camera model."""

    ANALYTIC = 1
    """Use the analytic (EWA) projection path."""

    UNSCENTED = 2
    """Use the unscented-transform projection path."""


class GaussianRenderMode(IntEnum):
    """
    Which per-Gaussian features :func:`fvdb_reality_capture.functional.evaluate_gaussian_sh` produces
    for rasterization.
    """

    FEATURES = 0
    """Spherical-harmonics evaluated features only, ``[C, N, D]``."""

    DEPTH = 1
    """View-space depth only, ``[C, N, 1]``. No spherical harmonics are evaluated."""

    FEATURES_AND_DEPTH = 2
    """Spherical-harmonics features with depth appended as the last channel, ``[C, N, D + 1]``."""
