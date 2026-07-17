# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from enum import IntEnum

__all__ = ["RollingShutterType", "CameraModel", "ProjectionMethod"]


class RollingShutterType(IntEnum):
    """
    Rolling shutter policy for camera projection / ray generation.

    Rolling shutter models treat different image rows/columns as having different exposure times.
    FVDB uses this to interpolate between per-camera start/end poses when generating rays.
    """

    NONE = 0
    """
    No rolling shutter: the start pose is used for all pixels.
    """

    VERTICAL = 1
    """
    Vertical rolling shutter: exposure time varies with image row (y).
    """

    HORIZONTAL = 2
    """
    Horizontal rolling shutter: exposure time varies with image column (x).
    """


class CameraModel(IntEnum):
    """
    Camera model for projection / ray generation.

    Notes:

    - ``PINHOLE`` and ``ORTHOGRAPHIC`` ignore distortion coefficients.
    - ``OPENCV_*`` variants use pinhole intrinsics plus OpenCV-style distortion. When distortion
      coefficients are provided, FVDB expects a packed layout:

      ``[k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]``

      Unused coefficients for a given model should be set to 0.
    """

    PINHOLE = 0
    """
    Ideal pinhole camera model (no distortion).
    """

    OPENCV_RADTAN_5 = 1
    """
    OpenCV radial-tangential distortion with 5 parameters (k1,k2,p1,p2,k3).
    """

    OPENCV_RATIONAL_8 = 2
    """
    OpenCV rational radial-tangential distortion with 8 parameters (k1..k6,p1,p2).
    """

    OPENCV_RADTAN_THIN_PRISM_9 = 3
    """
    OpenCV radial-tangential + thin-prism distortion with 9 parameters (k1,k2,p1,p2,k3,s1..s4).
    """

    OPENCV_THIN_PRISM_12 = 4
    """
    OpenCV rational radial-tangential + thin-prism distortion with 12 parameters
    (k1..k6,p1,p2,s1..s4).
    """

    ORTHOGRAPHIC = 5
    """
    Orthographic camera model (no distortion).
    """


class ProjectionMethod(IntEnum):
    """
    Projection implementation selector for Gaussian splatting camera models.
    """

    AUTO = 0
    """
    Choose the default implementation for the selected camera model.
    """

    ANALYTIC = 1
    """
    Use the analytic projection path.
    """

    UNSCENTED = 2
    """
    Use the unscented projection path.
    """
