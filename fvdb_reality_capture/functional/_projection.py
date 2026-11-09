# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Stage 1 of the composable pipeline: project 3D Gaussians into every camera."""

from __future__ import annotations

from typing import cast

import torch
from fvdb import functional as F

from ..enums import CameraModel, ProjectionMethod
from ._autograd import _ProjectGaussiansFn
from ._types import ProjectedGaussians


def requires_distortion_coeffs(camera_model: CameraModel) -> bool:
    """Whether a camera model is a lens-distortion model.

    Every model other than pinhole and orthographic distorts, needs distortion coefficients, and can
    only be projected with the unscented transform. Keeping this the single test means a distortion
    model added to fvdb is handled consistently by projection, rasterization and :class:`~fvdb_reality_capture.GaussianSplat3d`.

    Args:
        camera_model (CameraModel): The camera model.

    Returns:
        distorted (bool): ``True`` for the distortion models, ``False`` for pinhole and orthographic.
    """
    return CameraModel(camera_model) not in (CameraModel.PINHOLE, CameraModel.ORTHOGRAPHIC)


def check_distortion_coeffs(
    distortion_coeffs: torch.Tensor | None, camera_model: CameraModel, num_cameras: int, device: torch.device
) -> torch.Tensor | None:
    """Check packed distortion coefficients for a camera batch before a kernel reads them.

    Pinhole and orthographic cameras ignore the coefficients, so for them ``None`` is returned whatever was
    passed. For the distortion models the kernels read twelve coefficients per camera, so anything else
    is rejected here rather than rendering wrong pixels or tripping a device assert.

    Args:
        distortion_coeffs (torch.Tensor | None): Packed coefficients, ``[C, 12]``, or ``None``.
        camera_model (CameraModel): The batch's camera model; distortion models require the coefficients.
        num_cameras (int): ``C``.
        device (torch.device): Device of the Gaussians and cameras.

    Returns:
        distortion_coeffs (torch.Tensor | None): The coefficients the kernel should receive: ``None`` for
            camera models without distortion, the validated tensor otherwise.

    Raises:
        RuntimeError: If a distortion camera model has no coefficients, or its tensor is not a contiguous
            ``[C, 12]`` tensor on ``device``.
    """
    if not requires_distortion_coeffs(camera_model):
        return None
    if distortion_coeffs is None:
        raise RuntimeError("distortionCoeffs must be provided for OpenCV camera models")
    if list(distortion_coeffs.shape) != [num_cameras, 12]:
        raise RuntimeError(f"distortionCoeffs must have shape ({num_cameras}, 12)")
    if not distortion_coeffs.is_contiguous():
        raise RuntimeError("distortionCoeffs must be contiguous")
    if distortion_coeffs.device != device:
        raise RuntimeError(f"distortionCoeffs must be on {device}, got {distortion_coeffs.device}")
    return distortion_coeffs


def resolve_projection_method(camera_model: CameraModel, projection_method: ProjectionMethod) -> ProjectionMethod:
    """Replace :attr:`~fvdb_reality_capture.ProjectionMethod.AUTO` with the concrete method for a camera model.

    Pinhole and orthographic cameras default to the analytic projection; the distortion models default
    to the unscented transform, which is the only method that supports them.

    Args:
        camera_model (CameraModel): The camera model.
        projection_method (ProjectionMethod): The requested method, possibly ``AUTO``.

    Returns:
        projection_method (ProjectionMethod): ``ANALYTIC`` or ``UNSCENTED``.
    """
    if projection_method != ProjectionMethod.AUTO:
        return ProjectionMethod(projection_method)
    if requires_distortion_coeffs(camera_model):
        return ProjectionMethod.UNSCENTED
    return ProjectionMethod.ANALYTIC


def project_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    world_to_camera_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    image_width: int,
    image_height: int,
    near: float = 0.01,
    far: float = 1e10,
    camera_model: CameraModel = CameraModel.PINHOLE,
    projection_method: ProjectionMethod = ProjectionMethod.AUTO,
    distortion_coeffs: torch.Tensor | None = None,
    min_radius_2d: float = 0.0,
    eps_2d: float = 0.3,
    antialias: bool = False,
    *,
    accumulated_mean_2d_gradient_norms: torch.Tensor | None = None,
    accumulated_gradient_step_counts: torch.Tensor | None = None,
    accumulated_max_2d_radii: torch.Tensor | None = None,
) -> ProjectedGaussians:
    """Project 3D Gaussians onto the image planes of a set of cameras.

    With the analytic method the result is differentiable with respect to ``means``, ``quats``,
    ``log_scales`` and ``world_to_camera_matrices``. The unscented transform has no backward pass;
    to train through it, rasterize with :func:`rasterize_world_space_gaussians`, which
    differentiates through the 3D parameters directly.

    Args:
        means (torch.Tensor): Gaussian centers in world space, ``[N, 3]``.
        quats (torch.Tensor): Gaussian rotations as quaternions, ``[N, 4]``.
        log_scales (torch.Tensor): Natural-log scale factors, ``[N, 3]``.
        world_to_camera_matrices (torch.Tensor): World-to-camera transforms, ``[C, 4, 4]``, contiguous.
        projection_matrices (torch.Tensor): Camera intrinsics, ``[C, 3, 3]``, contiguous.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        near (float): Near clipping plane. Gaussians closer than this are culled.
        far (float): Far clipping plane. Gaussians farther than this are culled.
        camera_model (CameraModel): Camera model of all ``C`` cameras.
        projection_method (ProjectionMethod): Projection method; ``AUTO`` picks per camera model.
        distortion_coeffs (torch.Tensor | None): Packed OpenCV distortion coefficients, ``[C, 12]``,
            contiguous. Required for the OpenCV camera models and ignored otherwise.
        min_radius_2d (float): Gaussians whose projected radius is at most this many pixels are culled.
        eps_2d (float): Blur added to the projected covariance for numerical stability.
        antialias (bool): Compute opacity compensation factors for the blur added by ``eps_2d``.
        accumulated_mean_2d_gradient_norms (torch.Tensor | None): Optional ``[N]`` float accumulator that
            the analytic backward pass adds image-normalized 2D mean gradient norms into.
        accumulated_gradient_step_counts (torch.Tensor | None): Optional ``[N]`` ``int32`` accumulator of
            backward passes per Gaussian. Updated only when given together with the gradient-norm accumulator.
        accumulated_max_2d_radii (torch.Tensor | None): Optional ``[N]`` ``int32`` accumulator of the
            largest projected radius seen per Gaussian. Only updated alongside the other two.

    Returns:
        projected (ProjectedGaussians): The projected Gaussians.
    """
    if not projection_matrices.is_contiguous():
        raise RuntimeError("projectionMatrices must be contiguous")
    if not world_to_camera_matrices.is_contiguous():
        raise RuntimeError("worldToCameraMatrices must be contiguous")
    camera_model = CameraModel(camera_model)
    num_cameras = world_to_camera_matrices.size(0)
    distortion_coeffs = check_distortion_coeffs(distortion_coeffs, camera_model, num_cameras, means.device)

    resolved = resolve_projection_method(camera_model, projection_method)
    if requires_distortion_coeffs(camera_model) and resolved != ProjectionMethod.UNSCENTED:
        raise RuntimeError("OpenCV camera models require ProjectionMethod::UNSCENTED or AUTO")

    if resolved == ProjectionMethod.UNSCENTED:
        if distortion_coeffs is None:
            distortion_coeffs = torch.empty(num_cameras, 0, device=means.device, dtype=means.dtype)
        radii, means2d, depths, conics, compensations = F.project_gaussians_ut_fwd(
            means,
            quats,
            log_scales,
            world_to_camera_matrices,
            world_to_camera_matrices,
            projection_matrices,
            distortion_coeffs,
            camera_model,
            image_width,
            image_height,
            eps_2d,
            near,
            far,
            min_radius_2d,
            antialias,
        )
        if not antialias:
            compensations = None
    else:
        result = cast(
            tuple[torch.Tensor, ...],
            _ProjectGaussiansFn.apply(
                means,
                quats,
                log_scales,
                world_to_camera_matrices,
                projection_matrices,
                image_width,
                image_height,
                eps_2d,
                near,
                far,
                min_radius_2d,
                antialias,
                camera_model == CameraModel.ORTHOGRAPHIC,
                accumulated_mean_2d_gradient_norms,
                accumulated_gradient_step_counts,
                accumulated_max_2d_radii,
            ),
        )
        radii, means2d, depths, conics = result[:4]
        compensations = result[4] if antialias else None

    return ProjectedGaussians(
        radii=radii,
        means2d=means2d,
        depths=depths,
        conics=conics,
        compensations=compensations,
        image_width=image_width,
        image_height=image_height,
        camera_model=camera_model,
        projection_method=resolved,
    )
