# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Literal, Sequence

import numpy as np
import pyproj

from .sfm_scene import SfmScene


def _geo_ecef2enu_normalization_transform(points):
    """
    Compute a transformation matrix that converts ECEF coordinates to ENU coordinates.

    Args:
        point_cloud: Nx3 array of points in ECEF coordinates

    Returns:
        transform: 4x4 transformation matrix transforming ECEF to ENU coordinates
    """
    xorigin, yorigin, zorigin = np.median(points, axis=0)
    tform_ecef2lonlat = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    pt_lonlat = tform_ecef2lonlat.transform(xorigin, yorigin, zorigin)
    londeg, latdeg = pt_lonlat[0], pt_lonlat[1]

    # ECEF to ENU rotation matrix
    lon = np.deg2rad(londeg)
    lat = np.deg2rad(latdeg)
    rot = np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)],
            [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)],
        ]
    )

    tvec = np.array([xorigin, yorigin, zorigin])
    # Create SE(3) matrix (4x4 transformation matrix)
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = -rot @ tvec

    return transform


def _pca_normalization_transform(point_cloud):
    """
    Compute a transormation matrix that normalizes the scene using PCA on a set of input points

    Args:
        point_cloud: Nx3 array of points

    Returns:
        transform: 4x4 transformation matrix
    """
    # Compute centroid
    centroid = np.median(point_cloud, axis=0)

    # Translate point cloud to centroid
    translated_point_cloud = point_cloud - centroid

    # Compute covariance matrix
    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues (descending order) so that the z-axis
    # is the principal axis with the smallest eigenvalue.
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # Check orientation of eigenvectors. If the determinant of the eigenvectors is
    # negative, then we need to flip the sign of one of the eigenvectors.
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # Create rotation matrix
    rotation_matrix = eigenvectors.T

    # Create SE(3) matrix (4x4 transformation matrix)
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid

    return transform


def _camera_similarity_normalization_transform(c2w, strict_scaling=False, center_method="focus"):
    """
    Get a similarity transformation to normalize a scene given its camera -> world transformations

    Args:
        c2w: A set of camera -> world transformations [R|t] (N, 4, 4)
        strict_scaling: If set to true, use the maximum distance to any camera to rescale the scene
                        which may not be that robust. If false, use the median
        center_method: If set to 'focus' use the focus of the scene to center the cameras
                        If set to 'poses' use the center of the camera positions to center the cameras

    Returns:
        transform: A 4x4 normalization transform (4,4)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene.
    if center_method == "focus":
        # find the closest point to the origin for each camera's center ray
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        # use center of the camera positions
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale

    return transform


def percentile_filter_sfm_scene(
    sfm_scene: SfmScene,
    percentile_min: Sequence[float | int] | np.ndarray | None = None,
    percentile_max: Sequence[float | int] | np.ndarray | None = None,
) -> SfmScene:
    """
    Filter the points in an `SfmScene` based on percentile bounds.

    Args:
        sfm_scene: SfmScene object containing camera and point data
        percentile_min: Tuple of minimum percentiles (from 0 to 100) for x, y, z coordinates
            or None to use (0, 0, 0) (default: None)
        percentile_max: Tuple of maximum percentiles (from 0 to 100) for x, y, z coordinates
            or None to use (100, 100, 100) (default: None)

    Returns:
        SfmScene: A new SfmScene object with points filtered based on the specified percentile bounds.
    """
    if percentile_min is not None and percentile_max is None:
        percentile_max = np.array([100.0, 100.0, 100.0])

    if percentile_max is not None and percentile_min is None:
        percentile_min = np.array([0.0, 0.0, 0.0])
    if percentile_min is None and percentile_max is None:
        return sfm_scene

    assert (
        percentile_max is not None and percentile_min is not None
    ), "Both percentile_min and percentile_max must be provided or None."

    if len(percentile_min) != 3:
        raise ValueError(f"percentile_min must be a sequence of length 3. Got {percentile_min} instead.")
    if len(percentile_max) != 3:
        raise ValueError(f"percentile_max must be a sequence of length 3. Got {percentile_max} instead.")

    percentile_min = np.asarray(percentile_min, dtype=float)
    percentile_max = np.asarray(percentile_max, dtype=float)

    if np.all(percentile_min <= 0) and np.any(percentile_min >= 100):
        return sfm_scene

    percentile_min = np.clip(percentile_min, 0.0, 100.0)
    percentile_max = np.clip(percentile_max, 0.0, 100.0)

    points = sfm_scene.points
    lower_boundx = np.percentile(points[:, 0], percentile_min[0])
    upper_boundx = np.percentile(points[:, 0], percentile_max[0])

    lower_boundy = np.percentile(points[:, 1], percentile_min[1])
    upper_boundy = np.percentile(points[:, 1], percentile_max[1])

    lower_boundz = np.percentile(points[:, 2], percentile_min[2])
    upper_boundz = np.percentile(points[:, 2], percentile_max[2])

    good_map = np.logical_and.reduce(
        [
            points[:, 0] > lower_boundx,
            points[:, 0] < upper_boundx,
            points[:, 1] > lower_boundy,
            points[:, 1] < upper_boundy,
            points[:, 2] > lower_boundz,
            points[:, 2] < upper_boundz,
        ]
    )

    if np.sum(good_map) == 0:
        raise ValueError(
            f"No points found in the specified percentile range: "
            f"min={percentile_min}, max={percentile_max}. "
            "Please adjust the percentile values."
        )
    return sfm_scene.filter_points(good_map)


def normalize_sfm_scene(
    sfm_scene: SfmScene, normalization_type: Literal["pca", "none", "ecef2enu", "similarity"]
) -> tuple[SfmScene, np.ndarray]:
    """
    Normalize the SfmScene using the specified normalization type.

    Args:
        sfm_scene: SfmScene object containing camera and point data
        normalization_type: Type of normalization to apply. Options are "pca", "similarity", "ecef2enu", or "none".

    Returns:
        transformed_sfm_scene: The SfmScene object after applying the normalization transform.
        normalization_transform: 4x4 transformation matrix for normalizing the scene
    """
    points = sfm_scene.points
    world_to_camera_matrices = sfm_scene.camera_to_world_matrices

    # Normalize the world space.
    if normalization_type == "pca":
        normalization_transform = _pca_normalization_transform(points)
    elif normalization_type == "ecef2enu":
        normalization_transform = _geo_ecef2enu_normalization_transform(points)
    elif normalization_type == "similarity":
        camera_to_world_matrices = np.linalg.inv(world_to_camera_matrices)
        normalization_transform = _camera_similarity_normalization_transform(camera_to_world_matrices)
    elif normalization_type == "none":
        normalization_transform = np.eye(4)
    else:
        raise RuntimeError(f"Unknown normalization type {normalization_type}")

    return sfm_scene.transform(normalization_transform), normalization_transform
