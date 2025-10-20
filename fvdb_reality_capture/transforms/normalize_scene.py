# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any, Literal

import numpy as np
import pyproj

from fvdb_reality_capture.sfm_scene import SfmScene

from .base_transform import BaseTransform, transform


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
    eigenvectors *= -1.0  # Flip to right-handed coordinate system

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

    # Estimate the up vector of the scene as the average up vector of all the camera poses
    # Note that camera space coordinates are assumed to be x-right, y-down, z-forward.
    # To compute the up vector in world space, we therefore use the negative y-axis
    # (i.e. OpenCV convention)
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    # 1. Compute a rotation matrix that rotates the estimated world up-vector to align with +z
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

    # Compute new camera pose transformations in the aligned space
    R = R_align @ R
    t = (R_align @ t[..., None])[..., 0]

    # 2. Compute a centroid for the scene using one of two methods
    if center_method == "focus":
        # Use the "focus" of the scene defined as the closest point to the origin in the new aligned space
        # along each camera's center ray.
        lookat_vector = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
        nearest = t + (lookat_vector * -t).sum(-1)[:, None] * lookat_vector
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        # Use the median value of the camera positions to center the scene
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    # 3. Compute a rescaling factor so the scene fits within a unit cube
    # Use either the median or maximum distance to any camera position
    # to determine the scale factor
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))

    # Build the final similarity transform
    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align
    transform[:3, :] *= scale

    return transform


@transform
class NormalizeScene(BaseTransform):
    """
    A :class:`~base_transform.BaseTransform` which normalizes an :class:`~fvdb_reality_capture.sfm_scene.SfmScene`
    using a variety of approaches. This transform applies a rotation/translation/scaling to the entire scene, including
    both points and camera poses.

    The normalization types available are:

    * ``"pca"``: Normalizes by centering the scene about its median point, and rotating the point cloud to align with
      its principal axes.

    * ``"ecef2enu"``: Converts a scene whose points and camera poses are in
      `Earth-Centered, Earth-Fixed (ECEF) <https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system>`_
      coordinates to `East-North-Up (ENU) <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_ coordinates,
      centering the scene around the median point.

    * ``"similarity"``: Rotate the scene so that +z aligns with the average up vector of the cameras, center the scene
      around the median camera position, and rescale the scene to fit within a unit cube.

    * ``"none"``: Do not apply any normalization to the scene. Effectively a no-op.

    Example usage:

    .. code-block:: python

        from fvdb_reality_capture import transforms
        from fvdb_reality_capture.sfm_scene import SfmScene

        # Create a NormalizeScene transform to normalize the scene using PCA
        transform = transforms.NormalizeScene(normalization_type="pca")

        # Apply the transform to an SfmScene
        input_scene: SfmScene = ...
        output_scene: SfmScene = transform(input_scene)

    """

    version = "1.0.0"

    valid_normalization_types = ["pca", "ecef2enu", "similarity", "none"]

    def __init__(self, normalization_type: Literal["pca", "none", "ecef2enu", "similarity"]):
        """
        Create a new :class:`NormalizeScene` transform which normalizes an
        :class:`~fvdb_reality_capture.sfm_scene.SfmScene` using the specified normalization type.

        Normalization is applied to both the points and camera poses in the scene.

        Args:
            normalization_type (str): The type of normalization to apply. Options are ``"pca"``,
                ``"none"``, ``"ecef2enu"``, or ``"similarity"``.
        """
        super().__init__()
        if normalization_type not in self.valid_normalization_types:
            raise ValueError(
                f"Invalid normalization type '{normalization_type}'. "
                f"Valid options are: {', '.join(self.valid_normalization_types)}."
            )
        self._normalization_type = normalization_type
        self._normalization_transform = None
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Return a new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` which is the result of applying the
        normalization transform to the input scene.

        The normalization transform is computed based on the specified normalization type and the contents of
        the input scene. It is applied to both the points and camera poses in the scene.

        Args:
            input_scene (SfmScene): Input :class:`~fvdb_reality_capture.sfm_scene.SfmScene` object containing
                camera and point data

        Returns:
            output_scene (SfmScene): A new :class:`~fvdb_reality_capture.sfm_scene.SfmScene` after applying
                the normalization transform.
        """
        self._logger.info(f"Normalizing SfmScene with normalization type: {self._normalization_type}")

        normalization_transform = self._compute_normalization_transform(input_scene)
        if normalization_transform is None:
            self._logger.warning("Returning the input scene unchanged.")
            return input_scene

        return input_scene.apply_transformation_matrix(normalization_transform)

    def _compute_normalization_transform(self, input_scene: SfmScene) -> np.ndarray | None:
        """
        Compute the normalization transformation matrix for the scene based on the specified normalization type.

        Args:
            input_scene (SfmScene): The input scene to normalize.

        Returns:
            transformation_matrix (np.ndarray | None): The 4x4 normalization transformation matrix, or ``None``
                if the scene lacks points or camera matrices.
        """
        if self._normalization_transform is None:
            points = input_scene.points
            world_to_camera_matrices = input_scene.camera_to_world_matrices

            if points is None or len(points) == 0:
                self._logger.warning("No points found in the SfmScene.")
                return None
            if world_to_camera_matrices is None or len(world_to_camera_matrices) == 0:
                self._logger.warning("No camera matrices found in the SfmScene.")
                return None

            # Normalize the world space.
            if self._normalization_type == "pca":
                normalization_transform = _pca_normalization_transform(points)
            elif self._normalization_type == "ecef2enu":
                normalization_transform = _geo_ecef2enu_normalization_transform(points)
            elif self._normalization_type == "similarity":
                camera_to_world_matrices = np.linalg.inv(world_to_camera_matrices)
                normalization_transform = _camera_similarity_normalization_transform(camera_to_world_matrices)
            elif self._normalization_type == "none":
                normalization_transform = np.eye(4)
            else:
                raise RuntimeError(f"Unknown normalization type {self._normalization_type}")

            self._normalization_transform = normalization_transform
        return self._normalization_transform

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the :class:`NormalizeScene` transform for serialization.

        You can use this state dictionary to recreate the transform using :meth:`from_state_dict`.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {"name": self.name(), "version": self.version, "normalization_type": self._normalization_type}

    @staticmethod
    def name() -> str:
        """
        Return the name of the :class:`NormalizeScene` transform. **i.e.** ``"NormalizeScene"``.

        Returns:
            str: The name of the :class:`NormalizeScene` transform. **i.e.** ``"NormalizeScene"``.
        """
        return "NormalizeScene"

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "NormalizeScene":
        """
        Create a :class:`NormalizeScene` transform from a state dictionary generated with :meth:`state_dict`.

        Args:
            state_dict (dict): The state dictionary for the transform.

        Returns:
            transform (NormalizeScene): An instance of the :class:`NormalizeScene` transform.
        """
        if state_dict["name"] != "NormalizeScene":
            raise ValueError(f"Expected state_dict with name 'NormalizeScene', got {state_dict['name']} instead.")
        if "normalization_type" not in state_dict:
            raise ValueError("State dictionary must contain 'normalization_type' key.")

        normalization_type = state_dict["normalization_type"]
        return NormalizeScene(normalization_type)
