# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any, Literal

import numpy as np
import pyproj

from ..sfm_scene import SfmScene
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


@transform
class NormalizeScene(BaseTransform):
    """
    A transform which normalizes an SfmScene using a variety of approaches:
    - "pca": Normalizes the scene using PCA, centering and rotating the point cloud to align with principle axes.
    - "ecef2enu": Converts ECEF coordinates to ENU coordinates, centering the scene around the median point.
    - "similarity": Applies a similarity transformation to the scene based on camera positions, centering and scaling it.
    - "none": No normalization is applied, the scene remains unchanged.
    """

    version = "1.0.0"

    valid_normalization_types = ["pca", "ecef2enu", "similarity", "none"]

    def __init__(self, normalization_type: Literal["pca", "none", "ecef2enu", "similarity"]):
        """
        Initialize the NormalizeScene transform.

        Args:
            normalization_type (str): The type of normalization to apply.
                Options are "pca", "none", "ecef2enu", or "similarity".
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
        Normalize the SfmScene using the specified normalization type.

        Args:
            input_scene (SfmScene): Input SfmScene object containing camera and point data

        Returns:
            output_scene (SfmScene): A new SfmScene after applying the normalization transform.
        """
        self._logger.info(f"Normalizing SfmScene with normalization type: {self._normalization_type}")

        normalization_transform = self._compute_normalization_transform(input_scene)
        if normalization_transform is None:
            self._logger.warning("Returning the input scene unchanged.")
            return input_scene

        return input_scene.apply_transformation_matrix(normalization_transform)

    def _compute_normalization_transform(self, input_scene: SfmScene) -> np.ndarray | None:
        """
        Compute the normalization transform for the scene.

        Args:
            input_scene (SfmScene): The input scene to normalize.

        Returns:
            np.ndarray | None: The normalization transform, or None if the scene lacks points or camera matrices.
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

    def transform_camera_poses_to_scene_normalized_space(
        self, input_scene: SfmScene, camera_to_world_matrices: np.ndarray
    ) -> np.ndarray:
        """
        Transform points to the scene normalized space.
        """
        normalization_transform = self._compute_normalization_transform(input_scene)

        if normalization_transform is None:
            self._logger.warning("Returning the input poses unchanged.")
            return camera_to_world_matrices
        assert len(camera_to_world_matrices.shape) == 3 and camera_to_world_matrices.shape[1:] == (4, 4)

        new_camera_to_world_matrix = np.einsum("nij, ki -> nkj", camera_to_world_matrices, normalization_transform)
        scaling = np.linalg.norm(new_camera_to_world_matrix[:, 0, :3], axis=1)
        new_camera_to_world_matrix[:, :3, :3] = new_camera_to_world_matrix[:, :3, :3] / scaling[:, None, None]

        return new_camera_to_world_matrix

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the NormalizeScene transform for serialization.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {"name": self.name(), "version": self.version, "normalization_type": self._normalization_type}

    @staticmethod
    def name() -> str:
        """
        Return the name of the NormalizeScene transform.

        Returns:
            str: The name of the NormalizeScene transform.
        """
        return "NormalizeScene"

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "NormalizeScene":
        """
        Create a NormalizeScene transform from a state dictionary.

        Args:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.

        Returns:
            NormalizeScene: An instance of the NormalizeScene transform.
        """
        if state_dict["name"] != "NormalizeScene":
            raise ValueError(f"Expected state_dict with name 'NormalizeScene', got {state_dict['name']} instead.")
        if "normalization_type" not in state_dict:
            raise ValueError("State dictionary must contain 'normalization_type' key.")

        normalization_type = state_dict["normalization_type"]
        return NormalizeScene(normalization_type)
