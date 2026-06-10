# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
from abc import ABC, abstractmethod

import numpy as np
import pycolmap
from fvdb import CameraModel

from .sfm_metadata import SfmCameraMetadata


class Adapter(ABC):
    """
    Base class for translating source-specific scene conventions into FVDB scene conventions.
    """

    @abstractmethod
    def camera_metadata(self, camera_id: int) -> SfmCameraMetadata:
        """
        Return camera metadata for a source camera ID.
        """
        raise NotImplementedError

    @abstractmethod
    def image_camera_id(self, image_id: int) -> int:
        """
        Return the source camera ID for a source image ID.
        """
        raise NotImplementedError

    @abstractmethod
    def image_name(self, image_id: int) -> str:
        """
        Return the source image file name for a source image ID.
        """
        raise NotImplementedError

    @abstractmethod
    def world_to_camera_matrix(self, image_id: int) -> np.ndarray:
        """
        Return a 4x4 world-to-camera matrix for a source image ID.
        """
        raise NotImplementedError

    @abstractmethod
    def registered_image_ids(self) -> list[int]:
        """
        Return source image IDs that should be loaded into the scene.
        """
        raise NotImplementedError

    @abstractmethod
    def points_from_scene(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int], dict[int, np.ndarray]]:
        """
        Return source scene points and visibility data in the layout expected by the scene loader.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def visibility_cache_loader(self) -> str:
        """
        Return the loader identifier stored in visible-points cache metadata.
        """
        raise NotImplementedError


class COLMAPAdapter(Adapter):
    """
    Translate pycolmap conventions into FVDB scene conventions.
    """

    CAMERA_MODEL_NAMES = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE",
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
        5: "OPENCV_FISHEYE",
    }
    VISIBILITY_CACHE_LOADER = "pycolmap"

    def __init__(self, colmap_path: pathlib.Path):
        self._reconstruction = self._load_reconstruction(colmap_path)

    @staticmethod
    def _load_reconstruction(colmap_path: pathlib.Path) -> pycolmap.Reconstruction:
        """
        Load raw COLMAP data into a pycolmap reconstruction.
        """
        if not colmap_path.exists():
            raise FileNotFoundError(f"COLMAP directory {colmap_path} does not exist.")

        colmap_sparse_path = colmap_path / "sparse" / "0"
        if not colmap_sparse_path.exists():
            colmap_sparse_path = colmap_path / "sparse"
        if not colmap_sparse_path.exists():
            raise FileNotFoundError(f"COLMAP directory {colmap_sparse_path} does not exist.")

        return pycolmap.Reconstruction(colmap_sparse_path)

    def _camera(self, camera_id: int) -> pycolmap.Camera:
        return self._reconstruction.cameras[camera_id]

    def _image(self, image_id: int) -> pycolmap.Image:
        return self._reconstruction.images[image_id]

    def camera_metadata(self, camera_id: int) -> SfmCameraMetadata:
        source_camera = self._camera(camera_id)
        camera_model, distortion_coeffs = self.camera_model_and_distortion_coeffs(source_camera)
        fx, fy, cx, cy = self.camera_intrinsics(source_camera)
        img_width, img_height = self.camera_image_size(source_camera)
        return SfmCameraMetadata(
            img_width=img_width,
            img_height=img_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_model=camera_model,
            distortion_coeffs=distortion_coeffs,
        )

    def image_camera_id(self, image_id: int) -> int:
        return int(self._image(image_id).camera_id)

    def image_name(self, image_id: int) -> str:
        return str(self._image(image_id).name)

    @classmethod
    def camera_model_name(cls, cam: pycolmap.Camera) -> str:
        model_name = getattr(cam, "model_name", None)
        if model_name:
            return str(model_name)

        source_model = cam.model
        if isinstance(source_model, str):
            return source_model

        model_name = getattr(source_model, "name", None)
        if model_name is not None:
            return str(model_name)

        model_value = getattr(source_model, "value", source_model)
        if isinstance(model_value, np.integer):
            model_value = int(model_value)

        if isinstance(model_value, int) and model_value in cls.CAMERA_MODEL_NAMES:
            return cls.CAMERA_MODEL_NAMES[model_value]

        raise ValueError(f"Unknown COLMAP camera model {source_model}")

    @classmethod
    def camera_image_size(cls, source_camera: pycolmap.Camera) -> tuple[int, int]:
        return int(source_camera.width), int(source_camera.height)

    @classmethod
    def camera_intrinsics(cls, source_camera: pycolmap.Camera) -> tuple[float, float, float, float]:
        params = np.asarray(source_camera.params, dtype=np.float64)
        camera_model = cls.camera_model_name(source_camera)

        if camera_model == "SIMPLE_PINHOLE":
            fx, cx, cy = params[:3]
            return float(fx), float(fx), float(cx), float(cy)
        if camera_model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
            return float(fx), float(fy), float(cx), float(cy)
        if camera_model == "SIMPLE_RADIAL":
            fx, cx, cy = params[:3]
            return float(fx), float(fx), float(cx), float(cy)
        if camera_model == "RADIAL":
            fx, cx, cy = params[:3]
            return float(fx), float(fx), float(cx), float(cy)
        if camera_model in ("OPENCV", "OPENCV_FISHEYE"):
            fx, fy, cx, cy = params[:4]
            return float(fx), float(fy), float(cx), float(cy)

        raise ValueError(f"Unsupported COLMAP camera model {camera_model}")

    @classmethod
    def camera_model_and_distortion_coeffs(cls, source_camera: pycolmap.Camera) -> tuple[CameraModel, np.ndarray]:
        """
        Convert a COLMAP camera into the canonical FVDB camera model and packed distortion coefficients.

        Args:
            source_camera (pycolmap.Camera): The COLMAP camera object.

        Returns:
            tuple[CameraModel, np.ndarray]: The canonical camera model and distortion coefficients in
                FVDB packed layout ``[k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]``.
        """
        camera_model = cls.camera_model_name(source_camera)
        params = np.asarray(source_camera.params, dtype=np.float64)
        coeffs = np.zeros((12,), dtype=np.float32)

        if camera_model == "SIMPLE_PINHOLE":
            return CameraModel.PINHOLE, np.empty((0,), dtype=np.float32)
        if camera_model == "PINHOLE":
            return CameraModel.PINHOLE, np.empty((0,), dtype=np.float32)
        if camera_model == "SIMPLE_RADIAL":
            coeffs[0] = params[3]
            return CameraModel.OPENCV_RADTAN_5, coeffs
        if camera_model == "RADIAL":
            coeffs[0] = params[3]
            coeffs[1] = params[4]
            return CameraModel.OPENCV_RADTAN_5, coeffs
        if camera_model == "OPENCV":
            coeffs[0] = params[4]
            coeffs[1] = params[5]
            coeffs[6] = params[6]
            coeffs[7] = params[7]
            return CameraModel.OPENCV_RADTAN_5, coeffs
        if camera_model == "OPENCV_FISHEYE":
            raise ValueError("COLMAP OPENCV_FISHEYE cameras are not supported by fvdb.CameraModel")
        raise ValueError(f"Unsupported COLMAP camera model {camera_model}")

    def world_to_camera_matrix(self, image_id: int) -> np.ndarray:
        source_image = self._image(image_id)
        world_to_camera = np.asarray(source_image.cam_from_world().matrix(), dtype=np.float64)
        return np.vstack([world_to_camera, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)])

    def registered_image_ids(self) -> list[int]:
        try:
            return [int(image_id) for image_id in self._reconstruction.reg_image_ids()]
        except AttributeError:
            return [int(image_id) for image_id, image in self._reconstruction.images.items() if image.has_pose]

    def points_from_scene(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int], dict[int, np.ndarray]]:
        point3D_items = sorted(
            (int(point3D_id), point3D) for point3D_id, point3D in self._reconstruction.points3D.items()
        )
        num_points = len(point3D_items)

        point3D_ids = np.empty(num_points, dtype=np.uint64)
        points3D = np.empty((num_points, 3), dtype=np.float64)
        point3D_colors = np.empty((num_points, 3), dtype=np.uint8)
        point3D_errors = np.empty(num_points, dtype=np.float64)
        point3D_id_to_point3D_idx: dict[int, int] = {}
        point3D_id_to_images: dict[int, np.ndarray] = {}

        for point3D_idx, (point3D_id, point3D) in enumerate(point3D_items):
            point3D_ids[point3D_idx] = point3D_id
            points3D[point3D_idx] = np.asarray(point3D.xyz, dtype=np.float64)
            point3D_colors[point3D_idx] = np.asarray(point3D.color, dtype=np.uint8)
            point3D_errors[point3D_idx] = float(point3D.error)
            point3D_id_to_point3D_idx[point3D_id] = point3D_idx
            point3D_id_to_images[point3D_id] = np.array(
                [(int(track_el.image_id), int(track_el.point2D_idx)) for track_el in point3D.track.elements],
                dtype=np.uint32,
            ).reshape(-1, 2)

        return (
            points3D,
            point3D_ids,
            point3D_colors,
            point3D_errors,
            point3D_id_to_point3D_idx,
            point3D_id_to_images,
        )

    @property
    def visibility_cache_loader(self) -> str:
        return self.VISIBILITY_CACHE_LOADER
