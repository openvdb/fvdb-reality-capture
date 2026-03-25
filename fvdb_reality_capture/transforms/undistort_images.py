# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import os
import pathlib
from typing import Any, Literal

import cv2
import numpy as np
import tqdm
from fvdb import CameraModel

from fvdb_reality_capture.sfm_scene import SfmCache, SfmCameraMetadata, SfmPosedImageMetadata, SfmScene

from .base_transform import BaseTransform, transform


@transform
class UndistortImages(BaseTransform):
    """
    Materialize undistorted images and masks for an :class:`SfmScene` and return a new scene
    whose cameras describe that undistorted pixel space directly.
    """

    version = "1.0.0"

    def __init__(
        self,
        image_type: Literal["jpg", "png"] = "png",
        jpeg_quality: int = 98,
        alpha: float = 0.0,
        remap_interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST_EXACT,
    ):
        self._image_type = image_type
        self._jpeg_quality = jpeg_quality
        self._alpha = alpha
        self._remap_interpolation = remap_interpolation
        self._mask_interpolation = mask_interpolation
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    @staticmethod
    def _packed_distortion_coeffs_to_opencv_radtan5(distortion_coeffs: np.ndarray) -> np.ndarray:
        """
        Convert packed FVDB distortion coefficients into OpenCV's 5-coefficient radtan layout.

        This is used only by the undistortion transform, which relies on OpenCV APIs expecting
        coefficients ordered as ``[k1, k2, p1, p2, k3]``.

        Args:
            distortion_coeffs: Distortion coefficients in packed FVDB order
                ``[k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]``.

        Returns:
            np.ndarray: Distortion coefficients in OpenCV radtan-5 order ``[k1, k2, p1, p2, k3]``.
        """
        coeffs = np.zeros((5,), dtype=np.float32)
        coeffs[0] = distortion_coeffs[0]
        coeffs[1] = distortion_coeffs[1]
        coeffs[2] = distortion_coeffs[6]
        coeffs[3] = distortion_coeffs[7]
        coeffs[4] = distortion_coeffs[2]
        return coeffs

    @staticmethod
    def _undistorted_camera_metadata(
        camera_meta: SfmCameraMetadata,
        alpha: float,
    ) -> tuple[SfmCameraMetadata, np.ndarray | None, np.ndarray | None, tuple[int, int, int, int] | None]:
        """
        Build the output camera metadata and remap state for a single input camera.

        For distorted cameras this computes the OpenCV undistortion maps and ROI crop, then
        returns a pinhole camera describing the undistorted pixel space. For already-pinhole
        cameras it returns an equivalent pinhole camera and no remap state.

        Args:
            camera_meta: Input camera metadata in the source scene pixel space.
            alpha: OpenCV free-scaling parameter passed to ``getOptimalNewCameraMatrix``.

        Returns:
            tuple: ``(output_camera, map_x, map_y, roi)`` where ``output_camera`` describes the
            undistorted pixel space and the remaining values define the remap/crop operation.
        """
        if camera_meta.distortion_coeffs.size == 0:
            return (
                SfmCameraMetadata(
                    img_width=camera_meta.width,
                    img_height=camera_meta.height,
                    fx=camera_meta.fx,
                    fy=camera_meta.fy,
                    cx=camera_meta.cx,
                    cy=camera_meta.cy,
                    camera_model=CameraModel.PINHOLE,
                    distortion_coeffs=np.array([], dtype=np.float32),
                ),
                None,
                None,
                None,
            )

        if not camera_meta.can_undistort:
            raise NotImplementedError(
                f"Camera model {camera_meta.camera_model.name} does not support image undistortion"
            )

        distortion_parameters = UndistortImages._packed_distortion_coeffs_to_opencv_radtan5(
            camera_meta.distortion_coeffs
        )
        undistorted_proj_mat, undistort_roi = cv2.getOptimalNewCameraMatrix(
            camera_meta.projection_matrix,
            distortion_parameters,
            (camera_meta.width, camera_meta.height),
            alpha,
        )
        undistort_map_x, undistort_map_y = cv2.initUndistortRectifyMap(
            camera_meta.projection_matrix,
            distortion_parameters,
            None,
            undistorted_proj_mat,
            (camera_meta.width, camera_meta.height),
            cv2.CV_32FC1,  # type: ignore[arg-type]
        )

        roi_x, roi_y, roi_w, roi_h = (int(v) for v in undistort_roi)
        adjusted_proj_mat = undistorted_proj_mat.copy()
        adjusted_proj_mat[0, 2] -= roi_x
        adjusted_proj_mat[1, 2] -= roi_y
        undistorted_camera = SfmCameraMetadata(
            img_width=roi_w,
            img_height=roi_h,
            fx=float(adjusted_proj_mat[0, 0]),
            fy=float(adjusted_proj_mat[1, 1]),
            cx=float(adjusted_proj_mat[0, 2]),
            cy=float(adjusted_proj_mat[1, 2]),
            camera_model=CameraModel.PINHOLE,
            distortion_coeffs=np.array([], dtype=np.float32),
        )
        return undistorted_camera, undistort_map_x, undistort_map_y, (roi_x, roi_y, roi_w, roi_h)

    @staticmethod
    def _undistort_and_crop(
        image: np.ndarray,
        undistort_map_x: np.ndarray | None,
        undistort_map_y: np.ndarray | None,
        undistort_roi: tuple[int, int, int, int] | None,
        interpolation: int,
    ) -> np.ndarray:
        """
        Apply a precomputed OpenCV remap and ROI crop to an image-like array.

        Args:
            image: Input image or mask array in the source scene pixel space.
            undistort_map_x: OpenCV x-coordinate remap array, or ``None`` for a no-op.
            undistort_map_y: OpenCV y-coordinate remap array, or ``None`` for a no-op.
            undistort_roi: ROI crop in ``(x, y, width, height)`` format, or ``None`` for a no-op.
            interpolation: OpenCV interpolation enum to use during remapping.

        Returns:
            np.ndarray: The remapped and cropped image, or the original image if no remap state
            was provided.
        """
        if undistort_map_x is None or undistort_map_y is None or undistort_roi is None:
            return image
        if interpolation == cv2.INTER_NEAREST_EXACT:
            interpolation = cv2.INTER_NEAREST
        image_remap = cv2.remap(image, undistort_map_x, undistort_map_y, interpolation=interpolation)
        x, y, w, h = undistort_roi
        return image_remap[y : y + h, x : x + w]

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        if len(input_scene.images) == 0:
            self._logger.warning("No images found in the SfmScene. Returning the input scene unchanged.")
            return input_scene
        if len(input_scene.cameras) == 0:
            self._logger.warning("No cameras found in the SfmScene. Returning the input scene unchanged.")
            return input_scene
        if all(camera.distortion_coeffs.size == 0 for camera in input_scene.cameras.values()):
            self._logger.info("No distorted cameras found in the SfmScene. Returning the input scene unchanged.")
            return input_scene

        input_cache: SfmCache = input_scene.cache
        alpha_str = str(self._alpha).replace(" ", "_").replace(".", "_").replace("-", "neg")
        cache_prefix = (
            f"undistorted_{self._image_type}_q{self._jpeg_quality}_a{alpha_str}_"
            f"m{self._remap_interpolation}_mask{self._mask_interpolation}"
        )
        output_cache = input_cache.make_folder(cache_prefix, description="Undistorted images and masks")

        new_camera_metadata = {
            cam_id: self._undistorted_camera_metadata(camera_meta, self._alpha)
            for cam_id, camera_meta in input_scene.cameras.items()
        }

        num_zeropad = len(str(len(input_scene.images))) + 2
        num_masks = sum(
            len(str(image_meta.mask_path)) > 0 and os.path.exists(image_meta.mask_path)
            for image_meta in input_scene.images
        )

        regenerate_cache = False
        if output_cache.num_files != input_scene.num_images + num_masks:
            if output_cache.num_files == 0:
                self._logger.info("No undistorted images found in the cache.")
            else:
                self._logger.info(
                    f"Inconsistent number of undistorted files in the cache. Expected {input_scene.num_images + num_masks}, "
                    f"found {output_cache.num_files}. Clearing cache and regenerating undistorted images."
                )
            output_cache.clear_current_folder()
            regenerate_cache = True

        new_image_metadata: list[SfmPosedImageMetadata] = []
        for image_id in range(input_scene.num_images):
            if regenerate_cache:
                break

            image_meta = input_scene.images[image_id]
            cache_image_filename = f"image_{image_meta.image_id:0{num_zeropad}}"
            if not output_cache.has_file(cache_image_filename):
                self._logger.info(
                    f"Image {cache_image_filename} not found in the cache. Clearing cache and regenerating."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

            cache_file_meta = output_cache.get_file_metadata(cache_image_filename)
            value_meta = cache_file_meta["metadata"]
            value_quality = value_meta.get("quality", -1)
            value_mode = value_meta.get("remap_interpolation", -1)
            value_alpha = value_meta.get("alpha", None)
            if (
                cache_file_meta.get("data_type", "") != self._image_type
                or value_quality != self._jpeg_quality
                or value_mode != self._remap_interpolation
                or value_alpha != self._alpha
            ):
                self._logger.info(
                    "Output cache image metadata does not match expected format. Clearing the cache and regenerating."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

            mask_path = ""
            if len(str(image_meta.mask_path)) > 0 and os.path.exists(image_meta.mask_path):
                cache_mask_filename = f"mask_{image_meta.image_id:0{num_zeropad}}"
                if not output_cache.has_file(cache_mask_filename):
                    self._logger.info(
                        f"Mask {cache_mask_filename} not found in the cache. Clearing cache and regenerating."
                    )
                    output_cache.clear_current_folder()
                    regenerate_cache = True
                    break
                mask_file_meta = output_cache.get_file_metadata(cache_mask_filename)
                mask_path = str(mask_file_meta["path"])

            new_image_metadata.append(
                SfmPosedImageMetadata(
                    world_to_camera_matrix=image_meta.world_to_camera_matrix,
                    camera_to_world_matrix=image_meta.camera_to_world_matrix,
                    camera_metadata=new_camera_metadata[image_meta.camera_id][0],
                    camera_id=image_meta.camera_id,
                    image_path=str(cache_file_meta["path"]),
                    mask_path=mask_path,
                    point_indices=image_meta.point_indices,
                    image_id=image_meta.image_id,
                )
            )

        if regenerate_cache:
            new_image_metadata = []
            pbar = tqdm.tqdm(input_scene.images, unit="imgs", desc="Undistorting images")
            for image_meta in pbar:
                image_filename = pathlib.Path(image_meta.image_path).name
                image = cv2.imread(image_meta.image_path, cv2.IMREAD_UNCHANGED)
                assert image is not None, f"Failed to load image {image_meta.image_path}"
                pbar.set_description(f"Undistorting {image_filename}")

                undistorted_camera, undistort_map_x, undistort_map_y, undistort_roi = new_camera_metadata[
                    image_meta.camera_id
                ]
                undistorted_image = self._undistort_and_crop(
                    image, undistort_map_x, undistort_map_y, undistort_roi, self._remap_interpolation
                )
                cache_image_filename = f"image_{image_meta.image_id:0{num_zeropad}}"
                cache_file_meta = output_cache.write_file(
                    name=cache_image_filename,
                    data=undistorted_image,
                    data_type=self._image_type,
                    quality=self._jpeg_quality,
                    metadata={
                        "quality": self._jpeg_quality,
                        "remap_interpolation": self._remap_interpolation,
                        "alpha": self._alpha,
                    },
                )

                mask_path = ""
                if len(str(image_meta.mask_path)) > 0 and os.path.exists(image_meta.mask_path):
                    mask = cv2.imread(image_meta.mask_path, cv2.IMREAD_UNCHANGED)
                    assert mask is not None, f"Failed to load mask {image_meta.mask_path}"
                    undistorted_mask = self._undistort_and_crop(
                        mask, undistort_map_x, undistort_map_y, undistort_roi, self._mask_interpolation
                    )
                    cache_mask_filename = f"mask_{image_meta.image_id:0{num_zeropad}}"
                    cache_mask_meta = output_cache.write_file(
                        name=cache_mask_filename,
                        data=undistorted_mask,
                        data_type="png",
                        metadata={"remap_interpolation": self._mask_interpolation, "alpha": self._alpha},
                    )
                    mask_path = str(cache_mask_meta["path"])

                new_image_metadata.append(
                    SfmPosedImageMetadata(
                        world_to_camera_matrix=image_meta.world_to_camera_matrix,
                        camera_to_world_matrix=image_meta.camera_to_world_matrix,
                        camera_metadata=undistorted_camera,
                        camera_id=image_meta.camera_id,
                        image_path=str(cache_file_meta["path"]),
                        mask_path=mask_path,
                        point_indices=image_meta.point_indices,
                        image_id=image_meta.image_id,
                    )
                )
            pbar.close()

        output_scene = SfmScene(
            cameras={cam_id: value[0] for cam_id, value in new_camera_metadata.items()},
            images=new_image_metadata,
            points=input_scene.points,
            points_err=input_scene.points_err,
            points_rgb=input_scene.points_rgb,
            scene_bbox=input_scene.scene_bbox,
            transformation_matrix=input_scene.transformation_matrix,
            cache=output_cache,
        )
        return output_scene

    @staticmethod
    def name() -> str:
        return "UndistortImages"

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
            "version": self.version,
            "image_type": self._image_type,
            "jpeg_quality": self._jpeg_quality,
            "alpha": self._alpha,
            "remap_interpolation": self._remap_interpolation,
            "mask_interpolation": self._mask_interpolation,
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "UndistortImages":
        if state_dict["name"] != "UndistortImages":
            raise ValueError(f"Expected state_dict with name 'UndistortImages', got {state_dict['name']} instead.")
        return UndistortImages(
            image_type=state_dict["image_type"],
            jpeg_quality=state_dict["jpeg_quality"],
            alpha=state_dict["alpha"],
            remap_interpolation=state_dict["remap_interpolation"],
            mask_interpolation=state_dict["mask_interpolation"],
        )
