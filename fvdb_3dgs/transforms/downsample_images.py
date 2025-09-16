# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
from typing import Any, Literal

import cv2
import tqdm

from ..sfm_scene import SfmCache, SfmImageMetadata, SfmScene
from .base_transform import BaseTransform, transform


@transform
class DownsampleImages(BaseTransform):
    """
    A transform which dowsamples an image by a given factor, caching the results.

    You can specify the cached downsampled image type (e.g., "jpg" or "png"),
    the rescale sampling mode (e.g., `cv2.INTER_AREA`), and the rescaled JPEG quality.
    The downsampled images are saved in the cache with a prefix that includes the downsample factor,
    image type, sampling mode, and quality.
    If the downsampled images already exist in the cache, they will be reused instead of being recomputed.
    """

    version = "1.0.0"

    def __init__(
        self,
        image_downsample_factor: int,
        image_type: Literal["jpg", "png"] = "jpg",
        rescale_sampling_mode: int = cv2.INTER_AREA,
        rescaled_jpeg_quality: int = 100,
    ):
        """
        Create a new DownsampleImages transform instance with the specified downsampling factor
        and image caching parameters (image type, downsampling mode, and quality).

        Args:
            image_downsample_factor (int): The factor by which to downsample the images.
            image_type (str): The type of the cached downsampled images, either "jpg" or "png".
            rescale_sampling_mode (int): The OpenCV interpolation method to use for rescaling images.
            rescaled_jpeg_quality (int): The quality of the JPEG images when saving them to the cache (1-100).
        """
        super().__init__()
        self._image_downsample_factor = image_downsample_factor
        self._image_type = image_type
        self._rescale_sampling_mode = rescale_sampling_mode
        self._rescaled_jpeg_quality = rescaled_jpeg_quality
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Perform the downsampling transform on the input scene and cache.

        Args:
            input_scene (SfmScene): The input scene containing images to be downsampled.

        Returns:
            output_scene (SfmScene): A new SfmScene with paths to downsampled images.
        """
        if self._image_downsample_factor == 1:
            self._logger.info("Image downsample factor is 1, skipping downsampling.")
            return input_scene

        if len(input_scene.images) == 0:
            self._logger.warning("No images found in the SfmScene. Returning the input scene unchanged.")
            return input_scene
        if len(input_scene.cameras) == 0:
            self._logger.warning("No cameras found in the SfmScene. Returning the input scene unchanged.")
            return input_scene

        input_cache: SfmCache = input_scene.cache
        cache_prefix = f"downsampled_{self._image_downsample_factor}x_{self._image_type}_q{self._rescaled_jpeg_quality}_m{self._rescale_sampling_mode}"
        output_cache = input_cache.make_folder(
            cache_prefix, description=f"Rescaled images by a factor of {self._image_downsample_factor}"
        )

        new_camera_metadata = {}
        for cam_id, cam_meta in input_scene.cameras.items():
            rescaled_cam_w = int(cam_meta.width / self._image_downsample_factor)
            rescaled_cam_h = int(cam_meta.height / self._image_downsample_factor)
            new_camera_metadata[cam_id] = cam_meta.resize(rescaled_cam_w, rescaled_cam_h)

        self._logger.info(
            f"Rescaling images using downsample factor {self._image_downsample_factor}, "
            f"sampling mode {self._rescale_sampling_mode}, and quality {self._rescaled_jpeg_quality}."
        )

        self._logger.info(f"Attempting to load downsampled images from cache.")
        # How many zeros to pad the image index in the mask file names
        num_zeropad = len(str(len(input_scene.images))) + 2

        new_image_metadata = []

        regenerate_cache = False

        if output_cache.num_files != input_scene.num_images:
            if output_cache.num_files == 0:
                self._logger.info(f"No downsampled images found in the cache.")
            else:
                self._logger.info(
                    f"Inconsistent number of downsampled images in the cache. "
                    f"Expected {input_scene.num_images}, found {output_cache.num_files}. "
                    f"Clearing cache and regenerating downsampled images."
                )
            output_cache.clear_current_folder()
            regenerate_cache = True

        for image_id in range(input_scene.num_images):
            if regenerate_cache:
                break
            cache_image_filename = f"image_{image_id:0{num_zeropad}}"
            image_meta = input_scene.images[image_id]
            if not output_cache.has_file(cache_image_filename):
                self._logger.info(
                    f"Image {cache_image_filename} not found in the cache. " f"Clearing cache and regenerating."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

            cache_file_meta = output_cache.get_file_metadata(cache_image_filename)
            value_meta = cache_file_meta["metadata"]
            value_quality = value_meta.get("quality", -1)
            value_mode = value_meta.get("downsample_mode", -1)

            if (
                cache_file_meta.get("data_type", "") != self._image_type
                or value_quality != self._rescaled_jpeg_quality
                or value_mode != self._rescale_sampling_mode
            ):
                self._logger.info(
                    f"Output cache image metadata does not match expected format. "
                    f"Clearing the cache and regenerating downsampled images."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

            new_image_metadata.append(
                SfmImageMetadata(
                    world_to_camera_matrix=image_meta.world_to_camera_matrix,
                    camera_to_world_matrix=image_meta.camera_to_world_matrix,
                    camera_metadata=new_camera_metadata[image_meta.camera_id],
                    camera_id=image_meta.camera_id,
                    image_path=str(cache_file_meta["path"]),
                    mask_path=image_meta.mask_path,
                    point_indices=image_meta.point_indices,
                    image_id=image_meta.image_id,
                )
            )

        if regenerate_cache:
            new_image_metadata = []
            self._logger.info(
                f"Generating images downsampled by a factor of {self._image_downsample_factor} and saving to cache."
            )
            pbar = tqdm.tqdm(input_scene.images, unit="imgs")
            for _, image_meta in enumerate(pbar):
                image_filename = pathlib.Path(image_meta.image_path).name
                full_res_image_path = image_meta.image_path
                full_res_img = cv2.imread(full_res_image_path)
                assert full_res_img is not None, f"Failed to load image {full_res_image_path}"
                img_h, img_w = full_res_img.shape[:2]
                rescaled_img_h = int(img_h / self._image_downsample_factor)
                rescaled_img_w = int(img_w / self._image_downsample_factor)
                assert rescaled_img_w == new_camera_metadata[image_meta.camera_id].width, "Got mismatched widths!"
                assert rescaled_img_h == new_camera_metadata[image_meta.camera_id].height, "Got mismatched heights!"
                pbar.set_description(
                    f"Rescaling {image_filename} from {img_w} x {img_h} to {rescaled_img_w} x {rescaled_img_h}"
                )
                rescaled_image = cv2.resize(
                    full_res_img, (rescaled_img_w, rescaled_img_h), interpolation=self._rescale_sampling_mode
                )
                assert (
                    rescaled_image.shape[0] == rescaled_img_h and rescaled_image.shape[1] == rescaled_img_w
                ), f"Rescaled image {image_filename} has shape {rescaled_image.shape} but expected {rescaled_img_h, rescaled_img_w}"
                # Save the rescaled image to the cache
                cache_image_filename = f"image_{image_meta.image_id:0{num_zeropad}}"
                cache_file_meta = output_cache.write_file(
                    name=cache_image_filename,
                    data=rescaled_image,
                    data_type=self._image_type,
                    quality=self._rescaled_jpeg_quality,
                    metadata={
                        "quality": self._rescaled_jpeg_quality,
                        "downsample_mode": self._rescale_sampling_mode,
                    },
                )
                new_image_metadata.append(
                    SfmImageMetadata(
                        world_to_camera_matrix=image_meta.world_to_camera_matrix,
                        camera_to_world_matrix=image_meta.camera_to_world_matrix,
                        camera_metadata=new_camera_metadata[image_meta.camera_id],
                        camera_id=image_meta.camera_id,
                        image_path=str(cache_file_meta["path"]),
                        mask_path=image_meta.mask_path,
                        point_indices=image_meta.point_indices,
                        image_id=image_meta.image_id,
                    )
                )

            pbar.close()

            self._logger.info(
                f"Rescaled {input_scene.num_images} images by a factor of {self._image_downsample_factor} "
                f"and saved to cache with sampling mode {self._rescale_sampling_mode} and quality "
                f"{self._rescaled_jpeg_quality}."
            )

        output_scene = SfmScene(
            cameras=new_camera_metadata,
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
        """
        Return the name of the DownsampleImages transform.

        Returns:
            str: The name of the DownsampleImages transform.
        """
        return "DownsampleImages"

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the DownsampleImages transform for serialization.
        Returns:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.
        """
        return {
            "name": self.name(),
            "version": self.version,
            "image_downsample_factor": self._image_downsample_factor,
            "image_type": self._image_type,
            "rescale_sampling_mode": self._rescale_sampling_mode,
            "rescaled_jpeg_quality": self._rescaled_jpeg_quality,
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "DownsampleImages":
        """
        Create a DownsampleImages transform from a state dictionary.

        Args:
            state_dict (dict[str, Any]): A dictionary containing information to serialize/deserialize the transform.

        Returns:
            DownsampleImages: An instance of the DownsampleImages transform.
        """
        if state_dict["name"] != "DownsampleImages":
            raise ValueError(f"Expected state_dict with name 'DownsampleImages', got {state_dict['name']} instead.")

        return DownsampleImages(
            image_downsample_factor=state_dict["image_downsample_factor"],
            image_type=state_dict["image_type"],
            rescale_sampling_mode=state_dict["rescale_sampling_mode"],
            rescaled_jpeg_quality=state_dict["rescaled_jpeg_quality"],
        )
