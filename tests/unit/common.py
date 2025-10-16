# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from fvdb_reality_capture.sfm_scene import SfmCameraMetadata, SfmImageMetadata, SfmScene


def remove_point_indices_from_scene(scene: SfmScene) -> SfmScene:
    """
    Returns a copy of the scene with point indices removed from all images.
    """
    images_no_points = []
    for img_meta in scene.images:
        img_meta_no_points = SfmImageMetadata(
            world_to_camera_matrix=img_meta.world_to_camera_matrix,
            camera_to_world_matrix=img_meta.camera_to_world_matrix,
            camera_id=img_meta.camera_id,
            camera_metadata=img_meta.camera_metadata,
            image_path=img_meta.image_path,
            mask_path=img_meta.mask_path,
            point_indices=None,  # Remove point indices
            image_id=img_meta.image_id,
        )
        images_no_points.append(img_meta_no_points)
    ret = SfmScene(
        cameras=scene.cameras,
        images=images_no_points,
        points=scene.points,
        points_err=scene.points_err,
        points_rgb=scene.points_rgb,
        scene_bbox=scene.scene_bbox,
        transformation_matrix=scene.transformation_matrix,
        cache=scene.cache,
    )
    return ret


def sfm_camera_metadata_match(cam1: SfmCameraMetadata, cam2: SfmCameraMetadata) -> bool:
    """
    Return True if the two camera metadata objects are equivalent, False otherwise.

    Args:
        cam1 (SfmCameraMetadata): The first camera metadata object.
        cam2 (SfmCameraMetadata): The second camera metadata object.

    Returns:
        bool: True if the two camera metadata objects are equivalent, False otherwise.
    """
    if not np.allclose(cam1.projection_matrix, cam2.projection_matrix):
        return False
    if not np.allclose(cam1.fx, cam2.fx):
        return False
    if not np.allclose(cam1.fy, cam2.fy):
        return False
    if not np.allclose(cam1.cx, cam2.cx):
        return False
    if not np.allclose(cam1.cy, cam2.cy):
        return False
    if not np.allclose(cam1.fovx, cam2.fovx):
        return False
    if not np.allclose(cam1.fovy, cam2.fovy):
        return False
    if not cam1.width == cam2.width:
        return False
    if not cam1.height == cam2.height:
        return False
    if not cam1.camera_type == cam2.camera_type:
        return False
    if not np.allclose(cam1.aspect, cam2.aspect):
        return False
    if not np.allclose(cam1.distortion_parameters, cam2.distortion_parameters):
        return False
    if cam1.undistort_roi != cam2.undistort_roi:
        return False

    has_undistort_map_x1 = cam1.undistort_map_x is not None
    has_undistort_map_x2 = cam2.undistort_map_x is not None
    if has_undistort_map_x1 != has_undistort_map_x2:
        return False
    if has_undistort_map_x1 and has_undistort_map_x2:
        assert cam1.undistort_map_x is not None
        assert cam2.undistort_map_x is not None
        if not np.allclose(cam1.undistort_map_x, cam2.undistort_map_x):
            return False
    has_undistort_map_y1 = cam1.undistort_map_y is not None
    has_undistort_map_y2 = cam2.undistort_map_y is not None
    if has_undistort_map_y1 != has_undistort_map_y2:
        return False
    if has_undistort_map_y1 and has_undistort_map_y2:
        assert cam1.undistort_map_y is not None
        assert cam2.undistort_map_y is not None
        if not np.allclose(cam1.undistort_map_y, cam2.undistort_map_y):
            return False

    return True


def sfm_image_metadata_match(im1: SfmImageMetadata, im2: SfmImageMetadata) -> bool:
    """
    Return True if the two image metadata objects are equivalent, False otherwise.

    Args:
        im1 (SfmImageMetadata): The first image metadata object.
        im2 (SfmImageMetadata): The second image metadata object.

    Returns:
        bool: True if the two image metadata objects are equivalent, False otherwise.
    """
    if im1.image_id != im2.image_id:
        return False
    if im1.image_path != im2.image_path:
        return False
    if im1.mask_path != im2.mask_path:
        return False
    if not np.allclose(im1.camera_to_world_matrix, im2.camera_to_world_matrix):
        return False
    if not np.allclose(im1.world_to_camera_matrix, im2.world_to_camera_matrix):
        return False
    if im1.image_size != im2.image_size:
        return False
    im1_has_points = im1.point_indices is not None
    im2_has_points = im2.point_indices is not None
    if im1_has_points != im2_has_points:
        return False
    if im1_has_points and im2_has_points:
        assert im1.point_indices is not None
        assert im2.point_indices is not None
        if not np.array_equal(im1.point_indices, im2.point_indices):
            return False
    if im1.camera_id != im2.camera_id:
        return False
    return sfm_camera_metadata_match(im1.camera_metadata, im2.camera_metadata)


def sfm_scenes_match(scene1: SfmScene, scene2: SfmScene) -> bool:
    """
    Return True if the two SfmScene objects are equivalent, False otherwise.

    Args:
        scene1 (SfmScene): The first SfmScene object.
        scene2 (SfmScene): The second SfmScene object.

    Returns:
        bool: True if the two SfmScene objects are equivalent, False otherwise.
    """
    if not np.allclose(scene1.transformation_matrix, scene2.transformation_matrix):
        return False
    if scene1.num_images != scene2.num_images:
        return False
    if scene1.num_cameras != scene2.num_cameras:
        return False
    if not np.allclose(scene1.projection_matrices, scene2.projection_matrices):
        return False
    if not np.allclose(scene1.image_centers, scene2.image_centers):
        return False
    if not np.allclose(scene1.image_sizes, scene2.image_sizes):
        return False
    if len(scene1.images) != len(scene2.images):
        return False
    if not np.allclose(scene1.world_to_camera_matrices, scene2.world_to_camera_matrices):
        return False
    if not np.allclose(scene1.camera_to_world_matrices, scene2.camera_to_world_matrices):
        return False
    if len(scene1.cameras) != len(scene2.cameras):
        return False
    if not np.allclose(scene1.points, scene2.points):
        return False
    if not np.allclose(scene1.points_rgb, scene2.points_rgb):
        return False
    if not np.allclose(scene1.points_err, scene2.points_err):
        return False
    if not np.allclose(scene1.scene_bbox, scene2.scene_bbox):
        return False

    for im1, im2 in zip(scene1.images, scene2.images):
        if not sfm_image_metadata_match(im1, im2):
            return False
    for k, v in scene1.cameras.items():
        if k not in scene2.cameras:
            return False
        if not sfm_camera_metadata_match(v, scene2.cameras[k]):
            return False
    return True
