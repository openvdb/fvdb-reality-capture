# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time
from typing import Literal

import cv2
import numpy as np
import tyro

import fvdb_3dgs
from fvdb_3dgs.transforms import (
    Compose,
    DownsampleImages,
    FilterImagesWithLowPoints,
    NormalizeScene,
    PercentileFilterPoints,
)
from fvdb_3dgs.viewer import Viewer


def center_and_scale_scene(sfm_scene: fvdb_3dgs.SfmScene, scale: float) -> fvdb_3dgs.SfmScene:
    centroid = np.median(sfm_scene.points, axis=0)
    transform = np.diag([scale, scale, scale, 1.0])
    transform[0:3, 3] = -centroid * scale
    return sfm_scene.apply_transformation_matrix(transform)


def main(
    dataset_path: pathlib.Path,
    viewer_port: int = 8080,
    verbose: bool = False,
    image_downsample_factor: int = 8,
    show_images: bool = True,
    points_percentile_filter: float = 0.0,
    dataset_type: Literal["colmap", "e57"] = "colmap",
):
    """
    Visualize a scene in a saved checkpoint file.

    Args:
        ply_path (pathlib.Path): Path to a PLY file containing the Gaussian splat model.
        viewer_port (int): The port to expose the viewer server on
        verbose (bool): If True, then the viewer will log verbosely.
        device (str | torch.device): Device to use for computation (default is "cuda").
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    logger = logging.getLogger(__name__)

    viewer = Viewer(port=viewer_port, verbose=verbose)

    if dataset_type == "colmap":
        sfm_scene: fvdb_3dgs.SfmScene = fvdb_3dgs.SfmScene.from_colmap(dataset_path)
    elif dataset_type == "e57":
        sfm_scene = fvdb_3dgs.SfmScene.from_e57(dataset_path, point_downsample_factor=20)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    sfm_scene = Compose(
        DownsampleImages(image_downsample_factor),
        NormalizeScene("pca"),
        PercentileFilterPoints([points_percentile_filter] * 3, [100.0 - points_percentile_filter] * 3),
        FilterImagesWithLowPoints(min_num_points=5),
    )(sfm_scene)

    scene_extent = sfm_scene.points.max(0) - sfm_scene.points.min(0)
    axis_scale = 0.01 * float(np.linalg.norm(scene_extent))
    logger.info(f"Scene extent: {scene_extent}. Scaling camera view axis by {axis_scale}.")

    projection_matrices = np.stack([img.camera_metadata.projection_matrix for img in sfm_scene.images], axis=0)

    images = []
    for img in sfm_scene.images:
        cv_img = cv2.imread(str(img.image_path))
        if cv_img is None:
            raise ValueError(f"Could not read image {img.image_path}")
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        images.append(cv_img)

    viewer.camera_far = 100.0 * float(np.linalg.norm(scene_extent))
    viewer.register_camera_view(
        "cameras",
        sfm_scene.camera_to_world_matrices,
        projection_matrices,
        image_sizes=sfm_scene.image_sizes,
        images=images,
        frustum_line_width=2.0,
        frustum_scale=1.0 * axis_scale,
        axis_length=2.0 * axis_scale,
        axis_thickness=0.1 * axis_scale,
        show_images=show_images,
    )
    viewer.viser_server.scene.add_point_cloud(
        "points", sfm_scene.points, colors=sfm_scene.points_rgb, point_size=0.033 * axis_scale
    )

    logger.info("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
