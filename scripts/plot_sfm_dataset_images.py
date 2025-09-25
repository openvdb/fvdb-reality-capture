# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import pathlib

import cv2
import numpy as np
import tqdm

from fvdb_reality_capture.sfm_scene import SfmScene
from fvdb_reality_capture.training import SfmDataset
from fvdb_reality_capture.transforms import DownsampleImages

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/360_v2/garden")
    parser.add_argument("--image_downsample_factor", type=int, default=4)
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path)
    sfm_scene: SfmScene = SfmScene.from_colmap(dataset_path)
    transform = DownsampleImages(image_downsample_factor=args.image_downsample_factor)

    # Parse COLMAP data.
    dataset = SfmDataset(
        sfm_scene=sfm_scene,
        return_visible_points=True,
    )
    print(f"Dataset: {len(dataset)} images.")

    imsize = None
    for i, data in tqdm.tqdm(enumerate(dataset), desc="Plotting points"):
        image = data["image"].astype(np.uint8)
        # Make sure all images we write are the same size. We use the first image to determine the size of the video.
        # This is done because some images have slightly different sizes due to undistortion.
        imsize = image.shape if imsize is None else imsize
        if image.shape != imsize:
            new_image = np.zeros(imsize, dtype=np.uint8)
            new_image[: image.shape[0], : image.shape[1]] = image[: imsize[0], : imsize[1]]
            image = new_image
        points = data["points"]
        depths = data["depths"]
        for x, y in points:  # type: ignore
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        cv2.imwrite("{i:04d}_points.png", image)
