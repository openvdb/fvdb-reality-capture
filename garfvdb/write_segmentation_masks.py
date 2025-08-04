# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import json
import os

import cv2
import numpy as np
import torch
import tyro
from PIL import Image
from tqdm import tqdm


def convert_samdata_to_images(segmentation_data_filepath: str, output_dir: str):
    """
    Write debug visualization images from cached SAM segmentation data.

    This script loads cached segmentation data from SAM (Segment Anything Model)
    and generates debug images by overlaying each segmentation mask on the original
    images with a red transparent overlay. Useful for visually inspecting and
    debugging segmentation results.

    Args:
        segmentation_data_filepath: Path to the cached SAM segmentation data (.pt file)
        output_dir: Directory where debug images will be saved (organized by image index and mask ID)
    """
    os.makedirs(output_dir, exist_ok=True)
    segmentation_data = torch.load(segmentation_data_filepath)
    for i, (img, mask_ids) in tqdm(enumerate(zip(segmentation_data["images"], segmentation_data["mask_ids"]))):
        if not os.path.exists(os.path.join(output_dir, f"{i:03d}")):
            os.makedirs(os.path.join(output_dir, f"{i:03d}"), exist_ok=True)

        pixel_level_keys = mask_ids.numpy()
        colmap_img = Image.fromarray(img.cpu().numpy())

        # iterate over each mask ID
        num_masks = np.max(pixel_level_keys) + 1  # +1 because mask IDs start from 0
        for m in range(num_masks):
            # Create binary mask where any pixel with value i becomes 1, others become 0
            mask = np.any(pixel_level_keys == m, axis=-1).astype(np.uint8)  # Shape: (H, W)

            # resize mask to be the same size as the colmap image
            mask = cv2.resize(mask, (colmap_img.width, colmap_img.height))

            # Convert to PIL Image and save
            mask_image = Image.fromarray(mask * 255)  # Convert 0/1 to 0/255 for proper image display
            # overlay the mask on the image in red
            # Create a red overlay image
            red_overlay = Image.new("RGBA", colmap_img.size, (255, 0, 0, 0))  # Start with transparent red

            # Convert colmap image to RGBA if it isn't already
            if colmap_img.mode != "RGBA":
                colmap_img = colmap_img.convert("RGBA")

            # Create alpha channel: where mask is white (255), set alpha to 128 (50% opacity)
            # where mask is black (0), set alpha to 0 (transparent)
            alpha_data = np.array(mask_image)
            alpha_data = np.where(alpha_data == 255, 100, 0).astype(np.uint8)
            alpha_channel = Image.fromarray(alpha_data)
            red_overlay.putalpha(alpha_channel)

            # Composite the red overlay onto the colmap image
            result_img = Image.alpha_composite(colmap_img, red_overlay)
            # Convert back to RGB for JPEG saving
            result_img_rgb = result_img.convert("RGB")
            colmap_filename = os.path.join(output_dir, f"{i:03d}", f"{m:03d}.jpg")
            result_img_rgb.save(colmap_filename, quality=95)
            # print(f"Saved mask {i} for {key}/{camera_id} with shape {mask.shape}")


if __name__ == "__main__":
    tyro.cli(convert_samdata_to_images)
