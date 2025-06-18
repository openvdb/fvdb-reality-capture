#! /usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from garfvdb.dataset import SegmentationDataset
from garfvdb.model import GARfVDBModel
from garfvdb.util import pca_projection_fast
from torch.utils.data import DataLoader
from tqdm import tqdm


def main(gs_checkpoint_path: str, garfvdb_checkpoint_path: str, segmentation_dataset_path: str):
    torch.manual_seed(0)
    device = "cuda:0"
    dataset = SegmentationDataset(segmentation_dataset_path)
    grouping_scale_stats = torch.cat(dataset.scales)
    model = GARfVDBModel(gs_checkpoint_path, grouping_scale_stats, depth_samples=12, device=device)
    model.load_checkpoint(garfvdb_checkpoint_path)
    model.eval()
    # model.gs_model.save_ply("gs_model.ply")

    scale = 0.1
    batch_size = 2
    trainloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=dataset._collate_fn,
    )

    quantile_transformer = model.get_quantile_transformer()
    scale = torch.tensor([[scale]])
    scale = quantile_transformer(scale)[0, 0]

    for i, sample in tqdm(enumerate(trainloader)):
        for k, v in sample.items():
            sample[k] = v.to(device)

        image = sample["image"]
        intrinsics = sample["intrinsics"]
        cam_to_world = sample["cam_to_world"]
        h, w = sample["image_h"][0], sample["image_w"][0]

        with torch.no_grad():
            mask_output = model.get_mask_output(sample, scale)
            mask_output = mask_output.reshape((mask_output.shape[0], h, w, -1))

            world_to_cam_mats = torch.linalg.inv(cam_to_world.to(device)).contiguous()
            intrinsics_mats = intrinsics.to(device)

            gauss_output, _ = model.gs_model.render_images(
                image_width=w,
                image_height=h,
                world_to_camera_matrices=world_to_cam_mats,
                projection_matrices=intrinsics_mats,
                near=0.0,
                far=1e10,
            )
            gauss_output = torch.clamp(gauss_output, 0.0, 1.0)

        ### PCA to 3 colors
        pca_output = pca_projection_fast(mask_output.detach(), 3)

        ### Save images
        image = image.cpu().numpy()
        pca_output = (pca_output.cpu().numpy() * 255).astype(np.uint8)
        gauss_output = (gauss_output.cpu().numpy() * 255).astype(np.uint8)

        for j, (img, pca_img, gauss_img) in enumerate(zip(image, pca_output, gauss_output)):
            image_path = f"test_images/image_{i*batch_size + j}.png"
            # mask_output_path = f"test_images/mask_output_{i}.png"
            pca_output_path = f"test_images/pca_output_{i*batch_size + j}.png"
            gauss_output_path = f"test_images/gauss_output_{i*batch_size + j}.png"
            plt.imsave(image_path, img)
            plt.imsave(pca_output_path, pca_img)
            plt.imsave(gauss_output_path, gauss_img)


if __name__ == "__main__":
    tyro.cli(main)
