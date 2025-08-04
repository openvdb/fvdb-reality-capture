# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

import h5py
import torch
import torch.utils.data
import tqdm
import tyro
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from fvdb import GaussianSplat3d

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets import SfmDataset
from datasets.transforms import Compose, DownsampleImages, NormalizeScene

logging.basicConfig(level=logging.WARN)
logging.getLogger().name = "make_segmentation_dataset"


def convert_pixel_level_keys_to_masks(pixel_level_keys: torch.Tensor) -> torch.Tensor:
    # iterate over each mask ID
    num_masks = torch.max(pixel_level_keys) + 1  # +1 because mask IDs start from 0
    masks = []
    for i in range(num_masks):
        # Create binary mask where any pixel with value i becomes 1, others become 0
        mask = torch.any(pixel_level_keys == i, dim=-1)  # Shape: (H, W)
        masks.append(mask)
    return torch.stack(masks)


def get_garfield_sam_masks_for_image(image_path: str, sam_data_path: str) -> torch.Tensor:
    sam_data = h5py.File(sam_data_path, "r")
    image_path = Path(image_path)
    image_name = image_path.name
    dataset_name = "sam_fb_points_per_side_32_pred_iou_thresh_0.9_stability_score_thresh_0.9"
    for cam_id in sam_data[dataset_name]["image_filenames"].keys():
        sam_image_filename = Path(sam_data[dataset_name]["image_filenames"][cam_id][()].decode("utf-8"))

        if sam_image_filename.name == image_name:
            pixel_level_keys = torch.tensor(sam_data[dataset_name]["pixel_level_keys"][cam_id])
            logging.debug(f"pixel_level_keys.shape {pixel_level_keys.shape}")
            masks = convert_pixel_level_keys_to_masks(pixel_level_keys)
            logging.debug(f"masks.shape {masks.shape}")
            return masks

    raise ValueError(f"Could not find {image_name} in {sam_data_path}")


def get_sam2_checkpoint(ckpt_name="sam2.1_hiera_large.pt") -> Path:
    import requests

    url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{ckpt_name}"
    output_dir = Path("sam2/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ckpt_name
    if not output_path.exists():
        print(f"Downloading SAM2 checkpoint from {url}")
        r = requests.get(url)
        with open(output_path, "wb") as f:
            f.write(r.content)

    return output_path


@torch.inference_mode()
def make_segmentation_dataset(
    model: GaussianSplat3d,
    dataset: SfmDataset,
    max_scale,
    device="cuda",
    garfield_sam_data_path: Optional[str] = None,
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # If we're re-using the garfield sam data, we don't need to use SAM2
    if garfield_sam_data_path is None:
        sam2_checkpoint = get_sam2_checkpoint()
        model_cfg = "configs/sam2.1/sam2.1_hiera_l"
        sam2 = build_sam2(model_cfg, ckpt_path=sam2_checkpoint, device=device, apply_postprocessing=False)
        sam2_mask_generator = SAM2AutomaticMaskGenerator(
            sam2, points_per_side=40, pred_iou_thresh=0.80, stability_score_thresh=0.80
        )

    all_scales = []
    all_pixel_to_mask_ids = []
    all_mask_cdfs = []
    all_images = []
    all_cam_to_world = []
    all_intrinsics = []

    for data in tqdm.tqdm(dataloader):
        img = data["image"].squeeze()  # [H, W, 3]
        intrinsics = data["projection"].to(device).squeeze()
        cam_to_world = data["camera_to_world"].to(device).squeeze()
        world_to_cam = torch.linalg.inv(cam_to_world).contiguous()

        logging.debug("img h %s w %s" % (img.shape[0], img.shape[1]))

        g_ids, _ = model.render_top_contributing_gaussian_ids(
            num_samples=1,
            world_to_camera_matrices=world_to_cam.unsqueeze(0),
            projection_matrices=intrinsics.unsqueeze(0),
            image_width=img.shape[1],
            image_height=img.shape[0],
            near=0.01,
            far=1e10,
        )
        g_ids = g_ids.squeeze().unsqueeze(-1)  # [H, W, 1]

        logging.debug("g_ids.shape " + str(g_ids.shape))
        if g_ids.max() >= model.means.shape[0]:
            logging.debug("g_ids.max() " + str(g_ids.max()))
            logging.debug("model.means.shape[0] " + str(model.means.shape[0]))
            raise ValueError("g_ids.max() is greater than model.means.shape[0]")

        invalid_mask = g_ids == -1
        if invalid_mask.any():
            logging.debug("Found %d invalid (-1) ids" % (invalid_mask.sum().item()))

        world_pts = model.means[g_ids].squeeze(2)  # [H, W, 3]

        if garfield_sam_data_path is None:
            # Generate a set of masks for the current image using SAM2
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sam_masks = sam2_mask_generator.generate(img.cpu().numpy())
                sam_masks = sorted(sam_masks, key=(lambda x: x["area"]), reverse=True)
                sam_masks = torch.stack([torch.from_numpy(m["segmentation"]) for m in sam_masks]).to(
                    device
                )  # [M, H, W]
            # Erode masks to remove noise at the boundary.
            # We're going to compute the scale of each mask by taking the standard deviation of the 3D points
            # within that mask, and the points at the boundary of masks are usually noisy.
            eroded_masks = torch.conv2d(
                sam_masks.unsqueeze(1).float(),
                torch.full((3, 3), 1.0, device=device).view(1, 1, 3, 3),
                padding=1,
            )
            eroded_masks = (eroded_masks >= 5).squeeze(1)  # [M, H, W]
        else:
            eroded_masks = get_garfield_sam_masks_for_image(data["image_path"][0], garfield_sam_data_path).to(device)

        # mask out any pixels with invalid gaussian ids in the sam_masks
        eroded_masks = eroded_masks * (~invalid_mask.squeeze().unsqueeze(0))

        # Compute a 3D scale per mask which corresponds to the variance of the 3D points that fall within that mask
        # Filter out masks whose scale is too large since very scattered 3D points are likely noise
        logging.debug("world_pts " + str(world_pts.shape))
        logging.debug("scale " + str(world_pts[eroded_masks[1]].std(dim=0) * 2.0))
        scales = torch.stack([world_pts[mask].std(dim=0).norm() for mask in eroded_masks])  # [M]
        keep = scales < max_scale  # [M]
        eroded_masks = eroded_masks[keep]  # [M', H, W]
        scales = scales[keep]  # [M']

        # Compute a tensor that maps pixels to the set of masks which intersect that pixel (sorted by area)
        # i.e. pixel_to_mask_id[i, j] = [m1, m2, m3, ...] where m1, m2, ... are the integer ids of the masks
        # which contain pixel [i, j] and area(m1) <= area(m2) <= area(m3) <= ...
        max_masks = int(eroded_masks.sum(dim=0).max().item())
        pixel_to_mask_id = torch.full(
            (max_masks, eroded_masks.shape[1], eroded_masks.shape[2]), -1, dtype=torch.long, device=device
        )  # [MM, H, W]
        for m, mask in enumerate(eroded_masks):
            mask_clone = mask.clone()
            for i in range(max_masks):
                free = pixel_to_mask_id[i] == -1
                masked_area = mask_clone == 1
                right_index = free & masked_area
                if len(pixel_to_mask_id[i][right_index]) > 0:
                    pixel_to_mask_id[i][right_index] = m
                mask_clone[right_index] = 0
        pixel_to_mask_id = pixel_to_mask_id.permute(1, 2, 0)  # [H, W, MM]

        # We're going to use the SAM masks to group pixels for contrastive learning.
        # i.e. we're going to project features for each pixel into the image and push features corresponding to pixels
        #      with the same mask together, and pixels with different masks apart.
        # If we sample pixels, uniformly, we're going to overwhelmingly sample pixels in large masks, and small masks
        # will not get supervised. To fix this, we assign a weight to each mask which intersects a pixel. The weight
        # is proportional to the log probability of sampling that mask (under uniform sampling).
        # These weights are encoded as a CDF per-pixel which we use to choose which mask to use for loss computation
        # at training time

        # Get the unique ids of each mask, and the number of pixels each mask occupies (area)
        mask_ids, num_pix_per_mask = torch.unique(pixel_to_mask_id, return_counts=True)  # [N], [N]

        # Sort masks by their area
        mask_area_sort_ids = torch.argsort(num_pix_per_mask)
        mask_ids, num_pix_per_mask = mask_ids[mask_area_sort_ids], num_pix_per_mask[mask_area_sort_ids]  # [N], [N]
        num_pix_per_mask[0] = 0  # Remove the -1 mask which corresponds to no mask, [N]

        # The probability of any pixel landing in a mask is just the area of the mask over the area of the image
        probs = num_pix_per_mask / num_pix_per_mask.sum()  # [N]

        # Gather the probability values into pixel_to_mask_id, which produces a tensor where
        # each pixel has a list of probabilities that correspond to the masks that intersect that pixel
        mask_probs = torch.gather(probs, 0, pixel_to_mask_id.reshape(-1) + 1).view(pixel_to_mask_id.shape)  # [H, W, MM]

        # Compute a CDF for each pixel (which sums to 1) which weighs each mask by its log probability of being sampled
        # i.e. mask_cdf[i, j, k] is a cumulative probability weight used to select mask k for pixel [i, j]
        mask_cdf = torch.log(mask_probs)
        never_masked = mask_cdf.isinf()
        mask_cdf[never_masked] = 0.0
        mask_cdf = mask_cdf / (mask_cdf.sum(dim=-1, keepdim=True) + 1e-6)
        mask_cdf = torch.cumsum(mask_cdf, dim=-1)  # [H, W, MM]
        mask_cdf[never_masked] = 1.0

        all_scales.append(scales.cpu())
        all_pixel_to_mask_ids.append(pixel_to_mask_id.cpu())
        all_mask_cdfs.append(mask_cdf.cpu())
        all_images.append(img.cpu())
        all_cam_to_world.append(cam_to_world.cpu())
        all_intrinsics.append(intrinsics.cpu())

    return all_scales, all_pixel_to_mask_ids, all_mask_cdfs, all_images, all_cam_to_world, all_intrinsics


def main(
    checkpoint_path: Path,
    colmap_path: Path,
    output_filepath: Path,
    garfield_sam_data_path: Optional[str] = None,
    data_scale_factor: int = 1,
    normalization_type: Literal["pca", "none"] = "pca",
    device: str = "cuda",
):
    """Generate segmentation dataset for Garfvdb training.

    This script can operate in two modes:
    1. Generate fresh segmentation data using SAM2 (when garfield_sam_data_path is None)
    2. Convert existing segmentation data generated by Garfield by reading the HDF5 file

    The output contains pixel-to-mask mappings, mask scales, and camera information that can be
    used for contrastive learning during GARfVDB training.

    Args:
        checkpoint_path: Path to the 3D Gaussian Splatting checkpoint (.ply or .pth file)
        colmap_path: Path to the COLMAP reconstruction directory containing cameras, images, and points3D
        output_filepath: Path where the segmentation dataset will be saved (.pth file)
        garfield_sam_data_path: Optional path to Garfield's HDF5 segmentation file. If provided,
                               segmentation data will be read from this file instead of generating
                               new masks with SAM2
        data_scale_factor: Downsampling factor for input images (default: 1 = no downsampling)
        device: Device to run computations on (default: "cuda")
    """
    print("Loading checkpoint from ", checkpoint_path)
    if checkpoint_path.suffix == ".ply":
        gs3d = GaussianSplat3d.from_ply(checkpoint_path, device=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        gs3d = GaussianSplat3d.from_state_dict(checkpoint["splats"])

    transforms = [
        NormalizeScene(normalization_type=normalization_type),
        DownsampleImages(data_scale_factor),
    ]
    dataset = SfmDataset(
        colmap_path, test_every=1, split="all", transform=Compose(*transforms)
    )  # , image_downsample_factor=data_scale_factor)

    min = gs3d.means.min(dim=0)[0]
    max = gs3d.means.max(dim=0)[0]
    extents = torch.abs(max - min)

    scales, mask_ids, mask_cds, imgs, cam_to_worlds, intrinsics = make_segmentation_dataset(
        gs3d, dataset, device=device, max_scale=extents.max().item(), garfield_sam_data_path=garfield_sam_data_path
    )

    torch.save(
        {
            "scales": scales,
            "mask_ids": mask_ids,
            "mask_cdfs": mask_cds,
            "images": imgs,
            "cam_to_worlds": cam_to_worlds,
            "intrinsics": intrinsics,
        },
        output_filepath,
    )


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
