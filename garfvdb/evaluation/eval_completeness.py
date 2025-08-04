# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import json
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import tyro

sys.path.insert(0, str(Path(__file__).parent.parent))
from garfvdb.dataset import GARfVDBInput, SegmentationDataset
from garfvdb.model import GARfVDBModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from datasets import SfmDataset
from datasets.sfm_scene._colmap_utils.image import Image
from datasets.sfm_scene._colmap_utils.rotation import Quaternion
from datasets.transforms import (
    Compose,
    DownsampleImages,
    NormalizeScene,
    PercentileFilterPoints,
)
from evaluation.nerfstudio_to_colmap import nerfstudio_to_colmap_xform
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


def transform_nerfstudio_poses_to_original_space(
    poses: torch.Tensor,
    applied_transform: torch.Tensor,
    applied_scale: float,
) -> torch.Tensor:
    """
    Transforms the Nerfstudio camera poses in their normalized space back to the original world coordinate system.
    Args:
        poses: Poses in the transformed space
        applied_transform: Transform matrix applied in the data processing step
        applied_scale: Scale used in the data processing step
    Returns:
        Original poses
    """
    output_poses = poses
    output_poses[..., :3, 3] /= applied_scale
    inv_transform = torch.linalg.inv(
        torch.cat(
            (
                applied_transform,
                torch.tensor([[0, 0, 0, 1]], dtype=applied_transform.dtype, device=applied_transform.device),
            ),
            0,
        )
    )
    output_poses = torch.einsum("ij,bjk->bik", inv_transform, output_poses)

    return output_poses


def transform_camera_to_worlds_to_gs_scene_normalized_space(
    camera_to_worlds: torch.Tensor,
    colmap_dataset: SfmDataset,
) -> torch.Tensor:
    """
    Transforms the camera poses from the original COLMAP space to the scene normalized space the 3dgs model was trained in.

    Args:
        camera_to_worlds: Camera poses in the original COLMAP space
        colmap_dataset: COLMAP/Sfm dataset
    Returns:
        Camera poses in the scene normalized space
    """
    camera_to_worlds_np = camera_to_worlds.cpu().numpy()
    normalize_transform = colmap_dataset._transform.transforms[0]
    camera_to_worlds_np = normalize_transform.transform_camera_poses_to_scene_normalized_space(
        colmap_dataset._sfm_scene, camera_to_worlds_np
    )
    return torch.from_numpy(camera_to_worlds_np)


def transform_poses_to_training_space(
    poses: torch.Tensor,  # Float[Tensor, "num_poses 3 4"],
    applied_transform: torch.Tensor,  # Float[Tensor, "3 4"],
    applied_scale: float,
) -> torch.Tensor:  # Float[Tensor, "num_poses 3 4"]:
    """
    Transforms poses from original space to the same space used during training.
    This applies the SAME transformations that were applied to training data.
    Args:
        poses: Poses in the original space (like camera_path.json)
        applied_transform: Transform matrix applied in the data processing step
        applied_scale: Scale used in the data processing step
    Returns:
        Poses in the training space
    """
    # Convert to 4x4 matrices for easier manipulation
    output_poses = torch.cat(
        (
            poses,
            torch.tensor([[[0, 0, 0, 1]]], dtype=poses.dtype, device=poses.device).expand(poses.shape[0], -1, -1),
        ),
        dim=1,
    )

    # Apply the same transform that was applied during training
    transform_4x4 = torch.cat(
        (
            applied_transform,
            torch.tensor([[0, 0, 0, 1]], dtype=applied_transform.dtype, device=applied_transform.device),
        ),
        0,
    )

    output_poses = torch.einsum("ij,bjk->bik", transform_4x4, output_poses)

    # Apply the same scale that was applied during training
    output_poses[..., :3, 3] *= applied_scale

    # Return only the 3x4 part
    return output_poses[:, :3, :]


def read_camera_from_nerfstudio_json(
    camera_path_json_path: Path, device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Reads camera poses from a Nerfstudio camera path JSON file.

    Args:
        camera_path_json_path: Path to the camera path JSON file
        device: Device to load the camera poses to
    Returns:
        camera_to_worlds: Camera poses in the Nerfstudio space
        Ks: Camera intrinsics
        image_height: Image height
        image_width: Image width
    """
    with open(camera_path_json_path, "r") as f:
        camera_path_json = json.load(f)

    image_height = camera_path_json["render_height"]
    image_width = camera_path_json["render_width"]  # TODO: check if this is correct

    if "camera_type" not in camera_path_json or camera_path_json["camera_type"] != "perspective":
        raise ValueError(f"Camera type not supported.")

    c2ws = []
    Ks = []
    for camera in camera_path_json["camera_path"]:
        # pose
        c2w = torch.tensor(camera["camera_to_world"]).view(4, 4)
        c2ws.append(c2w)

        # field of view
        fov = camera["fov"]
        pp_h = image_height / 2.0
        focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
        Ks.append(torch.tensor([[focal_length, 0, image_width / 2], [0, focal_length, image_height / 2], [0, 0, 1]]))

    camera_to_worlds = torch.stack(c2ws, dim=0)
    Ks = torch.stack(Ks, dim=0)

    return camera_to_worlds.to(device), Ks.to(device), image_height, image_width


def load_coco_json(json_file_path: Path, img_height: int, img_width: int) -> Tuple[list, list]:
    """
    Load COCO (https://doi.org/10.1007/978-3-319-10602-1_48) JSON file and extract masks as NumPy arrays, merging masks with the same label,
    and return category names.

    :param json_file: Path to the COCO JSON file
    :param img_height: Height of the image
    :param img_width: Width of the image
    :return: Tuple containing:
             - List of masks for each image, with masks merged by label and sorted by category ID
             - Sorted list of category names
    """
    coco = COCO(json_file_path)
    image_masks = {}  # Temporary storage for masks
    sorted_masks = []  # Final storage for sorted masks
    category_names = {}

    for ann in coco.anns.values():
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        rle = ann["segmentation"]

        if type(rle) == list:
            rle = maskUtils.frPyObjects(rle, img_height, img_width)
        m = maskUtils.decode(rle)

        if image_id not in image_masks:
            image_masks[image_id] = {}
        if category_id not in image_masks[image_id]:
            image_masks[image_id][category_id] = m
        else:
            image_masks[image_id][category_id] = np.logical_or(image_masks[image_id][category_id], m)

        if category_id not in category_names:
            category_names[category_id] = coco.cats[category_id]["name"]

    # Sort images by image ID
    for image_id in sorted(image_masks):
        sorted_categories = sorted(image_masks[image_id].items())
        sorted_masks.append([mask.squeeze() for _, mask in sorted_categories])

    # Sort category names
    sorted_category_names = [name for _, name in sorted(category_names.items())]

    return sorted_masks, sorted_category_names


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Helper function to calculate mIoU (mean intersection over union) between two masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def run_completeness_exp(
    colmap_dataset_path: Path,
    garfvdb_checkpoint_path: Path,
    gsplat_checkpoint_path: Path,
    segmentation_dataset_path: Path,
    garfield_output_path: Path,
    eval_data_path: Path,
    scene_name: Literal["bouquet", "dozer_nerfgun_waldo", "keyboard", "ramen", "teatime"],
    results_path: Path,
):
    """
    Run completeness experiment, and save results including masks and IoU scores.
    "dozer_nerfgun_waldo" is the "livingroom" scene!

    Args:
        config_path: path to config file (e.g. outputs/.../config.yml)
        eval_data_path: path to eval data (e.g., eval_data/)
        scene_name: scene name
        results_path: where to save results (as image/png)
    """

    transforms = [
        NormalizeScene(normalization_type="pca"),
        PercentileFilterPoints(
            percentile_min=np.full((3,), 0.0),
            percentile_max=np.full((3,), 100.0),
        ),
        DownsampleImages(
            image_downsample_factor=1,
        ),
    ]

    colmap_dataset = SfmDataset(colmap_dataset_path, test_every=1, split="all", transform=Compose(*transforms))

    full_dataset = SegmentationDataset(segmentation_dataset_path)

    ### Model ###
    # Scale grouping stats
    grouping_scale_stats = torch.cat(full_dataset.scales)
    max_scale = grouping_scale_stats.max()
    # garfvdb model
    model = GARfVDBModel.create_from_checkpoint(
        garfvdb_checkpoint_path,
        gsplat_checkpoint_path,
        grouping_scale_stats,
    )
    model.eval()

    ### Cameras ###
    base_dir = eval_data_path / scene_name / "exp1_completeness"

    # Load camera poses from garfield eval data
    camera_poses_path = base_dir / "camera_path.json"

    camera_to_worlds, Ks, image_height, image_width = read_camera_from_nerfstudio_json(camera_poses_path)

    # Garfield output for dataparser transforms that are applied to the eval camera poses
    dataparser_transforms_path = garfield_output_path / "dataparser_transforms.json"
    with open(dataparser_transforms_path, "r") as f:
        dataparser_transforms = json.load(f)
    dataparser_transform = torch.tensor(dataparser_transforms["transform"])
    dataparser_scale: float = dataparser_transforms["scale"]

    # Transform Nerftudio normalized camera poses to original space
    camera_to_worlds = transform_nerfstudio_poses_to_original_space(
        camera_to_worlds, dataparser_transform, dataparser_scale
    ).contiguous()

    # Convert Nerftudio camera poses to COLMAP camera poses
    world_to_cameras = []
    for c2w in camera_to_worlds:
        q, t = nerfstudio_to_colmap_xform(c2w.cpu().numpy())
        world_to_cameras.append(torch.tensor(Image("", 0, Quaternion(q), t).world_to_cam_matrix()))
    world_to_cameras = torch.stack(world_to_cameras)
    camera_to_worlds = torch.linalg.inv(world_to_cameras)

    camera_to_worlds = transform_camera_to_worlds_to_gs_scene_normalized_space(camera_to_worlds, colmap_dataset)

    ### Experiment Data ###
    # Get GT masks
    gt_masks_path = base_dir / "groundtruth.json"
    gt_masks, _ = load_coco_json(gt_masks_path, image_height, image_width)
    gt_masks = np.array(gt_masks)  # (camera_idx, level, height, width)

    # Get experiment metadata
    experiment_info_path = base_dir / "experiment_info.json"
    with open(experiment_info_path, "r") as f:
        experiment_info = json.load(f)  # includes click position, and scales

    # Create canvas for displaying results!
    fig, axes = plt.subplots(len(camera_to_worlds), len(experiment_info["levels"]) + 1, figsize=(30, 20))

    iou_list = np.zeros((len(camera_to_worlds), len(experiment_info["levels"])))

    for i, (camera_to_world, K, click) in tqdm.tqdm(
        enumerate(zip(camera_to_worlds, Ks, experiment_info["click_position"])),
        total=len(camera_to_worlds),
    ):
        camera_to_world = camera_to_world.unsqueeze(0).float().to(model.device)
        world_to_camera = torch.linalg.inv(camera_to_world).contiguous()
        K = K.unsqueeze(0).float().to(model.device)
        input = GARfVDBInput(
            intrinsics=K,
            cam_to_world=camera_to_world,
            image_w=torch.tensor([image_width]),
            image_h=torch.tensor([image_height]),
        ).to(model.device)

        rgb, alpha = model.gs_model.render_images(
            image_width=image_width,
            image_height=image_height,
            world_to_camera_matrices=world_to_camera,
            projection_matrices=K,
            near=0.01,
            far=1e10,
            # sh_degree_to_use=0,
        )
        rgb = torch.clamp(rgb, 0.0, 1.0)

        rgb = rgb[0]
        alpha = alpha[0]
        for j in tqdm.trange(len(experiment_info["levels"]), desc="Level", leave=False):
            # Per paper: "we mine multiple masks from GARField across multiple scales
            # at 0.05 increments, where at each scale a mask is obtained based on
            # feature similarity thresholded at 0.9.
            # We report the maximum mIOU computed over all candidate masks.

            best_scale, best_mask, best_iou = None, None, 0.0
            top_10_scales, top_10_masks, top_10_ious = [], [], []
            scale_increment = max_scale / 4 / 100
            for curr_scale in tqdm.tqdm(
                np.arange(scale_increment, max_scale / 4 + scale_increment, scale_increment), desc="Scale", leave=False
            ):
                # Get outputs at the current scale.
                instance, alpha = model.get_mask_output(input, scale=curr_scale)
                instance = instance[0]
                alpha = alpha[0]

                # Calculate affinity between the click point and the rest of the rendered feats
                # Select region with >0.9 affinity (== "mask").
                click_feat = instance[click[1], click[0], :]
                affinity = instance @ click_feat
                mask = torch.where(affinity >= 0.90, 1, 0)

                # Calculate IoU between the rendered mask and the GT mask.
                iou = calculate_iou(mask.detach().cpu().numpy(), gt_masks[i][j])

                if iou >= best_iou:
                    best_iou, best_mask, best_scale = iou, mask, curr_scale

                if len(top_10_ious) < 5:
                    top_10_scales.append(curr_scale)
                    top_10_masks.append(mask)
                    top_10_ious.append(iou)
                else:
                    min_idx = np.argmin(top_10_ious)
                    if iou > top_10_ious[min_idx]:
                        top_10_scales[min_idx] = curr_scale
                        top_10_masks[min_idx] = mask
                        top_10_ious[min_idx] = iou

            iou_list[i, j] = best_iou

            if best_mask is None:
                continue

            import PIL

            # Display rendered RGB image on the leftmost column.
            if j == 0:
                import PIL.Image

                saved_jpg = base_dir / f"{i:05d}.jpg"
                saved_jpg = np.array(PIL.Image.open(saved_jpg)) / 255.0
                axes[i][0].imshow(saved_jpg)
                axes[i][0].set_title(f"Camera {i}")
                axes[i][0].axis("off")
                axes[i][0].scatter(click[0], click[1], s=3, c="red", marker="*")

            # Display rendered mask on the right columns, with RGB image overlaid.
            axes[i][j + 1].imshow(best_mask.detach().cpu().numpy())
            axes[i][j + 1].imshow(rgb.detach().cpu().numpy(), alpha=0.6)
            axes[i][j + 1].set_title(f"Scale {best_scale:.4f}, IoU {best_iou:.2f}")
            axes[i][j + 1].axis("off")

            # Also display the GT mask, but with a contour/boundary.
            gt_mask = gt_masks[i][j]
            axes[i][j + 1].contour(gt_mask, linewidths=0.2, colors="red")

            # Also display the click point.
            axes[i][j + 1].scatter(click[0], click[1], s=3, c="red", marker="*")

    mIoU_per_level = np.mean(iou_list, axis=0)
    mIoU_per_level = np.round(mIoU_per_level * 100, decimals=2)

    fig.suptitle(f"(1) Completeness experiment on {scene_name}; mIoU(s): {mIoU_per_level.tolist()}")
    fig.tight_layout()

    fig.savefig(results_path)
    print(f"Scene: {scene_name}, mIoU(s): {mIoU_per_level.tolist()}")


if __name__ == "__main__":
    tyro.cli(run_completeness_exp)
