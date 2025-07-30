# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import uuid
from typing import Sequence

import torch
from datasets import SfmDataset

from fvdb import GaussianSplat3d

from ..checkpoint import Checkpoint
from ..scene_optimization_runner import Config


def merge_checkpoints(
    checkpoint_list: Sequence[Checkpoint],
    run_name: str | None = None,
    config: Config | None = None,
    train_dataset: SfmDataset | None = None,
    eval_dataset: SfmDataset | None = None,
) -> Checkpoint:
    """
    Merge a list of checkpoints into a single checkpoint.
    The merged checkpoint will contain the splats, camera matrices, and other metadata from all checkpoints.

    Note: If you pass in a train_dataset and eval_dataset, they will be used as the datasets for
        the merged checkpoint. If you pass in None, the merged checkpoint will not contain any
        datasets, and the camera poses will simply be concatenated.

    Note: If you pass in a config, it will be used as the configuration for the merged checkpoint,
        otherwise the checkpoint will have no config.

    Note: If you pass `run_name=None`, the merged checkpoint will have a default run name of
        "merged_UUID" where UUID is a generated UUID.

    Note: The merged checkpoint will not contain the optimizer state, so it is not
        suitable for resuming training.

    Args:
        checkpoint_list (Sequence[Checkpoint]): List of Checkpoint objects to merge.
        run_name (str | None): Name for the merged checkpoint. If None, a default name will be generated
            of the form "merged_UUID" where UUID is a generated UUID.
        config (Config | None): Configuration for the merged checkpoint. If None, the merged checkpoint will not
            contain any configuration.
        train_dataset (SfmDataset | None): Training dataset for the merged checkpoint. If None, the merged
            checkpoint will not contain a training dataset and the dataset metadata (camera poses, etc.) will
            just be the concatenation of the dataset metadata in the input checkpoint list.
        eval_dataset (SfmDataset | None): Evaluation dataset for the merged checkpoint. If None, the merged
            checkpoint will not contain an evaluation dataset and the dataset metadata (camera poses, etc.) will
            just be the concatenation of the dataset metadata in the input checkpoint list.

    Returns:
        Checkpoint: A new Checkpoint object containing the merged data.
    """
    if not checkpoint_list:
        raise ValueError("The checkpoint list is empty.")

    new_splat: GaussianSplat3d = GaussianSplat3d.cat([cp.splats for cp in checkpoint_list])

    if train_dataset is None and eval_dataset is not None:
        raise ValueError("If eval_dataset is provided, train_dataset must also be provided.")
    if train_dataset is not None and eval_dataset is None:
        raise ValueError("If train_dataset is provided, eval_dataset must also be provided.")

    if run_name is None:
        run_name = f"merged_{uuid.uuid4()}"

    if train_dataset is not None and eval_dataset is not None:
        merged_checkpoint = Checkpoint.make_checkpoint(
            step=-1,
            run_name=run_name,
            model=new_splat,
            config=vars(config),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizer=None,
            pose_adjust_model=None,
            pose_adjust_optimizer=None,
            pose_adjust_scheduler=None,
        )
    else:
        camera_origins = torch.stack([cp.camera_to_world_matrices[:, :3, 3] for cp in checkpoint_list], dim=0)
        mean_origin = camera_origins.mean(dim=0, keepdim=True)
        dists = torch.norm(camera_origins - mean_origin, dim=-1)
        scene_scale = torch.max(dists).item()

        final_train_c2w = torch.cat([cp.final_training_poses for cp in checkpoint_list], dim=0)
        train_proj_matrices = torch.cat([cp.training_projection_matrices for cp in checkpoint_list], dim=0)
        train_image_sizes = torch.cat([cp.training_image_sizes for cp in checkpoint_list], dim=0)
        eval_c2w = torch.cat([cp.eval_poses for cp in checkpoint_list], dim=0)
        eval_proj_matrices = torch.cat([cp.eval_projection_matrices for cp in checkpoint_list], dim=0)
        eval_image_sizes = torch.cat([cp.eval_image_sizes for cp in checkpoint_list], dim=0)

        merged_checkpoint = Checkpoint.make_minimal_checkpoint(
            run_name=run_name,
            model=new_splat,
            scene_scale=scene_scale,
            training_camera_to_world_matrices=final_train_c2w,
            training_projection_matrices=train_proj_matrices,
            training_image_sizes=train_image_sizes,
            eval_camera_to_world_matrices=eval_c2w,
            eval_projection_matrices=eval_proj_matrices,
            eval_image_sizes=eval_image_sizes,
        )

    return merged_checkpoint
