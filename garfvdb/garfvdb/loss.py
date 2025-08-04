# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import math
from typing import Dict

import torch
from garfvdb.model import GARfVDBModel
from torch.nn import functional as F


def chunked_norm(features_flat, mask_0, mask_1, chunk_size=10000) -> torch.Tensor:
    """
    Calculate all individual L2 norms efficiently by pre-allocating result tensor
    and filling it chunk by chunks.

    Args:
        features_flat: Feature tensor [N, feature_dim]
        mask_0, mask_1: Index tensors for pairs
        chunk_size: Number of pairs to process at once

    Returns:
        torch.Tensor: All individual L2 norms [total_pairs]
    """
    total_pairs = len(mask_0)
    device = features_flat.device

    # Pre-allocate result tensor
    all_norms = torch.empty(total_pairs, dtype=torch.float32, device=device)

    # Process in chunks to avoid additional memory spikes
    for i in range(0, total_pairs, chunk_size):
        end_idx = min(i + chunk_size, total_pairs)

        # Get chunks of indices
        chunk_mask_0 = mask_0[i:end_idx]
        chunk_mask_1 = mask_1[i:end_idx]

        # Get corresponding features
        chunk_feats_0 = features_flat[chunk_mask_0]
        chunk_feats_1 = features_flat[chunk_mask_1]

        chunk_norms = torch.norm(chunk_feats_0 - chunk_feats_1, p=2, dim=-1)

        # Store directly in pre-allocated tensor
        all_norms[i:end_idx] = chunk_norms

        del chunk_feats_0, chunk_feats_1, chunk_norms

    return all_norms


def calculate_loss(
    model: GARfVDBModel,
    enc_feats: torch.Tensor,
    input: Dict[str, torch.Tensor],
    return_loss_images: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Calculate contrastive loss for GARfVDB instance grouping/segmentation.

    This function computes a multi-component contrastive loss that encourages points
    belonging to the same instance to have similar features while pushing apart
    features from different instances.

    Args:
        model (GARfVDBModel): The model instance used to compute MLP outputs and access
            configuration parameters like max_grouping_scale.
        enc_feats (torch.Tensor): Encoded features tensor of shape [batch_size*samples_per_img, feat_dim]
            containing the feature representations for each point/pixel.
        input (Dict[str, torch.Tensor]): Input dictionary containing:
            - "image": Input images of shape [B, H, W, C] or [B, num_samples, C]
            - "mask_id": Instance mask IDs of shape matching image spatial dims, where
                        each unique ID represents a different instance (-1 for invalid/background)
            - "scales": Scale values for hierarchical feature computation
        return_loss_images (bool, optional): Whether to compute and return loss visualization
            images in addition to the total loss. Since this is memory intensive, we use float16
            when computing loss across the whole image for memory efficiency.
            Defaults to False.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - "total_loss": Normalized total contrastive loss value (scalar)
            - "instance_loss_1_img": (if return_loss_images=True) Visualization of loss 1
              per pixel/point, normalized to [0,1]
            - "instance_loss_2_img": (if return_loss_images=True) Visualization of loss 2
              per pixel/point, normalized to [0,1]
            - "instance_loss_4_img": (if return_loss_images=True) Visualization of loss 4
              per pixel/point, normalized to [0,1]

    Loss Components:
        1. **Instance Loss 1**: Pulls together features of points with the same mask ID
           at the same scale. Computed as L2 distance between positive pairs.

        2. **Instance Loss 2**: Hierarchical consistency loss that also pulls together
           features of points with the same mask ID but at larger scales to enforce
           multi-scale consistency.

        3. **Instance Loss 4**: Pushes apart features of points with different mask IDs
           using a margin-based hinge loss (ReLU(margin - distance)).

    Implementation Details:
        - Uses block masking to prevent cross-image comparisons in batched inputs
        - Only considers upper triangular pairs to avoid double-counting
        - Excludes diagonal pairs (same point with itself) from grouping supervision
        - Filters out invalid pairs where mask_id == -1
        - Loss visualization images are computed by scattering loss values back to
          their corresponding spatial locations in the input image
        - Normalizes final loss by the total number of valid pairs considered
    """

    return_loss_dict = {}

    # Using a product of this form to accomodate 'image' inputs of the form [B, num_samples, C] and [B, H, W, C]
    samples_per_img = math.prod(input["image"].shape[1:-1])

    num_chunks = enc_feats.shape[0]

    input_id1 = input_id2 = input["mask_id"].flatten()

    # Debug prints
    logging.debug(
        f"calc_loss shapes: enc_feats={enc_feats.shape}, input_id1={input_id1.shape}, mask_id={input['mask_id'].shape}"
    )

    # Expand labels
    labels1_expanded = input_id1.unsqueeze(1).expand(-1, input_id1.shape[0])
    labels2_expanded = input_id2.unsqueeze(0).expand(input_id2.shape[0], -1)

    # Mask for positive/negative pairs across the entire matrix
    mask_full_positive = labels1_expanded == labels2_expanded
    mask_full_negative = ~mask_full_positive

    # # Debug print
    logging.debug(
        f"num_chunks = {num_chunks}, input_id1.shape[0] = {input_id1.shape[0]}, samples_per_img = {samples_per_img}"
    )

    # Create a block mask to only consider pairs within the same image -- no cross-image pairs
    block_mask = torch.kron(  # [samples_per_img*num_chunks, samples_per_img*num_chunks] dtype: torch.bool
        torch.eye(num_chunks, device=labels1_expanded.device, dtype=torch.bool),
        torch.ones(
            (samples_per_img, samples_per_img),
            device=labels1_expanded.device,
            dtype=torch.bool,
        ),
    )  # block-diagonal matrix, to consider only pairs within the same image

    logging.debug(f"block_mask.shape = {block_mask.shape}")

    # Only consider upper triangle to avoid double-counting
    block_mask = torch.triu(
        block_mask, diagonal=0
    )  # [samples_per_img*num_chunks, samples_per_img*num_chunks] dtype: torch.bool
    # Only consider pairs where both points are valid (-1 means not in mask / invalid)
    block_mask = block_mask * (labels1_expanded != -1) * (labels2_expanded != -1)

    # Mask for diagonal elements (i.e., pairs of the same point).
    # Don't consider these pairs for grouping supervision (pulling), since they are trivially similar.
    diag_mask = torch.eye(block_mask.shape[0], device=block_mask.device, dtype=torch.bool)

    scales = input["scales"]

    ####################################################################################
    # Grouping supervision
    ####################################################################################

    # Get instance features - will return a 3D tensor [batch_size, samples_per_img, feat_dim]
    instance_features = model.get_mlp_output(enc_feats, scales)

    # Flatten the instance features to match the masking operations
    # [batch_size, samples_per_img, feat_dim] -> [batch_size*samples_per_img, feat_dim]
    instance_features_flat = instance_features.reshape(-1, instance_features.shape[-1])

    # 1. If (A, s_A) and (A', s_A) in same group, then supervise the features to be similar
    mask = torch.where(mask_full_positive * block_mask * (~diag_mask))

    logging.debug(f"mask0 shape: {mask[0].shape}")
    logging.debug(f"mask1 shape: {mask[1].shape}")

    if mask[0].shape[0] > 1000000:
        instance_loss_1 = chunked_norm(instance_features_flat, mask[0], mask[1], chunk_size=10000)
    else:
        instance_loss_1 = torch.norm(instance_features_flat[mask[0]] - instance_features_flat[mask[1]], p=2, dim=-1)

    if not (mask[0] // samples_per_img == mask[1] // samples_per_img).all():
        logging.error("Loss Function: There's a camera cross-talk issue")

    instance_loss_1_sum = instance_loss_1.nansum()
    return_loss_dict["instance_loss_1"] = instance_loss_1_sum
    logging.debug(f"Loss 1: {instance_loss_1_sum.item()}, using {mask[0].shape[0]} pairs")
    del instance_loss_1_sum

    if return_loss_images:
        with torch.no_grad():
            loss_1_img = torch.zeros(input["image"].shape[:-1], device=instance_loss_1.device)
            # Use scatter_reduce_ to accumulate values
            loss_1_img.view(-1).scatter_reduce_(
                0,  # dim to reduce along
                mask[0],  # indices
                instance_loss_1,  # values to scatter
                reduce="sum",  # reduction operation
                include_self=True,  # include values in the output
            )
            # rescale loss_1_img to [0, 1]
            loss_1_img = loss_1_img / torch.max(loss_1_img)
            return_loss_dict["instance_loss_1_img"] = loss_1_img.detach()
            del loss_1_img
    del instance_loss_1

    # 2. If ", then also supervise them to be similar at s > s_A
    # if self.config.use_hierarchy_losses and (not self.config.use_single_scale):

    scale_diff = torch.max(torch.zeros_like(scales), (model.get_max_grouping_scale() - scales))
    larger_scale = scales + scale_diff * torch.rand(size=(1,), device=scales.device)

    # Get larger scale features and flatten
    larger_scale_instance_features = model.get_mlp_output(enc_feats, larger_scale)
    larger_scale_instance_features_flat = larger_scale_instance_features.reshape(
        -1, larger_scale_instance_features.shape[-1]
    )

    if mask[0].shape[0] > 1000000:
        instance_loss_2 = chunked_norm(larger_scale_instance_features_flat, mask[0], mask[1], chunk_size=10000)
    else:
        instance_loss_2 = torch.norm(
            larger_scale_instance_features_flat[mask[0]] - larger_scale_instance_features_flat[mask[1]], p=2, dim=-1
        )
    instance_loss_2_nansum = instance_loss_2.nansum()
    return_loss_dict["instance_loss_2"] = instance_loss_2_nansum
    logging.debug(f"Loss 2: {instance_loss_2_nansum.item()}, using {mask[0].shape[0]} pairs")
    del instance_loss_2_nansum

    if return_loss_images:
        with torch.no_grad():
            # image of instance_loss_2
            loss_2_img = torch.zeros(input["image"].shape[:-1], device=instance_loss_2.device)
            # Use scatter_reduce_ to accumulate values, using 'sum' as the reduction operation
            loss_2_img.view(-1).scatter_reduce_(
                0,  # dim to reduce along
                mask[0],  # indices
                instance_loss_2,  # values to scatter
                reduce="sum",  # reduction operation
                include_self=True,  # include values in the output
            )
            loss_2_img = loss_2_img / loss_2_img.max()
            return_loss_dict["instance_loss_2_img"] = loss_2_img.detach()
            del loss_2_img
    del instance_loss_2

    # 4. Also supervising A, B to be dissimilar at scales s_A, s_B respectively seems to help.
    mask = torch.where(mask_full_negative * block_mask)

    # if return_loss_images:
    #     instance_features_flat = instance_features_flat.to(torch.float16)

    margin = 1.0

    if mask[0].shape[0] > 1000000:
        instance_loss_4 = F.relu(margin - chunked_norm(instance_features_flat, mask[0], mask[1], chunk_size=10000))
    else:
        instance_loss_4 = F.relu(
            margin - torch.norm(instance_features_flat[mask[0]] - instance_features_flat[mask[1]], p=2, dim=-1)
        )
    instance_loss_4_nansum = instance_loss_4.to(torch.float32).nansum()
    return_loss_dict["instance_loss_4"] = instance_loss_4_nansum
    logging.debug(f"Loss 4: {instance_loss_4_nansum.item()}, using {mask[0].shape[0]} pairs")
    del instance_loss_4_nansum

    if return_loss_images:
        with torch.no_grad():
            #  image of instance_loss_4
            loss_4_img = torch.zeros(
                input["image"].shape[:-1], device=instance_loss_4.device, dtype=instance_loss_4.dtype
            )
            # Use scatter_reduce_ to accumulate values, using 'sum' as the reduction operation
            loss_4_img.view(-1).scatter_reduce_(
                0,  # dim to reduce along
                mask[0],  # indices
                instance_loss_4,  # values to scatter
                reduce="sum",  # reduction operation
                include_self=True,  # include values in the output
            )
            loss_4_img = loss_4_img / loss_4_img.max()
            return_loss_dict["instance_loss_4_img"] = loss_4_img.detach()
            del loss_4_img
    del instance_loss_4

    return_loss_dict["total_loss"] = (
        return_loss_dict["instance_loss_1"] + return_loss_dict["instance_loss_2"] + return_loss_dict["instance_loss_4"]
    ) / torch.sum(block_mask).float()

    return return_loss_dict
