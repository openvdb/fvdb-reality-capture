# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from fvdb_reality_capture.instance_segmentation.model import GARfVDBModel


def chunked_norm(
    features_flat: torch.Tensor,
    mask_0: torch.Tensor,
    mask_1: torch.Tensor,
    chunk_size: int = 10000,
) -> torch.Tensor:
    """Calculate L2 norms between feature pairs efficiently using chunked processing.

    Pre-allocates the result tensor and processes pairs in chunks to avoid
    memory spikes when computing norms for large numbers of feature pairs.

    Args:
        features_flat: Feature tensor of shape ``[N, feature_dim]``.
        mask_0: Index tensor for first element of pairs.
        mask_1: Index tensor for second element of pairs.
        chunk_size: Number of pairs to process at once.

    Returns:
        L2 norms of shape ``[total_pairs]`` for all feature pairs.
    """
    total_pairs = len(mask_0)
    device = features_flat.device

    # Pre-allocate result tensor
    all_norms = torch.empty(total_pairs, dtype=torch.float32, device=device)

    # Process in chunks to avoid memory spikes
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


def _sum_loss_per_view(
    per_pair_data: tuple[torch.Tensor, torch.Tensor],
    num_views: int,
    samples_per_img: int,
) -> torch.Tensor:
    """Sum per-pair loss values into per-view totals.

    Args:
        per_pair_data: Tuple of (pair_losses, pair_row_indices) where pair_row_indices
            are the row indices from ``torch.where`` for each pair.
        num_views: Number of views (B).
        samples_per_img: Number of samples per image (R).

    Returns:
        Per-view loss sums of shape ``[num_views]``.
    """
    pair_losses, pair_row_indices = per_pair_data
    view_ids = pair_row_indices // samples_per_img
    per_view = []
    for v in range(num_views):
        v_mask = view_ids == v
        per_view.append(pair_losses[v_mask].to(torch.float32).sum())
    return torch.stack(per_view)


def calculate_loss(
    model: GARfVDBModel,
    enc_feats: torch.Tensor,
    input: dict[str, torch.Tensor],
    return_loss_images: bool = False,
) -> dict[str, torch.Tensor]:
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
            - "mask_ids": Instance mask IDs of shape matching image spatial dims, where
                        each unique ID represents a different instance (-1 for invalid/background)
            - "scales": Scale values for hierarchical feature computation
        return_loss_images (bool, optional): Whether to compute and return loss visualization
            images in addition to the total loss. Since this is memory intensive, we use float16
            when computing loss across the whole image for memory efficiency.
            Defaults to False.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - "total_loss": Normalized total contrastive loss value (scalar).
            - "instance_loss_1_img": (if return_loss_images=True) Visualization of loss 1
              per pixel/point, normalized to [0, 1].
            - "instance_loss_2_img": (if return_loss_images=True) Visualization of loss 2
              per pixel/point, normalized to [0, 1].
            - "instance_loss_4_img": (if return_loss_images=True) Visualization of loss 4
              per pixel/point, normalized to [0, 1].

    Loss Components:
        1. **Instance Loss 1**: Pulls together features of points with the same mask ID
           at the same scale. Computed as L2 distance between positive pairs.

        2. **Instance Loss 2**: Hierarchical consistency loss that also pulls together
           features of points with the same mask ID but at larger scales to enforce
           multi-scale consistency.

        4. **Instance Loss 4**: Pushes apart features of points with different mask IDs
           using a margin-based hinge loss (ReLU(margin - distance)).

    Implementation Details:
        - Uses block masking to prevent cross-image comparisons in batched inputs.
        - Only considers upper triangular pairs to avoid double-counting.
        - Excludes diagonal pairs (same point with itself) from grouping supervision.
        - Filters out invalid pairs where mask_ids == -1.
        - Loss visualization images are computed by scattering loss values back to
          their corresponding spatial locations in the input image.
    """

    return_loss_dict = {}

    # Accommodate 'image' inputs of shape [B, num_samples, C] and [B, H, W, C]
    samples_per_img = math.prod(input["image"].shape[1:-1])

    num_chunks = enc_feats.shape[0]

    input_id1 = input_id2 = input["mask_ids"].flatten()

    logging.debug(
        f"calc_loss shapes: enc_feats={enc_feats.shape}, input_id1={input_id1.shape}, mask_ids={input['mask_ids'].shape}"
    )

    # Expand labels for pairwise comparison
    labels1_expanded = input_id1.unsqueeze(1).expand(-1, input_id1.shape[0])
    labels2_expanded = input_id2.unsqueeze(0).expand(input_id2.shape[0], -1)

    # Masks for positive/negative pairs across the entire matrix
    mask_full_positive = labels1_expanded == labels2_expanded
    mask_full_negative = ~mask_full_positive

    logging.debug(
        f"num_chunks = {num_chunks}, input_id1.shape[0] = {input_id1.shape[0]}, samples_per_img = {samples_per_img}"
    )

    # Block-diagonal matrix to consider only pairs within the same image (no cross-image pairs)
    block_mask = torch.kron(
        torch.eye(num_chunks, device=labels1_expanded.device, dtype=torch.bool),
        torch.ones(
            (samples_per_img, samples_per_img),
            device=labels1_expanded.device,
            dtype=torch.bool,
        ),
    )

    logging.debug(f"block_mask.shape = {block_mask.shape}")

    # Only consider upper triangle to avoid double-counting
    block_mask = torch.triu(block_mask, diagonal=0)
    # Only consider pairs where both points are valid (-1 indicates invalid/no mask)
    block_mask = block_mask * (labels1_expanded != -1) * (labels2_expanded != -1)

    # Exclude diagonal elements (same-point pairs are trivially similar)
    diag_mask = torch.eye(block_mask.shape[0], device=block_mask.device, dtype=torch.bool)

    scales = input["scales"]

    # --- Grouping supervision ---
    # Get instance features: [batch_size, samples_per_img, feat_dim]
    instance_features = model.get_mlp_output(enc_feats, scales)

    # Flatten for masking: [batch_size * samples_per_img, feat_dim]
    instance_features_flat = instance_features.reshape(-1, instance_features.shape[-1])

    # 1. If (A, s_A) and (A', s_A) are in the same group, supervise the features to be similar.
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
    return_loss_dict["instance_loss_1_per_pair"] = (instance_loss_1, mask[0])
    logging.debug(f"Loss 1: {instance_loss_1_sum.item()}, using {mask[0].shape[0]} pairs")
    del instance_loss_1_sum

    if return_loss_images:
        with torch.no_grad():
            loss_1_img = torch.zeros(input["image"].shape[:-1], device=instance_loss_1.device)
            # Accumulate loss values at pixel locations
            loss_1_img.view(-1).scatter_reduce_(0, mask[0], instance_loss_1, reduce="sum", include_self=True)
            # Normalize to [0, 1]
            loss_1_img = loss_1_img / torch.max(loss_1_img)
            return_loss_dict["instance_loss_1_img"] = loss_1_img.detach()
            del loss_1_img
    del instance_loss_1

    # 2. If (A, s_A) and (A', s_A) are in the same group, also supervise them to be similar at s > s_A.
    scale_diff = torch.max(torch.zeros_like(scales), (model.max_grouping_scale - scales))
    larger_scale = scales + scale_diff * torch.rand(size=(1,), device=scales.device)

    # Get larger scale features and flatten.
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
    return_loss_dict["instance_loss_2_per_pair"] = (instance_loss_2, mask[0])
    logging.debug(f"Loss 2: {instance_loss_2_nansum.item()}, using {mask[0].shape[0]} pairs")
    del instance_loss_2_nansum

    if return_loss_images:
        with torch.no_grad():
            loss_2_img = torch.zeros(input["image"].shape[:-1], device=instance_loss_2.device)
            loss_2_img.view(-1).scatter_reduce_(0, mask[0], instance_loss_2, reduce="sum", include_self=True)
            loss_2_img = loss_2_img / loss_2_img.max()
            return_loss_dict["instance_loss_2_img"] = loss_2_img.detach()
            del loss_2_img
    del instance_loss_2

    # 4. Supervise A, B to be dissimilar at scales s_A, s_B respectively.
    mask = torch.where(mask_full_negative * block_mask)

    margin = 1.0

    if mask[0].shape[0] > 1000000:
        instance_loss_4 = F.relu(margin - chunked_norm(instance_features_flat, mask[0], mask[1], chunk_size=10000))
    else:
        instance_loss_4 = F.relu(
            margin - torch.norm(instance_features_flat[mask[0]] - instance_features_flat[mask[1]], p=2, dim=-1)
        )
    instance_loss_4_nansum = instance_loss_4.to(torch.float32).nansum()
    return_loss_dict["instance_loss_4"] = instance_loss_4_nansum
    return_loss_dict["instance_loss_4_per_pair"] = (instance_loss_4, mask[0])
    logging.debug(f"Loss 4: {instance_loss_4_nansum.item()}, using {mask[0].shape[0]} pairs")
    del instance_loss_4_nansum

    if return_loss_images:
        with torch.no_grad():
            loss_4_img = torch.zeros(
                input["image"].shape[:-1], device=instance_loss_4.device, dtype=instance_loss_4.dtype
            )
            loss_4_img.view(-1).scatter_reduce_(0, mask[0], instance_loss_4, reduce="sum", include_self=True)
            loss_4_img = loss_4_img / loss_4_img.max()
            return_loss_dict["instance_loss_4_img"] = loss_4_img.detach()
            del loss_4_img
    del instance_loss_4

    # Per-view loss normalization: compute normalized loss per view, then average.
    # This ensures each view contributes equally regardless of its instance distribution
    # (views with more instances produce more pairs, which would otherwise dominate).
    if num_chunks > 1:
        per_view_pair_counts = torch.zeros(num_chunks, device=block_mask.device, dtype=torch.float32)
        for v in range(num_chunks):
            s = v * samples_per_img
            e = (v + 1) * samples_per_img
            per_view_pair_counts[v] = block_mask[s:e, s:e].sum().float()

        per_view_pair_counts = per_view_pair_counts.clamp(min=1.0)

        per_view_total = torch.zeros(num_chunks, device=block_mask.device, dtype=torch.float32)
        for loss_key in ("instance_loss_1", "instance_loss_2", "instance_loss_4"):
            per_view_total += _sum_loss_per_view(return_loss_dict[loss_key + "_per_pair"], num_chunks, samples_per_img)

        return_loss_dict["total_loss"] = (per_view_total / per_view_pair_counts).mean()
    else:
        pair_count = torch.sum(block_mask).float().clamp(min=1.0)
        return_loss_dict["total_loss"] = (
            return_loss_dict["instance_loss_1"]
            + return_loss_dict["instance_loss_2"]
            + return_loss_dict["instance_loss_4"]
        ) / pair_count

    for key in [k for k in return_loss_dict if k.endswith("_per_pair")]:
        del return_loss_dict[key]

    return return_loss_dict
