# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import torch

from fvdb_reality_capture.instance_segmentation.loss import _sum_loss_per_view, calculate_loss


class _IdentityMLP:
    """Stand-in for GARfVDBModel that returns encoder features unchanged at every scale."""

    max_grouping_scale = 1.0

    def get_mlp_output(self, enc_feats: torch.Tensor, scales) -> torch.Tensor:
        return enc_feats


def _reference_total_loss(features: torch.Tensor, mask_ids: torch.Tensor) -> torch.Tensor:
    """Per-view normalized loss computed pair by pair, straight from the definition.

    For each view, sum loss 1 and loss 2 over positive pairs and loss 4 over negative
    pairs, restricted to the upper triangle with both endpoints valid, divide by that
    view's pair count (upper triangle including the diagonal, both endpoints valid),
    then average over views. With the identity MLP loss 2 equals loss 1.
    """
    num_views, samples = mask_ids.shape
    per_view = []
    for v in range(num_views):
        total = torch.zeros(())
        count = 0
        for i in range(samples):
            for j in range(i, samples):
                if mask_ids[v, i] < 0 or mask_ids[v, j] < 0:
                    continue
                count += 1
                dist = torch.norm(features[v, i] - features[v, j])
                if mask_ids[v, i] == mask_ids[v, j]:
                    if i != j:
                        total = total + 2 * dist
                else:
                    total = total + torch.relu(1.0 - dist)
        per_view.append(total / max(count, 1))
    return torch.stack(per_view).mean()


def _make_input(features: torch.Tensor, mask_ids: torch.Tensor) -> dict:
    num_views, samples = mask_ids.shape
    return {
        "image": torch.zeros(num_views, samples, 3),
        "mask_ids": mask_ids,
        "scales": torch.full((num_views, samples), 0.2),
    }


def test_total_loss_single_view_is_sum_over_pair_count():
    torch.manual_seed(0)
    features = torch.randn(1, 6, 4)
    mask_ids = torch.tensor([[0, 0, 1, 1, 2, -1]])

    loss = calculate_loss(_IdentityMLP(), features, _make_input(features, mask_ids))["total_loss"]

    torch.testing.assert_close(loss, _reference_total_loss(features, mask_ids))


def test_total_loss_averages_per_view_normalized_losses():
    torch.manual_seed(0)
    features = torch.randn(3, 6, 4)
    # Views with very different instance and validity structure so that a global
    # sum / global count would differ from the per-view average.
    mask_ids = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5],
            [7, 7, -1, -1, -1, -1],
        ]
    )

    loss = calculate_loss(_IdentityMLP(), features, _make_input(features, mask_ids))["total_loss"]

    torch.testing.assert_close(loss, _reference_total_loss(features, mask_ids))


def test_sum_loss_per_view_matches_loop():
    torch.manual_seed(0)
    num_views, samples = 4, 5
    rows = torch.randint(0, num_views * samples, (40,))
    losses = torch.rand(40)

    summed = _sum_loss_per_view((losses, rows), num_views, samples)

    expected = torch.stack([losses[rows // samples == v].sum() for v in range(num_views)])
    torch.testing.assert_close(summed, expected)
