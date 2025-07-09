# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import time

import torch
import tyro
from utils import filter_splat_means, prune_large
from viewer import Viewer

from fvdb import GaussianSplat3d


def main(checkpoint_path: str):
    """
    Visualize a Gaussian Splat 3D model from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file containing the Gaussian Splat 3D model.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    splats = prune_large(checkpoint["splats"])
    splats = filter_splat_means(splats, [0.95, 0.95, 0.95, 0.95, 0.89, 0.999])
    model = GaussianSplat3d.from_state_dict(splats)

    viewer = Viewer()
    viewer.register_gaussian_splat_3d("Visualized Model", model)

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000000)  # Keep the viewer running


if __name__ == "__main__":
    tyro.cli(main)
