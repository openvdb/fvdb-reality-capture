# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time

import torch
import tyro
from fvdb import GaussianSplat3d

from fvdb_reality_capture.viewer import Viewer


def main(
    ply_path: pathlib.Path,
    viewer_port: int = 8080,
    verbose: bool = False,
    device: str | torch.device = "cuda",
):
    """
    Visualize a scene in a saved checkpoint file.

    Args:
        ply_path (pathlib.Path): Path to a PLY file containing the Gaussian splat model.
        viewer_port (int): The port to expose the viewer server on
        verbose (bool): If True, then the viewer will log verbosely.
        device (str | torch.device): Device to use for computation (default is "cuda").
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    viewer = Viewer(port=viewer_port, verbose=verbose)

    model, metadata = GaussianSplat3d.from_ply(ply_path, device)

    bbmin = torch.min(model.means, dim=0).values
    bbmax = torch.max(model.means, dim=0).values
    bbdiagonal = torch.norm(bbmax - bbmin).item()
    viewer.camera_far = 4 * bbdiagonal

    splat_view = viewer.register_gaussian_splat_3d(name="model", gaussian_scene=model)

    logger = logging.getLogger("visualize")
    logger.info("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
