# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib

from fvdb import GaussianSplat3d

from fvdb_reality_capture.tools import export_splats_to_usdz


def main(ply_path: pathlib.Path, out_usdz_path: pathlib.Path = pathlib.Path("out.usdz")):
    """
    Convert a PLY file containing Gaussian splats to a USDZ file.

    Args:
        ply_path (pathlib.Path): Path to the input PLY file containing Gaussian splats.
        out_usdz_path (pathlib.Path): Path to the output USDZ file (default is "out.usdz").
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading splats from {ply_path}")
    splats, _ = GaussianSplat3d.from_ply(ply_path)
    logger.info(f"Loaded Gaussian Splat model with {splats.num_gaussians} splats")
    logger.info(f"Exporting to USDZ at {out_usdz_path}")
    export_splats_to_usdz(splats, out_usdz_path)
    logger.info("Export complete")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
