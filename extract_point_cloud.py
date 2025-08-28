# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib

import point_cloud_utils as pcu
import torch
import tyro
from fvdb_3dgs.tools import point_cloud_from_splats

from fvdb import GaussianSplat3d


def main(
    ply_path: pathlib.Path,
    near: float = 0.1,
    far: float = 4.0,
    use_scene_scale_units: bool = True,
    depth_image_downsample_factor: int = 8,
    output_path: pathlib.Path = pathlib.Path("point_cloud.ply"),
    device: str = "cuda",
):
    """
    Extract a mesh from a saved checkpoint file.

    Args:
        ply_path (pathlib.Path): Path to the PLY containing the Gaussian splat model and training camera metadata.
        near (float): Near plane distance (as a multiple of the scene_scale) below which we'll ignore depth samples (default is 0.1).
        far (float): Far plane distance (as a multiple of the scene_scale) above which we'll ignore depth samples.
        use_scene_scale_units (bool): Whether to use scene scale units for the near, plane, far plane and truncation margin.
        output_path (pathlib.Path): Path to save the extracted mesh (default is "mesh.ply").
        device (str): Device to use for computation (default is "cuda").
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    logger = logging.getLogger("extract_point_cloud")

    logger.info(f"Loading PLY from {ply_path}")

    model, metadata = GaussianSplat3d.from_ply(ply_path)

    if "camera_to_world_matrices" not in metadata:
        raise ValueError("PLY file must contain 'camera_to_world_matrices'")
    assert isinstance(metadata["camera_to_world_matrices"], torch.Tensor)
    camera_to_world_matrices = metadata["camera_to_world_matrices"].to(device)

    if "projection_matrices" not in metadata:
        raise ValueError("PLY file must contain 'projection_matrices'")
    assert isinstance(metadata["projection_matrices"], torch.Tensor)
    projection_matrices = metadata["projection_matrices"].to(device)

    if "image_sizes" not in metadata:
        raise ValueError("PLY file must contain 'image_sizes'")
    assert isinstance(metadata["image_sizes"], torch.Tensor)
    image_sizes = metadata["image_sizes"]

    model = model.to(device)

    if use_scene_scale_units:
        if "scene_scale" not in metadata:
            raise ValueError("PLY file must contain 'scene_scale'")
        assert isinstance(metadata["scene_scale"], float)
        far = far * metadata["scene_scale"]
        near = near * metadata["scene_scale"]

    logger.info(
        f"Extracting point cloud from checkpoint using near={near:0.3f}, far={far:0.3f}, downsample factor={depth_image_downsample_factor}"
    )
    positions, colors = point_cloud_from_splats(
        model=model,
        camera_to_world_matrices=camera_to_world_matrices,
        projection_matrices=projection_matrices,
        image_sizes=image_sizes,
        near=near,
        far=far,  # Use the scene scale from the training dataset
        depth_image_downsample_factor=depth_image_downsample_factor,
        show_progress=True,
    )

    logger.info(f"Extracted {positions.shape[0]} points with colors.")
    positions, colors = positions.to(torch.float32).cpu().numpy(), colors.to(torch.float32).cpu().numpy()

    logger.info(f"Saving point cloud to {output_path}")
    pcu.save_mesh_vc(str(output_path), positions, colors)
    logger.info("Point cloud saved successfully.")


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
