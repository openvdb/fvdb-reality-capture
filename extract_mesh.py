# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib

import point_cloud_utils as pcu
import torch
import tyro
from fvdb import GaussianSplat3d

from fvdb_reality_capture.tools import mesh_from_splats


def main(
    ply_path: pathlib.Path,
    truncation_margin: float,
    grid_shell_thickness: float = 3.0,
    near: float = 0.1,
    far: float = 4.0,
    use_scene_scale_units: bool = True,
    alpha_threshold: float = 0.1,
    image_downsample_factor: int = 1,
    output_path: pathlib.Path = pathlib.Path("mesh.ply"),
    device: str = "cuda",
):
    """
    Extract a mesh from a saved checkpoint file.

    Args:
        ply_path (pathlib.Path): Path to the PLY containing the Gaussian splat model and training camera metadata.
        truncation_margin (float): Margin for truncating the mesh, in world units.
        grid_shell_thickness (float): Thickness of the TSDF grid shell in multiples of the truncation margin (default is 3.0).
            _i.e_. if truncation_margin is 0.1 and grid_shell_thickness is 3.0, the TSDF grid will extend 0.3 world units
            from the surface of the model.
        near (float): Near plane distance (as a multiple of the scene scale) below which we'll ignore depth samples (default is 0.1).
        far (float): Far plane distance (as a multiple of the scene scale) above which we'll ignore depth samples.
        alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model is transparent,
            usually indicating the background. (default is 0.1).
        image_downsample_factor (int): Factor by which to downsample the rendered images for
            depth estimation (default is 1, _i.e._ no downsampling).
        use_scene_scale_units (bool): Whether to use scene scale units for the near, plane, far plane and truncation margin.
        output_path (pathlib.Path): Path to save the extracted mesh (default is "mesh.ply").
        device (str): Device to use for computation (default is "cuda").
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    logger = logging.getLogger("extract_mesh")

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
        f"Extracting mesh from checkpoint using near={near:0.3f}, far={far:0.3f}, and truncation margin={truncation_margin:0.3f}"
    )

    v, f, c = mesh_from_splats(
        model=model,
        camera_to_world_matrices=camera_to_world_matrices,
        projection_matrices=projection_matrices,
        image_sizes=image_sizes,
        truncation_margin=truncation_margin,
        grid_shell_thickness=grid_shell_thickness,
        near=near,
        far=far,
        alpha_threshold=alpha_threshold,
        image_downsample_factor=image_downsample_factor,
        show_progress=True,
    )

    logger.info(f"Extracted mesh with {v.shape[0]} vertices and {f.shape[0]} faces.")

    v, f, c = v.to(torch.float32).cpu().numpy(), f.cpu().numpy(), c.to(torch.float32).cpu().numpy()

    logger.info(f"Saving mesh to {output_path}")
    pcu.save_mesh_vfc(str(output_path), v, f, c)
    logger.info("Mesh saved successfully.")


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
