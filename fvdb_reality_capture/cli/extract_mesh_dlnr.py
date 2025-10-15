# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib

import point_cloud_utils as pcu
import torch
import tyro
from fvdb import GaussianSplat3d

from fvdb_reality_capture.tools import mesh_from_splats_dlnr


def main(
    ply_path: pathlib.Path,
    truncation_margin: float,
    grid_shell_thickness: float = 3.0,
    baseline: float = 0.07,
    near: float = 4.0,
    far: float = 20.0,
    disparity_reprojection_threshold: float = 3.0,
    alpha_threshold: float = 0.1,
    image_downsample_factor: int = 1,
    dlnr_backbone: str = "middleburry",
    output_path: pathlib.Path = pathlib.Path("mesh.ply"),
    use_absolute_baseline: bool = False,
    device: str = "cuda",
):
    """
    Extract a mesh from a saved checkpoint file using TSDF fusion and depth maps estimated using the DLNR model.

    The algorithm renders a stereo pair of images from the Gaussian splat model by adding a small baseline to each camera position.
    (specified by the baseline parameter). It then uses DLNR to estimate depth maps for these stereo pairs which are fed to TSDF fusion.

    The mesh extraction algorithm is based on the paper:
    "GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views"
    (https://arxiv.org/abs/2404.01810)

    Args:
        ply_path (pathlib.Path): Path to the PLY containing the Gaussian splat model and training camera metadata.
        truncation_margin (float): Margin for truncating the mesh, in world units.
        grid_shell_thickness (float): Thickness of the TSDF grid shell in multiples of the truncation margin (default is 3.0).
            _i.e_. if truncation_margin is 0.1 and grid_shell_thickness is 3.0, the TSDF grid will extend 0.3 world units
            from the surface of the model.
        baseline (float): Baseline distance (as a fraction of the mean depth of each image) used
            for generating stereo pairs as input to the DLNR model (default is 0.07).
        near (float): Near plane distance (as a multiple of the baseline) below which we'll ignore depth samples (default is 4.0).
        far (float): Far plane distance (as a multiple of the baseline) above which we'll ignore depth samples (default is 20.0).
        disparity_reprojection_threshold (float): Reprojection error threshold for occlusion masking in pixels (default is 3.0).
        alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model is transparent,
            usually indicating the background. (default is 0.1).
        image_downsample_factor (int): Factor by which to downsample the rendered images for
            depth estimation (default is 1, _i.e._ no downsampling).
        dlnr_backbone (str): Backbone to use for the DLNR model, either "middleburry" or "sceneflow" (default is "middleburry").
        output_path (pathlib.Path): Path to save the extracted mesh (default is "mesh.ply").
        use_absolute_baseline (bool): If True, use the provided baseline as an absolute distance in world units (default is False).
        device (str): Device to use for computation (default is "cuda").
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    logger = logging.getLogger("extract_mesh_dlnr")

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

    if use_absolute_baseline:
        logger.info(f"Extracting mesh using DLNR stereo matching with a baseline of {baseline} world units")
    else:
        logger.info(
            f"Extracting mesh using DLNR stereo matching with a baseline of {baseline * 100:.1f}% of the mean depth of each image"
        )
    v, f, c = mesh_from_splats_dlnr(
        model=model,
        camera_to_world_matrices=camera_to_world_matrices,
        projection_matrices=projection_matrices,
        image_sizes=image_sizes,
        truncation_margin=truncation_margin,
        grid_shell_thickness=grid_shell_thickness,
        baseline=baseline,
        near=near,
        far=far,
        disparity_reprojection_threshold=disparity_reprojection_threshold,
        alpha_threshold=alpha_threshold,
        image_downsample_factor=image_downsample_factor,
        dlnr_backbone=dlnr_backbone,
        use_absolute_baseline=use_absolute_baseline,
        show_progress=True,
    )

    v, f, c = v.to(torch.float32).cpu().numpy(), f.cpu().numpy(), c.to(torch.float32).cpu().numpy()
    logger.info(f"Extracted mesh with {v.shape[0]} vertices and {f.shape[0]} faces.")

    logger.info(f"Saving mesh to {output_path}")
    pcu.save_mesh_vfc(str(output_path), v, f, c)
    logger.info("Mesh saved successfully.")


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
