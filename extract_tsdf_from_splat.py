# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib

import point_cloud_utils as pcu
import torch
import torch.utils.data
import tqdm
import tyro
from datasets import SfmDataset

from fvdb import GaussianSplat3d, JaggedTensor, gridbatch_from_dense


@torch.inference_mode()
def make_mesh_from_splat(
    model: GaussianSplat3d,
    dataset: SfmDataset,
    voxel_size: float,
    voxel_trunc_margin: float = 2.0,
    near_plane: float = 0.1,
    far_plane: float = 1e10,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Generate a mesh using TSDF fusion from depth maps rendered from a Gaussian splat.
    The TSDF fusion algorithm is based on the paper:
    "KinectFusion: Real-Time Dense Surface Mapping and Tracking" (https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/)

    Args:
        model: GaussianSplat3d model.
        dataset: A ColmapDataset.
        voxel_size: Size of the voxels in the TSDF grid.
        voxel_trunc_margin: Margin for truncating the TSDF. Units are in voxels.
        near_plane: Near plane distance for the depth images.
        far_plane: Far plane distance for the depth images.
        device: Device to use.
        dtype: Data type for the TSDF and weights. Default is torch.float16.

    Returns:
        mesh_vertices: A [num_vertices, 3] shaped tensor of mesh vertices.
        mesh_faces: A [num_faces, 3] shaped tensor of triangle mesh indices into the vertices.
        mesh_colors: A [num_vertices, 3] shaped tensor of RGB colors for each vertex.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    trunc_margin = voxel_size * voxel_trunc_margin
    accum_grid = gridbatch_from_dense(1, [1, 1, 1], voxel_sizes=voxel_size, device=device)  # type: ignore
    weights = JaggedTensor([torch.zeros(accum_grid.total_voxels, device=device, dtype=dtype)])
    tsdf = JaggedTensor([torch.zeros(accum_grid.total_voxels, device=device, dtype=dtype)])
    colors = JaggedTensor([torch.zeros(accum_grid.total_voxels, 3, device=device, dtype=torch.uint8)])

    pbar = tqdm.tqdm(dataloader, desc="Creating mesh from splat")
    for i, data in enumerate(pbar):
        img = data["image"].squeeze()
        projection_mats = data["K"].to(device)
        cam_to_world_mats = data["camtoworld"].to(device)
        world_to_cam = torch.linalg.inv(cam_to_world_mats).contiguous()

        # We set near and far planes to 0.0 and 1e10 respectively to avoid clipping
        # in the rendering process. Instead, we will use the provided near and far planes
        # to filter the depth images after rendering so pixels out of range will not be integrated
        # into the TSDF.
        rgbd, alphas = model.render_images_and_depths(
            world_to_cam, projection_mats, img.shape[1], img.shape[0], near=0.0, far=1e10
        )
        rgb_images = (rgbd[..., :3].clip_(min=0.0, max=1.0) * 255.0).to(torch.uint8)
        depth_images = (rgbd[..., -1].unsqueeze(-1) / alphas.clamp(min=1e-10)).to(dtype)
        weight_images = ((depth_images > near_plane) & (depth_images < far_plane)).to(dtype)

        accum_grid, tsdf, weights, colors = accum_grid.integrate_tsdf_with_features(
            trunc_margin,
            projection_mats.to(dtype),
            cam_to_world_mats.to(dtype),
            tsdf,
            colors,
            weights,
            depth_images,
            rgb_images,
            weight_images.squeeze(-1),
        )
        pbar.set_postfix({"accumulated_voxels": accum_grid.total_voxels})

        # TSDF fusion is a bit of a torture case for the PyTorch memory allocator since
        # it progressively allocates bigger tensors which don't fit in the memory pool,
        # causing the pool to grow larger and larger.
        # To avoid this, we synchronize the CUDA device and empty the cache after each image.
        del rgb_images, depth_images, weight_images
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # After integrating all the images, we prune the grid to remove empty voxels which have no weights.
    # This is done to reduce the size of the grid and speed up the marching cubes algorithm
    # which will be used to extract the mesh.
    new_grid = accum_grid.pruned_grid(weights > 0.0)
    filter_tsdf = new_grid.jagged_like(torch.zeros(new_grid.total_voxels, device=new_grid.device, dtype=dtype))
    filter_colors = new_grid.jagged_like(
        torch.zeros(new_grid.total_voxels, 3, device=new_grid.device, dtype=torch.uint8)
    )
    new_grid.inject_from(accum_grid, tsdf, filter_tsdf)
    new_grid.inject_from(accum_grid, colors, filter_colors)
    # print("NEW GRID VOXELS", new_grid.total_voxels)

    mesh_vertices, mesh_faces, _ = new_grid.marching_cubes(filter_tsdf, 0.0)
    mesh_vertices = mesh_vertices.jdata
    mesh_faces = mesh_faces.jdata
    mesh_colors = new_grid.sample_trilinear(mesh_vertices, filter_colors.to(dtype)).jdata / 255.0
    mesh_colors.clip_(min=0.0, max=1.0)

    return mesh_vertices.cpu().numpy(), mesh_faces.cpu().numpy(), mesh_colors.cpu().numpy()


def main(
    checkpoint_path: str,
    data_path: str,
    voxel_size: float,
    voxel_trunc_margin: float = 2.0,
    near_plane: float = 0.1,
    far_plane: float = 1e10,
    normalization_type: str = "pca",
    output_path: str = "mesh.ply",
    image_downsample_factor: int = 4,
    device: str = "cuda",
):
    """
    Main function of the script. This script generates a colored triangle mesh from a GaussianSplat3d model by
    rendering a depth and color image from each camera a dataset, and integrating them into a TSDF
    using the TSDF fusion algorithm.

    Args:
        checkpoint_path: Path to the GaussianSplat3d checkpoint.
        data_path: Path to the dataset.
        voxel_size: Size of the voxels in the TSDF grid.
        voxel_trunc_margin: Margin for truncating the TSDF. Units are in voxels.
        near_plane: Near plane distance for the depth images.
        far_plane: Far plane distance for the depth images.
        normalization_type: Normalization type for the dataset. Options are "pca", "similarity", "ecef2enu", or "none".
        image_downsample_factor: Downsample factor for the depth images.
        device: Device to use

    Returns:
        None
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GaussianSplat3d.from_state_dict(checkpoint["splats"])

    dataset = SfmDataset(
        pathlib.Path(data_path),
        split="all",
        image_downsample_factor=image_downsample_factor,
        normalization_type=normalization_type,
    )

    v, f, c = make_mesh_from_splat(
        model,
        dataset,
        voxel_size,
        voxel_trunc_margin=voxel_trunc_margin,
        near_plane=near_plane,
        far_plane=far_plane,
        device=device,
    )

    pcu.save_mesh_vfc(output_path, v, f, c)


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
