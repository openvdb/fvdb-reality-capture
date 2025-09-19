import pathlib

import numpy as np
import point_cloud_utils as pcu
import torch
import tqdm
from fvdb import Grid

from fvdb_3dgs import SfmScene
from fvdb_3dgs.training.load_e57 import load_e57_scene

# scene = SfmScene.from_e57("../data/hexagon_data/")

dataset_path = pathlib.Path('/home/bbartlett/Data1/nuRec/hexagon/')
scene: SfmScene = load_e57_scene(dataset_path, downsample_point_factor=1)

device = "cuda"
trunc_margin = 0.1
vox_factor = 20
points = torch.from_numpy(scene.points.astype(np.float32)).to(device)


splat_pts = []
splat_vals = []
for i in tqdm.tqdm(range(len(scene.images))):
    image_metadata = scene.images[i]
    camera_metadata = image_metadata.camera_metadata
    cam_to_world_matrix = (
        torch.from_numpy(image_metadata.camera_to_world_matrix).to(device).to(dtype=torch.float32, device=device)
    )
    world_to_cam_matrix = torch.linalg.inv(cam_to_world_matrix).contiguous().to(dtype=torch.float32, device=device)
    projection_matrix = (
        torch.from_numpy(camera_metadata.projection_matrix).to(device).to(dtype=torch.float32, device=device)
    )
    visible_points = torch.from_numpy(scene.points[image_metadata.point_indices]).to(dtype=torch.float32, device=device)
    visible_points_rgb = torch.from_numpy(scene.points_rgb[image_metadata.point_indices]).to(dtype=torch.float32, device=device)
    if visible_points.shape[0] == 0:
        continue
    points_cam = world_to_cam_matrix[:3, :3] @ visible_points.t() + world_to_cam_matrix[:3, 3:4]
    points_cam_neg = points_cam.clone()
    points_cam_neg[2, :] -= trunc_margin / vox_factor
    points_cam_neg_2 = points_cam.clone()
    points_cam_neg_2[2, :] -= trunc_margin / 1.0

    points_cam_pos = points_cam.clone()
    points_cam_pos[2, :] += trunc_margin / vox_factor
    points_cam_pos_2 = points_cam.clone()
    points_cam_pos_2[2, :] += trunc_margin / 1.0

    all_points_cam = torch.cat([points_cam, points_cam_neg, points_cam_pos, points_cam_neg_2, points_cam_pos_2], dim=1)
    points_world = (cam_to_world_matrix[:3, :3] @ all_points_cam + cam_to_world_matrix[:3, 3:4]).t()

    values = torch.cat([
        torch.zeros(points_cam.shape[1], device=device),
        -torch.ones(points_cam_neg.shape[1], device=device) * (trunc_margin / vox_factor),
        torch.ones(points_cam_pos.shape[1], device=device) * (trunc_margin / vox_factor),
        -torch.ones(points_cam_neg_2.shape[1], device=device) * trunc_margin,
        torch.ones(points_cam_pos_2.shape[1], device=device) * trunc_margin,
    ])

    splat_pts.append(points_world)
    splat_vals.append(values)

splat_pts = torch.cat(splat_pts, dim=0)
splat_vals = torch.cat(splat_vals, dim=0)

print(splat_pts.shape, splat_vals.shape)

# voxel_size = trunc_margin / voxel_factor

grid = Grid.from_points(splat_pts, voxel_size=trunc_margin / vox_factor)
splat_vals = grid.splat_trilinear(splat_pts, splat_vals.unsqueeze(-1))
v, f, n = grid.marching_cubes(splat_vals, 0.0)
print(v.shape, f.shape)
pcu.save_mesh_vf("mesh.ply", v.cpu().numpy(), f.cpu().numpy())
