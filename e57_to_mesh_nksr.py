import pathlib

import nksr
import numpy as np
import point_cloud_utils as pcu
import torch
import tqdm
from fvdb import Grid

from fvdb_3dgs.training.load_e57 import load_scan_for_nksr

dataset_path = pathlib.Path('/home/bbartlett/Data1/nuRec/hexagon/20042020-kantonalschule-_Setip_001.e57')
truncation_margin = 0.05

points, points_rgb, location = load_scan_for_nksr(dataset_path, downsample_point_factor=10)

location_per_point = np.repeat(location,points.shape[0],axis=0)


view_dirs = location_per_point-points

print("here")
points = np.ascontiguousarray(points)
print("here 1")
_, normals = pcu.estimate_point_cloud_normals_knn(points,8,view_directions=view_dirs)
print("here 2")
pcu.save_mesh_vn('mesh_norms.ply',points, normals)
print("save normals mesh")

device = "cuda"



# points_neg = points - normals * truncation_margin/2.0
# points_pos = points + normals * truncation_margin/2.0


# values = torch.cat([
#     torch.ones(points_pos.shape[0], device=device),
#     -torch.ones(points_neg.shape[0], device=device),
# ])

# points_all = torch.from_numpy(np.concatenate([points_pos, points_neg], axis=0).astype(np.float32)).to(device)


# grid = Grid.from_points(points_all, voxel_size=truncation_margin)

# splat_vals = grid.splat_trilinear(points_all, values.unsqueeze(-1))
# v, f, n = grid.marching_cubes(splat_vals, 0.0)

# pcu.save_mesh_vf("mesh_tsdf.ply", v.cpu().numpy(), f.cpu().numpy())


reconstructor = nksr.Reconstructor(device)

points = torch.from_numpy(points.astype(np.float32)).to(device)
points_rgb = torch.from_numpy(points_rgb.astype(np.float32)).to(device)
normals = torch.from_numpy(normals.astype(np.float32)).to(device)

field = reconstructor.reconstruct(points, normal=normals, detail_level=1, voxel_size=0.005)
field.set_texture_field(nksr.fields.PCNNField(points, points_rgb))
mesh = field.extract_dual_mesh(mise_iter=1)

pcu.save_mesh_vfc("mesh_nksr.ply", mesh.v.cpu().numpy(), mesh.f.cpu().numpy(), mesh.c.cpu().numpy()/255.)

print("done")





