.. code-block:: text

   usage: frgs mesh-basic [-h] [MESH-BASIC OPTIONS] PATH FLOAT

   Extract a triangle mesh from a saved Gaussian splat file with TSDF fusion using depth maps
   rendered from the Gaussian splat model. The algorithm proceeds in three steps:

   1. First, it renders depth and color/feature images from the Gaussian splat radiance field at each
   of the specified
      camera views.

   2. Second, it integrates the depths and colors/features into a sparse fvdb.Grid in a narrow band
      around the surface using sparse truncated signed distance field (TSDF) fusion.
      The result is a sparse voxel grid representation of the scene where each voxel stores a signed
   distance
      value and color (or other features).

   3. Third, it extracts a mesh using the sparse marching cubes algorithm implemented in
   fvdb.Grid.marching_cubes
      over the Grid and TSDF values. This step produces a triangle mesh with vertex colors sampled
   from the
      colors/features stored in the Grid.

   Example usage:

      # Extract a mesh from a Gaussian splat model saved in `model.pt` with a truncation margin of
   0.05
      frgs mesh-basic model.pt 0.05 --output-path mesh.ply

      # Extract a mesh from a Gaussian splat model saved in `model.ply` with a truncation margin of
   0.1
      # with a grid shell thickness of 5 voxels, near plane at 0.1x median depth, far plane at 2.0x
   median depth
      # of each images.
      frgs mesh-basic model.ply 0.1 --output-path mesh.ply --grid-shell-thickness 5.0 --near 0.1
   --far 2.0

   ╭─ positional arguments ─────────────────────────────────────────────────────────────────────────╮
   │ PATH                    Path to the input PLY or checkpoint file. Must end in .ply, .pt, or    │
   │                         .pth. (required)                                                       │
   │ FLOAT                   Truncation margin for TSDF volume. This is the distance (in world      │
   │                         units) that the TSDF values are truncated to. (required)               │
   ╰────────────────────────────────────────────────────────────────────────────────────────────────╯
   ╭─ options ──────────────────────────────────────────────────────────────────────────────────────╮
   │ -h, --help              show this help message and exit                                        │
   │ -g FLOAT, --grid-shell-thickness FLOAT                                                         │
   │                         The number of voxels along each axis to include in the TSDF volume.    │
   │                         This defines the resolution of the narrow band around the surface.     │
   │                         (default: 3.0)                                                         │
   │ -n FLOAT, --near FLOAT  Near plane distance for which depth values are considered valid. The   │
   │                         units depend on the `near_far_units` parameter. By default, this is a  │
   │                         multiple of the median depth of each image. (default: 0.2)             │
   │ -f FLOAT, --far FLOAT   Far plane distance for which depth values are considered valid. The    │
   │                         units depend on the `near_far_units` parameter. By default, this is a  │
   │                         multiple of the median depth of each image. (default: 1.5)             │
   │ -at FLOAT, --alpha-threshold FLOAT                                                             │
   │                         Alpha threshold to mask pixels where the Gaussian splat model is       │
   │                         transparent, usually indicating the background. (default is 0.1).      │
   │                         (default: 0.1)                                                         │
   │ -idf INT, --image-downsample-factor INT                                                        │
   │                         Factor by which to downsample the rendered images for depth estimation │
   │                         (default is 1, _i.e._ no downsampling). (default: 1)                   │
   │ -nfu {absolute,camera_extent,median_depth}, --near-far-units                                   │
   │ {absolute,camera_extent,median_depth}                                                          │
   │                         Which units to use for near and far clipping.                          │
   │                         - "absolute" means the near and far values are in world units.         │
   │                         - "camera_extent" means the near and far values are fractions of the   │
   │                         maximum distance from any camera to the centroid of all cameras (this  │
   │                         is good for orbit captures).                                           │
   │                         - "median_depth" means the near and far values are fractions of the    │
   │                         median depth of each image. This is good for captures where the        │
   │                         cameras are not evenly distributed around the scene.                   │
   │                         (default is "median_depth"). (default: median_depth)                   │
   │ -o PATH, --output-path PATH                                                                    │
   │                         Path to save the extracted mesh (default is "mesh.ply"). (default:     │
   │                         mesh.ply)                                                              │
   │ -d STR, --device STR    Extract a mesh from a Gaussian Splat reconstruction. (default: cuda)   │
   ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

