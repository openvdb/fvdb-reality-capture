.. code-block:: text

   usage: frgs mesh-dlnr [-h] [MESH-DLNR OPTIONS] PATH FLOAT

   Extract a mesh from a saved Gaussian splat file using TSDF fusion and depth maps estimated using
   the DLNR model.
   1. First, it renders stereo pairs of images from the Gaussian splat radiance field, and uses
      DLNR to compute depth maps from these stereo pairs in the frame of the first image in the pair.
      The result is a set of depth maps aligned with the rendered images.

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
      frgs mesh-dlnr model.pt 0.05 --output-path mesh.ply

      # Extract a mesh from a Gaussian splat model saved in `model.ply` with a truncation margin of
   0.1
      # with a grid shell thickness of 5 voxels
      frgs mesh-dlnr model.ply 0.1 --output-path mesh.ply --grid-shell-thickness 5.0

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
   │ -b FLOAT, --baseline FLOAT                                                                     │
   │                         Baseline distance (as a fraction of the mean depth of each image) used │
   │                         for generating stereo pairs as input to the DLNR model (default is     │
   │                         0.07). (default: 0.07)                                                 │
   │ -n FLOAT, --near FLOAT  Near plane distance (as a multiple of the baseline) for which depth    │
   │                         values are considered valid. (default: 4.0)                            │
   │ -f FLOAT, --far FLOAT   Far plane distance (as a multiple of the baseline) for which depth     │
   │                         values are considered valid. (default: 20.0)                           │
   │ -at FLOAT, --alpha-threshold FLOAT                                                             │
   │                         Alpha threshold to mask pixels where the Gaussian splat model is       │
   │                         transparent, usually indicating the background. (default is 0.1).      │
   │                         (default: 0.1)                                                         │
   │ -dt FLOAT, --disparity-reprojection-threshold FLOAT                                            │
   │                         Reprojection error threshold for occlusion masking in pixels (default  │
   │                         is 3.0). (default: 3.0)                                                │
   │ -idf INT, --image-downsample-factor INT                                                        │
   │                         Factor by which to downsample the rendered images for depth estimation │
   │                         (default is 1, _i.e._ no downsampling). (default: 1)                   │
   │ -db STR, --dlnr-backbone STR                                                                   │
   │                         Backbone to use for the DLNR model, either "middleburry" or            │
   │                         "sceneflow" (default is "middleburry"). (default: middleburry)         │
   │ -ab, --use-absolute-baseline, --no-use-absolute-baseline                                       │
   │                         If True, use the provided baseline as an absolute distance in world    │
   │                         units (default is False). (default: False)                             │
   │ -o PATH, --output-path PATH                                                                    │
   │                         Path to save the extracted mesh (default is "mesh.ply"). (default:     │
   │                         mesh.ply)                                                              │
   │ -d STR, --device STR    Extract a mesh from a Gaussian Splat reconstruction. (default: cuda)   │
   ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

