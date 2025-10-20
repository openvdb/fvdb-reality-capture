.. code-block:: text

    usage: frgs points [-h] [POINTS OPTIONS] PATH

    Extract a point cloud with colors/features from a Gaussian splat file by unprojecting depth images
    rendered from it. This algorithm can optionally filter out points near depth discontinuities using
    the following heurstic:
        1. Apply a small Gaussian filter to the depth images to reduce noise.
        2. Run a Canny edge detector on the depth immage to find
        depth discontinuities. The result is an image mask where pixels near depth edges are marked.
        3. Dilate the edge mask to remove depth samples near edges.
        4. Remove points from the point cloud where the corresponding depth pixel is marked in the
    dilated edge mask.

    Example usage:

        # Extract a point cloud from a Gaussian splat model saved in `model.pt`
        frgs points model.pt --output-path points.ply

        # Extract a point cloud from a Gaussian splat model saved in `model.ply`
        frgs points model.ply --output-path points.ply

    ╭─ positional arguments ─────────────────────────────────────────────────────────────────────────╮
    │ PATH                    input-path (required)                                                  │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ options ──────────────────────────────────────────────────────────────────────────────────────╮
    │ -h, --help              show this help message and exit                                        │
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
    │ -ces FLOAT, --canny-edge-std FLOAT                                                             │
    │                         Standard deviation for the Gaussian filter applied to the depth image  │
    │                         before Canny edge detection (default is 1.0). Set to 0.0 to disable    │
    │                         canny edge filtering. (default: 1.0)                                   │
    │ -cmd INT, --canny-mask-dilation INT                                                            │
    │                         Dilation size for the Canny edge mask (default is 5). (default: 5)     │
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
    │                         points.ply)                                                            │
    │ -d STR, --device STR    Device to use for computation (default is "cuda"). (default: cuda)     │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

