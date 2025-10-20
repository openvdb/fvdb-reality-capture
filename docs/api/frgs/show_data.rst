.. code-block:: text

    usage: frgs show-data [-h] [SHOW-DATA OPTIONS] PATH

    Visualize a scene in a dataset folder. This will plot the scene's cameras and point cloud in an
    interactive viewer shown in a browser window.

    The dataset folder should either contain a colmap dataset, a set of e57 files, a
    simple_directory dataset:

    COLMAP Data format: A folder should containining:
        - cameras.txt
        - images.txt
        - points3D.txt
        - A folder named "images" containing the image files.
        - An optional "masks" folder with the same layout as images containing masks of which pixels
    are valid.

    E57 format: A folder containing one or more .e57 files.

    Simple Directory format: A folder containing:
        - images/ A directory of images (jpg, png, etc).
        - An optional "masks/" folder with the same layout as images containing masks of which
    pixels are valid.
        - A cameras.json file containing camera intrinsics and extrinsics for each image. It should
    be a list of objects
            with the following format:
                "camera_name": "camera_0000",
                "width": 2048,
                "height": 2048,
                "camera_intrinsics": [], # 3x3 matrix in row-major order
                "world_to_camera": [], # 4x4 matrix in row-major order
                "image_path": "name_of_image_file_relative_to_images_folder"

    Example usage:

        # Visualize a Colmap dataset in the folder ./colmap_dataset
        frgs show-data ./colmap_dataset

        # Visualize an e57 dataset in the folder ./e57_dataset
        frgs show-data ./e57_dataset --dataset-type e57

        # Visualize a simple directory dataset in the folder ./simple_directory_dataset
        frgs show-data ./simple_directory_dataset --dataset-type simple_directory

        # Flip the up axis of the scene from -Z to +Z
        # It's -fu because that's what you say when your scene is backwards.
        frgs show-data ./colmap_dataset -fu

    ╭─ positional arguments ───────────────────────────────────────────────────────────────────────╮
    │ PATH                                                                                         │
    │     Path to the dataset folder. (required)                                                   │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ options ────────────────────────────────────────────────────────────────────────────────────╮
    │ -h, --help                                                                                   │
    │     show this help message and exit                                                          │
    │ -p INT, --viewer-port INT                                                                    │
    │     The port to expose the viewer server on. (default: 8080)                                 │
    │ -ip STR, --viewer-ip-address STR                                                             │
    │     The port to expose the viewer server on. (default: 127.0.0.1)                            │
    │ -v, --verbose, --no-verbose                                                                  │
    │     If True, then the viewer will log verbosely. (default: False)                            │
    │ -ppf FLOAT, --points-percentile-filter FLOAT                                                 │
    │     Percentile filter for points. Points with any coordinate below this percentile or above  │
    │     (100 - this percentile) will be removed from the point cloud. This can help remove       │
    │     outliers. Set to 0.0 to disable. (default: 0.0)                                          │
    │ -mpi INT, --min-points-per-image INT                                                         │
    │     Minimum number of points a camera must observe to be included in the viewer. (default:   │
    │     5)                                                                                       │
    │ -dt {colmap,simple_directory,e57}, --dataset-type {colmap,simple_directory,e57}              │
    │     Type of dataset to load. (default: colmap)                                               │
    │ -al FLOAT, --axis-length FLOAT                                                               │
    │     The length (in world units) of the axes drawn at each camera and at the origin.          │
    │     (default: 1.0)                                                                           │
    │ -fl FLOAT, --frustum-length FLOAT                                                            │
    │     Frustum length from the origin to the view plane in world units. (default: 1.0)          │
    │ -flw FLOAT, --frustum-line-width FLOAT                                                       │
    │     Scren space line width of the camera frustums. (default: 2.0)                            │
    │ -alw FLOAT, --axis-line-width FLOAT                                                          │
    │     Screen space line width of the axes. (default: 1.0)                                      │
    │ -fu, --flip-up-axis, --no-flip-up-axis                                                       │
    │     If true, flip the up axis of the scene from -Z to +Z (default: False)                    │
    │ -ps FLOAT, --point-size FLOAT                                                                │
    │     Size of the points in screen space. (default: 1.0)                                       │
    │ -pc {None}|{FLOAT FLOAT FLOAT}, --points-color {None}|{FLOAT FLOAT FLOAT}                    │
    │     If set to a color tuple, use this color for all points instead of their RGB values.      │
    │     Color values must be in the range [0.0, 1.0]. (default: None)                            │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────╯
