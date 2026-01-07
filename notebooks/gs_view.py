# %%
import argparse
import logging
import os
import termios
import time

import cv2
import fvdb
import matplotlib.pyplot as plt
import numpy as np
import torch

import fvdb_reality_capture as frc

# Let's use verbose logging to track what happens under the hood.
# For less output set level=logging.WARN. For more set level=logging.DEBUG
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s : %(message)s")

logger = logging.getLogger("main")


# Visualize the SfmScene interactively in a 3D viewer using fvdb.viz.Viewer
def visualize_sfm_scene(scene: frc.sfm_scene.SfmScene, name: str, center_scene: bool = False):

    viewer_scene = fvdb.viz.get_scene("SfmScene Visualization")
    viewer_scene.reset()
    # Optionally center the scene at the origin.
    # This is useful to visualize multiple scenes together without them being far apart.
    if center_scene:
        center_transform = np.eye(4)
        center_transform[:3, 3] = -np.median(scene.points, axis=0)
        scene = scene.apply_transformation_matrix(center_transform)

    # Plot the points in the SfmScene with their colors (which are uint8 by default but the viewer
    # expects float32 colors in [0,1]).
    # Each point is drawn as a small sphere with a 2 pixel radius.
    viewer_scene.add_point_cloud(
        name=f"{name} Points", points=scene.points, colors=scene.points_rgb.astype(np.float32) / 255.0, point_size=2.0
    )

    # Plot the cameras as coordinate frames with axis length 2 units,
    # and frustums whose distance from the origin to camera plane is 1 unit long.
    viewer_scene.add_cameras(
        f"{name} Cameras",
        camera_to_world_matrices=scene.camera_to_world_matrices,
        projection_matrices=scene.projection_matrices,
        axis_length=2,
        frustum_scale=2.5,
    )

    # Set the initial camera view to be at the position of the first posed image, in the SfmScene,
    # looking at the center of the 3D points, with Z as up (COLMAP SfM scenes use Z as up).
    viewer_scene.set_camera_lookat(
        eye=scene.image_camera_positions[0],
        center=np.zeros(3),
        up=np.array([0, 0, 1]),  # Z is up in COLMAP SfM scenes
    )
    return viewer_scene


# %%

if __name__ == "__main__":

    # _device_count = torch.cuda.device_count()
    # _device_names = [torch.cuda.get_device_name(d) for d in range(_device_count)]
    # logger.info(f"Found {_device_count} devices: " + str(_device_names))
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    _device_count = torch.cuda.device_count()
    _device_names = [torch.cuda.get_device_name(d) for d in range(_device_count)]
    logger.info(f"Found {_device_count} devices: " + str(_device_names))

    parser = argparse.ArgumentParser(description="view gs ply model")

    parser.add_argument("model_path", type=str, default="", help="path to the model .ply file")
    parser.add_argument("--ip_address", required=False, type=str, default="127.0.0.1", help="viewer ip address")
    parser.add_argument("--port", required=False, type=int, default=8017, help="viewer port")

    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        logger.error(f"File {model_path} not found.")
    dirname = os.path.basename(os.path.dirname(model_path))
    fvdb.viz.init(port=args.port)

    model, metadata = fvdb.GaussianSplat3d.from_ply(model_path)
    # Add our splat model to the viewer
    logger.info(f"Read model with {model.num_gaussians:_} splats from model_path")
    scene = fvdb.viz.get_scene(f"3DGS model at {dirname}")
    scene.add_gaussian_splat_3d("3DGS Radiance Field", model)

    origin = (-0.073, -0.06, 0.006)  # (0.251, -0.842, 0.028)
    lookAt = (-0.3826399, 0.9235322, -0.02598172)  # (-0.073, -0.06, 0.006)
    up = (0, 0, 1.0)
    scene.set_camera_lookat(eye=origin, center=lookAt, up=up)

    # fvdb.viz.show()

    input("Press any key to exit the script. ")

    # _scene = fvdb.viz.get_scene(f"3DGS model at {dirname}")
    origin = scene.camera_orbit_center
    lookAt = scene.camera_orbit_direction
    up = scene.camera_up_direction
    logger.info(f"origin={str(origin.numpy())}, lookAt={str(lookAt.numpy())}, {str(up)}")

    # scene1 = fvdb.viz.get_scene(f"3DGS model at {dirname}")

    # input("Press any key again to exit the script. ")
