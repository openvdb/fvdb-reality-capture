# %%
import argparse
import datetime
import json
import logging
import os
import sys
import time

import cv2
import fvdb
import matplotlib.pyplot as plt
import numpy as np
import point_cloud_utils as pcu
import torch

import fvdb_reality_capture as frc
import fvdb_reality_capture.transforms as fvtransforms
from fvdb_reality_capture.tools import (
    export_splats_to_usdz,
    mesh_from_splats,
    mesh_from_splats_dlnr,
)


# %%
# Visualize an image in an SfmScene and the 3D points visible from that images
# projected onto the image plane as blue dots.
def plot_image_from_scene(scene: frc.sfm_scene.SfmScene, image_id: int):
    image_meta: frc.sfm_scene.SfmPosedImageMetadata = scene.images[image_id]
    camera_meta: frc.sfm_scene.SfmCameraMetadata = image_meta.camera_metadata

    # Get the visible 3d points for this image
    visible_points_3d: np.ndarray = scene.points[image_meta.point_indices]

    # Project those points onto the image plane
    # 1. Get the world -> camera space transform and projection matrix
    world_to_cam_matrix: np.ndarray = image_meta.world_to_camera_matrix
    projection_matrix: np.ndarray = camera_meta.projection_matrix
    # 2. Transform world points to camera space
    visible_points_3d_cam_space = world_to_cam_matrix[:3, :3] @ visible_points_3d.T + world_to_cam_matrix[:3, 3:4]
    # 3. Transform camera space coordinates to image space
    visible_points_2d = projection_matrix @ visible_points_3d_cam_space
    visible_points_2d /= visible_points_2d[2]

    # Load the image and convert to RGB (OpenCV uses BGR by default)
    loaded_image = cv2.imread(image_meta.image_path)
    assert loaded_image is not None, f"Failed to load image at {image_meta.image_path}"
    loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)

    # Plot the image and projected points
    plt.title(f"SfmScene Image {image_id}")
    plt.axis("off")
    plt.imshow(loaded_image)
    plt.scatter(visible_points_2d[0], visible_points_2d[1], color="#432de9", marker=".", s=2)


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
@torch.no_grad()
def plot_reconstruction_results(model: fvdb.GaussianSplat3d, sfm_scene: frc.sfm_scene.SfmScene, image_id: int):
    # Get one of the images and its camera from the scene
    image_meta: frc.sfm_scene.SfmPosedImageMetadata = sfm_scene.images[image_id]
    camera_meta: frc.sfm_scene.SfmCameraMetadata = image_meta.camera_metadata
    camera_to_world_matrix = torch.from_numpy(image_meta.camera_to_world_matrix).to(
        device=model.device, dtype=torch.float32
    )
    projection_matrix = torch.from_numpy(camera_meta.projection_matrix).to(device=model.device, dtype=torch.float32)
    image_height, image_width = image_meta.image_size

    # Read the ground truth image from disk
    gt_image = cv2.imread(image_meta.image_path)
    assert gt_image is not None, f"Failed to load image at {image_meta.image_path}"
    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

    # Render the image and a depth map from the Gaussian splat model
    rendered_rgbd, alphas = model.render_images_and_depths(
        world_to_camera_matrices=camera_to_world_matrix.inverse().unsqueeze(0).contiguous(),
        projection_matrices=projection_matrix.unsqueeze(0),
        image_width=image_width,
        image_height=image_height,
        near=0.1,
        far=10000.0,
    )
    rgb = rendered_rgbd[0, ..., :3].cpu().numpy()
    depth = (rendered_rgbd[0, ..., 3] / alphas.squeeze()).cpu().numpy()
    rendered_image = np.clip(rgb, 0.0, 1.0)
    rendered_image = (rendered_image * 255).astype(np.uint8)

    # Plot the ground truth and rendered images side by side
    plt.figure(figsize=(25, 4.25))
    plt.suptitle(f"Image ID {image_id}")
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    plt.axis("off")
    plt.imshow(gt_image)
    plt.subplot(1, 3, 2)
    plt.title("Rendered from Gaussian Splat")
    plt.axis("off")
    plt.imshow(rendered_image)
    plt.subplot(1, 3, 3)
    plt.title("Rendered Depth Map from Gaussian Splat")
    plt.axis("off")
    plt.imshow(depth, cmap="turbo")
    plt.show()


# %%
VIZ = True
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train fVDB gs with optional mesh reconstruction")

    parser.add_argument("data_dir", type=str, default="", help="path to the model .ply file")
    parser.add_argument("--results_dir", type=str, default="", help="path to the result dir ()")

    parser.add_argument("--ip_address", required=False, type=str, default="127.0.0.1", help="viewer ip address")
    parser.add_argument("--port", required=False, type=int, default=8016, help="viewer port")

    parser.add_argument("--downsample_factor", required=False, type=int, default=4, help="image downsample factor")
    parser.add_argument(
        "--scene_normalization", required=False, type=str, default="none", help="Scene normalization: [none|pca]"
    )
    parser.add_argument("--means_lr", required=False, type=float, default=1.6e-5, help="means learning rate")
    parser.add_argument("--max_epochs", required=False, type=int, default=13, help="Number of training epochs")
    parser.add_argument("--refine_start_epoch", required=False, type=int, default=2, help="densification start")
    parser.add_argument("--refine_stop_epoch", required=False, type=int, default=7)
    parser.add_argument("--sh_degree", required=False, type=int, default=3)
    parser.add_argument("--increase_sh_degree_every_epoch", required=False, type=int, default=3)
    parser.add_argument("--save_at_percent", metavar="int", type=float, nargs="+", default=[30, 50, 80, 100])
    parser.add_argument("--optimize_camera_poses", action="store_true")
    parser.add_argument("--random_bkgd", action="store_true")
    parser.add_argument("--opacity_reg", required=False, type=float, default=0.2)
    parser.add_argument("--scale_reg", required=False, type=float, default=0.2)
    parser.add_argument("--remove_gaussians_outside_scene_bbox", action="store_true")
    # [-45, -45, -5, 45, 45, 25]
    parser.add_argument("--scene_bbox", metavar="float", type=float, nargs=6, default=[])

    parser.add_argument("--plot_images", action="store_true")
    parser.add_argument("--build_mesh", action="store_true")
    parser.add_argument("--build_mesh_dlnr", action="store_true")

    args = parser.parse_args()
    logger = logging.getLogger("main")
    logger.info(f"Found {torch.cuda.device_count()} devices: {torch.cuda.get_device_name(0)}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logger.info(f"Found {torch.cuda.device_count()} devices: {torch.cuda.get_device_name(0)}")

    colmap_path = args.data_dir
    colmap_sparse_path = os.path.join(colmap_path, "sparse", "0")
    if not os.path.exists(colmap_sparse_path):
        colmap_sparse_path = os.path.join(colmap_path, "sparse")
    if not os.path.exists(colmap_sparse_path):
        raise FileNotFoundError(f"COLMAP directory {colmap_sparse_path} does not exist.")

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    results_dir = os.path.join(args.results_dir, f"gs_{now}")
    if results_dir[0] != "/":
        results_dir = os.path.join(args.data_dir, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "command_line.txt"), "w") as f:
        f.write(" ".join(sys.argv))

    with open(os.path.join(results_dir, "args.json"), "w") as fp:
        json.dump(args.__dict__, fp)

    # Let's use verbose logging to track what happens under the hood.
    # For less output set level=logging.WARN. For more set level=logging.DEBUG
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s : %(message)s")

    # Initialize the fvdb.viz module for interactive 3D visualization.
    # This will spin up a small HTTP server in the background.
    fvdb.viz.init(port=args.port)

    logger.info(f"Loading COLMAP scene from {args.data_dir}")
    sfm_scene = frc.sfm_scene.SfmScene.from_colmap(args.data_dir)

    if args.plot_images:
        # Plot three images from the scene and their visible 3D points alongside each other
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 3, 1)
        plot_image_from_scene(sfm_scene, 8)
        plt.subplot(1, 3, 2)
        plot_image_from_scene(sfm_scene, 16)
        plt.subplot(1, 3, 3)
        plot_image_from_scene(sfm_scene, 32)
        plt.show()

    # View the SfmScene interactively in a 3D viewer
    # viz_scene = visualize_sfm_scene(sfm_scene, "Raw SfmScene", center_scene=True)
    # fvdb.viz.show()

    if args.remove_gaussians_outside_scene_bbox and len(args.scene_bbox) == 6:
        scene_dict = sfm_scene.state_dict()
        scene_dict["scene_bbox"] = args.scene_bbox
        sfm_scene = frc.sfm_scene.SfmScene.from_state_dict(scene_dict)

    # Clean up and resize the SfmScene using a transform pipeline which downsamples images,
    # aligns the scene with its principle axes and centers it at (0, 0, 0),
    # filters outlier points, and removes images with too few points.
    cleanup_and_resize_transform = fvtransforms.Compose(
        fvtransforms.DownsampleImages(
            image_downsample_factor=args.downsample_factor, image_type="jpg", rescaled_jpeg_quality=95
        ),
        fvtransforms.NormalizeScene(normalization_type=args.scene_normalization),
        # fvtransforms.PercentileFilterPoints(percentile_min=3.0, percentile_max=97.0),
        # fvtransforms.FilterImagesWithLowPoints(min_num_points=50),
    )
    cleaned_sfm_scene = cleanup_and_resize_transform(sfm_scene)

    print(f"Original scene had {len(sfm_scene.points)} points and {len(sfm_scene.images)} images")
    print(f"Cleaned scene has {len(cleaned_sfm_scene.points)} points and {len(cleaned_sfm_scene.images)} images")

    # Visualize the transformed scene so we can see the effect of the cleanup and resizing
    # Note that we don't have to center the scene here since the normalization transform already did that.
    if True:
        visualize_sfm_scene(cleaned_sfm_scene, "Cleaned SfmScene", center_scene=False)

    # %%
    optConfig = frc.radiance_fields.GaussianSplatOptimizerConfig(
        means_lr=args.means_lr,
        spatial_scale_mode=(
            frc.radiance_fields.SpatialScaleMode.MEDIAN_CAMERA_DEPTH
            if cleaned_sfm_scene.has_visible_point_indices
            else frc.radiance_fields.SpatialScaleMode.SCENE_DIAGONAL_PERCENTILE  # ABSOLUTE_UNITS
        ),
    )
    gsConfig = frc.radiance_fields.GaussianSplatReconstructionConfig(
        max_epochs=args.max_epochs,
        refine_start_epoch=args.refine_start_epoch,
        refine_stop_epoch=args.refine_stop_epoch,
        remove_gaussians_outside_scene_bbox=args.remove_gaussians_outside_scene_bbox,
        sh_degree=args.sh_degree,
        increase_sh_degree_every_epoch=args.increase_sh_degree_every_epoch,
        save_at_percent=args.save_at_percent,
        optimize_camera_poses=args.optimize_camera_poses,
        random_bkgd=args.random_bkgd,
        opacity_reg=args.opacity_reg,
        scale_reg=args.scale_reg,
    )

    tr_scene = fvdb.viz.get_scene("Training Visualization")
    # tr_scene.add_gaussian_splat_3d("Reconstructed Gaussian Splat Radiance Field", model)
    # We'll just reconstruct our scene using the default settings, which are good in most cases.
    # See the documentation for `frc.GaussianSplatReconstruction` for details on the available options.
    # Note that this process can take a while depending on the size of your scene.
    runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
        cleaned_sfm_scene,
        viz_scene=tr_scene,
        viz_update_interval_epochs=1,
        device="cuda:0",
        optimizer_config=optConfig,
        config=gsConfig,
        use_every_n_as_val=200,
    )
    runner.optimize(show_progress=True)

    model: fvdb.GaussianSplat3d = runner.model

    # Save the model as a PLY file for viewing in external tools like SuperSplat
    model.save_ply(os.path.join(results_dir, "reconstructed_model.ply"), metadata=runner.reconstruction_metadata)

    # Save the model as a USDZ file for viewing in tools like NVIDIA's Isaac Sim
    export_splats_to_usdz(model, out_path=os.path.join(results_dir, "reconstructed_model.usdz"))

    print(
        f"Reconstructed Gaussian Splat Model has {model.num_gaussians}, is on device {model.device}, and renders images with {model.num_channels} channels."
    )
    if args.plot_images:
        plot_reconstruction_results(model, cleaned_sfm_scene, image_id=80)
        plot_reconstruction_results(model, cleaned_sfm_scene, image_id=16)
        plot_reconstruction_results(model, cleaned_sfm_scene, image_id=100)

    # Add our splat model to the viewer
    scene = fvdb.viz.get_scene("Gaussian Splat Model Visualization")
    scene.add_gaussian_splat_3d("Reconstructed Gaussian Splat Radiance Field", model)

    scene.add_cameras(
        "Input Cameras",
        camera_to_world_matrices=cleaned_sfm_scene.camera_to_world_matrices,
        projection_matrices=cleaned_sfm_scene.projection_matrices,
        axis_length=1,
        frustum_scale=1.5,
    )

    # Set up the viewer's initial camera to be positioned at the first camera in the SfmScene
    # looking at the center of the scene. This should give a good initial view of the model.
    camera_position = cleaned_sfm_scene.images[0].origin
    camera_lookat_point = model.means.mean(dim=0)
    scene.set_camera_lookat(eye=camera_position, center=camera_lookat_point, up=(0, 0, 1))  # Colmap uses Z as up
    fvdb.viz.show()

    if args.build_mesh:
        # cleaned_sfm_scene = sfm_scene
        # The truncation margin determines the width of the narrow band around the surface in which we compute the TSDF.
        # A larger margin will produce coarser voxels, while a smaller margin will produce finer voxels but may miss some surface details.
        # Here we pick a truncation margin of 0.25 world units in our scene.
        truncation_margin = 0.5

        # This function returns a tensor of vertices, faces, and colors for the mesh.
        # The vertices have shape (num_vertices, 3), the faces have shape (num_faces, 3),
        # and the colors have shape (num_vertices, 3). The colors are in the range [0, 1].
        v, f, c = mesh_from_splats(
            model,
            cleaned_sfm_scene.camera_to_world_matrices,
            cleaned_sfm_scene.projection_matrices,
            cleaned_sfm_scene.image_sizes,
            truncation_margin,
        )

        # Save the mesh as a PLY file for viewing in external tools using point_cloud_utils (https://fwilliams.info/point-cloud-utils/) [3]
        pcu.save_mesh_vfc(
            os.path.join(results_dir, "reconstructed_mesh.ply"), v.cpu().numpy(), f.cpu().numpy(), c.cpu().numpy()
        )

        print(f"Reconstructed mesh with {v.shape[0]:,} vertices and {f.shape[0]:,} faces")

    if args.build_mesh_dlnr:
        # The truncation margin determines the width of the narrow band around the surface in which we compute the TSDF.
        # A larger margin will produce coarser voxels, while a smaller margin will produce finer voxels but may miss some surface details.
        # Here we pick a truncation margin of 0.25 world units in our scene.
        truncation_margin = 0.25

        # This function has virtually the same interface as `mesh_from_splats`.
        # It returns a tensor of vertices, faces, and colors for the mesh.
        # The vertices have shape (num_vertices, 3), the faces have shape (num_faces, 3),
        # and the colors have shape (num_vertices, 3). The colors are in the range [0, 1].
        v, f, c = mesh_from_splats_dlnr(
            model,
            cleaned_sfm_scene.camera_to_world_matrices,
            cleaned_sfm_scene.projection_matrices,
            cleaned_sfm_scene.image_sizes,
            truncation_margin,
        )

        # Save the mesh as a PLY file for viewing in external tools using point_cloud_utils (https://fwilliams.info/point-cloud-utils/) [3]
        pcu.save_mesh_vfc(
            os.path.join(results_dir, "reconstructed_mesh_dlnr.ply"), v.cpu().numpy(), f.cpu().numpy(), c.cpu().numpy()
        )

        print(f"Reconstructed mesh with {v.shape[0]:,} vertices and {f.shape[0]:,} faces")

    input("Press any key to exit the script. ")
