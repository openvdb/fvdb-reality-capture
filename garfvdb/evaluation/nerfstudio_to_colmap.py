#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import json
import os
import struct
from pathlib import Path
from typing import Dict

import numpy as np


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def nerfstudio_to_colmap_xform(c2w, keep_original_world_coordinate=False):
    """
    Convert nerfstudio camera-to-world matrix back to COLMAP quaternion and translation.

    Args:
        c2w: 4x4 camera-to-world transformation matrix from nerfstudio
        keep_original_world_coordinate: whether the original conversion preserved world coordinates

    Returns:
        quat_vec: quaternion vector [w, x, y, z] in COLMAP format
        transform_vec: translation vector in COLMAP format
    """
    c2w_copy = np.array(c2w).copy()

    # Reverse the coordinate system transformations in opposite order
    if not keep_original_world_coordinate:
        # Reverse the Z negation
        c2w_copy[2, :] *= -1
        # Reverse the Y and Z row swap
        c2w_copy = c2w_copy[np.array([0, 2, 1, 3]), :]

    # Convert from OpenGL back to OpenCV coordinate system
    c2w_copy[0:3, 1:3] *= -1

    # Invert to get world-to-camera matrix
    w2c = np.linalg.inv(c2w_copy)

    # Extract rotation and translation
    rotation = w2c[0:3, 0:3]
    translation = w2c[0:3, 3]

    # Convert rotation matrix back to quaternion
    quat_vec = rotmat2qvec(rotation)

    return quat_vec, translation


def camera_path_to_colmap_xform(c2w):
    """
    Convert camera_path.json camera-to-world matrix to COLMAP quaternion and translation.

    This is different from nerfstudio_to_colmap_xform because camera_path.json files
    contain poses in OpenGL coordinates directly from the viewer, without the additional
    transformations applied during nerfstudio's data processing pipeline.

    Args:
        c2w: 4x4 camera-to-world transformation matrix from camera_path.json

    Returns:
        quat_vec: quaternion vector [w, x, y, z] in COLMAP format
        transform_vec: translation vector in COLMAP format
    """
    # Create a copy to avoid modifying the input
    c2w_copy = np.array(c2w).copy()

    # Convert from OpenGL to OpenCV coordinate system
    # OpenGL: +X right, +Y up, +Z back
    # OpenCV: +X right, +Y down, +Z forward
    c2w_copy[0:3, 1:3] *= -1

    # Invert to get world-to-camera matrix
    w2c = np.linalg.inv(c2w_copy)

    # Extract rotation and translation
    rotation = w2c[0:3, 0:3]
    translation = w2c[0:3, 3]

    # Convert rotation matrix back to quaternion
    quat_vec = rotmat2qvec(rotation)

    return quat_vec, translation


def write_cameras_bin(cameras: Dict, output_file: str):
    """Write cameras to COLMAP cameras.bin format."""
    with open(output_file, "wb") as f:
        # Write number of cameras
        f.write(struct.pack("Q", len(cameras)))  # Use Q for uint64

        for camera_id, camera_data in sorted(cameras.items()):
            # Camera header: camera_id, model, width, height
            f.write(struct.pack("I", camera_id))  # camera_id (uint32)
            f.write(struct.pack("i", camera_data["model"]))  # model (int32)
            f.write(struct.pack("Q", camera_data["width"]))  # width (uint64)
            f.write(struct.pack("Q", camera_data["height"]))  # height (uint64)

            # Camera parameters (variable length double array)
            params = camera_data["params"]
            f.write(struct.pack(f"{len(params)}d", *params))


def write_images_bin(images: Dict, output_file: str):
    """Write images to COLMAP images.bin format."""
    with open(output_file, "wb") as f:
        # Write number of images
        f.write(struct.pack("Q", len(images)))  # Use Q for uint64

        for image_id, image_data in sorted(images.items()):
            # Image header
            f.write(struct.pack("I", image_id))  # image_id (uint32)

            # Quaternion (4 doubles: qw, qx, qy, qz)
            quat = image_data["quaternion"]
            f.write(struct.pack("4d", *quat))

            # Translation (3 doubles: tx, ty, tz)
            trans = image_data["translation"]
            f.write(struct.pack("3d", *trans))

            # Camera ID
            f.write(struct.pack("I", image_data["camera_id"]))  # camera_id (uint32)

            # Image name (null-terminated string)
            name = image_data["name"].encode("utf-8") + b"\x00"
            f.write(name)

            # Number of 2D points (we'll use 0 for now)
            f.write(struct.pack("Q", 0))  # num_points2D (uint64)


def write_points3d_bin(output_file: str):
    """Write empty points3D.bin file."""
    with open(output_file, "wb") as f:
        # Write number of 3D points (0)
        f.write(struct.pack("Q", 0))  # Use Q for uint64


def convert_nerfstudio_to_colmap(input_path: Path, output_path: Path) -> None:
    """
    Convert nerfstudio scene to COLMAP binary format
    """

    # Load nerfstudio transforms
    transforms_path = input_path / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found at {transforms_path}")

    print(f"Loading transforms from {transforms_path}")
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract camera parameters
    width = int(transforms["frames"][0]["w"])
    height = int(transforms["frames"][0]["h"])

    # Handle different focal length formats
    if "fl_x" in transforms["frames"][0] and "fl_y" in transforms["frames"][0]:
        fx = transforms["frames"][0]["fl_x"]
        fy = transforms["frames"][0]["fl_y"]
    elif "focal_length" in transforms["frames"][0]:
        fx = fy = transforms["frames"][0]["focal_length"]
    else:
        fx = fy = max(width, height)

    cx = transforms["frames"][0].get("cx", width / 2.0)
    cy = transforms["frames"][0].get("cy", height / 2.0)

    print(f"width: {width}, height: {height}")
    print(f"fx: {fx}, fy: {fy}")
    print(f"cx: {cx}, cy: {cy}")

    # Check for distortion parameters
    k1 = transforms.get("k1", 0.0)
    k2 = transforms.get("k2", 0.0)
    p1 = transforms.get("p1", 0.0)
    p2 = transforms.get("p2", 0.0)

    # Determine camera model and parameters
    if abs(k1) > 1e-10 or abs(k2) > 1e-10 or abs(p1) > 1e-10 or abs(p2) > 1e-10:
        # OPENCV model (4)
        camera_model = 4
        params = [fx, fy, cx, cy, k1, k2, p1, p2]
    elif abs(fx - fy) > 1e-6:
        # PINHOLE model (1)
        camera_model = 1
        params = [fx, fy, cx, cy]
    else:
        # SIMPLE_PINHOLE model (0)
        camera_model = 0
        params = [fx, cx, cy]

    print(f"Camera model: {camera_model}, params: {params}")

    # Prepare camera data
    cameras = {1: {"model": camera_model, "width": width, "height": height, "params": params}}

    # Convert camera poses
    frames = transforms.get("frames", [])
    images = {}

    print(f"Converting {len(frames)} camera poses...")
    for image_id, frame in enumerate(frames, start=1):
        file_path = frame["file_path"]
        if file_path.startswith("./"):
            file_path = file_path[2:]

        # Convert pose
        transform_matrix = frame["transform_matrix"]
        quaternion, translation = nerfstudio_to_colmap_xform(transform_matrix)

        images[image_id] = {
            "quaternion": quaternion,
            "translation": translation,
            "camera_id": 1,
            "name": os.path.basename(file_path),
        }

        if image_id <= 3:  # Debug first few
            print(f"  Image {image_id}: {images[image_id]['name']}")
            print(f"    Quat: {quaternion}")
            print(f"    Trans: {translation}")

    # Write COLMAP binary files
    print(f"Writing COLMAP files to {output_path}")

    cameras_file = output_path / "cameras.bin"
    images_file = output_path / "images.bin"
    points3d_file = output_path / "points3D.bin"

    write_cameras_bin(cameras, str(cameras_file))
    write_images_bin(images, str(images_file))
    write_points3d_bin(str(points3d_file))

    # Verify files were created
    for file_path, name in [
        (cameras_file, "cameras.bin"),
        (images_file, "images.bin"),
        (points3d_file, "points3D.bin"),
    ]:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {name}: {size} bytes")

            # Quick verification
            if name == "cameras.bin" and size > 0:
                with open(file_path, "rb") as f:
                    num_cameras = struct.unpack("Q", f.read(8))[0]
                    print(f"   Contains {num_cameras} camera(s)")

            elif name == "images.bin" and size > 0:
                with open(file_path, "rb") as f:
                    num_images = struct.unpack("Q", f.read(8))[0]
                    print(f"   Contains {num_images} image(s)")
        else:
            print(f"❌ {name}: Not created")

    print("Conversion complete!")


def main():
    """
    Convert nerfstudio scene's transforms.json to COLMAP cameras and images binary format
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description="Nerfstudio to COLMAP converter")
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to nerfstudio scene directory containing transforms.json"
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to output COLMAP sparse directory")

    args = parser.parse_args()

    try:
        convert_nerfstudio_to_colmap(args.input, args.output)
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
