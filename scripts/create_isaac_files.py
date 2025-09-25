#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import sys
import pathlib
from typing import Dict, Union, Optional
import numpy as np
import point_cloud_utils as pcu
from pathlib import Path
from fvdb import GaussianSplat3d
import torch
from ply_to_usdz import export_to_usdz  # Import from same directory


def convert_mesh(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    bbox: list[float] | None = None,
    resolution: int = 100_000,
):
    """
    Convert a mesh to watertight format, optionally cropping to a bounding box.

    Args:
        input_path (pathlib.Path): Path to input mesh file (PLY format)
        output_path (pathlib.Path): Path to save the processed mesh
        bbox (list[float], optional): Bounding box coordinates [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    # Load the mesh with vertices and faces
    v, f = pcu.load_mesh_vf(str(input_path))

    print("Converting Mesh to OBJ")
    if bbox is not None:
        # Unpack bounding box coordinates
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        # Create mask for vertices within bbox
        mask = (
            (v[:, 0] >= min_x)
            & (v[:, 0] <= max_x)
            & (v[:, 1] >= min_y)
            & (v[:, 1] <= max_y)
            & (v[:, 2] >= min_z)
            & (v[:, 2] <= max_z)
        )

        # Get indices of vertices to keep
        keep_indices = np.where(mask)[0]

        # Create mapping from old vertex indices to new ones
        old_to_new = np.full(v.shape[0], -1)
        old_to_new[keep_indices] = np.arange(len(keep_indices))

        # Filter vertices
        v = v[keep_indices]

        # Filter faces - keep only faces where all vertices are within bounds
        valid_faces = []
        for face in f:
            if all(old_to_new[idx] != -1 for idx in face):
                # Remap vertex indices
                new_face = [old_to_new[idx] for idx in face]
                valid_faces.append(new_face)

        f = np.array(valid_faces, dtype=np.int32)
        # print the new bounds
        print(f"Cropped to:")
        print(f"  min: {v.min(axis=0)}")
        print(f"  max: {v.max(axis=0)}")

    # Make mesh watertight
    # See https://github.com/hjwdzh/Manifold for details
    resolution = resolution
    v_watertight, f_watertight = pcu.make_mesh_watertight(v, f, resolution=resolution)
    print(f"\nWatertight mesh has {v_watertight.shape[0]} vertices and {f_watertight.shape[0]} faces")

    # Convert to the expected types
    v_clean = v_watertight.astype(np.float32)
    f_clean = f_watertight.astype(np.int32)

    # Write OBJ file
    with open(output_path, "w") as f:
        # Write vertices
        for v in v_clean:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write faces (OBJ uses 1-based indexing)
        for face in f_clean:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"Saved watertight mesh to {output_path}")


def convert_splat(input_ply, bbox, output_usdz):
    model, metadata = GaussianSplat3d.from_ply(str(input_ply))  # Convert Path to string

    # Get positions for bounds info
    xyz = model.means.cpu().numpy()
    min_bounds = xyz.min(axis=0)
    max_bounds = xyz.max(axis=0)
    print(f"Converting Splat to USDZ")

    if bbox is not None:
        # Create mask for points within bbox
        min_x, min_y, min_z, max_x, max_y, max_z = bbox

        mask = (
            (xyz[:, 0] >= min_x)
            & (xyz[:, 0] <= max_x)
            & (xyz[:, 1] >= min_y)
            & (xyz[:, 1] <= max_y)
            & (xyz[:, 2] >= min_z)
            & (xyz[:, 2] <= max_z)
        )
        mask = torch.from_numpy(mask).to(model.device)
        # Create new model with only points in bbox using mask indexing
        model = model[mask]
        print(f"Cropped from {len(xyz)} to {len(model.means)} points")

    # Create new metadata dictionary with only compatible types
    new_metadata: Optional[Dict[str, Union[str, int, float, torch.Tensor]]] = {
        "sh_degree": int(model.sh_degree),  # Ensure it's an int
    }

    # Add bbox to metadata if provided
    if bbox is not None:
        new_metadata["bbox"] = torch.tensor(bbox, dtype=torch.float32)  # Convert to tensor

    # If original metadata exists, only copy compatible values
    if metadata is not None:
        for key, value in metadata.items():
            # Only copy compatible values
            if isinstance(value, (str, int, float, torch.Tensor)):
                new_metadata[key] = value

    # Export to USDZ
    export_to_usdz(model, output_usdz)  # export_to_usdz already handles Path objects


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop a mesh and/or splat model to a given bounding box")

    # Input/Output arguments
    parser.add_argument("--input-splat", type=Path, help="Input splat file (PLY format)")
    parser.add_argument("--output-path", type=Path, help="Output file path (no extension)")
    parser.add_argument("--input-mesh", type=Path, help="Input mesh file (PLY/OBJ format)")
    # add resolution for mesh
    parser.add_argument("--resolution", type=int, help="Resolution for mesh", required=False, default=100_000)
    # Optional bounding box arguments
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=6,
        metavar=("MIN_X", "MIN_Y", "MIN_Z", "MAX_X", "MAX_Y", "MAX_Z"),
        help="Optional bounding box coordinates to crop the model: min_x min_y min_z max_x max_y max_z",
        required=False,
    )

    args = parser.parse_args()

    # Validate that at least one input is provided
    if not args.input_splat and not args.input_mesh:
        parser.error("At least one of --input-splat or --input-mesh must be provided")

    # Process splat if input is provided
    if not args.output_path:
        parser.error("--output-path is required")

    # Create output paths with extensions
    usdz_output_path = args.output_path.with_suffix(".usdz")
    mesh_output_path = args.output_path.with_suffix(".obj")

    # Process splat if input is provided
    if args.input_splat:
        convert_splat(args.input_splat, args.bbox, usdz_output_path)

    # Process mesh if input is provided
    if args.input_mesh:
        convert_mesh(args.input_mesh, mesh_output_path, args.bbox, args.resolution)
