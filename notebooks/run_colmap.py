# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import glob
import shutil
import logging
import subprocess
from argparse import ArgumentParser

log = logging.getLogger("colmap_pipeline")
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(handler)


def run_command(cmd, step_name):
    """Run a shell command, letting output pass through directly to the console."""
    log.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with code {result.returncode}")
    log.info(f"{step_name} completed successfully.")


parser = ArgumentParser("COLMAP reconstruction pipeline")
parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")
parser.add_argument("--skip_features", action="store_true", help="Skip feature extraction")
parser.add_argument("--skip_matching", action="store_true", help="Skip feature matching")
parser.add_argument("--skip_ba", action="store_true", help="Skip bundle adjustment")
parser.add_argument("--split_ba", action="store_true", help="Use hierarchical mapper instead of regular mapper")
parser.add_argument(
    "--refine_ba", action="store_true", help="Run additional triangulation and BA refinement iterations"
)
parser.add_argument("--refine_iters", default=1, type=int, help="Number of refinement iterations")
parser.add_argument("--global_ba", action="store_true", help="Run global bundle adjustment after mapper")
parser.add_argument("--source_path", "-s", required=True, type=str, help="Path to source data directory")
parser.add_argument("--camera", default="PINHOLE", type=str, help="Camera model (e.g., PINHOLE, OPENCV, RADIAL)")
parser.add_argument("--colmap_executable", default="", type=str, help="Path to COLMAP executable (default: colmap)")
parser.add_argument("--geoalign", action="store_true", help="Align model using GPS/geographic data")
parser.add_argument("--align", action="store_true", help="Align model to plane")
parser.add_argument("--multicamera", action="store_true", help="Use single camera per folder instead of single camera")
parser.add_argument("--max_image_size", default=7096, type=int, help="Maximum image size for feature extraction")
parser.add_argument("--max_sift_features", default=20000, type=int, help="Maximum number of SIFT features per image")
parser.add_argument("--sift_threads", default=14, type=int, help="Number of threads for SIFT extraction")
parser.add_argument("--mask_path", default=None, type=str, help="Path to image masks directory")
parser.add_argument("--random_seed", default=None, type=int, help="Random seed for deterministic reconstruction")

args = parser.parse_args()
colmap_command = args.colmap_executable if args.colmap_executable else "colmap"
use_gpu = 0 if args.no_gpu else 1

single_camera_tag = "single_camera"
if args.multicamera:
    single_camera_tag = "single_camera_per_folder"

max_image_size = args.max_image_size
sift_max_features = args.max_sift_features
sift_matches = sift_max_features * 3

match_gpu = use_gpu
if sift_max_features > 32768:
    log.warning("SIFT features > 32768, disabling GPU for matching")
    match_gpu = 0

if not args.skip_features:
    log.info("Starting feature extraction...")
    for f in glob.glob(os.path.join(args.source_path, "database.db*")):
        os.remove(f)

    sparse_dir = os.path.join(args.source_path, "sparse")
    if os.path.isdir(sparse_dir):
        shutil.rmtree(sparse_dir)

    feat_extracton_cmd = (
        f"{colmap_command} feature_extractor "
        f"--database_path {args.source_path}/database.db "
        f"--image_path {args.source_path}/images_raw "
        f"--ImageReader.camera_model {args.camera} "
        f"--ImageReader.{single_camera_tag} 1 "
        f"--FeatureExtraction.use_gpu {use_gpu} "
        f"--SiftExtraction.max_image_size {max_image_size} "
        f"--SiftExtraction.max_num_features {sift_max_features} "
        f"--FeatureExtraction.num_threads {args.sift_threads}"
    )

    if args.mask_path is not None:
        feat_extracton_cmd += f" --ImageReader.mask_path {args.mask_path}"

    run_command(feat_extracton_cmd, "Feature extraction")

if not args.skip_matching:
    log.info("Starting feature matching...")
    matcher_type = "exhaustive_matcher"
    feat_matching_cmd = (
        f"{colmap_command} {matcher_type} "
        f"--database_path {args.source_path}/database.db "
        f"--FeatureMatching.use_gpu {match_gpu} "
        f"--FeatureMatching.max_num_matches {sift_matches} "
        f"--FeatureMatching.guided_matching 1"
    )

    run_command(feat_matching_cmd, "Feature matching")

sparse_fn = os.path.join(args.source_path, "sparse")
ba_fn = os.path.join(sparse_fn, "0")

if not args.skip_ba:
    log.info("Starting mapper and bundle adjustment...")
    if os.path.isdir(sparse_fn):
        shutil.rmtree(sparse_fn)
    os.makedirs(ba_fn)

    if not args.split_ba:
        mapper_cmd = (
            f"{colmap_command} mapper "
            f"--database_path {args.source_path}/database.db "
            f"--image_path {args.source_path}/images_raw "
            f"--output_path {sparse_fn} "
            f"--Mapper.ba_global_max_num_iterations 50 "
            f"--Mapper.ba_use_gpu {use_gpu}"
        )

        if args.random_seed is not None:
            mapper_cmd += f" --Mapper.init_min_tri_angle {args.random_seed}"

        run_command(mapper_cmd, "Mapper")
    else:
        mapper_cmd = (
            f"{colmap_command} hierarchical_mapper "
            f"--database_path {args.source_path}/database.db "
            f"--image_path {args.source_path}/images_raw "
            f"--output_path {sparse_fn} "
            f"--Mapper.tri_min_angle 0.75 "
            f"--Mapper.ba_use_gpu {use_gpu}"
        )

        if args.random_seed is not None:
            mapper_cmd += f" --Mapper.init_min_tri_angle {args.random_seed}"

        run_command(mapper_cmd, "Hierarchical mapper")

if args.global_ba and not args.skip_ba:
    log.info("Running global bundle adjustment...")
    global_ba_cmd = (
        f"{colmap_command} bundle_adjuster "
        f"--input_path {ba_fn} "
        f"--output_path {ba_fn} "
        f"--BundleAdjustment.use_gpu {use_gpu}"
    )

    run_command(global_ba_cmd, "Global bundle adjustment")


if args.refine_ba:
    log.info(f"Running {args.refine_iters} refinement iteration(s)...")
    for i in range(args.refine_iters):
        log.info(f"Refinement iteration {i+1}/{args.refine_iters}...")

        tri_cmd = (
            f"{colmap_command} point_triangulator "
            f"--input_path {ba_fn} "
            f"--database_path {args.source_path}/database.db "
            f"--image_path {args.source_path}/images_raw "
            f"--output_path {ba_fn} "
            f"--Mapper.tri_min_angle 0.75"
        )

        run_command(tri_cmd, "Point triangulator")

        bundle_cmd = (
            f"{colmap_command} bundle_adjuster "
            f"--input_path {ba_fn} "
            f"--output_path {ba_fn} "
            f"--BundleAdjustment.use_gpu {use_gpu}"
        )

        run_command(bundle_cmd, "Bundle adjuster")

    log.info("Refinement completed successfully.")


if args.geoalign or args.align:
    log.info("Starting model alignment...")
    align_fn = os.path.join(args.source_path, "sparse", "aligned")
    if os.path.isdir(align_fn):
        shutil.rmtree(align_fn)
    os.makedirs(align_fn)

    if args.geoalign:
        ref = "1"
        align_type = "ecef"
        log.info("Using geographic alignment (ECEF)")
    else:
        ref = "0"
        align_type = "plane"
        log.info("Using plane alignment")

    alignment_cmd = (
        f"{colmap_command} model_aligner "
        f"--input_path {ba_fn} "
        f"--output_path {align_fn} "
        f"--database_path {args.source_path}/database.db "
        f"--ref_is_gps {ref} "
        f"--alignment_type {align_type} "
        f"--alignment_max_error 3.0"
    )

    run_command(alignment_cmd, "Model aligner")

    for f in glob.glob(os.path.join(ba_fn, "*.bin")):
        os.remove(f)
    for f in glob.glob(os.path.join(align_fn, "*")):
        shutil.copy2(f, ba_fn)
    shutil.rmtree(align_fn)
    log.info("Model alignment completed successfully.")

log.info("Starting image undistortion...")
undistort_cmd = (
    f"{colmap_command} image_undistorter "
    f"--image_path {args.source_path}/images_raw "
    f"--input_path {ba_fn} "
    f"--output_path {args.source_path} "
    f"--output_type COLMAP "
    f"--max_image_size {max_image_size}"
)

run_command(undistort_cmd, "Image undistorter")

dist_fn = os.path.join(args.source_path, "sparse", "distorted")
if os.path.isdir(dist_fn):
    shutil.rmtree(dist_fn)
shutil.copytree(ba_fn, dist_fn)
for f in glob.glob(os.path.join(sparse_fn, "*.bin")):
    shutil.copy2(f, ba_fn)
    os.remove(f)
log.info("Image undistortion completed successfully.")
log.info("COLMAP reconstruction pipeline completed!")
