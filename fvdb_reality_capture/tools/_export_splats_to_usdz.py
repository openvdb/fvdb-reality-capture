# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

# pip install msgpack numpy usd-core types-usd
import gzip
import io
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
from fvdb import GaussianSplat3d
from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class NamedUSDStage:
    filename: str
    stage: Usd.Stage

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.stage.Export(str(out_dir / self.filename))

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=self.filename, delete=False) as temp_file:
            temp_file_path = temp_file.name
        self.stage.GetRootLayer().Export(temp_file_path)
        with open(temp_file_path, "rb") as file:
            usd_data = file.read()
        zip_file.writestr(self.filename, usd_data)
        os.unlink(temp_file_path)


def _initialize_usd_stage():
    """
    Initialize a new USD stage with standard settings.

    Returns:
        Usd.Stage: A new USD stage with standard settings
    """
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1)
    stage.SetMetadata("upAxis", "Z")

    # Define xform containing everything.
    world_path = "/World"
    UsdGeom.Xform.Define(stage, world_path)
    stage.SetMetadata("defaultPrim", world_path[1:])

    return stage


def _serialize_usd_stage_to_bytes(stage: Usd.Stage) -> bytes:
    """
    Export a USD stage to a temporary file and read it back as bytes.

    Args:
        stage: The USD stage to export

    Returns:
        bytes: The exported USD stage content
    """
    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as temp_file:
        temp_file_path = temp_file.name

    stage.GetRootLayer().Export(temp_file_path)

    with open(temp_file_path, "rb") as f:
        content = f.read()

    os.unlink(temp_file_path)
    return content


def _serialize_nurec_usd(
    model_file, positions: np.ndarray, normalizing_transform: np.ndarray = np.eye(4)
) -> NamedUSDStage:
    """
    Create a USD file for the 3DGS model.

    Args:
        model_file: NamedSerialized object containing the compressed msgpack data
        positions: Positions extracted from PLY file for AABB calculation
        normalizing_transform: 4x4 transformation matrix to normalize the scene (defaults to identity)

    Returns:
        NamedUSDStage object containing the USD stage
    """
    logger.info("Creating USD file containing NuRec model")

    # Calculate AABB from positions
    min_coord = np.min(positions, axis=0)
    max_coord = np.max(positions, axis=0)
    logger.info(f"Model bounding box: min={min_coord}, max={max_coord}")

    # Convert numpy values to Python floats
    min_x, min_y, min_z = float(min_coord[0]), float(min_coord[1]), float(min_coord[2])
    max_x, max_y, max_z = float(max_coord[0]), float(max_coord[1]), float(max_coord[2])

    min_list = [min_x, min_y, min_z]
    max_list = [max_x, max_y, max_z]

    # Initialize the USD stage with standard settings
    stage = _initialize_usd_stage()

    # Set up render settings
    render_settings = {
        "rtx:rendermode": "RaytracedLighting",
        "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
        "rtx:post:histogram:enabled": False,
        "rtx:post:registeredCompositing:invertToneMap": True,
        "rtx:post:registeredCompositing:invertColorCorrection": True,
        "rtx:material:enableRefraction": False,
        "rtx:post:tonemap:op": 2,
        "rtx:raytracing:fractionalCutoutOpacity": False,
        "rtx:matteObject:visibility:secondaryRays": True,
    }
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", render_settings)

    # Define UsdVol::Volume
    from pxr import UsdVol

    gauss_path = "/World/gauss"
    gauss_volume = UsdVol.Volume.Define(stage, gauss_path)
    gauss_prim = gauss_volume.GetPrim()

    # Apply normalizing transform (identity by default)
    # Default conversion matrix from 3DGRUT to USDZ
    default_conv_tf = np.array(
        [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    normalizing_inverse = np.linalg.inv(normalizing_transform)
    corrected_matrix = normalizing_inverse @ default_conv_tf

    # Apply transform directly to the gauss volume
    matrix_op = gauss_volume.AddTransformOp()
    matrix_op.Set(Gf.Matrix4d(*corrected_matrix.flatten()))

    # Define nurec volume properties
    gauss_prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)

    # Enable transform of UsdVol::Volume to take effect
    gauss_prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    # Define field assets and link to volumetric Gaussians prim
    density_field_path = gauss_path + "/density_field"
    density_field = stage.DefinePrim(density_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("density", density_field_path)

    emissive_color_field_path = gauss_path + "/emissive_color_field"
    emissive_color_field = stage.DefinePrim(emissive_color_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("emissiveColor", emissive_color_field_path)

    # Set file paths for field assets
    nurec_relative_path = "./" + model_file.filename
    density_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    density_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("density")
    density_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float")
    density_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("density")

    emissive_color_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    emissive_color_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    emissive_color_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float3")
    emissive_color_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("emissiveColor")

    # Set identity color correction matrix
    emissive_color_field.CreateAttribute("omni:nurec:ccmR", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([1.0, 0.0, 0.0, 0.0])
    )
    emissive_color_field.CreateAttribute("omni:nurec:ccmG", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([0.0, 1.0, 0.0, 0.0])
    )
    emissive_color_field.CreateAttribute("omni:nurec:ccmB", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([0.0, 0.0, 1.0, 0.0])
    )

    # Set extent and crop boundaries
    gauss_prim.GetAttribute("extent").Set([min_list, max_list])

    # Set zero offset
    gauss_offset = [0.0, 0.0, 0.0]
    gauss_prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(gauss_offset))

    # Set crop bounds
    min_vec = Gf.Vec3d(min_x, min_y, min_z)
    max_vec = Gf.Vec3d(max_x, max_y, max_z)
    gauss_prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(min_vec)
    gauss_prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(max_vec)

    # Create empty proxy mesh relationship for forward compatibility
    gauss_prim.CreateRelationship("proxy")

    return NamedUSDStage(filename="gauss.usda", stage=stage)


def update_render_settings(stage: Usd.Stage, referenced_layer: Sdf.Layer) -> None:
    """
    Update render settings from a referenced layer.

    Args:
        stage: The stage to update
        referenced_layer: The layer containing render settings to copy
    """
    if "renderSettings" not in referenced_layer.customLayerData:
        return  # Do nothing if render settings are not present in the referenced layer

    new_render_settings = referenced_layer.customLayerData["renderSettings"]
    current_render_settings = stage.GetRootLayer().customLayerData.get("renderSettings", {})
    if current_render_settings is None:
        current_render_settings = {}

    current_render_settings.update(new_render_settings)
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", current_render_settings)


def serialize_usd_default_layer(gauss_stage: NamedUSDStage) -> NamedUSDStage:
    """
    Create a default USD layer that references the gauss stage.

    Args:
        gauss_stage: The NamedUSDStage object containing the gauss USD stage

    Returns:
        NamedUSDStage: The default USD stage with the gauss reference
    """
    stage = _initialize_usd_stage()

    # The delegate captures all errors about dangling references, effectively silencing them.
    delegate = UsdUtils.CoalescingDiagnosticDelegate()

    # Create a reference to the gauss stage
    prim = stage.OverridePrim(f"/World/{Path(gauss_stage.filename).stem}")
    # Assume that all reference paths are in the same directory, so that they are also valid relative file paths.
    prim.GetReferences().AddReference(gauss_stage.filename)

    # Copy render settings from the gauss stage's layer
    gauss_layer = gauss_stage.stage.GetRootLayer()
    if "renderSettings" in gauss_layer.customLayerData:
        update_render_settings(stage, gauss_layer)

    # Return as NamedUSDStage
    return NamedUSDStage(filename="default.usda", stage=stage)


def write_to_usdz(file_path: Path, model_file, gauss_usd: NamedUSDStage, default_usd: NamedUSDStage) -> None:
    """
    Write the USDZ file containing the model data and USD stages.

    Args:
        file_path: Path to write the USDZ file to
        model_file: The compressed model data
        gauss_usd: The gauss USD stage
        default_usd: The default USD stage
    """
    # Make sure path to usdz-file exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        # Save default.usda first (required by USDZ spec)
        default_usd.save_to_zip(zip_file)

        # Save the model file and gauss USD stage
        model_file.save_to_zip(zip_file)
        gauss_usd.save_to_zip(zip_file)

    logger.info(f"USDZ file created successfully at {file_path}")


@dataclass(kw_only=True)
class NamedSerialized:
    """
    Class to store serialized data with a filename.
    """

    filename: str
    serialized: str | bytes

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        """
        Save the serialized data to a zip file.

        Args:
            zip_file: Zip file to save the data to
        """
        zip_file.writestr(self.filename, self.serialized)


def _fill_state_dict_tensors(
    template: dict[str, Any],
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    features_albedo: np.ndarray,
    features_specular: np.ndarray,
    n_active_features: int,
    dtype=np.float16,
) -> None:
    """
    Helper function to fill the state dict tensors in a template.

    Args:
        template: Template dictionary to fill
        positions: Gaussian positions (N, 3)
        rotations: Gaussian rotations (N, 4)
        scales: Gaussian scales (N, 3)
        densities: Gaussian densities (N, 1)
        features_albedo: Gaussian albedo features (N, 3)
        features_specular: Gaussian specular features (N, M)
        n_active_features: Active SH degree
        dtype: Data type to convert to (default: np.float16)
    """
    # Convert data to specified format for efficiency
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.positions"] = positions.astype(dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.rotations"] = rotations.astype(dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.scales"] = scales.astype(dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.densities"] = densities.astype(dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_albedo"] = features_albedo.astype(
        dtype
    ).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_specular"] = features_specular.astype(
        dtype
    ).tobytes()

    # Create empty extra_signal tensor
    extra_signal = np.zeros((positions.shape[0], 0), dtype=dtype)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.extra_signal"] = extra_signal.tobytes()

    # Store n_active_features as binary data (64-bit integer)
    n_active_features_binary = np.array([n_active_features], dtype=np.int64).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.n_active_features"] = n_active_features_binary

    # Store shapes
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.positions.shape"] = list(positions.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.rotations.shape"] = list(rotations.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.scales.shape"] = list(scales.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.densities.shape"] = list(densities.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_albedo.shape"] = list(features_albedo.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_specular.shape"] = list(
        features_specular.shape
    )
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.extra_signal.shape"] = list(extra_signal.shape)
    # Empty array for scalar value
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.n_active_features.shape"] = []


def fill_3dgut_template(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    features_albedo: np.ndarray,
    features_specular: np.ndarray,
    n_active_features: int,
    density_activation: str = "sigmoid",
    scale_activation: str = "exp",
    rotation_activation: str = "normalize",
    density_kernel_degree: int = 2,
    density_kernel_density_clamping: bool = False,
    density_kernel_min_response: float = 0.0113,
    radiance_sph_degree: int = 3,
    transmittance_threshold: float = 0.001,
    global_z_order: bool = False,
    n_rolling_shutter_iterations: int = 5,
    ut_alpha: float = 1.0,
    ut_beta: float = 2.0,
    ut_kappa: float = 0.0,
    ut_require_all_sigma_points: bool = False,
    image_margin_factor: float = 0.1,
    rect_bounding: bool = True,
    tight_opacity_bounding: bool = True,
    tile_based_culling: bool = True,
    k_buffer_size: int = 0,
) -> dict[str, Any]:
    """
    Create and fill the 3DGUT JSON template with gaussian data.

    Args:
        positions: Gaussian positions (N, 3)
        rotations: Gaussian rotations (N, 4)
        scales: Gaussian scales (N, 3)
        densities: Gaussian densities (N, 1)
        features_albedo: Gaussian albedo features (N, 3)
        features_specular: Gaussian specular features (N, M)
        n_active_features: Active SH degree

        Render parameters interfaced between 3DGRUT and NuRec:

        density_kernel_degree: Kernel degree for density computation
        density_activation: Activation function for density
        scale_activation: Activation function for scale
        rotation_activation: Activation function for rotation
        density_kernel_density_clamping: Whether to clamp density kernel
        density_kernel_min_response: Minimum response for density kernel
        radiance_sph_degree: SH degree for radiance
        transmittance_threshold: Threshold for transmittance (min_transmittance in 3DGRUT)

        3DGUT-specific splatting parameters:

        global_z_order: Whether to use global z-order
        n_rolling_shutter_iterations: Number of rolling shutter iterations
        ut_alpha: Alpha parameter for unscented transform
        ut_beta: Beta parameter for unscented transform
        ut_kappa: Kappa parameter for unscented transform
        ut_require_all_sigma_points: Whether to require all sigma points
        image_margin_factor: Image margin factor (ut_in_image_margin_factor in 3DGRUT)
        rect_bounding: Whether to use rectangular bounding
        tight_opacity_bounding: Whether to use tight opacity bounding
        tile_based_culling: Whether to use tile-based culling
        k_buffer_size: Size of the k-buffer

    Returns:
        Dictionary with the filled 3DGUT template
    """
    template = {
        "nre_data": {
            "version": "0.2.576",
            "model": "nre",
            "config": {
                "layers": {
                    "gaussians": {
                        "name": "sh-gaussians",
                        "device": "cuda",
                        "density_activation": density_activation,
                        "scale_activation": scale_activation,
                        "rotation_activation": rotation_activation,
                        "precision": 16,
                        "particle": {
                            "density_kernel_planar": False,  # TODO: Does this have an equivalent in 3DGRUT?
                            "density_kernel_degree": density_kernel_degree,
                            "density_kernel_density_clamping": density_kernel_density_clamping,
                            "density_kernel_min_response": density_kernel_min_response,
                            "radiance_sph_degree": radiance_sph_degree,
                        },
                        "transmittance_threshold": transmittance_threshold,
                    }
                },
                "renderer": {
                    "name": "3dgut-nrend",
                    "log_level": 3,
                    "force_update": False,
                    "update_step_train_batch_end": False,
                    "per_ray_features": False,
                    "global_z_order": global_z_order,
                    "projection": {
                        "n_rolling_shutter_iterations": n_rolling_shutter_iterations,
                        "ut_dim": 3,  # TODO: Does this have an equivalent in 3DGRUT?
                        "ut_alpha": ut_alpha,
                        "ut_beta": ut_beta,
                        "ut_kappa": ut_kappa,
                        "ut_require_all_sigma_points": ut_require_all_sigma_points,
                        "image_margin_factor": image_margin_factor,
                        "min_projected_ray_radius": 0.5477225575051661,
                    },
                    "culling": {
                        "rect_bounding": rect_bounding,
                        "tight_opacity_bounding": tight_opacity_bounding,
                        "tile_based": tile_based_culling,
                        "near_clip_distance": 0.2,  # TODO: Does this have an equivalent in 3DGRUT?
                        # TODO: Does this have an equivalent in 3DGRUT?
                        "far_clip_distance": 3.402823466e38,
                    },
                    "render": {"mode": "kbuffer", "k_buffer_size": k_buffer_size},
                },
                "name": "gaussians_primitive",
                "appearance_embedding": {"name": "skip-appearance", "embedding_dim": 0, "device": "cuda"},
                "background": {"name": "skip-background", "device": "cuda", "composite_in_linear_space": False},
            },
            "state_dict": {
                "._extra_state": {"obj_track_ids": {"gaussians": []}},
                ".gaussians_nodes.gaussians.positions": None,
                ".gaussians_nodes.gaussians.rotations": None,
                ".gaussians_nodes.gaussians.scales": None,
                ".gaussians_nodes.gaussians.densities": None,
                ".gaussians_nodes.gaussians.extra_signal": None,
                ".gaussians_nodes.gaussians.features_albedo": None,
                ".gaussians_nodes.gaussians.features_specular": None,
                ".gaussians_nodes.gaussians.n_active_features": None,
                # Shapes
                ".gaussians_nodes.gaussians.positions.shape": None,
                ".gaussians_nodes.gaussians.rotations.shape": None,
                ".gaussians_nodes.gaussians.scales.shape": None,
                ".gaussians_nodes.gaussians.densities.shape": None,
                ".gaussians_nodes.gaussians.extra_signal.shape": None,
                ".gaussians_nodes.gaussians.features_albedo.shape": None,
                ".gaussians_nodes.gaussians.features_specular.shape": None,
                ".gaussians_nodes.gaussians.n_active_features.shape": None,
            },
        }
    }

    # Fill in the state dict tensors
    _fill_state_dict_tensors(
        template, positions, rotations, scales, densities, features_albedo, features_specular, n_active_features
    )

    return template


@torch.no_grad()
def export_splats_to_usdz(
    model: GaussianSplat3d,
    out_path: str | Path,
) -> None:
    """
    Export an :class:`fvdb.GaussianSplat3d` model to a USDZ file.

    Args:
        model (fvdb.GaussianSplat3d): The Gaussian Splat model to save to a usdz file
        out_path (str | Path): The output path for the usdz file. If the file extension is not ``.usdz``,
            it will be added. *e.g.*, ``./scene`` will save to ``./scene.usdz``.
    """

    if isinstance(out_path, str):
        out_path = Path(out_path)
    out_path = out_path.with_suffix(".usdz")
    means = model.means.cpu().numpy()
    quats = model.quats.cpu().numpy()
    log_scales = model.log_scales.cpu().numpy()
    logit_opacities = model.logit_opacities.cpu().numpy()
    sh0 = model.sh0.cpu().numpy()
    shN = model.shN.cpu().numpy()
    n_sh_coeffs = model.num_sh_bases

    usdz_params = {
        "positions": means,
        "rotations": quats,
        "scales": log_scales,
        "densities": logit_opacities,
        "features_albedo": sh0,
        "features_specular": shN,
        "n_active_features": n_sh_coeffs,
        "density_kernel_degree": 2,
        # Common renderer configuration parameters
        "density_activation": "sigmoid",
        "scale_activation": "exp",
        "rotation_activation": "normalize",  # Always normalize for rotations
        "density_kernel_density_clamping": True,
        "density_kernel_min_response": 0.0113,
        "radiance_sph_degree": 3,  # TODO: Adapt to actual number of SH coeffs
        "transmittance_threshold": 0.0001,
        "global_z_order": True,
        "n_rolling_shutter_iterations": 5,
        "ut_alpha": 1.0,
        "ut_beta": 2.0,
        "ut_kappa": 0.0,
        "ut_require_all_sigma_points": False,
        "image_margin_factor": 0.1,
        "rect_bounding": True,
        "tight_opacity_bounding": True,
        "tile_based_culling": True,
        "k_buffer_size": 0,
    }

    template = fill_3dgut_template(**usdz_params)

    # Compress the data
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
        packed = msgpack.packb(template)
        f.write(packed)  # type: ignore

    model_file = NamedSerialized(filename=out_path.stem + ".nurec", serialized=buffer.getvalue())

    # Create USD representations
    gauss_usd = _serialize_nurec_usd(model_file, means, np.eye(4))
    default_usd = serialize_usd_default_layer(gauss_usd)

    # Write the final USDZ file
    write_to_usdz(out_path, model_file, gauss_usd, default_usd)
