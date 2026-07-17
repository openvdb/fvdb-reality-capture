# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any, Callable, Literal, Mapping, cast

import fvdb
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from fvdb_reality_capture.radiance_fields.gaussian_splatting import GaussianSplat3d
from fvdb_reality_capture.instance_segmentation.config import GARfVDBConfig
from fvdb_reality_capture.instance_segmentation.training.dataset import GARfVDBInput
from fvdb_reality_capture.instance_segmentation.util import rgb_to_sh


class SparseConvWithSkips(torch.nn.Module):
    """Sparse convolutional network with skip connections for sparse grids.

    Attributes:
        kernel_size: Convolution kernel size.
        relu: ReLU activation function.
    """

    def __init__(self, num_grids: int, feature_dim: int) -> None:
        """Initialize the sparse convolutional network.

        Args:
            num_grids: Number of grids to create convolutional layers for.
        """
        super().__init__()

        self.kernel_size = 3
        self.relu = torch.nn.ReLU()

        # Create 3 convolutional layers per grid with Xavier initialization
        for i in range(num_grids):
            for j in range(3):
                self.add_module(
                    f"conv_{i}_{j}", fvdb.nn.SparseConv3d(feature_dim, feature_dim, self.kernel_size, bias=False)
                )
                torch.nn.init.xavier_uniform_(self.get_submodule(f"conv_{i}_{j}").weight)

    @nvtx.range("SparseConvWithSkips.forward")
    def forward(self, data: fvdb.JaggedTensor, grid_batch: fvdb.GridBatch) -> fvdb.JaggedTensor:
        result = []

        for i in range(len(data)):
            in_data = data[i]
            in_grid = grid_batch[i]

            plan = fvdb.ConvolutionPlan.from_grid_batch(kernel_size=self.kernel_size, stride=1, source_grid=in_grid)

            x = self.get_submodule(f"conv_{i}_0")(in_data, plan)

            out_grid = plan.target_grid

            x = self.relu(x, out_grid)

            x = self.get_submodule(f"conv_{i}_1")(x, plan)
            x = self.relu(x, out_grid)

            # Third conv layer with residual skip connection
            x = self.get_submodule(f"conv_{i}_2")(x, plan)
            x.data.jdata += in_data.jdata

            result.append(x)

        return fvdb.JaggedTensor(result)


class GARfVDBModel(torch.nn.Module):
    """A PyTorch module that implements a segmentation method using Gaussian Splatting and ƒVDB.

    This model implements a segmentation method inspired by GARField which uses a trained Gaussian
    Splatting model as the renderable radiance field instead of a NeRF.  The encoded features are
    sampled from a batch of fVDB grids and then optionally passed through a sparse convolutional
    network before being concatenated with a scale value and passed to an MLP.  The MLP output is
    then used to predict a mask for each pixel.

    The model consists of several key components:
    1. A Gaussian Splatting model for 3D scene representation
    2. Feature encoding grids or per-gaussian features
    3. An optional sparse convolutional network for encoded feature processing [currently WIP]
    4. An MLP for feature transformation

    Attributes:
        device (torch.device): The device to run the model on (CPU/GPU)
        model_config (GARfVDBConfig): Configuration parameters for the model
        gs_model (GaussianSplat3d): The underlying Gaussian Splatting model
        quantile_transformer (Callable): Function to normalize scales
        encoder_grids (Optional[fvdb.nn.VDBTensor]): Feature encoding grids when use_grid is True
        encoder_convnet (Optional[SparseConvWithSkips]): Sparse convolutional encoder network
        mlp (torch.nn.Sequential): MLP for feature transformation
    """

    ### Attributes ###
    mlp: torch.nn.Sequential
    encoder_gridbatch_features_data: torch.nn.Parameter | torch.Tensor
    encoder_gridbatch: fvdb.GridBatch
    encoder_convnet: SparseConvWithSkips | None

    @staticmethod
    def _state_dict_post_hook(module, state_dict, prefix, local_metadata):
        """State dict post hook to also provide the encoder gridbatch.

        Returns:
            dict: The state dictionary of the model with the encoder gridbatch if it is used.
        """
        if module.model_config.use_grid:
            state_dict[f"{prefix}encoder_gridbatch"] = module.encoder_gridbatch.cpu()

    def __init__(
        self,
        gs_model: GaussianSplat3d,
        scale_stats: torch.Tensor | None,
        model_config: GARfVDBConfig = GARfVDBConfig(),
        device: str | torch.device = torch.device("cuda:0"),
        *,
        encoder_gridbatch: fvdb.GridBatch | None = None,
        encoder_features: torch.Tensor | None = None,
        max_scale: torch.Tensor | None = None,
        quantile_values: torch.Tensor | None = None,
        output_quantiles: torch.Tensor | None = None,
    ):
        """Initialize the GARfVDBModel from gsplat checkpoint and scale statistics from the entire training dataset.

        Args:
            gs_model: GaussianSplat3D model
            scale_stats: [N] Tensor of scale statistics from the entire training dataset.
                May be ``None`` when loading precomputed quantile buffers.
            model_config: Model configuration
            device: Device to use
        """

        super().__init__()
        # Because the encoder gridbatch is not a tensor that would be returned by the state_dict method,
        # we need to register a post hook to add it to the state dict.
        self.register_state_dict_post_hook(self._state_dict_post_hook)

        self.device = torch.device(device)
        self.model_config = model_config
        self.gs_model = gs_model
        if self.model_config.use_grid and (self.model_config.num_grids < 4 or self.model_config.num_grids % 2 != 0):
            raise ValueError("GARfVDBConfig.num_grids must be an even integer greater than or equal to 4")

        # Quantile lookup tensors are buffers so the portable inference artifact does not
        # need the training-time mask statistics.
        if scale_stats is not None:
            fitted_max_scale, fitted_quantile_values, fitted_output_quantiles = self._fit_quantiles(
                scale_stats, device=self.device
            )
            max_scale = fitted_max_scale
            quantile_values = fitted_quantile_values
            output_quantiles = fitted_output_quantiles
        elif max_scale is None or quantile_values is None or output_quantiles is None:
            raise ValueError(
                "Provide scale_stats when training, or max_scale, quantile_values, and output_quantiles when loading."
            )

        self.register_buffer("_max_scale", max_scale.detach().to(self.device))
        self.register_buffer("_quantile_values", quantile_values.detach().to(self.device))
        self.register_buffer("_output_quantiles", output_quantiles.detach().to(self.device))

        # --- Encoded Features ---
        # When use_grid=True: Use GARField-style encoding with 3D feature grids at
        # multiple scales. Features are sampled using Gaussian means and weighted
        # by transmittance. When use_grid=False: Store features per-Gaussian and
        # render directly.
        if self.model_config.use_grid:
            if encoder_gridbatch is not None:
                if encoder_features is None:
                    raise ValueError("encoder_features is required when encoder_gridbatch is provided")
                if encoder_gridbatch.grid_count != self.model_config.num_grids:
                    raise ValueError(
                        f"Expected {self.model_config.num_grids} encoder grids, got {encoder_gridbatch.grid_count}."
                    )
                if encoder_features.shape != (
                    encoder_gridbatch.total_voxels,
                    self.model_config.grid_feature_dim,
                ):
                    raise ValueError(
                        "Encoder feature shape does not match the loaded NanoVDB topology: "
                        f"expected {(encoder_gridbatch.total_voxels, self.model_config.grid_feature_dim)}, "
                        f"got {tuple(encoder_features.shape)}."
                    )
                self.encoder_gridbatch = encoder_gridbatch.to(self.device)
                self.encoder_gridbatch_features_data = torch.nn.Parameter(
                    encoder_features.detach().to(self.device).contiguous()
                )
                num_grids = [self.model_config.num_grids // 2, self.model_config.num_grids // 2]
            else:
                # GARField encoder uses two sets of grids with exponentially-spaced resolutions:
                # - 12 grids from 16 to 256 voxels per axis (coarse to medium)
                # - 12 grids from 256 to 2048 voxels per axis (medium to fine)
                # Each grid has 8 feature channels
                resolution_range = [(16, 256), (256, 2048)]
                num_grids = [self.model_config.num_grids // 2, self.model_config.num_grids // 2]
                # Get the spatial extent of the Gaussian means
                means = self.gs_model.means.detach()
                extent = means.max(dim=0).values - means.min(dim=0).values
                max_extent = extent.max().item()

                # Calculate the worldspace voxel sizes for each grid across the specified ranges
                voxel_sizes = []
                for res_range, n_grids in zip(resolution_range, num_grids):
                    growth_rate = np.exp((np.log(res_range[1]) - np.log(res_range[0])) / (n_grids - 1))
                    for i in range(n_grids):
                        res = np.round(res_range[0] * (growth_rate**i))
                        voxel_sizes.append([max_extent / res for _ in range(3)])

                # Create the encoder grids
                points = fvdb.JaggedTensor([means for _ in range(len(voxel_sizes))])
                self.encoder_gridbatch = fvdb.GridBatch.from_points(
                    points=points,
                    voxel_sizes=voxel_sizes,  # type: ignore
                )
                # The original GARField encoder grids store features at voxel corners
                self.encoder_gridbatch = self.encoder_gridbatch.dual_grid()

                # Initialize the encoded features
                enc_features = (
                    torch.randn(
                        [self.encoder_gridbatch.total_voxels, self.model_config.grid_feature_dim], device=device
                    )
                    * 1e-3
                )
                # Store as a parameter to make it a leaf tensor
                self.encoder_gridbatch_features_data = torch.nn.Parameter(enc_features)

            if self.model_config.use_grid_conv:
                self.encoder_convnet = SparseConvWithSkips(
                    num_grids=sum(num_grids), feature_dim=self.model_config.grid_feature_dim
                ).to(device)

        else:
            # Initialize the per-gaussian features and register as a parameter
            self.gs_features = torch.nn.Parameter(
                torch.zeros(
                    [self.gs_model.means.shape[0], 1, self.model_config.num_grids * self.model_config.grid_feature_dim],
                    device=device,
                )
            )

            # Create reusable GaussianSplat3d for rendering - sh0 will be updated from gs_features
            self._gs_model_for_render = fvdb.GaussianSplat3d.from_tensors(
                means=self.gs_model.means.detach(),
                quats=self.gs_model.quats.detach(),
                log_scales=self.gs_model.log_scales.detach(),
                logit_opacities=self.gs_model.logit_opacities.detach(),
                sh0=self.gs_features.detach(),  # placeholder, will be overwritten
                shN=torch.zeros(
                    [self.gs_model.means.shape[0], 0, self.model_config.num_grids * self.model_config.grid_feature_dim],
                    device=device,
                ),
            )

        # --- MLP ---
        # GARField-style MLP ("instance net") with configurable hidden layers.
        # NOTE: GARField had used 4 hidden layers with 256 units each, 256 output channels
        # Input: concatenation of multi-scale encoder features and spatial scale encoding
        if self.model_config.use_grid:
            i_channels = self.model_config.grid_feature_dim * np.sum(num_grids) + 1
        else:
            i_channels = self.model_config.num_grids * self.model_config.grid_feature_dim + 1
        o_channels = self.model_config.mlp_output_dim
        n_neurons = self.model_config.mlp_hidden_dim
        hidden_layers = self.model_config.mlp_num_layers

        # Create MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(i_channels, n_neurons, bias=True),
            torch.nn.ReLU(),
            *[
                layer
                for _ in range(hidden_layers)
                for layer in (torch.nn.Linear(n_neurons, n_neurons, bias=True), torch.nn.ReLU())
            ],
            torch.nn.Linear(n_neurons, o_channels, bias=False),
        ).to(device)

        # Initialize MLP: Kaiming for ReLU layers, Xavier for final layer, zero biases
        mlp_layers = [layer for layer in self.mlp if isinstance(layer, torch.nn.Linear)]
        for i, layer in enumerate(mlp_layers):
            is_final_layer = i == len(mlp_layers) - 1
            if is_final_layer:
                torch.nn.init.xavier_uniform_(layer.weight)
            else:
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    ### Properties ###
    @property
    def enc_features(self) -> fvdb.JaggedTensor:
        """Get the encoded features.

        Returns:
            fvdb.JaggedTensor: The encoded features
        """
        return self.encoder_gridbatch.jagged_like(self.encoder_gridbatch_features_data)

    @property
    def max_grouping_scale(self) -> torch.Tensor:
        """Get the maximum grouping scale.

        Returns:
            torch.Tensor: The maximum grouping world space scale
        """
        return self._max_scale

    @property
    def quantile_transformer(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """The scale quantile transformer.

        Returns:
            Callable: The quantile transformer
        """
        return self._transform_scale

    @staticmethod
    def create_from_state_dict(
        state_dict: dict[str, Any],
        model_config: GARfVDBConfig,
        gs_model: GaussianSplat3d,
        scale_stats: torch.Tensor,
    ) -> "GARfVDBModel":
        """Load the model from the given state dictionary.

        Args:
            state_dict: State dictionary of the model
            model_config: Model configuration
            gs_model: GaussianSplat3d model
            scale_stats: Scale statistics

        Returns:
            GARfVDBModel: The loaded model
        """
        if model_config.use_grid:
            model = GARfVDBModel(
                gs_model,
                scale_stats,
                model_config,
                gs_model.device,
                encoder_gridbatch=state_dict["encoder_gridbatch"],
                encoder_features=state_dict["encoder_gridbatch_features_data"],
            )
        else:
            model = GARfVDBModel(gs_model, scale_stats, model_config, gs_model.device)
            model.gs_features = torch.nn.Parameter(state_dict["gs_features"])

        network_state = {
            key: value
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor) and key not in {"encoder_gridbatch_features_data", "gs_features"}
        }
        model.load_state_dict(network_state, strict=False)
        return model

    @staticmethod
    def _fit_quantiles(
        scales: torch.Tensor,
        device: torch.device,
        distribution: Literal["uniform", "normal"] = "normal",
        n_quantiles: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit the lookup tensors used to normalize grouping scales."""
        scales = scales.flatten()
        positive_scales = scales[scales > 0]
        if positive_scales.numel() == 0:
            raise ValueError("At least one positive grouping scale is required")
        max_scale = torch.max(positive_scales)
        scales = positive_scales[positive_scales < max_scale]
        if scales.numel() == 0:
            scales = positive_scales

        n_quantiles = min(n_quantiles, max(2, scales.numel()))
        quantile_levels = torch.linspace(0, 1, n_quantiles, device=scales.device, dtype=scales.dtype)
        quantile_values = torch.quantile(scales.detach(), quantile_levels).to(device)
        quantile_levels = quantile_levels.to(device)
        if distribution == "normal":
            clamped_levels = quantile_levels.clamp(1e-7, 1 - 1e-7)
            output_quantiles = torch.erfinv(2 * clamped_levels - 1) * (2**0.5)
        else:
            output_quantiles = quantile_levels
        return max_scale.to(device), quantile_values, output_quantiles

    def _transform_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Transform scales using the persisted piecewise-linear quantile lookup."""
        original_shape = x.shape
        x_flat = x.reshape(-1)

        indices = torch.searchsorted(self._quantile_values, x_flat).clamp(1, self._quantile_values.numel() - 1)

        lower_q = self._quantile_values[indices - 1]
        upper_q = self._quantile_values[indices]

        denom = upper_q - lower_q
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        weight = ((x_flat - lower_q) / denom).clamp(0, 1)

        lower_out = self._output_quantiles[indices - 1]
        upper_out = self._output_quantiles[indices]
        transformed = lower_out + weight * (upper_out - lower_out)

        return transformed.reshape(original_shape)

    def network_state_dict(self) -> dict[str, torch.Tensor]:
        """Return dense model tensors, excluding the NanoVDB encoder feature payload."""
        return {
            key: value.detach().cpu().contiguous()
            for key, value in self.state_dict().items()
            if isinstance(value, torch.Tensor) and key not in {"encoder_gridbatch_features_data", "gs_features"}
        }

    @classmethod
    def from_artifact_components(
        cls,
        *,
        gs_model: GaussianSplat3d,
        model_config: GARfVDBConfig,
        encoder_gridbatch: fvdb.GridBatch,
        encoder_features: torch.Tensor,
        network_state: Mapping[str, torch.Tensor],
        device: str | torch.device,
    ) -> "GARfVDBModel":
        """Construct an inference model from NanoVDB and safetensors payloads."""
        required = {"_max_scale", "_quantile_values", "_output_quantiles"}
        missing = required.difference(network_state)
        if missing:
            raise ValueError(f"Network payload is missing required scale tensors: {sorted(missing)}")
        model = cls(
            gs_model=gs_model,
            scale_stats=None,
            model_config=model_config,
            device=device,
            encoder_gridbatch=encoder_gridbatch,
            encoder_features=encoder_features,
            max_scale=network_state["_max_scale"],
            quantile_values=network_state["_quantile_values"],
            output_quantiles=network_state["_output_quantiles"],
        )
        incompatible = model.load_state_dict(dict(network_state), strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if key != "encoder_gridbatch_features_data"]
        missing_network = [key for key in incompatible.missing_keys if key != "encoder_gridbatch_features_data"]
        if unexpected or missing_network:
            raise ValueError(
                f"Network payload does not match GARfVDBConfig; missing={missing_network}, unexpected={unexpected}."
            )
        model.eval()
        return model

    @nvtx.range("GARfVDBModel.get_encoded_features")
    def get_encoded_features(self, input: GARfVDBInput) -> torch.Tensor:
        """Get the per-pixel encoder features for the given input images or pixel samples.

        Args:
            input: GARfVDBInput data

        Returns:
            Encoded feature tensor [B, R, S, F] or [B, H, W, S, F]
        """
        img_w = input["image_w"][0]
        img_h = input["image_h"][0]
        if not self.model_config.use_grid:
            # Update sh0 with current gs_features before rendering
            self._gs_model_for_render.sh0 = self.gs_features

            intrinsics = input["projection"]
            world_to_cam = input["world_to_camera"]
            pixel_coords = input.get("pixel_coords", None)  # [B, rays_per_image, 2]

            if pixel_coords is not None:
                B, rays_per_image = pixel_coords.shape[:2]
                flat_pixel_coords = pixel_coords.reshape(-1, 2)
                offsets = torch.arange(0, B + 1, device=pixel_coords.device, dtype=torch.long) * rays_per_image
                pixels_to_render = fvdb.JaggedTensor.from_data_and_offsets(flat_pixel_coords, offsets)

                features, _ = self._gs_model_for_render.sparse_render_images(
                    pixels_to_render=pixels_to_render,
                    world_to_camera_matrices=world_to_cam,
                    projection_matrices=intrinsics,
                    image_width=img_w,
                    image_height=img_h,
                    near=0.01,
                    far=1e10,
                )
                feature_size = features.eshape[-1]

                cam_batch_size, num_rays_per_cam = pixel_coords.shape[:2]
                # reshape features to [B, R, S, F]
                return features.jdata.reshape(len(features), num_rays_per_cam, -1, feature_size)

            else:
                features, _ = self._gs_model_for_render.render_images(
                    image_width=img_w,
                    image_height=img_h,
                    world_to_camera_matrices=world_to_cam,
                    projection_matrices=intrinsics,
                    near=0.01,
                    far=1e10,
                    sh_degree_to_use=0,
                )
                feature_size = features.shape[-1]
                # reshape features to [B, H, W, S, F]
                return features.reshape(len(features), img_h, img_w, -1, feature_size)
        else:
            intrinsics: torch.Tensor = input["projection"]
            world_to_cam: torch.Tensor = input["world_to_camera"]
            pixel_coords: torch.Tensor | None = input.get("pixel_coords", None)  # [B, rays_per_image, 2]

            # Render the IDs of the gaussians that contribute to each pixel, which we will use to sample the feature grids
            with torch.no_grad():
                if pixel_coords is not None:
                    B, rays_per_image = pixel_coords.shape[:2]
                    flat_pixel_coords = pixel_coords.reshape(-1, 2)
                    offsets = torch.arange(0, B + 1, device=pixel_coords.device, dtype=torch.long) * rays_per_image
                    pixels_to_render = fvdb.JaggedTensor.from_data_and_offsets(flat_pixel_coords, offsets)

                    ids, weights = self.gs_model.sparse_render_contributing_gaussian_ids(
                        pixels_to_render=pixels_to_render,
                        top_k_contributors=self.model_config.depth_samples,
                        world_to_camera_matrices=world_to_cam,
                        projection_matrices=intrinsics,
                        image_width=img_w,
                        image_height=img_h,
                        near=0.01,
                        far=1e10,
                    )

                else:
                    ids, weights = self.gs_model.render_contributing_gaussian_ids(
                        top_k_contributors=self.model_config.depth_samples,
                        image_width=img_w,
                        image_height=img_h,
                        world_to_camera_matrices=world_to_cam,
                        projection_matrices=intrinsics,
                        near=0.01,
                        far=1e10,
                    )

                # TODO: Investigate NaN values in contributing gaussian weights
                if torch.isnan(weights.jdata).any():
                    logging.warning("NaN values detected in contributing gaussian weights")
                    raise ValueError("NaN values in contributing gaussian weights")

            if self.model_config.enc_feats_one_idx_per_ray:
                # Stochastically pick the depth_sample for each pixel based on the weights as probabilities
                # Convert weights to probabilities if they aren't already normalized

                # # Check for rays where all weights are 0
                # zero_mask = torch.tensor(weights.lshape) == 0  # [C,[R]]
                # if zero_mask.jdata.any():
                #     logging.warning(f"WARNING: Found {zero_mask.jdata.sum().item()} rays with all 0 weights")
                #     # For each ray with all zeros, set the first depth sample to 1e-10
                #     weights = torch.where(
                #         zero_mask.unsqueeze(-1),  # [B, R, 1] or [B, H, W, 1]
                #         torch.cat([torch.full_like(weights[..., :1], 1e-10), weights[..., 1:]], dim=-1),
                #         weights,
                #     )
                # Sample one index per ray based on probabilities
                # NOTE:  This probability would be much simpler if the division were just broadcastable
                probs = []
                for per_cam_weights, per_cam_weight_sum in zip(weights, weights.jsum(dim=0, keepdim=True)):
                    cam_probs = []

                    for per_ray_weights, per_ray_weight_sum in zip(per_cam_weights, per_cam_weight_sum):
                        if len(per_ray_weights) == 0:
                            cam_probs.append(torch.empty([0, per_ray_weights.eshape[-1]]))
                        else:
                            cam_probs.append(fvdb.relu(per_ray_weights.jdata / (per_ray_weight_sum.jdata + 1e-10)))
                    probs.append(cam_probs)
                probs = fvdb.JaggedTensor(probs)
                # shape: [B, R]
                depth_sample_indices = []
                for per_cam_probs in probs:
                    depth_sample_indices_cam = []
                    for per_ray_probs in per_cam_probs:
                        if len(per_ray_probs) == 0:
                            depth_sample_indices_cam.append(torch.empty([0]))
                        else:
                            depth_sample_indices_cam.append(torch.multinomial(per_ray_probs.jdata, num_samples=1))
                    depth_sample_indices.append(depth_sample_indices_cam)

                depth_sample_indices = fvdb.JaggedTensor(depth_sample_indices)
                # get 1 id per ray based on the depth_sample_indices
                single_ids = []
                for per_cam_ids, per_cam_depth_sample_indices in zip(ids, depth_sample_indices):

                    cam_ids = []
                    for per_ray_ids, per_ray_depth_sample_indices in zip(per_cam_ids, per_cam_depth_sample_indices):
                        if len(per_ray_ids) == 0:
                            cam_ids.append(torch.empty([0]))
                        else:
                            cam_ids.append(per_ray_ids.jdata[per_ray_depth_sample_indices.jdata])
                            # cam_ids.append(
                            #     torch.gather(per_ray_ids.jdata, dim=1, index=per_ray_depth_sample_indices.jdata)
                            # )
                    single_ids.append(cam_ids)
                ids = fvdb.JaggedTensor(single_ids)
                # ids = torch.gather(ids.squeeze(-1), dim=2, index=depth_sample_indices)  # [B, R, 1] or [B, H, W, 1]

            # Filter out invalid IDs (-1 indicates no valid gaussian for that position)
            valid_mask = ids.jdata >= 0
            valid_ids = ids.jdata[valid_mask]

            # Get unique among valid IDs only
            unique_valid_ids, valid_inverse = torch.unique(valid_ids, return_inverse=True)

            # Look up gaussian positions for valid unique IDs only
            unique_world_pts = self.gs_model.means[unique_valid_ids]

            # sample the encoder grids at the unique world points
            grid_count = self.encoder_gridbatch.grid_count
            num_unique_pts = len(unique_valid_ids)

            if num_unique_pts > 0:
                tiled_pts = unique_world_pts.repeat(grid_count, 1)  # [grid_count * num_unique_pts, 3]
                offsets = (
                    torch.arange(0, grid_count + 1, device=unique_world_pts.device, dtype=torch.long) * num_unique_pts
                )
                enc_grid_sample_pts = fvdb.JaggedTensor.from_data_and_offsets(tiled_pts, offsets)
                with nvtx.range("sample_encoder_grids"):
                    if self.model_config.use_grid_conv:
                        conv_output = self.encoder_convnet(self.enc_features, self.encoder_gridbatch)
                        unique_enc_feats = conv_output.sample_trilinear(enc_grid_sample_pts)
                    else:
                        unique_enc_feats = self.encoder_gridbatch.sample_trilinear(
                            enc_grid_sample_pts, self.enc_features
                        )

                feat_dim = unique_enc_feats.jdata.shape[-1]
                # jdata is [grid_count * num_unique_pts, feat_dim]
                # reshape to [grid_count, num_unique_pts, feat_dim], transpose, flatten
                unique_cam_enc_feats = (
                    unique_enc_feats.jdata.reshape(grid_count, num_unique_pts, feat_dim)
                    .transpose(0, 1)
                    .reshape(num_unique_pts, grid_count * feat_dim)
                )  # [num_unique_pts, F]

                # Re-assemble enc_feats: valid positions get looked-up features, invalid get zeros
                total_samples = ids.jdata.shape[0]
                feat_dim_total = unique_cam_enc_feats.shape[-1]
                enc_feats_data = torch.zeros(
                    total_samples, feat_dim_total, device=ids.jdata.device, dtype=unique_cam_enc_feats.dtype
                )
                enc_feats_data[valid_mask] = unique_cam_enc_feats[valid_inverse]
            else:
                # Edge case: no valid IDs at all
                # Determine feature dimension from encoder config
                feat_dim_total = grid_count * self.enc_features.shape[-1]
                total_samples = ids.jdata.shape[0]
                enc_feats_data = torch.zeros(total_samples, feat_dim_total, device=ids.jdata.device)

            enc_feats = ids.jagged_like(enc_feats_data)

            if not self.model_config.enc_feats_one_idx_per_ray:
                # Weighted sum of the enc_feats and transmittance weights
                enc_feats.jdata = enc_feats.jdata * weights.jdata.unsqueeze(-1)
                # When every pixel has exactly 1 contributor, fvdb returns a
                # 1-level JaggedTensor [C,[R]] instead of 2-level [C,[R,[K]]].
                # jsum(dim=0) on a 1-level tensor would collapse pixels within
                # each camera rather than depth samples within each pixel.
                if isinstance(enc_feats.lshape[0], list):
                    enc_feats = enc_feats.jsum(dim=0, keepdim=True)

            epsilon = 1e-6
            enc_feats.jdata = enc_feats.jdata / (torch.linalg.norm(enc_feats.jdata, dim=-1, keepdim=True) + epsilon)

            # Convert enc_feats to regular Tensor
            feature_size = enc_feats.eshape[-1]
            if pixel_coords is not None:
                # reshape enc_feats to [B, R, S, F]
                cam_batch_size, num_rays_per_cam = pixel_coords.shape[:2]
                return enc_feats.jdata.reshape(cam_batch_size, num_rays_per_cam, -1, feature_size)
            else:
                # reshape enc_feats to [B, H, W, S, F]
                return enc_feats.jdata.reshape(len(enc_feats), img_h, img_w, -1, feature_size)

    @nvtx.range("GARfVDBModel.get_mlp_output")
    def get_mlp_output(self, enc_feats: torch.Tensor, scales: torch.Tensor | float) -> torch.Tensor:
        """Get the MLP output for the given encoder features and scales.

        Args:
            enc_feats: Encoder features [B, R, S, F] or [B, H, W, S, F]
            scales: Either a scalar (float/int) for uniform scale across all pixels,
                    or a tensor [B, R] or [B, H, W] for per-pixel scales

        Returns:
            MLP output [B, R, F] or [B, H, W, F]
        """
        epsilon = 1e-5

        # Process scales through quantile transformer and concatenate with features
        if isinstance(scales, (int, float)):
            # transform once and use F.pad to append constant value
            scale_tensor = torch.tensor([scales], device=enc_feats.device, dtype=enc_feats.dtype)
            scale_value = self.quantile_transformer(scale_tensor).item()
            # Pad last dimension with constant scale value: (left_pad, right_pad)
            in_feats = torch.nn.functional.pad(enc_feats, (0, 1), value=scale_value)
        else:
            # transform per-pixel scales (used during training)
            # scales: [B, R] or [B, H, W] -> scales_quant: [B, R, S, 1] or [B, H, W, S, 1]
            scales_quant = self.quantile_transformer(scales)
            scales_quant = scales_quant.unsqueeze(-1).expand(enc_feats.shape[:-1]).unsqueeze(-1)
            in_feats = torch.cat([enc_feats, scales_quant], dim=-1)

        # Apply MLP
        gfeats = self.mlp(in_feats)
        norms = gfeats.norm(dim=-1, keepdim=True)
        gfeats = gfeats / (norms + epsilon)

        # Reshape back to 3D
        return gfeats

    def _sample_encoder_grids_at_gaussians(self) -> torch.Tensor:
        # repeat the gaussian means for each grid
        num_points = self.gs_model.means.shape[0]
        grid_count = self.encoder_gridbatch.grid_count
        repeated_data = self.gs_model.means.repeat(grid_count, 1)
        # Create offsets: [0, N, 2N, 3N, ..., grid_count*N]
        offsets = torch.arange(
            0, (grid_count + 1) * num_points, num_points, device=self.gs_model.means.device, dtype=torch.long
        )
        world_pts = fvdb.JaggedTensor.from_data_and_offsets(repeated_data, offsets)

        if self.model_config.use_grid_conv:
            conv_output = self.encoder_convnet(self.encoder_grids)
            enc_feats = conv_output.sample_trilinear(world_pts)
        else:
            with nvtx.range("encoder_gridbatch.sample_trilinear"):
                enc_feats = self.encoder_gridbatch.sample_trilinear(world_pts, self.enc_features)
        # enc_feats.jdata is [grid_count * N, F], we want [N, grid_count * F]
        num_features = enc_feats.jdata.shape[-1]
        enc_feats = enc_feats.jdata.view(grid_count, num_points, num_features).permute(1, 0, 2).reshape(num_points, -1)
        return enc_feats

    @nvtx.range("GARfVDBModel.get_mask_output")
    def get_mask_output(self, input: GARfVDBInput, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the mask output for the given input and scale

        This is a convenience function used in inference or evaluation to get the mask output for a given scale.

        Args:
            input: GARfVDBInput data
            scale: Chosen scale for the masks

        Returns:
            Mask output [B, H, W, mlp_output_dim]
        """
        img_w = input["image_w"][0]
        img_h = input["image_h"][0]

        intrinsics = input["projection"]
        world_to_cam = input["world_to_camera"]

        if self.model_config.use_grid:
            # Obtain per-gaussian features from the encoder grids
            enc_feats = self._sample_encoder_grids_at_gaussians()

            # Set them as the sh0 features
            enc_feats_state_dict = self.gs_model.state_dict()
            enc_feats_state_dict["sh0"] = rgb_to_sh(enc_feats).unsqueeze(1).contiguous()
            with nvtx.range("GaussianSplat3d.from_state_dict"):
                gs3d_enc_feats = GaussianSplat3d.from_state_dict(enc_feats_state_dict)
        else:
            with nvtx.range("update_gs_features"):
                # Update sh0 with current gs_features and reuse the model
                self._gs_model_for_render.sh0 = self.gs_features
                gs3d_enc_feats = self._gs_model_for_render

        # Render the image
        with nvtx.range("gs3d_enc_feats.render_images"):
            img_feats, alpha = gs3d_enc_feats.render_images(
                image_width=img_w,
                image_height=img_h,
                world_to_camera_matrices=world_to_cam,
                projection_matrices=intrinsics,
                near=0.01,
                far=1e10,
                sh_degree_to_use=0,
            )

        epsilon = 1e-6
        img_feats = img_feats / (torch.linalg.norm(img_feats, dim=-1, keepdim=True) + epsilon)

        # Apply MLP
        gfeats = self.get_mlp_output(img_feats, scale)

        return gfeats, alpha

    def get_gaussian_affinity_output(self, scale: float) -> torch.Tensor:
        """Get per-Gaussian affinity features for the given scale.

        This is a convenience function used during inference or evaluation to
        compute per-Gaussian feature embeddings (affinities) by sampling
        encoder grid features or using the current Gaussian features, normalizing
        them, and passing them through the MLP at the specified scale.

        Args:
            scale: Chosen scale for the affinity MLP output.

        Returns:
            A tensor of per-Gaussian affinity features.
        """
        if self.model_config.use_grid:
            # Obtain per-gaussian features from the encoder grids
            enc_feats = self._sample_encoder_grids_at_gaussians()
            enc_feats = rgb_to_sh(enc_feats)
        else:
            enc_feats = self.gs_features
            enc_feats = rgb_to_sh(enc_feats)

        epsilon = 1e-6
        enc_feats = enc_feats / (torch.linalg.norm(enc_feats, dim=-1, keepdim=True) + epsilon)

        # Apply MLP
        gfeats = self.get_mlp_output(enc_feats, scale)

        return gfeats
