# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import logging
import math
from typing import Callable, Dict, Iterator, Literal, Optional, Union

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import fvdb
from fvdb import GaussianSplat3d

from .config import ModelConfig
from .dataset import GARfVDBInput
from .util import rgb_to_sh


class SparseConvWithSkips(torch.nn.Module):
    """
    PyTorch module that implements an fVDB sparse convolutional network with skip connections.
    """

    def __init__(self, num_grids: int):
        super().__init__()

        self.relu = fvdb.nn.ReLU()

        # Initialize weights and ensure they require gradients
        for i in range(num_grids):
            for j in range(3):
                self.add_module(f"conv_{i}_{j}", fvdb.nn.SparseConv3d(8, 8, 3, bias=False))
                torch.nn.init.xavier_uniform_(self.get_submodule(f"conv_{i}_{j}").weight)

    def forward(self, input: fvdb.nn.VDBTensor) -> fvdb.nn.VDBTensor:
        result = []
        for i in range(len(input)):
            in_grid = input[i]

            x = self.get_submodule(f"conv_{i}_0")(in_grid)
            x = self.relu(x)

            x = self.get_submodule(f"conv_{i}_1")(x)
            x = self.relu(x)

            # # Third conv layer with skip connection
            x = self.get_submodule(f"conv_{i}_2")(x)
            x.data.jdata += in_grid.data.jdata

            result.append(x)

        return fvdb.jcat(result)


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
    3. An optional sparse convolutional network for encoded feature processing
    4. An MLP for feature transformation

    Attributes:
        device (torch.device): The device to run the model on (CPU/GPU)
        model_config (ModelConfig): Configuration parameters for the model
        gs_model (GaussianSplat3d): The underlying Gaussian Splatting model
        quantile_transformer (Callable): Function to normalize scales
        encoder_grids (Optional[fvdb.nn.VDBTensor]): Feature encoding grids when use_grid is True
        encoder_convnet (Optional[SparseConvWithSkips]): Sparse convolutional encoder network
        mlp (torch.nn.Sequential): MLP for feature transformation
    """

    def __init__(
        self,
        gsplat_checkpoint_path: str,
        scale_stats: torch.Tensor,
        model_config: ModelConfig = ModelConfig(),
        device: Union[str, torch.device] = torch.device("cuda"),
    ):
        """Initialize the GARfVDBModel from gsplat checkpoint and scale statistics from the entire training dataset.

        Args:
            gsplat_checkpoint_path: Path to the GaussianSplat3D checkpoint
            scale_stats: [N] Tensor of scale statistics from the entire training dataset
            model_config: Model configuration
            device: Device to use
        """

        super().__init__()

        self.device = device
        self.model_config = model_config

        ### Load the GaussianSplat3D model ###
        checkpoint = torch.load(gsplat_checkpoint_path, map_location=device)

        state_dict = {
            "sh0": checkpoint["splats"]["sh0"],  # [N, 1, 3]
            "shN": checkpoint["splats"]["shN"],  # [N, K, 3]
            "quats": checkpoint["splats"]["quats"],  # [N, 4]
            "means": checkpoint["splats"]["means"],  # [N, 3]
            "log_scales": checkpoint["splats"]["log_scales"],  # [N, 3]
            "logit_opacities": checkpoint["splats"]["logit_opacities"],  # [N]
            "requires_grad": checkpoint["splats"]["requires_grad"],  # [N]
            "track_max_2d_radii_for_grad": checkpoint["splats"]["track_max_2d_radii_for_grad"],  # [N]
        }

        # if this checkpoint was saved with the old SH shapes, we need to move it to be N-first
        if state_dict["sh0"].shape[0] == 1:
            state_dict["sh0"] = state_dict["sh0"].permute(1, 0, 2)
            state_dict["shN"] = state_dict["shN"].permute(1, 0, 2)

        self.gs_model = GaussianSplat3d.from_state_dict(state_dict)

        # Build the quantile transformer
        self.quantile_transformer = self._get_quantile_func(scale_stats)

        ###  Encoded Features ###
        # When `use_grid` is True, we will use the GARField method of encoding features at different scales using
        # a set of 3D feature grids at a range of scene scales.  At training time, these features are sampled from the
        # grids using the 3D centers of the rendered gaussians and weighted by their transmittance.
        # When `use_grid` is False, we will store the encoded features per-3d-guassian and render the features directly
        if self.model_config.use_grid:
            # GARField Encoder grids consist of two sets of 3D feature grids:
            # 1. 12 grids of number of voxels along each axis ranging from 16 -> 256
            # 2. 12 grids of number of voxels along each axis ranging from 256 -> 2048
            # Each grid has 8 feature channels
            resolution_range = [(16, 256), (256, 2048)]
            num_grids = [12, 12]
            # get the spatial extent of the means
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
            enc_gridbatch = fvdb.gridbatch_from_points(
                points=points,
                voxel_sizes=voxel_sizes,  # type: ignore
            )
            # The original GARField encoder grids store features at voxel corners
            enc_gridbatch = enc_gridbatch.dual_grid()

            # Initialize the encoded features
            enc_features = 1e-4 * torch.rand(
                [enc_gridbatch.total_voxels, self.model_config.grid_feature_dim], device=device
            )
            enc_features = enc_gridbatch.jagged_like(enc_features)

            self.encoder_grids = fvdb.nn.VDBTensor(enc_gridbatch, enc_features).requires_grad_(True)

            if self.model_config.use_grid_conv:
                self.add_module("encoder_convnet", SparseConvWithSkips(num_grids=sum(num_grids)).to(device))

        else:
            # Initialize the per-gaussian features
            gs_features = torch.nn.Parameter(
                torch.zeros(
                    [self.gs_model.means.shape[0], 1, self.model_config.gs_features],
                    device=device,
                )
            )
            state_dict["sh0"] = gs_features
            self.gs_model.load_state_dict(state_dict)
            # have to set requires_grad here because 'load_state_dict' will set all params to the 'require_grad' value
            # from the state dict
            self.gs_model.sh0.requires_grad = True

        ### MLP ###
        # GARField MLP ('instance net') uses 4 hidden layers with 256 units each, 256 output channels
        # input channels is the concatenation of the encoder grids and a spatial scale encoding
        if self.model_config.use_grid:
            i_channels = self.model_config.grid_feature_dim * np.sum(num_grids) + 1
        else:
            i_channels = self.model_config.gs_features + 1
        o_channels = self.model_config.mlp_output_dim
        n_neurons = self.model_config.mlp_hidden_dim
        hidden_layers = self.model_config.mlp_num_layers
        self.add_module(
            "mlp",
            torch.nn.Sequential(
                torch.nn.Linear(i_channels, n_neurons, bias=False),
                torch.nn.ReLU(),
                *[torch.nn.Linear(n_neurons, n_neurons, bias=False), torch.nn.ReLU()] * hidden_layers,
                torch.nn.Linear(n_neurons, o_channels, bias=False),
            ).to(device),
        )

        # Initialize the MLP weights
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def get_max_grouping_scale(self) -> float:
        """Get the maximum grouping scale.

        Returns:
            float: The maximum grouping world space scale
        """
        return self.model_config.max_grouping_scale

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Return an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse: if True, then yields parameters of this module and all submodules. Otherwise, yields only parameters that are direct members of this module.

        Returns:
            Iterator over module parameters
        """
        if self.model_config.use_grid:
            if self.model_config.use_grid_conv:
                return itertools.chain(
                    self.mlp.parameters(), [self.encoder_grids.data.jdata], self.encoder_convnet.parameters()
                )
            else:
                return itertools.chain(self.mlp.parameters(), [self.encoder_grids.data.jdata])
        else:
            return itertools.chain(self.mlp.parameters(), [self.gs_model.sh0])

    def load_checkpoint(self, checkpoint_path: str):
        """Load the model checkpoint from the given path.

        Args:
            checkpoint_path: Path to the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if self.model_config.use_grid:
            self.encoder_grids.data.jdata = checkpoint["encoder_grids"]
        else:
            self.gs_model.sh0 = checkpoint["sh0"]
        self.mlp.load_state_dict(checkpoint["mlp"])

    def get_quantile_transformer(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the scale quantile transformer.

        Returns:
            Callable: The quantile transformer
        """
        return self.quantile_transformer

    def _get_quantile_func(
        self, scales: torch.Tensor, distribution: Literal["uniform", "normal"] = "normal"
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Use 3D scale statistics to normalize scales with quantile transformer.

        Args:
            scales: [N] Tensor of scales
            distribution: Distribution to use for the quantile transformer

        Returns:
            Callable: The quantile transformer
        """
        scales = scales.flatten()
        scales = scales[(scales > 0) & (scales < self.get_max_grouping_scale())]

        scales = scales.detach().cpu().numpy()  # type: ignore

        # Calculate quantile transformer
        quantile_transformer = QuantileTransformer(output_distribution=distribution)
        quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

        def quantile_transformer_func(scales):
            # This function acts as a wrapper for QuantileTransformer.
            # QuantileTransformer expects a numpy array, while we have a torch tensor.
            return torch.Tensor(quantile_transformer.transform(scales.cpu().numpy())).to(scales.device)

        return quantile_transformer_func

    @staticmethod
    def write_points_to_ply(points, filename, ascii=True):
        """
        Write 3D points to a PLY file.

        Args:
            points: [N, 3] Tensor of 3D points
            filename: Output PLY file path
            ascii: If True, write in ASCII format, otherwise binary
        """
        points_np = points.detach().cpu().numpy()
        with open(filename, "w") as f:
            f.write("ply\n")
            f.write(f"format {'ascii' if ascii else 'binary_little_endian'} 1.0\n")
            f.write("comment Generated from PyTorch tensor\n")
            f.write(f"element vertex {points_np.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")

            if ascii:
                for point in points_np:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")
                    # f.write("\n")
            else:
                # Close and reopen in binary mode for binary format
                f.close()
                import struct

                with open(filename, "ab") as f:
                    for point in points_np:
                        f.write(struct.pack("<fff", *point))

    def get_encoded_features(self, input: GARfVDBInput) -> torch.Tensor:
        """Get the per-pixel encoder features for the given input images or pixel samples.

        Args:
            input: GARfVDBInput data

        Returns:
            Encoded feature tensor [B, R, S, F] or [B, H, W, S, F]
        """
        if not self.model_config.use_grid:
            intrinsics = input["intrinsics"]
            cam_to_world = input["cam_to_world"]
            world_to_cam = torch.linalg.inv(cam_to_world).contiguous()
            pixel_coords = input.get("pixel_coords", None)  # [B, rays_per_image, 2]

            img_w = int(input["image_w"][0].item())
            img_h = int(input["image_h"][0].item())

            features, opacity = self.gs_model.render_images(
                image_width=img_w,
                image_height=img_h,
                world_to_camera_matrices=world_to_cam,
                projection_matrices=intrinsics,
                near=0.01,
                far=1e10,
                sh_degree_to_use=0,
            )

            # # select just the cam_pts that we need from the whole image if pixel_coords is specified
            if pixel_coords is not None:
                batch_indices = torch.arange(features.shape[0]).view(-1, 1).expand(-1, pixel_coords.shape[1])
                features = features[batch_indices, pixel_coords[:, :, 0], pixel_coords[:, :, 1]]
                opacity = opacity[batch_indices, pixel_coords[:, :, 0], pixel_coords[:, :, 1]]
            return features
        else:
            intrinsics = input["intrinsics"]
            cam_to_world = input["cam_to_world"]
            world_to_cam = torch.linalg.inv(cam_to_world).contiguous()
            pixel_coords = input.get("pixel_coords", None)  # [B, rays_per_image, 2]

            img_w = int(input["image_w"][0].item())
            img_h = int(input["image_h"][0].item())

            # Render the IDs of the gaussians that contribute to each pixel, which we will use to sample the feature grids
            # TODO: Sparse/masked rendering is not implemented yet, so we are rendering the entire image and then selecting
            #       only the samples required for each pixel
            with torch.no_grad():
                if pixel_coords is not None:
                    ids, weights = self.gs_model.sparse_render_top_contributing_gaussian_ids(
                        pixels_to_render=fvdb.JaggedTensor(pixel_coords.unbind()),
                        num_samples=self.model_config.depth_samples,
                        image_width=img_w,
                        image_height=img_h,
                        world_to_camera_matrices=world_to_cam,
                        projection_matrices=intrinsics,
                        near=0.01,
                        far=1e10,
                        projection_type="perspective",
                    )
                    ids = torch.stack(ids.unbind())  # [B, R, depth_samples]
                    weights = torch.stack(weights.unbind())  # [B, R, depth_samples]

                else:
                    ids, weights = self.gs_model.render_top_contributing_gaussian_ids(
                        num_samples=self.model_config.depth_samples,
                        image_width=img_w,
                        image_height=img_h,
                        world_to_camera_matrices=world_to_cam,
                        projection_matrices=intrinsics,
                        near=0.01,
                        far=1e10,
                        projection_type="perspective",
                    )

                # TOOD:  There are Nans coming out of weights, need to fix this
                # if there's any nans print a warning
                if torch.isnan(weights).any():
                    logging.warning("WARNING: Nans in weights")
                    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

            # stochastically pick the depth_sample for each pixel based on the weights as probabilities
            # Convert weights to probabilities if they aren't already normalized

            # Check for rays where all weights are -1
            zero_mask = weights.sum(dim=-1) == 0  # [B, R] or [B, H, W]
            if zero_mask.any():
                logging.warning(f"WARNING: Found {zero_mask.sum().item()} rays with all 0 weights")
                # For each ray with all zeros, set the first depth sample to 1e-10
                weights = torch.where(
                    zero_mask.unsqueeze(-1),  # [B, R, 1] or [B, H, W, 1]
                    torch.cat([torch.full_like(weights[..., :1], 1e-10), weights[..., 1:]], dim=-1),
                    weights,
                )

            probs = torch.relu(weights / (weights.sum(dim=-1, keepdim=True) + 1e-10))  # Normalize to ensure sum=1

            # Sample one index per ray based on probabilities
            # shape: [B, R]
            depth_sample_indices = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).reshape(
                probs.shape[:-1]
            )
            depth_sample_indices = depth_sample_indices.unsqueeze(-1)  # [B, R, 1] or [B, H, W, 1]
            # get 1 id per ray based on the depth_sample_indices
            selected_ids = torch.gather(ids.squeeze(-1), dim=2, index=depth_sample_indices)  # [B, R, 1] or [B, H, W, 1]

            # unique ids
            ids_3dgs = selected_ids % self.gs_model.means.shape[0]
            unique_ids, unique_ids_inverse = torch.unique(ids_3dgs, return_inverse=True)
            unique_world_pts = self.gs_model.means[unique_ids]

            # # debug test whether any of the points don't fall inside the grids' voxels
            # in_voxels = self.encoder_grids.grid.points_in_active_voxel(unique_world_pts)
            # num_not_in_voxels = (in_voxels.jdata == 0).sum().item()
            # if num_not_in_voxels > 0:
            #     print(f"num_not_in_voxels: {num_not_in_voxels}")

            # sample the encoder grids at the unique world points
            enc_grid_sample_pts = fvdb.JaggedTensor([unique_world_pts for _ in range(self.encoder_grids.grid_count)])
            if self.model_config.use_grid_conv:
                conv_output = self.encoder_convnet(self.encoder_grids)
                unique_enc_feats = conv_output.sample_trilinear(enc_grid_sample_pts)
            else:
                unique_enc_feats = self.encoder_grids.sample_trilinear(enc_grid_sample_pts)
            unique_cam_enc_feats = torch.cat(unique_enc_feats.unbind(), dim=-1)  # [unique_ids, F]
            # re-assemble the enc_feats based on the original ids
            enc_feats = unique_cam_enc_feats[unique_ids_inverse].squeeze(-2)  # [B, R, S, F]

            # Weighted sum of the enc_feats and transmittance weights
            # weights = weights.unsqueeze(-1)  # [B, R, S, 1] or [B, H, W, S, 1]

            # multiply by weights
            # enc_feats = enc_feats * weights
            # enc_feats = enc_feats.sum(dim=-2)  # [B, R, F] or [B, H, W, F]
            # print("enc_Feats shape ", enc_feats.shape)

            enc_feats = enc_feats / torch.linalg.norm(enc_feats, dim=-1, keepdim=True)

            return enc_feats

    def get_mlp_output(self, enc_feats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Get the MLP output for the given encoder features and scales.

        Args:
            enc_feats: Encoder features [B, R, S, F] or [B, H, W, S, F]
            scales: Scales [B, R] or [B, H, W]

        Returns:
            MLP output [B, R, F] or [B, H, W, F]
        """
        epsilon = 1e-5

        # Debug print to understand shapes
        logging.debug(f"get_mlp_output shapes: enc_feats={enc_feats.shape}, scales={scales.shape}")

        # Process scales
        # Flatten for processing
        scales_flat = scales.reshape(-1)
        scales_flat = scales_flat.contiguous().view(-1, 1)
        scales_flat = self.quantile_transformer(scales_flat).to(scales_flat.device)

        in_feats = torch.cat([enc_feats, scales_flat.reshape(scales.shape).unsqueeze(-1)], dim=-1)

        # Apply MLP
        gfeats = self.mlp(in_feats)
        norms = gfeats.norm(dim=-1, keepdim=True)
        gfeats = gfeats / (norms + epsilon)

        # Reshape back to 3D
        return gfeats

    def get_mask_output(self, input: GARfVDBInput, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the mask output for the given input and scale

        This is a convenience function used in inference or evaluation to get the mask output for a given scale.

        Args:
            input: GARfVDBInput data
            scale: Chosen scale for the masks

        Returns:
            Mask output [B, H, W, mlp_output_dim]
        """
        img_w = int(input["image_w"][0].item())
        img_h = int(input["image_h"][0].item())

        intrinsics = input["intrinsics"]
        cam_to_world = input["cam_to_world"]
        world_to_cam = torch.linalg.inv(cam_to_world).contiguous()

        # Obtain per-gaussian features from the encoder grids
        world_pts = fvdb.JaggedTensor([self.gs_model.means for _ in range(self.encoder_grids.grid_count)])
        if self.model_config.use_grid_conv:
            conv_output = self.encoder_convnet(self.encoder_grids)
            enc_feats = conv_output.sample_trilinear(world_pts)
        else:
            enc_feats = self.encoder_grids.sample_trilinear(world_pts)
        enc_feats = torch.cat(enc_feats.unbind(), dim=-1)  # [N, F]
        enc_feats = enc_feats / torch.linalg.norm(enc_feats, dim=-1, keepdim=True)

        # Apply MLP
        epsilon = 1e-5
        scales = torch.full_like(enc_feats[:, :1], scale)

        gfeats = self.mlp(torch.cat([enc_feats, scales], dim=-1))
        norms = gfeats.norm(dim=-1, keepdim=True)
        gfeats = gfeats / (norms + epsilon)

        # Set them as the sh0 features
        enc_feats_state_dict = self.gs_model.state_dict()
        enc_feats_state_dict["sh0"] = rgb_to_sh(gfeats).unsqueeze(1).contiguous()

        gs3d_enc_feats = GaussianSplat3d.from_state_dict(enc_feats_state_dict)

        # Render the image
        img, alpha = gs3d_enc_feats.render_images(
            image_width=img_w,
            image_height=img_h,
            world_to_camera_matrices=world_to_cam,
            projection_matrices=intrinsics,
            projection_type="perspective",
            near=0.01,
            far=1e10,
            sh_degree_to_use=0,
        )

        img = img / torch.linalg.norm(img, dim=-1, keepdim=True)

        return img, alpha
