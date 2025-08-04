#! /usr/bin/env python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import tqdm
import tyro
import viser
from garfvdb.config import Config
from garfvdb.dataset import (
    GARfVDBInput,
    GARfVDBInputCollateFn,
    RandomSamplePixels,
    RandomSelectMaskIDAndScale,
    Resize,
    SegmentationDataset,
    TransformDataset,
)
from garfvdb.loss import calculate_loss
from garfvdb.model import GARfVDBModel
from garfvdb.optim import ExponentaLRWithRampUpScheduler
from garfvdb.util import calculate_pca_projection, pca_projection_fast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from viz import CameraState, Viewer

logging.basicConfig(level=logging.ERROR)
logging.getLogger().name = "garfvdb"


class Runner:
    def __init__(
        self,
        cfg: Config,
        checkpoint_path: Path,
        segmentation_dataset_path: str,
        device: Union[str, torch.device] = "cuda",
        disable_viewer: bool = False,
    ):
        self.config = cfg
        self.device = device

        self.val_every_n_steps = 500
        self.disable_viewer = disable_viewer

        # Create tensorboard writer with timestamp for unique runs
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/segmentation_training/{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"TensorBoard logs will be saved to: {log_dir}")

        ### Dataset ###
        full_dataset = SegmentationDataset(segmentation_dataset_path)

        # Split into train and validation sets
        val_split = 0.1
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size - 1, 1], generator=torch.Generator().manual_seed(42)
        )

        # For training, randomly sample pixels from each image
        self.train_dataset = TransformDataset(
            self.train_dataset,
            Compose(
                [
                    RandomSamplePixels(self.config.sample_pixels_per_image, scale_bias_strength=0.0),
                    RandomSelectMaskIDAndScale(),
                ]
            ),
        )
        self.val_dataset = TransformDataset(
            self.val_dataset,
            Compose(
                [Resize(1 / 2), RandomSamplePixels(self.config.sample_pixels_per_image), RandomSelectMaskIDAndScale()]
            ),
        )
        # For testing, use the full image
        self.test_dataset = TransformDataset(self.test_dataset, Compose([Resize(1 / 6), RandomSelectMaskIDAndScale()]))

        ### Model ###
        # Scale grouping stats
        grouping_scale_stats = torch.cat(full_dataset.scales)
        # Calculate max scale from dataset
        self.max_scale = float(torch.max(grouping_scale_stats).item())
        self.model = GARfVDBModel(
            checkpoint_path,
            grouping_scale_stats,
            model_config=self.config.model,
            device=device,
        )

        # Optimizer
        # Different parameter groups with separate learning rates
        lr = 1e-5
        param_groups = [
            {"params": self.model.mlp.parameters(), "lr": lr},  # Base learning rate for MLP
        ]

        # Add grid parameters with different learning rate if using grid
        if self.config.model.use_grid:
            # For encoder_grids, we need to access the data.jdata parameter as shown in the model's parameters() method
            param_groups.append({"params": [self.model.encoder_grids.data.jdata], "lr": lr})
            if self.config.model.use_grid_conv:
                param_groups.append({"params": self.model.encoder_convnet.parameters(), "lr": lr})

        else:
            # For sh0 model, add the sh0 parameter
            param_groups.append({"params": [self.model.gs_model.sh0], "lr": lr})  # Lower learning rate for sh0

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=lr / self.config.batch_size,
            eps=1e-15,
            # param_groups,
            weight_decay=1e-6,
        )

        # Add ExponentaLRWithRampUpScheduler for each parameter group
        base_lr = self.optimizer.param_groups[0]["lr"]  # MLP learning rate (1e-4)
        # grid_lr = self.optimizer.param_groups[1]["lr"]  # Grid/sh0 learning rate (5e-5)

        # Create scheduler for all parameters using base_lr as reference
        self.scheduler = ExponentaLRWithRampUpScheduler(
            optimizer=self.optimizer,
            lr_init=base_lr,
            lr_final=1e-6,
            max_steps=self.config.num_train_iters,
        )

        # Store the full dataset for viewer
        self.full_dataset = full_dataset

        # Initialize viewer sliders to default values
        self.viewer_scale = self.max_scale * 0.1  # Start at 10% of max scale
        self.viewer_mask_blend = 0.5  # Start with 50% blending

        # PCA freezing variables
        self.freeze_pca = False
        self.frozen_pca_projection = None  # Store the V matrix from calculate_pca_projection

        # Viewer
        if not self.disable_viewer:
            self.server = viser.ViserServer(port=8080, verbose=False)

            # Set default up axis
            self.current_up_axis = "+z"
            self.server.scene.set_up_direction(self.current_up_axis)
            self.client_up_axis_dropdowns = {}
            self.client_scale_sliders = {}
            self.client_mask_blend_sliders = {}
            self.client_freeze_pca_checkboxes = {}

            @self.server.on_client_connect
            def _(client: viser.ClientHandle) -> None:
                # up axis dropdown
                up_axis_dropdown = client.gui.add_dropdown(
                    "Up Axis",
                    options=["+x", "-x", "+y", "-y", "+z", "-z"],
                    initial_value=self.current_up_axis,
                )

                # scale slider
                scale_slider = client.gui.add_slider(
                    "Scale",
                    min=0.0,
                    max=self.max_scale,
                    step=0.01,
                    initial_value=self.viewer_scale,
                )

                # mask blend slider
                mask_blend_slider = client.gui.add_slider(
                    "Mask Blend",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=self.viewer_mask_blend,
                )

                # freeze PCA checkbox
                freeze_pca_checkbox = client.gui.add_checkbox(
                    "Freeze PCA Projection",
                    initial_value=self.freeze_pca,
                )

                self.client_up_axis_dropdowns[client.client_id] = up_axis_dropdown
                self.client_scale_sliders[client.client_id] = scale_slider
                self.client_mask_blend_sliders[client.client_id] = mask_blend_slider
                self.client_freeze_pca_checkboxes[client.client_id] = freeze_pca_checkbox

                # Add callback for up axis changes
                @up_axis_dropdown.on_update
                def _on_up_axis_change(event) -> None:
                    self.viewer.set_up_axis(client.client_id, event.target.value)

                # Add callback for scale changes
                @scale_slider.on_update
                def _on_scale_change(event) -> None:
                    self.viewer_scale = event.target.value
                    # Trigger re-render when scale changes
                    self.viewer.rerender(None)

                # Add callback for mask blend changes
                @mask_blend_slider.on_update
                def _on_mask_blend_change(event) -> None:
                    self.viewer_mask_blend = event.target.value
                    # Trigger re-render when mask blend changes
                    self.viewer.rerender(None)

                # Add callback for freeze PCA checkbox
                @freeze_pca_checkbox.on_update
                def _on_freeze_pca_change(event) -> None:
                    self.freeze_pca = event.target.value
                    if not self.freeze_pca:
                        # Clear frozen PCA parameters when unfreezing
                        self.frozen_pca_projection = None
                    # Trigger re-render when freeze state changes
                    self.viewer.rerender(None)

            # Add client disconnect handler to clean up
            @self.server.on_client_disconnect
            def _(client: viser.ClientHandle) -> None:
                if client.client_id in self.client_up_axis_dropdowns:
                    del self.client_up_axis_dropdowns[client.client_id]
                if client.client_id in self.client_scale_sliders:
                    del self.client_scale_sliders[client.client_id]
                if client.client_id in self.client_mask_blend_sliders:
                    del self.client_mask_blend_sliders[client.client_id]
                if client.client_id in self.client_freeze_pca_checkboxes:
                    del self.client_freeze_pca_checkboxes[client.client_id]

            self.viewer = Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )
            # Add camera frames to the viewer
            self.camera_frames = []
            self.camera_labels = []

            # Calculate scene scale from camera positions
            camera_positions = []
            for i, data in enumerate(self.train_dataset):
                cam_to_world = data["cam_to_world"].cpu().numpy()
                camera_positions.append(cam_to_world[:3, 3])
            camera_positions = np.array(camera_positions)

            # Calculate scene bounds and scale
            scene_center = np.mean(camera_positions, axis=0)
            scene_radius = np.max(np.linalg.norm(camera_positions - scene_center, axis=1))
            # Scale factors for axes - adjust these multipliers to change relative size
            axes_length = scene_radius * 0.0005  # 5% of scene radius
            axes_radius = axes_length * 0.1  # 10% of axes length

            for i, data in enumerate(self.train_dataset):
                cam_to_world = data["cam_to_world"].cpu().numpy()
                # Create a frame for each camera with scaled axes
                frame = self.server.scene.add_frame(
                    f"camera_{i}",
                    wxyz=self._matrix_to_quaternion(cam_to_world[:3, :3]),
                    position=cam_to_world[:3, 3],
                    axes_length=axes_length,  # Scaled based on scene size
                    axes_radius=axes_radius,  # Scaled based on scene size
                )
                K = data["intrinsics"].cpu().numpy()
                H, W = data["image"].shape[:2]
                fy = K[1, 1]
                frustum = self.server.scene.add_camera_frustum(
                    f"camera_{i}/frustum",
                    fov=2 * np.arctan2(H / 2, fy),
                    aspect=W / H,
                    image=data["image_full"].cpu().numpy(),
                )
                self.camera_frames.append(frame)

    # pose optimization
    def _matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion."""
        # Ensure R is a proper rotation matrix
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = -R

        # Convert to quaternion
        q = np.zeros(4)
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            q[0] = 0.25 * S
            q[1] = (R[2, 1] - R[1, 2]) / S
            q[2] = (R[0, 2] - R[2, 0]) / S
            q[3] = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S
            q[3] = 0.25 * S
        return q

    def _apply_pca_projection(
        self, features: torch.Tensor, n_components: int = 3, valid_feature_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply PCA projection, either using frozen parameters or computing fresh ones."""

        if self.freeze_pca and self.frozen_pca_projection is not None:
            # Use frozen PCA projection matrix
            return pca_projection_fast(features, n_components, V=self.frozen_pca_projection, mask=valid_feature_mask)
        else:
            # Compute fresh PCA
            try:
                result = pca_projection_fast(features, n_components, mask=valid_feature_mask)

                # Store projection matrix if freezing is enabled
                if self.freeze_pca:
                    if valid_feature_mask is not None:
                        features = features[valid_feature_mask]
                    self.frozen_pca_projection = calculate_pca_projection(features, n_components, center=True)

                return result
            except Exception as e:
                if "failed to converge" in str(e):
                    # Fallback: return zeros with correct shape
                    logging.warning("PCA failed to converge, returning zero projection")
                    B, H, W, C = features.shape
                    return torch.zeros(B, H, W, n_components, device=features.device, dtype=features.dtype)
                else:
                    raise e

    def _filter_features_for_robust_pca(self, features: torch.Tensor, valid_feature_mask: torch.Tensor) -> torch.Tensor:
        """Filter features using valid_feature_mask to replace invalid areas with noise."""
        B, H, W, C = features.shape

        # Expand to match feature
        if valid_feature_mask.dim() == 2:
            valid_mask = valid_feature_mask.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        else:
            valid_mask = valid_feature_mask

        invalid_mask = ~valid_mask

        # Count invalid pixels
        num_invalid = invalid_mask.sum()
        total_pixels = invalid_mask.numel()

        if num_invalid > 0:
            logging.debug(f"Replacing {num_invalid} invalid features out of {total_pixels} with noise")

            filtered_features = features.clone()

            # Generate noise with appropriate scale based on valid features
            if valid_mask.any():
                # Use std of valid features to scale noise
                valid_features = features[valid_mask.unsqueeze(-1).expand(-1, -1, -1, C)]
                if len(valid_features) > 0:
                    noise_scale = max(valid_features.std().item() * 0.1, 1e-4)
                else:
                    noise_scale = 1e-4
            else:
                noise_scale = 1e-4

            # Replace invalid areas with noise
            noise = torch.randn_like(features) * noise_scale
            invalid_mask_expanded = invalid_mask.unsqueeze(-1).expand(-1, -1, -1, C)
            filtered_features[invalid_mask_expanded] = noise[invalid_mask_expanded]
        else:
            filtered_features = features

        return filtered_features

    def train(self):
        trainloader = itertools.cycle(
            DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.batch_size * 2,
                persistent_workers=True,
                pin_memory=True,
                collate_fn=GARfVDBInputCollateFn,
            )
        )

        # Create validation dataloader
        valloader = DataLoader(
            self.val_dataset,
            batch_size=min(self.config.batch_size, 4),
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=GARfVDBInputCollateFn,
        )

        testloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=GARfVDBInputCollateFn,
        )

        self.model.train()
        self.optimizer.zero_grad()

        pbar = tqdm.tqdm(range(self.config.num_train_iters))

        # Gradient accumulation settings
        gradient_accumulation_steps = self.config.accumulate_grad_steps
        accumulated_loss = 0.0

        for step in pbar:
            if not self.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            minibatch = next(trainloader)

            # Move to device
            for k, v in minibatch.items():
                minibatch[k] = v.to(self.device)

            # Debug prints for first iteration
            if step == 0:
                logging.debug(f"Training with sample_pixels_per_image={self.config.sample_pixels_per_image}")
                logging.debug(f"Image size: {minibatch['image_full'].shape}")
                logging.debug(f"Batch size: {minibatch['image'].shape[0]}")
                logging.debug(f"scales shape: {minibatch['scales'].shape}")
                logging.debug(f"mask_id shape: {minibatch['mask_id'].shape}")
                logging.debug(f"pixel_coords shape: {minibatch['pixel_coords'].shape}")
                logging.debug(f"mean2d shape: {self.model.gs_model.means.shape}")
                logging.debug(f"Using gradient accumulation over {gradient_accumulation_steps} steps")

            ### Forward pass ###
            cam_enc_feats = self.model.get_encoded_features(minibatch)

            # loss = self.model.calc_loss(cam_enc_feats, minibatch, step)
            loss_dict = calculate_loss(self.model, cam_enc_feats, minibatch)
            loss = loss_dict["total_loss"]

            # Log loss components to tensorboard
            for key, value in loss_dict.items():
                self.writer.add_scalar(f"train/{key}", value.item(), step)

            # Scale loss by accumulation steps to maintain same effective learning rate
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Zero gradients only at the beginning of accumulation cycle
            if step % gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            loss.backward()

            # Perform optimizer step and gradient clipping only after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                # self.scheduler.step()

                # Reset accumulated loss
                accumulated_loss = 0.0

            # For display purposes, show the scaled loss
            display_loss = loss.item() * gradient_accumulation_steps

            pbar.set_postfix(loss=f"{display_loss:.4g}")
            del loss
            torch.cuda.empty_cache()

            # Save model
            if (step + 1) % 1000 == 0:
                self.model.save_checkpoint(f"checkpoints/segmentation_checkpoint_{step}.pt")

            # Evaluate on validation set periodically
            if (step + 1) % self.val_every_n_steps == 0 or step == self.config.num_train_iters - 1:
                val_loss = 0
                with torch.no_grad():
                    for val_batch in valloader:
                        for k, v in val_batch.items():
                            val_batch[k] = v.to(self.device)
                        val_enc_feats = self.model.get_encoded_features(val_batch)
                        val_loss_dict = calculate_loss(self.model, val_enc_feats, val_batch)
                        val_loss += val_loss_dict["total_loss"].item()
                val_loss /= len(valloader)

                # Log validation loss to tensorboard
                self.writer.add_scalar("val/loss", val_loss, step)

                # Log a sample validation image
                with torch.no_grad():
                    cam_to_world_matrix = val_batch["cam_to_world"]  # .cpu().numpy()
                    world_to_cam_matrix = torch.linalg.inv(cam_to_world_matrix).contiguous()
                    projection_matrix = val_batch["intrinsics"]  # .cpu().numpy()
                    img_w, img_h = val_batch["image_full"].shape[2], val_batch["image_full"].shape[1]
                    beauty_output, _ = self.model.gs_model.render_images(
                        world_to_cam_matrix,
                        projection_matrix,
                        img_w,
                        img_h,
                        0.01,
                        1e10,
                        "perspective",
                        # 0,  # sh_degree_to_use
                    )
                    beauty_output = torch.clamp(beauty_output, 0.0, 1.0).cpu()
                    self.writer.add_image("val/beauty_output", beauty_output.permute(0, 3, 1, 2).cpu()[0], step)
                    beauty_gt = val_batch["image_full"]
                    self.writer.add_image("val/beauty_gt", beauty_gt.permute(0, 3, 1, 2)[0].cpu(), step)

                    val_batch_zero = next(iter(valloader)).to(self.device)
                    desired_scale = torch.max(val_batch_zero["scales"]) * 0.05

                    val_mask_output, mask_alpha = self.model.get_mask_output(val_batch_zero, desired_scale.item())
                    logging.debug(f"mask_alpha zeros: {mask_alpha.sum()}")
                    # beauty_output = val_batch_zero["image_full"]
                    pca_output = pca_projection_fast(val_mask_output, 3, mask=mask_alpha.squeeze(-1) > 0)
                    ### Save images

                    pca_output = (pca_output.cpu() * 255).type(torch.uint8)  # [B, H, W, 3]
                    self.writer.add_image("val/sample_mask", pca_output.permute(0, 3, 1, 2)[0], step)
                    alpha = 0.7
                    blended = (beauty_output * (1 - alpha) + pca_output * alpha).type(torch.uint8)
                    # Permute dimensions from [B, H, W, C] to [B, C, H, W] for TensorBoard
                    blended = blended.permute(0, 3, 1, 2)
                    self.writer.add_image("val/sample_image", blended[0], step)

                # Log test loss images
                if self.config.log_test_images:
                    with torch.no_grad():
                        for test_batch_idx, test_batch in enumerate(testloader):
                            if test_batch_idx != 0:
                                break

                            # Log the image
                            # Permute from [H, W, C] to [C, H, W] for TensorBoard
                            test_image = test_batch["image"][0].cpu().permute(2, 0, 1)
                            self.writer.add_image("test/sample_image", test_image, step)
                            # Move input to device
                            for k, v in test_batch.items():
                                test_batch[k] = v.to(self.device)
                            # Set scales to 0.1
                            test_batch["scales"] = torch.full_like(test_batch["scales"], 0.05)
                            test_enc_feats = self.model.get_encoded_features(test_batch)

                            # NOTE: If this starts using too much memory, we can move the image loss calculation to the CPU
                            test_loss_dict = calculate_loss(
                                self.model, test_enc_feats, test_batch, return_loss_images=True
                            )
                            for key in ("instance_loss_1", "instance_loss_2", "instance_loss_4"):
                                self.writer.add_image(
                                    f"test/{key}_img", test_loss_dict[f"{key}_img"].detach().cpu(), step
                                )
                            self.model.to(self.device)

            # Update the viewer
            if not self.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_pixels_in_minibatch = (
                    minibatch["image"].shape[0] * minibatch["image"].shape[1] * minibatch["image"].shape[2]
                )
                num_train_rays_per_sec = num_pixels_in_minibatch * num_train_steps_per_sec
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_pixels_in_minibatch)

        # Close tensorboard writer
        self.writer.close()

    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: CameraState, img_wh: Tuple[int, int]):
        """Callable function for the viewer that renders a blend of the image and mask."""
        img_w, img_h = img_wh
        cam_to_world_matrix = camera_state.c2w
        projection_matrix = camera_state.get_K(img_wh)
        world_to_cam_matrix = torch.linalg.inv(
            torch.from_numpy(cam_to_world_matrix).float().to(self.device)
        ).contiguous()
        projection_matrix = torch.from_numpy(projection_matrix).float().to(self.device)

        # Create a mock input for the model
        mock_input = GARfVDBInput(
            {
                "intrinsics": projection_matrix.unsqueeze(0),
                "cam_to_world": torch.from_numpy(cam_to_world_matrix).float().to(self.device).unsqueeze(0),
                "image_w": torch.tensor([img_w]).to(self.device),
                "image_h": torch.tensor([img_h]).to(self.device),
            }
        )

        try:
            # Render the beauty image
            beauty_colors, _ = self.model.gs_model.render_images(
                world_to_cam_matrix[None],
                projection_matrix[None],
                img_w,
                img_h,
                0.01,
                1e10,
                "perspective",
                # 0,  # sh_degree_to_use
            )
            beauty_rgb = beauty_colors[0, ..., :3]

            # Render the mask features
            mask_features_output, mask_alpha = self.model.get_mask_output(mock_input, self.viewer_scale)

            # Apply PCA projection
            mask_pca = self._apply_pca_projection(
                mask_features_output, 3, valid_feature_mask=mask_alpha.squeeze(-1) > 0
            )[0]

            # Blend between beauty image and mask based on slider value
            alpha = self.viewer_mask_blend
            blended_rgb = beauty_rgb * (1 - alpha) + mask_pca * alpha

            return np.clip(blended_rgb.cpu().numpy(), 0.0, 1.0)

        except Exception as e:
            logging.warning(f"Error in viewer render function: {e}")
            # Return a fallback image on error
            return np.zeros((img_h, img_w, 3), dtype=np.float32)


def train_segmentation(
    checkpoint_path: Path,
    segmentation_dataset_path: str,
    config: Config = Config(),
    disable_viewer: bool = False,
):
    torch.manual_seed(0)
    runner = Runner(config, checkpoint_path, segmentation_dataset_path, disable_viewer=disable_viewer)
    runner.train()
    if not disable_viewer:
        logging.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(train_segmentation)
