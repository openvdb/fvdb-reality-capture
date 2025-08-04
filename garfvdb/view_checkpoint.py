# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from garfvdb.dataset import SegmentationDataset
from garfvdb.model import GARfVDBInput, GARfVDBModel
from garfvdb.util import calculate_pca_projection, pca_projection_fast
from viz import CameraState, Viewer

np.set_printoptions(suppress=True)
import tyro
import viser


def main(segmentation_dataset_path: Path, garfvdb_checkpoint_path: Path, gsplat_checkpoint_path: Path):

    device = torch.device("cuda")

    full_dataset = SegmentationDataset(segmentation_dataset_path)
    grouping_scale_stats = torch.cat(full_dataset.scales)

    model = GARfVDBModel.create_from_checkpoint(garfvdb_checkpoint_path, gsplat_checkpoint_path, grouping_scale_stats)

    client_up_axis_dropdowns = {}
    client_scale_sliders = {}
    client_mask_blend_sliders = {}
    client_freeze_pca_checkboxes = {}
    frozen_pca_projection = None

    freeze_pca = False
    viewer_scale = 0.1
    viewer_mask_blend = 0.5
    current_up_axis = "+z"

    def _apply_pca_projection(
        features: torch.Tensor, n_components: int = 3, valid_feature_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply PCA projection, either using frozen parameters or computing fresh ones."""
        nonlocal frozen_pca_projection

        if freeze_pca and frozen_pca_projection is not None:
            # Use frozen PCA projection matrix
            return pca_projection_fast(features, n_components, V=frozen_pca_projection, mask=valid_feature_mask)
        else:
            # Compute fresh PCA
            try:
                result = pca_projection_fast(features, n_components, mask=valid_feature_mask)

                # Store projection matrix if freezing is enabled
                if freeze_pca:
                    if valid_feature_mask is not None:
                        features = features[valid_feature_mask]
                    frozen_pca_projection = calculate_pca_projection(features, n_components, center=True)

                return result
            except RuntimeError as e:
                if "failed to converge" in str(e):
                    # Fallback: return zeros with correct shape
                    logging.warning("PCA failed to converge, returning zero projection")
                    B, H, W, C = features.shape
                    return torch.zeros(B, H, W, n_components, device=features.device, dtype=features.dtype)
                else:
                    raise e

    @torch.no_grad()
    def _viewer_render_fn(camera_state: CameraState, img_wh: Tuple[int, int]):
        """Callable function for the viewer that renders a blend of the image and mask."""
        img_w, img_h = img_wh
        cam_to_world_matrix = camera_state.c2w
        projection_matrix = camera_state.get_K(img_wh)
        world_to_cam_matrix = torch.linalg.inv(torch.from_numpy(cam_to_world_matrix).float().to(device)).contiguous()
        projection_matrix = torch.from_numpy(projection_matrix).float().to(device)

        # Create a mock input for the model
        mock_input = GARfVDBInput(
            {
                "intrinsics": projection_matrix.unsqueeze(0),
                "cam_to_world": torch.from_numpy(cam_to_world_matrix).float().to(device).unsqueeze(0),
                "image_w": torch.tensor([img_w]).to(device),
                "image_h": torch.tensor([img_h]).to(device),
            }
        )

        try:
            # Render the beauty image
            beauty_colors, _ = model.gs_model.render_images(
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
            mask_features_output, mask_alpha = model.get_mask_output(mock_input, viewer_scale)

            # Apply PCA projection
            mask_pca = _apply_pca_projection(mask_features_output, 3, valid_feature_mask=mask_alpha.squeeze(-1) > 0)[0]

            # Blend between beauty image and mask based on slider value
            alpha = viewer_mask_blend
            blended_rgb = beauty_rgb * (1 - alpha) + mask_pca * alpha

            return np.clip(blended_rgb.cpu().numpy(), 0.0, 1.0)

        except Exception as e:
            logging.warning(f"Error in viewer render function: {e}")
            # Return a fallback image on error
            return np.zeros((img_h, img_w, 3), dtype=np.float32)

    server = viser.ViserServer(port=8080, verbose=False)
    server.scene.set_up_direction(current_up_axis)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # up axis dropdown
        up_axis_dropdown = client.gui.add_dropdown(
            "Up Axis",
            options=["+x", "-x", "+y", "-y", "+z", "-z"],
            initial_value=current_up_axis,
        )

        # scale slider
        scale_slider = client.gui.add_slider(
            "Scale",
            min=0.0,
            max=model.max_scale,
            step=0.01,
            initial_value=viewer_scale,
        )

        # mask blend slider
        mask_blend_slider = client.gui.add_slider(
            "Mask Blend",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=viewer_mask_blend,
        )

        # freeze PCA checkbox
        freeze_pca_checkbox = client.gui.add_checkbox(
            "Freeze PCA Projection",
            initial_value=freeze_pca,
        )

        client_up_axis_dropdowns[client.client_id] = up_axis_dropdown
        client_scale_sliders[client.client_id] = scale_slider
        client_mask_blend_sliders[client.client_id] = mask_blend_slider
        client_freeze_pca_checkboxes[client.client_id] = freeze_pca_checkbox

        # Add callback for up axis changes
        @up_axis_dropdown.on_update
        def _on_up_axis_change(event) -> None:
            viewer.set_up_axis(client.client_id, event.target.value)

        # Add callback for scale changes
        @scale_slider.on_update
        def _on_scale_change(event) -> None:
            nonlocal viewer_scale
            viewer_scale = event.target.value
            # Trigger re-render when scale changes
            viewer.rerender(None)

        # Add callback for mask blend changes
        @mask_blend_slider.on_update
        def _on_mask_blend_change(event) -> None:
            nonlocal viewer_mask_blend
            viewer_mask_blend = event.target.value
            # Trigger re-render when mask blend changes
            viewer.rerender(None)

        # Add callback for freeze PCA checkbox
        @freeze_pca_checkbox.on_update
        def _on_freeze_pca_change(event) -> None:
            nonlocal freeze_pca, frozen_pca_projection
            freeze_pca = event.target.value
            if not freeze_pca:
                # Clear frozen PCA parameters when unfreezing
                frozen_pca_projection = None
            # Trigger re-render when freeze state changes
            viewer.rerender(None)

    # Add client disconnect handler to clean up
    @server.on_client_disconnect
    def _(client: viser.ClientHandle) -> None:
        if client.client_id in client_up_axis_dropdowns:
            del client_up_axis_dropdowns[client.client_id]
        if client.client_id in client_scale_sliders:
            del client_scale_sliders[client.client_id]
        if client.client_id in client_mask_blend_sliders:
            del client_mask_blend_sliders[client.client_id]
        if client.client_id in client_freeze_pca_checkboxes:
            del client_freeze_pca_checkboxes[client.client_id]

    viewer = Viewer(
        server=server,
        render_fn=_viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
