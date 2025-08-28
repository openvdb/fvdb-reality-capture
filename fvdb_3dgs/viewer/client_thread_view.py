# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import abc

import torch


class ClientThreadRenderingView(abc.ABC):
    """
    Abstract base class for rendering views in a client thread.
    This class defines the interface for rendering views that can be used in a client thread.
    """

    @abc.abstractmethod
    def _render(
        self,
        current_frame: torch.Tensor | None,
        current_depth: torch.Tensor | None,
        world_to_cam_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
        img_width: int,
        img_height: int,
        near: float,
        far: float,
        camera_model: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Render the frame with the given camera and light settings.
        """
        pass

    @abc.abstractmethod
    def layout_gui(self):
        pass
