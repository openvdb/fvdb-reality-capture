# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch
from fvdb.types import (
    NumericMaxRank2,
    NumericMaxRank3,
    to_Mat33fBatch,
    to_Mat44fBatch,
    to_Vec2iBatch,
)


def validate_camera_matrices_and_image_sizes(
    camera_to_world_matrices: NumericMaxRank3,
    projection_matrices: NumericMaxRank3,
    image_sizes: NumericMaxRank2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Validate the shapes of the camera matrices and image sizes. This function converts input
    tensor-like objects to torch.Tensors and checks their shapes.

    The constraints are:
    - camera_to_world_matrices must have shape (C, 4, 4) where C is the number of cameras.
    - projection_matrices must have shape (C, 3, 3) where C is the number of cameras.
    - image_sizes must have shape (C, 2) where C is the number of cameras.

    Args:
        camera_to_world_matrices (torch.Tensor): A (C, 4, 4)-shaped Tensor-like object containing camera to world
            matrices where C is the number of camera views.
        projection_matrices (torch.Tensor): A (C, 3, 3)-shaped Tensor-like object containing perspective projection matrices
            where C is the number of camera views.
        image_sizes (torch.Tensor): A (C, 2)-shaped Tensor-like object containing the width and height of each image.

        Returns:
            valid_camera_to_world_matrices (torch.Tensor): The validated camera to world matrices with shape (C, 4, 4).
            valid_projection_matrices (torch.Tensor): The validated projection matrices with shape (C, 3, 3).
            valid_image_sizes (torch.Tensor): The validated image sizes with shape (C, 2).
    """
    camera_to_world_matrices = to_Mat44fBatch(camera_to_world_matrices)

    num_cameras = camera_to_world_matrices.shape[0]
    if camera_to_world_matrices.shape != (num_cameras, 4, 4):
        raise ValueError(
            f"Expected camera_to_world_matrices to have shape (C, 4, 4) where C is the number of cameras, but got {camera_to_world_matrices.shape}"
        )

    projection_matrices = to_Mat33fBatch(projection_matrices)
    if projection_matrices.shape != (num_cameras, 3, 3):
        raise ValueError(
            f"Expected projection_matrices to have shape (C, 3, 3) where C is the number of cameras, but got {projection_matrices.shape}"
        )

    image_sizes = to_Vec2iBatch(image_sizes)
    if image_sizes.shape != (num_cameras, 2):
        raise ValueError(
            f"Expected image_sizes to have shape (C, 2) where C is the number of cameras, but got {image_sizes.shape}"
        )

    return camera_to_world_matrices, projection_matrices, image_sizes
