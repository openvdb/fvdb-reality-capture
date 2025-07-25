# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch
import torch.nn.functional as F


class CameraPoseAdjustment(torch.nn.Module):
    """Camera pose optimization module for 3D Gaussian Splatting.

    This module enables optimization of camera poses defined by their Camera-to-World
    transform. It's generally used to jointly optimize camera poses during training of a
    3D Gaussian Splatting model.

    The model learns a transformation *delta* which applies to the original camera-to-world
    transforms in a dataset.

    The delta is represented as a 9D vector `[dx, dy, dz, r1, r2, r3, r4, r5, r6]`
    which encodes a change in translation and a change in rotation.
    The nine components of the vector are:
    - `[dx, dy, dz]`: translation deltas in world coordinates
    - `[r1, r2, r3, r4, r5, r6]`: 6D rotation representation for stable optimization in machine
    learning, as described in "On the Continuity of Rotation Representations in Neural Networks"
    (Zhou et al., 2019). This representation is preferred over Euler angles or quaternions for
    optimization stability and avoids singularities.

    The module uses an embedding layer to learn these deltas for a fixed number of cameras
    specified at initialization. Generally, this is the number of cameras in the training dataset.

    You apply this module to a batch of camera-to-world transforms by passing the transforms
    and their corresponding camera indices (in the range `[0, num_cameras-1]`() to the
    `forward` method. The module will return the updated camera-to-world transforms
    after applying the learned deltas.

    Attributes:
        pose_embeddings (torch.nn.Embedding): Embedding layer for learning camera pose deltas.
    """

    def __init__(self, num_poses: int, init_std: float = 1e-4):
        """
        Create a new `CameraPoseAdjustment` module for storing changes in camera-to-world transforms
        for a fixed number of poses (`num_poses`).

        Args:
            num_poses (int): Number of poses to learn deltas for.
            init_std (float): Standard deviation for the normal distribution used to initialize
                the pose embeddings.
        """
        super().__init__()

        # Change in positions (3D) + Change in rotations (6D)
        self.pose_embeddings: torch.nn.Embedding = torch.nn.Embedding(num_poses, 9)

        torch.nn.init.normal_(self.pose_embeddings.weight, std=init_std)

        # Identity rotation in 6D representation
        self.register_buffer("_identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    @property
    def num_poses(self) -> int:
        """
        Return the number of poses this module is initialized for.

        Returns:
            int: The number of poses (cameras) this module can adjust.
        """
        return self.pose_embeddings.num_embeddings

    @staticmethod
    def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation described in [1] to a rotation matrix.

        This method uses the Gram-Schmid orthogonalization schemed described in Section B of [1].

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035

        Args:
            d6 (torch.Tensor): 6D rotation representation tensor with shape (*, 6)

        Returns:
            torch.Tensor: batch of rotation matrices with shape (*, 3, 3)
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def forward(self, cam_to_world_matrices: torch.Tensor, image_ids: torch.Tensor) -> torch.Tensor:
        """Adjust camera pose based on deltas.

        Args:
            cam_to_world_matrices (torch.Tensor): A batch of camera to world transformations
                to adjust. Tnsor of shape (*, 4, 4) where B is the batch size.
            image_ids (torch.Tensor): Indices of images in the batch in the range
            `[0, self.num_poses -1]`. Tensor of shape (*,).

        Returns:
            torch.Tensor: A batch of updated cam_to_world_matrices where we've applied the
                learned deltas for camera ids to the input camera-to-world transforms.
                i.e. `output[i] = cam_to_world_matrices[i] @ transform[image_ids[i]]`
        """
        if cam_to_world_matrices.shape[:-2] != image_ids.shape:
            raise ValueError("`cam_to_world_matrices` and `camera_ids` must have the same batch shape.")
        if cam_to_world_matrices.shape[-2:] != (4, 4):
            raise ValueError("`cam_to_world_matrices` must have shape (..., 4, 4).")

        batch_shape = cam_to_world_matrices.shape[:-2]
        pose_deltas = self.pose_embeddings(image_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = self._rotation_6d_to_matrix(drot + self._identity.expand(*batch_shape, -1))  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(cam_to_world_matrices, transform)
