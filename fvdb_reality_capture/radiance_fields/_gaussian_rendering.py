# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import torch
from fvdb import CameraModel, GaussianSplat3d, ProjectionMethod

from .gaussian_splat_dataset import SfmDataset

if TYPE_CHECKING:
    from .gaussian_splat_reconstruction import GaussianSplatReconstructionConfig


@dataclass
class RenderOutputs:
    """
    Primary tensors returned by a render backend.

    This groups the main image-like outputs produced by a backend into a named structure so
    call sites can access them by meaning rather than by tuple position. The ``image`` tensor
    contains the primary rendered channels, ``alpha`` contains the accumulated opacity image,
    and ``depth`` is included only when the selected render path also produces a depth channel.
    """

    image: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor | None = None


RenderBackendName = Literal["image_space", "world_space"]


def projection_method_from_config(value: str) -> ProjectionMethod:
    mapping = {
        "auto": ProjectionMethod.AUTO,
        "analytic": ProjectionMethod.ANALYTIC,
        "unscented": ProjectionMethod.UNSCENTED,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported projection_method {value}")
    return mapping[value]


def _camera_model_from_batch(camera_models: torch.Tensor) -> CameraModel:
    unique_camera_models = torch.unique(camera_models.to(dtype=torch.int32))
    if unique_camera_models.numel() != 1:
        raise NotImplementedError("Rendering a minibatch with multiple camera models is not supported")
    return CameraModel(int(unique_camera_models.item()))


def _distortion_coeffs_for_batch(camera_model: CameraModel, distortion_coeffs: torch.Tensor) -> torch.Tensor | None:
    if camera_model in (CameraModel.PINHOLE, CameraModel.ORTHOGRAPHIC):
        return None
    return distortion_coeffs


class RenderBackend(Protocol):
    """
    Interface implemented by Gaussian splat rendering backends.

    A render backend encapsulates one concrete strategy for rendering Gaussian splats, such as
    image-space projection or world-space rendering. Backends are responsible for validating that
    the scene cameras are compatible with the chosen rendering path and for producing the tensors
    consumed during training and evaluation.
    """

    def validate_scene_cameras(
        self,
        model: GaussianSplat3d,
        dataset: SfmDataset,
        config: "GaussianSplatReconstructionConfig",
        device: torch.device,
    ) -> None:
        """
        Validate that the backend can render the camera models present in a scene.

        Implementations should raise an exception early if the dataset contains camera models,
        distortion settings, or batching patterns that the backend cannot support.

        Args:
            model (GaussianSplat3d): Gaussian splat model that will be rendered by the backend.
            dataset (SfmDataset): Dataset whose cameras should be validated.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            device (torch.device): Device on which validation probes should run.
        """
        ...

    def forward_train(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
        crop: tuple[int, int, int, int],
    ) -> RenderOutputs:
        """
        Render a training crop and return the tensors needed for loss computation.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Full image width in pixels before cropping.
            image_height (int): Full image height in pixels before cropping.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.
            crop (tuple[int, int, int, int]): Crop rectangle as ``(origin_w, origin_h, width, height)``.

        Returns:
            RenderOutputs: The rendered training crop, alpha image, and optional depth image.
        """
        ...

    def forward_eval(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> RenderOutputs:
        """
        Render a full evaluation image and return the primary image outputs.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Output image width in pixels.
            image_height (int): Output image height in pixels.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.

        Returns:
            RenderOutputs: The rendered image, alpha image, and optional depth image.
        """
        ...


class ImageSpaceRenderBackend:
    """
    Backend that projects Gaussians in image space before rasterization.

    This backend first projects Gaussians into each camera view using the image-space FVDB APIs
    and then rasterizes those projected Gaussians. It is the natural fit for the classic 3DGS
    rendering path and for renderers that need projected-gaussian intermediates during training.
    """

    def validate_scene_cameras(
        self,
        model: GaussianSplat3d,
        dataset: SfmDataset,
        config: "GaussianSplatReconstructionConfig",
        device: torch.device,
    ) -> None:
        """
        Probe the scene cameras to ensure image-space rendering supports them.

        Args:
            model (GaussianSplat3d): Gaussian splat model used for the probe render.
            dataset (SfmDataset): Dataset whose cameras should be validated.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            device (torch.device): Device on which validation probes should run.
        """
        if config.batch_size > 1 and torch.unique(torch.from_numpy(dataset.camera_models)).numel() > 1:
            raise NotImplementedError("batch_size > 1 is not supported for scenes with multiple camera models")
        self._probe(model, dataset, config, device, render_depth=config.sparse_depth_reg > 0.0)

    def forward_train(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
        crop: tuple[int, int, int, int],
    ) -> RenderOutputs:
        """
        Render a cropped training view using image-space projection and rasterization.

        This path first projects Gaussians into the target camera, then rasterizes just the
        requested crop window. If sparse depth regularization is enabled, it uses the FVDB API
        that also produces a depth channel.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Full image width in pixels before cropping.
            image_height (int): Full image height in pixels before cropping.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.
            crop (tuple[int, int, int, int]): Crop rectangle as ``(origin_w, origin_h, width, height)``.

        Returns:
            RenderOutputs: The rendered crop, alpha image, and optional depth image.
        """
        world_to_camera_matrices = world_to_camera_matrices.contiguous()
        projection_matrices = projection_matrices.contiguous()
        distortion_coeffs = distortion_coeffs.contiguous()
        camera_model = _camera_model_from_batch(camera_models)
        distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model, distortion_coeffs)
        projection_method = projection_method_from_config(config.projection_method)
        projection_function = (
            model.project_gaussians_for_images_and_depths
            if config.sparse_depth_reg > 0.0
            else model.project_gaussians_for_images
        )
        projected_gaussians = projection_function(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=config.near_plane,
            far=config.far_plane,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs_arg,
            sh_degree_to_use=sh_degree_to_use,
            min_radius_2d=config.min_radius_2d,
            eps_2d=config.eps_2d,
            antialias=config.antialias,
        )
        crop_origin_w, crop_origin_h, crop_w, crop_h = crop
        rendered, alphas = model.render_from_projected_gaussians(
            projected_gaussians,
            crop_width=crop_w,
            crop_height=crop_h,
            crop_origin_w=crop_origin_w,
            crop_origin_h=crop_origin_h,
            tile_size=config.tile_size,
        )
        image = rendered[..., : model.num_channels]
        depth = rendered[..., -1:] if rendered.shape[-1] == model.num_channels + 1 else None
        return RenderOutputs(image=image, alpha=alphas, depth=depth)

    def forward_eval(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> RenderOutputs:
        """
        Render a full evaluation image using image-space projection and rasterization.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Output image width in pixels.
            image_height (int): Output image height in pixels.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.

        Returns:
            RenderOutputs: The rendered image and alpha image, plus depth when provided by the
            selected render path.
        """
        world_to_camera_matrices = world_to_camera_matrices.contiguous()
        projection_matrices = projection_matrices.contiguous()
        distortion_coeffs = distortion_coeffs.contiguous()
        camera_model = _camera_model_from_batch(camera_models)
        distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model, distortion_coeffs)
        image, alpha = model.render_images(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=config.near_plane,
            far=config.far_plane,
            camera_model=camera_model,
            projection_method=projection_method_from_config(config.projection_method),
            distortion_coeffs=distortion_coeffs_arg,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=config.tile_size,
            min_radius_2d=config.min_radius_2d,
            eps_2d=config.eps_2d,
            antialias=config.antialias,
        )
        return RenderOutputs(image=image, alpha=alpha)

    @staticmethod
    def _probe(
        model: GaussianSplat3d,
        dataset: SfmDataset,
        config: "GaussianSplatReconstructionConfig",
        device: torch.device,
        render_depth: bool,
    ) -> None:
        """
        Run a lightweight backend probe over the scene camera models.

        This helper renders a minimal example for each distinct camera model present in the
        dataset so unsupported camera/distortion combinations fail early, before optimization
        starts.

        Args:
            model (GaussianSplat3d): Gaussian splat model used for the probe render.
            dataset (SfmDataset): Dataset whose distinct camera models should be probed.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            device (torch.device): Device on which validation probes should run.
            render_depth (bool): Whether the probe should use the depth-producing render API.
        """
        if len(dataset) == 0:
            return
        seen: set[int] = set()
        projection_method = projection_method_from_config(config.projection_method)
        with torch.no_grad():
            for dataset_idx, scene_idx in enumerate(dataset.indices):
                camera_model = int(dataset.sfm_scene.images[scene_idx].camera_metadata.camera_model)
                if camera_model in seen:
                    continue
                seen.add(camera_model)
                datum = dataset[dataset_idx]
                world_to_camera = datum["world_to_camera"].unsqueeze(0).to(device)
                projection = datum["projection"].unsqueeze(0).to(device)
                distortion_coeffs = datum["distortion_coeffs"].unsqueeze(0).to(device)
                world_to_camera = world_to_camera.contiguous()
                projection = projection.contiguous()
                distortion_coeffs = distortion_coeffs.contiguous()
                height, width = datum["image"].shape[:2]
                camera_model_enum = CameraModel(camera_model)
                distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model_enum, distortion_coeffs)
                if render_depth:
                    model.project_gaussians_for_images_and_depths(
                        world_to_camera_matrices=world_to_camera,
                        projection_matrices=projection,
                        image_width=width,
                        image_height=height,
                        near=config.near_plane,
                        far=config.far_plane,
                        camera_model=camera_model_enum,
                        projection_method=projection_method,
                        distortion_coeffs=distortion_coeffs_arg,
                        sh_degree_to_use=0,
                        min_radius_2d=config.min_radius_2d,
                        eps_2d=config.eps_2d,
                        antialias=config.antialias,
                    )
                else:
                    model.project_gaussians_for_images(
                        world_to_camera_matrices=world_to_camera,
                        projection_matrices=projection,
                        image_width=width,
                        image_height=height,
                        near=config.near_plane,
                        far=config.far_plane,
                        camera_model=camera_model_enum,
                        projection_method=projection_method,
                        distortion_coeffs=distortion_coeffs_arg,
                        sh_degree_to_use=0,
                        min_radius_2d=config.min_radius_2d,
                        eps_2d=config.eps_2d,
                        antialias=config.antialias,
                    )


class WorldSpaceRenderBackend:
    """
    Backend that renders directly from world-space Gaussians.

    This backend uses the world-space FVDB rendering APIs directly. It avoids the explicit
    projected-gaussian intermediate used by the image-space backend while preserving the same
    high-level interface expected by :class:`GaussianSplatReconstruction`.
    """

    def validate_scene_cameras(
        self,
        model: GaussianSplat3d,
        dataset: SfmDataset,
        config: "GaussianSplatReconstructionConfig",
        device: torch.device,
    ) -> None:
        """
        Probe the scene cameras to ensure world-space rendering supports them.

        Args:
            model (GaussianSplat3d): Gaussian splat model used for the probe render.
            dataset (SfmDataset): Dataset whose cameras should be validated.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            device (torch.device): Device on which validation probes should run.
        """
        if config.batch_size > 1 and torch.unique(torch.from_numpy(dataset.camera_models)).numel() > 1:
            raise NotImplementedError("batch_size > 1 is not supported for scenes with multiple camera models")
        if len(dataset) == 0:
            return
        seen: set[int] = set()
        projection_method = projection_method_from_config(config.projection_method)
        with torch.no_grad():
            for dataset_idx, scene_idx in enumerate(dataset.indices):
                camera_model = int(dataset.sfm_scene.images[scene_idx].camera_metadata.camera_model)
                if camera_model in seen:
                    continue
                seen.add(camera_model)
                datum = dataset[dataset_idx]
                world_to_camera = datum["world_to_camera"].unsqueeze(0).to(device)
                projection = datum["projection"].unsqueeze(0).to(device)
                distortion_coeffs = datum["distortion_coeffs"].unsqueeze(0).to(device)
                world_to_camera = world_to_camera.contiguous()
                projection = projection.contiguous()
                distortion_coeffs = distortion_coeffs.contiguous()
                height, width = datum["image"].shape[:2]
                camera_model_enum = CameraModel(camera_model)
                distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model_enum, distortion_coeffs)
                render_function = (
                    model.render_images_and_depths_from_world
                    if config.sparse_depth_reg > 0.0
                    else model.render_images_from_world
                )
                render_function(
                    world_to_camera_matrices=world_to_camera,
                    projection_matrices=projection,
                    image_width=width,
                    image_height=height,
                    near=config.near_plane,
                    far=config.far_plane,
                    camera_model=camera_model_enum,
                    projection_method=projection_method,
                    distortion_coeffs=distortion_coeffs_arg,
                    sh_degree_to_use=0,
                    tile_size=config.tile_size,
                    min_radius_2d=config.min_radius_2d,
                    eps_2d=config.eps_2d,
                    antialias=config.antialias,
                )

    def forward_train(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
        crop: tuple[int, int, int, int],
    ) -> RenderOutputs:
        """
        Render a cropped training view directly from world-space Gaussians.

        The world-space renderer produces a full image for the requested camera batch, after which
        this backend slices out the requested crop region so the rest of the reconstruction code
        can work with the same crop-based interface as the image-space backend.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Full image width in pixels before cropping.
            image_height (int): Full image height in pixels before cropping.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.
            crop (tuple[int, int, int, int]): Crop rectangle as ``(origin_w, origin_h, width, height)``.

        Returns:
            RenderOutputs: The rendered crop, alpha image, and optional depth image.
        """
        world_to_camera_matrices = world_to_camera_matrices.contiguous()
        projection_matrices = projection_matrices.contiguous()
        distortion_coeffs = distortion_coeffs.contiguous()
        camera_model = _camera_model_from_batch(camera_models)
        distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model, distortion_coeffs)
        render_function = (
            model.render_images_and_depths_from_world
            if config.sparse_depth_reg > 0.0
            else model.render_images_from_world
        )
        rendered, alpha = render_function(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=config.near_plane,
            far=config.far_plane,
            camera_model=camera_model,
            projection_method=projection_method_from_config(config.projection_method),
            distortion_coeffs=distortion_coeffs_arg,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=config.tile_size,
            min_radius_2d=config.min_radius_2d,
            eps_2d=config.eps_2d,
            antialias=config.antialias,
        )
        crop_origin_w, crop_origin_h, crop_w, crop_h = crop
        rendered = rendered[:, crop_origin_h : crop_origin_h + crop_h, crop_origin_w : crop_origin_w + crop_w]
        alpha = alpha[:, crop_origin_h : crop_origin_h + crop_h, crop_origin_w : crop_origin_w + crop_w]
        image = rendered[..., : model.num_channels]
        depth = rendered[..., -1:] if rendered.shape[-1] == model.num_channels + 1 else None
        return RenderOutputs(image=image, alpha=alpha, depth=depth)

    def forward_eval(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> RenderOutputs:
        """
        Render a full evaluation image directly from world-space Gaussians.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Output image width in pixels.
            image_height (int): Output image height in pixels.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.

        Returns:
            RenderOutputs: The rendered image and alpha image, plus depth when provided by the
            selected render path.
        """
        world_to_camera_matrices = world_to_camera_matrices.contiguous()
        projection_matrices = projection_matrices.contiguous()
        distortion_coeffs = distortion_coeffs.contiguous()
        camera_model = _camera_model_from_batch(camera_models)
        distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model, distortion_coeffs)
        image, alpha = model.render_images_from_world(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=config.near_plane,
            far=config.far_plane,
            camera_model=camera_model,
            projection_method=projection_method_from_config(config.projection_method),
            distortion_coeffs=distortion_coeffs_arg,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=config.tile_size,
            min_radius_2d=config.min_radius_2d,
            eps_2d=config.eps_2d,
            antialias=config.antialias,
        )
        return RenderOutputs(image=image, alpha=alpha)


def make_render_backend(name: RenderBackendName) -> RenderBackend:
    if name == "image_space":
        return ImageSpaceRenderBackend()
    if name == "world_space":
        return WorldSpaceRenderBackend()
    raise ValueError(f"Unsupported render_backend {name}")
