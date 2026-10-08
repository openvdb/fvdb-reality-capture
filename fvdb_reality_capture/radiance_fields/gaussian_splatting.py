# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import math
import pathlib
from typing import Any, Mapping, Sequence, TypeVar, overload

import torch
from fvdb import functional as fvdb_functional
from fvdb.grid import Grid
from fvdb.grid_batch import GridBatch
from fvdb.jagged_tensor import JaggedTensor
from fvdb.types import DeviceIdentifier, cast_check, resolve_device

from ..enums import CameraModel, GaussianRenderMode, ProjectionMethod
from ..functional import (
    Crop,
    GaussianTileIntersection,
    ProjectedGaussians,
    as_pixel_jagged,
    compute_gaussian_opacities,
    evaluate_gaussian_sh,
    intersect_gaussian_tiles,
    intersect_gaussian_tiles_sparse,
    project_gaussians,
    rasterize_contributing_gaussian_ids,
    rasterize_contributing_gaussian_ids_sparse,
    rasterize_num_contributing_gaussians,
    rasterize_num_contributing_gaussians_sparse,
    rasterize_screen_space_gaussians,
    rasterize_screen_space_gaussians_sparse,
    pad_crop,
    rasterize_world_space_gaussians,
    sh_degree_from_coefficients,
    validate_crop,
)
from ..functional._autograd import (
    _EvaluateGaussianSHFn,
    _ProjectGaussiansJaggedFn,
    _RasterizeScreenSpaceGaussiansFn,
)

JaggedTensorOrTensorT = TypeVar("JaggedTensorOrTensorT", JaggedTensor, torch.Tensor)


class ProjectedGaussianSplats:
    """
    A class representing a set of Gaussian splats projected onto a batch of 2D image planes.

    A :class:`ProjectedGaussianSplats` instance contains the 2D projections of 3D Gaussian splats, which can be used to render
    images onto the image planes. Instances of this class are created by calling the :meth:`GaussianSplat3d.project_gaussians_for_images`,
    :meth:`GaussianSplat3d.project_gaussians_for_images_and_depths`, etc. methods.

    .. note::

        The reason to have a separate class for projected Gaussian splats is to be able to run projection once, and then render
        the splats multiple times (e.g. rendering crops) without re-projecting them each time. This can save significant computation time.
    """

    __PRIVATE__ = object()

    def __init__(
        self,
        *,
        projected: ProjectedGaussians,
        render_quantities: torch.Tensor,
        logit_opacities: torch.Tensor,
        antialias: bool,
        eps_2d: float,
        near_plane: float,
        far_plane: float,
        min_radius_2d: float,
        sh_degree_to_use: int,
        opacities: torch.Tensor,
        _private: Any = None,
    ) -> None:
        """
        Private constructor. Use :meth:`GaussianSplat3d.project_gaussians_for_images` or similar methods to create instances.

        ``opacities`` are the per-camera opacities of ``projected``, ``(C, N)``, computed by the projection
        helper alongside the features so the projection is a consistent snapshot of the model.
        """
        if _private is not self.__PRIVATE__:
            raise ValueError(
                "ProjectedGaussianSplats constructor is private. Use GaussianSplat3d.project_gaussians_for_images or similar methods instead."
            )
        self._projected = projected
        self._render_quantities = render_quantities
        self._logit_opacities = logit_opacities
        self._antialias = antialias
        self._eps_2d = eps_2d
        self._near_plane = near_plane
        self._far_plane = far_plane
        self._min_radius_2d = min_radius_2d
        self._sh_degree_to_use = sh_degree_to_use
        self._opacities = opacities

    @property
    def projected_gaussians(self) -> ProjectedGaussians:
        """
        Return the underlying :class:`fvdb_reality_capture.functional.ProjectedGaussians`, the stage-1 output of the
        composable pipeline, for use with the functions in :mod:`fvdb_reality_capture.functional`.

        Returns:
            projected_gaussians (ProjectedGaussians): The projected Gaussians without features or opacities.
        """
        return self._projected

    def tile_intersection(self, tile_size: int = 16) -> GaussianTileIntersection:
        """
        Compute the tile intersections of the projected Gaussians for a tile size.

        Nothing is cached, so holding a projection does not hold tile buffers. A caller rendering several
        crops from one projection should keep the result and pass it to :meth:`GaussianSplat3d.render_from_projected_gaussians`
        as ``tiles`` (see its docstring example) or to the rasterization stages in :mod:`fvdb_reality_capture.functional`.

        Args:
            tile_size (int): The tile side length in pixels. Default is 16.

        Returns:
            tiles (GaussianTileIntersection): The tile intersections, as consumed by the rasterization stages
                in :mod:`fvdb_reality_capture.functional`.
        """
        return intersect_gaussian_tiles(self._projected, tile_size=tile_size, opacities=self.opacities)

    @property
    def logit_opacities(self) -> torch.Tensor:
        """
        Return the logit opacities of the Gaussians that were projected.

        Returns:
            logit_opacities (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number of projected Gaussians.
        """
        return self._logit_opacities

    @property
    def antialias(self) -> bool:
        """
        Return whether antialiasing was enabled during the projection of the Gaussian splats.

        Returns:
            antialias (bool): ``True`` if antialiasing was enabled during projection, ``False`` otherwise.
        """
        return self._antialias

    @property
    def inv_covar_2d(self) -> torch.Tensor:
        """
        The inverse of the 2D covariance matrices of the Gaussians projected into each image plane. These define the
        spatial extent of ellipses for each splatted Gaussian. Note that
        since covariance matrices are symmetric, we pack them into a tensor of shape ``(num_projected_gaussians, 3)``
        where each covariance matrix is represented as ``(Cxx, Cxy, Cyy)``.

        Returns:
            inv_covar_2d (torch.Tensor): A tensor of shape ``(C, N, D)`` representing the packed inverse 2D covariance matrices,
                where ``C`` is the number of image planes, ``N`` is the number of projected Gaussians, and ``D`` is number of feature channels for each
                Gaussian (see :attr:`GaussianSplat3d.num_channels`).
        """
        return self._projected.conics

    @property
    def depths(self) -> torch.Tensor:
        """
        Return the depth of each projected Gaussian in each image plane. The depth is defined as the
        distance from the camera to the mean of the Gaussian along the camera's viewing direction.

        Returns:
            depths (torch.Tensor): A tensor of shape ``(C, N)`` representing the depth of each projected Gaussian, where
                ``C`` is the number of image planes, and ``N`` is the number of projected Gaussians.
        """
        return self._projected.depths

    @property
    def eps_2d(self) -> float:
        """
        Return the epsilon value used during the projection of the Gaussian splats to avoid
        numerical issues. This value is used to clamp very small radii during projection.

        Returns:
            eps_2d (float): The epsilon value used during projection.
        """
        return self._eps_2d

    @property
    def far_plane(self) -> float:
        """
        Return the far plane distance used during the projection of the Gaussian splats.

        Returns:
            far_plane (float): The far plane distance.
        """
        return self._far_plane

    @property
    def image_height(self) -> int:
        """
        Return the height of the image planes used during the projection of the Gaussian splats.

        Returns:
            image_height (int): The height of the image planes.
        """
        return self._projected.image_height

    @property
    def image_width(self) -> int:
        """
        Return the width of the image planes used during the projection of the Gaussian splats.

        Returns:
            image_width (int): The width of the image planes.
        """
        return self._projected.image_width

    @property
    def means2d(self) -> torch.Tensor:
        """
        Return the 2D projected means (in pixel units) of the Gaussians in each image plane.

        Returns:
            means2d (torch.Tensor): A tensor of shape ``(C, N, 2)`` representing the 2D projected means,
                where ``C`` is the number of image planes, ``N`` is the number of projected Gaussians,
                and the last dimension contains the (x, y) coordinates of the means in pixel space.
        """
        return self._projected.means2d

    @property
    def min_radius_2d(self) -> float:
        """
        Return the minimum radius (in pixels) used to clip Gaussians during projection. Gaussians
        whose radius projected to less than this value are ignored to avoid numerical issues.

        Returns:
            min_radius_2d (float): The minimum radius used during projection.
        """
        return self._min_radius_2d

    @property
    def near_plane(self) -> float:
        """
        Return the near plane distance used during the projection of the Gaussian splats.

        Returns:
            near_plane (float): The near plane distance.
        """
        return self._near_plane

    @property
    def opacities(self) -> torch.Tensor:
        """
        Return the opacities of each projected Gaussian in each image plane.

        They are computed when the Gaussians are projected, together with the features, so they reflect
        the model at that moment and carry a gradient exactly when the projection was made with one.

        Returns:
            opacities (torch.Tensor): A tensor of shape ``(C, N)`` representing the opacity of each projected Gaussian, where
                ``C`` is the number of image planes, and ``N`` is the number of projected Gaussians.
        """
        return self._opacities

    @property
    def camera_model(self) -> CameraModel:
        """
        Return the camera model used during projection.

        Returns:
            camera_model (CameraModel): The camera model used during projection.
        """
        return self._projected.camera_model

    @property
    def projection_method(self) -> ProjectionMethod:
        """
        Return the resolved projection method used during projection.

        Returns:
            projection_method (ProjectionMethod): The resolved projection method.
        """
        return self._projected.projection_method

    @property
    def radii(self) -> torch.Tensor:
        """
        Return the per-axis 2D radii (in pixels) of each projected Gaussian in each image plane.
        Entry ``(c, n, 0)`` is the half-width of the AABB along x and ``(c, n, 1)`` is the
        half-width along y. A Gaussian is considered visible iff both axes are positive.

        Returns:
            radii (torch.Tensor): A tensor of shape ``(C, N, 2)`` representing the per-axis 2D
                radius of each projected Gaussian.
        """
        return self._projected.radii

    @property
    def render_quantities(self) -> torch.Tensor:
        """
        Return the render quantities of each projected Gaussian in each image plane. The render quantities
        are used for shading and lighting calculations during rendering.

        Returns:
            render_quantities (torch.Tensor): A tensor of shape ``(C, N, D)`` representing the render quantities of each projected Gaussian,
                where ``C`` is the number of image planes, ``N`` is the number of projected Gaussians, and ``D`` is the number of feature
                channels for each Gaussian (see :attr:`GaussianSplat3d.num_channels`).
        """
        return self._render_quantities

    @property
    def sh_degree_to_use(self) -> int:
        """
        Return the spherical harmonic degree used during the projection of the Gaussian splats.

        .. note::

            This indicates up to which degree the spherical harmonics coefficients were projected
            for each Gaussian. For example, if this value is ``0``, only the diffuse (degree 0) coefficients
            were projected. If this value is ``2``, coefficients up to degree 2 were projected.

        Returns:
            sh_degree_to_use (int): The spherical harmonic degree used during projection.
        """
        return self._sh_degree_to_use


class GaussianSplat3d:
    """
    An efficient data structure representing a Gaussian splat radiance field in 3D space.

    A :class:`GaussianSplat3d` instance contains a set of 3D Gaussian splats, each defined by its mean position,
    orientation (quaternion), scale, opacity, and spherical harmonics coefficients for color representation.

    Together, these define a radiance field which can be volume rendered to produce images and depths from
    arbitrary viewpoints. This class provides a variety of methods for rendering and manipulating Gaussian splats radiance fields.
    World-to-camera matrices supplied to its rendering methods are interpreted as rigid transforms
    with orthonormal rotation blocks.
    These include:

    - Rendering images with arbitrary channels using spherical harmonics for view-dependent color
      representation (:meth:`render_images`, :meth:`render_images_and_depths`).
    - Rendering depth maps (:meth:`render_depths`, :meth:`render_images_and_depths`).
    - Rendering features at arbitrary sparse pixel locations (:meth:`sparse_render_images`, :meth:`sparse_render_images_and_depths`).
    - Rendering depths at arbitrary sparse pixel locations (:meth:`sparse_render_depths`).
    - Computing which gaussians contribute to each pixel in an image plane
      (:meth:`render_num_contributing_gaussians`, :meth:`render_contributing_gaussian_ids`).
    - Computing the set of Gaussians which contribute to a set of sparse pixel locations
      (:meth:`sparse_render_num_contributing_gaussians`, :meth:`sparse_render_contributing_gaussian_ids`).
    - Saving and loading Gaussian splat data to/from PLY files (:meth:`save_to_ply`, :meth:`from_ply`).
    - Slicing, indexing, and masking Gaussians to create new :class:`GaussianSplat3d` instances.
    - Concatenating multiple :class:`GaussianSplat3d` instances into a single instance (:meth:`cat`).

    Background
    -----------

    Mathematically, the radiance field represented by a :class:`GaussianSplat3d` is defined as a sum of anisotropic 3D Gaussians,
    with view-dependent features represented using spherical harmonics. The radiance field :math:`R(x, v)` accepts as
    input a 3D position :math:`x \\in \\mathbb{R}^3` and a viewing direction :math:`v \\in \\mathbb{S}^2`, and is defined as:

    .. math::

        R(x, v) = \\sum_{i=1}^{N} o_i \\cdot \\alpha_i(x) \\cdot SH(v; C_i)

        \\alpha_i(x) = \\exp\\left(-\\frac{1}{2}(x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i)\\right)

        \\Sigma_i = R(q_i)^T \\cdot \\text{diag}(S_i) \\cdot R(q_i)

    where:

    - :math:`N` is the number of Gaussians (see :attr:`num_gaussians`).
    - :math:`\\mu_i \\in \\mathbb{R}^3` is the mean of the i-th Gaussian (see :attr:`means`).
    - :math:`\\Sigma_i \\in \\mathbb{R}^{3 \\times 3}` is the covariance matrix of the i-th Gaussian,
      defined by its scale diagonal scale :math:`S_i \\in \\mathbb{R}^3` (see :attr:`scales`) and orientation
      quaternion :math:`q_i \\in \\mathbb{R}^4` (see :attr:`quats`).
    - :math:`o_i \\in [0, 1]` is the opacity of the i-th Gaussian (see :attr:`opacities`).
    - :math:`SH(v; C_i)` is the spherical harmonics function evaluated at direction :math:`v` with coefficients :math:`C_i`.
    - :math:`R(q_i)` is the rotation matrix corresponding to the quaternion :math:`q_i`.

    To render images from a :class:`GaussianSplat3d`, you volume render the radiance field using

    .. math::

        I(u, v) = \\int_{t \\in r(u, v)} T(t) R(r(t), d) dt

    where :math:`r(u, v)` is the camera ray through pixel :math:`(u, v)`, :math:`d` is the viewing direction of the ray,
    and :math:`T(t) = \\exp\\left(-\\int_{0}^{t} R(r(s), s) ds\\right)` is the accumulated transmittance along the ray up to distance :math:`t`.

    and to render depths you compute

    .. math::

        D(u, v) = \\int_{t \\in r(u, v)} t \\cdot T(t) \\sum_{i=1}^{N} o_i \\cdot \\alpha_i(r(t), d) dt

    """

    PLY_VERSION_STRING = "fvdb_ply 1.0.0"
    """
    Version string written to PLY files saved using the :meth:`save_to_ply` method.
    This string will be written in the comment section of the PLY file to identify
    the version of the fvdb library used to save the file. The comment will have the form
    ``comment fvdb_gs_ply <PLY_VERSION_STRING>``.
    """

    __PRIVATE__ = object()

    _TENSOR_FIELDS = ("_means", "_quats", "_log_scales", "_logit_opacities", "_sh0", "_shN")
    _ACCUM_TENSOR_FIELDS = (
        "_accumulated_mean_2d_gradient_norms",
        "_accumulated_gradient_step_counts",
        "_accumulated_max_2d_radii",
    )

    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor,
        accumulate_mean_2d_gradients: bool = False,
        accumulate_max_2d_radii: bool = False,
        accumulated_mean_2d_gradient_norms: torch.Tensor | None = None,
        accumulated_gradient_step_counts: torch.Tensor | None = None,
        accumulated_max_2d_radii_tensor: torch.Tensor | None = None,
        _private: Any = None,
    ) -> None:
        """
        Initializes the :class:`GaussianSplat3d` with tensors directly.
        This constructor is private and should not be used directly.

        .. note::

            You should never call this constructor directly. Instead, use the
            :meth:`from_tensors` or :meth:`from_ply` class methods to create new instances of
            :class:`GaussianSplat3d`.
        """
        if _private is not self.__PRIVATE__:
            raise ValueError("GaussianSplat3d constructor is private. Use from_tensors or from_ply instead.")
        self._means = means
        self._quats = quats
        self._log_scales = log_scales
        self._logit_opacities = logit_opacities
        self._sh0 = sh0
        self._shN = shN
        self._accumulate_mean_2d_gradients = accumulate_mean_2d_gradients
        self._accumulate_max_2d_radii = accumulate_max_2d_radii
        self._accumulated_mean_2d_gradient_norms = accumulated_mean_2d_gradient_norms
        self._accumulated_gradient_step_counts = accumulated_gradient_step_counts
        self._accumulated_max_2d_radii = accumulated_max_2d_radii_tensor

    @classmethod
    def from_tensors(
        cls,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor,
        accumulate_mean_2d_gradients: bool = False,
        accumulate_max_2d_radii: bool = False,
        detach: bool = False,
    ) -> "GaussianSplat3d":
        """
        Create a new :class:`GaussianSplat3d` from the provided tensors. This constructs a new
        Gaussian splat radiance field with the specified means, orientations, scales, opacities, and spherical harmonics coefficients.

        .. note::

            The :class:`GaussianSplat3d` stores the log of scales scales (:attr:`log_scales`) rather than the scales
            directly. This ensures numerical stability, especially when optimizing the scales, since each gaussian
            is defined as :math:`\\exp(R(q)^T S R(q))` where :math:`R(q)` is rotation matrix defined by the unit quaternion of the Gaussian,
            and :math:`S = diag(exp(log_scales))`.


        .. note::

            The :class:`GaussianSplat3d` stores the logit of opacities (:attr:`logit_opacities`) rather than the opacities
            directly. The actual opacities are obtained by applying the sigmoid function to the logit opacities.
            This ensures opacities are always in the range ``[0, 1]`` and improves numerical stability during optimization.

        Args:
            means (torch.Tensor): Tensor of shape ``(N, 3)`` representing the means of the gaussians, where ``N`` is the number of gaussians.
            quats (torch.Tensor): Tensor of shape ``(N, 4)`` representing the quaternions (orientations) of the gaussians, where ``N`` is the number of gaussians.
            log_scales (torch.Tensor): Tensor of shape ``(N, 3)`` representing the log scales of the gaussians, where ``N`` is the number of gaussians.
            logit_opacities (torch.Tensor): Tensor of shape ``(N,)`` representing the logit opacities of the gaussians, where ``N`` is the number of gaussians.
            sh0 (torch.Tensor): Tensor of shape ``(N, 1, D)`` representing the diffuse SH coefficients
                where ``D`` is the number of channels (see :attr:`num_channels`).
            shN (torch.Tensor): Tensor of shape ``(N, K-1, D)`` representing the directionally
                varying SH coefficients where ``D`` is the number of channels (see :attr:`num_channels`),
                and ``K`` is the number of spherical harmonic bases (see :attr:`num_sh_bases`).
            accumulate_mean_2d_gradients (bool, optional): If ``True``, tracks the average norm of the
                gradient of projected means for each Gaussian during the backward pass of projection.
                This is useful for some optimization techniques, such as the one in the `original paper <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>`_.
                Defaults to ``False``.
            accumulate_max_2d_radii (bool, optional): If ``True``, tracks the maximum 2D radii for each Gaussian
                during the backward pass of projection. This is useful for some optimization techniques, such as the one in the `original paper <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>`_.
                Defaults to ``False``.
            detach (bool, optional): If ``True``, creates copies of the input tensors and detaches them
                from the computation graph. Defaults to ``False``.
        """

        if detach:
            means = means.detach().clone()
            quats = quats.detach().clone()
            log_scales = log_scales.detach().clone()
            logit_opacities = logit_opacities.detach().clone()
            sh0 = sh0.detach().clone()
            shN = shN.detach().clone()
        return GaussianSplat3d(
            means=means,
            quats=quats,
            log_scales=log_scales,
            logit_opacities=logit_opacities,
            sh0=sh0,
            shN=shN,
            accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
            accumulate_max_2d_radii=accumulate_max_2d_radii,
            _private=cls.__PRIVATE__,
        )

    @classmethod
    def from_ply(
        cls, filename: pathlib.Path | str, device: DeviceIdentifier = "cuda"
    ) -> "tuple[GaussianSplat3d, dict[str, str | int | float | torch.Tensor]]":
        """
        Create a `GaussianSplat3d` instance from a PLY file.

        Args:
            filename (str): The name of the file to load the PLY data from.
            device (torch.device): The device to load the data onto. Default is "cuda".

        Returns:
            splats (GaussianSplat3d): An instance of GaussianSplat3d initialized with the data from the PLY file.
            metadata (dict[str, str | int | float | torch.Tensor]): A dictionary of metadata where the keys are strings and the
                values are either strings, ints, floats, or tensors. Can be empty if no metadata is saved in the PLY file.
        """
        device = resolve_device(device)
        if isinstance(filename, pathlib.Path):
            filename = str(filename)
        means, quats, log_scales, logit_opacities, sh0, shN, metadata = fvdb_functional.load_gaussian_ply(
            filename, device
        )
        return (
            cls(
                means=means,
                quats=quats,
                log_scales=log_scales,
                logit_opacities=logit_opacities,
                sh0=sh0,
                shN=shN,
                _private=cls.__PRIVATE__,
            ),
            metadata,
        )

    @overload
    def __getitem__(self, index: slice) -> "GaussianSplat3d": ...

    @overload
    def __getitem__(self, index: torch.Tensor) -> "GaussianSplat3d": ...

    def __getitem__(self, index: slice | torch.Tensor) -> "GaussianSplat3d":
        """
        Select Gaussians using either an integer index tensor, a boolean mask tensor, or a slice.

        .. note::

            If :attr:`accumulate_mean_2d_gradients` or :attr:`accumulate_max_2d_radii` is enabled on this
            :class:`GaussianSplat3d` instance, the returned :class:`GaussianSplat3d` will also contain
            the corresponding accumulated values.

        Example usage:

        .. code-block:: python

            # Using a slice
            gs_subset = gsplat3d[10:20] # Selects Gaussians from index 10 to 19

            # Using an integer index tensor
            indices = torch.tensor([0, 2, 4, 6])
            gs_subset = gsplat3d[indices] # Selects Gaussians at indices 0, 2, 4, and 6

            # Using a boolean mask tensor

            mask = torch.tensor([True, False, True, False, ...]) # Length must be num_gaussians
            gs_subset = gsplat3d[mask] # Selects Gaussians where mask is True

        Args:
            index (slice | torch.Tensor): A slice object or a 1D tensor containing either integer indices or a boolean mask.

        Returns:
            gaussian_splat_3d (GaussianSplat3d): A new instance of :class:`GaussianSplat3d` containing only the selected Gaussians.

        """
        if isinstance(index, slice):
            idx = index
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1:
                raise ValueError("Expected 'index' to be a 1D tensor.")
            if index.dtype == torch.bool:
                if len(index) != self.num_gaussians:
                    raise ValueError(
                        f"Expected 'index_or_mask' to have the same length as the number of Gaussians ({self.num_gaussians}), "
                        f"but got {len(index)}."
                    )
            elif index.dtype not in (torch.int64, torch.int32):
                raise ValueError("Expected 'index' to be a boolean or integer (int32 or int64) tensor.")
            idx = index
        else:
            raise TypeError("Expected 'index' to be a slice or a torch.Tensor.")
        return self._select(idx)

    @overload
    def __setitem__(self, index: slice, value: "GaussianSplat3d") -> None: ...

    @overload
    def __setitem__(self, index: torch.Tensor, value: "GaussianSplat3d") -> None: ...

    def __setitem__(self, index: torch.Tensor | slice, value: "GaussianSplat3d") -> None:
        """
        Set the values of Gaussians in this :class:`GaussianSplat3d` instance using either an integer index tensor,
        a boolean mask tensor, or a slice.

        .. note::

            If using integer indices with duplicate indices, the Gaussian set from ``value`` at the duplicate indices will
            overwrite in a random order.

        .. note::

            If :attr:`accumulate_mean_2d_gradients` or :attr:`accumulate_max_2d_radii` is enabled on this
            :class:`GaussianSplat3d` instance, the corresponding accumulated values will also be updated
            for the selected Gaussians, based on the values from the ``value`` instance. If ``value`` does not have
            these accumulations enabled, the accumulated values for the selected Gaussians will be reset to zero.

        Example:

        .. code-block:: python

            # Using a slice
            gs_subset: GaussianSplat3d = ...  # Some GaussianSplat3d instance with 10 Gaussians
            gsplat3d[10:20] = gs_subset  # Sets Gaussians from index 10 to 19

            # Using an integer index tensor
            indices = torch.tensor([0, 2, 4, 6])
            gs_subset: GaussianSplat3d = ...  # Some GaussianSplat3d instance with 4 Gaussians
            gsplat3d[indices] = gs_subset  # Sets Gaussians at indices 0, 2, 4, and 6

            # Using a boolean mask tensor
            mask = torch.tensor([True, False, True, False, ...])  # Length must be num_gaussians
            gs_subset: GaussianSplat3d = ...  # Some GaussianSplat3d instance with num unmasked Gaussians
            gsplat3d[mask] = gs_subset  # Sets Gaussians where mask is True

        Args:
            index (torch.Tensor | slice): A slice object or a 1D tensor containing either integer indices or a boolean mask.
            value (GaussianSplat3d): The :class:`GaussianSplat3d` instance containing the new values to set.
                Must have the same number of Gaussians as the selected indices or mask.
        """
        if isinstance(index, slice):
            idx = index
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1:
                raise ValueError("Expected 'index' to be a 1D tensor.")
            if index.dtype == torch.bool:
                if len(index) != self.num_gaussians:
                    raise ValueError(
                        f"Expected 'index' to have the same length as the number of Gaussians ({self.num_gaussians}), "
                        f"but got {len(index)}."
                    )
            elif index.dtype not in (torch.int64, torch.int32):
                raise ValueError("Expected 'index' to be a boolean or integer (int32 or int64) tensor.")
            idx = index
        else:
            raise TypeError("Expected 'index' to be a slice or a torch.Tensor")
        for field in self._TENSOR_FIELDS:
            getattr(self, field).data[idx] = getattr(value, field)
        for field in self._ACCUM_TENSOR_FIELDS:
            src_val = getattr(value, field)
            dst_val = getattr(self, field)
            if dst_val is not None:
                if src_val is not None:
                    dst_val.data[idx] = src_val
                else:
                    dst_val.data[idx] = 0

    def _select(self, idx: slice | torch.Tensor) -> "GaussianSplat3d":
        return GaussianSplat3d(
            means=self._means[idx],
            quats=self._quats[idx],
            log_scales=self._log_scales[idx],
            logit_opacities=self._logit_opacities[idx],
            sh0=self._sh0[idx],
            shN=self._shN[idx],
            accumulate_mean_2d_gradients=self._accumulate_mean_2d_gradients,
            accumulate_max_2d_radii=self._accumulate_max_2d_radii,
            accumulated_mean_2d_gradient_norms=(
                self._accumulated_mean_2d_gradient_norms[idx]
                if self._accumulated_mean_2d_gradient_norms is not None
                else None
            ),
            accumulated_gradient_step_counts=(
                self._accumulated_gradient_step_counts[idx]
                if self._accumulated_gradient_step_counts is not None
                else None
            ),
            accumulated_max_2d_radii_tensor=(
                self._accumulated_max_2d_radii[idx] if self._accumulated_max_2d_radii is not None else None
            ),
            _private=self.__PRIVATE__,
        )

    def detach(self) -> "GaussianSplat3d":
        """
        Return a new :class:`GaussianSplat3d` instance whose tensors are detached from the computation graph.
        This is useful when you want to stop tracking gradients for this instance.

        Returns:
            gaussian_splat (GaussianSplat3d): A new :class:`GaussianSplat3d` instance whose
                tensors are detached.
        """
        return GaussianSplat3d(
            means=self._means.detach().clone(),
            quats=self._quats.detach().clone(),
            log_scales=self._log_scales.detach().clone(),
            logit_opacities=self._logit_opacities.detach().clone(),
            sh0=self._sh0.detach().clone(),
            shN=self._shN.detach().clone(),
            accumulate_mean_2d_gradients=self._accumulate_mean_2d_gradients,
            accumulate_max_2d_radii=self._accumulate_max_2d_radii,
            accumulated_mean_2d_gradient_norms=(
                self._accumulated_mean_2d_gradient_norms.detach().clone()
                if self._accumulated_mean_2d_gradient_norms is not None
                else None
            ),
            accumulated_gradient_step_counts=(
                self._accumulated_gradient_step_counts.detach().clone()
                if self._accumulated_gradient_step_counts is not None
                else None
            ),
            accumulated_max_2d_radii_tensor=(
                self._accumulated_max_2d_radii.detach().clone() if self._accumulated_max_2d_radii is not None else None
            ),
            _private=self.__PRIVATE__,
        )

    def detach_(self) -> None:
        """
        Detaches this :class:`GaussianSplat3d` instance from the computation graph in place.
        This modifies the current instance to stop tracking gradients.

        .. note::

            This method modifies the current instance and does not return a new instance.

        """
        for field in self._TENSOR_FIELDS:
            setattr(self, field, getattr(self, field).detach_())

    @staticmethod
    def cat(
        splats: "Sequence[GaussianSplat3d]",
        accumulate_mean_2d_gradients: bool = False,
        accumulate_max_2d_radii: bool = False,
        detach: bool = False,
    ) -> "GaussianSplat3d":
        """
        Concatenates a sequence of :class:`GaussianSplat3d` instances into a single :class:`GaussianSplat3d` instance.

        The returned :class:`GaussianSplat3d` will contain all the Gaussians from the input instances,
        in the order they were provided.

        .. note::

            All input :class:`GaussianSplat3d` instances must have the same number of channels
            and spherical harmonic degree.

        .. note::

            If ``accumulate_mean_2d_gradients`` is ``True``, the concatenated instance will track the average norm
            of projected mean gradients for each Gaussian during the backward pass of projection. This value
            is copied over from each input instance if they were tracking it, and initialized to zero otherwise.

        .. note::

            If ``accumulate_max_2d_radii`` is ``True``, the concatenated instance will track the maximum 2D radii
            for each Gaussian during the backward pass of projection. This value is copied over from each input
            instance if they were tracking it, and initialized to zero otherwise.


        Args:
            splats (Sequence[GaussianSplat3d]): A sequence of :class:`GaussianSplat3d` instances to concatenate.
            accumulate_mean_2d_gradients (bool): If True, copies over the accumulated mean 2D gradients
                for each :class:`GaussianSplat3d` into the new one, or initializes it to zero if the input
                instance was not tracking it.
                Defaults to ``False``.
            accumulate_max_2d_radii (bool): If ``True``, copies the accumulated maximum 2D radii
                for each :class:`GaussianSplat3d` into the concatenated one, or initializes it to zero if the input
                instance was not tracking it.
                Defaults to ``False``.
            detach (bool): If ``True``, detaches the concatenated :class:`GaussianSplat3d` from the computation graph.
                Defaults to ``False``.

        Returns:
            GaussianSplat3d: A new instance of GaussianSplat3d containing the concatenated Gaussians.
        """

        def _cat_field(field: str) -> torch.Tensor:
            t = torch.cat([getattr(s, field) for s in splats], dim=0)
            return t.detach().clone() if detach else t

        accum_grad_norms: torch.Tensor | None = None
        accum_step_counts: torch.Tensor | None = None
        accum_max_radii: torch.Tensor | None = None

        if accumulate_mean_2d_gradients:
            parts_gn: list[torch.Tensor] = []
            parts_sc: list[torch.Tensor] = []
            for s in splats:
                n = s.num_gaussians
                gn = s._accumulated_mean_2d_gradient_norms
                sc = s._accumulated_gradient_step_counts
                parts_gn.append(gn if gn is not None else torch.zeros(n, device=s.device, dtype=s.dtype))
                parts_sc.append(sc if sc is not None else torch.zeros(n, device=s.device, dtype=torch.int32))
            accum_grad_norms = torch.cat(parts_gn, dim=0)
            accum_step_counts = torch.cat(parts_sc, dim=0)

        if accumulate_max_2d_radii:
            parts_mr: list[torch.Tensor] = []
            for s in splats:
                n = s.num_gaussians
                mr = s._accumulated_max_2d_radii
                parts_mr.append(mr if mr is not None else torch.zeros(n, device=s.device, dtype=torch.int32))
            accum_max_radii = torch.cat(parts_mr, dim=0)

        return GaussianSplat3d(
            means=_cat_field("_means"),
            quats=_cat_field("_quats"),
            log_scales=_cat_field("_log_scales"),
            logit_opacities=_cat_field("_logit_opacities"),
            sh0=_cat_field("_sh0"),
            shN=_cat_field("_shN"),
            accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
            accumulate_max_2d_radii=accumulate_max_2d_radii,
            accumulated_mean_2d_gradient_norms=accum_grad_norms,
            accumulated_gradient_step_counts=accum_step_counts,
            accumulated_max_2d_radii_tensor=accum_max_radii,
            _private=GaussianSplat3d.__PRIVATE__,
        )

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "GaussianSplat3d":
        """
        Creates a :class:`GaussianSplat3d` instance from a state dictionary generated by :meth:`state_dict`.
        This method is typically used to load a saved state of the :class:`GaussianSplat3d` instance.

        A state dictionary must contains the following keys which are all the required parameters to initialize a :class:`GaussianSplat3d`.
        Here ``N`` denotes the number of Gaussians (see :attr:`num_gaussians`)

        - ``'means'``: Tensor of shape ``(N, 3)`` representing the means of the Gaussians.
        - ``'quats'``: Tensor of shape ``(N, 4)`` representing the quaternions of the Gaussians.
        - ``'log_scales'``: Tensor of shape ``(N, 3)`` representing the log scales of the Gaussians.
        - ``'logit_opacities'``: Tensor of shape ``(N,)`` representing the logit opacities of the Gaussians.
        - ``'sh0'``: Tensor of shape ``(N, 1, D)`` representing the diffuse SH coefficients
          where ``D`` is the number of channels (see :attr:`num_channels`).
        - ``'shN'``: Tensor of shape ``(N, K-1, D)`` representing the directionally varying SH
          coefficients where ``D`` is the number of channels (see :attr:`num_channels`), and ``K``
          is the number of spherical harmonic bases (see :attr:`num_sh_bases`).
        - ``'accumulate_max_2d_radii'``: bool Tensor with a single element indicating
          whether to track the maximum 2D radii for gradients.
        - ``'accumulate_mean_2d_gradients'``: bool Tensor with a single element indicating whether
          to track the average norm of the gradient of projected means for each Gaussian.

        It can also optionally contain the following keys:

        - ``'accumulated_gradient_step_counts'``: Tensor of shape ``(N,)`` representing the
          accumulated gradient step counts for each Gaussian.
        - ``'accumulated_max_2d_radii'``: Tensor of shape ``(N,)`` representing the maximum
          2D projected radius for each Gaussian across every iteration of optimization.
        - ``'accumulated_mean_2d_gradient_norms'``: Tensor of shape ``(N,)`` representing the
          average norm of the gradient of projected means for each Gaussian across every iteration of optimization.

        Args:
            state_dict (dict[str, torch.Tensor]): A dictionary containing the state of the :class:`GaussianSplat3d` instance, usually generated via the :meth:`state_dict` method.

        Returns:
            gaussian_splat (GaussianSplat3d): An instance of :class:`GaussianSplat3d` initialized with the provided state dictionary.
        """
        accum_mean_2d_grads = bool(state_dict.get("accumulate_mean_2d_gradients", torch.tensor(False)).item())
        accum_max_2d = bool(state_dict.get("accumulate_max_2d_radii", torch.tensor(False)).item())
        return cls(
            means=state_dict["means"],
            quats=state_dict["quats"],
            log_scales=state_dict["log_scales"],
            logit_opacities=state_dict["logit_opacities"],
            sh0=state_dict["sh0"],
            shN=state_dict["shN"],
            accumulate_mean_2d_gradients=accum_mean_2d_grads,
            accumulate_max_2d_radii=accum_max_2d,
            accumulated_mean_2d_gradient_norms=state_dict.get("accumulated_mean_2d_gradient_norms"),
            accumulated_gradient_step_counts=state_dict.get("accumulated_gradient_step_counts"),
            accumulated_max_2d_radii_tensor=state_dict.get("accumulated_max_2d_radii"),
            _private=cls.__PRIVATE__,
        )

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the Tensors managed by this :class:`GaussianSplat3d` instance is stored.

        Returns:
            device (torch.device): The device of this :class:`GaussianSplat3d` instance.
        """
        return self._means.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of of the tensors managed by this :class:`GaussianSplat3d` instance
        (e.g., ``torch.float32``, ``torch.float64``).

        Returns:
            torch.dtype: The data type of the tensors managed by this :class:`GaussianSplat3d` instance.
        """
        return self._means.dtype

    @property
    def sh_degree(self) -> int:
        """
        Returns the degree of the spherical harmonics used in the Gaussian splatting representation.
        This value is 0 for diffuse SH coefficients and >= 1 for directionally varying SH coefficients.

        .. note::

            This is **not** the same as the number of spherical harmonics bases (see :attr:`num_sh_bases`).
            The relationship between the degree and the number of bases is given by
            :math:`K = (sh\\_degree + 1)^2`, where :math:`K` is the number of spherical harmonics bases.

        Returns:
            sh_degree (int): The degree of the spherical harmonics.
        """
        return sh_degree_from_coefficients(self._shN)

    @property
    def num_channels(self) -> int:
        """
        Returns the number of channels in the Gaussian splatting representation.
        For example, if you are rendering RGB images, this method will return 3.

        Returns:
            num_channels (int): The number of channels.
        """
        return self._sh0.size(-1)

    @property
    def num_gaussians(self) -> int:
        """
        Returns the number of Gaussians in the Gaussian splatting representation.
        This is the total number of individual gaussian splats that are being used to represent the scene.

        Returns:
            num_gaussians (int): The number of Gaussians.
        """
        return self._means.size(0)

    @property
    def num_sh_bases(self) -> int:
        """
        Returns the number of spherical harmonics (SH) bases used in the Gaussian splatting representation.

        .. note::

            The number of SH bases is related to the SH degree (see :attr:`sh_degree`) by the formula
            :math:`K = (sh\\_degree + 1)^2`, where :math:`K` is the number of spherical harmonics bases.

        Returns:
            num_sh_bases (int): The number of spherical harmonics bases.
        """
        return (self.sh_degree + 1) ** 2

    @property
    def log_scales(self) -> torch.Tensor:
        """
        Returns the log of the scales for each Gaussian. Gaussians are represented in 3D space,
        as ellipsoids defined by their means, orientations (quaternions), and scales. *i.e.*

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        .. note::

            The :class:`GaussianSplat3d` stores the log of scales scales (:attr:`log_scales`) rather than the scales
            directly. This ensures numerical stability, especially when optimizing the scales.
            To read the scales directly, see the :attr:`scales` property (which is read-only).

        Returns:
            log_scales (torch.Tensor): A tensor of shape ``(N, 3)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the log of the scale of a Gaussian in 3D space.
        """
        return self._log_scales

    @log_scales.setter
    def log_scales(self, value: torch.Tensor) -> None:
        """
        Sets the log of the scales for each Gaussian. Gaussians are represented in 3D space,
        as ellipsoids defined by their means, orientations (quaternions), and scales. *i.e.*

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        .. note::

            The :class:`GaussianSplat3d` stores the log of scales scales (:attr:`log_scales`) rather than the scales
            directly. This ensures numerical stability, especially when optimizing the scales.
            To read the scales directly, see the :attr:`scales` property (which is read-only).

        Args:
            value (torch.Tensor): A tensor of shape ``(N, 3)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the log of the
                scale of a Gaussian in 3D space.

        """
        self._log_scales = cast_check(value, torch.Tensor, "log_scales")

    @property
    def logit_opacities(self) -> torch.Tensor:
        """
        Return the logit (inverse of sigmoid) of the opacities of each Gaussian in the scene.

        .. note::

            The :class:`GaussianSplat3d` stores the logit of opacities (:attr:`logit_opacities`) rather than the opacities
            directly. The actual opacities are obtained by applying the sigmoid function to the logit opacities.
            To read the opacities directly, see the :attr:`opacities` property (which is read-only).

        Returns:
            logit_opacities (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the logit of the opacity of a Gaussian in 3D space.
        """
        return self._logit_opacities

    @logit_opacities.setter
    def logit_opacities(self, value: torch.Tensor) -> None:
        """
        Set the logit (inverse of sigmoid) of the opacities of each Gaussian in the scene.

        .. note::

            The :class:`GaussianSplat3d` stores the logit of opacities (:attr:`logit_opacities`) rather than the opacities
            directly. The actual opacities are obtained by applying the sigmoid function to the logit opacities.
            To read the opacities directly, see the :attr:`opacities` property (which is read-only).

        Args:
            value (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the logit of the opacity of a Gaussian in 3D space.
        """
        self._logit_opacities = cast_check(value, torch.Tensor, "logit_opacities")

    @property
    def means(self) -> torch.Tensor:
        """
        Return the means (3d positions) of the Gaussians in this :class:`GaussianSplat3d`.
        The means represent the center of each Gaussian in 3D space.
        *i.e* each Gaussian :math:`g_i` is defined as:

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.


        Returns:
            torch.Tensor: A tensor of shape (N, 3) where N is the number
                of Gaussians (see `num_gaussians`). Each row represents the mean of a Gaussian in 3D space.
        """
        return self._means

    @means.setter
    def means(self, value: torch.Tensor) -> None:
        """
        Sets the means (3d positions) of the Gaussians in this :class:`GaussianSplat3d`.
        The means represent the center of each Gaussian in 3D space.
        *i.e* each Gaussian :math:`g_i` is defined as:

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        Args:
            value (torch.Tensor): A tensor of shape ``(N, 3)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the mean of a Gaussian in 3D space.
        """
        self._means = cast_check(value, torch.Tensor, "means")

    @property
    def quats(self) -> torch.Tensor:
        """
        Returns the unit quaternions representing the orientation of the covariance of the Gaussians in this :class:`GaussianSplat3d`.
        *i.e* each Gaussian :math:`g_i` is defined as:

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        Returns:
            quats (torch.Tensor): A tensor of shape ``(N, 4)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the unit quaternion of a Gaussian in 3D space.
        """
        return self._quats

    @quats.setter
    def quats(self, value: torch.Tensor) -> None:
        """
        Sets the unit quaternions representing the orientation of the covariance of the Gaussians in this :class:`GaussianSplat3d`.
        *i.e* each Gaussian :math:`g_i` is defined as:

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        Args:
            value (torch.Tensor): A tensor of shape ``(N, 4)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the unit quaternion of a Gaussian in 3D space.
        """
        self._quats = cast_check(value, torch.Tensor, "quats")

    @property
    def requires_grad(self) -> bool:
        """
        Returns whether the tensors tracked by this :class:`GaussianSplat3d` instance are set to require gradients.
        This is typically set to True if you want to optimize the parameters of the Gaussians.

        Example:

        .. code-block:: python

            gsplat3d = GaussianSplat3d(...)  # Some GaussianSplat3d instance
            gsplat3d.requires_grad = True  # Enable gradient tracking for optimization

            assert gsplat3d.means.requires_grad  # Now the means will require gradients
            assert gsplat3d.covariances.requires_grad  # Now the covariances will require gradients
            assert gsplat3d.logit_opacities.requires_grad  # Now the logit opacities will require gradients
            assert gsplat3d.log_scales.requires_grad  # Now the log scales will require gradients
            assert gsplat3d.sh0.requires_grad  # Now the SH coefficients will require gradients
            assert gsplat3d.shN.requires_grad  # Now the SH coefficients will require gradients

        Returns:
            requires_grad (bool): ``True`` if gradients are required, ``False`` otherwise.
        """
        return self._means.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """
        Sets whether the tensors tracked by this :class:`GaussianSplat3d` instance require gradients.
        This is typically set to True if you want to optimize the parameters of the Gaussians.

        Example:

        .. code-block:: python

            gsplat3d = GaussianSplat3d(...)  # Some GaussianSplat3d instance
            gsplat3d.requires_grad = True  # Enable gradient tracking for optimization

            assert gsplat3d.means.requires_grad  # Now the means will require gradients
            assert gsplat3d.covariances.requires_grad  # Now the covariances will require gradients
            assert gsplat3d.logit_opacities.requires_grad  # Now the logit opacities will require gradients
            assert gsplat3d.log_scales.requires_grad  # Now the log scales will require gradients
            assert gsplat3d.sh0.requires_grad  # Now the SH coefficients will require gradients
            assert gsplat3d.shN.requires_grad  # Now the SH coefficients will require gradients

        Returns:
            requires_grad (bool): ``True`` if gradients are required, ``False`` otherwise.
        """
        v = cast_check(value, bool, "requires_grad")
        for field in self._TENSOR_FIELDS:
            getattr(self, field).requires_grad_(v)

    @property
    def sh0(self) -> torch.Tensor:
        """
        Returns the diffuse spherical harmonics coefficients of the Gaussians in this :class:`GaussianSplat3d`.
        These coefficients are used to represent the diffuse color/feature of each Gaussian.

        Returns:
            sh0 (torch.Tensor): A tensor of shape ``(N, 1, D)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`), and ``D`` is the number of channels (see :attr:`num_channels`).
                Each row represents the diffuse SH coefficients for a Gaussian.
        """
        return self._sh0

    @sh0.setter
    def sh0(self, value: torch.Tensor) -> None:
        """
        Sets the diffuse spherical harmonics coefficients of the Gaussians in this :class:`GaussianSplat3d`.
        These coefficients are used to represent the diffuse color/feature of each Gaussian.

        Args:
            value (torch.Tensor): A tensor of shape ``(N, 1, D)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`), and ``D`` is the number of channels (see :attr:`num_channels`).
                Each row represents the diffuse SH coefficients for a Gaussian.
        """
        self._sh0 = cast_check(value, torch.Tensor, "sh0")

    @property
    def shN(self) -> torch.Tensor:
        """
        Returns the directionally varying spherical harmonics coefficients of the Gaussians in the scene.
        These coefficients are used to represent a direction dependent color/feature of each Gaussian.

        Returns:
            torch.Tensor: A tensor of shape (N, K-1, D) where N is the number
                of Gaussians (see `num_gaussians`), D is the number of channels (see `num_channels`),
                and K is the number of spherical harmonic bases (see `num_sh_bases`).
                Each row represents the directionally varying SH coefficients for a Gaussian.
        """
        return self._shN

    @shN.setter
    def shN(self, value: torch.Tensor) -> None:
        """
        Sets the directionally varying spherical harmonics coefficients of the Gaussians in this :class:`GaussianSplat3d`.
        These coefficients are used to represent a direction dependent color/feature of each Gaussian.

        Args:
            value (torch.Tensor): A tensor of shape ``(N, K-1, D)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`), ``D`` is the number of channels (see :attr:`num_channels`),
                and ``K`` is the number of spherical harmonic bases (see :attr:`num_sh_bases`).
                Each row represents the directionally varying SH coefficients for a Gaussian.
        """
        self._shN = cast_check(value, torch.Tensor, "shN")

    @property
    def opacities(self) -> torch.Tensor:
        """
        Returns the opacities of the Gaussians in the Gaussian splatting representation.
        The opacities encode the visibility of each Gaussian in the scene.

        .. note::

            This property is read only. :class:`GaussianSplat3d` stores the logit (inverse of sigmoid)
            of the opacities to ensure numerical stability, which you can modify. See :attr:`logit_opacities`.

        Returns:
            opacities (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
                Each element represents the opacity of a Gaussian.
        """
        return torch.sigmoid(self._logit_opacities)

    @property
    def scales(self) -> torch.Tensor:
        """
        Returns the scales of the Gaussians in the Gaussian splatting representation. The scales are
        the eigenvalues of the covariance matrix of each Gaussian.
        *i.e* each Gaussian :math:`g_i` is defined as:

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        .. note::

            This property is read only. :class:`GaussianSplat3d` stores the log of scales to ensure numerical stability,
            which you can modify. See :attr:`log_scales`.

        Returns:
            scales (torch.Tensor): A tensor of shape ``(N, 3)`` where ``N`` is the number
                of Gaussians. Each row represents the scale of a Gaussian in 3D space.
        """
        return torch.exp(self._log_scales)

    @property
    def accumulated_gradient_step_counts(self) -> torch.Tensor:
        """
        Returns the accumulated gradient step counts for each Gaussian.

        If this :class:`GaussianSplat3d` instance
        is set to track accumulated gradients (*i.e*  :attr:`accumulate_mean_2d_gradients` is ``True``),
        then this tensor contains the number of Gradient steps that have been applied to each Gaussian during optimization.

        If :attr:`accumulate_mean_2d_gradients` is ``False``, this property will be an empty tensor.

        .. note::

            To reset the counts, call call the :meth:`reset_accumulated_gradient_state` method.

        Returns:
            step_counts (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
                Each element represents the accumulated gradient step count for a Gaussian.
        """
        return self._accumulated_gradient_step_counts

    @property
    def accumulated_max_2d_radii(self) -> torch.Tensor:
        """
        Returns the maximum 2D projected radius (in pixels) for each Gaussian across all calls to `render_*` functions.
        This is used by certain optimization techniques to ensure that the Gaussians do not become too large or too small during the optimization process.

        If :this :class:`GaussianSplat3d` instance is set to track maximum 2D radii
        (*i.e* :attr:`accumulate_max_2d_radii` is ``True``), then this tensor contains the maximum 2D radius for each Gaussian.

        If :attr:`accumulate_max_2d_radii` is ``False``, this property will be an empty tensor.

        .. note::

            To reset the maximum radii to zero, you can call the :meth:`reset_accumulated_gradient_state` method.

        Returns:
            max_radii (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
                Each element represents the maximum 2D radius for a Gaussian across all optimization iterations.

        """
        return self._accumulated_max_2d_radii

    @property
    def accumulate_max_2d_radii(self) -> bool:
        """
        Returns whether to track the maximum 2D projected radius of each Gaussian across calls to `render_*` functions.
        This is used by certain optimization techniques to ensure that the Gaussians do not become too large or too small during the optimization process.


        .. seealso::

            See :attr:`accumulated_max_2d_radii` for the actual maximum radii values.

        Returns:
            accumulate_max_radii (bool): ``True`` if the maximum 2D radii are being tracked across rendering calls, ``False`` otherwise.
        """
        return self._accumulate_max_2d_radii

    @accumulate_max_2d_radii.setter
    def accumulate_max_2d_radii(self, value) -> None:
        """
        Sets whether to track the maximum 2D projected radius of each Gaussian across calls to `render_*` functions.
        This is used by certain optimization techniques to ensure that the Gaussians do not become too large or too small during the optimization process.

        .. seealso::

            See :attr:`accumulated_max_2d_radii` for the actual maximum radii values.

        Args:
            value (bool): ``True`` if the maximum 2D radii are being tracked across rendering calls, ``False`` otherwise.
        """
        self._accumulate_max_2d_radii = cast_check(value, bool, "accumulate_max_2d_radii")

    @property
    def accumulate_mean_2d_gradients(self) -> bool:
        """
        Returns whether to track the average norm of the gradient of projected means for each Gaussian during the backward pass of projection.
        This property is used by certain optimization techniques to split/prune/duplicate Gaussians.
        The accumulated 2d gradient norms are defined as follows:

        .. math::

            \\sum_{t=1}^{T} \\| \\partial_{L_t} \\mu_i^{2D} \\|_2

        where :math:`\\mu_i^{2D}` is the projection of the mean of Gaussian :math:`g_i` onto the image plane,
        and :math:`L_t` is the loss at iteration :math:`t`.

        .. seealso::

            See :attr:`accumulated_mean_2d_gradient_norms` for the actual average norms of the gradients.

        Returns:
            accumulate_mean_2d_grads (bool): ``True`` if the average norm of the gradient of projected means is being tracked, ``False`` otherwise.
        """
        return self._accumulate_mean_2d_gradients

    @accumulate_mean_2d_gradients.setter
    def accumulate_mean_2d_gradients(self, value: bool) -> None:
        """
        Sets whether to track the average norm of the gradient of projected means for each Gaussian during the backward pass of projection.
        This property is used by certain optimization techniques to split/prune/duplicate Gaussians.
        The accumulated 2d gradient norms are defined as follows:

        .. math::

            \\sum_{t=1}^{T} \\| \\partial_{L_t} \\mu_i^{2D} \\|_2

        where :math:`\\mu_i^{2D}` is the projection of the mean of Gaussian :math:`g_i` onto the image plane,
        and :math:`L_t` is the loss at iteration :math:`t`.

        .. seealso::

            See :attr:`accumulated_mean_2d_gradient_norms` for the actual average norms of the gradients.

        Args:
            value (bool): ``True`` if the average norm of the gradient of projected means is being tracked, ``False`` otherwise.
        """
        self._accumulate_mean_2d_gradients = cast_check(value, bool, "accumulate_mean_2d_gradients")

    @property
    def accumulated_mean_2d_gradient_norms(self) -> torch.Tensor:
        """
        Returns the average norm of the gradient of projected (2D) means for each Gaussian across every backward pass.
        This is used by certain optimization techniques to split/prune/duplicate Gaussians.
        The accumulated 2d gradient norms are defined as follows:

        .. math::

            \\sum_{t=1}^{T} \\| \\partial_{L_t} \\mu_i^{2D} \\|_2

        where :math:`\\mu_i^{2D}` is the projection of the mean of Gaussian :math:`g_i` onto the image plane,
        and :math:`L_t` is the loss at iteration :math:`t`.

        .. note::

            To reset the accumulated norms, call the :meth:`reset_accumulated_gradient_state` method.

        Returns:
            accumulated_grad_2d_norms (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
                Each element represents the average norm of the gradient of projected means for a Gaussian across all optimization iterations.
                The norm is computed in 2D space, i.e., the projected means.
        """
        return self._accumulated_mean_2d_gradient_norms

    # ---------------------------------------------------------------------------
    #  Private rendering helpers
    # ---------------------------------------------------------------------------

    def _projection_accumulators(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Create the enabled densification accumulators if missing or stale and return them."""
        num_gaussians = self.num_gaussians
        device = self._means.device
        grad_norms: torch.Tensor | None = None
        step_counts: torch.Tensor | None = None
        max_radii: torch.Tensor | None = None
        if self._accumulate_mean_2d_gradients:
            gn = self._accumulated_mean_2d_gradient_norms
            if gn is None or gn.numel() != num_gaussians:
                gn = torch.zeros(num_gaussians, device=device, dtype=self._means.dtype)
                self._accumulated_mean_2d_gradient_norms = gn
            sc = self._accumulated_gradient_step_counts
            if sc is None or sc.numel() != num_gaussians:
                sc = torch.zeros(num_gaussians, device=device, dtype=torch.int32)
                self._accumulated_gradient_step_counts = sc
            grad_norms, step_counts = gn, sc
        if self._accumulate_max_2d_radii:
            mr = self._accumulated_max_2d_radii
            if mr is None or mr.numel() != num_gaussians:
                mr = torch.zeros(num_gaussians, device=device, dtype=torch.int32)
                self._accumulated_max_2d_radii = mr
            max_radii = mr
        return grad_norms, step_counts, max_radii

    def _project(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel,
        projection_method: ProjectionMethod,
        distortion_coeffs: torch.Tensor | None,
        min_radius_2d: float,
        eps_2d: float,
        antialias: bool,
    ) -> ProjectedGaussians:
        """Stage 1 for this model's Gaussians, wiring in the enabled densification accumulators."""
        # Every projection wires in the enabled accumulators, as on main. World-space rendering reaches
        # the analytic backward only through antialiasing compensations, with a zero 2D-mean gradient, and
        # the kernel still counts that as a step; leaving the gradient accumulators out of world-space
        # projections is part of the training-backend follow-up.
        grad_norms, step_counts, max_radii = self._projection_accumulators()
        return project_gaussians(
            self._means,
            self._quats,
            self._log_scales,
            world_to_camera_matrices,
            projection_matrices,
            image_width,
            image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            accumulated_mean_2d_gradient_norms=grad_norms,
            accumulated_gradient_step_counts=step_counts,
            accumulated_max_2d_radii=max_radii,
        )

    def _project_and_opacities(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel,
        projection_method: ProjectionMethod,
        distortion_coeffs: torch.Tensor | None,
        min_radius_2d: float,
        eps_2d: float,
        antialias: bool,
    ) -> tuple[ProjectedGaussians, torch.Tensor]:
        """Stage 1 plus the per-camera opacities every later stage takes, computed once."""
        projected = self._project(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )
        return projected, compute_gaussian_opacities(self._logit_opacities, projected)

    def _features(
        self,
        projected: ProjectedGaussians,
        world_to_camera_matrices: torch.Tensor,
        sh_degree_to_use: int,
        render_mode: GaussianRenderMode,
    ) -> torch.Tensor:
        """Stage 2 for this model's spherical-harmonics coefficients."""
        return evaluate_gaussian_sh(
            self._means, self._sh0, self._shN, world_to_camera_matrices, projected, sh_degree_to_use, render_mode
        )

    def _project_for(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel,
        projection_method: ProjectionMethod,
        distortion_coeffs: torch.Tensor | None,
        min_radius_2d: float,
        eps_2d: float,
        antialias: bool,
        sh_degree_to_use: int,
        render_mode: GaussianRenderMode,
    ) -> ProjectedGaussianSplats:
        """Stages 1 and 2, bundled for later rendering with :meth:`render_from_projected_gaussians`."""
        projected, opacities = self._project_and_opacities(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )
        render_quantities = self._features(projected, world_to_camera_matrices, sh_degree_to_use, render_mode)
        return ProjectedGaussianSplats(
            projected=projected,
            render_quantities=render_quantities,
            logit_opacities=self._logit_opacities,
            antialias=antialias,
            eps_2d=eps_2d,
            near_plane=near,
            far_plane=far,
            min_radius_2d=min_radius_2d,
            sh_degree_to_use=sh_degree_to_use,
            opacities=opacities,
            _private=ProjectedGaussianSplats.__PRIVATE__,
        )

    def _render_dense(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel,
        projection_method: ProjectionMethod,
        distortion_coeffs: torch.Tensor | None,
        sh_degree_to_use: int,
        tile_size: int,
        min_radius_2d: float,
        eps_2d: float,
        antialias: bool,
        backgrounds: torch.Tensor | None,
        masks: torch.Tensor | None,
        render_mode: GaussianRenderMode,
        world_space: bool,
        crop: Crop | None = None,
        crop_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All four stages for dense images, in screen space or world space."""
        projected, opacities = self._project_and_opacities(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )
        features = self._features(projected, world_to_camera_matrices, sh_degree_to_use, render_mode)
        tiles = intersect_gaussian_tiles(projected, tile_size=tile_size, opacities=opacities)
        if world_space:
            return rasterize_world_space_gaussians(
                self._means,
                self._quats,
                self._log_scales,
                projected,
                features,
                opacities,
                world_to_camera_matrices,
                projection_matrices,
                tiles,
                distortion_coeffs=distortion_coeffs,
                backgrounds=backgrounds,
                masks=masks,
                crop=crop,
                crop_masks=crop_masks,
            )
        return rasterize_screen_space_gaussians(
            projected,
            features,
            opacities,
            tiles,
            backgrounds=backgrounds,
            masks=masks,
            crop=crop,
            crop_masks=crop_masks,
        )

    def _render_sparse(
        self,
        pixels_to_render: JaggedTensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel,
        projection_method: ProjectionMethod,
        distortion_coeffs: torch.Tensor | None,
        sh_degree_to_use: int,
        tile_size: int,
        min_radius_2d: float,
        eps_2d: float,
        antialias: bool,
        backgrounds: torch.Tensor | None,
        masks: torch.Tensor | None,
        render_mode: GaussianRenderMode,
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """All four stages for an arbitrary set of pixels, in the requested pixel order."""
        projected, opacities = self._project_and_opacities(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )
        features = self._features(projected, world_to_camera_matrices, sh_degree_to_use, render_mode)
        sparse_tiles = intersect_gaussian_tiles_sparse(
            pixels_to_render, projected, tile_size=tile_size, opacities=opacities
        )
        return rasterize_screen_space_gaussians_sparse(
            projected, features, opacities, sparse_tiles, backgrounds=backgrounds, tile_masks=masks
        )

    @staticmethod
    def _sparse_result(
        pixels_to_render: JaggedTensor | torch.Tensor, features: JaggedTensor, alphas: JaggedTensor
    ) -> tuple[Any, Any]:
        """Return sparse results as JaggedTensors, or as ``[C, P, D]`` tensors for tensor pixel input."""
        if isinstance(pixels_to_render, torch.Tensor):
            return torch.stack(features.unbind(), dim=0), torch.stack(alphas.unbind(), dim=0)
        return features, alphas

    def project_gaussians_for_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> ProjectedGaussianSplats:
        """
        Projects this :class:`GaussianSplat3d` onto one or more image planes for rendering depth images in those planes.
        You can render depth images from the projected Gaussians by calling :meth:`render_projected_gaussians`.

        .. note::

            The reason to have a separate projection and rendering step is to enable rendering crops of an image without
            having to project the Gaussians again.


        .. note::

            All images being rendered must have the same width and height.


        .. seealso::

            :class:`fvdb_reality_capture.ProjectedGaussianSplats` for the projected Gaussians representation.

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Project the Gaussians for rendering depth images onto C image planes
            projected_gaussians = gaussian_splat_3d.project_gaussians_for_depths(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the C images
                image_height, # height of the C images
                near, # near clipping plane
                far) # far clipping plane

            # Now render a crop of size 100x100 starting at (10, 10) from the projected Gaussians
            # in each image plane.
            # Returns a tensor of shape [C, 100, 100, 1] containing the depth images,
            # and a tensor of shape [C, 100, 100, 1] containing the final alpha (opacity) values
            # of each pixel.
            cropped_depth_images_1, cropped_alphas = gaussian_splat_3d.render_from_projected_gaussians(
                projected_gaussians,
                crop_width=100,
                crop_height=100,
                crop_origin_w=10,
                crop_origin_h=10)

            # To get the depth images, divide the last channel by the alpha values
            true_depths_1 = cropped_images_1[..., -1:] / cropped_alphas

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the world-to-camera transformation matrices for ``C`` cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note that all images must have the same width.
            image_height (int): The height of the images to be rendered. Note that all images must have the same height.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            projected_gaussians (ProjectedGaussianSplats): An instance of ProjectedGaussianSplats containing the projected Gaussians.
                This object contains the projected 2D representations of the Gaussians, which can be used for rendering depth images or further processing.

        """
        return self._project_for(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            sh_degree_to_use=-1,
            render_mode=GaussianRenderMode.DEPTH,
        )

    def project_gaussians_for_images(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        sh_degree_to_use: int = -1,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> ProjectedGaussianSplats:
        """
        Projects this :class:`GaussianSplat3d` onto one or more image planes for rendering multi-channel (see :attr:`num_channels`) images in those planes.
        You can render images from the projected Gaussians by calling :meth:`render_projected_gaussians`.

        .. note::

            The reason to have a separate projection and rendering step is to enable rendering crops of an image without
            having to project the Gaussians again.


        .. note::

            All images being rendered must have the same width and height.


        .. seealso::

            :class:`fvdb_reality_capture.ProjectedGaussianSplats` for the projected Gaussians representation.

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Project the Gaussians for rendering images onto C image planes
            projected_gaussians = gaussian_splat_3d.project_gaussians_for_images(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the C images
                image_height, # height of the C images
                near, # near clipping plane
                far) # far clipping plane

            # Now render a crop of size 100x100 starting at (10, 10) from the projected Gaussians
            # in each image plane.
            # Returns a tensor of shape [C, 100, 100, D] containing the images (where D is num_channels),
            # and a tensor of shape [C, 100, 100, 1] containing the final alpha (opacity) values
            # of each pixel.
            cropped_images_1, cropped_alphas = gaussian_splat_3d.render_from_projected_gaussians(
                projected_gaussians,
                crop_width=100,
                crop_height=100,
                crop_origin_w=10,
                crop_origin_h=10)

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the world-to-camera transformation matrices for ``C`` cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note that all images must have the same width.
            image_height (int): The height of the images to be rendered. Note that all images must have the same height.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            projected_gaussians (ProjectedGaussianSplats): An instance of ProjectedGaussianSplats containing the projected Gaussians.
                This object contains the projected 2D representations of the Gaussians, which can be used for rendering images or further processing.

        """
        return self._project_for(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            sh_degree_to_use=sh_degree_to_use,
            render_mode=GaussianRenderMode.FEATURES,
        )

    def project_gaussians_for_images_and_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        sh_degree_to_use: int = -1,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> ProjectedGaussianSplats:
        """
        Projects this :class:`GaussianSplat3d` onto one or more image planes for rendering multi-channel (see :attr:`num_channels`) images with depths
        in the last channel.
        You can render images+depths from the projected Gaussians by calling :meth:`render_projected_gaussians`.

        .. note::

            The reason to have a separate projection and rendering step is to enable rendering crops of an image without
            having to project the Gaussians again.


        .. note::

            All images being rendered must have the same width and height.


        .. seealso::

            :class:`fvdb_reality_capture.ProjectedGaussianSplats` for the projected Gaussians representation.

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Project the Gaussians for rendering images onto C image planes
            projected_gaussians = gaussian_splat_3d.project_gaussians_for_images_and_depths(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the C images
                image_height, # height of the C images
                near, # near clipping plane
                far) # far clipping plane

            # Now render a crop of size 100x100 starting at (10, 10) from the projected Gaussians
            # in each image plane.
            # Returns a tensor of shape [C, 100, 100, D] containing the images (where D is num_channels + 1 for depth),
            # and a tensor of shape [C, 100, 100, 1] containing the final alpha (opacity) values
            # of each pixel. Binning the Gaussians into tiles once and passing the result lets several
            # crops share it.
            tiles = projected_gaussians.tile_intersection()
            cropped_images_1, cropped_alphas = gaussian_splat_3d.render_from_projected_gaussians(
                projected_gaussians,
                crop_width=100,
                crop_height=100,
                crop_origin_w=10,
                crop_origin_h=10,
                tiles=tiles)

            cropped_images = cropped_images_1[..., :-1]  # Extract image channels

            # Divide by alpha to get the final true depth values
            cropped_depths = cropped_images_1[..., -1:] / cropped_alphas  # Extract depth channel

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the world-to-camera transformation matrices for ``C`` cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note that all images must have the same width.
            image_height (int): The height of the images to be rendered. Note that all images must have the same height.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            projected_gaussians (ProjectedGaussianSplats): An instance of ProjectedGaussianSplats containing the projected Gaussians.
                This object contains the projected 2D representations of the Gaussians, which can be used for rendering images or further processing.

        """
        return self._project_for(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            sh_degree_to_use=sh_degree_to_use,
            render_mode=GaussianRenderMode.FEATURES_AND_DEPTH,
        )

    def render_from_projected_gaussians(
        self,
        projected_gaussians: ProjectedGaussianSplats,
        crop_width: int = -1,
        crop_height: int = -1,
        crop_origin_w: int = -1,
        crop_origin_h: int = -1,
        tile_size: int | None = None,
        backgrounds: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        tiles: GaussianTileIntersection | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render a set of images from Gaussian splats that have already been projected onto image planes
        (See for example :meth:`project_gaussians_for_images`).
        This method is useful when you want to render images from pre-computed projected Gaussians,
        for example, when rendering crops of images without having to re-project the Gaussians.

        .. note::

            A negative value for any of ``crop_width``, ``crop_height``, ``crop_origin_w`` and ``crop_origin_h``
            means its default: the full image width or height, or an origin of zero. All four negative (the
            default) renders the full image.

        .. note::

            The output always has the requested crop size. Where the crop runs past the image boundary,
            the part inside the image is rendered and the rest is filled with the background at zero
            alpha; a crop that lies entirely outside the image is all background. The stage function
            returns the clipped part only (empty for a crop entirely outside); this method pads it. Either
            way the output is differentiable with respect to the projection whenever the projection is.


        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Project the Gaussians for rendering images onto C image planes
            projected_gaussians = gaussian_splat_3d.project_gaussians_for_images_and_depths(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the C images
                image_height, # height of the C images
                near, # near clipping plane
                far) # far clipping plane

            # Now render a crop of size 100x100 starting at (10, 10) from the projected Gaussians
            # in each image plane.
            # Returns a tensor of shape [C, 100, 100, D] containing the images (where D is num_channels + 1 for depth),
            # and a tensor of shape [C, 100, 100, 1] containing the final alpha (opacity) values
            # of each pixel. Binning the Gaussians into tiles once and passing the result lets several
            # crops share it.
            tiles = projected_gaussians.tile_intersection()
            cropped_images_1, cropped_alphas = gaussian_splat_3d.render_from_projected_gaussians(
                projected_gaussians,
                crop_width=100,
                crop_height=100,
                crop_origin_w=10,
                crop_origin_h=10,
                tiles=tiles)

            cropped_images = cropped_images_1[..., :-1]  # Extract image channels

            # Divide by alpha to get the final true depth values
            cropped_depths = cropped_images_1[..., -1:] / cropped_alphas  # Extract depth channel


        Args:
            projected_gaussians (ProjectedGaussianSplats): An instance of :class:`fvdb_reality_capture.ProjectedGaussianSplats`
                containing the projected Gaussians after spherical harmonic evaluation. This object should have been created by calling
                :meth:`project_gaussians_for_images`, :meth:`project_gaussians_for_depths`,
                :meth:`project_gaussians_for_images_and_depths`, etc.
            crop_width (int): The width of the crop to render. If -1, the full image width is used.
                Default is -1. A crop that runs past the image edge is filled with the background at zero
                alpha outside the image, so the output always has the requested size.
            crop_height (int): The height of the crop to render. If -1, the full image height is used.
                Default is -1.
            crop_origin_w (int): The x-coordinate of the top-left corner of the crop. If -1, the crop starts at (0, 0).
                Default is -1.
            crop_origin_h (int): The y-coordinate of the top-left corner of the crop. If -1, the crop starts at (0, 0).
                Default is -1.
            tile_size (int | None): The size of the tiles to use for rendering. Taken from ``tiles`` when
                they are given, and 16 otherwise; passing both raises if they disagree. You shouldn't set
                this parameter unless you really know what you are doing.
            backgrounds (torch.Tensor | None): Optional background colors of shape ``(C, D)``.
                If ``None``, background is treated as 0.
            masks (torch.Tensor | None): Optional per-pixel boolean mask in crop coordinates, of the requested
                crop size ``(C, cropH, cropW)`` or of its size after clipping to the image, on the projection's
                device. ``True`` means render, ``False`` means skip (filled with background). Without a crop it
                is a full-image mask.
            tiles (GaussianTileIntersection | None): The tile intersections of ``projected_gaussians`` at
                ``tile_size``, from :meth:`ProjectedGaussianSplats.tile_intersection`. Computed here when
                ``None``. Pass them when rendering several crops from one projection, so the Gaussians are
                binned into tiles once rather than once per crop.


        Returns:
            rendered_images (torch.Tensor): A tensor of shape ``(C, H, W, D)`` where ``C`` is the number of image planes,
                ``H`` is the height of the rendered images, ``W`` is the width of the rendered images, and ``D`` is the
                number of channels (e.g., RGB, RGBD, etc.).
            alpha_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of cameras,
                ``H`` is the height of the images, and ``W`` is the width of the images.
                Each element represents the alpha value (opacity) at a pixel such that 0 <= alpha < 1,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        pg = projected_gaussians
        projected = pg.projected_gaussians
        width, height = projected.image_width, projected.image_height
        crop_w = crop_width if crop_width >= 0 else width
        crop_h = crop_height if crop_height >= 0 else height
        origin_w = crop_origin_w if crop_origin_w >= 0 else 0
        origin_h = crop_origin_h if crop_origin_h >= 0 else 0
        is_crop = crop_w != width or crop_h != height or origin_w != 0 or origin_h != 0
        requested_h, requested_w = crop_h, crop_w
        if tiles is not None:
            if tile_size is not None and tiles.tile_size != tile_size:
                raise ValueError(f"tiles were computed at tile_size {tiles.tile_size}, not the requested {tile_size}")
            tile_size = tiles.tile_size
        elif tile_size is None:
            tile_size = 16
        # The stage function clips the crop at the image edge (to nothing, if it lies entirely outside),
        # checks the crop mask against the requested or clipped crop size, and returns the clipped part; it
        # is padded back below, so the output has the requested size.
        crop = (origin_w, origin_h, crop_w, crop_h) if is_crop else None
        images, alphas = rasterize_screen_space_gaussians(
            projected,
            pg.render_quantities,
            pg.opacities,
            tiles if tiles is not None else pg.tile_intersection(tile_size),
            backgrounds=backgrounds,
            masks=masks if crop is None else None,
            crop=crop,
            crop_masks=masks if crop is not None else None,
        )
        return pad_crop(images, alphas, requested_h, requested_w, backgrounds)

    def render_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        tile_size: int = 16,
        min_radius_2d: float = 0.3,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render ``C`` depth maps from this :class:`GaussianSplat3d` from ``C`` camera views.

        .. note::

            All depth maps being rendered must have the same width and height.


        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Render depth maps from C camera views
            # depth_images is a tensor of shape [C, H, W, 1]
            # alpha_images is a tensor of shape [C, H, W, 1]
            depth_images, alpha_images = gaussian_splat_3d.render_depths(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the depth maps
                image_height, # height of the depth maps
                near, # near clipping plane
                far) # far clipping plane

            true_depths = depth_images / alpha_images  # Get true depth values by dividing by alpha

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for ``C`` cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the depth maps to be rendered. Note these are the same for all depth maps being rendered.
            image_height (int): The height of the depth maps to be rendered. Note these are the same for all depth maps being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.
            backgrounds (torch.Tensor | None): Optional background colors of shape ``(C, 1)``.
                If ``None``, background is treated as 0.
            masks (torch.Tensor | None): Optional per-pixel boolean mask of shape ``(C, H, W)``.
                ``True`` means render, ``False`` means skip (filled with background).

        Returns:
            depth_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the depth maps, and ``W`` is the width of the depth maps.
                Each element represents the depth value at that pixel in the depth map.
            alpha_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, and ``W`` is the width of the images.
                Each element represents the alpha value (opacity) at a pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        return self._render_dense(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            sh_degree_to_use=-1,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
            masks=masks,
            render_mode=GaussianRenderMode.DEPTH,
            world_space=False,
        )

    def sparse_render_depths(
        self,
        pixels_to_render: JaggedTensorOrTensorT,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        tile_size: int = 16,
        min_radius_2d: float = 0.3,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[JaggedTensorOrTensorT, JaggedTensorOrTensorT]:
        """
        Render ``C`` collections of sparse depth values from this :class:`GaussianSplat3d` from ``C`` camera views
        at the specified pixel locations.

        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # pixels_to_render is a tensor of shape [C, P, 2] containing pixel coordinates to render
            # Render sparse depth values from C camera views at specified pixel locations
            # depth_values is a tensor of shape [C, P, 1]
            # alpha_values is a tensor of shape [C, P, 1]
            depth_values, alpha_values = gaussian_splat_3d.sparse_render_depths(
                pixels_to_render, # tensor of shape [C, P, 2]
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the images
                image_height, # height of the images
                near, # near clipping plane
                far) # far clipping plane

            true_depths = depth_values / alpha_values  # Get true depth values by dividing by alpha

        Args:
            pixels_to_render (torch.Tensor | JaggedTensor): A tensor of shape ``(C, P, 2)`` or a JaggedTensor where ``C`` is the number of camera views,
                and ``P`` is the number of pixel coordinates to render per camera. Each pixel coordinate is represented as (y, x) (row, col).
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.
            backgrounds (torch.Tensor | None): Optional background depths of shape ``(C, 1)``.
                If ``None``, background is treated as 0.
            masks (torch.Tensor | None): Optional per-tile boolean mask of shape
                ``(C, tileH, tileW)`` where ``tileH = ceil(image_height / tile_size)`` and
                ``tileW = ceil(image_width / tile_size)``. ``True`` means the tile is rendered,
                ``False`` means the tile is skipped and its pixels receive the background value
                with zero alpha.

        Returns:
            depth_values (torch.Tensor | JaggedTensor): A tensor of shape ``(C, P, 1)`` or a JaggedTensor where ``C`` is the number of camera views,
                and ``P`` is the number of pixel coordinates rendered per camera. Each element represents the depth value at that pixel.
            alpha_values (torch.Tensor | JaggedTensor): A tensor of shape ``(C, P, 1)`` or a JaggedTensor where ``C`` is the number of camera views,
                and ``P`` is the number of pixel coordinates rendered per camera. Each element represents the alpha value (opacity) at that pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        pixels_jt = as_pixel_jagged(pixels_to_render)
        features, alphas = self._render_sparse(
            pixels_to_render=pixels_jt,
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            sh_degree_to_use=-1,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
            masks=masks,
            render_mode=GaussianRenderMode.DEPTH,
        )
        return self._sparse_result(pixels_to_render, features, alphas)

    def render_images(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        sh_degree_to_use: int = -1,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render ``C`` multi-channel images (see :attr:`num_channels`) from this :class:`GaussianSplat3d` from ``C`` camera views.

        .. note::

            All images being rendered must have the same width and height.


        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Render images from C camera views.
            # images is a tensor of shape [C, H, W, D] where D is the number of channels
            # alpha_images is a tensor of shape [C, H, W, 1]
            images, alpha_images = gaussian_splat_3d.render_images(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the images
                image_height, # height of the images
                near, # near clipping plane
                far) # far clipping plane

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.
            backgrounds (torch.Tensor | None): Optional background colors of shape ``(C, D)``.
                If ``None``, background is treated as 0.
            masks (torch.Tensor | None): Optional per-pixel boolean mask of shape ``(C, H, W)``.
                ``True`` means render, ``False`` means skip (filled with background).


        Returns:
            images (torch.Tensor): A tensor of shape ``(C, H, W, D)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, ``W`` is the width of the images, and ``D`` is the number of channels.
            alpha_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, and ``W`` is the width of the images.
                Each element represents the alpha value (opacity) at a pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        return self._render_dense(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
            masks=masks,
            render_mode=GaussianRenderMode.FEATURES,
            world_space=False,
        )

    def render_images_from_world(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        sh_degree_to_use: int = -1,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        crop: Crop | None = None,
        crop_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render dense images by rasterizing directly from world-space 3D Gaussians.

        This is similar to :meth:`render_images`, but the rasterization step is performed in 3D
        using per-pixel rays against the Gaussian ellipsoids (instead of rasterizing 2D conics
        produced by a projection step). This enables gradients w.r.t. Gaussian geometry
        (``means``, ``quats``, ``log_scales``) through rasterization, which is useful for
        Unscented Transform (UT)-based OpenCV camera models.

        Notes:
            - This is **dense-only**: outputs are dense tensors of shape ``(C, H, W, ...)``.
            - Tile intersection data is still computed from a (non-differentiable) projection
              step, so gradients can be discontinuous when small parameter changes cause a Gaussian
              to enter/leave a tile (or switch which tiles it overlaps).
            - Background compositing follows standard "over" alpha compositing. If
              ``backgrounds`` is provided, the output color is:

              ``color = sum_i (feat_i * alpha_i * T_i) + T_final * background``

              where ``T_final`` is the remaining transmittance at the end of rasterization, and
              ``alpha = 1 - T_final``.
            - ``masks`` is a **per-pixel** boolean mask of shape ``(C, H, W)``.
              ``True`` means render, ``False`` means skip (filled with background).


        Example:

        .. code-block:: python

            images, alphas = gaussian_splat_3d.render_images_from_world(
                world_to_camera_matrices,  # [C,4,4]
                projection_matrices,       # [C,3,3]
                image_width=640,
                image_height=480,
                near=0.01,
                far=1e10,
                camera_model=fvdb_reality_capture.CameraModel.OPENCV_RATIONAL_8,
                distortion_coeffs=dist_coeffs,  # [C,12]
                backgrounds=bg,                 # [C,D]
                masks=pixel_mask,              # [C,H,W] (optional)
            )

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)``.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)``.
            image_width (int): Output image width ``W``.
            image_height (int): Output image height ``H``.
            near (float): Near clipping plane.
            far (float): Far clipping plane.
            camera_model (CameraModel): Semantic camera model used for ray generation.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape
                ``(C, 12)`` and packed layout ``[k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]``.
                Required for :class:`CameraModel.OPENCV_*` camera models; use a zero-filled
                tensor to represent no distortion. For :class:`CameraModel.PINHOLE` and
                :class:`CameraModel.ORTHOGRAPHIC`, pass ``None`` or a ``(C, 12)`` tensor,
                which is ignored.
            sh_degree_to_use (int): SH degree to use. ``-1`` means use all available SH bases.
            tile_size (int): Tile size (in pixels). ``tileH = ceil(H / tile_size)``,
                ``tileW = ceil(W / tile_size)``.
            min_radius_2d (float): Minimum projected radius (in pixels) used for tiling/culling.
            eps_2d (float): Padding used during tiling/projection to avoid numerical issues.
            antialias (bool): If ``True``, applies opacity correction (when available) when using
                ``eps_2d > 0.0``.
            backgrounds (torch.Tensor | None): Optional background colors of shape ``(C, D)``,
                where ``D`` is :attr:`num_channels`. If ``None``, background is treated as 0.
            masks (torch.Tensor | None): Optional per-pixel boolean mask of shape ``(C, H, W)``.
                ``True`` means render, ``False`` means skip (filled with background).

            crop (tuple[int, int, int, int] | None): Optional ``(origin_w, origin_h, width, height)`` window
                to render instead of the full image, clipped to the image; tiles outside it are skipped and
                the output has the clipped size; a crop entirely outside the image gives an empty render.
            crop_masks (torch.Tensor | None): Optional per-pixel boolean mask in crop coordinates, of the crop's
                requested or clipped size, as an alternative to the image-coordinate ``masks`` when a crop is given.

        Returns:
            images (torch.Tensor): Rendered images of shape ``(C, H, W, D)``.
            alpha_images (torch.Tensor): Alpha images of shape ``(C, H, W, 1)``.
        """
        return self._render_dense(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
            masks=masks,
            render_mode=GaussianRenderMode.FEATURES,
            world_space=True,
            crop=crop,
            crop_masks=crop_masks,
        )

    def render_depths_from_world(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        crop: Crop | None = None,
        crop_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render dense depth images by rasterizing directly from world-space 3D Gaussians.

        This mirrors :meth:`render_images_from_world`, but renders depth-only outputs with the
        same camera-model and projection-method dispatch. ``crop`` and ``crop_masks`` behave as there.
        """
        return self._render_dense(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            sh_degree_to_use=-1,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
            masks=masks,
            render_mode=GaussianRenderMode.DEPTH,
            world_space=True,
            crop=crop,
            crop_masks=crop_masks,
        )

    def sparse_render_images(
        self,
        pixels_to_render: JaggedTensorOrTensorT,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        sh_degree_to_use: int = -1,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[JaggedTensorOrTensorT, JaggedTensorOrTensorT]:
        """
        Render ``C`` collections of multi-channel features (see :attr:`num_channels`) from this :class:`GaussianSplat3d` from ``C`` camera views
        at the specified pixel locations.

        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # pixels_to_render is a tensor of shape [C, P, 2] containing pixel coordinates to render
            # Render sparse images from C camera views at specified pixel locations
            # features is a tensor of shape [C, P, D] where D is the number of channels
            # alphas is a tensor of shape [C, P, 1]
            features, alphas = gaussian_splat_3d.sparse_render_images(
                pixels_to_render, # tensor of shape [C, P, 2]
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the images
                image_height, # height of the images
                near, # near clipping plane
                far) # far clipping plane

        Args:
            pixels_to_render (torch.Tensor | JaggedTensor): A tensor of shape ``(C, P, 2)`` or a :class:`~fvdb.JaggedTensor` where ``C`` is the number of camera views,
                and ``P`` is the number of pixel coordinates to render per camera. Each pixel coordinate is represented as (y, x) (row, col).
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.
            backgrounds (torch.Tensor | None): Optional background colors of shape ``(C, D)``
                where ``D`` is :attr:`num_channels`. If ``None``, background is treated as 0.
            masks (torch.Tensor | None): Optional per-tile boolean mask of shape
                ``(C, tileH, tileW)`` where ``tileH = ceil(image_height / tile_size)`` and
                ``tileW = ceil(image_width / tile_size)``. ``True`` means the tile is rendered,
                ``False`` means the tile is skipped and its pixels receive the background value
                with zero alpha.

        Returns:
            features (torch.Tensor | JaggedTensor): A tensor of shape ``(C, P, D)`` or a
                :class:`~fvdb.JaggedTensor` where ``C`` is the number of camera views,
                ``P`` is the number of pixel coordinates rendered per camera, and ``D`` is the number of channels.
            alpha_images (torch.Tensor | JaggedTensor): A tensor of shape ``(C, P, 1)`` or a :class:`~fvdb.JaggedTensor`
                where ``C`` is the number of camera views, and ``P`` is the number of pixel coordinates rendered per camera.
                Each element represents the alpha value (opacity) at that pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        pixels_jt = as_pixel_jagged(pixels_to_render)
        features, alphas = self._render_sparse(
            pixels_to_render=pixels_jt,
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
            masks=masks,
            render_mode=GaussianRenderMode.FEATURES,
        )
        return self._sparse_result(pixels_to_render, features, alphas)

    def sparse_render_images_and_depths(
        self,
        pixels_to_render: JaggedTensorOrTensorT,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        sh_degree_to_use: int = -1,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[JaggedTensorOrTensorT, JaggedTensorOrTensorT]:
        """
        Render ``C`` collections of sparse multi-channel features (see :attr:`num_channels`) with depth as
        the last channel from this :class:`GaussianSplat3d` from ``C`` camera views at the specified pixel locations.

        Example:

            .. code-block:: python

                # Assume gaussian_splat_3d is an instance of GaussianSplat3d
                # pixels_to_render is a tensor of shape [C, P, 2] containing pixel coordinates to render
                # Render sparse images with depth from C camera views at specified pixel locations
                # features is a tensor of shape [C, P, D + 1] where D is the number of channels
                # alphas is a tensor of shape [C, P, 1]
                features, alphas = gaussian_splat_3d.sparse_render_images_and_depths(
                    pixels_to_render, # tensor of shape [C, P, 2]
                    world_to_camera_matrices, # tensor of shape [C, 4, 4]
                    projection_matrices, # tensor of shape [C, 3, 3]
                    image_width, # width of the images
                    image_height, # height of the images
                    near, # near clipping plane
                    far) # far clipping plane

        Args:
            pixels_to_render (torch.Tensor | JaggedTensor): A tensor of shape ``(C, P, 2)`` or a :class:`~fvdb.JaggedTensor` where ``C`` is the number of camera views,
                and ``P`` is the number of pixel coordinates to render per camera. Each pixel coordinate is represented as (y, x) (row, col).
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.
            backgrounds (torch.Tensor | None): Optional background values of shape ``(C, D+1)``
                where ``D`` is :attr:`num_channels` (the last element is the background depth).
                If ``None``, background is treated as 0.
            masks (torch.Tensor | None): Optional per-tile boolean mask of shape
                ``(C, tileH, tileW)`` where ``tileH = ceil(image_height / tile_size)`` and
                ``tileW = ceil(image_width / tile_size)``. ``True`` means the tile is rendered,
                ``False`` means the tile is skipped and its pixels receive the background value
                with zero alpha.

        Returns:
            features_with_depths (torch.Tensor | JaggedTensor): A tensor of shape ``(C, P, D + 1)`` or a
                :class:`~fvdb.JaggedTensor` where ``C`` is the number of camera views,
                ``P`` is the number of pixel coordinates rendered per camera, and ``D`` is the number of channels. The last channel
                represents the depth value at that pixel.
            alpha_images (torch.Tensor | JaggedTensor): A tensor of shape ``(C, P, 1)`` or a :class:`~fvdb.JaggedTensor`
                where ``C`` is the number of camera views, and ``P`` is the number of pixel coordinates rendered per camera.
                Each element represents the alpha value (opacity) at that pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        pixels_jt = as_pixel_jagged(pixels_to_render)
        features, alphas = self._render_sparse(
            pixels_to_render=pixels_jt,
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
            masks=masks,
            render_mode=GaussianRenderMode.FEATURES_AND_DEPTH,
        )
        return self._sparse_result(pixels_to_render, features, alphas)

    def render_images_and_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        sh_degree_to_use: int = -1,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render ``C`` multi-channel images (see :attr:`num_channels`) with depth as the last channel from this :class:`GaussianSplat3d` from ``C`` camera views.

        .. note::

            All images being rendered must have the same width and height.


        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Render images with depth maps from C camera views.
            # images is a tensor of shape [C, H, W, D + 1] where D is the number of channels
            # alpha_images is a tensor of shape [C, H, W, 1]
            images, alpha_images = gaussian_splat_3d.render_images_and_depths(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the images
                image_height, # height of the images
                near, # near clipping plane
                far) # far clipping plane

            images = images[..., :-1]  # Extract image channels

            depths = images[..., -1:] / alpha_images  # Extract depth channel by dividing by alpha

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.
            backgrounds (torch.Tensor | None): Optional background colors of shape ``(C, D+1)``.
                If ``None``, background is treated as 0.
            masks (torch.Tensor | None): Optional per-pixel boolean mask of shape ``(C, H, W)``.
                ``True`` means render, ``False`` means skip (filled with background).

        Returns:
            images (torch.Tensor): A tensor of shape ``(C, H, W, D + 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, ``W`` is the width of the images, and ``D`` is the number of channels.
            alpha_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, and ``W`` is the width of the images.
                Each element represents the alpha value (opacity) at a pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        return self._render_dense(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
            masks=masks,
            render_mode=GaussianRenderMode.FEATURES_AND_DEPTH,
            world_space=False,
        )

    def render_images_and_depths_from_world(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        sh_degree_to_use: int = -1,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        crop: Crop | None = None,
        crop_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render dense RGBD images by rasterizing directly from world-space 3D Gaussians.

        This mirrors :meth:`render_images_from_world`, but returns image channels with depth in the
        final channel while using the same camera-model and projection-method dispatch. ``crop`` and
        ``crop_masks`` behave as there.
        """
        return self._render_dense(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
            masks=masks,
            render_mode=GaussianRenderMode.FEATURES_AND_DEPTH,
            world_space=True,
            crop=crop,
            crop_masks=crop_masks,
        )

    def render_num_contributing_gaussians(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Renders ``C`` images where each pixel contains the number of contributing Gaussians for that pixel from ``C`` camera views.

        .. note::

            All images being rendered must have the same width and height.


        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Render images from C camera views.
            # images is a tensor of shape [C, H, W, D] where D is the number of channels
            # alpha_images is a tensor of shape [C, H, W, 1]
            num_gaussians, alpha_images = gaussian_splat_3d.render_num_contributing_gaussians(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the images
                image_height, # height of the images
                near, # near clipping plane
                far) # far clipping plane

            num_gaussians_cij = num_gaussians[c, i, j, 0]  # Number of contributing Gaussians at pixel (i, j) in camera c

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, ``W`` is the width of the images.
                Each element represents the number of contributing Gaussians at that pixel.
            alpha_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, and ``W`` is the width of the images.
                Each element represents the alpha value (opacity) at a pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        with torch.no_grad():
            projected, opacities = self._project_and_opacities(
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                camera_model=camera_model,
                projection_method=projection_method,
                distortion_coeffs=distortion_coeffs,
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
            )
            tiles = intersect_gaussian_tiles(projected, tile_size=tile_size, opacities=opacities)
            return rasterize_num_contributing_gaussians(projected, opacities, tiles)

    @overload
    def sparse_render_num_contributing_gaussians(
        self,
        pixels_to_render: torch.Tensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def sparse_render_num_contributing_gaussians(
        self,
        pixels_to_render: JaggedTensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...

    def sparse_render_num_contributing_gaussians(
        self,
        pixels_to_render: JaggedTensor | torch.Tensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[JaggedTensor | torch.Tensor, JaggedTensor | torch.Tensor]:
        """
        Renders the number of Gaussians which contribute to each pixel specified in the input.

        .. seealso::

            :meth:`render_num_contributing_gaussians` for rendering dense images of contributing Gaussians.


        Args:
            pixels_to_render (torch.Tensor | JaggedTensor): A :class:`fvdb.JaggedTensor` of shape ``(C, R_c, 2)`` representing the
                pixels to render for each camera, where ``C`` is the number of camera views and ``R_c`` is the
                number of pixels to render per camera. Each value is an (x, y) pixel coordinate.
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            num_contributing_gaussians (torch.Tensor | JaggedTensor): A tensor of shape ``(C, R)`` (if this method was called with ``pixels_to_render`` as a :class:`torch.Tensor`)
                or a :class:`fvdb.JaggedTensor` of shape ``(C, R_c)`` (if this method was called with ``pixels_to_render`` as a :class:`fvdb.JaggedTensor`)
                where ``C`` is the number of cameras, and ``R``/``R_c`` is the number of pixels to render per camera.
                Each element represents the number of contributing Gaussians at that pixel.
            alphas (torch.Tensor | JaggedTensor): A tensor of shape ``(C, R)`` (if this method was called with ``pixels_to_render`` as a :class:`torch.Tensor`)
                or a :class:`fvdb.JaggedTensor` of shape ``(C, R_c)`` (if this method was called with ``pixels_to_render`` as a :class:`fvdb.JaggedTensor`)
                where ``C`` is the number of cameras, and ``R``/``R_c`` is the number of pixels to render per camera.
                Each element represents the alpha value (opacity) at that pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        pixels_jt = as_pixel_jagged(pixels_to_render)
        with torch.no_grad():
            projected, opacities = self._project_and_opacities(
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                camera_model=camera_model,
                projection_method=projection_method,
                distortion_coeffs=distortion_coeffs,
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
            )
            sparse_tiles = intersect_gaussian_tiles_sparse(
                pixels_jt, projected, tile_size=tile_size, opacities=opacities
            )
            counts, alphas = rasterize_num_contributing_gaussians_sparse(projected, opacities, sparse_tiles)
        return self._sparse_result(pixels_to_render, counts, alphas)

    def render_contributing_gaussian_ids(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        top_k_contributors: int = 0,
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Render the IDs of the Gaussians that are the contributors to the rendered images' pixels and the value of their weighted contributions to the rendered pixels.

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.
            top_k_contributors (int): If greater than 0, returns only the top ``k`` most opaque Gaussians contributing to each pixel.
                If 0 (default), returns all contributing Gaussians per pixel.

        Returns:
            ids (fvdb.JaggedTensor): A ``[[C1P1 + C1P2 + ... C1P(imageWidth * imageHeight), 1], ... [CNP1 + CNP2 + ... CNP(imageWidth * imageHeight), 1]]``
                jagged tensor containing the IDs of the contributing Gaussians of each rendered pixel for each camera.
            weights (fvdb.JaggedTensor): A ``[[C1P1 + C1P2 + ... C1P(imageWidth * imageHeight), 1], ... [CNP1 + CNP2 + ... CNP(imageWidth * imageHeight), 1]]``
                jagged tensor containing the weights of the contributing Gaussians of each rendered pixel for each camera. The weights are in row-major order and
                sum to 1 for each pixel if that pixel is opaque (alpha=1).
        """
        with torch.no_grad():
            projected, opacities = self._project_and_opacities(
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                camera_model=camera_model,
                projection_method=projection_method,
                distortion_coeffs=distortion_coeffs,
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
            )
            tiles = intersect_gaussian_tiles(projected, tile_size=tile_size, opacities=opacities)
            return rasterize_contributing_gaussian_ids(projected, opacities, tiles, top_k_contributors)

    @overload
    def sparse_render_contributing_gaussian_ids(
        self,
        pixels_to_render: torch.Tensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        top_k_contributors: int = 0,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...

    @overload
    def sparse_render_contributing_gaussian_ids(
        self,
        pixels_to_render: JaggedTensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        top_k_contributors: int = 0,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...

    def sparse_render_contributing_gaussian_ids(
        self,
        pixels_to_render: JaggedTensor | torch.Tensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        camera_model: CameraModel = CameraModel.PINHOLE,
        projection_method: ProjectionMethod = ProjectionMethod.AUTO,
        distortion_coeffs: torch.Tensor | None = None,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        top_k_contributors: int = 0,
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Render the IDs of the Gaussians that are the contributors to the rendered images'
        pixels and the value of their weighted contributions to the rendered pixels. This
        function will render only a sparse subset of the pixels in the overall image, as specified
        by the ``pixels_to_render`` parameter.

        Args:
            pixels_to_render (torch.Tensor | JaggedTensor): A :class:`torch.Tensor` of shape ``(C, R, 2)``
                or a :class:`fvdb.JaggedTensor` of shape ``(C, R_c, 2)`` representing the
                pixels to render for each camera, where ``C`` is the number of camera views and ``R``/``R_c`` is the
                number of pixels to render per camera. Each value is an (x, y) pixel coordinate.
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for ``C`` cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            camera_model (CameraModel): Semantic camera model for projection. Default is
                :attr:`fvdb_reality_capture.CameraModel.PINHOLE`.
            projection_method (ProjectionMethod): Projection implementation selector. Default is
                :attr:`fvdb_reality_capture.ProjectionMethod.AUTO`.
            distortion_coeffs (torch.Tensor | None): Distortion coefficients with shape ``(C, 12)``.
                Required for :class:`CameraModel.OPENCV_*` camera models. For
                :class:`CameraModel.PINHOLE` and :class:`CameraModel.ORTHOGRAPHIC`, pass
                ``None`` or a ``(C, 12)`` tensor, which is ignored. To represent no
                distortion with an OpenCV camera model, pass a zero-filled tensor.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.
            top_k_contributors (int): If greater than 0, returns only the top ``k`` most opaque Gaussians contributing to each pixel,
                If 0 (default), returns all contributing Gaussians per pixel.

        Returns:
            ids (fvdb.JaggedTensor): A ``[[C1P1 + C1P2 + ... C1PN1, 1], ... [CNP1 + CNP2 + ... CNPNN, 1]]`` jagged tensor
                containing the IDs of the contributing Gaussians of each rendered pixel for each camera. The IDs are in row-major order.
            weights (fvdb.JaggedTensor): A ``[[C1P1 + C1P2 + ... C1PN1, 1], ... [CNP1 + CNP2 + ... CNPNN, 1]]`` jagged tensor
                containing the weights of the contributing Gaussians of each rendered pixel for each camera. The weights are in row-major order and sum to 1 for each pixel if that pixel is opaque (alpha=1).
        """
        pixels_jt = as_pixel_jagged(pixels_to_render)
        with torch.no_grad():
            projected, opacities = self._project_and_opacities(
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                camera_model=camera_model,
                projection_method=projection_method,
                distortion_coeffs=distortion_coeffs,
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
            )
            sparse_tiles = intersect_gaussian_tiles_sparse(
                pixels_jt, projected, tile_size=tile_size, opacities=opacities
            )
            return rasterize_contributing_gaussian_ids_sparse(projected, opacities, sparse_tiles, top_k_contributors)

    def relocate_gaussians(
        self,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        ratios: torch.Tensor,
        binomial_coeffs: torch.Tensor,
        n_max: int,
        min_opacity: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Relocate Gaussians by adjusting opacity and scale based on replication ratio.

        Args:
            log_scales (torch.Tensor): Log scales of the Gaussians to relocate [N, 3].
            logit_opacities (torch.Tensor): Logit opacities of the Gaussians to relocate [N].
            ratios (torch.Tensor): Replication ratios per Gaussian [N].
            binomial_coeffs (torch.Tensor): Binomial coefficients table [nMax, nMax].
            n_max (int): Maximum replication ratio (size of binomial table).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (logit_opacities_new [N], log_scales_new [N, 3]).
        """
        return fvdb_functional.mcmc_relocate_gaussians(
            log_scales, logit_opacities, ratios, binomial_coeffs, n_max, min_opacity
        )

    def add_noise_to_means(self, noise_scale: float, t: float = 0.005, k: float = 100.0) -> None:
        """
        Add noise to the Gaussian positions (means), scaled by ``noise_scale``.

        Args:
            noise_scale (float): Noise scale factor applied to scale-dependent noise.
            t (float): Parameter t for noise scaling. Defaults to 0.005.
            k (float): Parameter k for noise scaling. Defaults to 100.0.
        """
        fvdb_functional.mcmc_add_noise_to_means(
            self._means, self._log_scales, self._logit_opacities, self._quats, noise_scale, t, k
        )

    def reset_accumulated_gradient_state(self) -> None:
        """
        Reset the accumulated projected gradients of the mans if :attr:`accumulate_mean_2d_gradients` is ``True``,
        and the accumulated max 2D radii if :attr:`accumulate_max_2d_radii` is ``True``.

        The values of :attr:`accumulated_projected_mean_2d_gradients`, :attr:`accumulated_max_2d_radii`,
        and :attr:`accumulated_gradient_step_counts` will be zeroed out after this call.

        .. seealso::
            :meth:`accumulate_mean_2d_gradients` :meth:`accumulate_max_2d_radii` which control if
            we accumulate these values during rendering and backward passes.

        .. seealso::
            :attr:`accumulated_mean_2d_gradient_norms` :attr:`accumulated_max_2d_radii` :attr:`accumulated_gradient_step_counts`
            for the actual accumulated state being reset.

        """
        if self._accumulated_mean_2d_gradient_norms is not None:
            self._accumulated_mean_2d_gradient_norms.zero_()
        if self._accumulated_gradient_step_counts is not None:
            self._accumulated_gradient_step_counts.zero_()
        if self._accumulated_max_2d_radii is not None:
            self._accumulated_max_2d_radii.zero_()

    def save_ply(
        self, filename: pathlib.Path | str, metadata: Mapping[str, str | int | float | torch.Tensor] | None = None
    ) -> None:
        """
        Save this :class:`GaussianSplat3d` to a PLY file. and include any metadata provided.

        Args:
            filename (pathlib.Path | str): The path to the PLY file to save.
            metadata (dict[str, str | int | float | torch.Tensor] | None): An optional dictionary of metadata
                where the keys are strings and the values are either strings, ints, floats, or tensors. Defaults to ``None``,
        """
        if isinstance(filename, pathlib.Path):
            filename = str(filename)
        fvdb_functional.save_gaussian_ply(
            filename, self._means, self._quats, self._log_scales, self._logit_opacities, self._sh0, self._shN, metadata
        )

    @overload
    def to(self, dtype: torch.dtype | None = None) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        device: DeviceIdentifier | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        other: torch.Tensor,
    ) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        other: "GaussianSplat3d",
    ) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        other: Grid,
    ) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        other: GridBatch,
    ) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        other: JaggedTensor,
    ) -> "GaussianSplat3d": ...

    def to(
        self,
        *args,
        **kwargs,
    ) -> "GaussianSplat3d":
        """
        Move the :class:`GaussianSplat3d` instance to a different device or change its data type or both.

        Args:
            other (DeviceIdentifier | torch.Tensor | GaussianSplat3d | Grid | GridBatch | JaggedTensor):
                The target :class:`torch.Device`, :class:`torch.Tensor`,
                :class:`~fvdb.Grid`, :class:`~fvdb.GridBatch`, :class:`~fvdb.JaggedTensor`,
                or :class:`~fvdb_reality_capture.GaussianSplat3d` instance to which the
                :class:`GaussianSplat3d` instance should be moved.
            device (DeviceIdentifier, optional): The target ``device`` to move the :class:`GaussianSplat3d` instance to.
            dtype (torch.dtype, optional): The target data type for the :class:`GaussianSplat3d` instance.

        Returns:
           gaussian_splat_3d (GaussianSplat3d): A new instance of :class:`GaussianSplat3d` with the specified device and/or data type.
        """

        # All values passed by keyword arguments
        if len(args) == 0:
            if len(kwargs) == 1:
                # .to(device=...) or .to(other=...)
                if "device" in kwargs:
                    device = kwargs["device"]
                    dtype = kwargs.get("dtype", self.dtype)
                elif "other" in kwargs:
                    other = kwargs["other"]
                    if isinstance(other, (torch.Tensor, JaggedTensor, GaussianSplat3d)):
                        device = other.device
                        dtype = other.dtype
                    elif isinstance(other, (GridBatch, Grid)):
                        device = other.device
                        dtype = self.dtype
                else:
                    raise TypeError(
                        f"Invalid keyword arguments for to(): {kwargs}. Expected 'device' or 'other' and optionally 'dtype'."
                    )
            elif len(kwargs) == 2:
                # .to(device=..., dtype=...) or .to(dtype=..., device=...)
                if "device" in kwargs and "dtype" in kwargs:
                    device = kwargs["device"]
                    dtype = kwargs["dtype"]
                else:
                    raise TypeError(
                        f"Invalid keyword arguments for to(): {kwargs}. Expected 'device' or 'other' and optionally 'dtype'."
                    )
            else:
                raise TypeError(
                    f"Invalid keyword arguments for to(): {kwargs}. Expected 'device' or 'other' and optionally 'dtype'."
                )

        elif len(args) == 1 and isinstance(args[0], (torch.Tensor, GaussianSplat3d, JaggedTensor)):
            # .to(other)
            device = args[0].device
            dtype = args[0].dtype
        elif len(args) == 1 and isinstance(args[0], (GridBatch, Grid)):
            # .to(other)
            device = args[0].device
            dtype = self.dtype
        elif len(args) == 1:
            # .to(device)
            device = args[0]
            dtype = kwargs.get("dtype", self.dtype)
        elif len(args) == 2:
            # .to(device, dtype)
            device = args[0]
            dtype = args[1]
        else:
            raise TypeError(
                f"Invalid arguments for to(): {args}. Expected a DeviceIdentifier, torch.Tensor, GaussianSplat3d, GridBatch, or JaggedTensor."
            )

        device = resolve_device(device, inherit_from=self)
        dtype = self.dtype if dtype is None else cast_check(dtype, torch.dtype, "dtype")

        def _move(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None:
                return None
            if t.is_floating_point():
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)

        return GaussianSplat3d(
            means=_move(self._means),  # type: ignore[arg-type]
            quats=_move(self._quats),  # type: ignore[arg-type]
            log_scales=_move(self._log_scales),  # type: ignore[arg-type]
            logit_opacities=_move(self._logit_opacities),  # type: ignore[arg-type]
            sh0=_move(self._sh0),  # type: ignore[arg-type]
            shN=_move(self._shN),  # type: ignore[arg-type]
            accumulate_mean_2d_gradients=self._accumulate_mean_2d_gradients,
            accumulate_max_2d_radii=self._accumulate_max_2d_radii,
            accumulated_mean_2d_gradient_norms=_move(self._accumulated_mean_2d_gradient_norms),
            accumulated_gradient_step_counts=_move(self._accumulated_gradient_step_counts),
            accumulated_max_2d_radii_tensor=_move(self._accumulated_max_2d_radii),
            _private=GaussianSplat3d.__PRIVATE__,
        )

    def set_state(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor,
    ) -> None:
        """
        Set the underlying tensors managed by this :class:`GaussianSplat3d` instance.

        Note: If :attr:`accumulate_mean_2d_gradients` and/or :attr:`accumulate_max_2d_radii` are ``True``,
        this method will reset the gradient state (see :meth:`reset_accumulated_gradient_state`).

        Args:
            means (torch.Tensor): Tensor of shape ``(N, 3)`` representing the means of the Gaussians.
                ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
            quats (torch.Tensor): Tensor of shape ``(N, 4)`` representing the quaternions of the Gaussians.
                ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
            log_scales (torch.Tensor): Tensor of shape ``(N, 3)`` representing the log scales of the Gaussians.
                ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
            logit_opacities (torch.Tensor): Tensor of shape ``(N,)`` representing the logit opacities of the Gaussians.
                ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
            sh0 (torch.Tensor): Tensor of shape ``(N, 1, D)`` representing the diffuse SH coefficients
                where ``N`` is the number of Gaussians (see :attr:`num_gaussians`), and ``D`` is the number of channels (see :attr:`num_channels`).
            shN (torch.Tensor): Tensor of shape ``(N, K-1, D)`` representing the directionally
                varying SH coefficients where ``N`` is the number of Gaussians (see :attr:`num_gaussians`),
                ``D`` is the number of channels (see :attr:`num_channels`),
                and ``K`` is the number of spherical harmonic bases (see :attr:`num_sh_bases`).
        """
        self._means = means
        self._quats = quats
        self._log_scales = log_scales
        self._logit_opacities = logit_opacities
        self._sh0 = sh0
        self._shN = shN
        self._accumulated_mean_2d_gradient_norms = None
        self._accumulated_gradient_step_counts = None
        self._accumulated_max_2d_radii = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Return a dictionary containing the state of the GaussianSplat3d instance.
        This is useful for serializing the state of the object for saving or transferring.

        A state dictionary always contains the following keys where ``N`` denotes the number of Gaussians (see :attr:`num_gaussians`):

        - ``'means'``: Tensor of shape ``(N, 3)`` representing the means of the Gaussians.
        - ``'quats'``: Tensor of shape ``(N, 4)`` representing the quaternions of the Gaussians.
        - ``'log_scales'``: Tensor of shape ``(N, 3)`` representing the log scales of the Gaussians.
        - ``'logit_opacities'``: Tensor of shape ``(N,)`` representing the logit opacities of the Gaussians.
        - ``'sh0'``: Tensor of shape ``(N, 1, D)`` representing the diffuse SH coefficients
          where ``D`` is the number of channels (see :attr:`num_channels`).
        - ``'shN'``: Tensor of shape ``(N, K-1, D)`` representing the directionally varying SH
          coefficients where ``D`` is the number of channels (see :attr:`num_channels`), and ``K``
          is the number of spherical harmonic bases (see :attr:`num_sh_bases`).
        - ``'accumulate_max_2d_radii'``: bool Tensor with a single element indicating
          whether to track the maximum 2D radii for gradients.
        - ``'accumulate_mean_2d_gradients'``: bool Tensor with a single element indicating whether
          to track the average norm of the gradient of projected means for each Gaussian.

        It can also optionally contain the following keys if :attr:`accumulate_mean_2d_gradients` and/or :attr:`accumulate_max_2d_radii` are set to ``True``:

        - ``'accumulated_gradient_step_counts'``: Tensor of shape ``(N,)`` representing the
          accumulated gradient step counts for each Gaussian.
        - ``'accumulated_max_2d_radii'``: Tensor of shape ``(N,)`` representing the maximum
          2D projected radius for each Gaussian across every iteration of optimization.
        - ``'accumulated_mean_2d_gradient_norms'``: Tensor of shape ``(N,)`` representing the
          average norm of the gradient of projected means for each Gaussian across every iteration of optimization.


        .. seealso:: :meth:`from_state_dict` for constructing a :class:`GaussianSplat3d` from a state dictionary.

        Returns:
            state_dict (dict[str, torch.Tensor]): A dictionary containing the state of
                the :class:`GaussianSplat3d` instance.
        """
        d: dict[str, torch.Tensor] = {
            "means": self._means,
            "quats": self._quats,
            "log_scales": self._log_scales,
            "logit_opacities": self._logit_opacities,
            "sh0": self._sh0,
            "shN": self._shN,
            "accumulate_mean_2d_gradients": torch.tensor(self._accumulate_mean_2d_gradients),
            "accumulate_max_2d_radii": torch.tensor(self._accumulate_max_2d_radii),
        }
        if self._accumulated_mean_2d_gradient_norms is not None:
            d["accumulated_mean_2d_gradient_norms"] = self._accumulated_mean_2d_gradient_norms
        if self._accumulated_gradient_step_counts is not None:
            d["accumulated_gradient_step_counts"] = self._accumulated_gradient_step_counts
        if self._accumulated_max_2d_radii is not None:
            d["accumulated_max_2d_radii"] = self._accumulated_max_2d_radii
        return d


# TODO: Make a batched class to encapsulate this jagged rendering pipeline.
def gaussian_render_jagged(
    means: JaggedTensor,  # [N1 + N2 + ..., 3]
    quats: JaggedTensor,  # [N1 + N2 + ..., 4]
    scales: JaggedTensor,  # [N1 + N2 + ..., 3]
    opacities: JaggedTensor,  # [N1 + N2 + ...]
    sh_coeffs: JaggedTensor,  # [N1 + N2 + ..., K, 3]
    viewmats: JaggedTensor,  # [C1 + C2 + ..., 4, 4]
    Ks: JaggedTensor,  # [C1 + C2 + ..., 3, 3]
    image_width: int,
    image_height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    sh_degree_to_use: int = -1,
    tile_size: int = 16,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    antialias: bool = False,
    render_depth_channel: bool = False,
    return_debug_info: bool = False,
    ortho: bool = False,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Render Gaussian splats with jagged (variable-length) batched inputs.

    This function composes differentiable projection, SH evaluation, tile intersection,
    and rasterization stages, each backed by Python ``torch.autograd.Function`` wrappers
    around the underlying CUDA/CPU dispatch kernels.

    Args:
        means: Jagged tensor of Gaussian centers ``[sum(N_i), 3]``.
        quats: Jagged tensor of Gaussian quaternions ``[sum(N_i), 4]``.
        scales: Jagged tensor of Gaussian scales ``[sum(N_i), 3]``.
        opacities: Jagged tensor of Gaussian opacities ``[sum(N_i)]``.
        sh_coeffs: Jagged tensor of SH coefficients ``[sum(N_i), K, D]``.
        viewmats: Jagged tensor of rigid world-to-camera matrices ``[sum(C_i), 4, 4]``.
        Ks: Jagged tensor of intrinsic matrices ``[sum(C_i), 3, 3]``.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        near_plane: Near clipping plane distance.
        far_plane: Far clipping plane distance.
        sh_degree_to_use: SH degree to evaluate (``-1`` means use all available bases).
        tile_size: Rasterization tile size in pixels.
        radius_clip: Minimum 2-D radius for projected Gaussians.
        eps2d: Epsilon added to 2-D covariance diagonal for numerical stability.
        antialias: Whether to apply antialiasing compensation to opacities.
        render_depth_channel: If ``True``, append a depth channel to the rendered colors.
        return_debug_info: If ``True``, return intermediate tensors in the debug dict.
        ortho: Use orthographic projection.
        backgrounds: Optional per-camera background colors ``[total_cameras, D, H, W]``.
        masks: Optional per-camera masks ``[total_cameras, 1, H, W]``.

    Returns:
        A tuple ``(rendered_images, rendered_alphas, debug_info)`` where
        ``rendered_images`` has shape ``[total_cameras, D, H, W]`` and
        ``rendered_alphas`` has shape ``[total_cameras, 1, H, W]``.
    """
    ccz = viewmats.jdata.size(0)  # total cameras across all batches

    # --- Build cross-batch index arrays ---
    # TODO: This indexing logic is convoluted but there is no better way without
    # custom CUDA kernels.  Given Gaussians with shape [sum(N_i), ...] and cameras
    # with shape [sum(C_i), ...], we compute the cross-product of each batch's
    # Gaussians with that batch's cameras, producing a flat tensor of shape
    # [sum(C_i * N_i), ...].  We need to track two index arrays:
    #   camera_ids:   shape [sum(C_i * N_i)], values in [0, sum(C_i))
    #   gaussian_ids: shape [sum(C_i * N_i)], values in [0, sum(N_i))
    # g_sizes: [N1, N2, ...], c_sizes: [C1, C2, ...]
    g_sizes = means.joffsets[1:] - means.joffsets[:-1]
    c_sizes = Ks.joffsets[1:] - Ks.joffsets[:-1]

    # camera_ids: flat index into viewmats.jdata for each (gaussian, camera) pair
    tt = g_sizes.repeat_interleave(c_sizes)
    camera_ids = torch.arange(ccz, device=means.device, dtype=torch.int32).repeat_interleave(tt, 0)

    # gaussian_ids: flat index into means.jdata for each pair
    dd0 = means.joffsets[:-1].repeat_interleave(c_sizes, 0)
    dd1 = means.joffsets[1:].repeat_interleave(c_sizes, 0)
    shifts = dd0[1:] - dd1[:-1]
    shifts = torch.cat([torch.tensor([0], device=means.device), shifts])
    shifts_cumsum = shifts.cumsum(0, dtype=torch.int32)
    gaussian_ids = torch.arange(camera_ids.size(0), device=means.device, dtype=torch.int32)
    gaussian_ids = gaussian_ids + shifts_cumsum.repeat_interleave(tt, 0)

    # --- Differentiable projection ---
    radii, means2d, depths, conics, compensations = _ProjectGaussiansJaggedFn.apply(
        g_sizes,
        means.jdata,
        quats.jdata,
        scales.jdata,
        c_sizes,
        viewmats.jdata,
        Ks.jdata,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        ortho,
    )

    # Gather opacities per (gaussian, camera) pair
    opacities_batched = opacities.jdata[gaussian_ids]
    if antialias:
        opacities_batched = opacities_batched * compensations
    opacities_batched = opacities_batched.contiguous()

    debug_info: dict[str, torch.Tensor] = {}
    if return_debug_info:
        debug_info["camera_ids"] = camera_ids
        debug_info["gaussian_ids"] = gaussian_ids
        debug_info["radii"] = radii
        debug_info["means2d"] = means2d
        debug_info["depths"] = depths
        debug_info["conics"] = conics
        debug_info["opacities"] = opacities_batched

    # --- Differentiable SH evaluation ---
    K = sh_coeffs.jdata.size(-2)
    actual_sh_degree = int(math.sqrt(K) - 1) if sh_degree_to_use < 0 else sh_degree_to_use

    # Permute [total_G, K, D] → [K, total_G, D], then gather by gaussian_ids → [K, nnz, D]
    sh_coeffs_batched = sh_coeffs.jdata.permute(1, 0, 2)[:, gaussian_ids, :]

    if actual_sh_degree == 0:
        sh0 = sh_coeffs_batched[0, :, :].unsqueeze(0)  # [1, nnz, D]
        render_quantities = _EvaluateGaussianSHFn.apply(
            actual_sh_degree,
            1,
            means.jdata,
            viewmats.jdata,
            camera_ids,
            gaussian_ids,
            sh0.permute(1, 0, 2),  # [nnz, 1, D]
            None,
            radii.unsqueeze(0),  # [1, nnz, 2] (per-axis)
        )
    else:
        sh0 = sh_coeffs_batched[0, :, :].unsqueeze(0)  # [1, nnz, D]
        shN = sh_coeffs_batched[1:, :, :]  # [K-1, nnz, D]
        render_quantities = _EvaluateGaussianSHFn.apply(
            actual_sh_degree,
            1,
            means.jdata,
            viewmats.jdata,
            camera_ids,
            gaussian_ids,
            sh0.permute(1, 0, 2),  # [nnz, 1, D]
            shN.permute(1, 0, 2),  # [nnz, K-1, D]
            radii.unsqueeze(0),  # [1, nnz, 2] (per-axis)
        )
    render_quantities = render_quantities.squeeze(0)  # [nnz, D]

    if render_depth_channel:
        render_quantities = torch.cat([render_quantities, depths[gaussian_ids].unsqueeze(-1)], dim=-1)

    # --- Non-differentiable tile intersection ---
    num_tiles_h = math.ceil(image_height / tile_size)
    num_tiles_w = math.ceil(image_width / tile_size)
    tile_offsets, tile_gaussian_ids_t = fvdb_functional.intersect_gaussian_tiles(
        means2d,
        radii,
        depths,
        ccz,
        tile_size,
        num_tiles_h,
        num_tiles_w,
        camera_ids=camera_ids,
        conics=conics,
        opacities=opacities_batched,
    )
    if return_debug_info:
        debug_info["tile_offsets"] = tile_offsets
        debug_info["tile_gaussian_ids"] = tile_gaussian_ids_t

    # --- Differentiable rasterization ---
    rendered_images, rendered_alphas = _RasterizeScreenSpaceGaussiansFn.apply(
        means2d,
        conics,
        render_quantities,
        opacities_batched,
        image_width,
        image_height,
        0,  # image_origin_w
        0,  # image_origin_h
        tile_size,
        tile_offsets,
        tile_gaussian_ids_t,
        False,  # absgrad
        backgrounds,
        masks,
    )

    return rendered_images, rendered_alphas, debug_info


def evaluate_spherical_harmonics(
    sh_degree: int,
    num_cameras: int,
    means: torch.Tensor,
    sh0: torch.Tensor,
    radii: torch.Tensor,
    shN: torch.Tensor | None = None,
    world_to_camera_matrices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate spherical harmonics to compute view-dependent features/colors.

    Args:
        sh_degree: Degree of spherical harmonics to use (0-3 typically).
        num_cameras: Number of camera views (C).
        means: World-space Gaussian means with shape [N, 3].
        sh0: DC term coefficients with shape [N, 1, D].
        radii: Per-axis projected radii with shape [C, N, 2] (int32). A point
               is masked out unless both axes are positive.
        shN: Higher-order SH coefficients with shape [N, K-1, D] where
             K = (sh_degree+1)^2. Required when sh_degree > 0.
        world_to_camera_matrices: Rigid world-to-camera transforms with shape
            [C, 4, 4]. Rotation blocks must be orthonormal. Required when
            sh_degree > 0.

    Returns:
        Tensor of shape [C, N, D] containing the evaluated features/colors.
    """
    if sh0.ndim != 3:
        raise ValueError(f"sh0 must be 3-dimensional [N, 1, D], got {sh0.ndim}D")
    if sh0.shape[1] != 1:
        raise ValueError(f"sh0.shape[1] must be 1, got {sh0.shape[1]}")
    if radii.ndim != 3 or radii.shape[-1] != 2:
        raise ValueError(f"radii must be shape [C, N, 2], got {tuple(radii.shape)}")
    if sh0.shape[0] != radii.shape[1]:
        raise ValueError(f"sh0.shape[0] ({sh0.shape[0]}) must match radii.shape[1] ({radii.shape[1]})")
    if sh_degree > 0 and world_to_camera_matrices is None:
        raise ValueError("world_to_camera_matrices is required when sh_degree > 0")
    if world_to_camera_matrices is None:
        world_to_camera_matrices = sh0.new_empty(0)
    if shN is None:
        N, _, D = sh0.shape
        shN = sh0.new_empty(N, 0, D)

    empty_ids = torch.empty(0, dtype=torch.int32, device=means.device)
    return _EvaluateGaussianSHFn.apply(
        sh_degree,
        num_cameras,
        means,
        world_to_camera_matrices,
        empty_ids,
        empty_ids,
        sh0,
        shN,
        radii,
    )


__all__ = [
    "GaussianSplat3d",
    "ProjectedGaussianSplats",
    "gaussian_render_jagged",
    "evaluate_spherical_harmonics",
]
