# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as nnf
import torch.optim
from fvdb import GaussianSplat3d
from scipy.special import logit

from fvdb_reality_capture.sfm_scene import SfmScene

from .base_gaussian_splat_optimizer import BaseGaussianSplatOptimizer


class InsertionGrad2dThresholdMode(str, Enum):
    """
    The `GaussianSplatOptimizer` uses a threshold on the accumulated norm of 2D mean gradients to use during refinement.

    There are several modes for computing this threshold, specified by the config. These modes let you adapt the
    refinement behavior to the statistics of the gradients during training.
    """

    CONSTANT = "constant"
    """
    Always use the fixed threshold specified by ``self._config.insertion_grad_2d_threshold``. This mode with a default
    value (``0.0002``) will produce okay results, but may not be optimal for all types of captures.
    """

    PERCENTILE_FIRST_ITERATION = "percentile_first_iteration"
    """
    During the first refinement step, set the threshold to the given percentile of the gradients. For all subsequent
    refinement steps, use that fixed threshold. Using this mode will have similar behavior to ``CONSTANT`` but will
    adapt to the scale of the gradients which can be more robust across different capture types.
    """

    PERCENTILE_EVERY_ITERATION = "percentile_every_iteration"
    """
    During **every** refinement step, set the threshold to the given percentile of the gradients. For highly detailed
    scenes, this mode may be useful to adaptively insert more Gaussians as the model learns more detail. This generally
    produces many more Gaussians and more detailed results at the cost of more memory and compute.
    """


class SpatialScaleMode(str, Enum):
    """
    How to interpret 3D optimization scale thresholds (insertion_scale_3d_threshold, deletion_scale_3d_threshold)
    and learning rates. These thresholds specified in a unitless space, and are subsequently multipled by a spatial scale
    computed from the scene being optimized. There are several heuristics for computing this spatial scale, specified by the config:
    """

    ABSOLUTE_UNITS = "absolute_units"
    """
    Use the thresholds and learning rates *as-is*, in absolute world units (e.g. meters).
    """

    MEDIAN_CAMERA_DEPTH = "median_camera_depth"
    """
    Compute the median depth of SfmPoints across of all cameras in the scene, and use that as the spatial scale.
    """

    MAX_CAMERA_DEPTH = "max_camera_depth"
    """
    Compute the maximum depth of SfmPoints across all cameras in the scene, and use that as the spatial scale.
    """

    MAX_CAMERA_TO_CENTROID = "max_camera_diagonal"
    """
    Compute the maximum distance from any camera to the centroid of all camera positions
    (good for orbits around an object).
    """

    SCENE_DIAGONAL_PERCENTILE = "relative_to_scene_diagonal"
    """
    Compute the axis-aligned bounding box of all points within the 5th to 95th percentile range along each axis, and
    use the given percentile of the length of the diagonal of this box as the spatial scale.
    """


@dataclass
class GaussianSplatOptimizerConfig:
    """
    Parameters for configuring the ``GaussianSplatOptimizer``.
    """

    max_gaussians: int = -1
    """The maximum number of Gaussians to allow in the model. If -1, no limit."""

    insertion_grad_2d_threshold_mode: InsertionGrad2dThresholdMode = InsertionGrad2dThresholdMode.CONSTANT
    """
    Whether to use a fixed threshold for :obj:`insertion_grad_2d_threshold` (constant), a value computed as a percentile of
    the distribution of screen space mean gradients on the first iteration, or a percentile value
    computed at each refinement step.

    See :class:`InsertionGrad2dThresholdMode` for details on the available modes.
    """

    deletion_opacity_threshold: float = 0.005
    """
    If a Gaussian's opacity drops below this value, delete it during refinement.
    """

    deletion_scale_3d_threshold: float = 0.1
    """
    If a Gaussian's 3d scale is above this value, then delete it during refinement.
    """

    deletion_scale_2d_threshold: float = 0.15
    """
    If the maximum projected size of a Gaussian between refinement steps exceeds this value then delete it during
    refinement.

    .. note:: This parameter is only used if set :obj:`use_screen_space_scales_for_refinement_until` is greater than 0.
    """

    insertion_grad_2d_threshold: float = (
        0.0002 if insertion_grad_2d_threshold_mode == InsertionGrad2dThresholdMode.CONSTANT else 0.9
    )
    """
    Threshold value on the accumulated norm of projected mean gradients between refinement steps to
    determine whether a Gaussian has high error and is a candidate for duplication or splitting.

    .. note:: If :obj:`insertion_grad_2d_threshold_mode` is :obj:`InsertionGrad2dThresholdMode.CONSTANT`, then this value
              is used directly as the threshold, and **must be positive**.

    .. note:: If :obj:`insertion_grad_2d_threshold_mode` is :obj:`InsertionGrad2dThresholdMode.PERCENTILE_FIRST_ITERATION`
              or :obj:`InsertionGrad2dThresholdMode.PERCENTILE_EVERY_ITERATION`, then this value must be in the
              range ``(0.0, 1.0)`` (exclusive).
    """

    insertion_scale_3d_threshold: float = 0.01
    """
    Duplicate high-error (determined by :obj:`insertion_grad_2d_threshold`) Gaussians whose 3d scale is below this value.
    These Gaussians are too small to capture the detail in the region they cover, so we duplicate them to
    allow them to specialize.
    """

    insertion_scale_2d_threshold: float = 0.05
    """
    Split high-error (determined by :obj:`insertion_grad_2d_threshold`) Gaussians whose maximum projected
    size exceeds this value. These Gaussians are too large to capture the detail in the region they cover,
    so we split them to allow them to specialize.

    .. note:: This parameter is only used if set :obj:`use_screen_space_scales_for_refinement_until` is greater than 0.

    """

    opacity_updates_use_revised_formulation: bool = False
    """
    When splitting Gaussians, whether to update the opacities of the new Gaussians using the revised formulation from
    `*"Revising Densification in Gaussian Splatting"* <https://arxiv.org/abs/2404.06109>`_.
    This removes a bias which weighs newly split Gaussians contribution to the image more heavily than
    older Gaussians.
    """

    insertion_split_factor: int = 2
    """
    When splitting Gaussians during insertion, this value specifies the total number of new Gaussians that will
    replace each selected source Gaussian. The original is removed and replaced by :obj:`insertion_split_factor` new
    Gaussians. *e.g.* if this value is 2, each split Gaussian is replaced by 2 new smaller Gaussians
    (the original is removed). This value must be >= 2.
    """

    insertion_duplication_factor: int = 2
    """
    When duplicating Gaussians during insertion, this value specifies the total number of copies (including
    the original) that will result for each selected source Gaussian. The original is kept, and
    ``insertion_duplication_factor - 1`` new identical copies are added. *e.g.* if this value is 3,
    each duplicated Gaussian becomes 3 copies of itself (the original plus 2 new). This value must be >= 2.
    """

    reset_opacities_every_n_refinements: int = 30
    """
    If set to a positive value, then clamp all opacities to be at most twice the value of
    :obj:`deletion_opacity_threshold` every time :func:`GaussianSplatOptimizer.refine` is called
    :obj:`reset_opacities_every_n_refinements` times. This prevents Gaussians from becoming completely occluded by
    denser Gaussians and thus unable to be optimized.
    """

    use_scales_for_deletion_after_n_refinements: int = reset_opacities_every_n_refinements
    """
    If set to a positive value, then after ``use_scales_for_deletion_after_n_refinements`` calls to
    :func:`GaussianSplatOptimizer.refine`, use the 3D scales of the Gaussians to determine whether to delete them.
    This will delete Gaussians that have grown
    too large in 3D space and are not contributing to the optimization.

    By default, this value matches :obj:`reset_opacities_every_n_refinements` so that both behaviors are enabled at the
    same time.
    """

    use_screen_space_scales_for_refinement_until: int = 0
    """
    If set to a positive value, then use threshold the maximum projected size of Gaussians between refinement steps
    to decide whether to split or delete Gaussians that are too large. This behavior is enabled until
    :func:`GaussianSplatOptimizer.refine` has been called ``use_screen_space_scales_for_refinement_until`` times.
    After that, only 3D scales are used for refinement.
    """

    spatial_scale_mode: SpatialScaleMode = SpatialScaleMode.MEDIAN_CAMERA_DEPTH
    """
    How to interpret 3D optimization scale thresholds and learning rates (*i.e.* :obj:`insertion_scale_3d_threshold`,
    :obj:`deletion_scale_3d_threshold`, and :obj:`means_lr`). These are scaled by a spatial scale computed from
    the scene, so they are relative to the size of the scene being optimized.

    See :class:`SpatialScaleMode` for details on the available modes.
    """

    spatial_scale_multiplier: float = 1.1
    """
    Multiplier to apply to the spatial scale computed from the scene to get a slightly larger scale.
    """

    means_lr: float = 1.6e-4
    """
    Learning rate for the means of the Gaussians. This is also scaled by the spatial scale computed from the scene.

    See :obj:`spatial_scale_mode` for details on how the spatial scale is computed.
    """

    log_scales_lr: float = 5e-3
    """
    Learning rate for the log scales of the Gaussians.
    """

    quats_lr: float = 1e-3
    """
    Learning rate for the quaternions of the Gaussians.
    """

    logit_opacities_lr: float = 5e-2
    """
    Learning rate for the logit opacities of the Gaussians.
    """

    sh0_lr: float = 2.5e-3
    """
    Learning rate for the diffuse spherical harmonics (order 0).
    """

    shN_lr: float = 2.5e-3 / 20
    """
    Learning rate for the specular spherical harmonics (order > 0).
    """


class GaussianSplatOptimizer(BaseGaussianSplatOptimizer):
    """
    Optimizer for reconstructing a scene using Gaussian Splat radiance fields over a collection of posed images.

    The optimizer uses an Adam optimizer to optimize the parameters of a ``fvdb.GaussianSplat3d`` model, and
    provides utilities to refine the model by inserting and deleting Gaussians based on their contribution to the
    optimization. The tools here mostly follow the algorithm in the original Gaussian Splatting paper
    (https://arxiv.org/abs/2308.04079).

    .. note:: You should not call the constructor of this class directly. Instead use :func:`from_model_and_config`
              or :func:`from_state_dict`.

    """

    __PRIVATE__ = object()

    def __init__(
        self,
        model: GaussianSplat3d,
        optimizer: torch.optim.Adam,
        config: GaussianSplatOptimizerConfig,
        spatial_scale: float,
        refine_count: int,
        step_count: int,
        _private: Any = None,
    ):
        """
        Create a new ``GaussianSplatOptimizer`` instance from a model, optimizer and a config.

        .. note:: You should not call this constructor directly.  Instead use :func:`from_model_and_config`
                  or :func:`from_state_dict`.

        Args:
            model (GaussianSplat3d): The ``GaussianSplat3d`` model to optimize.
            optimizer (torch.optim.Adam): The optimizer for the model.
            config (GaussianSplatOptimizerConfig): Configuration options for the optimizer.
            spatial_scale (float): A spatial scale for the scene used to interpret 3D scale thresholds in the config.
            refine_count (int): The number of times :func:`refine()` has been called on this optimizer.
            step_count (int): The number of times :func:`step()` has been called on this optimizer.
            _private (Any): A private object to prevent direct instantiation. Must be
                :obj:`GaussianSplatOptimizer.__PRIVATE__`.
        """
        if _private is not self.__PRIVATE__:
            raise RuntimeError(
                "GaussianSplatOptimizer must be created using from_model_and_config() or from_state_dict()"
            )
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        # How many timeds we've called step() on this optimizer
        self._step_count = step_count

        # How many times we've called refine() on this optimizer
        self._refine_count = refine_count

        # A spatial scale for the scene used to interpret 3D scale thresholds in the config
        self._spatial_scale = spatial_scale

        self._config = config
        self._model = model
        self._model.accumulate_mean_2d_gradients = True  # Make sure we track the 2D gradients for refinement
        if self._config.use_screen_space_scales_for_refinement_until > 0:
            self._model.accumulate_max_2d_radii = True  # Make sure we track the max 2D radii for refinement
        if self._config.insertion_split_factor < 2:
            raise ValueError("insertion_split_factor must be >= 2")
        if self._config.insertion_duplication_factor < 2:
            raise ValueError("insertion_duplication_factor must be >= 2")

        # This hook counts the number of times we call backward between zeroing the gradients.
        # To determine whether a Gaussian should be split or duplicated, we threshold the *average*
        # gradient of its 2D mean with respect to the loss.
        # If we call backward multiple times per iteration (e.g. for different losses) or if we're accumulating gradients,
        # then the denominator of the average is the total number of backward calls since the last zero_grad().
        # This hook corrects the count even if backward() is called multiple times per iteration.
        self._num_grad_accumulation_steps = 1  # Number of times we've called backward since zeroing the gradients

        def _count_accumulation_steps_backward_hook(_):
            self._num_grad_accumulation_steps += 1

        self._model.means.register_hook(_count_accumulation_steps_backward_hook)

        # The actual numeric value to use when thresholding the 2D gradient to decide whether to grow a Gaussian.
        # This depends on the mode specified in the config.
        self._insertion_grad_2d_abs_threshold: float | None = (
            self._config.insertion_grad_2d_threshold
            if self._config.insertion_grad_2d_threshold_mode == InsertionGrad2dThresholdMode.CONSTANT
            else None
        )
        if self._config.insertion_grad_2d_threshold_mode != InsertionGrad2dThresholdMode.CONSTANT:
            if not (0.0 < self._config.insertion_grad_2d_threshold < 1.0):
                raise ValueError(
                    "insertion_grad_2d_threshold must be in the range (0.0, 1.0) when using percentile modes"
                )
        else:
            if self._insertion_grad_2d_abs_threshold is None or self._insertion_grad_2d_abs_threshold <= 0.0:
                raise ValueError("insertion_grad_2d_threshold must be > 0.0 when using CONSTANT mode")

        self._optimizer = optimizer

        # Store the decay exponent for the means learning rate schedule so we can serialize it
        self._means_lr_decay_exponent = 1.0

    def reset_learning_rates_and_decay(self, batch_size: int, expected_steps: int):
        """
        Set the learning rates and learning rate decay factor based on the batch size and the expected
        number of optimization steps (*i.e.* the number of times :func:`step()` is called).

        This is useful if you want to change the batch size or expected number of steps after creating
        the optimizer.

        Args:
            batch_size (int): The batch size used for training. This is used to scale the learning rates.
            expected_steps (int): The expected number of optimization steps.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if expected_steps <= 0:
            raise ValueError("expected_steps must be > 0")

        self._means_lr_decay_exponent = 0.01 ** (1.0 / expected_steps)

        # Scale the learning rate and momentum parameters (epsilon, betas) based on batch size,
        # reference: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # Note that this will not make the training exactly equivalent to the original INRIA
        # Gaussian splat implementation.
        # See https://arxiv.org/pdf/2402.18824v1 for more details.
        lr_batch_rescale = math.sqrt(float(batch_size))

        # Store learning rates in a dictionary so we can look them up
        # using param_group['name'] for each parameter group in the optimizer.
        reset_lr_values = {
            "means": self._config.means_lr * self._spatial_scale * lr_batch_rescale,
            "log_scales": self._config.log_scales_lr * lr_batch_rescale,
            "quats": self._config.quats_lr * lr_batch_rescale,
            "logit_opacities": self._config.logit_opacities_lr * lr_batch_rescale,
            "sh0": self._config.sh0_lr * lr_batch_rescale,
            "shN": self._config.shN_lr * lr_batch_rescale,
        }

        rescaled_betas = (1.0 - batch_size * (1.0 - 0.9), 1.0 - batch_size * (1.0 - 0.999))
        for param_group in self._optimizer.param_groups:
            param_group["betas"] = rescaled_betas
            param_group["lr"] = reset_lr_values[param_group["name"]]
            param_group["eps"] = 1e-15 / lr_batch_rescale

    @classmethod
    def from_model_and_scene(
        cls,
        model: GaussianSplat3d,
        sfm_scene: SfmScene,
        config: GaussianSplatOptimizerConfig = GaussianSplatOptimizerConfig(),
    ) -> "GaussianSplatOptimizer":
        """
        Create a new ``GaussianSplatOptimizer`` instance from a model and config.

        Args:
            model (GaussianSplat3d): The ``GaussianSplat3d`` model to optimize.
            config (GaussianSplatOptimizerConfig): Configuration options for the optimizer.
            means_lr_scale (float): A scale factor to apply to the means learning rate.
            means_lr_decay_exponent (float): The exponent used for decaying the means learning rate.
            batch_size (int): The batch size used for training. This is used to scale the learning rates.

        Returns:
            GaussianSplatOptimizer: A new :class:`GaussianSplatOptimizer` instance.
        """

        spatial_scale = (
            cls._compute_spatial_scale(sfm_scene, config.spatial_scale_mode) * config.spatial_scale_multiplier
        )
        optimizer = GaussianSplatOptimizer._make_optimizer(model, spatial_scale, config)

        return cls(
            model=model,
            optimizer=optimizer,
            config=config,
            spatial_scale=spatial_scale,
            refine_count=0,
            step_count=0,
            _private=cls.__PRIVATE__,
        )

    @classmethod
    def from_state_dict(cls, model: GaussianSplat3d, state_dict: dict[str, Any]) -> "GaussianSplatOptimizer":
        """
        Create a new :class:`GaussianSplatOptimizer` instance from a model and a state dict.

        Args:
            model (GaussianSplat3d): The :class:`GaussianSplat3d` model to optimize.
            state_dict (dict[str, Any]): A state dict previously obtained from :func:`state_dict`.
        Returns:
            optimizer (GaussianSplatOptimizer): A new :class:`GaussianSplatOptimizer` instance.
        """
        if "version" not in state_dict:
            raise ValueError("State dict is missing version information")
        if state_dict["version"] not in (3,):
            raise ValueError(f"Unsupported version: {state_dict['version']}")

        config = GaussianSplatOptimizerConfig(**state_dict["config"])

        # We pass in 1.0 for the means_lr_scale since this is already baked into the optimizer state
        # which we load below.
        optimizer = GaussianSplatOptimizer._make_optimizer(model=model, means_lr_scale=1.0, config=config)
        optimizer.load_state_dict(state_dict["optimizer"])

        optimizer = cls(
            model=model,
            optimizer=optimizer,
            spatial_scale=state_dict["spatial_scale"],
            config=config,
            step_count=state_dict["step_count"],
            refine_count=state_dict["refine_count"],
            _private=cls.__PRIVATE__,
        )
        optimizer._insertion_grad_2d_abs_threshold = state_dict["insertion_grad_2d_abs_threshold"]
        optimizer._means_lr_decay_exponent = state_dict["means_lr_decay_exponent"]

        return optimizer

    def step(self):
        """
        Step the optimizer (updating the model's parameters) and decay the learning rate of the means.
        """
        self._optimizer.step()
        self._step_count += 1
        # Decay the means learning rate
        for param_group in self._optimizer.param_groups:
            if param_group["name"] == "means":
                param_group["lr"] *= self._means_lr_decay_exponent
                return

    def zero_grad(self, set_to_none: bool = False):
        """
        Zero the gradients of all tensors being optimized.

        Args:
            set_to_none (bool): If ``True``, set the gradients to ``None`` instead of zeroing them.
                This can be more memory efficient.
        """
        self._num_grad_accumulation_steps = 0
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """
        Return a serializable state dict for the optimizer.

        Returns:
            state_dict (dict[str, Any]): A state dict containing the state of the optimizer.
        """
        return {
            "optimizer": self._optimizer.state_dict(),
            "means_lr_decay_exponent": self._means_lr_decay_exponent,
            "insertion_grad_2d_abs_threshold": self._insertion_grad_2d_abs_threshold,
            "num_grad_accumulation_steps": self._num_grad_accumulation_steps,
            "config": vars(self._config),
            "spatial_scale": self._spatial_scale,
            "step_count": self._step_count,
            "refine_count": self._refine_count,
            "version": 3,
        }

    @torch.no_grad()
    def filter_gaussians(self, indices_or_mask: torch.Tensor):
        """
        Filter the Gaussians in the model to only those specified by the given indices or mask
        and update the optimizer state accordingly. This can be used to delete, shuffle, or duplicate
        the Gaussians during optimization.

        Args:
            indices_or_mask (torch.Tensor): A 1D tensor of indices or a boolean mask indicating which Gaussians to keep.
        """

        def _copy_param_and_grad(param: torch.Tensor) -> torch.Tensor:
            new_param = param[indices_or_mask]
            new_param.grad = param.grad[indices_or_mask] if param.grad is not None else None
            return new_param

        self._model.set_state(
            means=_copy_param_and_grad(self._model.means),
            quats=_copy_param_and_grad(self._model.quats),
            log_scales=_copy_param_and_grad(self._model.log_scales),
            logit_opacities=_copy_param_and_grad(self._model.logit_opacities),
            sh0=_copy_param_and_grad(self._model.sh0),
            shN=_copy_param_and_grad(self._model.shN),
        )
        self._update_optimizer_params_and_state(lambda x: x[indices_or_mask])

    @torch.no_grad()
    def refine(self, zero_gradients: bool = True) -> dict[str, int]:
        """
        Perform a step of refinement by inserting Gaussians where more detail is needed and deleting Gaussians that are
        not contributing to the optimization. Refinement happens via three mechanisms:

        **Duplication**: Make :obj:`~GaussianSplatOptimizerConfig.insertion_duplication_factor` copies of a Gaussian.

          We duplicate a Gaussian if its 3D size is below some threshold and the gradient of its projected means over
          time is high on average. Intuitively, this means the Gaussian is not taking up a lot of space in the scene,
          but consistently wants to change positions when viewed from different cameras. Likely this Gaussian is stuck
          trying to represent too much of the scene and should be split into multiple copies.

        **Splitting**: Split a Gaussian into :obj:`~GaussianSplatOptimizerConfig.insertion_split_factor` smaller ones.

          We split a Gaussian when its 3D size exceeds a threshold value and the gradient of its projected mean over
          time is high on average. In this case, a Gaussian is likely too large for the amount of detail it represents
          and should be split to capture detail in the image.

        **Deletion**: Removing a Gaussian from the scene.

          We delete a Gaussian if its opacity falls below a threshold since it is not contributing much to
          rendered images.


        Args:
            zero_gradients (bool): If ``True``, zero the gradients after refinement.

        Returns:
            refine_stats (dict[str, int]): A dictionary containing statistics about the refinement step with the keys:

                * ``"num_duplicated"``: The number of Gaussians that were duplicated.
                * ``"num_split"``: The number of Gaussians that were split.
                * ``"num_deleted"``: The number of Gaussians that were deleted.
        """

        use_screen_space_scales = self._refine_count < self._config.use_screen_space_scales_for_refinement_until
        if not use_screen_space_scales and self._model.accumulate_max_2d_radii:
            # We no longer need to track the max 2D radii since we're not using them for refinement
            self._model.accumulate_max_2d_radii = False

        is_duplicated, is_split = self._compute_insertion_masks(use_screen_space_scales)

        use_scales_for_deletion = self._refine_count > self._config.use_scales_for_deletion_after_n_refinements
        is_deleted = self._compute_deletion_mask(use_scales_for_deletion, use_screen_space_scales)

        # We won't insert Gaussians which are up for deletion since they will be deleted anyway
        is_duplicated.logical_and_(~is_deleted)
        is_split.logical_and_(~is_deleted)

        # Get the new Gaussians to add from splitting and duplication
        duplication_indices = torch.where(is_duplicated)[0]
        split_indices = torch.where(is_split)[0]

        num_split = len(split_indices)
        num_duplicated = len(duplication_indices)
        num_deleted = int(is_deleted.sum().item())

        # Should we reset opacities after refinement?
        should_reset_opacities = (
            self._refine_count > 0
            and self._config.reset_opacities_every_n_refinements > 0
            and self._refine_count % self._config.reset_opacities_every_n_refinements == 0
        )

        # The net number of Gaussians added to the total number of Gaussians after refinement
        num_added_gaussians = (
            num_duplicated * (self._config.insertion_duplication_factor - 1)
            + num_split * (self._config.insertion_split_factor - 1)
            - num_deleted
        )
        num_gaussians_before_refinement = self._model.num_gaussians
        num_gaussians_after_refinement = num_gaussians_before_refinement + num_added_gaussians
        if self._config.max_gaussians > 0 and num_gaussians_after_refinement > self._config.max_gaussians:
            self._logger.warning(
                f"Refinement would insert a net of {num_added_gaussians} leading to {num_gaussians_after_refinement} which exceeds max_gaussians ({self._config.max_gaussians}), skipping refinement step"
            )
            if should_reset_opacities:
                self._reset_opacities()

            self._refine_count += 1
            return {"num_duplicated": 0, "num_split": 0, "num_deleted": 0}

        # Get indices of Gaussians which are preserved during refinement
        kept_indices = torch.where(~(is_split | is_deleted))[0]

        duplicated_gaussians = self._compute_duplicated_gaussians(duplication_indices)
        split_gaussians = self._compute_split_gaussians(split_indices)

        def _cat_parameter(param: torch.Tensor, name: str) -> torch.Tensor:
            ret = torch.cat([param[kept_indices], duplicated_gaussians[name], split_gaussians[name]], dim=0)
            num_added_gaussians = duplicated_gaussians[name].shape[0] + split_gaussians[name].shape[0]
            # If you want to preserve gradients, we'll do so by creating a new tensor and copying
            # over the gradients of the kept parameters, and setting the gradients of the new parameters to zero.
            if param.grad is not None and not zero_gradients:
                ret.grad = torch.cat(
                    [
                        param.grad[kept_indices],
                        torch.zeros(num_added_gaussians, *param.shape[1:], dtype=param.dtype, device=param.device),
                    ],
                    dim=0,
                )
            else:
                ret.grad = None
            return ret

        # We no longer need the accumulated gradient state since we've used it to compute masks for refinement
        # Reset it so we can start accumulating for the next refinement step
        self._model.reset_accumulated_gradient_state()
        self._model.set_state(
            means=_cat_parameter(self._model.means, "means"),
            quats=_cat_parameter(self._model.quats, "quats"),
            log_scales=_cat_parameter(self._model.log_scales, "log_scales"),
            logit_opacities=_cat_parameter(self._model.logit_opacities, "logit_opacities"),
            sh0=_cat_parameter(self._model.sh0, "sh0"),
            shN=_cat_parameter(self._model.shN, "shN"),
        )

        def update_state_function(x: torch.Tensor):
            num_kept = kept_indices.shape[0]
            total_gaussians = (
                num_kept
                + num_duplicated * (self._config.insertion_duplication_factor - 1)
                + num_split * self._config.insertion_split_factor
            )
            ret = torch.zeros(total_gaussians, *x.shape[1:], dtype=x.dtype, device=x.device)
            ret[0:num_kept] = x[kept_indices]
            return ret

        self._update_optimizer_params_and_state(update_state_function)

        if should_reset_opacities:
            self._reset_opacities()

        self._refine_count += 1
        self._logger.debug(
            f"Optimizer refinement (step {self._step_count:,}): {num_duplicated:,} duplicated, {num_split:,} split, {num_deleted:,} pruned. "
            f"Before refinement model had {num_gaussians_before_refinement:,} Gaussians, after refinement has {num_gaussians_after_refinement:,} Gaussians."
        )
        return {"num_duplicated": num_duplicated, "num_split": num_split, "num_deleted": num_deleted}

    @torch.no_grad()
    def _reset_opacities(self):
        """
        Clamp the logit_opacities of all Gaussians to be less than or equal to a small value above
        the deletion threshold. This is useful to call periodically during optimization to prevent
        Gaussians from becoming completely occluded by denser Gaussians, and thus unable to be optimized.
        """
        # Clamp all opacities to be less than or equal to twice the deletion threshold
        clip_value = logit(self._config.deletion_opacity_threshold * 2.0)
        self._model.logit_opacities.clamp_max_(clip_value)
        # This operation invalidates any existing gradients since the tracked
        # adam states no longer make sense after clamping, and we want any gradient
        # steps after this to not be influenced by previous gradients.
        self._model.logit_opacities.grad = None
        self._update_optimizer_params_and_state(
            lambda x: x.zero_(), parameter_names={"logit_opacities"}, reset_adam_step_counts=True
        )
        self._logger.debug(
            f"Reset all opacities to be <= 2 * deletion_opacity_threshold ({self._config.deletion_opacity_threshold * 2.0:.3f})"
        )

    @staticmethod
    def _make_optimizer(model, means_lr_scale, config):
        """
        Make an Adam optimizer for the given model and config.
        This is just a helper function used by the constructors since this logic is shared and verbose.

        Args:
            model (GaussianSplat3d): The model to optimize.
            means_lr_scale (float): A scale factor to apply to the means learning rate.
            config (GaussianSplatOptimizerConfig): The configuration for the optimizer.

        Returns:
            optimizer (torch.optim.Adam): An Adam optimizer for the model.
        """
        return torch.optim.Adam(
            [
                {"params": model.means, "lr": means_lr_scale * config.means_lr, "name": "means"},
                {
                    "params": model.log_scales,
                    "lr": config.log_scales_lr,
                    "name": "log_scales",
                },
                {"params": model.quats, "lr": config.quats_lr, "name": "quats"},
                {
                    "params": model.logit_opacities,
                    "lr": config.logit_opacities_lr,
                    "name": "logit_opacities",
                },
                {"params": model.sh0, "lr": config.sh0_lr, "name": "sh0"},
                {"params": model.shN, "lr": config.shN_lr, "name": "shN"},
            ],
            eps=1e-15,
            betas=(0.9, 0.999),
            fused=True,
        )

    @torch.no_grad()
    def _compute_insertion_grad_2d_threshold(self, accumulated_mean_2d_gradients: torch.Tensor) -> float:
        """
        Compute the threshold on the accumulated norm of 2D mean gradients to use during refinement.

        There are several modes for computing this threshold, specified by the config:

        - ``CONSTANT``: Always use the fixed threshold specified by ``self._config.insertion_grad_2d_threshold``.
        - ``PERCENTILE_FIRST_ITERATION``: During the first refinement step, set the threshold to the given percentile
        of the gradients. For all subsequent refinement steps, use that fixed threshold.
        - ``PERCENTILE_EVERY_ITERATION``: During every refinement step, set the threshold to the given percentile
        of the gradients.

        These modes let you adapt the refinement behavior to the statistics of the gradients during training.
        Generally ``CONSTANT`` with a default value (0.0002) will produce okay results, but may not be optimal for all types
        of captures. Using ``PERCENTILE_FIRST_ITERATION`` will have similar behavior to ``CONSTANT`` but will adapt to
        the scale of the gradients, which can be more robust across different capture types.
        For highly detailed scenes, ``PERCENTILE_EVERY_ITERATION`` may be useful to adaptively insert more Gaussians
        as the model learns more detail. This generally produces many more Gaussians and more detailed results at the
        cost of more memory and compute.

        Args:
            accumulated_mean_2d_gradients (torch.Tensor): The average norm of the projected mean gradients of shape (num_gaussians,).
                This is typically obtained from ``model.accumulated_mean_2d_gradient_norms / model.accumulated_gradient_step_counts``
                where ``model.accumulated_gradient_step_counts`` is the number of optimization steps since the last refinement.

        Returns:
            insertion_grad_2d_threshold (float): The threshold value to use for deciding whether to insert Gaussians during refinement.
        """

        # Helper to compute the quantile of the gradients, using NumPy if we have too many Gaussians for torch.quantile
        # which has a cap at 2**24 elements
        def _grad_2d_quantile(quantile: float) -> float:
            if self._model.num_gaussians > 2**24:
                # torch.quantile has a cap at 2**24 elements so fall back to NumPy for large numbers of Gaussians
                self._logger.debug("Using numpy to compute gradient percentile threshold")
                return float(np.quantile(accumulated_mean_2d_gradients.cpu().numpy(), quantile))
            else:
                return torch.quantile(accumulated_mean_2d_gradients, quantile).item()

        # Determine the threshold for the 2D projected gradient based on the selected mode
        if self._config.insertion_grad_2d_threshold_mode == InsertionGrad2dThresholdMode.CONSTANT:
            # In CONSTANT mode, we always use the fixed threshold specified by self._grow_grad2d_threshold
            assert self._insertion_grad_2d_abs_threshold is not None
            return self._insertion_grad_2d_abs_threshold

        elif self._config.insertion_grad_2d_threshold_mode == InsertionGrad2dThresholdMode.PERCENTILE_FIRST_ITERATION:
            # In PERCENTILE_FIRST_ITERATION mode, we set the threshold to the given percentile of the gradients
            # during the first refinement step, and then use that fixed threshold for all subsequent steps
            if self._insertion_grad_2d_abs_threshold is None:
                self._insertion_grad_2d_abs_threshold = _grad_2d_quantile(self._config.insertion_grad_2d_threshold)
                self._logger.debug(
                    f"Setting fixed grad2d threshold to {self._insertion_grad_2d_abs_threshold:.6f} corresponding to the "
                    f"({self._config.insertion_grad_2d_threshold * 100:.1f} percentile)"
                )
            assert self._insertion_grad_2d_abs_threshold is not None
            return self._insertion_grad_2d_abs_threshold

        elif self._config.insertion_grad_2d_threshold_mode == InsertionGrad2dThresholdMode.PERCENTILE_EVERY_ITERATION:
            # In PERCENTILE_EVERY_ITERATION mode, we set the threshold to the given percentile of the gradients
            # during every refinement step
            return _grad_2d_quantile(self._config.insertion_grad_2d_threshold)

        else:
            raise RuntimeError("Invalid mode for insertion_grad_2d_threshold.")

    @torch.no_grad()
    def _compute_insertion_masks(
        self, use_screen_space_scales_for_splitting: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute boolean masks indicating which Gaussians should be duplicated or split.

        Args:
            use_screen_space_scales_for_splitting: If set to true, use the tracked screen space scales to decide whether to split.
                Note that the model must have been configured to track these scales by
                setting ``GaussianSplat3d.accumulate_max_2d_radii = True``.

        Returns:
            duplication_mask (torch.Tensor): A boolean mask indicating which Gaussians should be duplicated.
            split_mask (torch.Tensor): A boolean mask indicating which Gaussians should be split.
        """
        # We use the average norm of the gradients of the projected Gaussians with respect to the
        # loss (accumulated since the last refinement step) to decide which Gaussians to duplicate or split.

        # model.accumulated_gradient_step_counts is the number of times a Gaussian has been projected
        # to an image (i.e. included in the loss gradient computation)
        # model.accumulated_mean_2d_gradient_norms is the sum of norms of the gradients of the
        # projected Gaussians (dL/dμ2D) since the last refinement step.
        count = self._model.accumulated_gradient_step_counts.clamp_min(1)
        if self._num_grad_accumulation_steps > 1:
            # Multiply the 2D gradient count by the number of times we've called backward since the last zero_grad()
            # to get the correct average if we're calling backward multiple times per iteration.
            count *= self._num_grad_accumulation_steps
        avg_norm_of_projected_mean_gradients = self._model.accumulated_mean_2d_gradient_norms / count

        # If the average norm of 2D projected gradients is high, that Gaussian is likely introducing
        # a lot of error into the reconstruction, and is a candidate for duplication or splitting.
        # We use the configured threshold to determine what "high" means.
        # If the 3D scale is small, we duplicate the Gaussian to allow it to specialize.
        # If the 3D scale is large, we split the Gaussian to allow it to specialize.
        # Duplication and splitting are mutually exclusive in the current logic:
        # a Gaussian is either duplicated (if small) or split (if large), but not both.
        is_grad_high = avg_norm_of_projected_mean_gradients > self._compute_insertion_grad_2d_threshold(
            avg_norm_of_projected_mean_gradients
        )
        is_small = self._model.log_scales.max(dim=-1).values <= np.log(
            self._config.insertion_scale_3d_threshold * self._spatial_scale
        )
        is_duplicated = is_grad_high & is_small

        # If the Gaussian is high error and its 3d spatial size is large, split the Gaussian
        is_large = ~is_small
        is_split = is_grad_high & is_large
        # Additionally, if a Gaussian's maximum projected size between refinement steps is too large, split it
        # This is only done if use_screen_space_scales_for_splitting is True
        if use_screen_space_scales_for_splitting:
            if not self._model.accumulate_max_2d_radii:
                raise ValueError(
                    "use_screen_space_scales_for_splitting is set to True but the model is not configured to "
                    + "track screen space scales. Set model.accumulate_max_2d_radii = True."
                )
            is_split.logical_or_(self._model.accumulated_max_2d_radii > self._config.insertion_scale_2d_threshold)

        return is_duplicated, is_split

    @torch.no_grad()
    def _compute_deletion_mask(
        self, use_scales_for_deletion: bool, use_screen_space_scales_for_deletion: bool
    ) -> torch.Tensor:
        """
        Compute a boolean mask indicating which Gaussians should be deleted.

        .. note:: If you pass in ``use_screen_space_scales_for_deletion``, the model must have been configured to
                  track these scales by setting ``model.accumulate_max_2d_radii = True``.

        Args:
            use_scales_for_deletion: If set to true, use a threshold on the 3D scales to delete Gaussians that are too large.
            use_screen_space_scales_for_deletion: If set to true, use a threshold on the maximum 2D projected scale
                between refinements to delete Gaussians that are too large.

        Returns:
            deletion_mask (torch.Tensor): A boolean mask indicating which Gaussians should be deleted.
        """
        # Delete a Gaussians if its opacity is below the threshold (meaning it doesn't contribute much to rendered images)
        is_deleted = self._model.logit_opacities < logit(self._config.deletion_opacity_threshold)

        # If you specify it, we will also delete Gaussians that are too large since they are likely not contributing
        # meaningfully to the reconstruction.
        if use_scales_for_deletion:
            is_too_big = self._model.log_scales.max(dim=-1).values > np.log(
                self._config.deletion_scale_3d_threshold * self._spatial_scale
            )
            is_deleted.logical_or_(is_too_big)

            # We can also use the tracked screen space scales to delete Gaussians that
            # project to a very large size in screen space between refinement steps.
            # This is only done if use_screen_space_scales_for_deletion is True
            if use_screen_space_scales_for_deletion:
                # Here we track the maximum size a Gaussian has projected to in screen space between refinement steps
                # if it's too big, we delete it
                has_projected_too_big = self._model.accumulated_max_2d_radii > self._config.deletion_scale_2d_threshold
                is_deleted.logical_or_(has_projected_too_big)
        return is_deleted

    @torch.no_grad()
    def _update_optimizer_params_and_state(
        self,
        optimizer_fn: Callable[[torch.Tensor], torch.Tensor],
        parameter_names: set[str] | None = None,
        reset_adam_step_counts: bool = False,
    ):
        """
        After changing the tensors in the model (e.g. after refinement or resetting opacities),
        we need to update the optimizer params to point to the new tensors, and fix the adam moments
        accordingly.

        If reset_adam_step_counts is True, we will also reset the Adam step counts to zero.
        This method copies the model's tensors into the optimizer's param groups so they continue to be optimized.
        It also applies the Adam moments for each parameter being updated 'exp_avg' and 'exp_avg_sq'.

        Args:
            optimizer_fn (Callable[[torch.Tensor], torch.Tensor]): A function to apply to each Adam moment Tensor for
                each parameter. Accepts the old moment Tensor and returns the new moment Tensor.
            parameter_names (set[str] | None): If provided, only update the parameter groups with these names.
                If ``None``, update all parameter groups.
            reset_adam_step_counts (bool): If ``True``, reset the Adam step counts to zero for all parameters being updated.
        """
        for i, param_group in enumerate(self._optimizer.param_groups):
            parameter_name = param_group["name"]
            if parameter_names is not None and parameter_name not in parameter_names:
                continue
            assert len(param_group["params"]) == 1, "Expected one parameter tensor per param group"
            old_parameter = param_group["params"][0]
            optimizer_state = self._optimizer.state[old_parameter]
            del self._optimizer.state[old_parameter]
            for key, value in optimizer_state.items():
                if key != "step":
                    optimizer_state[key] = optimizer_fn(value)
                elif reset_adam_step_counts:
                    optimizer_state[key].zero_()
            new_parameter = getattr(self._model, parameter_name)
            new_parameter.requires_grad = True
            self._optimizer.state[new_parameter] = optimizer_state
            self._optimizer.param_groups[i]["params"] = [new_parameter]

        if self._model.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _compute_revised_opacities(self, indices: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute opacity values for new Gaussians being inserted using the revised formulation from
        the paper "Revising Densification in Gaussian Splatting" (https://arxiv.org/abs/2404.06109).
        This removes a bias which weighs newly split Gaussians contribution to the image more heavily than
        older Gaussians.

        Args:
            indices (torch.Tensor): A 1D tensor of indices indicating which Gaussians to compute revised opacities for.
            eps (float): A small value to clamp the opacities to avoid numerical issues when computing the logit.

        Returns:
            revised_opacities (torch.Tensor): A tensor of revised logit opacities of shape ``(len(indices),)``.
        """
        # Update opacity values for the new Gaussians using the revised formulation from
        # the paper "Revising Densification in Gaussian Splatting" (https://arxiv.org/abs/2404.06109).
        # This removes a bias which weighs newly split Gaussians contribution to the image more heavily than
        # older Gaussians.
        safe_opacities = self._model.opacities[indices].clamp(min=eps, max=1.0 - eps)
        return torch.logit(1.0 - torch.sqrt(1.0 - safe_opacities))

    @torch.no_grad()
    def _compute_duplicated_gaussians(self, duplication_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute the new Gaussians to add by duplicating the Gaussians at the given indices.

        Returns a dictionary containing the new Gaussians to add with keys:

        * ``"means"``: The means of the new Gaussians of shape ``[(D-1)*M, 3]`` where ``D`` is the duplication
        factor and ``M`` is the number of duplicated Gaussians.
        * ``"quats"``: The quaternions of the new Gaussians of shape ``[(D-1)*M, 4]``.
        * ``"log_scales"``: The log scales of the new Gaussians of shape ``[(D-1)*M, 3]``.
        * ``"logit_opacities"``: The logit opacities of the new Gaussians of shape ``[(D-1)*M]``.
        * ``"sh0"``: The SH0 coefficients of the new Gaussians of shape ``[(D-1)*M, 1, 3]``.
        * ``"shN"``: The SHN coefficients of the new Gaussians of shape ``[(D-1)*M, K-1, 3]``.

        Args:
            duplication_indices (torch.Tensor): A 1D tensor of indices indicating which Gaussians to duplicate.

        Returns:
            duplicated_gaussian_parameters (dict[str, torch.Tensor]): A dictionary containing the new Gaussians to add.

        """
        duplication_factor = self._config.insertion_duplication_factor
        if duplication_factor < 2:
            raise ValueError("duplication_factor must be >= 2")

        if duplication_indices.numel() == 0:
            return {
                "means": torch.empty((0, 3), device=self._model.device),
                "quats": torch.empty((0, 4), device=self._model.device),
                "log_scales": torch.empty((0, 3), device=self._model.device),
                "logit_opacities": torch.empty((0,), device=self._model.device),
                "sh0": torch.empty((0, 1, 3), device=self._model.device),
                "shN": torch.empty((0, self._model.shN.shape[1], 3), device=self._model.device),
            }

        num_new_gaussians = duplication_factor - 1  # We already have one copy of each Gaussian in the model

        means_to_add = self._model.means[duplication_indices].repeat(num_new_gaussians, 1)  # [(D-1)*M, 3]
        log_scales_to_add = self._model.log_scales[duplication_indices].repeat(num_new_gaussians, 1)  # [(D-1)*M, 3]
        quats_to_add = self._model.quats[duplication_indices].repeat(num_new_gaussians, 1)  # [(D-1)*M, 4]
        sh0_to_add = self._model.sh0[duplication_indices].repeat(num_new_gaussians, 1, 1)  # [(D-1)*M, 1, 3]
        shN_to_add = self._model.shN[duplication_indices].repeat(num_new_gaussians, 1, 1)  # [(D-1)*M, K-1, 3]

        if self._config.opacity_updates_use_revised_formulation:
            logit_opacities_to_add = self._compute_revised_opacities(duplication_indices)  # [M,]
            logit_opacities_to_add = logit_opacities_to_add.repeat(num_new_gaussians)  # [(D-1)*M,]
        else:
            logit_opacities_to_add = self._model.logit_opacities[duplication_indices].repeat(
                num_new_gaussians
            )  # [(D-1)*M,]

        return {
            "means": means_to_add,
            "quats": quats_to_add,
            "log_scales": log_scales_to_add,
            "logit_opacities": logit_opacities_to_add,
            "sh0": sh0_to_add,
            "shN": shN_to_add,
        }

    @torch.no_grad()
    def _compute_split_gaussians(self, split_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute the new Gaussians to add by splitting the Gaussians at the given indices.

        Returns a dictionary containing the new Gaussians to add with keys:

        - ``"means"``: The means of the new Gaussians of shape ``[S*M, 3]`` where ``S`` is the split factor
        and ``M`` is the number of split Gaussians.
        - ``"quats"``: The quaternions of the new Gaussians of shape ``[S*M, 4]``.
        - ``"log_scales"``: The log scales of the new Gaussians of shape ``[S*M, 3]``.
        - ``"logit_opacities"``: The logit opacities of the new Gaussians of shape ``[S*M]``.
        - ``"sh0"``: The SH0 coefficients of the new Gaussians of shape ``[S*M, 1, 3]``.
        - ``"shN"``: The SHN coefficients of the new Gaussians of shape ``[S*M, K-1, 3]``.

        Args:
            split_indices (torch.Tensor): A 1D tensor of indices indicating which Gaussians to split.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the new Gaussians to add.

        """
        split_factor = self._config.insertion_split_factor
        if split_indices.numel() == 0:
            return {
                "means": torch.empty((0, 3), device=self._model.device),
                "quats": torch.empty((0, 4), device=self._model.device),
                "log_scales": torch.empty((0, 3), device=self._model.device),
                "logit_opacities": torch.empty((0,), device=self._model.device),
                "sh0": torch.empty((0, 1, 3), device=self._model.device),
                "shN": torch.empty((0, self._model.shN.shape[1], 3), device=self._model.device),
            }
        if split_factor < 2:
            raise ValueError("split_factor must be >= 2")

        split_scales = self._model.scales[split_indices]  # [M, 3]
        split_quats = nnf.normalize(self._model.quats[split_indices], dim=-1)  # [M, 4]
        rotmats = self._unit_quats_to_rotation_matrices(split_quats)  # [M, 3, 3]
        split_mean_offsets = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            split_scales,
            torch.randn(split_factor, split_indices.shape[0], 3, device=self._model.device),
        )  # [S, N, 3]

        means_to_add = (self._model.means[split_indices] + split_mean_offsets).reshape(-1, 3)  # [S*M, 3]
        quats_to_add = self._model.quats[split_indices].repeat(split_factor, 1)  # [S*M, 4]
        sh0_to_add = self._model.sh0[split_indices].repeat(split_factor, 1, 1)  # [S*M, 1, 3]
        shN_to_add = self._model.shN[split_indices].repeat(split_factor, 1, 1)  # [S*M, K-1, 3]

        # Scale down each split Gaussian's scale by a factor of 0.8 * split_factor to keep the
        # overall volume of the split Gaussians roughly the same as the original Gaussian.
        # The 0.8 factor comes from the original INRIA implementation, and was determined empirically.
        scales_denominator_factor = 0.8 * split_factor
        log_scales_to_add = torch.log(split_scales / scales_denominator_factor).repeat(split_factor, 1)  # [S*M, 3]

        if self._config.opacity_updates_use_revised_formulation:
            logit_opacities_to_add = self._compute_revised_opacities(split_indices)  # [M,]
            logit_opacities_to_add = logit_opacities_to_add.repeat(split_factor)  # [S*M]
        else:
            logit_opacities_to_add = self._model.logit_opacities[split_indices].repeat(split_factor)  # [S*M]

        return {
            "means": means_to_add,
            "quats": quats_to_add,
            "log_scales": log_scales_to_add,
            "logit_opacities": logit_opacities_to_add,
            "sh0": sh0_to_add,
            "shN": shN_to_add,
        }

    @classmethod
    def _compute_spatial_scale(cls, sfm_scene: SfmScene, mode: SpatialScaleMode) -> float:
        """
        Calculate a spatial scale for the scene based on the given mode.

        Args:
            sfm_scene (SfmScene): The scene to calculate the scale for.
            mode (SpatialScaleMode): The mode to use for calculating the scale.
        Returns:
            spatial_scale (float): The calculated spatial scale.
        """

        if not sfm_scene.has_visible_point_indices and mode in (
            SpatialScaleMode.MEDIAN_CAMERA_DEPTH,
            SpatialScaleMode.MAX_CAMERA_DEPTH,
        ):
            raise ValueError(f"Cannot use {mode} when SfmScene.has_visible_point_indices is False")

        if mode == SpatialScaleMode.ABSOLUTE_UNITS:
            return 1.0
        elif mode == SpatialScaleMode.MEDIAN_CAMERA_DEPTH or mode == SpatialScaleMode.MAX_CAMERA_DEPTH:
            # Compute the median distance from the SfmPoints seen by each camera to the position of the camera
            median_depth_per_camera = []
            for image_meta in sfm_scene.images:
                assert (
                    image_meta.point_indices is not None
                ), "SfmScene.has_visible_point_indices is True but image has no point indices"

                # Don't use cameras that don't see any points in the estimate
                if len(image_meta.point_indices) == 0:
                    continue
                points = sfm_scene.points[image_meta.point_indices]
                dist_to_points = np.linalg.norm(points - image_meta.origin, axis=1)
                median_dist = np.median(dist_to_points)
                median_depth_per_camera.append(median_dist)
            if mode == SpatialScaleMode.MEDIAN_CAMERA_DEPTH:
                return float(np.median(median_depth_per_camera))
            elif mode == SpatialScaleMode.MAX_CAMERA_DEPTH:
                return float(np.max(median_depth_per_camera))
        elif mode == SpatialScaleMode.MAX_CAMERA_TO_CENTROID:
            origins = np.stack([cam.origin for cam in sfm_scene.images], axis=0)
            centroid = np.mean(origins, axis=0)
            dists = np.linalg.norm(origins - centroid, axis=1)
            return float(np.max(dists))
        elif mode == SpatialScaleMode.SCENE_DIAGONAL_PERCENTILE:
            percentiles = np.percentile(sfm_scene.points, [5, 95], axis=1)
            bbox_min, bbox_max = percentiles[:, 0], percentiles[:, 1]
            scene_diag = np.linalg.norm(bbox_max - bbox_min)
            return float(scene_diag)
        else:
            raise ValueError(f"Unknown spatial scale mode: {mode}")

    @staticmethod
    def _unit_quats_to_rotation_matrices(quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor of unit quaternions (encoding 3d rotations) to a tensor of rotation matrices.

        Args:
            quaternions (torch.Tensor): A Tensor of unit quaternions in wxyz convention with shape  ``[*, 4]``

        Returns:
            rotation_matrices (torch.Tensor): A tensor of rotation matrices of shape ``[*, 3, 3]``
        """
        assert quaternions.shape[-1] == 4, quaternions.shape
        w, x, y, z = torch.unbind(quaternions, dim=-1)
        mat = torch.stack(
            [
                1 - 2 * (y**2 + z**2),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x**2 + z**2),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x**2 + y**2),
            ],
            dim=-1,
        )
        return mat.reshape(quaternions.shape[:-1] + (3, 3))
