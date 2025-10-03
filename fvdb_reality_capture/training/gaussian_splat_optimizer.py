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


class InsertionGrad2dThresholdMode(str, Enum):
    """
    The `GaussianSplatOptimizer` uses a threshold on the accumulated norm of 2D mean gradients to use during refinement.

    There are several modes for computing this threshold, specified by the config:
    - CONSTANT: Always use the fixed threshold specified by self._config.insertion_grad_2d_threshold.
    - PERCENTILE_FIRST_ITERATION: During the first refinement step, set the threshold to the given percentile of the gradients.
        For all subsequent refinement steps, use that fixed threshold.
    - PERCENTILE_EVERY_ITERATION: During every refinement step, set the threshold to the given percentile of the gradients.

    These modes let you adapt the refinement behavior to the statistics of the gradients during training.
    Generally CONSTANT with a default value (0.0002) will produce okay results, but may not be optimal for all types
    of captures. Using PERCENTILE_FIRST_ITERATION will have similar behavior to CONSTANT but will adapt to the scale of the gradients
    which can be more robust across different capture types.
    For highly detailed, scenes, PERCENTILE_EVERY_ITERATION may be useful to adaptively insert more Gaussians as the model learns more detail.
    This generally produces many more Gaussians and more detailed results at the cost of more memory and compute.
    """

    CONSTANT = "constant"
    PERCENTILE_FIRST_ITERATION = "percentile_first_iteration"
    PERCENTILE_EVERY_ITERATION = "percentile_every_iteration"


@dataclass
class GaussianSplatOptimizerConfig:
    """
    Parameters for configuring the `GaussianSplatOptimizer`.
    """

    # The maximum number of Gaussians to allow in the model. If -1, no limit.
    max_gaussians: int = -1

    # Whether to use a fixed threshold for insertion_grad_2d_threshold (constant),
    # a value computed as a percentile of the grad_2d distribution on the first iteration
    # or a percentile value computed at each refinement step
    insertion_grad_2d_threshold_mode: InsertionGrad2dThresholdMode = InsertionGrad2dThresholdMode.CONSTANT

    # If a Gaussian's opacity drops below this value, delete it
    deletion_opacity_threshold: float = 0.005

    # If a Gaussian's 3d scale drops below this value (units are specified by scale_3d_threshold_units) then delete it.
    deletion_scale_3d_threshold: float = 0.1

    # If a projected Gaussian's 2d scale drops below this value (units are specified by scale_2d_threshold_units) then delete it.
    # Note this parameter is only used if you call refine with use_screen_space_scales=True
    deletion_scale_2d_threshold: float = 0.15

    # Threshold value on the accumulated norm of projected mean gradients between refinement scale to
    # determine whether a Gaussian has high error and is a candidate for duplication or splitting.
    # This value must be positive if using CONSTANT mode, or in the range (0.0, 1.0) if using
    # PERCENTILE_FIRST_ITERATION or PERCENTILE_EVERY_ITERATION modes.
    insertion_grad_2d_threshold: float = 0.0002 if insertion_grad_2d_threshold_mode == "constant" else 0.9

    # Duplicate high-error (determined by insertion_grad_2d_threshold) Gaussians whose 3d scale is below this value.
    # These Gaussians are too small to capture the detail in the region they cover, so we duplicate them to
    # allow them to specialize.
    insertion_scale_3d_threshold: float = 0.01

    # Split high-error (determined by insertion_grad_2d_threshold) Gaussians whose maximum projected
    # size exceeds this value. These Gaussians are too large to capture the detail in the region they cover,
    # so we split them to allow them to specialize.
    # Note this parameter is only used if you call refine with use_screen_space_scales=True
    insertion_scale_2d_threshold: float = 0.05

    # When splitting Gaussinas, update the opacities of the new Gaussians using the revised formulation from
    # "Revising Densification in Gaussian Splatting" (https://arxiv.org/abs/2404.06109).
    # This removes a bias which weighs newly split Gaussians contribution to the image more heavily than
    # older Gaussians.
    opacity_updates_use_revised_formulation: bool = False

    # When splitting Gaussians, during insertion, how many new Gaussians to create per split Gaussian.
    # _e.g._ if this value is 2, each split Gaussian becomes 2 smaller Gaussians.
    # This value must be >= 2.
    insertion_split_factor: int = 2

    # When duplicating Gaussians, during insertion, how many new Gaussians to create per duplicated Gaussian.
    # _e.g._ if this value is 3, each duplicated Gaussian becomes 3 copies of itself
    # This value must be >= 2.
    insertion_duplication_factor: int = 2

    # Learning rate for the means
    means_lr: float = 1.6e-4
    # Learning rate for the log scales
    log_scales_lr: float = 5e-3
    # Learning rate for the quaternions
    quats_lr: float = 1e-3
    # Learning rate for the logit opacities
    logit_opacities_lr: float = 5e-2
    # Learning rate for the spherical harmonics of order 0
    sh0_lr: float = 2.5e-3
    # Learning rate for the spherical harmonics of order N (N > 0)
    shN_lr: float = 2.5e-3 / 20


class GaussianSplatOptimizer:
    """
    Optimizer for reconstructing a scene using Gaussian Splat radiance fields over a collection of posed images.

    The optimizer uses an Adam optimizer to optimize the parameters of a `fvdb.GaussianSplat3d` model, and
    provides utilities to refine the model by inserting and deleting Gaussians based on their contribution to the optimization.
    The tools here mostly follow the algorithm in the original Gaussian Splatting paper (https://arxiv.org/abs/2308.04079).
    """

    __PRIVATE__ = object()

    def __init__(
        self,
        model: GaussianSplat3d,
        optimizer: torch.optim.Adam,
        means_lr_decay_exponent: float,
        config: GaussianSplatOptimizerConfig,
        _private: Any = None,
    ):
        """
        Create a new `GaussianSplatOptimizer` instance froom a model, optimizers and a config.

        Note: You should not call this constructor directly. Instead use `from_model_and_config()` or `from_state_dict()`.

        Args:
            model (GaussianSplat3d): The `GaussianSplat3d` model to optimize.
            optimizers (dict[str, torch.optim.Adam]): A dictionary of optimizers for each parameter group in the model.
            means_lr_decay_exponent (float): The exponent used for decaying the means learning rate.
            config (GaussianSplatOptimizerConfig): Configuration options for the optimizer.
            _private (Any): A private object to prevent direct instantiation. Must be `GaussianSplatOptimizer.__PRIVATE__`.
        """
        if _private is not self.__PRIVATE__:
            raise RuntimeError(
                "GaussianSplatOptimizer must be created using from_model_and_config() or from_state_dict()"
            )
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._model = model
        self._model.accumulate_mean_2d_gradients = True  # Make sure we track the 2D gradients for refinement
        self._config = config
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
        self._means_lr_decay_exponent = means_lr_decay_exponent

    @classmethod
    def from_model_and_config(
        cls,
        model: GaussianSplat3d,
        config: GaussianSplatOptimizerConfig = GaussianSplatOptimizerConfig(),
        means_lr_decay_exponent: float = 1.0,
        batch_size: int = 1,
    ) -> "GaussianSplatOptimizer":
        """
        Create a new `GaussianSplatOptimizer` instance from a model and config.

        Args:
            model (GaussianSplat3d): The `GaussianSplat3d` model to optimize.
            config (GaussianSplatOptimizerConfig): Configuration options for the optimizer.
            means_lr_decay_exponent (float): The exponent used for decaying the means learning rate.
            batch_size (int): The batch size used for training. This is used to scale the learning rates.

        Returns:
            GaussianSplatOptimizer: A new `GaussianSplatOptimizer` instance.
        """

        optimizer = GaussianSplatOptimizer._make_optimizer(model, batch_size, config)

        return cls(
            model=model,
            optimizer=optimizer,
            means_lr_decay_exponent=means_lr_decay_exponent,
            config=config,
            _private=cls.__PRIVATE__,
        )

    @classmethod
    def from_state_dict(cls, model: GaussianSplat3d, state_dict: dict[str, Any]) -> "GaussianSplatOptimizer":
        """
        Create a new `GaussianSplatOptimizer` instance from a model and a state dict.

        Args:
            model (GaussianSplat3d): The `GaussianSplat3d` model to optimize.
            state_dict (dict[str, Any]): A state dict previously obtained from `state_dict()`.

        Returns:
            GaussianSplatOptimizer: A new `GaussianSplatOptimizer` instance.
        """
        if "version" not in state_dict:
            raise ValueError("State dict is missing version information")
        if state_dict["version"] not in (3,):
            raise ValueError(f"Unsupported version: {state_dict['version']}")

        config = GaussianSplatOptimizerConfig(**state_dict["config"])
        optimizer = GaussianSplatOptimizer._make_optimizer(model, batch_size=1, config=config)
        optimizer.load_state_dict(state_dict["optimizer"])

        optimizer = cls(
            model=model,
            optimizer=optimizer,
            means_lr_decay_exponent=state_dict["means_lr_decay_exponent"],
            config=config,
            _private=cls.__PRIVATE__,
        )
        optimizer._insertion_grad_2d_abs_threshold = state_dict["insertion_grad_2d_abs_threshold"]

        return optimizer

    def step(self):
        """
        Step the optimizers (updating the model's parameters) and decay the learning rate of the means.
        """
        self._optimizer.step()
        # Decay the means learning rate
        for param_group in self._optimizer.param_groups:
            if param_group["name"] == "means":
                param_group["lr"] *= self._means_lr_decay_exponent
                return

    def zero_grad(self, set_to_none: bool = False):
        """
        Zero the gradients of all tensors being optimized.

        Args:
            set_to_none (bool): If True, set the gradients to None instead of zeroing them. This can be more memory efficient.
        """
        self._num_grad_accumulation_steps = 0
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """
        Return a serializable state dict for the optimizer.

        Returns:
            dict[str, Any]: A state dict containing the state of the optimizer.
        """
        return {
            "optimizer": self._optimizer.state_dict(),
            "means_lr_decay_exponent": self._means_lr_decay_exponent,
            "insertion_grad_2d_abs_threshold": self._insertion_grad_2d_abs_threshold,
            "num_grad_accumulation_steps": self._num_grad_accumulation_steps,
            "config": vars(self._config),
            "version": 3,
        }

    @torch.no_grad()
    def filter_gaussians(self, indices_or_mask: torch.Tensor):
        """
        Filter the Gaussians in the model to only those specified by the given indices or mask
        and update the optimizer state accordingly. This can be used to delete, shuffle, or expand
        the Gaussians during optimization.

        Args:
            indices_or_mask (torch.Tensor): A 1D tensor of indices or a boolean mask indicating which Gaussians to keep.
                It must have shape (num_gaussians,).
        """
        self._model = self._model[indices_or_mask]
        self._update_optimizer_for_model(lambda x: x[indices_or_mask])

    @torch.no_grad()
    def reset_opacities(self):
        """
        Clamp the logit_opacities of all Gaussians to a small value above the deletion threshold.
        This is useful to call periodically during optimization to prevent Gaussians from becoming
        completely occluded by denser Gaussians, and thus unable to be optimized.
        """
        # Clamp all opacities to be less than or equal to twice the deletion threshold
        value = self._config.deletion_opacity_threshold * 2.0
        self._model.logit_opacities.clamp_(max=torch.logit(torch.tensor(value)).item())
        self._update_optimizer_for_model(lambda x: x.zero_(), parameter_names={"logit_opacities"})

    @torch.no_grad()
    def refine(self, use_scales_for_deletion, use_screen_space_scales) -> tuple[int, int, int]:
        """
        Perform a step of refinement by inserting Gaussians where more detail is needed and deleting Gaussians that are not contributing to the optimization.
        Refinement happens via three mechanisms:

        **Duplication**: Make two exact copies of a Gaussian.
          - We duplicate a Gaussian if it's 3D size is below some threshold and the gradient of its projected means over time is high on
            average. Intuitively, this means the Gaussian is not taking up a lot of space in the scene, but consistently wants to change positions
            when viewed from different images. Likely this Gaussian is stuck trying to represent too much of the scene and should
            be split into multiple copies.

        **Splitting**: Split a Gaussian into two smaller ones.
          - We split a Gaussian when its 3D size exceeds a threshold value and the gradient of its projected mean over time is high on average.
            In this case, a Gaussian is likely too large for the amount of detail it represents and should be split to capture detail in the image.

        **Deletion**: Removing a Gaussian from the scene.
          - We delete a Gaussian if its opacity falls below a threshold since it is not contributing much to rendered images.


        Args:
            use_scales_for_deletion: If set to True, use the 3D scales to decide whether to delete Gaussians that are too large.
            use_screen_space_scales: If set to true, threshold the maximum projected size of Gaussians between refinement steps
                                     to decide whether to split or delete Gaussians that are too large.
                                     Note that the model must have been configured to track these scales by setting
                                     `GaussianSplat3d.accumulate_max_2d_radii = True`.

        Returns:
            num_duplicated (int): The number of Gaussians that were duplicated.
            num_split (int): The number of Gaussians that were split.
            num_deleted (int): The number of Gaussians that were deleted.
        """

        is_duplicated, is_split = self._compute_insertion_masks(use_screen_space_scales)
        duplication_indices = torch.where(is_duplicated)[0]
        num_duplicated = len(duplication_indices)

        split_indices = torch.where(is_split)[0]
        num_split = len(split_indices)

        is_deleted = self._compute_deletion_mask(use_scales_for_deletion, use_screen_space_scales)
        num_deleted = int(is_deleted.sum().item())

        # Gaussians which are not split and not deleted are kept as-is
        kept_indices = torch.where(~(is_split | is_deleted))[0]

        # Get the new Gaussians to add from splitting and duplication
        duplicated_gaussians = self._compute_duplicated_gaussians(duplication_indices)
        split_gaussians = self._compute_split_gaussians(split_indices)

        # We no longer need the accumulated gradient state since we've used it to compute masks for refinement
        # Reset it so we can start accumulating for the next refinement step
        self._model.reset_accumulated_gradient_state()
        self._model.set_state(
            means=torch.cat(
                [self._model.means[kept_indices], duplicated_gaussians["means"], split_gaussians["means"]], dim=0
            ),
            quats=torch.cat(
                [self._model.quats[kept_indices], duplicated_gaussians["quats"], split_gaussians["quats"]], dim=0
            ),
            log_scales=torch.cat(
                [
                    self._model.log_scales[kept_indices],
                    duplicated_gaussians["log_scales"],
                    split_gaussians["log_scales"],
                ],
                dim=0,
            ),
            logit_opacities=torch.cat(
                [
                    self._model.logit_opacities[kept_indices],
                    duplicated_gaussians["logit_opacities"],
                    split_gaussians["logit_opacities"],
                ],
                dim=0,
            ),
            sh0=torch.cat([self._model.sh0[kept_indices], duplicated_gaussians["sh0"], split_gaussians["sh0"]], dim=0),
            shN=torch.cat([self._model.shN[kept_indices], duplicated_gaussians["shN"], split_gaussians["shN"]], dim=0),
        )

        def update_state_function(x: torch.Tensor):
            total_gaussians = kept_indices.shape[0] + num_duplicated + num_split * 2
            ret = torch.zeros([total_gaussians] + list(x.shape[1:]), dtype=x.dtype, device=x.device)
            ret[: kept_indices.shape[0]] = x[kept_indices]
            return ret

        self._update_optimizer_for_model(update_state_function)

        return num_duplicated, num_split, num_deleted

    @staticmethod
    def _make_optimizer(model, batch_size, config):
        """
        Make an Adam optimizer for the given model and config.
        This is just a helper function used by the constructors since this logic is shared and verbose.

        Args:
            model (GaussianSplat3d): The model to optimize.
            batch_size (int): The batch size used for training. This is used to scale the learning rates.
            config (GaussianSplatOptimizerConfig): The configuration for the optimizer.

        Returns:
            torch.optim.Adam: An Adam optimizer for the model.
        """
        # Scale learning rate based on batch size, reference:
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # Note that this would not make the training exactly equivalent to the original INRIA
        # Gaussian splat implementation.
        # See https://arxiv.org/pdf/2402.18824v1 for more details.
        lr_batch_rescale = math.sqrt(float(batch_size))
        return torch.optim.Adam(
            [
                {"params": model.means, "lr": config.means_lr * lr_batch_rescale, "name": "means"},
                {
                    "params": model.log_scales,
                    "lr": config.log_scales_lr * lr_batch_rescale,
                    "name": "log_scales",
                },
                {"params": model.quats, "lr": config.quats_lr * lr_batch_rescale, "name": "quats"},
                {
                    "params": model.logit_opacities,
                    "lr": config.logit_opacities_lr * lr_batch_rescale,
                    "name": "logit_opacities",
                },
                {"params": model.sh0, "lr": config.sh0_lr * lr_batch_rescale, "name": "sh0"},
                {"params": model.shN, "lr": config.shN_lr * lr_batch_rescale, "name": "shN"},
            ],
            eps=1e-15 / lr_batch_rescale,
            betas=(1.0 - batch_size * (1.0 - 0.9), 1.0 - batch_size * (1.0 - 0.999)),
        )

    def _compute_insertion_grad_2d_threshold(self, accumulated_mean_2d_gradients: torch.Tensor) -> float:
        """
        Compute the threshold on the accumulated norm of 2D mean gradients to use during refinement.

        There are several modes for computing this threshold, specified by the config:
        - CONSTANT: Always use the fixed threshold specified by self._config.insertion_grad_2d_threshold.
        - PERCENTILE_FIRST_ITERATION: During the first refinement step, set the threshold to the given percentile of the gradients.
            For all subsequent refinement steps, use that fixed threshold.
        - PERCENTILE_EVERY_ITERATION: During every refinement step, set the threshold to the given percentile of the gradients.

        These modes let you adapt the refinement behavior to the statistics of the gradients during training.
        Generally CONSTANT with a default value (0.0002) will produce okay results, but may not be optimal for all types
        of captures. Using PERCENTILE_FIRST_ITERATION will have similar behavior to CONSTANT but will adapt to the scale of the gradients
        which can be more robust across different capture types.
        For highly detailed, scenes, PERCENTILE_EVERY_ITERATION may be useful to adaptively insert more Gaussians as the model learns more detail.
        This generally produces many more Gaussians and more detailed results at the cost of more memory and compute.

        Args:
            accumulated_mean_2d_gradients (torch.Tensor): The average norm of the projected mean gradients of shape (num_gaussians,).
                This is typically obtained from `model.accumulated_mean_2d_gradient_norms / model.accumulated_gradient_step_counts`
                where `model.accumulated_gradient_step_counts` is the number of optimization steps since the last refinement.

        Returns:
            float: The threshold value to use for deciding whether to insert Gaussians during refinement.
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

    def _compute_insertion_masks(
        self, use_screen_space_scales_for_splitting: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute boolean masks indicating which Gaussians should be duplicated or split.

        Args:
            use_screen_space_scales_for_splitting: If set to true, use the tracked screen space scales to decide whether to split.
                                              Note that the model must have been configured to track these scales by setting
                                              `GaussianSplat3d.accumulate_max_2d_radii = True`.

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

        # If the average norm of 2D projected gradients is high, that Gaussians is likely introducing
        # a lot of error into the reconstruction, and is a candidate for duplication or splitting.
        # We use the configured threshold to determine what "high" means.
        # If the 3D scale is small, we duplicate the Gaussian to allow it to specialize.
        # If the 3D scale is large, we split the Gaussian to allow it to specialize.
        # Note that a Gaussian can be both duplicated and split if it meets both criteria.
        # In that case, we will create 3 new Gaussians (2 from duplication and 2 from splitting, with
        # one of the split Gaussians replacing the original).
        is_grad_high = avg_norm_of_projected_mean_gradients > self._compute_insertion_grad_2d_threshold(
            avg_norm_of_projected_mean_gradients
        )
        is_small = self._model.scales.max(dim=-1).values <= self._config.insertion_scale_3d_threshold
        is_duplicated = is_grad_high & is_small

        # If the Gaussian is high error and its 3d spatial size is large, split the Gaussian
        is_large = ~is_small
        is_split = is_grad_high & is_large
        # If a Gaussian is high error, and it projects to a large size in screen space, split it
        # This is only done if use_screen_space_scales_for_splitting is True
        if use_screen_space_scales_for_splitting:
            if not self._model.accumulate_max_2d_radii:
                raise ValueError(
                    "use_screen_space_scales_for_splitting is set to True but the model is not configured to "
                    + "track screen space scales. Set model.accumulate_max_2d_radii = True."
                )
            is_split |= self._model.accumulated_max_2d_radii > self._config.insertion_scale_2d_threshold

        return is_duplicated, is_split

    @torch.no_grad()
    def _compute_deletion_mask(
        self, use_scales_for_deletion: bool, use_screen_space_scales_for_deletion: bool
    ) -> torch.Tensor:
        """
        Compute a boolean mask indicating which Gaussians should be deleted.

        Args:
            use_scales_for_deletion: If set to true, use a threshold on the the 3D scales to delete Gaussians that are too large.
            use_screen_space_scales_for_deletion: If set to true, use a threshold on the the maximum 2D projected scale
                between refinements to delete Gaussians that are too large.
                Note: the model must have been configured to track these scales by setting
                `model.accumulate_max_2d_radii = True`.

        Returns:
            deletion_mask (torch.Tensor): A boolean mask indicating which Gaussians should be deleted.
        """
        # Delete a Gaussians if its opacity is below the threshold (meaning it doesn't contribute much to rendered images)
        is_deleted = self._model.opacities.flatten() < self._config.deletion_opacity_threshold

        # If you specify it, we will also delete Gaussians that are too large since they are likely not contributing
        # meaningfully to the reconstruction.
        if use_scales_for_deletion:
            is_too_big = self._model.scales.max(dim=-1).values > self._config.deletion_scale_3d_threshold
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
    def _update_optimizer_for_model(
        self,
        optimizer_fn: Callable[[torch.Tensor], torch.Tensor],
        parameter_names: set[str] | None = None,
    ):
        """
        After changing the tensors in the model (e.g. after refinement or resetting opacities),
        we need to update the optimizer state to point to the new tensors.

        This method copies the model's tensors into the optimizer's param groups so they continue to be optimized.
        It also applies the the Adam moments for each parameter being updated 'exp_avg' and 'exp_avg_sq'.

        Args:
            optimizer_fn (Callable[[torch.Tensor], torch.Tensor]): A function to apply to each Adam moment Tensor for each parameter.
                Accepts the old moment Tensor and returns the new moment Tensor.
            parameter_names (set[str] | None): If provided, only update the parameter groups with these names. If None, update all parameter groups.
        """
        for i, param_group in enumerate(self._optimizer.param_groups):
            parameter_name = param_group["name"]
            if parameter_names is not None and param_group["name"] not in parameter_names:
                continue
            assert len(param_group["params"]) == 1, "Expected one parameter tensor per param group"
            old_parameter = param_group["params"][0]
            optimizer_state = self._optimizer.state[old_parameter]
            del self._optimizer.state[old_parameter]
            for key, value in optimizer_state.items():
                if key != "step":
                    optimizer_state[key] = optimizer_fn(value)
            new_parameter = getattr(self._model, parameter_name)
            new_parameter.requires_grad = True
            self._optimizer.state[new_parameter] = optimizer_state
            self._optimizer.param_groups[i]["params"] = [new_parameter]

        if self._model.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _compute_duplicated_gaussians(self, duplication_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute the new Gaussians to add by duplicating the Gaussians at the given indices.

        Args:
            duplication_indices (torch.Tensor): A 1D tensor of indices indicating which Gaussians to duplicate.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the new Gaussians to add with keys (where D is the duplication factor and M is the number of duplicated Gaussians):
                - "means": The means of the new Gaussians of shape [(D-1)*M, 3] where D is the duplication factor and M is the number of duplicated Gaussians.
                - "quats": The quaternions of the new Gaussians of shape [(D-1)*M, 4].
                - "log_scales": The log scales of the new Gaussians of shape [(D-1)*M, 3].
                - "logit_opacities": The logit opacities of the new Gaussians of shape [(D-1)*M].
                - "sh0": The SH0 coefficients of the new Gaussians of shape [(D-1)*M, 1, 3].
                - "shN": The SHN coefficients of the new Gaussians of shape [(D-1)*M, K-1, 3].
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
            # Update opacity values for the new Gaussians using the revised formulation from
            # the paper "Revising Densification in Gaussian Splatting" (https://arxiv.org/abs/2404.06109).
            # This removes a bias which weighs newly split Gaussians contribution to the image more heavily than
            # older Gaussians.
            logit_opacities_to_add = torch.logit(1.0 - torch.sqrt(1.0 - self._model.opacities[duplication_indices]))
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

        Args:
            split_indices (torch.Tensor): A 1D tensor of indices indicating which Gaussians to split.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the new Gaussians to add with keys:
                - "means": The means of the new Gaussians of shape [S*M, 3] where S is the split factor and M is the number of split Gaussians.
                - "quats": The quaternions of the new Gaussians of shape [S*M, 4].
                - "log_scales": The log scales of the new Gaussians of shape [S*M, 3].
                - "logit_opacities": The logit opacities of the new Gaussians of shape [S*M].
                - "sh0": The SH0 coefficients of the new Gaussians of shape [S*M, 1, 3].
                - "shN": The SHN coefficients of the new Gaussians of shape [S*M, K-1, 3].
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
        log_scales_to_add = torch.log(split_scales / (0.8 * split_factor)).repeat(split_factor, 1)  # [S*M, 3]
        quats_to_add = self._model.quats[split_indices].repeat(split_factor, 1)  # [S*M, 4]
        sh0_to_add = self._model.sh0[split_indices].repeat(split_factor, 1, 1)  # [S*M, 1, 3]
        shN_to_add = self._model.shN[split_indices].repeat(split_factor, 1, 1)  # [S*M, K-1, 3]

        if self._config.opacity_updates_use_revised_formulation:
            # Update opacity values for the new Gaussians using the revised formulation from
            # the paper "Revising Densification in Gaussian Splatting" (https://arxiv.org/abs/2404.06109).
            # This removes a bias which weighs newly split Gaussians contribution to the image more heavily than
            # older Gaussians.
            logit_opacities_to_add = torch.logit(1.0 - torch.sqrt(1.0 - self._model.opacities[split_indices]))
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

    @staticmethod
    def _unit_quats_to_rotation_matrices(quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor of unit quaternions (encoding 3d rotations) to a tensor of rotation matrices.

        Args:
            quaternions (torch.Tensor): A Tensor of unit quaternions in wxyz convention with shape [*, 4]

        Returns:
            rotation_matrices (torch.Tensor): A tensor of rotation matrices of shape [*, 3, 3]
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
