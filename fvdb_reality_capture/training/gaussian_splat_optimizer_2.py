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

    # If a Gaussian's 3d scale drops below this value (units specfied by scale_3d_threshold_units) then delete it
    deletion_scale_3d_threshold: float = 0.1

    # If a projected Gaussian's 2d scale drops below this value (units specfied by scale_3d_threshold_units) then delete it
    deletion_scale_2d_threshold: float = 0.15

    # Duplicate or split Gaussians where the accumulated gradients of its 2d mean is above this value
    # and whose 3d and 2d scales exceed insertion_scale_3d_threshold and insertion_scale_2d_threshold
    insertion_grad_2d_threshold: float = 0.0002 if insertion_grad_2d_threshold_mode == "constant" else 0.9

    # Duplicate or split Gaussians whose 3d scale exceeds this value and whose
    # accumulated 2d gradient exceeds insertion_grad_2d_threshold
    insertion_scale_3d_threshold: float = 0.01

    # Duplicate or split Gaussians whose 2d scale exceeds this value and whose accumulated 2d gradient
    # exceeds insertion_grad_2d_threshold
    insertion_scale_2d_threshold: float = 0.05

    # When splitting Gaussinas, update the opacities of the new Gaussians using the revised formulation from
    # "Revising Densification in Gaussian Splatting" (https://arxiv.org/abs/2404.06109).
    # This removes a bias which weighs newly split Gaussians contribution to the image more heavily than
    # older Gaussians.
    opacity_updates_use_revised_formulation: bool = False

    # TODO: Document
    use_absolute_gradients: bool = False

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


class _OptimizerState(torch.optim.Adam):
    def __init__(self, model: GaussianSplat3d, batch_size, config: GaussianSplatOptimizerConfig):
        self._model = model
        self._model.accumulate_mean_2d_gradients = True  # Make sure we track the 2D gradients for refinement

        # Scale learning rate based on batch size, reference:
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # Note that this would not make the training exactly equivalent to the original INRIA
        # Gaussian splat implementation.
        # See https://arxiv.org/pdf/2402.18824v1 for more details.
        lr_batch_rescale = math.sqrt(float(batch_size))
        self._optimizer = torch.optim.Adam(
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

    @property
    def optimizer(self) -> torch.optim.Adam:
        return self._optimizer

    @property
    def model(self) -> GaussianSplat3d:
        return self._model

    def step(self, means_lr_decay: float):
        self._optimizer.step()
        if means_lr_decay != 1.0:
            # Decay the means learning rate
            for param_group in self._optimizer.param_groups:
                if param_group["name"] == "means":
                    param_group["lr"] *= means_lr_decay
                    return
            raise RuntimeError("Means parameter group not found in optimizer")

    def zero_grad(self):
        self._optimizer.zero_grad()

    @staticmethod
    def _normalized_quat_to_rotmat(quat_: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized quaternion to rotation matrix.

        Args:
            quat: Normalized quaternion in wxyz convension. (..., 4)

        Returns:
            Rotation matrix (..., 3, 3)
        """
        assert quat_.shape[-1] == 4, quat_.shape
        w, x, y, z = torch.unbind(quat_, dim=-1)
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
        return mat.reshape(quat_.shape[:-1] + (3, 3))

    @torch.no_grad
    def _update_optimizer_state(
        self, optimizer_state_update_function: Callable[[torch.Tensor], torch.Tensor], parameter_name: str | None = None
    ):
        """
        Update the state tracked by the adam optimizer (_i.e._ the running averages of the gradients, and their squares)
        using the given function.

        The optimizer manages one tensor per parameter being optimized (means, log_scales, quats, logit_opacities, sh0, shN),
        and has one param_group per parameter.

        An adam param_group has the form (in PyTorch 2.8.0):
        ```
        optimizer.param_groups[i] = {
                'lr': lr,
                'name': name,
                'betas': (beta1, beta2),
                'eps': eps,
                'weight_decay': weight_decay,
                'amsgrad': amsgrad,
                'maximize': False,
                'foreach': None,
                'capturable': False,
                'differentiable': False,
                'fused': None,
                'decoupled_weight_decay': False,
                'initial_lr': lr_at_creation,
                'params': [parameter_tensor],
            }
        ```
        where the parameter_tensor is a reference to the tensor being optimized (e.g. model.means)

        The adam optimizer also keeps a running average of the gradients, their squares, and the step count
        in optimizer.state, keyed by the parameter tensor.
        _i.e._
        ```
        optimizer.state[parameter_tensor] = {
            'step': step,     # Tensor with a single integer tracking the step count
            'exp_avg': ...,   # Tensor with the same shape as the parameter tensor, tracking the running average of the gradients
            'exp_avg_sq': ... # Tensor with the same shape as the parameter tensor, tracking the running average of the squared gradients
            }
        ```

        This function updates optimizer.state[parameter_tensor][key] for each key in the state dict
        using the given optimizer_state_update_function, except for the 'step' key which is left unchanged.

        Args:
            optimizer_state_update_function (Callable[[torch.Tensor], torch.Tensor]):
                A function which takes a state tensor and returns an updated state tensor. Runs for
                every state tensor in the optimizer.
            parameter_name (str | None): If given, only update the state for the parameter with this name.
        """

        assert len(self._optimizer.param_groups) == 6, "Expected 6 parameter groups"
        print("updating params")
        for i, param_group in enumerate(self._optimizer.param_groups):
            if parameter_name is not None and param_group["name"] != parameter_name:
                continue
            print("updating parameter", param_group["name"])
            assert len(param_group["params"]) == 1, "Expected one parameter tensor per param group"
            p = param_group["params"][0]
            new_state = self._optimizer.state[p]
            del self._optimizer.state[p]
            for key, value in new_state.items():
                if key != "step":
                    new_state[key] = optimizer_state_update_function(value)
            new_parameter = getattr(self._model, param_group["name"])
            self._optimizer.param_groups[i]["params"] = [new_parameter]
            self._optimizer.state[new_parameter] = new_state
        print("done param update")

    @torch.no_grad
    def clip_opacities(self, value: float):
        """
        Clip the logit_opacities of each Gaussian so that it's opacity is less than or equal to <value>.

        Args:
            value (float): The new opacity value to set for all Gaussians.
        """
        # self._model.logit_opacities = torch.logit(
        #     torch.min(self._model.opacities, torch.full_like(self._model.opacities, value).logit_())
        # )
        self._model.logit_opacities = torch.clamp(
            self._model.logit_opacities, max=torch.logit(torch.tensor(value)).item()
        )
        self._update_optimizer_state(lambda x: torch.clamp(x, -10.0, 10.0), parameter_name="logit_opacities")

    @torch.no_grad
    def delete_gaussians(self, mask: torch.Tensor):
        """
        Delete Gaussians where mask is False.

        Args:
            mask (torch.Tensor): A boolean mask of shape (num_gaussians,) indicating which Gaussians to keep.

        """
        self._model.set_state(
            means=self._model.means[mask],
            quats=self._model.quats[mask],
            log_scales=self._model.log_scales[mask],
            logit_opacities=self._model.logit_opacities[mask],
            sh0=self._model.sh0[mask],
            shN=self._model.shN[mask],
        )
        self._update_optimizer_state(lambda x: x[mask])

    @torch.no_grad
    def insert_gaussians_by_duplication(
        self, mask: torch.Tensor, duplication_factor: int, use_revised_opacity_update: bool
    ):
        """
        Insert new Gaussians by duplicating those where mask is True. Each Gaussian to be
        duplicated is copied `duplication_factor` times (including the original).

        Args:
            mask (torch.Tensor): A boolean mask of shape (num_gaussians,) indicating which Gaussians to duplicate.
            duplication_factor (int): The number of copies to make of each selected Gaussian (including the original).
                Must be >= 2.
        """
        if duplication_factor < 2:
            raise ValueError("duplication_factor must be >= 2")

        indices = torch.where(mask)[0]  # Indices of Gaussians to duplicate, shape [M,]

        # Concatenate copies of the selected parameters to the end of the model parameters
        num_new_gaussians = duplication_factor - 1  # We already have one copy of each Gaussian in the model
        means_to_add = self._model.means[indices].repeat(num_new_gaussians, 1)  # [M*(D-1), 3]
        quats_to_add = self._model.quats[indices].repeat(num_new_gaussians, 1)  # [M*(D-1), 4]
        log_scales_to_add = self._model.log_scales[indices].repeat(num_new_gaussians, 1)  # [M*(D-1), 3]
        sh0_to_add = self._model.sh0[indices].repeat(num_new_gaussians, 1, 1)  # [M*(D-1), 1, 3]
        shN_to_add = self._model.shN[indices].repeat(num_new_gaussians, 1, 1)  # [M*(D-1), K-1, 3]

        # Update opacity values for the new Gaussians using the revised formulation from
        # the paper "Revising Densification in Gaussian Splatting" (https://arxiv.org/abs/2404.06109).
        if use_revised_opacity_update:
            logit_opacities_to_add = torch.sigmoid(1.0 - torch.sqrt(1.0 - self._model.opacities[indices]))
            logit_opacities_to_add = logit_opacities_to_add.repeat(num_new_gaussians)  # [M*(D-1),]
        else:
            logit_opacities_to_add = self._model.logit_opacities[indices].repeat(num_new_gaussians)  # [M*(D-1),]

        self._model.set_state(
            means=torch.cat([self._model.means, means_to_add], dim=0),
            quats=torch.cat([self._model.quats, quats_to_add], dim=0),
            log_scales=torch.cat([self._model.log_scales, log_scales_to_add], dim=0),
            logit_opacities=torch.cat([self._model.logit_opacities, logit_opacities_to_add], dim=0),
            sh0=torch.cat([self._model.sh0, sh0_to_add], dim=0),
            shN=torch.cat([self._model.shN, shN_to_add], dim=0),
        )

        def update_state_function(x: torch.Tensor) -> torch.Tensor:
            zpad = torch.zeros((len(indices) * num_new_gaussians, *x.shape[1:]), device=x.device)
            return torch.cat([x, zpad])

        self._update_optimizer_state(update_state_function)

    @torch.no_grad
    def insert_by_splitting(self, mask: torch.Tensor, split_factor: int, use_revised_opacity_update: bool):
        """
        Insert new Gaussians by splitting those where mask is True. Each Gaussian to be
        split is divided into `split_factor` parts. The means of the split Gaussians are chosen
        by sampling the original Gaussian, and the scales are divided `0.8 * split_factor`.
        The quaternions and spherical harmonics are copied from the original Gaussian.

        Args:
            mask (torch.Tensor): A boolean mask of shape (num_gaussians,) indicating which Gaussians to split.
            split_factor (int): The number of parts to split each selected Gaussian into. Must be >= 2.
        """
        if split_factor < 2:
            raise ValueError("split_factor must be >= 2")

        split_indices = torch.where(mask)[0]
        other_indices = torch.where(~mask)[0]
        print(
            self.model.means.shape,
            mask.shape,
            split_indices.shape,
            other_indices.shape,
            split_indices.max(),
            other_indices.max(),
        )

        device = self._model.device

        split_scales = self._model.scales[split_indices]
        split_quats = nnf.normalize(self._model.quats[split_indices], dim=-1)
        rotation_matrices = self._normalized_quat_to_rotmat(split_quats)  # [M, 3, 3]
        split_mean_offsets = torch.einsum(
            "nij,nj,bnj->bni",
            rotation_matrices,
            split_scales,
            torch.randn(split_factor, split_indices.shape[0], 3, device=device),
        )  # [S, N, 3]

        means_to_add = (self._model.means[split_indices] + split_mean_offsets).reshape(-1, 3)  # [S*M, 3]
        log_scales_to_add = torch.log(split_scales / (0.8 * split_factor)).repeat(split_factor, 1)  # [S*M, 3]
        quats_to_add = self._model.quats[split_indices].repeat(split_factor, 1)  # [S*M, 4]
        sh0_to_add = self._model.sh0[split_indices].repeat(split_factor, 1, 1)  # [S*M, 1, 3]
        shN_to_add = self._model.shN[split_indices].repeat(split_factor, 1, 1)  # [S*M, K-1, 3]

        # Update opacity values for the new Gaussians using the revised formulation from
        # the paper "Revising Densification in Gaussian Splatting" (https://arxiv.org/abs/2404.06109).
        if use_revised_opacity_update:
            logit_opacities_to_add = torch.sigmoid(1.0 - torch.sqrt(1.0 - self._model.opacities[split_indices]))
            logit_opacities_to_add = logit_opacities_to_add.repeat(split_factor)  # [S*M]
        else:
            logit_opacities_to_add = self._model.logit_opacities[split_indices].repeat(split_factor)  # [S*M]

        self._model.set_state(
            means=torch.cat([self._model.means[other_indices], means_to_add], dim=0),
            quats=torch.cat([self._model.quats[other_indices], quats_to_add], dim=0),
            log_scales=torch.cat([self._model.log_scales[other_indices], log_scales_to_add], dim=0),
            logit_opacities=torch.cat([self._model.logit_opacities[other_indices], logit_opacities_to_add], dim=0),
            sh0=torch.cat([self._model.sh0[other_indices], sh0_to_add], dim=0),
            shN=torch.cat([self._model.shN[other_indices], shN_to_add], dim=0),
        )

        def update_state_function(x: torch.Tensor) -> torch.Tensor:
            zero_params = torch.zeros((split_factor * len(split_indices), *x.shape[1:]), device=x.device)
            return torch.cat([x[other_indices], zero_params], dim=0)

        self._update_optimizer_state(update_state_function)


class GaussianSplatOptimizer:
    """
    Optimzier for training Gaussian Splat radiance fields over a collection of posed images.

    This optimizer uses Adam with a fixed learning rate for each parameter in a Gaussian Radiance field
    (i.e. means, covariances, opacities, spherical harmonics).
    It also handles splitting/duplicating/deleting Gaussians based on their opacity and gradients following the
    algorithm in the original Gaussian Splatting paper (https://arxiv.org/abs/2308.04079).
    """

    __PRIVATE__ = object()

    def __init__(
        self,
        state: _OptimizerState,
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

        self._state = state

        self._config = config

        # This hook counts the number of times we call backward between zeroing the gradients.
        # To determine whether a Gaussian should be split or duplicated, we threshold the *average*
        # gradient of its 2D mean with respect to the loss.
        # If we call backward multiple times per iteration (e.g. for different losses) or if we're accumulating gradients,
        # then the denominator of the average is the total number of backward calls since the last zero_grad().
        # This hook corrects the count even if backward() is called multiple times per iteration.
        self._num_grad_accumulation_steps = 1  # Number of times we've called backward since zeroing the gradients

        def _count_accumulation_steps_backward_hook(_):
            self._num_grad_accumulation_steps += 1

        self._state.model.means.register_hook(_count_accumulation_steps_backward_hook)

        # The actual numeric value to use when thresholding the 2D gradient to decide whether to grow a Gaussian.
        # This depends on the mode specified in the config.
        self._insertion_grad_2d_abs_threshold: float | None = (
            self._config.insertion_grad_2d_threshold
            if self._config.insertion_grad_2d_threshold_mode == InsertionGrad2dThresholdMode.CONSTANT
            else None
        )

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

        state = _OptimizerState(model=model, batch_size=batch_size, config=config)

        return cls(
            state=state,
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
        # if "version" not in state_dict:
        #     raise ValueError("State dict is missing version information")
        # if state_dict["version"] not in (3,):
        #     raise ValueError(f"Unsupported version: {state_dict['version']}")

        # config = GaussianSplatOptimizerConfig(**state_dict["config"])
        # optimizers = GaussianSplatOptimizer._make_optimizers(model, batch_size=1, config=config)
        # for name, optimizer in optimizers.items():
        #     optimizer.load_state_dict(state_dict["optimizers"][name])
        # means_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=1.0)

        # optimizer = cls(
        #     model=model,
        #     optimizers=optimizers,
        #     means_lr_scheduler=means_lr_scheduler,
        #     means_lr_decay_exponent=state_dict["means_lr_decay_exponent"],
        #     config=config,
        #     _private=cls.__PRIVATE__,
        # )
        # optimizer._insertion_grad_2d_abs_threshold = state_dict["insertion_grad_2d_abs_threshold"]

        # return optimizer
        raise NotImplementedError("from_state_dict is not implemented yet")

    def step(self):
        """
        Step the optimizers and update the learning rate schedulers, updating parameters of the model.
        """
        self._state.step(means_lr_decay=self._means_lr_decay_exponent)

    def zero_grad(self, set_to_none: bool = False):
        """
        Zero the gradients of all tensors being optimized.
        """
        self._num_grad_accumulation_steps = 0
        self._state.zero_grad()

    def state_dict(self) -> dict[str, Any]:
        """
        Return a serializable state dict for the optimizer.

        Returns:
            dict[str, Any]: A state dict containing the state of the optimizer.
        """
        return {}
        # return {
        #     "optimizers": {name: optimizer.state_dict() for name, optimizer in self._optimizers.items()},
        #     "means_lr_scheduler": self._means_lr_scheduler.state_dict(),
        #     "means_lr_decay_exponent": self._means_lr_decay_exponent,
        #     "insertion_grad_2d_abs_threshold": self._insertion_grad_2d_abs_threshold,
        #     "num_grad_accumulation_steps": self._num_grad_accumulation_steps,
        #     "config": vars(self._config),
        #     "version": 3,
        # }

    @torch.no_grad()
    def refine_gaussians(self, use_scales: bool = False, use_screen_space_scales: bool = False):
        if use_screen_space_scales:
            if not self._state.model.accumulate_max_2d_radii:
                raise ValueError(
                    "use_screen_space_scales is set to True but the model is not configured to "
                    + "track screen space scales. Set model.accumulate_max_2d_radii = True."
                )
        # Grow the number of Gaussians via:
        # 1. Duplicating those whose loss gradients are high and spatial size are small (i.e. have small eigenvals)
        # 2. Splitting those whose loss gradients are high and spatial size are large (i.e. have large eigenvals)
        #    or whose (2D projected) spatial extent simply exceeds the threshold self.grow_scale2d_threshold
        #
        # Note that splitting a Gaussian with mean μ and covariance Σ is implemented by sampling two new means
        # μ1, μ2 from N(μ, Σ), and setting the covariances Σ1 and Σ2 by dividing the eigenvalues of Σ by 1.6.
        n_dupli, n_split = self._grow_gs(use_screen_space_scales)
        # Prune Gaussians whose opacity is below a threshold or whose screen space spatial extent is too large
        n_prune = self._prune_gs(use_scales, use_screen_space_scales)
        # Reset running statistics used to determine which Gaussians to add/split/prune
        self._state.model.reset_accumulated_gradient_state()

        self._did_first_refinement = True
        return n_dupli, n_split, n_prune

    @torch.no_grad()
    def reset_opacities(self):
        """
        Reset the opacities to the given (post-sigmoid) value.
        """
        # Reset all opacities to twice the deletion threshold
        value = self._config.deletion_opacity_threshold * 2.0

        self._state.clip_opacities(value)

    def _compute_insertion_grad_2d_threshold(self, accumulated_mean_2d_gradients: torch.Tensor) -> float:
        # Helper to compute the quantile of the gradients, using NumPy if we have too many Gaussians for torch.quantile
        # which has a cap at 2**24 elements
        def _grad_2d_quantile(quantile: float) -> float:
            if self._state.model.num_gaussians > 2**24:
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
    def _grow_gs(self, use_screen_space_scales) -> tuple[int, int]:
        """
        Grow the number of Gaussians via:
          1. Duplicating those whose loss gradients are high and spatial size are small (i.e. have small eigenvals)
          2. Splitting those whose loss gradients are high and spatial size are large (i.e. have large eigenvals)
             or whose (2D projected) spatial extent simply exceeds the threshold self.grow_scale2d_threshold

        Note: Splitting a Gaussian with mean μ and covariance Σ is implemented by sampling two new means
              μ1, μ2 from N(μ, Σ), and setting the covariances Σ1 and Σ2 by dividing the eigenvalues of Σ by 1.6.

        Args:
            use_screen_space_scales: If set to true, use the tracked screen space scales to decide whether to split.
                                     Note that the model must have been configured to track these scales by setting
                                     GaussianSplat3d.track_max_2d_radii = True.
        """

        model = self._state.model
        # We use the average gradient ( over the the last N steps) of the projected Gaussians with respect to the
        # loss to decide which Gaussians to add/split/prune
        # count is the number of times a Gaussian has been projected (i.e. included in the loss gradient computation)
        # grad_2d is the sum of the gradients of the projected Gaussians (dL/dμ2D) over the last N steps
        count = model.accumulated_gradient_step_counts.clamp_min(1)
        if self._num_grad_accumulation_steps > 1:
            count *= self._num_grad_accumulation_steps

        grads = model.accumulated_mean_2d_gradient_norms / count
        device = grads.device

        # If the 2D projected gradient is high and the spatial size is small, duplicate the Gaussian
        is_grad_high = grads > self._compute_insertion_grad_2d_threshold(grads)
        is_small = model.scales.max(dim=-1).values <= self._config.insertion_scale_3d_threshold
        is_dupli = is_grad_high & is_small
        n_dupli: int = int(is_dupli.sum().item())

        # If the 2D projected gradient is high and the spatial size is large, split the Gaussian
        is_large = ~is_small
        is_split = is_grad_high & is_large
        # If the 2D projected spatial extent exceeds the threshold, split the Gaussian
        if use_screen_space_scales:
            is_split |= model.accumulated_max_2d_radii > self._config.insertion_scale_2d_threshold
        n_split: int = int(is_split.sum().item())

        # Hardcode these for now but could be made configurable
        dup_factor = 2
        split_factor = 2

        # First duplicate the Gaussians
        if n_dupli > 0:
            self._state.insert_gaussians_by_duplication(
                mask=is_dupli,
                duplication_factor=dup_factor,
                use_revised_opacity_update=self._config.opacity_updates_use_revised_formulation,
            )

        # Track new Gaussians added by duplication so we we don't split them
        is_split = torch.cat([is_split] + [torch.zeros(n_dupli, dtype=torch.bool, device=device)] * (dup_factor - 1))

        # Now split the Gaussians
        if n_split > 0:
            self._state.insert_by_splitting(
                mask=is_split,
                split_factor=split_factor,
                use_revised_opacity_update=self._config.opacity_updates_use_revised_formulation,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(self, use_scales: bool = False, use_screen_space_scales: bool = False) -> int:
        model = self._state.model

        # Prune any Gaussians whose opacity is below the threshold or whose (2D projected) spatial extent is too large
        is_prune = model.opacities.flatten() < self._config.deletion_opacity_threshold
        if use_scales:
            is_too_big = model.scales.max(dim=-1).values > self._config.deletion_scale_3d_threshold
            # The INRIA code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but it doesn't really get used
            if use_screen_space_scales:
                is_too_big |= model.accumulated_max_2d_radii > self._config.deletion_scale_2d_threshold

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            self._state.delete_gaussians(~is_prune)

        return int(n_prune)
