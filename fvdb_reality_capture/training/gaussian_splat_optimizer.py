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
import torch.nn.functional as F
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
        Step the optimizers and update the learning rate schedulers, updating parameters of the model.
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
    def refine_gaussians(self, use_scales: bool = False, use_screen_space_scales: bool = False):
        if use_screen_space_scales:
            if not self._model.accumulate_max_2d_radii:
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
        self._model.reset_accumulated_gradient_state()

        self._did_first_refinement = True
        return n_dupli, n_split, n_prune

    @torch.no_grad()
    def reset_opacities(self):
        """
        Reset the opacities to the given (post-sigmoid) value.
        """

        # Reset all opacities to twice the deletion threshold
        value = self._config.deletion_opacity_threshold * 2.0

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            if name == "logit_opacities":
                return torch.clamp(p, max=torch.logit(torch.tensor(value)).item())
            else:
                raise ValueError(f"Unexpected parameter name: {name}")

        def optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(v)

        # update the parameters and the state in the optimizers
        params = self._update_optimizer(param_fn, optimizer_fn, "logit_opacities")
        self._model.logit_opacities = params["logit_opacities"]

    @staticmethod
    def _make_optimizer(model, batch_size, config):
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

        # We use the average gradient ( over the the last N steps) of the projected Gaussians with respect to the
        # loss to decide which Gaussians to add/split/prune
        # count is the number of times a Gaussian has been projected (i.e. included in the loss gradient computation)
        # grad_2d is the sum of the gradients of the projected Gaussians (dL/dμ2D) over the last N steps
        count = self._model.accumulated_gradient_step_counts.clamp_min(1)
        if self._num_grad_accumulation_steps > 1:
            count *= self._num_grad_accumulation_steps

        grads = self._model.accumulated_mean_2d_gradient_norms / count
        device = grads.device

        # If the 2D projected gradient is high and the spatial size is small, duplicate the Gaussian
        is_grad_high = grads > self._compute_insertion_grad_2d_threshold(grads)
        is_small = self._model.scales.max(dim=-1).values <= self._config.insertion_scale_3d_threshold
        is_dupli = is_grad_high & is_small
        n_dupli: int = int(is_dupli.sum().item())

        # If the 2D projected gradient is high and the spatial size is large, split the Gaussian
        is_large = ~is_small
        is_split = is_grad_high & is_large

        # If the 2D projected spatial extent exceeds the threshold, split the Gaussian
        if use_screen_space_scales:
            is_split |= self._model.accumulated_max_2d_radii > self._config.insertion_scale_2d_threshold
        n_split: int = int(is_split.sum().item())

        # Hardcode these for now but could be made configurable
        dup_factor = 1  # 1 means one gaussian becomes 2, 2 means one gaussian becomes 3, etc.
        split_factor = 2

        # First duplicate the Gaussians
        if n_dupli > 0:
            self.duplicate_gaussians(mask=is_dupli, dup_factor=dup_factor)

        # Track new Gaussians added by duplication so we we don't split them
        is_split = torch.cat([is_split] + [torch.zeros(n_dupli, dtype=torch.bool, device=device)] * dup_factor)

        # Now split the Gaussians
        if n_split > 0:
            self.subdivide_gaussians(mask=is_split, split_factor=split_factor)
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(self, use_scales: bool = False, use_screen_space_scales: bool = False) -> int:
        # Prune any Gaussians whose opacity is below the threshold or whose (2D projected) spatial extent is too large
        is_prune = self._model.opacities.flatten() < self._config.deletion_opacity_threshold
        if use_scales:
            is_too_big = self._model.scales.max(dim=-1).values > self._config.deletion_scale_3d_threshold
            # The INRIA code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but it doesn't really get used
            if use_screen_space_scales:
                is_too_big |= self._model.accumulated_max_2d_radii > self._config.deletion_scale_2d_threshold

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            self.remove_gaussians(mask=is_prune)

        return int(n_prune)

    @torch.no_grad()
    def subdivide_gaussians(self, mask: torch.Tensor, split_factor: int = 2):
        """
        Split the Gaussian with the given mask.

        Args:
            mask: A boolean mask with shape [num_means,] indicating which Gaussians to split.
            split_factor: The number of splits for each Gaussian. Default: 4.
        """

        def _normalized_quat_to_rotmat(quat_: torch.Tensor) -> torch.Tensor:
            """Convert normalized quaternion to rotation matrix.

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

        device = mask.device
        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = self._model.scales[sel]  # [N,]
        quats = F.normalize(self._model.quats[sel], dim=-1)
        rotmats = _normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(split_factor, len(scales), 3, device=device),
        )  # [S, N, 3]

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            repeats = [split_factor] + [1] * (p.dim() - 1)
            cat_dim = 0
            if name == "means":
                p_split = (p[sel] + samples).reshape(-1, 3)  # [S*N, 3]
                p_rest = p[rest]
            elif name == "log_scales":
                # TODO: Adjust scale factor for splitting
                p_split = torch.log(scales / 1.6).repeat(split_factor, 1)  # [2N, 3]
                p_rest = p[rest]
            elif name == "logit_opacities" and self._config.opacity_updates_use_revised_formulation:
                new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
                p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
                p_rest = p[rest]
            else:
                p_split = p[sel].repeat(repeats)
                p_rest = p[rest]
            p_new = torch.cat([p_rest, p_split], dim=cat_dim)
            return p_new

        def optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            v_split = torch.zeros((split_factor * len(sel), *v.shape[1:]), device=device)
            v_rest = v[rest]
            return torch.cat([v_rest, v_split], dim=0)

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(param_fn, optimizer_fn)

    @torch.no_grad()
    def duplicate_gaussians(self, mask: torch.Tensor, dup_factor: int = 1):
        """Duplicate the Gaussian with the given mask.

        Args:
            mask: A boolean mask of shape [num_means,] indicating which Gaussians to duplicate.
            dup_factor: The number of times to duplicate the selected Gaussians.
        """
        device = mask.device
        sel = torch.where(mask)[0]

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            repeats = [dup_factor] + [1] * (p.dim() - 1)
            p_sel = p[sel]
            return torch.cat([p, p_sel.repeat(repeats)], dim=0)

        def optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            zpad = torch.zeros((len(sel) * dup_factor, *v.shape[1:]), device=device)
            return torch.cat([v, zpad])

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(param_fn, optimizer_fn)

    @torch.no_grad()
    def remove_gaussians(self, mask: torch.Tensor):
        """Remove the Gaussian with the given mask.

        Args:
            mask: A boolean mask of shape [num_means,] indicating which Gaussians to remove.
        """
        sel = torch.where(~mask)[0]

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return p[sel]

        def optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            return v[sel]

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(param_fn, optimizer_fn)

    @torch.no_grad()
    def _update_optimizer(
        self,
        param_fn: Callable[[str, torch.Tensor], torch.Tensor],
        optimizer_fn: Callable[[str, str, torch.Tensor], torch.Tensor],
        parameter_name: str | None = None,
    ) -> dict[str, torch.Tensor]:
        # Each optimizer only manages one tensor, so it should have exactly one param_group with a single parameter
        #
        # An adam param_group has the form (in PyTorch 2.8.0):
        # {
        #     'lr': lr,
        #     'name': name,
        #     'betas': (beta1, beta2),
        #     'eps': eps,
        #     'weight_decay': weight_decay,
        #     'amsgrad': amsgrad,
        #     'maximize': False,
        #     'foreach': None,
        #     'capturable': False,
        #     'differentiable': False,
        #     'fused': None,
        #     'decoupled_weight_decay': False,
        #     'initial_lr': lr_at_creation,
        #     'params': [parameter_tensor],
        # }
        # where the parameter_tensor is a reference to the tensor being optimized (e.g. model.means)
        #
        # The adam optimizer also keeps a running average of the gradients, their squares, and the step count

        params = dict()
        for i, param_group in enumerate(self._optimizer.param_groups):
            if parameter_name is not None and param_group["name"] != parameter_name:
                continue
            assert len(param_group["params"]) == 1, "Expected one parameter tensor per param group"
            p = param_group["params"][0]
            name = param_group["name"]
            new_state = self._optimizer.state[p]
            del self._optimizer.state[p]
            for key, value in new_state.items():
                if key != "step":
                    new_state[key] = optimizer_fn(name, key, value)
            new_parameter = param_fn(name, p)
            new_parameter.requires_grad = True
            self._optimizer.param_groups[i]["params"] = [new_parameter]
            self._optimizer.state[new_parameter] = new_state
            params[name] = new_parameter
        return params

    @torch.no_grad()
    def _update_param_with_optimizer(
        self,
        param_fn: Callable[[str, torch.Tensor], torch.Tensor],
        optimizer_fn: Callable[[str, str, torch.Tensor], torch.Tensor],
        names: list[str] | None = None,
    ):
        """Update the parameters and the state in the optimizers with defined functions.

        Args:
            param_fn: A function that takes the name of the parameter and the parameter itself,
                and returns the new parameter.
            optimizer_fn: A function that takes the key of the optimizer state and the state value,
                and returns the new state value.
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            names: A list of key names to update. If None, update all. Default: None.
        """
        params = {
            "means": self._model.means,
            "log_scales": self._model.log_scales,
            "quats": self._model.quats,
            "logit_opacities": self._model.logit_opacities,
            "sh0": self._model.sh0,
            "shN": self._model.shN,
        }
        if names is None:
            # If names is not provided, update all parameters
            names = list(params.keys())

        params = self._update_optimizer(param_fn, optimizer_fn)
        self._model.set_state(
            means=params["means"],
            quats=params["quats"],
            log_scales=params["log_scales"],
            logit_opacities=params["logit_opacities"],
            sh0=params["sh0"],
            shN=params["shN"],
        )
