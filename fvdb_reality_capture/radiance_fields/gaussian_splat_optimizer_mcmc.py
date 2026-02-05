# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from fvdb import GaussianSplat3d

from fvdb_reality_capture.radiance_fields.base_gaussian_splat_optimizer import (
    BaseGaussianSplatOptimizer,
    splat_optimizer,
)
from fvdb_reality_capture.radiance_fields.gaussian_splat_optimizer import (
    GaussianSplatOptimizer,
    GaussianSplatOptimizerConfig,
)
from fvdb_reality_capture.sfm_scene.sfm_scene import SfmScene


@dataclass
class GaussianSplatOptimizerMCMCConfig(GaussianSplatOptimizerConfig):
    """
    Parameters for configuring the ``GaussianSplatOptimizerMCMC``.
    """

    # Override base class defaults to match the MCMC paper:
    # "3D Gaussian Splatting as Markov Chain Monte Carlo" (https://arxiv.org/abs/2404.09591)
    initial_opacity: float = 0.5
    """
    Initial opacity of each Gaussian for MCMC optimization.

    Default: ``0.5`` (matches the MCMC paper, different from base 3DGS default of 0.1).
    """

    initial_covariance_scale: float = 0.1
    """
    Initial scale of each Gaussian for MCMC optimization.

    Default: ``0.1`` (matches the MCMC paper, different from base 3DGS default of 1.0).
    """

    noise_lr: float = 5e5
    """
    The learning rate for the noise added to the positions of the Gaussians.

    Default: ``5e5``.
    """

    insertion_rate: float = 1.05
    """
    The rate at which new Gaussians are inserted per step.

    Default: ``1.05`` (i.e., 5% more Gaussians per refinement step).
    """

    binomial_coeffs_n_max: int = 51
    """
    Maximum replication ratio used by the MCMC relocation kernel when computing updated opacities/scales
    for relocated/duplicated Gaussians.

    This controls the size of the binomial coefficient lookup table passed into
    :meth:`fvdb.GaussianSplat3d.relocate_gaussians`.

    Default: ``51``.
    """

    opacity_regularization: float = 0.01
    """
    Weight for opacity regularization loss :math:`L_{opacity} = \\frac{1}{N} \\sum_i |opacity_i|`.

    This loss encourages the opacities of the Gaussians to be small, which in turn encourages Gaussians to
    disapear in areas where they are not needed.

    Default: ``0.01``.
    """

    scale_regularization: float = 0.01
    """
    Weight for scale regularization loss :math:`L_{scale} = \\frac{1}{N} \\sum_i |scale_i|`.

    This loss encourages the scales of the Gaussians to be small, which in turn encourages Gaussians to
    disapear in areas where they are not needed.

    Default: ``0.01``.
    """

    def make_optimizer(self, model: GaussianSplat3d, sfm_scene: SfmScene) -> "GaussianSplatOptimizerMCMC":
        return GaussianSplatOptimizerMCMC.from_model_and_scene(
            model=model,
            sfm_scene=sfm_scene,
            config=self,
        )


@splat_optimizer
class GaussianSplatOptimizerMCMC(BaseGaussianSplatOptimizer):
    """
    MCMC optimizer for Gaussian Splat radiance fields.
    The optimizer uses an MCMC sampler to optimize the parameters of a ``fvdb.GaussianSplat3d`` model, and
    provides utilities to refine the model by inserting and deleting Gaussians based on their contribution to the
    optimization. The tools here mostly follow the algorithm in the Gaussian Splatting as Markov Chain Monte Carlo (MCMC)
    [paper](https://arxiv.org/abs/2404.09591).

    .. note:: You should not call the constructor of this class directly. Instead use :func:`from_model_and_scene`
              or :func:`from_state_dict`.
    """

    __PRIVATE__ = object()

    def __init__(
        self,
        model: GaussianSplat3d,
        config: GaussianSplatOptimizerMCMCConfig,
        optimizer: torch.optim.Adam,
        spatial_scale: float,
        refine_count: int,
        step_count: int,
        _private: Any = None,
    ):
        """
        Create a new ``GaussianSplatOptimizerMCMC`` instance from a model, optimizer and a config.

        Args:
            model (GaussianSplat3d): The ``GaussianSplat3d`` model to optimize.
            config (GaussianSplatOptimizerMCMCConfig): Configuration options for the optimizer.
            optimizer (torch.optim.Adam): The optimizer for the model.
            spatial_scale (float): A spatial scale for the scene used to interpret 3D scale thresholds in the config.
            refine_count (int): The number of times :func:`refine()` has been called on this optimizer.
            step_count (int): The number of times :func:`step()` has been called on this optimizer.
            _private (Any): A private object to prevent direct instantiation. Must be
                :obj:`GaussianSplatOptimizerMCMC.__PRIVATE__`.
        """
        if _private is not self.__PRIVATE__:
            raise RuntimeError(
                "GaussianSplatOptimizerMCMC must be created using from_model_and_config() or from_state_dict()"
            )
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        # How many times we've called step() on this optimizer
        self._step_count = step_count

        # How many times we've called refine() on this optimizer
        self._refine_count = refine_count

        # A spatial scale for the scene used to interpret 3D scale thresholds in the config
        self._spatial_scale = spatial_scale

        self._config = config
        self._model = model
        self._optimizer = optimizer

        # Store the decay exponent for the means learning rate schedule so we can serialize it
        self._means_lr_decay_exponent = 1.0
        self._means_lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._binomial_coeffs: torch.Tensor | None = None
        self._binomial_coeffs_n_max: int | None = None

        # Ensure we have a scheduler ready (does nothing unless _means_lr_decay_exponent != 1.0)
        self._ensure_means_lr_scheduler()

    def _ensure_means_lr_scheduler(self) -> None:
        """
        Ensure we have a `torch.optim.lr_scheduler` that decays only the "means" param group learning rate.

        We use `MultiplicativeLR` since our schedule is "multiply LR by a constant each optimizer step".
        """
        if self._means_lr_scheduler is not None:
            return

        def _one(_: int) -> float:
            return 1.0

        def _means_lambda(_: int, _self: "GaussianSplatOptimizerMCMC" = self) -> float:
            # Read exponent dynamically in case it gets updated via reset_learning_rates_and_decay().
            return float(_self._means_lr_decay_exponent)

        lr_lambdas: list[Callable[[int], float]] = []
        for pg in self._optimizer.param_groups:
            if pg.get("name") == "means":
                lr_lambdas.append(_means_lambda)
            else:
                lr_lambdas.append(_one)

        self._means_lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self._optimizer,
            lr_lambda=lr_lambdas,
        )

    def _get_binomial_coeffs(self) -> tuple[torch.Tensor, int]:
        """
        Get (and cache) the binomial coefficient table for the configured n_max on the current device.
        """
        n_max = int(self._config.binomial_coeffs_n_max)
        if n_max <= 0:
            raise ValueError("n_max must be > 0")
        if (
            self._binomial_coeffs is None
            or self._binomial_coeffs_n_max != n_max
            or self._binomial_coeffs.device != self._model.device
        ):
            self._binomial_coeffs = self._build_binomial_coeffs(n_max=n_max, device=self._model.device)
            self._binomial_coeffs_n_max = n_max
        return self._binomial_coeffs, n_max

    @staticmethod
    def _build_binomial_coeffs(n_max: int, device: torch.device) -> torch.Tensor:
        """
        Build a binomial coefficients lookup table of shape [n_max, n_max] as float32.
        Matches the reference implementation used in fvdb-core tests.

        Args:
            n_max (int): The maximum number of binomial coefficients to compute.
            device (torch.device): The device to compute the binomial coefficients on.

        Returns:
            torch.Tensor: A tensor of shape [n_max, n_max] containing the binomial coefficients.
        """
        coeffs = torch.zeros((n_max, n_max), device=device, dtype=torch.float32)
        for row in range(n_max):
            coeffs[row, 0] = 1.0
            coeffs[row, row] = 1.0
            for k in range(1, row):
                coeffs[row, k] = coeffs[row - 1, k - 1] + coeffs[row - 1, k]
        return coeffs

    @classmethod
    def from_model_and_scene(
        cls,
        model: GaussianSplat3d,
        sfm_scene: SfmScene,
        config: GaussianSplatOptimizerMCMCConfig = GaussianSplatOptimizerMCMCConfig(),
    ) -> "GaussianSplatOptimizerMCMC":
        """
        Create a new ``GaussianSplatOptimizerMCMC`` instance from a model and config.

        Args:
            model (GaussianSplat3d): The ``GaussianSplat3d`` model to optimize.
            sfm_scene (SfmScene): The ``SfmScene`` containing the scene data.
            config (GaussianSplatOptimizerMCMCConfig): Configuration options for the optimizer.
        """

        spatial_scale = sfm_scene.spatial_scale(config.spatial_scale_mode) * config.spatial_scale_multiplier
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
    def from_state_dict(cls, model: GaussianSplat3d, state_dict: dict[str, Any]) -> "GaussianSplatOptimizerMCMC":
        """
        Create a new ``GaussianSplatOptimizerMCMC`` instance from a model and a state dict.
        """
        if "version" not in state_dict:
            raise ValueError("State dict is missing version information")
        if state_dict["version"] not in (1,):
            raise ValueError(f"Unsupported version: {state_dict['version']}")

        config = GaussianSplatOptimizerMCMCConfig(**state_dict["config"])

        # We pass in 1.0 for the means_lr_scale since this is already baked into the optimizer state
        # which we load below.
        adam_optimizer = GaussianSplatOptimizer._make_optimizer(model=model, means_lr_scale=1.0, config=config)
        adam_optimizer.load_state_dict(state_dict["optimizer"])

        optimizer = cls(
            model=model,
            optimizer=adam_optimizer,
            spatial_scale=state_dict["spatial_scale"],
            config=config,
            step_count=state_dict["step_count"],
            refine_count=state_dict["refine_count"],
            _private=cls.__PRIVATE__,
        )
        optimizer._means_lr_decay_exponent = state_dict["means_lr_decay_exponent"]
        optimizer._ensure_means_lr_scheduler()

        return optimizer

    def step(self):
        """
        Step the optimizer (updating the model's parameters) and decay the learning rate of the means.
        """

        # MCMC optimization step adds noise to the positions of the Gaussians
        means_lr: float | None = None
        for pg in self._optimizer.param_groups:
            if pg.get("name") == "means":
                means_lr = float(pg["lr"])
                break
        if means_lr is None:
            raise RuntimeError("Could not find 'means' param group in optimizer")
        noise_scale = float(self._config.noise_lr) * means_lr
        if noise_scale != 0.0:
            self._model.add_noise_to_means(noise_scale=noise_scale)

        self._optimizer.step()
        self._step_count += 1
        # Decay the means learning rate (using a scheduler so only the "means" param group is affected)
        self._ensure_means_lr_scheduler()
        assert self._means_lr_scheduler is not None
        self._means_lr_scheduler.step()

    def zero_grad(self, set_to_none: bool = False):
        """
        Zero the gradients of all tensors being optimized.

        Args:
            set_to_none (bool): If ``True``, set the gradients to ``None`` instead of zeroing them.
                This can be more memory efficient.
        """
        self._optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def refine(self, zero_gradients: bool = True) -> dict[str, int]:
        """
        Perform a step of refinement by relocating and adding Gaussians.
        """
        num_gaussians_before_refinement = self._model.num_gaussians

        # teleport GSs
        num_relocated = self._relocate()

        # add new GSs
        if self._config.max_gaussians > 0:
            num_target = min(self._config.max_gaussians, int(self._config.insertion_rate * self._model.num_gaussians))
        else:
            num_target = int(self._config.insertion_rate * self._model.num_gaussians)
        num_added = max(0, num_target - self._model.num_gaussians)
        if num_added > 0:
            self._sample_add(num_added)

        if zero_gradients:
            self._model.log_scales.grad = None
            self._model.logit_opacities.grad = None
            self._model.quats.grad = None
            self._model.means.grad = None
            self._model.sh0.grad = None
            self._model.shN.grad = None

        self._refine_count += 1
        self._logger.debug(
            f"MCMC Optimizer refinement (step {self._step_count:,}): {num_relocated:,} relocated, {num_added:,} added. "
            f"Before refinement model had {num_gaussians_before_refinement:,} Gaussians, after refinement has {self._model.num_gaussians:,} Gaussians."
        )

        return {"num_relocated": num_relocated, "num_added": num_added}

    def regularization_loss(self) -> torch.Tensor:
        """
        Compute a loss regularizing the scales and opacities of the Gaussians in the model.

        This loss encourages the opacities and scales of the Gaussians to be small, encouraging Gaussians to disapear
        in areas where they are not needed (to later be relocated to more productive areas).

        Returns:
            reg_loss (torch.Tensor): A scalar tensor representing the regularization loss.
        """

        # Rgularize opacity to ensure Gaussian's don't become too opaque
        loss = self._config.opacity_regularization * self._model.opacities.mean()

        # Regularize scales to ensure Gaussians don't become too large
        loss = loss + self._config.scale_regularization * self._model.scales.mean()

        return loss

    @staticmethod
    @torch.no_grad()
    def _multinomial_sample(weights: torch.Tensor, n: int, replacement: bool = True) -> torch.Tensor:
        """Sample from a distribution using torch.multinomial or numpy.random.choice.

        This function adaptively chooses between `torch.multinomial` and `numpy.random.choice`
        based on the number of elements in `weights`. If the number of elements exceeds
        the torch.multinomial limit (2^24), it falls back to using `numpy.random.choice`.

        Args:
            weights (Tensor): A 1D tensor of weights for each element.
            n (int): The number of samples to draw.
            replacement (bool): Whether to sample with replacement. Default is True.

        Returns:
            Tensor: A 1D tensor of sampled indices.
        """
        num_elements = weights.size(0)
        if num_elements <= 2**24:
            return torch.multinomial(weights, n, replacement=replacement)
        else:
            weights = weights / weights.sum()
            weights_np = weights.detach().cpu().numpy()
            sampled_idxs_np = np.random.choice(num_elements, size=n, p=weights_np, replace=replacement)
            sampled_idxs = torch.from_numpy(sampled_idxs_np)
            return sampled_idxs.to(weights.device)

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
    def _relocate(self) -> int:
        """Inplace relocate some dead Gaussians to the location of a sample of live ones.

        Returns:
            int: The number of Gaussians relocated.
        """
        dead_mask = self._model.opacities <= self._config.deletion_opacity_threshold
        n_gs = int(dead_mask.sum().item())
        if n_gs > 0:
            dead_indices = dead_mask.nonzero(as_tuple=True)[0]
            alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
            n = len(dead_indices)

            # Sample for new GSs
            probs = self._model.opacities[alive_indices].flatten()  # ensure its shape is [N,]
            if probs.numel() == 0:
                return 0
            # multinomial requires sum(probs) > 0
            if float(probs.sum().item()) == 0.0:
                probs = torch.ones_like(probs)
            sampled_idxs = self._multinomial_sample(probs, n, replacement=True)
            sampled_idxs = alive_indices[sampled_idxs]
            ratios = torch.bincount(sampled_idxs, minlength=self._model.num_gaussians)[sampled_idxs] + 1
            binomial_coeffs, n_max = self._get_binomial_coeffs()
            ratios = ratios.to(dtype=torch.int32)
            new_logit_opacities, new_log_scales = self._model.relocate_gaussians(
                log_scales=self._model.log_scales[sampled_idxs],
                logit_opacities=self._model.logit_opacities[sampled_idxs],
                ratios=ratios,
                binomial_coeffs=binomial_coeffs,
                n_max=n_max,
                min_opacity=self._config.deletion_opacity_threshold,
            )

            self._model.log_scales[sampled_idxs] = new_log_scales
            self._model.logit_opacities[sampled_idxs] = new_logit_opacities
            for param_name in ["log_scales", "logit_opacities", "quats", "means", "sh0", "shN"]:
                param = getattr(self._model, param_name)
                param[dead_indices] = param[sampled_idxs]

            def zero_sampled_gradients(x: torch.Tensor) -> torch.Tensor:
                x[sampled_idxs] = 0
                return x

            self._update_optimizer_params_and_state(
                optimizer_fn=zero_sampled_gradients,
                parameter_names={"log_scales", "logit_opacities", "quats", "means", "sh0", "shN"},
                reset_adam_step_counts=False,
            )
        return n_gs

    @torch.no_grad()
    def _sample_add(self, n: int) -> int:
        """Sample new Gaussians from the model.

        Args:
            n (int): The number of new Gaussians to sample.

        Returns:
            int: The number of new Gaussians sampled.
        """
        probs = self._model.opacities.flatten()  # ensure its shape is [N,]
        if probs.numel() == 0:
            return 0
        if float(probs.sum().item()) == 0.0:
            probs = torch.ones_like(probs)
        sampled_idxs = self._multinomial_sample(probs, n, replacement=True)
        ratios = torch.bincount(sampled_idxs, minlength=self._model.num_gaussians)[sampled_idxs] + 1
        binomial_coeffs, n_max = self._get_binomial_coeffs()
        ratios = ratios.to(dtype=torch.int32)
        new_logit_opacities, new_log_scales = self._model.relocate_gaussians(
            log_scales=self._model.log_scales[sampled_idxs],
            logit_opacities=self._model.logit_opacities[sampled_idxs],
            ratios=ratios,
            binomial_coeffs=binomial_coeffs,
            n_max=n_max,
            min_opacity=self._config.deletion_opacity_threshold,
        )

        self._model.log_scales[sampled_idxs] = new_log_scales
        self._model.logit_opacities[sampled_idxs] = new_logit_opacities

        # Extend model tensors with sampled copies (after relocation adjustment)
        self._model.set_state(
            means=torch.cat([self._model.means, self._model.means[sampled_idxs]], dim=0),
            quats=torch.cat([self._model.quats, self._model.quats[sampled_idxs]], dim=0),
            log_scales=torch.cat([self._model.log_scales, self._model.log_scales[sampled_idxs]], dim=0),
            logit_opacities=torch.cat([self._model.logit_opacities, self._model.logit_opacities[sampled_idxs]], dim=0),
            sh0=torch.cat([self._model.sh0, self._model.sh0[sampled_idxs]], dim=0),
            shN=torch.cat([self._model.shN, self._model.shN[sampled_idxs]], dim=0),
        )

        def zero_extend_sampled_gradients(x: torch.Tensor) -> torch.Tensor:
            x = torch.cat([x, torch.zeros(n, *x.shape[1:], dtype=x.dtype, device=x.device)])
            return x

        self._update_optimizer_params_and_state(
            optimizer_fn=zero_extend_sampled_gradients,
            parameter_names={"log_scales", "logit_opacities", "quats", "means", "sh0", "shN"},
            reset_adam_step_counts=False,
        )
        return n

    def state_dict(self) -> dict[str, Any]:
        """
        Return a serializable state dict for the optimizer.

        Returns:
            state_dict (dict[str, Any]): A state dict containing the state of the optimizer.
        """
        return {
            "name": self.__class__.name(),
            "optimizer": self._optimizer.state_dict(),
            "means_lr_decay_exponent": self._means_lr_decay_exponent,
            "config": vars(self._config),
            "spatial_scale": self._spatial_scale,
            "step_count": self._step_count,
            "refine_count": self._refine_count,
            "version": 1,
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

        # Ensure scheduler exists so subsequent steps apply the updated decay exponent.
        # If a scheduler already exists, it reads _means_lr_decay_exponent dynamically.
        self._ensure_means_lr_scheduler()
