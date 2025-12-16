import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch
from fvdb import GaussianSplat3d

from fvdb_reality_capture.radiance_fields.base_gaussian_splat_optimizer import (
    BaseGaussianSplatOptimizer,
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

    noise_lr: float = 5e5
    """
    The learning rate for the noise added to the positions of the Gaussians.
    """
    insertion_rate = 1.05
    """
    The rate at which new Gaussians are inserted per step.
    """


class GaussianSplatOptimizerMCMC(BaseGaussianSplatOptimizer):
    """
    MCMC optimizer for Gaussian Splat radiance fields.
    The optimizer uses an MCMC sampler to optimize the parameters of a ``fvdb.GaussianSplat3d`` model, and
    provides utilities to refine the model by inserting and deleting Gaussians based on their contribution to the
    optimization. The tools here mostly follow the algorithm in the Gaussian Splating as Markov Chain Monte Carlo (MCMC)
    [paper](https://arxiv.org/abs/2404.09591).

    .. note:: You should not call the constructor of this class directly. Instead use :func:`from_model_and_config`
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

        # How many timeds we've called step() on this optimizer
        self._step_count = step_count

        # How many times we've called refine() on this optimizer
        self._refine_count = refine_count

        # A spatial scale for the scene used to interpret 3D scale thresholds in the config
        self._spatial_scale = spatial_scale

        self._config = config
        self._model = model
        self._optimizer = optimizer

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

        return optimizer

    def step(self):
        """
        Step the optimizer (updating the model's parameters) and decay the learning rate of the means.
        """

        # MCMC optimization step adds noise to the positions of the Gaussians
        self._model.add_noise_to_means(noise_scale=self._config.noise_lr * self._optimizer.param_groups["means"]["lr"])

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

    def refine(self) -> dict[str, int]:
        """
        Perform a step of refinement by relocating and adding Gaussians.
        """
        num_gaussians_before_refinement = self._model.num_gaussians

        # teleport GSs
        num_relocated = self._relocate()

        # add new GSs
        num_target = min(self._config.max_gaussians, int(self._config.insertion_rate * self._model.num_gaussians))
        num_added = max(0, num_target - self._model.num_gaussians)
        if num_added > 0:
            self._sample_add(num_added)

        self._model.log_scales.grad = None
        self._model.logit_opacities.grad = None
        self._model.quats.grad = None
        self._model.means.grad = None
        self._model.sh0.grad = None
        self._model.shN.grad = None

        if self.verbose:
            f"MCMC Optimizer refinement (step {self._step_count:,}): {num_relocated:,} relocated, {num_added:,} added. "
            f"Before refinement model had {num_gaussians_before_refinement:,} Gaussians, after refinement has {self._model.num_gaussians:,} Gaussians."

        return {"num_relocated": num_relocated, "num_added": num_added}

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
        dead_mask = self._model.opacities() <= self._config.deletion_opacity_threshold
        n_gs = dead_mask.sum().item()
        if n_gs > 0:
            dead_indices = dead_mask.nonzero(as_tuple=True)[0]
            alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
            n = len(dead_indices)

            # Sample for new GSs
            eps = torch.finfo(torch.float32).eps
            probs = self._model.opacities()[alive_indices].flatten()  # ensure its shape is [N,]
            sampled_idxs = self._multinomial_sample(probs, n, replacement=True)
            sampled_idxs = alive_indices[sampled_idxs]
            new_logit_opacities, new_log_scales = self._model.relocate_gaussians(
                log_scales=self._model.log_scales[sampled_idxs],
                logit_opacities=self._model.logit_opacities[sampled_idxs],
                ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
                binomial_coeffs=self._binomial_coeffs,
                n_max=self._config.n_max,
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
        """Sample new Gaussians from the model."""
        probs = self._model.opacities().flatten()  # ensure its shape is [N,]
        sampled_idxs = self._multinomial_sample(probs, n, replacement=True)
        new_logit_opacities, new_log_scales = self._model.relocate_gaussians(
            log_scales=self._model.log_scales[sampled_idxs],
            logit_opacities=self._model.logit_opacities[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
            binomial_coeffs=self._binomial_coeffs,
            n_max=self._config.n_max,
        )

        self._model.log_scales[sampled_idxs] = new_log_scales
        self._model.logit_opacities[sampled_idxs] = new_logit_opacities

        for param_name in ["log_scales", "logit_opacities", "quats", "means", "sh0", "shN"]:
            param = getattr(self._model, param_name)
            param = torch.cat([param, param[sampled_idxs]])

        def zero_extend_sampled_gradients(x: torch.Tensor) -> torch.Tensor:
            x = torch.cat([x, torch.zeros(n, *x.shape[1:], dtype=x.dtype, device=x.device)])
            return x

        self._update_optimizer_params_and_state(
            optimizer_fn=zero_extend_sampled_gradients,
            parameter_names={"log_scales", "logit_opacities", "quats", "means", "sh0", "shN"},
            reset_adam_step_counts=False,
        )

    def state_dict(self) -> dict[str, Any]:
        """
        Return a serializable state dict for the optimizer.

        Returns:
            state_dict (dict[str, Any]): A state dict containing the state of the optimizer.
        """
        return {
            "optimizer": self._optimizer.state_dict(),
            "means_lr_decay_exponent": self._means_lr_decay_exponent,
            "config": vars(self._config),
            "spatial_scale": self._spatial_scale,
            "step_count": self._step_count,
            "refine_count": self._refine_count,
            "version": 1,
        }
