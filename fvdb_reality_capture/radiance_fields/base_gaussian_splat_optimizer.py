# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import torch
from fvdb import GaussianSplat3d

# Keeps track of names of registered optimizers and their classes.
REGISTERED_OPTIMIZERS = {}


DerivedOptimizer = TypeVar("DerivedOptimizer", bound=type)


def splat_optimizer(cls: DerivedOptimizer) -> DerivedOptimizer:
    """
    Decorator to register an optimizer class which inherits from :class:`BaseGaussianSplatOptimizer`.

    Args:
        cls: The optimizer class to register.

    Returns:
        cls: The registered optimizer class.
    """
    if not issubclass(cls, BaseGaussianSplatOptimizer):
        raise TypeError(f"Optimizer {cls} must inherit from BaseGaussianSplatOptimizer.")

    if cls.name() in REGISTERED_OPTIMIZERS:
        del REGISTERED_OPTIMIZERS[cls.name()]

    REGISTERED_OPTIMIZERS[cls.name()] = cls

    return cls


class BaseGaussianSplatOptimizer(ABC):
    """
    Base class for optimizers that reconstruct a scene using Gaussian Splat radiance fields over a collection of posed images.

    This class defines the interface for optimizers that optimize the parameters of a `fvdb.GaussianSplat3d` model, and
    provides utilities to refine the model by inserting and deleting Gaussians based on their contribution to the optimization.

    Currently, the only concrete implementation is :class:`GaussianSplatOptimizer`, which implements the algorithm in the
    `original Gaussian Splatting paper <https://arxiv.org/abs/2308.04079>`_.
    """

    @classmethod
    def name(cls) -> str:
        """
        Stable name used for optimizer (de)serialization and registry lookup.

        By default we use the class name. Override in subclasses if you need a different stable identifier.
        """
        return cls.__name__

    @classmethod
    @abstractmethod
    def from_state_dict(cls, model: GaussianSplat3d, state_dict: dict[str, Any]) -> "BaseGaussianSplatOptimizer":
        """
        Abstract method to create a new :class:`BaseGaussianSplatOptimizer` instance from a model and a state dict (obtained from :meth:`state_dict`).

        Args:
            model (GaussianSplat3d): The `GaussianSplat3d` model to optimize.
            state_dict (dict[str, Any]): A state dict previously obtained from :meth:`state_dict`.

        Returns:
            optimizer (BaseGaussianSplatOptimizer): A new :class:`BaseGaussianSplatOptimizer` instance.
        """
        # Backwards compatibility: older checkpoints may not have stored the optimizer class name.
        # Infer from version/keys where possible.
        if "name" not in state_dict:
            version = state_dict.get("version", None)
            if version == 3 or "insertion_grad_2d_abs_threshold" in state_dict:
                state_dict["name"] = "GaussianSplatOptimizer"
            elif version == 1 and "insertion_grad_2d_abs_threshold" not in state_dict:
                state_dict["name"] = "GaussianSplatOptimizerMCMC"
            else:
                raise ValueError(
                    "Optimizer state dict is missing 'name' and optimizer type could not be inferred. "
                    f"Keys: {sorted(state_dict.keys())}"
                )

        OptimizerType = REGISTERED_OPTIMIZERS.get(state_dict["name"], None)
        if OptimizerType is None:
            raise ValueError(
                f"Optimizer '{state_dict['name']}' is not registered. Optimizer classes must be registered "
                f"with the `splat_optimizer` decorator which will be called when the optimizer is defined. "
                f"Ensure the optimizer class uses the `splat_optimizer` decorator and was imported before calling from_state_dict."
            )
        return OptimizerType.from_state_dict(model, state_dict)

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """
        Abstract method to return a serializable state dict for the optimizer.

        Returns:
            state_dict (dict[str, Any]): A state dict containing the state of the optimizer.
        """
        pass

    @abstractmethod
    def reset_learning_rates_and_decay(self, batch_size: int, expected_steps: int) -> None:
        """
        Abstract method to set the learning rates and learning rate decay factor based on the batch size and the expected
        number of optimization steps (times :meth:`step` is called).

        This is useful if you want to change the batch size or expected number of steps after creating
        the optimizer.

        Args:
            batch_size (int): The batch size used for training. This is used to scale the learning rates.
            expected_steps (int): The expected number of optimization steps.
        """
        pass

    @abstractmethod
    def step(self):
        """
        Abstract method to step the optimizer (updating the model's parameters).
        """
        pass

    @abstractmethod
    def zero_grad(self, set_to_none: bool = False):
        """
        Abstract method to zero the gradients of all tensors being optimized.

        Args:
            set_to_none (bool): If ``True``, set the gradients to None instead of zeroing them. This can be more memory efficient.
        """
        pass

    @abstractmethod
    def filter_gaussians(self, indices_or_mask: torch.Tensor):
        """
        Abstract method to filter the Gaussians in the model based on the given indices or mask, and update the corresponding
        optimizer state accordingly. This can be used to delete, shuffle, or duplicate the Gaussians during optimization.

        Args:
            indices_or_mask (torch.Tensor): A 1D tensor of indices or a boolean mask indicating which Gaussians to keep.
        """
        pass

    @abstractmethod
    def refine(self, zero_gradients: bool = True) -> dict[str, Any]:
        """
        Abstract method to refine the model by inserting and deleting Gaussians based on their contribution to the optimization.

        Args:
            zero_gradients (bool): If True, zero the gradients of all tensors being optimized after refining.

        Returns:
            refinement_stat (dict[str, Any]): A dictionary containing statistics about the refinement step.
        """
        pass

    @abstractmethod
    def regularization_loss(self) -> torch.Tensor:
        """
        Abstract method to compute the regularization loss for the current model parameters.

        Returns:
            reg_loss (torch.Tensor): A scalar tensor representing the regularization loss.
        """
        pass
