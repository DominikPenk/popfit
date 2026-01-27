from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn as nn

from ..core.model import Model
from ..core.variable import Variable
from .base import Optimizer


class MultistartVariable(Variable):
    def __init__(
        self,
        value: Optional[float | torch.Tensor] = None,
        bounds: Optional[Sequence[Any] | torch.Tensor] = None,
        *,
        population: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            value=value,
            bounds=bounds,
            population=population,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        self.mask = nn.Buffer(torch.all(torch.isfinite(self._bounds), dim=0))
        self.z = nn.Parameter(self.theta_to_z(population))
        pass

    @classmethod
    def from_base_variable(
        cls: type[MultistartVariable], base: Variable
    ) -> MultistartVariable:
        return cls(
            value=base.optimal,
            bounds=base.bounds,
            population=base.population,
            dtype=base.dtype,
            device=base.device,
        )

    def to_base_variable(self) -> Variable:
        return Variable(
            value=self.optimal,
            bounds=self.bounds,
            population=self.z_to_theta(),
            dtype=self.dtype,
            device=self.device,
        )

    def theta_to_z(self, theta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if not torch.any(self.mask):
            return theta

        safe_lower = torch.where(self.mask, self.lower_bound, 0.0)
        safe_upper = torch.where(self.mask, self.upper_bound, 1.0)
        val = (theta - safe_lower) / (safe_upper - safe_lower)
        val = torch.clamp(val, eps, 1.0 - eps)
        val = torch.logit(val)
        return torch.where(self.mask, val, theta)

    def z_to_theta(self) -> torch.Tensor:
        if not torch.any(self.mask):
            return self.z

        safe_lower = torch.where(self.mask, self.lower_bound, 0.0)
        safe_upper = torch.where(self.mask, self.upper_bound, 1.0)
        val = safe_lower + (safe_upper - safe_lower) * torch.sigmoid(self.z)
        return torch.where(self.mask, val, self.z)

    @property
    def population(self) -> torch.Tensor:
        return self.z_to_theta()


class MultistartOptimizer(Optimizer[MultistartVariable]):
    VariableType = MultistartVariable

    def __init__(
        self,
        model: Model,
        optimizer_cls: type[torch.optim.Optimizer],
        *,
        lr: float = 1e-2,
        population_size: int = 64,
        invalid_handing: Literal["ignore", "resample"] = "resample",
        **optimizer_args,
    ) -> None:
        super().__init__(
            model, population_size=population_size, invalid_handling=invalid_handing
        )
        self.optimizer_cls = optimizer_cls
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self.optimizer_args = dict(optimizer_args)
        self.optimizer_args["lr"] = lr

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer_cls(
            self.model.parameters(),
            **self.optimizer_args,
        )

    def step(self, losses: torch.Tensor) -> float:
        if self._optimizer is None:
            self._optimizer = self.get_optimizer()

        self.update_global_best(losses)

        valid_mask = self.validate_population(losses)

        if not torch.any(valid_mask):
            raise RuntimeError(
                "All individuals are invalid; cannot perform optimization step."
            )

        loss = losses[valid_mask].mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return self.global_best_loss

    @property
    def lr(self) -> float:
        return self.optimizer_args["lr"]

    @lr.setter
    def lr(self, value: float) -> None:
        self.optimizer_args["lr"] = value


class MultistartAdam(MultistartOptimizer):
    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        decoupled_weight_decay: bool = False,
        population_size: int = 64,
        invalid_handing: Literal["ignore", "resample"] = "resample",
    ) -> None:
        super().__init__(
            model,
            torch.optim.Adam,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=decoupled_weight_decay,
            population_size=population_size,
            invalid_handing=invalid_handing,
        )


class MultistartAdamW(MultistartOptimizer):
    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        population_size: int = 64,
        invalid_handing: Literal["ignore", "resample"] = "resample",
    ) -> None:
        super().__init__(
            model,
            torch.optim.AdamW,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            population_size=population_size,
            invalid_handing=invalid_handing,
        )


class MultistartGD(MultistartOptimizer):
    def __init__(
        self,
        model: Model,
        lr: float = 1e-2,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        *,
        foreach: bool | None = None,
        differentiable: bool = False,
        fused: bool | None = None,
        population_size: int = 64,
        invalid_handing: Literal["ignore", "resample"] = "resample",
    ) -> None:
        super().__init__(
            model,
            torch.optim.SGD,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
            population_size=population_size,
            invalid_handing=invalid_handing,
        )


class MultistartRMSprop(MultistartOptimizer):
    def __init__(
        self,
        model: Model,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        capturable: bool = False,
        foreach: bool | None = None,
        differentiable: bool = False,
        population_size: int = 64,
        invalid_handing: Literal["ignore", "resample"] = "resample",
    ) -> None:
        super().__init__(
            model,
            torch.optim.RMSprop,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            capturable=capturable,
            foreach=foreach,
            differentiable=differentiable,
            population_size=population_size,
            invalid_handing=invalid_handing,
        )
