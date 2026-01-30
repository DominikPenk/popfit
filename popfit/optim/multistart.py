from __future__ import annotations

from typing import Literal, Optional

import torch

from ..core.model import Model
from ..parametrization import SigmoidBounded
from .base import Optimizer


class MultistartOptimizer(Optimizer):
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

    def start_optimization(self) -> None:
        super().start_optimization()
        for variable in self.model.variables():
            variable.push_parametrization(SigmoidBounded(variable.latent_bounds))

    def finalize_optimization(self) -> None:
        for variable in self.model.variables():
            param = variable.pop_parametrization()
            if not isinstance(param, SigmoidBounded):
                raise RuntimeError(
                    "Expected all variables to be parameterized with SigmoidBounded"
                )

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
