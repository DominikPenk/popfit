from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn as nn

from ..core import Model, Variable
from .base import Optimizer


class PSOVariable(Variable):
    """
    Curve parameter with PSO-specific attributes for Particle Swarm Optimization.

    Extends Variable to include velocity and personal best tracking for PSO.

    Args:
        value (torch.Tensor): Initial parameter values.
        bounds (tuple[torch.Tensor, torch.Tensor]): Lower and upper bounds for the parameter.
        unit (str, optional): Unit of the parameter.
        description (str, optional): Description of the parameter.
        velocity (torch.Tensor, optional): Initial velocity for the parameter.
        personal_best (torch.Tensor, optional): Initial personal best value.
    """

    def __init__(
        self,
        value: Optional[float | torch.Tensor] = None,
        bounds: Optional[Sequence[Any] | torch.Tensor] = None,
        *,
        population: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        personal_best: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            value, bounds=bounds, population=population, dtype=dtype, device=device
        )
        velocity = torch.zeros_like(self.population) if velocity is None else velocity
        personal_best = (
            self.population.clone().detach()
            if personal_best is None
            else torch.as_tensor(personal_best, dtype=dtype, device=device)
        )

        if velocity.shape != self.population.shape:
            raise ValueError(
                f"velocity must have the same shape as value. {velocity.shape} vs. {self.population.shape}"
            )

        if personal_best.shape != self.population.shape:
            raise ValueError(
                f"personal_best must have the same shape as population. {personal_best.shape} vs. {self.population.shape}"
            )

        self.velocity = velocity
        self.personal_best = nn.Buffer(self.population.clone())

    @classmethod
    def from_base_variable(
        cls,
        base: Variable,
    ) -> "PSOVariable":
        """
        Create a PSOVariable from a base Variable.

        Args:
            base (Variable): The base parameter to convert.

        Returns:
            PSOVariable: The PSO-augmented parameter.
        """
        return cls(
            value=base.optimal,
            bounds=base.bounds,
            population=base.population,
            dtype=base.dtype,
            device=base.device,
        )

    def to_base_variable(self):
        """
        Convert this PSOVariable back to a standard Variable.

        Returns:
            Variable: The parameter in the original (bounded) space.
        """
        return Variable(
            value=self.optimal,
            bounds=self.bounds,
            population=self.population,
            dtype=self.dtype,
            device=self.device,
        )


class PSO(Optimizer[PSOVariable]):
    """
    Simple Particle Swarm Optimization (PSO) optimizer for curve fitting.

    This optimizer maintains a population of parameter candidates (particles), each with velocity and personal best tracking.
    It updates particles according to the PSO update rule to minimize a user-supplied loss function.

    Args:
        params (Union[Mapping[str, Variable], Iterable[Variable]]): Parameters to optimize.
        population_size (int, optional): Number of particles in the swarm. Defaults to 500.
        inertia (float, optional): Inertia coefficient. Defaults to 0.7.
        cognitive (float, optional): Cognitive (personal best) coefficient. Defaults to 1.5.
        social (float, optional): Social (global best) coefficient. Defaults to 1.7.
    """

    VariableType = PSOVariable

    def __init__(
        self,
        model: Model,
        *,
        population_size: int = 500,
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.7,
        invalid_handling: Literal["resample", "ignore"] = "ignore",
    ) -> None:
        super().__init__(
            model, population_size=population_size, invalid_handling=invalid_handling
        )

        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        self.personal_best_loss = torch.full(
            (self.population_size,),
            float("inf"),
        )

    @torch.no_grad()
    def step(self, losses: torch.Tensor) -> float:
        if losses.ndim != 1:
            raise ValueError("loss tensor must be 1D")

        # Set invalid losses to infinity to avoid updating bests
        self.update_global_best(losses)
        valid_mask = self.validate_population(losses)
        losses = torch.where(valid_mask, losses, float("inf"))

        self._update_personal_bests(losses)
        self._update_particles()
        return self.global_best_loss

    def reset(self) -> float:
        """Reset the optimizer to initial state."""
        self.personal_best_loss.fill_(float("inf"))
        self.global_best_loss = float("inf")
        for variable in self.variables():
            variable.global_best.copy_(variable.population[0].detach().clone())
        return self.global_best_loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_personal_bests(self, losses: torch.Tensor) -> None:
        if self.personal_best_loss.device != losses.device:
            self.personal_best_loss = self.personal_best_loss.to(losses.device)

        better_mask = losses < self.personal_best_loss
        if not torch.any(better_mask):
            return

        self.personal_best_loss = torch.where(
            better_mask, losses, self.personal_best_loss
        )

        for variable in self.variables():
            mask = self._expand_mask(better_mask, variable.population)
            updated = torch.where(
                mask, variable.population.detach(), variable.personal_best
            )
            variable.personal_best.copy_(updated)

    def _update_particles(self) -> None:
        for name, variables in self.named_variables():
            r1 = torch.rand_like(variables.population)
            r2 = torch.rand_like(variables.population)
            cognitive = variables.personal_best - variables.population
            social_target = self._broadcast_best(name, variables.population)
            social = social_target - variables.population
            variables.velocity.mul_(self.inertia)
            variables.velocity.add_(self.cognitive * r1 * cognitive)
            variables.velocity.add_(self.social * r2 * social)
            variables.population.add_(variables.velocity)
            variables.clamp_to_bounds()

    @staticmethod
    def _expand_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        if mask.shape == like.shape:
            return mask
        if mask.ndim != 1:
            raise ValueError("mask must be 1D")
        view_shape = (mask.shape[0],) + (1,) * (like.ndim - 1)
        return mask.view(view_shape).expand_as(like)

    def _broadcast_best(self, name: str, like: torch.Tensor) -> torch.Tensor:
        best_value = self.model.get_variable(name).global_best
        value = best_value
        while value.dim() < like.dim():
            value = value.unsqueeze(0)
        return value.expand_as(like)
