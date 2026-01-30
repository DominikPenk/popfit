from typing import Literal

import torch

from ..core import Model, Spec, Variable
from .base import Optimizer


class PSOSpec(Spec):
    def __init__(self, variable: Variable) -> None:
        velocity = torch.zeros_like(variable.population)
        personal_best = variable.population.detach().clone()
        super().__init__(velocity=velocity, personal_best=personal_best)


class PSO(Optimizer):
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

    def start_optimization(self) -> None:
        super().start_optimization()
        for variable in self.model.variables():
            variable.spec += PSOSpec(variable)

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

    def finalize_optimization(self) -> None:
        for variable in self.model.variables():
            vel = variable.spec.pop("velocity")
            pb = variable.spec.pop("personal_best")
            if not isinstance(vel, torch.Tensor):
                raise RuntimeError(
                    f"Expected spec member 'velocity' to be a torch.Tensor, got {type(vel)}"
                )
            if not isinstance(pb, torch.Tensor):
                raise RuntimeError(
                    f"Expected spec member 'personal_best' to be a torch.Tensor, got {type(pb)}"
                )

    def reset(self) -> float:
        """Reset the optimizer to initial state."""
        self.personal_best_loss.fill_(float("inf"))
        self.global_best_loss = float("inf")
        for variable in self.model.variables():
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

        for variable in self.model.variables():
            mask = self._expand_mask(better_mask, variable.population)
            variable.spec["personal_best"] = torch.where(
                mask, variable.population.detach(), variable.spec["personal_best"]
            )

    def _update_particles(self) -> None:
        for name, variables in self.model.named_variables():
            r1 = torch.rand_like(variables.population)
            r2 = torch.rand_like(variables.population)
            cognitive = variables.spec["personal_best"] - variables.population
            social_target = self._broadcast_best(name, variables.population)
            social = social_target - variables.population
            variables.spec["velocity"].mul_(self.inertia)
            variables.spec["velocity"].add_(self.cognitive * r1 * cognitive)
            variables.spec["velocity"].add_(self.social * r2 * social)
            variables.population.add_(variables.spec["velocity"])
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
