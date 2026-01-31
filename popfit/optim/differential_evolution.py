from typing import Literal

import torch

from ..core import Model
from .base import Optimizer


class DifferentialEvolution(Optimizer):
    """
    Differential Evolution optimizer (DE/rand/1/bin).

    Args:
        model: Model containing Variables
        population_size: Number of individuals
        F: Differential weight (mutation scale)
        CR: Crossover probability
    """

    def __init__(
        self,
        model: Model,
        *,
        population_size: int = 200,
        F: float = 0.8,
        CR: float = 0.9,
        invalid_handling: Literal["resample", "ignore"] = "ignore",
    ) -> None:
        super().__init__(
            model, population_size=population_size, invalid_handling=invalid_handling
        )
        self.F = F
        self.CR = CR

    @torch.no_grad()
    def step(self, losses: torch.Tensor) -> float:
        if losses.ndim != 1:
            raise ValueError("loss tensor must be 1D")

        self.update_global_best(losses)
        valid_mask = self.validate_population(losses)
        losses = torch.where(valid_mask, losses, float("inf"))

        # Create trial population
        self._mutate_and_crossover()

        return self.global_best_loss

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _mutate_and_crossover(self) -> None:
        for variable in self.model.variables():
            pop = variable.population

            N = pop.shape[0]
            device = pop.device

            # --- Mutation indices ---
            ia, ib, ic = DifferentialEvolution._draw_indices(N, device)

            trial = pop[ia] + self.F * (pop[ib] - pop[ic])

            # --- Binomial crossover ---
            pop_view = pop.view(N, -1)
            cross_mask = torch.rand_like(pop_view) < self.CR

            # Ensure at least one dimension crosses
            D = pop_view.shape[1]
            j_rand = torch.randint(0, D, (N,), device=device)
            cross_mask[torch.arange(N), j_rand] = True

            cross_mask = cross_mask.view(pop.shape)

            trial = torch.where(cross_mask, trial, pop)
            variable.population.copy_(trial)
            variable.clamp_to_bounds()

    @staticmethod
    def _draw_indices(N: int, device: torch.device):
        ia = torch.randint(1, N, (N,), device=device)
        ib = torch.randint(1, N, (N,), device=device)
        ic = torch.randint(1, N, (N,), device=device)

        i = torch.arange(N, device=device)

        ia = torch.where(ia == i, (ia + 1) % N, ia)
        ib = torch.where((ib == i) | (ib == ia), (ib + 2) % N, ib)
        ic = torch.where((ic == i) | (ic == ia) | (ic == ib), (ic + 3) % N, ic)

        return ia, ib, ic
