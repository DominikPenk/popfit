from typing import Literal, Self

import torch

from ..core import Model, init


class Optimizer:
    def __init__(
        self,
        model: Model,
        population_size: int,
        *,
        invalid_handling: Literal["resample", "ignore"] = "ignore",
    ) -> None:
        self.model = model
        self.population_size = population_size
        self.global_best_loss = float("inf")
        self.invalid_handling = invalid_handling

    def __enter__(self) -> Self:
        self.start_optimization()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finalize_optimization()

    def start_optimization(self) -> None:
        self.model.reset_population(self.population_size)

    def finalize_optimization(self) -> None:
        pass

    def step(self, losses: torch.Tensor) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    # --------------------------------------------------------------
    # Step utilities
    # --------------------------------------------------------------

    def get_valid_mask(self, losses: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(losses)

    def validate_population(self, losses: torch.Tensor) -> torch.Tensor:
        if losses.shape != (self.population_size,):
            raise ValueError(
                f"Expected losses to have shape ({self.population_size},), got {losses.shape} instead."
            )

        valid_mask = self.get_valid_mask(losses)
        if torch.all(valid_mask):
            return valid_mask

        if self.invalid_handling == "resample":
            invalid_mask = ~valid_mask
            for variable in self.model.variables():
                init.uniform_population(variable, mask=invalid_mask)
        return valid_mask

    def update_global_best(self, losses: torch.Tensor) -> None:
        if losses.shape != (self.population_size,):
            raise ValueError(
                f"Expected losses to have shape ({self.population_size},), got {losses.shape} instead."
            )

        save_losses = torch.where(torch.isfinite(losses), losses, float("inf"))
        current_best_loss, current_idx = torch.min(save_losses, dim=0)
        if current_best_loss.item() < self.global_best_loss:
            self.global_best_loss = current_best_loss.item()
            for variable in self.model.variables():
                best_value = variable.population[current_idx].detach().clone()
                variable.global_best.copy_(best_value)
