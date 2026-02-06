from typing import Optional

import torch

from popfit.core.variable import Variable


def populate_variable(
    variable: Variable,
    population: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    global_best: Optional[torch.Tensor] = None,
    check_shape: bool = True,
) -> None:
    dtype = variable.dtype
    device = variable.device

    if check_shape and population.shape[1:] != variable.shape:
        raise RuntimeError(
            f"Expected population of shape (n,) + {variable.shape}, got {population.shape}"
        )

    if check_shape and global_best is not None and global_best.shape != variable.shape:
        raise RuntimeError(
            f"Expected global_best of shape {variable.shape}, got {global_best.shape}"
        )

    if mask is not None and population.shape != variable.population.shape:
        raise RuntimeError(
            "Cannot modify variable population shape when mask is not None"
        )

    if mask is not None and mask.shape != population.shape[0:1]:
        raise RuntimeError(
            f"Expected mask of shape ({population.shape[0]},), got {mask.shape}"
        )

    population = torch.as_tensor(population, dtype=dtype, device=device)
    if mask is not None:
        population[~mask] = variable.population[~mask]

    if global_best is not None:
        global_best = torch.as_tensor(global_best, device=device, dtype=dtype)

    with torch.no_grad():
        if mask is None and population.shape != variable.population.shape:
            requires_grad = variable.population.requires_grad
            variable.population = torch.nn.Parameter(
                population, requires_grad=requires_grad
            )
        else:
            variable.population.copy_(population)
        if global_best is not None:
            variable.global_best.copy_(global_best)


def empty_population(
    variable: Variable,
    *,
    num_samples: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """Creates an empty population tensor of the given shape and device.

    Args:
        variable (Variable): The Variable object containing bounds.
        num_samples (int): Number of samples to initialize.
    """
    num_samples = num_samples or variable.population_size
    shape = (num_samples,) + variable.shape
    populate_variable(variable, torch.empty(shape), mask=mask)


def uniform_population(
    variable: Variable,
    *,
    num_samples: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """Samples uniform values within the bounds of a Variable's population.

    Args:
        variable (Variable): The Variable object containing bounds.
        num_samples (Optional[int], optional): Number of samples to generate. If None, uses default.

    Returns:
        torch.Tensor: Sampled tensor.
    """
    if not torch.all(torch.isfinite(variable.latent_bounds)):
        raise ValueError("uniform_population only supported for bounded variables")
    num_samples = num_samples or variable.population_size
    dist = torch.distributions.Uniform(variable.lower_bound, variable.upper_bound)
    populate_variable(
        variable,
        dist.sample((num_samples,)),
        mask=mask,
        check_shape=num_samples is None,
    )


def normal_population(
    variable: Variable,
    *,
    num_samples: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
) -> None:
    if not torch.all(torch.isfinite(variable.latent_bounds)):
        raise ValueError("normal_population only supported for bounded variables")
    num_samples = num_samples or variable.population_size
    low, high = variable.latent_bounds.unbind(dim=0)
    loc = 0.5 * (low + high)
    scale = (high - low) / 6.0
    dist = torch.distributions.Normal(loc, scale)

    population = dist.sample((num_samples,))

    invalid = (population < low) | (population > high)

    while invalid.any():
        replacements = dist.sample((num_samples,))
        population[invalid] = replacements[invalid]
        invalid = (population < low) | (population > high)

    populate_variable(variable, population, mask=mask, check_shape=num_samples is None)
