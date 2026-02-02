import math
from typing import Literal, NamedTuple

import torch

from ..core import Model, Spec, Variable
from .base import Optimizer

_MEAN = "cma_mean"
_C = "cma_c"
_SIGMA = "cma_sigma"
_PC = "cma_pc"
_PS = "cma_ps"
_GEN = "cma_generation"


class CMAState(NamedTuple):
    mean: torch.Tensor
    C: torch.Tensor
    sigma: torch.Tensor
    pc: torch.Tensor
    ps: torch.Tensor


def _update_distribution(
    *,
    mean: torch.Tensor,
    C: torch.Tensor,
    sigma: torch.Tensor,
    pc: torch.Tensor,
    ps: torch.Tensor,
    X: torch.Tensor,
    weights: torch.Tensor,
    mu_eff: float,
) -> CMAState:
    d = mean.numel()

    # --- mean update ---
    new_mean = torch.sum(weights[:, None] * X, dim=0)
    y = (new_mean - mean) / sigma

    # --- strategy parameters ---
    c_sigma = (mu_eff + 2.0) / (d + mu_eff + 5.0)
    c_c = (4.0 + mu_eff / d) / (d + 4.0 + 2.0 * mu_eff / d)

    # --- p_sigma ---
    eps = 1e-12
    L = torch.linalg.cholesky(C + eps * torch.eye(d, device=C.device, dtype=C.dtype))

    C_inv_sqrt_y = torch.linalg.solve_triangular(
        L, y.unsqueeze(1), upper=False
    ).squeeze(1)

    ps = (1.0 - c_sigma) * ps + math.sqrt(
        c_sigma * (2.0 - c_sigma) * mu_eff
    ) * C_inv_sqrt_y

    # --- p_c ---
    pc = (1.0 - c_c) * pc + math.sqrt(c_c * (2.0 - c_c) * mu_eff) * y

    # --- covariance ---
    artmp = (X - mean) / sigma

    c1 = 2.0 / ((d + 1.3) ** 2 + mu_eff)
    c_mu = min(
        1.0 - c1,
        2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((d + 2.0) ** 2 + mu_eff),
    )

    rank_mu = torch.sum(
        weights[:, None, None] * torch.einsum("ni,nj->nij", artmp, artmp),
        dim=0,
    )

    C = (1.0 - c1 - c_mu) * C + c1 * torch.outer(pc, pc) + c_mu * rank_mu
    C = 0.5 * (C + C.T)

    # --- step size ---
    chi_n = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (d + 1.0)) - 1.0)

    sigma = sigma * torch.exp(
        (c_sigma / d_sigma) * (torch.norm(ps) / chi_n - 1.0)
    ).clamp(1e-12, 1e3)

    return CMAState(
        mean=new_mean,
        C=C,
        sigma=sigma,
        pc=pc,
        ps=ps,
    )


def _sample_distribution(
    size: int,
    mean: torch.Tensor,
    C: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    # Flatten mean to match covariance dimension
    mean_flat = mean.flatten()
    d = mean_flat.numel()

    # Numerical safety for Cholesky
    eps = 1e-12
    L = torch.linalg.cholesky(C + eps * torch.eye(d, device=C.device, dtype=C.dtype))

    # Sample standard normal
    z = torch.randn(
        (size, d),
        device=mean.device,
        dtype=mean.dtype,
    )

    # Transform samples
    y = z @ L.T  # (λ, d)
    x = mean_flat + sigma * y  # (λ, d)

    # Reshape back to variable shape
    x = x.view(size, *mean.shape)
    return x


class CMASpec(Spec):
    """CMA-ES specific state for a single optimization variable.

    This class stores all internal CMA-ES parameters for a variable, including
    the mean vector, covariance matrix, step-size, evolution paths, and generation counter.

    Args:
        variable (Variable): The variable to optimize.
        sigma (float, optional): Initial step-size. Defaults to 0.3.
    """

    def __init__(self, variable: Variable, *, sigma: float = 0.3):
        _, *shape = variable.population.shape
        dim = int(torch.tensor(shape).prod())

        device = variable.device
        dtype = variable.dtype

        mean = torch.empty_like(variable.global_best)

        super().__init__(
            **{
                _MEAN: mean,
                _C: torch.eye(dim, device=device, dtype=dtype),
                _SIGMA: torch.as_tensor(sigma, device=device, dtype=dtype),
                _PC: torch.zeros(dim, device=device, dtype=dtype),
                _PS: torch.zeros(dim, device=device, dtype=dtype),
                _GEN: 0,
            }
        )


class BlockCMAES(Optimizer):
    """Block-wise CMA-ES optimizer for population-based gradient-free optimization.

    This optimizer implements the CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    algorithm in a block-wise fashion, maintaining separate CMA states per variable.
    It can handle invalid evaluations by resampling or ignoring them.

    Args:
        model (Model): The model containing Variables to optimize.
        population_size (int, optional): Number of candidate solutions per generation. Default is 32.
        sigma (float, optional): Initial global step-size. Default is 0.3.
        invalid_handling (Literal["resample", "ignore"], optional): How to handle invalid solutions.
            Defaults to "ignore".

    Attributes:
        lambda_ (int): Population size used internally (same as `population_size`).
        mu (int): Number of elite individuals used to update mean.
        weights (torch.Tensor): Recombination weights for elite individuals.
        mu_eff (float): Effective number of parents.
        sigma0 (float): Initial step-size.

    References:
        - Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in
          evolution strategies. Evolutionary computation, 9(2), 159-195.
    """

    def __init__(
        self,
        model: Model,
        *,
        population_size: int = 32,
        sigma: float = 0.3,
        invalid_handling: Literal["resample", "ignore"] = "ignore",
    ) -> None:
        super().__init__(
            model,
            population_size=population_size,
            invalid_handling=invalid_handling,
        )

        self.lambda_ = population_size
        self.mu = population_size // 2

        weights = torch.log(torch.arange(1, self.mu + 1, dtype=torch.float32))
        self.weights = weights.max() - weights
        self.weights /= self.weights.sum()

        self.mu_eff = 1.0 / torch.sum(self.weights**2).item()

        self.sigma0 = sigma

    def start_optimization(self) -> None:
        """Initialize the optimizer state for all Variables.

        Adds a CMASpec to each Variable storing its CMA-ES parameters.
        """
        super().start_optimization()
        device: torch.device | None = None

        for variable in self.model.variables():
            variable.spec += CMASpec(variable, sigma=self.sigma0)
            if device is None:
                device = variable.device
        self.weights = self.weights.to(device)

    @torch.no_grad()
    def step(self, losses: torch.Tensor) -> float:
        """Perform one CMA-ES generation step.

        This method updates the internal CMA state (mean, covariance, step-size)
        and samples a new population for the next generation.

        Args:
            losses (torch.Tensor): Tensor of shape `(population_size,)` containing the
                evaluation losses of the current population.

        Returns:
            float: Best loss observed after this step.
        """
        # Set invalid losses to infinity to avoid updating bests
        valid_mask = self.validate_population(losses)
        losses = torch.where(valid_mask, losses, float("inf"))
        self.update_global_best(losses)

        order = torch.argsort(losses)
        elites = order[: self.mu]

        for variable in self.model.variables():
            if variable.spec[_GEN] == 0:
                variable.spec[_MEAN] = variable.global_best
            self._update_variable(variable, elites)
            self._sample_variable(variable)
        return self.global_best_loss

    def finalize_optimization(self) -> None:
        """Clean up internal CMA-ES state from all Variables.

        Removes all CMASpec entries (_MEAN, _C, _SIGMA, _PC, _PS, _GEN) from each Variable.
        """
        for variable in self.model.variables():
            variable.spec.pop(_MEAN)
            variable.spec.pop(_C)
            variable.spec.pop(_SIGMA)
            variable.spec.pop(_PC)
            variable.spec.pop(_PS)
            variable.spec.pop(_GEN)

    @torch.no_grad()
    def _update_variable(self, variable: Variable, elites: torch.Tensor) -> None:
        """Update the CMA-ES internal state for a single Variable.

        Args:
            variable (Variable): The Variable to update.
            elites (torch.Tensor): Indices of elite individuals in the population.
        """
        spec = variable.spec

        mean = spec[_MEAN].flatten()
        C = spec[_C]
        sigma = spec[_SIGMA]
        pc = spec[_PC]
        ps = spec[_PS]

        d = mean.numel()
        X = variable.population[elites].reshape(self.mu, d)

        state = _update_distribution(
            mean=mean,
            C=C,
            sigma=sigma,
            pc=pc,
            ps=ps,
            X=X,
            weights=self.weights,
            mu_eff=self.mu_eff,
        )

        spec[_MEAN] = state.mean.view_as(spec[_MEAN])
        spec[_C] = state.C
        spec[_SIGMA] = state.sigma
        spec[_PC] = state.pc
        spec[_PS] = state.ps
        spec[_GEN] += 1

    @torch.no_grad()
    def _sample_variable(self, variable: Variable) -> None:
        """Sample a new population for a Variable using its CMA-ES parameters.

        Args:
            variable (Variable): The Variable to sample new candidates for.
        """
        spec = variable.spec

        x = _sample_distribution(
            size=self.population_size, mean=spec[_MEAN], C=spec[_C], sigma=spec[_SIGMA]
        )

        variable.population.copy_(x)
        variable.clamp_to_bounds()
