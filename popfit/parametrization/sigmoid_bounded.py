from typing import Optional

import torch
import torch.nn as nn

from .base import Parametrization


class SigmoidBounded(Parametrization):
    def __init__(self, bounds: torch.Tensor, eps: Optional[float] = None) -> None:
        super().__init__()
        self.mask = nn.Buffer(torch.all(torch.isfinite(bounds), dim=0))
        self.bounds = nn.Buffer(bounds.detach().clone())
        self._any = torch.any(self.mask)
        self.eps = eps or torch.finfo(bounds.dtype).eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self._any:
            return z

        safe_lower = torch.where(self.mask, self.bounds[0], 0.0)
        safe_upper = torch.where(self.mask, self.bounds[1], 1.0)
        val = safe_lower + (safe_upper - safe_lower) * torch.sigmoid(z)
        return torch.where(self.mask, val, z)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.any(self.mask):
            return x

        safe_lower = torch.where(self.mask, self.bounds[0], 0.0)
        safe_upper = torch.where(self.mask, self.bounds[1], 1.0)
        val = (x - safe_lower) / (safe_upper - safe_lower)
        val = torch.clamp(val, self.eps, 1.0 - self.eps)
        val = torch.logit(val)
        return torch.where(self.mask, val, x)

    def forward_bounds(self, bounds: torch.Tensor) -> torch.Tensor:
        return self.bounds

    def inverse_bounds(self, bounds: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                torch.full_like(self.bounds[0], float("-inf")),
                torch.full_like(self.bounds[1], float("inf")),
            ]
        )
