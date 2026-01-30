import torch
import torch.nn as nn


class Parametrization(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_bounds(self, bounds: torch.Tensor) -> torch.Tensor:
        return bounds

    def inverse_bounds(self, bounds: torch.Tensor) -> torch.Tensor:
        return bounds
