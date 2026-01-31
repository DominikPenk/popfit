import math
from typing import Optional

import torch
import torch.nn as nn

from ..core import Model, Variable, init


class Linear(Model):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Variable(
            shape=(out_features, in_features), device=device, dtype=dtype
        )

        if bias:
            self.bias = Variable(shape=(out_features,), device=device, dtype=dtype)
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def reset_population(self, size: int) -> None:
        init.empty_population(self.weight, num_samples=size)
        if self.bias is not None:
            init.empty_population(self.bias, num_samples=size)

        bound = math.sqrt(6.0 / (self.in_features + self.out_features))
        nn.init.uniform_(self.weight.population, -bound, bound)

        if self.bias is not None:
            bound = 1 / (self.in_features**0.5) if self.in_features > 0 else 0
            nn.init.uniform_(self.bias.population.data, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return nn.functional.linear(
                x,
                self.weight.global_best,
                self.bias.global_best if self.bias is not None else None,
            )

        if x.ndim == 2:
            x = x.unsqueeze(0).expand(self.weight.population_size, -1, -1)

        out = torch.bmm(x, self.weight.population.transpose(1, 2))

        if self.bias is not None:
            out = out + self.bias.population.unsqueeze(1)

        return out
