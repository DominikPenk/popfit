from __future__ import annotations

from functools import partialmethod
from typing import Any, Iterable, Optional, Sequence, overload

import torch
import torch.nn as nn


class Variable(nn.Module):
    def __init__(
        self,
        value: Optional[float | torch.Tensor] = None,
        bounds: Optional[Sequence[Any] | torch.Tensor] = None,
        *,
        shape: Optional[torch.Size | tuple[int, ...]] = None,
        population: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        dtype = dtype or torch.get_default_dtype()
        device = device or torch.get_default_device()

        inferred_shapes: dict[str, torch.Size] = {}

        if shape is not None:
            inferred_shapes["shape"] = torch.Size(shape)
        if value is not None:
            inferred_shapes["value"] = torch.as_tensor(value).shape
        if population is not None:
            inferred_shapes["population"] = population.shape[1:]
        if bounds is not None:
            b_tensor = torch.as_tensor(bounds)
            inferred_shapes["bounds"] = b_tensor.shape[1:]

        if not inferred_shapes:
            raise ValueError(
                "Could not infer variable shape at least one of value, shape, population, or bounds must be provided"
            )

        main_source, self._var_shape = next(iter(inferred_shapes.items()))
        for source, shape in inferred_shapes.items():
            if shape != self._var_shape:
                raise ValueError(
                    f"Shape of {source} does not match {main_source}. {shape} vs. {self._var_shape}"
                )

        # Set bounds
        if bounds is not None:
            b_tensor = torch.as_tensor(bounds, dtype=dtype, device=device)
            if b_tensor.shape == (2, *self._var_shape):
                low, high = b_tensor[0], b_tensor[1]
            else:
                # Try to broadcast (e.g., user passed tuple of two floats or two tensors)
                low = torch.as_tensor(bounds[0], dtype=dtype, device=device).expand(
                    self._var_shape
                )
                high = torch.as_tensor(bounds[1], dtype=dtype, device=device).expand(
                    self._var_shape
                )

            if torch.any(low > high):
                raise ValueError(
                    "Upper bound must be greater than or equal to lower bound for all elements."
                )
        else:
            low = torch.full(self._var_shape, float("-inf"), dtype=dtype, device=device)
            high = torch.full(self._var_shape, float("inf"), dtype=dtype, device=device)
        self._bounds = nn.Buffer(torch.stack([low, high], dim=0))

        if value is not None:
            value_tensor = torch.as_tensor(value, dtype=dtype, device=device)
        else:
            value_tensor = self.get_domain_center()

        pop_tensor = (
            torch.as_tensor(population, dtype=dtype, device=device)
            if population is not None
            else torch.empty((0, *self._var_shape), dtype=dtype, device=device)
        )

        self.global_best = nn.Buffer(value_tensor)
        self.population = nn.Parameter(pop_tensor, requires_grad=requires_grad)

        self.clamp_to_bounds(clamp_best=True)

        self._initialized = True

    def get_domain_center(self):
        mid = 0.5 * self._bounds.sum(dim=0)
        mid = torch.where(torch.isfinite(mid), mid, torch.zeros_like(mid))
        mid = mid.clamp_(min=self._bounds[0], max=self._bounds[1])
        return mid

    @classmethod
    def from_base_variable(
        cls: type[Variable],
        base_variable: Variable,
    ) -> Variable:
        if cls is not Variable:
            raise NotImplementedError(
                "from_base_variable must be implemented by subclasses."
            )
        return base_variable

    def to_base_variable(self) -> Variable:
        if type(self) is not Variable:
            raise NotImplementedError(
                "to_base_variable must be implemented by subclasses."
            )
        return self

    @overload
    def sample_uniform(self, *, mask: torch.Tensor) -> None: ...

    @overload
    def sample_uniform(self, *, num_samples: int) -> None: ...

    def sample_uniform(
        self, *, num_samples: Optional[int] = None, mask: Optional[torch.Tensor] = None
    ) -> None:
        if num_samples is None and mask is None:
            raise RuntimeError("At least one of num_samples and mask must be None")

        num_samples = num_samples if num_samples is not None else self.population_size

        low, high = self._bounds[0], self._bounds[1]
        dist = torch.distributions.Uniform(low, high)
        samples = dist.sample((num_samples,))

        if mask is not None:
            self.population.data[mask] = samples[mask]
        else:
            self.population.data = samples

    def empty(self, num_samples: int) -> None:
        self.population.data = torch.empty(
            (num_samples, *self._var_shape), dtype=self.dtype, device=self.device
        )

    def clamp_to_bounds(self, clamp_best: bool = False) -> None:
        self.population.data.clamp_(min=self._bounds[0], max=self._bounds[1])
        if clamp_best:
            self.global_best.clamp_(min=self._bounds[0], max=self._bounds[1])

    @classmethod
    def _unwrap(cls, obj) -> torch.Tensor:
        if isinstance(obj, Variable):
            return obj.population if obj.training else obj.global_best
        else:
            return obj

    # --------------------------------------------------------------
    # Operators
    # --------------------------------------------------------------

    def __bool__(self):
        raise RuntimeError("PopParameter cannot be used as a boolean value")

    def __getitem__(self, index: Any) -> torch.Tensor:
        return Variable._unwrap(self)[index]

    def _binary_op(self, op: str, other: Any):
        self_val = Variable._unwrap(self)
        other_val = Variable._unwrap(other)
        return getattr(self_val, op)(other_val)

    def _iop(self, op: str, other: Any):
        if self.training:
            getattr(self.population, op)(Variable._unwrap(other))
        else:
            getattr(self.global_best, op)(Variable._unwrap(other))
        return self

    __add__ = partialmethod(_binary_op, "__add__")
    __sub__ = partialmethod(_binary_op, "__sub__")
    __mul__ = partialmethod(_binary_op, "__mul__")
    __truediv__ = partialmethod(_binary_op, "__truediv__")
    __floordiv__ = partialmethod(_binary_op, "__floordiv__")
    __pow__ = partialmethod(_binary_op, "__pow__")
    __matmul__ = partialmethod(_binary_op, "__matmul__")
    __mod__ = partialmethod(_binary_op, "__mod__")

    __iadd__ = partialmethod(_iop, "__iadd__")
    __isub__ = partialmethod(_iop, "__isub__")
    __imul__ = partialmethod(_iop, "__imul__")
    __itruediv__ = partialmethod(_iop, "__itruediv__")
    __ifloordiv__ = partialmethod(_iop, "__ifloordiv__")
    __ipow__ = partialmethod(_iop, "__ipow__")
    __imatmul__ = partialmethod(_iop, "__imatmul__")
    __imod__ = partialmethod(_iop, "__imod__")

    def __radd__(self, other: Any) -> torch.Tensor:
        other_val = Variable._unwrap(other)
        return other_val + Variable._unwrap(self)

    def __rsub__(self, other: Any) -> torch.Tensor:
        other_val = Variable._unwrap(other)
        return other_val - Variable._unwrap(self)

    def __rmul__(self, other: Any) -> torch.Tensor:
        other_val = Variable._unwrap(other)
        return other_val * Variable._unwrap(self)

    def __rtruediv__(self, other: Any) -> torch.Tensor:
        other_val = Variable._unwrap(other)
        return other_val / Variable._unwrap(self)

    def __rfloordiv__(self, other: Any) -> torch.Tensor:
        other_val = Variable._unwrap(other)
        return other_val // Variable._unwrap(self)

    def __rpow__(self, other: Any) -> torch.Tensor:
        other_val = Variable._unwrap(other)
        return other_val ** Variable._unwrap(self)

    def __rmod__(self, other: Any) -> torch.Tensor:
        other_val = Variable._unwrap(other)
        return other_val % Variable._unwrap(self)

    def __neg__(self) -> torch.Tensor:
        return -Variable._unwrap(self)

    def __pos__(self) -> torch.Tensor:
        return Variable._unwrap(self)

    def __abs__(self) -> torch.Tensor:
        return abs(Variable._unwrap(self))

    # --------------------------------------------------------------
    # Torch function override
    # --------------------------------------------------------------

    def __torch_function__(
        self,
        func,
        types: Iterable[type],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        if kwargs is None:
            kwargs = {}

        args = tuple(Variable._unwrap(a) for a in args)
        kwargs = {k: Variable._unwrap(v) for k, v in kwargs.items()}

        return func(*args, **kwargs)

    # --------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @bounds.setter
    def bounds(self, value: Optional[Sequence[Any]]) -> None:
        if value is None:
            self._bounds[0].fill_(float("-inf"))
            self._bounds[1].fill_(float("inf"))
            return

        bounds_tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if bounds_tensor.shape != (2, *self._var_shape):
            raise ValueError(
                f"Expected bound of shape {2, *self._var_shape} got {bounds_tensor.shape}"
            )
        self._bounds = bounds_tensor

    @property
    def lower_bound(self) -> torch.Tensor:
        return self._bounds[0]

    @lower_bound.setter
    def lower_bound(self, value: Optional[float | torch.Tensor]) -> None:
        value = value if value is not None else float("-inf")
        value = torch.as_tensor(
            value, device=self._bounds.device, dtype=self._bounds.dtype
        )
        value = value.expand_as(self._bounds[0])
        self._bounds[0] = value

    @property
    def upper_bound(self) -> torch.Tensor:
        return self._bounds[1]

    @upper_bound.setter
    def upper_bound(self, value: Optional[float | torch.Tensor]) -> None:
        value = value if value is not None else float("inf")
        value = torch.as_tensor(
            value, device=self._bounds.device, dtype=self._bounds.dtype
        )
        value = value.expand_as(self._bounds[0])
        self._bounds[1] = value

    @property
    def population_size(self) -> int:
        return self.population.shape[0]

    @property
    def var_shape(self) -> torch.Size:
        return self._var_shape

    @property
    def optimal(self) -> torch.Tensor:
        return self.global_best

    @property
    def device(self) -> torch.device:
        return self.population.device

    @property
    def dtype(self) -> torch.dtype:
        return self.population.dtype

    @property
    def data(self) -> torch.Tensor:
        if self.training:
            return self.population
        else:
            return self.global_best
