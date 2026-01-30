from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .variable import Variable


class Spec:
    def __init__(self, **meta: Any) -> None:
        self.meta = dict(meta)

    def __getitem__(self, key: str) -> Any:
        return self.meta[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.meta[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove a metadata key."""
        del self.meta[key]

    def __repr__(self) -> str:
        return f"<Spec {self.meta!r}>"

    def __bool__(self) -> bool:
        return bool(self.meta)

    def pop(self, key: str, default: Any = None) -> Any:
        """Remove a key and return its value; return default if missing."""
        return self.meta.pop(key, default)

    def format(self, variable: Variable) -> str:
        return self.default_format(variable)

    def default_format(self, variable: Variable) -> str:
        label = self.meta.get("label") or self.meta.get("name") or "Variable"
        units = self.meta.get("units")
        unit = f" [{units}]" if units else ""
        value = variable.optimal if not variable.training else variable.population
        return f"{label}{unit}: {value}"

    def __add__(self, other: Spec) -> Spec:
        if not isinstance(other, Spec):
            return NotImplemented
        return CompositeSpec(self, other)


class CompositeSpec(Spec):
    def __init__(self, *specs: Spec):
        self.specs: tuple[Spec, ...] = tuple(
            s
            for spec in specs
            for s in (spec.specs if isinstance(spec, CompositeSpec) else (spec,))
        )

        meta = {}
        for spec in self.specs:
            if any(k in meta for k in spec.meta.keys()):
                warnings.warn("Duplicate keys will be overwritten by latest value")
            meta.update(spec.meta)

        super().__init__(**meta)

    def format(self, variable: Variable) -> str:
        # last formatter wins
        for spec in reversed(self.specs):
            if spec.format is not Spec.format:
                return spec.format(variable)
        return super().format(variable)


class UnitSpec(Spec):
    def __init__(self, units: str, name: Optional[str] = None, **meta):
        if name:
            meta["name"] = name
        super().__init__(units=units, **meta)
