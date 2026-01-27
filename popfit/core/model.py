from typing import Iterator, Optional

import torch.nn as nn

from .variable import Variable


def _reset_population(module: nn.Module, size: int) -> None:
    for child in module.children():
        if hasattr(child, "reset_population"):
            child.reset_population(size)  # type: ignore
        elif isinstance(child, Variable):
            child.sample_uniform(num_samples=size)
        else:
            _reset_population(child, size)


class Model(nn.Module):
    def add_variable(self, name: str, variable: Optional[Variable]) -> None:
        """Register a Variable as a submodule of the model.

        Args:
            name: The name of the Variable.
            variable: The Variable to register.
        """
        if not isinstance(variable, Variable):
            raise TypeError("variable must be an instance of Variable.")
        self.add_module(name, variable)

    def named_variables(
        self,
        memo: set[Variable] | None = None,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Variable]]:
        """Iterator over model Variables, yielding both the name and the Variable.

        Yields:
            An iterator over (name, Variable) pairs.
        """
        if memo is None:
            memo = set()

        for name, module in self.named_modules():
            if not isinstance(module, Variable):
                continue
            if remove_duplicate and module in memo:
                continue
            memo.add(module)
            yield name, module

    def variables(self) -> Iterator[Variable]:
        """Iterator over model Variables.

        Yields:
            An iterator over Variables.
        """
        for _, variable in self.named_variables():
            yield variable

    def get_variable(self, name: str) -> Variable:
        """Get a Variable by name.

        Args:
            name: The name of the Variable to retrieve.

        Returns:
            The Variable with the specified name.

        Raises:
            KeyError: If no Variable with the specified name exists.
        """
        variable = self.get_submodule(name)
        if not isinstance(variable, Variable):
            raise TypeError(f"The module '{name}' is not a Variable.")
        return variable

    def replace_variable(self, name: str, new_variable: Variable) -> Variable:
        """Replace a Variable in the model with a new Variable.

        Args:
            name: The name of the Variable to replace.
            new_variable: The new Variable to insert.

        Raises:
            KeyError: If no Variable with the specified name exists.
        """
        if not isinstance(new_variable, Variable):
            raise TypeError("new_variable must be an instance of Variable.")

        if "." in name:
            parent_name, leaf_name = name.rsplit(".", 1)
            parent = self.get_submodule(parent_name)
        else:
            leaf_name = name
            parent = self

        old = getattr(parent, leaf_name)
        if not isinstance(old, Variable):
            raise TypeError(f"The module '{name}' is not a Variable.")

        setattr(parent, leaf_name, new_variable)
        return old

    def reset_population(self, size: int) -> None:
        _reset_population(self, size)
