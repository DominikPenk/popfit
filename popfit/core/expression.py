from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn


class Expression(nn.Module):
    @property
    def value(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def is_leaf(self) -> bool:
        raise NotImplementedError

    @property
    def label(self) -> str:
        raise NotImplementedError

    def operands(self) -> Iterator[Expression]:
        return iter(())

    def walk(self) -> Iterator[Expression]:
        yield self
        for operand in self.operands():
            yield from operand.walk()

    def walk_postorder(self) -> Iterator[Expression]:
        for operand in self.operands():
            yield from operand.walk_postorder()
        yield self

    def leaves(self) -> Iterator[Expression]:
        for node in self.walk():
            if node.is_leaf:
                yield node

    def print_tree(self, prefix: str = "", is_last: bool = True):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + self.label)

        prefix += "    " if is_last else "│   "
        operands = list(self.operands())
        for i, child in enumerate(operands):
            child.print_tree(prefix, i == len(operands) - 1)

    def __bool__(self):
        raise RuntimeError(
            "Expression cannot be used as a boolean. "
            "Use .value or explicit reductions instead."
        )

    # -----------------
    # Addition
    # -----------------
    def __add__(self, other) -> Expression:
        return AddExpression(self, as_expression(other))

    def __radd__(self, other) -> Expression:
        return AddExpression(as_expression(other), self)

    # -----------------
    # Subtraction
    # -----------------
    def __sub__(self, other) -> Expression:
        return SubExpression(self, as_expression(other))

    def __rsub__(self, other) -> Expression:
        return SubExpression(as_expression(other), self)

    # -----------------
    # Multiplication
    # -----------------
    def __mul__(self, other) -> Expression:
        return MulExpression(self, as_expression(other))

    def __rmul__(self, other) -> Expression:
        return MulExpression(as_expression(other), self)

    # -----------------
    # Division
    # -----------------
    def __truediv__(self, other) -> Expression:
        return TrueDivExpression(self, as_expression(other))

    def __rtruediv__(self, other) -> Expression:
        return TrueDivExpression(as_expression(other), self)

    def __floordiv__(self, other) -> Expression:
        return DivExpression(self, as_expression(other))

    def __rfloordiv__(self, other) -> Expression:
        return DivExpression(as_expression(other), self)

    # -----------------
    # Comparison
    # -----------------
    def __lt__(self, other) -> Expression:
        return LessExpression(self, as_expression(other))

    def __le__(self, other) -> Expression:
        return LessEqualExpression(self, as_expression(other))

    def __gt__(self, other) -> Expression:
        return GreaterExpression(self, as_expression(other))

    def __ge__(self, other) -> Expression:
        return GreaterEqualExpression(self, as_expression(other))

    # -----------------
    # Power
    # -----------------
    def __pow__(self, other) -> Expression:
        return PowExpression(self, as_expression(other))

    def __rpow__(self, other) -> Expression:
        return PowExpression(as_expression(other), self)

    # -----------------
    # Unary ops
    # -----------------
    def __neg__(self) -> Expression:
        return NegExpression(self)

    def __pos__(self) -> Expression:
        return self

    def __abs__(self) -> Expression:
        return AbsExpression(self)


class ConstantExpression(Expression):
    def __init__(self, value: torch.Tensor | float | int):
        self.value_ = value

    @property
    def is_leaf(self) -> bool:
        return True

    @property
    def value(self):
        return self.value_

    @property
    def label(self):
        return str(self.value_)


def as_expression(x: Expression | torch.Tensor | float | int) -> Expression:
    if isinstance(x, Expression):
        return x
    return ConstantExpression(x)


class BinaryExpression(Expression):
    def __init__(self, a: Expression, b: Expression):
        super().__init__()
        self.a = a
        self.b = b

    @property
    def is_leaf(self) -> bool:
        return False

    def operands(self) -> Iterator[Expression]:
        yield self.a
        yield self.b


class AddExpression(BinaryExpression):
    @property
    def value(self) -> torch.Tensor:
        return self.a.value + self.b.value

    def label(self) -> str:
        return "+"


class SubExpression(BinaryExpression):
    @property
    def value(self) -> torch.Tensor:
        return self.a.value - self.b.value

    def label(self) -> str:
        return "-"


class MulExpression(BinaryExpression):
    @property
    def value(self) -> torch.Tensor:
        return self.a.value * self.b.value

    def label(self) -> str:
        return "*"


class TrueDivExpression(BinaryExpression):
    @property
    def value(self) -> torch.Tensor:
        return self.a.value / self.b.value

    def label(self) -> str:
        return "/"


class DivExpression(BinaryExpression):
    @property
    def value(self) -> torch.Tensor:
        return self.a.value // self.b.value

    def label(self) -> str:
        return "//"


class LessExpression(BinaryExpression):
    @property
    def value(self) -> torch.Tensor:
        return self.a.value < self.b.value

    def label(self) -> str:
        return "<"


class LessEqualExpression(BinaryExpression):
    @property
    def value(self) -> torch.Tensor:
        return self.a.value <= self.b.value

    def label(self) -> str:
        return "<="


class GreaterExpression(BinaryExpression):
    @property
    def value(self) -> torch.Tensor:
        return self.a.value > self.b.value

    def label(self) -> str:
        return ">"


class GreaterEqualExpression(BinaryExpression):
    @property
    def value(self) -> torch.Tensor:
        return self.a.value >= self.b.value

    def label(self) -> str:
        return ">="


class PowExpression(BinaryExpression):
    @property
    def value(self) -> torch.Tensor:
        return self.a.value**self.b.value

    def label(self) -> str:
        return "^"


class NegExpression(Expression):
    def __init__(self, a: Expression):
        super().__init__()
        self.a = a

    @property
    def value(self) -> torch.Tensor:
        return -self.a.value

    @property
    def is_leaf(self) -> bool:
        return False

    def label(self) -> str:
        return "-"

    def operands(self) -> Iterator[Expression]:
        yield self.a


class AbsExpression(Expression):
    def __init__(self, a: Expression):
        super().__init__()
        self.a = a

    @property
    def value(self) -> torch.Tensor:
        return abs(self.a.value)

    @property
    def is_leaf(self) -> bool:
        return False

    def label(self) -> str:
        return "abs"

    def operands(self) -> Iterator[Expression]:
        yield self.a
