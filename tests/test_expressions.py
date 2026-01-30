import pytest
import torch

from popfit.core.expression import Expression, as_expression
from popfit.core.variable import Variable


@pytest.fixture
def scalar_var():
    return Variable(value=2.0, shape=())


@pytest.fixture
def vector_var():
    var = Variable(value=torch.tensor([1.0, 2.0]), shape=(2,))
    var.eval()
    return var


@pytest.mark.parametrize("value", [1, 1.5, -3])
def test_as_expression_wraps_scalars(value):
    expr = as_expression(value)
    assert isinstance(expr, Expression)


def test_as_expression_identity():
    x = as_expression(1.0)
    assert as_expression(x) is x


def test_expression_cannot_be_bool():
    x = as_expression(1.0)
    with pytest.raises(RuntimeError):
        bool(x)


def test_leaf_operands_empty():
    x = as_expression(1.0)
    assert list(x.operands()) == []


# ----------------------
# Arithmetic operators
# ----------------------


def test_add_scalar():
    expr = as_expression(2) + 3
    assert expr.value == 5


def test_sub_scalar():
    expr = as_expression(5) - 3
    assert expr.value == 2


def test_mul_scalar():
    expr = as_expression(4) * 2
    assert expr.value == 8


def test_div_scalar():
    expr = as_expression(8) / 2
    assert expr.value == 4


def test_pow_scalar():
    expr = as_expression(2) ** 3
    assert expr.value == 8


def test_add_tensor(vector_var):
    expr = vector_var + 1
    assert torch.allclose(expr.value, torch.tensor([2.0, 3.0]))


def test_mul_tensor(vector_var):
    expr = 2 * vector_var
    assert torch.allclose(expr.value, torch.tensor([2.0, 4.0]))


def test_div_tensor(vector_var):
    expr = vector_var / 2
    assert torch.allclose(expr.value, torch.tensor([0.5, 1.0]))


def test_negation():
    expr = -as_expression(3)
    assert expr.value == -3


def test_abs_positive():
    expr = abs(as_expression(torch.tensor([1, 2])))
    assert torch.allclose(expr.value, torch.tensor([1, 2]))


def test_abs_negative():
    expr = abs(as_expression(torch.tensor([-1, 2])))
    assert torch.allclose(expr.value, torch.tensor([1, 2]))


# ----------------------
# Comparisons
# ----------------------
def test_less_expression():
    expr = as_expression(1) < 2
    assert expr.value is True


def test_greater_expression_tensor(vector_var):
    expr = vector_var > 1
    assert torch.equal(expr.value, torch.tensor([False, True]))


# ----------------------
# Expression tree structure
# ----------------------


@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
        lambda a, b: a // b,
        lambda a, b: a**b,
        lambda a, b: a < b,
        lambda a, b: a <= b,
        lambda a, b: a > b,
        lambda a, b: a >= b,
    ],
    ids=["+", "-", "*", "/", "//", "**", "<", "<=", ">", ">="],
)
def test_binary_operands(op):
    a = as_expression(1)
    b = as_expression(2)
    expr: Expression = op(a, b)

    assert isinstance(expr, Expression)
    assert list(expr.operands()) == [a, b]


def test_neg_operands():
    a = as_expression(1)
    expr = -a

    ops = list(expr.operands())
    assert ops == [a]


# ----------------------
# Nested expressions
# ----------------------


def test_nested_expression_evaluation():
    expr = (as_expression(2) + 1) * (as_expression(3) - 1)
    assert expr.value == 6


# ----------------------
# Device propagation
# ----------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_expression_cuda():
    v = Variable(value=1.0, device="cuda")
    v.eval()
    expr = v + 1

    assert expr.value.device == v.global_best.device


def test_no_tensor_materialization_for_scalars():
    expr = as_expression(1) + 2
    assert isinstance(expr.value, (int, float, torch.Tensor))


def test_expression_is_not_boolean():
    expr = as_expression(1) < 2
    with pytest.raises(RuntimeError):
        if expr:  # type ignore
            pass
