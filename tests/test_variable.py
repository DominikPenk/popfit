# test_variable_full.py
import operator

import pytest
import torch

from popfit import Variable


# ----------------------
# Fixtures
# ----------------------
@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return request.param


@pytest.fixture
def variable1(device):
    pop = torch.tensor([1.0, 2.0, 3.0], device=device)
    return Variable(value=10.0, population=pop.clone(), device=device)


@pytest.fixture
def variable2(device):
    pop = torch.tensor([4.0, 5.0, 6.0], device=device)
    return Variable(value=20.0, population=pop.clone(), device=device)


@pytest.fixture
def scalar():
    return 2.0


@pytest.fixture
def tensor(device):
    return torch.tensor([7.0, 8.0, 9.0], device=device)


# ----------------------
# Operators
# ----------------------

binary_ops = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "truediv": lambda a, b: a / b,
    "pow": lambda a, b: a**b,
}

unary_ops = {
    "neg": operator.neg,
    "pos": operator.pos,
    "abs": abs,
}

inplace_ops = {
    "iadd": operator.iadd,
    "isub": operator.isub,
    "imul": operator.imul,
    "itruediv": operator.itruediv,
}

torch_funcs = [torch.sum, torch.exp, torch.prod, torch.min, torch.max]

# ----------------------
# Binary Operators
# ----------------------


@pytest.mark.parametrize(
    "op_func", list(binary_ops.values()), ids=list(binary_ops.keys())
)
@pytest.mark.parametrize("train_mode", [True, False], ids=["train", "eval"])
def test_binary_var_scalar(variable1, scalar, op_func, train_mode):
    variable1.train(train_mode)
    target = variable1.population if train_mode else variable1.global_best
    expected = op_func(target, scalar)
    result = op_func(variable1, scalar)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    "op_func", list(binary_ops.values()), ids=list(binary_ops.keys())
)
@pytest.mark.parametrize("train_mode", [True, False], ids=["train", "eval"])
def test_binary_scalar_var(variable1, scalar, op_func, train_mode):
    variable1.train(train_mode)
    target = variable1.population if train_mode else variable1.global_best
    expected = op_func(scalar, target)
    result = op_func(scalar, variable1)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    "op_func", list(binary_ops.values()), ids=list(binary_ops.keys())
)
@pytest.mark.parametrize("train_mode", [True, False], ids=["train", "eval"])
def test_binary_var_tensor(variable1, tensor, op_func, train_mode):
    variable1.train(train_mode)
    target = variable1.population if train_mode else variable1.global_best
    expected = op_func(target, tensor)
    result = op_func(variable1, tensor)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    "op_func", list(binary_ops.values()), ids=list(binary_ops.keys())
)
@pytest.mark.parametrize("train_mode", [True, False], ids=["train", "eval"])
def test_binary_tensor_var(variable1, tensor, op_func, train_mode):
    variable1.train(train_mode)
    target = variable1.population if train_mode else variable1.global_best
    expected = op_func(tensor, target)
    result = op_func(tensor, variable1)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    "op_func", list(binary_ops.values()), ids=list(binary_ops.keys())
)
@pytest.mark.parametrize("train_mode", [True, False], ids=["train", "eval"])
def test_binary_var_var(variable1, variable2, op_func, train_mode):
    variable1.train(train_mode)
    variable2.train(train_mode)
    target1 = variable1.population if train_mode else variable1.global_best
    target2 = variable2.population if train_mode else variable2.global_best
    expected = op_func(target1, target2)
    result = op_func(variable1, variable2)
    assert torch.allclose(result, expected)


# ----------------------
# Unary Operators
# ----------------------


@pytest.mark.parametrize(
    "op_func", list(unary_ops.values()), ids=list(unary_ops.keys())
)
@pytest.mark.parametrize("train_mode", [True, False], ids=["train", "eval"])
def test_unary(variable1, op_func, train_mode):
    variable1.train(train_mode)
    expected = op_func(Variable._unwrap(variable1))
    result = op_func(variable1)
    assert torch.allclose(result, expected)


# ----------------------
# In-place Operators
# ----------------------


@pytest.mark.parametrize(
    "op_func", list(inplace_ops.values()), ids=list(inplace_ops.keys())
)
@pytest.mark.parametrize("train_mode", [True, False], ids=["train", "eval"])
def test_inplace_scalar(variable1, scalar, op_func, train_mode):
    with torch.no_grad():
        variable1.train(train_mode)
        t = Variable._unwrap(variable1).clone()
        op_func(variable1, scalar)
        op_func(t, scalar)
        assert torch.allclose(variable1, t)


@pytest.mark.parametrize(
    "op_func", list(inplace_ops.values()), ids=list(inplace_ops.keys())
)
def test_inplace_tensor(variable1, tensor, op_func):
    with torch.no_grad():
        variable1.train()
        t = Variable._unwrap(variable1).clone()
        op_func(variable1, tensor)
        op_func(t, tensor)
        assert torch.allclose(variable1, t)


@pytest.mark.parametrize(
    "op_func", list(inplace_ops.values()), ids=list(inplace_ops.keys())
)
def test_inplace_op_raises_without_no_grad_in_train(variable1, scalar, op_func):
    variable1.train()
    with pytest.raises(RuntimeError):
        op_func(variable1, scalar)


# ----------------------
# Indexing / Slicing
# ----------------------


def test_indexing_train(variable1):
    variable1.train()
    pop = variable1.population
    assert variable1[0].item() == pop[0].item()
    assert variable1[1:].tolist() == pop[1:].tolist()


def test_indexing_eval(variable1):
    variable1.eval()
    g = variable1.global_best
    # Should broadcast
    assert variable1.data.item() == g.item()


# ----------------------
# PyTorch functions
# ----------------------


@pytest.mark.parametrize("func", torch_funcs)
def test_torch_funcs_train(variable1, func):
    variable1.train()
    assert torch.allclose(func(variable1), func(variable1.population))


@pytest.mark.parametrize("func", torch_funcs)
def test_torch_funcs_eval(variable1, func):
    variable1.eval()
    g = variable1.global_best
    assert torch.allclose(func(variable1), func(g))


# ----------------------
# Boolean safety
# ----------------------


def test_bool_error(variable1):
    with pytest.raises(RuntimeError):
        bool(variable1)


# ----------------------
# Comparison Operators
# ----------------------


def test_bounds_initialization_none():
    var = Variable(value=1.0)
    lower, upper = var.bounds
    assert lower == float("-inf")
    assert upper == float("inf")
    assert var.lower_bound == float("-inf")
    assert var.upper_bound == float("inf")


def test_bounds_initialization_tuple():
    var = Variable(bounds=(0.0, 1.0))
    assert torch.allclose(var.bounds, torch.as_tensor([0.0, 1.0]))
    assert var.lower_bound == 0.0
    assert var.upper_bound == 1.0


def test_bounds_setter_updates_bounds():
    var = Variable(bounds=(0.0, 1.0))
    assert torch.allclose(var.bounds, torch.as_tensor([0.0, 1.0]))
    var.bounds = (1.0, 2.0)
    assert torch.allclose(var.bounds, torch.as_tensor([1.0, 2.0]))
    assert var.lower_bound == 1.0
    assert var.upper_bound == 2.0


def test_bounds_setter_none():
    var = Variable(bounds=(0.0, 1.0))
    assert torch.allclose(var.bounds, torch.as_tensor([0.0, 1.0]))
    var.bounds = None
    assert torch.allclose(var.bounds, torch.as_tensor([float("-inf"), float("inf")]))
    assert var.lower_bound == float("-inf")
    assert var.upper_bound == float("inf")


def test_bounds_setter_invalid_length():
    var = Variable(shape=())
    with pytest.raises(ValueError):
        var.bounds = (0.0,)


def test_bounds_setter_low_higher_than_high():
    var = Variable(shape=(1, 2))
    with pytest.raises(ValueError):
        var.bounds = (2.0, 1.0)


def test_lower_bound_setter():
    var = Variable(bounds=(0.0, 1.0))
    assert torch.allclose(var.bounds, torch.as_tensor([0.0, 1.0]))
    var.lower_bound = 0.5
    assert var.lower_bound == 0.5
    assert torch.allclose(var.bounds, torch.as_tensor([0.5, 1.0]))


def test_upper_bound_setter():
    var = Variable(bounds=(0.0, 1.0))
    assert torch.allclose(var.bounds, torch.as_tensor([0.0, 1.0]))
    var.upper_bound = 2.0
    assert var.upper_bound == 2.0
    assert torch.allclose(var.bounds, torch.as_tensor([0.0, 2.0]))
