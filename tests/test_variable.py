# test_variable_full.py

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


# ----------------------
# Boolean safety
# ----------------------


def test_bool_error(variable1):
    with pytest.raises(RuntimeError):
        bool(variable1)


# ----------------------
# Bounds
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


# ----------------------
# Value
# ----------------------


def test_value_is_global_best_in_eval():
    var = Variable(value=10, population=torch.as_tensor([0.0, 1.0]))
    var.eval()
    assert var.value.shape == ()
    assert var.value.item() == pytest.approx(10)

    assert var.latent_value.shape == ()
    assert var.value.item() == pytest.approx(10)


def test_value_is_population_in_train():
    pop = torch.as_tensor([0.0, 1.0])
    var = Variable(value=10, population=pop)
    var.train()
    assert torch.allclose(pop, var.value)
    assert torch.allclose(pop, var.latent_value)
