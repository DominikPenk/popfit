import pytest
import torch

from popfit.core.variable import Variable
from popfit.parametrization.sigmoid_bounded import SigmoidBounded


@pytest.fixture
def bounds():
    return torch.tensor([[0.0, -1.0], [1.0, 1.0]])


@pytest.fixture
def variable(bounds):
    return Variable(
        value=torch.as_tensor([0.0, 0.3]),
        population=torch.as_tensor([[1.0, 0.5], [0.0, 0.5]]),
        bounds=bounds,
    )


# ----------------------
# Sigmoid-bounded parametrization
# ----------------------


def test_sigmoid_bounded_forward_inverse(bounds):
    z = torch.tensor([[0.0, 1.0], [-1.0, 0.5]])
    param = SigmoidBounded(bounds=bounds)

    x = param.forward(z)
    z_recovered = param.inverse(x)

    # Forward followed by inverse returns original z
    assert torch.allclose(z, z_recovered)


def test_sigmoid_bounded_updates_population_and_global_best(variable: Variable):
    param = SigmoidBounded(bounds=variable.bounds)

    o_pop = variable.population.detach().clone()
    o_opt = variable.global_best.detach().clone()

    z_pop = param.inverse(variable.population)
    z_opt = param.inverse(variable.global_best)

    variable.push_parametrization(param)

    assert torch.allclose(z_pop, variable.population)
    assert torch.allclose(z_opt, variable.global_best)

    param = variable.pop_parametrization()

    assert torch.allclose(o_pop, variable.population, atol=1e-6)
    assert torch.allclose(o_opt, variable.global_best, atol=1e-6)


def test_sigmoid_bounded_does_not_change_model_value(variable: Variable):
    param = SigmoidBounded(bounds=variable.bounds)

    o_pop = variable.population.detach().clone()
    o_opt = variable.global_best.detach().clone()

    variable.push_parametrization(param)

    variable.train()
    assert torch.allclose(variable.value, o_pop, atol=1e-6)
    variable.eval()
    assert torch.allclose(variable.value, o_opt, atol=1e-6)

    param = variable.pop_parametrization()

    variable.train()
    assert torch.allclose(variable.value, o_pop, atol=1e-6)
    variable.eval()
    assert torch.allclose(variable.value, o_opt, atol=1e-6)


def test_forward_bounds_inverse_bounds():
    bounds = torch.tensor([[0.0, -1.0], [1.0, 1.0]])
    param = SigmoidBounded(bounds=bounds)

    forward_bounds = param.forward_bounds(bounds)
    inverse_bounds = param.inverse_bounds(bounds)

    # SigmoidBounded leaves forward_bounds unchanged
    assert torch.all(forward_bounds == bounds)

    # Inverse bounds should be unbounded
    assert torch.all(inverse_bounds[0] == float("-inf"))
    assert torch.all(inverse_bounds[1] == float("inf"))


def test_sigmoid_bounded_updates_variable_bounds(variable: Variable):
    param = SigmoidBounded(bounds=variable.bounds)

    variable.push_parametrization(param)

    assert torch.all(variable.latent_bounds[0] == float("-inf"))
    assert torch.all(variable.latent_bounds[1] == float("inf"))


def test_sigmoid_bounded_does_not_change_model_bounds(
    variable: Variable, bounds: torch.Tensor
):
    param = SigmoidBounded(bounds=variable.bounds)

    variable.push_parametrization(param)

    assert torch.allclose(variable.bounds, bounds)
