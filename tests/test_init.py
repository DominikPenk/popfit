import pytest
import torch

from popfit.core.init import (
    empty_population,
    normal_population,
    populate_variable,
    uniform_population,
)
from popfit.core.variable import Variable


@pytest.fixture
def scalar_variable():
    return Variable(
        value=0.0,
        bounds=(-1.0, 1.0),
        population=torch.zeros((5,)),
    )


@pytest.fixture
def vector_variable():
    return Variable(
        value=torch.zeros(3),
        bounds=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        population=torch.zeros((4, 3)),
    )


@pytest.fixture
def unbounded_variable():
    return Variable(
        value=0.0,
        population=torch.zeros((6,)),
    )


def test_populate_variable_replaces_population(vector_variable):
    new_pop = torch.randn(4, 3)
    populate_variable(vector_variable, new_pop)

    assert torch.allclose(vector_variable.population, new_pop)


def test_populate_variable_shape_mismatch_raises(vector_variable):
    bad_pop = torch.randn(4, 4)

    with pytest.raises(RuntimeError, match="Expected population of shape"):
        populate_variable(vector_variable, bad_pop)


def test_populate_variable_global_best_shape_check(vector_variable):
    bad_best = torch.randn(4)

    with pytest.raises(RuntimeError, match="Expected global_best of shape"):
        populate_variable(
            vector_variable,
            vector_variable.population,
            global_best=bad_best,
        )


def test_populate_variable_mask_shape_check(vector_variable):
    mask = torch.tensor([True, False])  # wrong length

    with pytest.raises(RuntimeError, match="Expected mask of shape"):
        populate_variable(
            vector_variable,
            vector_variable.population,
            mask=mask,
        )


def test_populate_variable_masked_update(vector_variable):
    original = vector_variable.population.clone()

    new_pop = torch.ones_like(original)
    mask = torch.tensor([True, False, True, False])

    populate_variable(vector_variable, new_pop, mask=mask)

    assert torch.allclose(
        vector_variable.population[mask],
        new_pop[mask],
    )
    assert torch.allclose(
        vector_variable.population[~mask],
        original[~mask],
    )


def test_populate_variable_mask_disallows_shape_change(vector_variable):
    new_pop = torch.randn(10, 3)
    mask = torch.tensor([True, False, True, False])

    with pytest.raises(RuntimeError, match="Cannot modify variable population shape"):
        populate_variable(vector_variable, new_pop, mask=mask, check_shape=False)


def test_populate_variable_updates_global_best(vector_variable):
    best = torch.tensor([0.1, 0.2, 0.3])

    populate_variable(
        vector_variable,
        vector_variable.population,
        global_best=best,
    )

    assert torch.allclose(vector_variable.global_best, best)


def test_empty_population_creates_empty_tensor(vector_variable):
    empty_population(vector_variable, num_samples=7)

    assert vector_variable.population.shape == (7, 3)
    assert vector_variable.population.dtype == vector_variable.dtype
    assert vector_variable.population.device == vector_variable.device


def test_empty_population_with_mask(vector_variable):
    original = vector_variable.population.clone()
    mask = torch.tensor([True, False, True, False])

    empty_population(vector_variable, mask=mask)

    assert vector_variable.population.shape == original.shape
    assert torch.allclose(vector_variable.population[~mask], original[~mask])


def test_uniform_population_requires_bounded_variable(unbounded_variable):
    with pytest.raises(ValueError, match="only supported for bounded variables"):
        uniform_population(unbounded_variable)


def test_uniform_population_within_bounds(vector_variable):
    uniform_population(vector_variable, num_samples=100)

    pop = vector_variable.population
    assert torch.all(pop >= vector_variable.lower_bound)
    assert torch.all(pop <= vector_variable.upper_bound)


def test_normal_population_requires_bounded_variable(unbounded_variable):
    with pytest.raises(ValueError, match="only supported for bounded variables"):
        normal_population(unbounded_variable)


def test_normal_population_respects_bounds(vector_variable):
    normal_population(vector_variable, num_samples=200)

    pop = vector_variable.population
    assert torch.all(pop >= vector_variable.lower_bound)
    assert torch.all(pop <= vector_variable.upper_bound)


def test_normal_population_centered(vector_variable):
    normal_population(vector_variable, num_samples=500)

    mean = vector_variable.population.mean(dim=0)
    center = 0.5 * (vector_variable.lower_bound + vector_variable.upper_bound)

    assert torch.allclose(mean, center, atol=0.2)


def test_normal_population_masked(vector_variable):
    original = vector_variable.population.clone()
    mask = torch.tensor([True, False, True, False])

    normal_population(vector_variable, mask=mask)

    assert not torch.allclose(
        vector_variable.population[mask],
        original[mask],
    )
    assert torch.allclose(
        vector_variable.population[~mask],
        original[~mask],
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_population_preserves_dtype(dtype):
    var = Variable(
        value=0.0,
        bounds=(-1.0, 1.0),
        shape=(),
        population=torch.zeros((4,), dtype=dtype),
        dtype=dtype,
    )

    uniform_population(var)

    assert var.population.dtype == dtype
