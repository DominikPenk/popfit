import pytest
import torch
import torch.nn as nn

from popfit import Model, Variable


@pytest.fixture
def model():
    return Model()


@pytest.fixture
def var():
    return Variable(value=10.0, population=torch.tensor([1.0, 2.0, 3.0]))


class SimpleModel(Model):
    def __init__(self):
        super().__init__()
        self.a = Variable(value=1.0)
        self.b = Variable(value=2.0)


class ComplexModel(Model):
    def __init__(self):
        super().__init__()
        self.curve1 = SimpleModel()
        self.curve2 = SimpleModel()


## Registration Tests
def test_register_variable_success(model, var):
    model.register_variable("test_var", var)
    assert model.test_var is var


def test_register_variable_wrong_type(model):
    with pytest.raises(TypeError):
        model.register_variable("bad_var", nn.Linear(10, 10))


## Retrieval Tests
def test_get_variable_success(model, var):
    model.register_variable("test_var", var)
    retrieved = model.get_variable("test_var")
    assert retrieved is var


def test_get_nested_variable_success(model, var):
    model = ComplexModel()
    retrieved = model.get_variable("curve1.a")
    assert retrieved is model.curve1.a


def test_get_variable_not_found(model):
    with pytest.raises(AttributeError):  # PyTorch get_submodule raises AttributeError
        model.get_variable("non_existent")


def test_get_variable_wrong_type(model):
    model.add_module("not_a_var", nn.ReLU())
    with pytest.raises(TypeError):
        model.get_variable("not_a_var")


## Iterator Tests
def test_named_variables_iteration(model):
    v1, v2 = Variable(shape=()), Variable(shape=())
    model.register_variable("v1", v1)
    model.register_variable("v2", v2)

    vars_dict = dict(model.named_variables())
    assert len(vars_dict) == 2
    assert vars_dict["v1"] is v1


def test_variables_iterator(model):
    v1 = Variable(shape=())
    model.register_variable("v1", v1)
    assert next(model.variables()) is v1


## Replacement Tests
def test_replace_variable_shallow(model):
    old_var = Variable(shape=())
    new_var = Variable(shape=())
    model.register_variable("v1", old_var)

    returned_old = model.replace_variable("v1", new_var)
    assert returned_old is old_var
    assert model.v1 is new_var


def test_replace_variable_nested(model):
    # Setup nested structure: model -> sub (nn.Module) -> target (Variable)
    sub = nn.Sequential()
    old_var = Variable(shape=())
    new_var = Variable(shape=())
    sub.add_module("target", old_var)
    model.add_module("sub", sub)

    model.replace_variable("sub.target", new_var)
    assert model.sub.target is new_var


def test_replace_variable_errors(model, var):
    model.add_module("not_a_var", nn.Linear(1, 1))

    # Test replacing something that isn't a Variable
    with pytest.raises(TypeError, match="is not a Variable"):
        model.replace_variable("not_a_var", var)

    # Test providing a non-Variable replacement
    model.register_variable("real_var", Variable(shape=()))
    with pytest.raises(TypeError, match="must be an instance of Variable"):
        model.replace_variable("real_var", nn.ReLU())


def test_recursive_model():
    complex_model = ComplexModel()
    variables = dict(complex_model.named_variables())

    assert "curve1.a" in variables
    assert "curve1.b" in variables
    assert "curve2.a" in variables
    assert "curve2.b" in variables

    assert variables["curve1.a"] is complex_model.curve1.a
    assert variables["curve2.b"] is complex_model.curve2.b
    assert variables["curve1.a"] is complex_model.curve1.a
    assert variables["curve1.b"] is complex_model.curve1.b
