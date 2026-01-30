# test_spec.py
import warnings

from popfit.core.spec import CompositeSpec, Spec, UnitSpec


# Dummy Variable class for testing purposes
class DummyVariable:
    def __init__(self, training=False, optimal=42, population=99):
        self.training = training
        self.optimal = optimal
        self.population = population


# -------------------------------
# Spec Tests
# -------------------------------


def test_spec_initialization_and_repr():
    meta = {"label": "TestVar", "units": "m"}
    spec = Spec(**meta)
    assert spec.meta == meta
    assert repr(spec) == f"<Spec {meta!r}>"


def test_spec_default_format_training_and_optimal():
    var = DummyVariable(training=False, optimal=10, population=20)
    spec = Spec(label="X", units="m")
    result = spec.default_format(var)
    assert result == "X [m]: 10"

    var.training = True
    result = spec.default_format(var)
    assert result == "X [m]: 20"


def test_spec_format_uses_default_by_default():
    var = DummyVariable()
    spec = Spec(label="A")
    assert spec.format(var) == spec.default_format(var)


def test_spec_bool_conversion():
    assert not Spec()
    assert Spec(label="A")


# -------------------------------
# CompositeSpec Tests
# -------------------------------


def test_composite_spec_flattening_and_meta_merge():
    s1 = Spec(label="A", units="m")
    s2 = Spec(label="B", description="second")
    composite = CompositeSpec(s1, s2)

    # Flattened specs tuple
    assert composite.specs == (s1, s2)

    # Meta should merge, last key wins
    assert composite.meta["label"] == "B"
    assert composite.meta["units"] == "m"
    assert composite.meta["description"] == "second"


def test_composite_spec_duplicate_warning():
    s1 = Spec(label="A")
    s2 = Spec(label="B")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CompositeSpec(s1, s2)
        assert any("Duplicate keys" in str(warning.message) for warning in w)


def test_composite_spec_format_customization():
    class CustomSpec(Spec):
        def format(self, variable):
            return "custom"

    s1 = Spec(label="A")
    s2 = CustomSpec(label="B")
    composite = CompositeSpec(s1, s2)
    var = DummyVariable()
    # Last custom formatter should be used
    assert composite.format(var) == "custom"


def test_composite_spec_nested_flattening():
    s1 = Spec(label="A")
    s2 = Spec(label="B")
    nested = CompositeSpec(s1, CompositeSpec(s2))
    # Should flatten to (s1, s2)
    assert nested.specs == (s1, s2)


# -------------------------------
# UnitSpec Tests
# -------------------------------


def test_unitspec_metadata_and_format():
    uspec = UnitSpec("m", name="Length", description="test variable")
    assert uspec.meta["units"] == "m"
    assert uspec.meta["name"] == "Length"
    assert uspec.meta["description"] == "test variable"

    var = DummyVariable()
    result = uspec.format(var)
    assert "Length [m]:" in result


def test_unitspec_name_none():
    uspec = UnitSpec("kg")
    assert "name" not in uspec.meta or uspec.meta.get("name") is None


# -------------------------------
# Operator Tests
# -------------------------------


def test_spec_add_operator_creates_composite():
    s1 = Spec(label="A")
    s2 = Spec(label="B")
    composite = s1 + s2
    assert isinstance(composite, CompositeSpec)
    assert composite.specs == (s1, s2)


# -------------------------------
# Edge Cases
# -------------------------------


def test_empty_meta_spec():
    spec = Spec()
    assert spec.meta == {}
    var = DummyVariable()
    # Should default label
    assert spec.default_format(var).startswith("Variable")


def test_composite_spec_all_defaults():
    s1 = Spec()
    s2 = Spec()
    composite = CompositeSpec(s1, s2)
    var = DummyVariable()
    # Format falls back to Spec default
    assert composite.format(var).startswith("Variable")
