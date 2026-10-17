"""Flat topology owns containers; typed injections own native field declarations."""
from dataclasses import MISSING, FrozenInstanceError, dataclass, field
from typing import Optional, get_args

import pytest

from objectstate import ObjectState
from objectstate import lazy_factory


@dataclass
class Nested:
    count: int = 3


@dataclass
class Configuration:
    nested: Nested = field(default_factory=Nested)
    number: int = 5
    nullable: Optional[int] = None


def test_container_kind_reads_owned_child_parameters():
    state = ObjectState(Configuration(), scope_id="p44-kinds")
    assert state.is_nested_container("nested")
    assert not state.is_nested_container("number")
    assert not state.is_nested_container("nullable")
    state.parameters["nested"] = None
    assert state.is_nested_container("nested")
    state.parameters["number"] = Nested()
    assert not state.is_nested_container("number")


def test_container_kind_does_not_probe_instance_attributes():
    class Guarded(Nested):
        def __getattribute__(self, name):
            raise AssertionError("container classification resolved a payload attribute")

    state = ObjectState(Configuration(), scope_id="p44-no-resolution")
    state.parameters["nested"] = Guarded()
    assert state.is_nested_container("nested")


@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("ui_hidden", [False, True])
def test_injection_carries_the_declared_field(optional, ui_hidden, monkeypatch):
    monkeypatch.setattr(lazy_factory, "_pending_injections", {})

    @dataclass
    class GlobalFieldKinds:
        pass

    @dataclass
    class Injected:
        count: int = 7

    decorate = lazy_factory.create_global_default_decorator(GlobalFieldKinds)
    decorate(optional=optional, inherit_as_none=False, ui_hidden=ui_hidden)(Injected)
    declaration, = lazy_factory._pending_injections[GlobalFieldKinds]
    assert declaration.config_class is Injected
    assert declaration.field_name == "injected"
    assert declaration.nullable is optional
    name, annotation, native_field = declaration.field_tuple()
    assert name == "injected"
    assert ObjectState._get_nested_dataclass_type(annotation) is Injected
    assert native_field.metadata == {"ui_hidden": ui_hidden}
    with pytest.raises(FrozenInstanceError):
        declaration.nullable = not optional
    if optional:
        assert type(None) in get_args(annotation)
        assert native_field.default is None
        assert native_field.default_factory is MISSING
    else:
        assert annotation is Injected
        assert native_field.default is MISSING
        assert native_field.default_factory is Injected


def test_empty_dataclass_without_children_is_a_snapshot_leaf():
    @dataclass
    class Empty:
        pass

    @dataclass
    class EmptyConfiguration:
        empty: Empty = field(default_factory=Empty)

    state = ObjectState(EmptyConfiguration(), scope_id="p44-empty-leaf")
    assert list(state.parameters) == ["empty"]
    assert not state.is_nested_container("empty")
    assert state.get_resolved_value("empty") == Empty()
