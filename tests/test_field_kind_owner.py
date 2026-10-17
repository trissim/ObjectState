"""The state owns runtime containers; dataclass Field owns injection defaults."""
from dataclasses import MISSING, Field, dataclass, field
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


def test_container_kind_reads_canonical_stored_value():
    state = ObjectState(Configuration(), scope_id="p44-kinds")
    assert state.is_nested_container("nested")
    assert not state.is_nested_container("number")
    assert not state.is_nested_container("nullable")
    state.parameters["nested"] = None
    assert not state.is_nested_container("nested")


def test_container_kind_does_not_probe_instance_attributes():
    class Guarded(Nested):
        def __getattribute__(self, name):
            raise AssertionError("container classification resolved a payload attribute")

    state = ObjectState(Configuration(), scope_id="p44-no-resolution")
    state.parameters["nested"] = Guarded()
    assert state.is_nested_container("nested")


@pytest.mark.parametrize("optional", [False, True])
def test_injection_carries_the_declared_field(optional, monkeypatch):
    monkeypatch.setattr(lazy_factory, "_pending_injections", {})

    @dataclass
    class GlobalFieldKinds:
        pass

    @dataclass
    class Injected:
        count: int = 7

    decorate = lazy_factory.create_global_default_decorator(GlobalFieldKinds)
    decorate(optional=optional, inherit_as_none=False)(Injected)
    declaration, = lazy_factory._pending_injections["GlobalFieldKinds"]["configs_to_inject"]
    assert isinstance(declaration, Field)
    assert declaration.name == "injected"
    assert ObjectState._get_nested_dataclass_type(declaration.type) is Injected
    if optional:
        assert type(None) in get_args(declaration.type)
        assert declaration.default is None
        assert declaration.default_factory is MISSING
    else:
        assert declaration.type is Injected
        assert declaration.default is MISSING
        assert declaration.default_factory is Injected
