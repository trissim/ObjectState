"""Field behavior selected once from its annotation, not from its current payload."""
from dataclasses import dataclass, field, is_dataclass
from typing import Union, get_args
import re

from python_introspect import UnifiedParameterAnalyzer


def camel_to_snake(name):
    first = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', first).lower()


@dataclass(frozen=True)
class FieldDeclaration:
    annotation: object

    @staticmethod
    def of(annotation):
        if type(None) in get_args(annotation):
            return NullableField(annotation)
        return ConfigurationField(annotation) if is_dataclass(annotation) else FieldDeclaration(annotation)

    @classmethod
    def for_parameter(cls, target, path):
        declaration = cls.of(target)
        for name in path.split('.'):
            try:
                parameter = UnifiedParameterAnalyzer.analyze(declaration.root_type)[name]
            except KeyError as error:
                raise AttributeError(name) from error
            declaration = cls.of(parameter.param_type)
        return declaration

    @property
    def root_type(self):
        return self.annotation

    def leaf_paths(self, path):
        return (path,)

    def record_change(self, registry, path, scope):
        for leaf in self.leaf_paths(path):
            registry.record_snapshot(f'edit {leaf}', scope)

    def read(self, state, path):
        return state.get_resolved_value(path)

    def injection(self, name, hidden):
        return name, self.annotation, field(default_factory=self.annotation, metadata={'ui_hidden': hidden})


class ConfigurationField(FieldDeclaration):
    def leaf_paths(self, path):
        return ()

    def read(self, state, path):
        from objectstate.object_state import FieldProxy
        return FieldProxy(state, path, self.annotation)


class NullableField(FieldDeclaration):
    @property
    def inner(self):
        return FieldDeclaration.of(Union[tuple(t for t in get_args(self.annotation) if t is not type(None))])

    @property
    def root_type(self):
        return self.inner.root_type

    def leaf_paths(self, path):
        return self.inner.leaf_paths(path)

    def read(self, state, path):
        return self.inner.read(state, path)

    def injection(self, name, hidden):
        return name, self.annotation, field(default=None, metadata={'ui_hidden': hidden})


@dataclass(frozen=True)
class ConfigInjection:
    declaration: FieldDeclaration
    ui_hidden: bool

    @property
    def config_class(self):
        return self.declaration.root_type

    def definition(self):
        return self.declaration.injection(camel_to_snake(self.config_class.__name__), self.ui_hidden)
