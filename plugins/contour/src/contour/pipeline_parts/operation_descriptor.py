"""pipeline: operation descriptor."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ..application.processing import OperationParameterSpec
from contour.i18n import choice_label, operation_name, parameter_label, tr

from contour.pipeline_parts.module_state import _OPERATIONS

@dataclass(frozen=True, slots=True)
class OperationDescriptor:
    type_name: str
    display_name: str
    parameters: tuple[OperationParameterSpec, ...]
    handler: OperationCallable
    public: bool = True

    def default_parameters(self) -> dict[str, Any]:
        return {spec.name: spec.default for spec in self.parameters}

def register_operation(descriptor: OperationDescriptor) -> None:
    _OPERATIONS[descriptor.type_name] = descriptor

def get_operation_descriptor(operation_name: str) -> OperationDescriptor:
    try:
        return _OPERATIONS[operation_name]
    except KeyError as exc:
        raise KeyError(tr("unsupported_pipeline_operation", operation=operation_name)) from exc

def available_operations() -> list[OperationDescriptor]:
    return [descriptor for descriptor in _OPERATIONS.values() if descriptor.public]

def get_operation_display_name(operation_key: str, language: str | None = None) -> str:
    descriptor = get_operation_descriptor(operation_key)
    return operation_name(descriptor.type_name, default=descriptor.display_name, language=language)

def get_parameter_display_label(spec: OperationParameterSpec, language: str | None = None) -> str:
    return parameter_label(spec.name, default=spec.label, language=language)

def get_choice_display_label(parameter_name: str, value: str, language: str | None = None) -> str:
    return choice_label(parameter_name, value, default=value, language=language)
