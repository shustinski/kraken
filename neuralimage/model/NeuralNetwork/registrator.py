import enum
import inspect
from typing import Dict, Type, TypedDict

import torch.nn as nn


class ModelType(enum.Enum):
    stable = 'stable'
    deprecated = 'deprecated'
    experimental = 'experimental'


class ModelRegistryEntry(TypedDict):
    model_class: Type[nn.Module]
    model_type: ModelType


_MODEL_REGISTRY: Dict[str, ModelRegistryEntry] = {}


def _normalize_model_type(value: ModelType | str | None) -> ModelType:
    if isinstance(value, ModelType):
        return value
    normalized = str(value or ModelType.stable.value).strip().lower()
    for model_type in ModelType:
        if normalized == model_type.value:
            return model_type
    raise ValueError(f'Неизвестный тип модели: {value!r}')


def register_model(
    name: str | None = None,
    *,
    model_type: ModelType | str = ModelType.stable,
):
    """Register a ``torch.nn.Module`` subclass in the global model registry."""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if not issubclass(cls, nn.Module):
            raise TypeError('Только подклассы torch.nn.Module могут быть зарегистрированы')
        model_name = name or cls.__name__
        if model_name in _MODEL_REGISTRY:
            raise KeyError(f'Модель с именем {model_name!r} уже зарегистрирована')
        _MODEL_REGISTRY[model_name] = {
            'model_class': cls,
            'model_type': _normalize_model_type(model_type),
        }
        return cls

    return decorator


def get_registered_models() -> Dict[str, Type[nn.Module]]:
    """Return a name -> model class mapping."""
    return {
        name: entry['model_class']
        for name, entry in _MODEL_REGISTRY.items()
    }


def get_registered_model_registry() -> Dict[str, ModelRegistryEntry]:
    """Return a copy of the full registry metadata."""
    return {
        name: {
            'model_class': entry['model_class'],
            'model_type': entry['model_type'],
        }
        for name, entry in _MODEL_REGISTRY.items()
    }


def get_registered_model_names_by_type() -> dict[ModelType, list[str]]:
    grouped: dict[ModelType, list[str]] = {model_type: [] for model_type in ModelType}
    for model_name, entry in _MODEL_REGISTRY.items():
        grouped[entry['model_type']].append(model_name)
    return grouped


def model_supports_init_kwarg(name: str, kwarg: str) -> bool:
    try:
        model_cls = _MODEL_REGISTRY[str(name)]['model_class']
    except KeyError:
        return False
    try:
        signature = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return False
    return str(kwarg) in signature.parameters


def create_model(name: str, *args, **kwargs) -> nn.Module:
    """Create a registered model instance by its registry name."""
    try:
        model_cls = _MODEL_REGISTRY[name]['model_class']
    except KeyError as exc:
        raise ValueError(f'Неизвестная модель {name!r}. Доступные: {list(_MODEL_REGISTRY)}') from exc
    return model_cls(*args, **kwargs)
