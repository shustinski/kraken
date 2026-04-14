import enum
import inspect
from typing import Callable, Dict, Type, TypedDict

import torch.nn as nn


class ModelType(enum.Enum):
    stable = 'stable'
    deprecated = 'deprecated'
    experimental = 'experimental'


class ModelRegistryEntry(TypedDict):
    model_class: Type[nn.Module]
    model_type: ModelType


_MODEL_REGISTRY: Dict[str, ModelRegistryEntry] = {}
_LOADED_MODEL_MODULES: set[str] = set()


def _load_cnn_models() -> None:
    from . import CNN_Models  # noqa: F401


def _load_dual_scale_models() -> None:
    from . import dual_scale_models  # noqa: F401


def _load_transformer_segmentation_models() -> None:
    from . import transformer_segmentation  # noqa: F401


_MODEL_MODULE_LOADERS: Dict[str, Callable[[], None]] = {
    'cnn_models': _load_cnn_models,
    'dual_scale_models': _load_dual_scale_models,
    'transformer_segmentation': _load_transformer_segmentation_models,
}

_MODEL_CATALOG: Dict[str, tuple[str, ModelType]] = {
    'S 660k': ('cnn_models', ModelType.deprecated),
    'M 720k': ('cnn_models', ModelType.deprecated),
    'Unet 21.6M': ('cnn_models', ModelType.deprecated),
    'Wellnet 86.5M': ('cnn_models', ModelType.deprecated),
    'Wellnet2': ('cnn_models', ModelType.deprecated),
    'Wellnet2 mini': ('cnn_models', ModelType.deprecated),
    'EfficientUNet': ('cnn_models', ModelType.stable),
    'EfficientUNetMax': ('cnn_models', ModelType.experimental),
    'UNET++': ('cnn_models', ModelType.experimental),
    'Transformer': ('cnn_models', ModelType.experimental),
    'FrameUnet': ('dual_scale_models', ModelType.experimental),
    'Swin UPerNet B': ('transformer_segmentation', ModelType.experimental),
    'Swin UPerNet L': ('transformer_segmentation', ModelType.experimental),
    'Mask2Former Swin B': ('transformer_segmentation', ModelType.experimental),
    'Mask2Former Swin L': ('transformer_segmentation', ModelType.experimental),
}


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


def _ensure_model_loaded(name: str) -> None:
    entry = _MODEL_CATALOG.get(str(name))
    if entry is None:
        return
    module_key, _model_type = entry
    if module_key in _LOADED_MODEL_MODULES:
        return
    _MODEL_MODULE_LOADERS[module_key]()
    _LOADED_MODEL_MODULES.add(module_key)


def _ensure_all_models_loaded() -> None:
    for model_name in _MODEL_CATALOG:
        _ensure_model_loaded(model_name)


def get_registered_models() -> Dict[str, Type[nn.Module]]:
    """Return a name -> model class mapping."""
    _ensure_all_models_loaded()
    return {
        name: entry['model_class']
        for name, entry in _MODEL_REGISTRY.items()
    }


def get_registered_model_registry() -> Dict[str, ModelRegistryEntry]:
    """Return a copy of the full registry metadata."""
    _ensure_all_models_loaded()
    return {
        name: {
            'model_class': entry['model_class'],
            'model_type': entry['model_type'],
        }
        for name, entry in _MODEL_REGISTRY.items()
    }


def get_registered_model_names_by_type() -> dict[ModelType, list[str]]:
    grouped: dict[ModelType, list[str]] = {model_type: [] for model_type in ModelType}
    for model_name, (_module_name, model_type) in _MODEL_CATALOG.items():
        grouped[model_type].append(model_name)
    return grouped


def model_supports_init_kwarg(name: str, kwarg: str) -> bool:
    _ensure_model_loaded(name)
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
    _ensure_model_loaded(name)
    try:
        model_cls = _MODEL_REGISTRY[name]['model_class']
    except KeyError as exc:
        raise ValueError(f'Неизвестная модель {name!r}. Доступные: {list(_MODEL_REGISTRY)}') from exc
    return model_cls(*args, **kwargs)
