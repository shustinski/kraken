import enum
import inspect
from importlib import import_module
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
_LOADED_MODEL_MODULES: set[str] = set()
_MODEL_CATALOG: Dict[str, tuple[str, str, ModelType]] = {
    'S 660k': ('model.NeuralNetwork.CNN_Models', 'SmallFCNN', ModelType.deprecated),
    'M 720k': ('model.NeuralNetwork.CNN_Models', 'MediumFCNN', ModelType.deprecated),
    'Unet 21.6M': ('model.NeuralNetwork.CNN_Models', 'Unet', ModelType.deprecated),
    'Wellnet 86.5M': ('model.NeuralNetwork.CNN_Models', 'Wellnet', ModelType.deprecated),
    'Wellnet2': ('model.NeuralNetwork.CNN_Models', 'Wellnet2', ModelType.deprecated),
    'Wellnet2 mini': ('model.NeuralNetwork.CNN_Models', 'Wellnet2Mini', ModelType.deprecated),
    'EfficientUNet': ('model.NeuralNetwork.CNN_Models', 'EfficientUNet', ModelType.stable),
    'EfficientUNetMax': ('model.NeuralNetwork.CNN_Models', 'EfficientUNetMax', ModelType.experimental),
    'UNET++': ('model.NeuralNetwork.CNN_Models', 'UnetPlusPlus', ModelType.experimental),
    'Transformer': ('model.NeuralNetwork.CNN_Models', 'ImageBinarizationTransformer', ModelType.experimental),
    'FrameUnet': ('model.NeuralNetwork.dual_scale_models', 'QuasiDualScaleUNet', ModelType.experimental),
    'Swin UPerNet B': ('model.NeuralNetwork.transformer_segmentation', 'SwinUPerNetB', ModelType.experimental),
    'Swin UPerNet L': ('model.NeuralNetwork.transformer_segmentation', 'SwinUPerNetL', ModelType.experimental),
    'Mask2Former Swin B': ('model.NeuralNetwork.transformer_segmentation', 'Mask2FormerSwinB', ModelType.experimental),
    'Mask2Former Swin L': ('model.NeuralNetwork.transformer_segmentation', 'Mask2FormerSwinL', ModelType.experimental),
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
    module_name, _class_name, _model_type = entry
    if module_name in _LOADED_MODEL_MODULES:
        return
    import_module(module_name)
    _LOADED_MODEL_MODULES.add(module_name)


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
    for model_name, (_module_name, _class_name, model_type) in _MODEL_CATALOG.items():
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
