from typing import Callable, Dict, Type
import torch.nn as nn

# ----------------------------------------------------------------------
# Регистратор (registry)
# ----------------------------------------------------------------------
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str | None = None):
    """
    Декоратор, регистрирующий класс модели в глобальном реестре.
    Если `name` не указан – берётся имя класса.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if not issubclass(cls, nn.Module):
            raise TypeError("Только подклассы torch.nn.Module могут быть зарегистрированы")
        model_name = name or cls.__name__
        if model_name in _MODEL_REGISTRY:
            raise KeyError(f"Модель с именем {model_name!r} уже зарегистрирована")
        _MODEL_REGISTRY[model_name] = cls
        return cls
    return decorator


def get_registered_models() -> Dict[str, Type[nn.Module]]:
    """Возвращает копию реестра (чтобы пользователь случайно не изменил его)."""
    return dict(_MODEL_REGISTRY)


def create_model(name: str, *args, **kwargs) -> nn.Module:
    """Фабричный метод – создать экземпляр модели по её имени."""
    try:
        model_cls = _MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Неизвестная модель {name!r}. Доступные: {list(_MODEL_REGISTRY)}") from exc
    return model_cls(*args, **kwargs)