from __future__ import annotations

import dataclasses
import types
from collections import abc
from datetime import datetime
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints


def encode_model(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return encode_model(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: encode_model(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, abc.Mapping):
        return {str(key): encode_model(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [encode_model(item) for item in value]
    raise TypeError(f"cannot encode projection model value {type(value).__name__}")


def decode_model(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    if annotation in {Any, object}:
        return value

    supertype = getattr(annotation, "__supertype__", None)
    if supertype is not None:
        return annotation(decode_model(supertype, value))

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in {Union, types.UnionType}:
        errors: list[Exception] = []
        for candidate in arguments:
            if candidate is type(None):
                continue
            try:
                return decode_model(candidate, value)
            except (TypeError, ValueError, KeyError) as exc:
                errors.append(exc)
        if errors:
            raise errors[-1]
        return value
    if origin in {abc.Mapping, dict}:
        key_type, item_type = arguments or (str, Any)
        return {
            decode_model(key_type, key): decode_model(item_type, item)
            for key, item in value.items()
        }
    if origin in {tuple, list, set, frozenset, abc.Sequence}:
        item_type = arguments[0] if arguments else Any
        decoded = [decode_model(item_type, item) for item in value]
        if origin is list:
            return decoded
        if origin is set:
            return set(decoded)
        if origin is frozenset:
            return frozenset(decoded)
        return tuple(decoded)

    if annotation is datetime:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        if not isinstance(value, abc.Mapping):
            raise TypeError(f"expected object for {annotation.__name__}")
        hints = get_type_hints(annotation)
        return annotation(
            **{
                field.name: decode_model(hints.get(field.name, field.type), value[field.name])
                for field in dataclasses.fields(annotation)
                if field.name in value
            }
        )
    if annotation in {str, int, float, bool}:
        return annotation(value)
    return value


__all__ = ["decode_model", "encode_model"]
