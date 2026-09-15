"""Explicit JSON codec for existing configuration dataclasses (never pickle)."""

from __future__ import annotations
import dataclasses
import enum
import importlib
from pathlib import Path

CONFIG_MODULES = (
    "neuralimage.lib.data_interfaces",
    "neuralimage.active_learning.config",
    "neuralimage.augmentations.sem_config",
    "neuralimage.preprocessing.config",
    "neuralimage.targets.config",
    "neuralimage.uncertainty.config",
)


def encode(value):
    if dataclasses.is_dataclass(value):
        return {
            "$type": type(value).__module__ + ":" + type(value).__name__,
            "fields": {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)},
        }
    if isinstance(value, enum.Enum):
        return {"$enum": type(value).__module__ + ":" + type(value).__name__, "value": value.value}
    if isinstance(value, Path):
        return {"$path": str(value)}
    if isinstance(value, tuple):
        return {"$tuple": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "$type" in value or "$enum" in value:
        module, name = value.get("$type", value.get("$enum")).split(":")
        if module not in CONFIG_MODULES:
            raise ValueError("Unknown configuration type")
        cls = getattr(importlib.import_module(module), name)
        if "$enum" in value and isinstance(cls, type) and issubclass(cls, enum.Enum):
            return cls(value["value"])
        if "$type" in value and isinstance(cls, type) and dataclasses.is_dataclass(cls):
            return cls(**{k: decode(v) for k, v in value["fields"].items()})
        raise ValueError("Not a configuration dataclass or enum")
    if "$path" in value:
        return Path(value["$path"])
    if "$tuple" in value:
        return tuple(decode(v) for v in value["$tuple"])
    return {k: decode(v) for k, v in value.items()}
