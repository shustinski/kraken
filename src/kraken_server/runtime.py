"""Environment-selected Server composition without insecure implicit fallback."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .app import create_app
from .configuration import ServerConfig, default_config_path


def _factory(reference: str) -> Callable[[], Any]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise TypeError("KRAKEN_SERVER_COMPOSITION must use module:function syntax")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError("Configured server composition is not callable")
    return value


def create_app_from_environment() -> Any:
    if os.environ.get("KRAKEN_SERVER_DEVELOPMENT") == "1":
        return create_app(development=True)
    configured_path = os.environ.get("KRAKEN_SERVER_CONFIG", "").strip()
    candidate = Path(configured_path) if configured_path else default_config_path()
    if candidate.is_file():
        ServerConfig.load(candidate).apply_to_environment()
    reference = os.environ.get("KRAKEN_SERVER_COMPOSITION", "")
    if not reference:
        raise RuntimeError("KRAKEN_SERVER_COMPOSITION is required outside explicit development mode")
    composed = _factory(reference)()
    if hasattr(composed, "router"):
        return composed
    if not isinstance(composed, dict):
        raise TypeError("Server composition must return a FastAPI app or create_app keyword mapping")
    return create_app(**composed)


__all__ = ["create_app_from_environment"]
