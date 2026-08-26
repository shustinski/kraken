"""Cartograph: local SEM-frame navigation and 3×3 stitching for Kraken.

This plugin does not build a global mosaic. The v1 compute unit is a sliding
3×3 window: nominal placement, constrained phase-correlation pairs, cycle
checks, robust translation-only poses, and a local diagnostic mosaic.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CartographPlugin",
    "CartographWindow",
    "RegisterLocalWindow",
    "RunLocalVerticalSlice",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CartographPlugin": (".plugin.plugin", "CartographPlugin"),
    "CartographWindow": (".presentation.qt.window", "CartographWindow"),
    "RegisterLocalWindow": (".application.local_registration", "RegisterLocalWindow"),
    "RunLocalVerticalSlice": (".application.pipeline", "RunLocalVerticalSlice"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
