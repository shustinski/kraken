"""Public API for the Karakal package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .version import __version__

__all__ = [
    "__version__",
    "KarakalMainWindow",
    "KarakalPlugin",
    "KarakalPresenter",
    "KarakalSettingsService",
    "KarakalWidget",
]


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "KarakalMainWindow": (".app.main_window", "KarakalMainWindow"),
    "KarakalWidget": (".app.main_window", "KarakalWidget"),
    "KarakalPlugin": (".plugin.plugin", "KarakalPlugin"),
    "KarakalPresenter": (".app.presenter", "KarakalPresenter"),
    "KarakalSettingsService": (".infra.services", "KarakalSettingsService"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
