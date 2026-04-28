"""Public API for the Karakal package."""

from __future__ import annotations

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


def __getattr__(name: str) -> Any:
    if name in {"KarakalMainWindow", "KarakalWidget"}:
        from .app.main_window import KarakalMainWindow, KarakalWidget

        return {
            "KarakalMainWindow": KarakalMainWindow,
            "KarakalWidget": KarakalWidget,
        }[name]
    if name == "KarakalPlugin":
        from .plugin.plugin import KarakalPlugin

        return KarakalPlugin
    if name == "KarakalPresenter":
        from .app.presenter import KarakalPresenter

        return KarakalPresenter
    if name == "KarakalSettingsService":
        from .infra.services import KarakalSettingsService

        return KarakalSettingsService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
