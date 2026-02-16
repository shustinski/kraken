from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_window import MainView
    from .settings_panel import SettingsPanel, SlidingPanel

__all__ = ["MainView", "SettingsPanel", "SlidingPanel"]


def __getattr__(name: str):
    if name == "MainView":
        from .main_window import MainView

        return MainView
    if name in {"SettingsPanel", "SlidingPanel"}:
        from .settings_panel import SettingsPanel, SlidingPanel

        return {"SettingsPanel": SettingsPanel, "SlidingPanel": SlidingPanel}[name]
    raise AttributeError(name)
