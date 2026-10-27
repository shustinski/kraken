"""Infrastructure exports; headless codecs must not initialize GUI settings."""
from importlib import import_module

__all__ = [
    "WidgetAppearanceSettingsStore",
    "WidgetDisplaySettingsStore",
    "WidgetGamificationProfileStore",
    "WidgetMetalPresetSettingsStore",
    "WidgetPathSettingsStore",
    "WidgetSessionSettingsStore",
    "WidgetViaPresetSettingsStore",
    "WidgetWindowSettingsStore",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".settings_store", __name__), name)
    globals()[name] = value
    return value
