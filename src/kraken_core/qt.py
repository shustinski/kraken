from __future__ import annotations

import os
import sys

from .styles import plugin_icon_path, shared_icon_path


def configure_application_identity(app, *, app_id: str, icon_name: str = "kraken") -> None:
    """Set runtime icon metadata for Qt apps, including Windows taskbar grouping."""
    if hasattr(app, "setDesktopFileName"):
        app.setDesktopFileName(app_id)
    if hasattr(app, "setApplicationName"):
        app.setApplicationName(app_id)
    _set_windows_app_user_model_id(app_id)

    try:
        from PyQt6.QtGui import QIcon
    except ImportError:
        return

    icon_path = resolve_icon_path(icon_name)
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))


def resolve_icon_path(icon_name: str):
    suffixes = (".ico", ".png") if os.name == "nt" else (".png", ".ico")
    for suffix in suffixes:
        if icon_name != "kraken":
            candidate = plugin_icon_path(icon_name, suffix=suffix)
            if candidate.exists():
                return candidate
        candidate = shared_icon_path(icon_name, suffix=suffix)
        if candidate.exists():
            return candidate
    if icon_name != "kraken":
        return resolve_icon_path("kraken")
    return None


def _set_windows_app_user_model_id(app_id: str) -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
    except Exception:
        return
