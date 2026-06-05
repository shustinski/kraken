"""Helpers for loading the Karakal application icon."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QWidget

_ICON_CANDIDATES = (
    "resources/icons/karakal_light.ico",
    "resources/icons/karakal_light.png",
    "resources/icons/karakal.ico",
    "resources/icons/karakal.png",
)


def _resource_roots() -> tuple[Path, ...]:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(getattr(sys, "_MEIPASS"))
        return (base / "karakal", base)
    module_path = Path(__file__).resolve()
    return (module_path.parents[1], module_path.parents[2])


def _icon_path_candidates(root: Path, relative_path: str) -> tuple[Path, ...]:
    direct = root / relative_path
    if direct.is_file():
        return (direct,)
    filename = Path(relative_path).name
    if not root.exists():
        return ()
    matches = tuple(path for path in root.rglob(filename) if path.is_file())
    return matches


def karakal_icon() -> QIcon:
    """Return the bundled Karakal icon, or an empty icon if unavailable."""

    for root in _resource_roots():
        for relative_path in _ICON_CANDIDATES:
            for candidate in _icon_path_candidates(root, relative_path):
                icon = QIcon(str(candidate))
                if not icon.isNull():
                    return icon
    return QIcon()


def apply_karakal_icon(target: QWidget | None = None) -> QIcon:
    """Apply the Karakal icon to the current app and an optional widget."""

    icon = karakal_icon()
    app = QApplication.instance()
    if app is not None:
        app.setWindowIcon(icon)
    if target is not None:
        target.setWindowIcon(icon)
    return icon
