from __future__ import annotations

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QAbstractScrollArea, QWidget

try:
    import shiboken6
except ImportError:
    shiboken6 = None


def qt_object_is_valid(obj: QObject | None) -> bool:
    if obj is None:
        return False
    if shiboken6 is not None:
        return bool(shiboken6.isValid(obj))
    try:
        obj.parent()
        return True
    except RuntimeError:
        return False


def safe_viewport(widget: QAbstractScrollArea | None) -> QWidget | None:
    if widget is None or not qt_object_is_valid(widget):
        return None
    try:
        viewport = widget.viewport()
    except RuntimeError:
        return None
    return viewport if qt_object_is_valid(viewport) else None
