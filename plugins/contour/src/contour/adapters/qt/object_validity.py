from __future__ import annotations

try:
    import shiboken6
except ImportError:
    shiboken6 = None


def qt_object_is_valid(obj: object | None) -> bool:
    if obj is None:
        return False
    if shiboken6 is not None:
        return bool(shiboken6.isValid(obj))
    try:
        obj.parent()  # type: ignore[attr-defined]
        return True
    except RuntimeError:
        return False


def safe_viewport(widget: object | None) -> object | None:
    if not qt_object_is_valid(widget):
        return None
    try:
        viewport = widget.viewport()  # type: ignore[attr-defined]
    except RuntimeError:
        return None
    return viewport if qt_object_is_valid(viewport) else None
