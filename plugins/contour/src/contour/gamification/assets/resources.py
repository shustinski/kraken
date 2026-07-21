from __future__ import annotations

from importlib import import_module


def ensure_gamification_qt_resources_registered() -> bool:
    """Register compiled Qt resources when a generated ``gamification_rc`` exists.

    Add new sprite sheets to ``gamification.qrc`` and compile it into
    ``gamification_rc.py`` during packaging if the application should load only
    from ``:/gamification/...`` paths. In source-tree/dev runs the animation
    registry also has a package-file fallback so missing compiled resources never
    break contour workflows.
    """

    try:
        module = import_module("contour.gamification.assets.gamification_rc")
    except ModuleNotFoundError:
        return False
    init = getattr(module, "qInitResources", None)
    if callable(init):
        init()
    return True


__all__ = ["ensure_gamification_qt_resources_registered"]
