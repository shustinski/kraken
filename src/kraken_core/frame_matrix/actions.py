"""Frame actions a matrix offers in its context menu."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ActionTitle = str | Callable[[Any], str]
ActionPredicate = Callable[[Any], bool]


@dataclass(frozen=True, slots=True)
class FrameMatrixAction:
    """One context-menu action of a frame matrix.

    ``title`` and ``is_visible`` receive the ``FrameContext`` of the click, so a
    title can name the selection size and an action can hide itself.
    """

    id: str
    title: ActionTitle
    is_visible: ActionPredicate | None = None

    def visible_for(self, context: Any) -> bool:
        return self.is_visible is None or bool(self.is_visible(context))

    def title_for(self, context: Any) -> str:
        return self.title(context) if callable(self.title) else str(self.title)


__all__ = ["FrameMatrixAction"]
