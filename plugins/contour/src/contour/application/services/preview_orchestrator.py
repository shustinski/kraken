"""Preview orchestration service.

Holds the debounce timer and the request-id bookkeeping used by
:class:`PolygonExtractionWidget` so the widget itself can focus on UI. The
orchestrator is intentionally Qt-aware (it wraps :class:`QTimer` and integrates
with :class:`QThreadPool`) but knows nothing about the editor scene or tabs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QTimer

if TYPE_CHECKING:
    pass


@dataclass
class _RequestCounters:
    prepared: int = 0
    preview: int = 0
    auto_tune: int = 0


class PreviewOrchestrator(QObject):
    """Track in-flight preview/prepare/auto-tune requests and debounce UI triggers."""

    def __init__(self, parent: QObject | None = None, *, debounce_ms: int = 150) -> None:
        super().__init__(parent)
        self._counters = _RequestCounters()
        self._pending_debounced: Callable[[], None] | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self._fire_debounced)

    def next_prepared_id(self) -> int:
        self._counters.prepared += 1
        return self._counters.prepared

    def next_preview_id(self) -> int:
        self._counters.preview += 1
        return self._counters.preview

    def next_auto_tune_id(self) -> int:
        self._counters.auto_tune += 1
        return self._counters.auto_tune

    @property
    def current_prepared_id(self) -> int:
        return self._counters.prepared

    @property
    def current_preview_id(self) -> int:
        return self._counters.preview

    @property
    def current_auto_tune_id(self) -> int:
        return self._counters.auto_tune

    def schedule(self, action: Callable[[], None]) -> None:
        """Debounce *action* — subsequent calls collapse until the timer fires."""
        self._pending_debounced = action
        self._timer.start()

    def flush(self) -> None:
        """Immediately invoke any pending debounced action."""
        if self._timer.isActive():
            self._timer.stop()
            self._fire_debounced()

    def _fire_debounced(self) -> None:
        action = self._pending_debounced
        self._pending_debounced = None
        if action is not None:
            action()


__all__ = ["PreviewOrchestrator"]
