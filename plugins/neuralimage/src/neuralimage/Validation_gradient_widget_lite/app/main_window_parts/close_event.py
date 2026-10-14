"""main_window: close event."""
from __future__ import annotations


class CloseEventMixin:
    def shutdown(self) -> None:
        self._presenter.shutdown()

    def closeEvent(self, event) -> None:
        self._presenter.shutdown()
        super().closeEvent(event)
