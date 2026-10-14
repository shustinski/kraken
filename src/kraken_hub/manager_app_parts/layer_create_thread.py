"""manager_app: layer create thread."""
from __future__ import annotations
from pathlib import Path
from threading import Event
from PyQt6.QtCore import QThread, pyqtSignal

class _LayerCreateThread(QThread):
    progress = pyqtSignal(int, int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function, *, cancelled: Event, parent=None) -> None:
        super().__init__(parent)
        self._function = function
        self.cancelled = cancelled

    def run(self) -> None:
        try:
            result = self._function(
                progress=lambda done, total, label: self.progress.emit(done, total, label),
                cancelled=self.cancelled,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)

class _BackgroundCallThread(QThread):
    """Run one blocking service call without occupying the Qt event loop."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function, *, parent=None) -> None:
        super().__init__(parent)
        self._function = function

    def run(self) -> None:
        try:
            result = self._function()
        except Exception as exc:  # noqa: BLE001 - Qt boundary reports service failures
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)

def _volume_key(path: Path) -> str:
    text = str(path)
    if text.startswith("\\\\"):
        parts = [part for part in text.split("\\") if part]
        return "\\\\" + "\\".join(parts[:2]).casefold()
    return path.anchor.casefold()
