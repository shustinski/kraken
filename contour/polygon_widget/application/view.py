from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMainWindow

from ..widget import PolygonExtractionWidget


def _try_apply_app_icon(window: QMainWindow) -> None:
    try:
        from lib.runtime_paths import resolve_internal_path
    except Exception:
        return
    icon_path = resolve_internal_path("icon.png")
    if Path(icon_path).exists():
        window.setWindowIcon(QIcon(str(icon_path)))


class PolygonWidgetMainView(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._presenter: Any | None = None
        self._widget = PolygonExtractionWidget(self)
        self.setCentralWidget(self._widget)
        _try_apply_app_icon(self)

    @property
    def widget(self) -> PolygonExtractionWidget:
        return self._widget

    def set_presenter(self, presenter: Any) -> None:
        self._presenter = presenter

    def set_window_title(self, title: str) -> None:
        self.setWindowTitle(title)

    def resize_window(self, width: int, height: int) -> None:
        self.resize(width, height)

    def set_ui_language(self, language: str) -> None:
        self._widget.set_ui_language(language)

    def set_input_directory(self, path: str) -> None:
        self._widget.set_input_directory(path)

    def set_output_directory(self, path: str) -> None:
        self._widget.set_output_directory(path)

    def set_cif_directory(self, path: str) -> None:
        self._widget.set_cif_directory(path)

    def set_pipeline(self, payload: dict) -> None:
        self._widget.set_pipeline(payload)

    def load_images(self, paths: list[str]) -> None:
        self._widget.load_images(paths)

    def show_status_message(self, message: str, timeout_ms: int = 0) -> None:
        self.statusBar().showMessage(message, timeout_ms)

    def bind_log_message(self, handler: Callable[[str], None]) -> None:
        self._widget.logMessage.connect(handler)

    def bind_image_processed(self, handler: Callable[[str, list], None]) -> None:
        self._widget.imageProcessed.connect(handler)


class PolygonWidgetStandaloneWindow(PolygonWidgetMainView):
    """Backward-compatible alias for the standalone main window."""

