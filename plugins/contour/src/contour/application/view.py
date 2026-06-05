from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QCloseEvent, QIcon
from PyQt6.QtWidgets import QMainWindow
from kraken_core.theme import add_theme_menu, apply_app_theme

from ..widget import PolygonExtractionWidget
from .styles import resolve_style_path

WINDOW_SCREEN_MARGIN_PX = 32
MIN_INITIAL_WINDOW_WIDTH = 640
MIN_INITIAL_WINDOW_HEIGHT = 420


def _try_apply_app_icon(window: QMainWindow) -> None:
    icon_path = resolve_style_path("icons", "icon.png")
    if icon_path.exists():
        window.setWindowIcon(QIcon(str(icon_path)))


def _bounded_initial_window_size(
    width: int,
    height: int,
    available_width: int,
    available_height: int,
) -> QSize:
    max_width = max(320, int(available_width) - WINDOW_SCREEN_MARGIN_PX)
    max_height = max(240, int(available_height) - WINDOW_SCREEN_MARGIN_PX)

    def _clamp(requested: int, preferred_minimum: int, maximum: int) -> int:
        requested = max(1, int(requested))
        if maximum < preferred_minimum:
            return maximum
        return min(max(requested, preferred_minimum), maximum)

    return QSize(
        _clamp(width, MIN_INITIAL_WINDOW_WIDTH, max_width),
        _clamp(height, MIN_INITIAL_WINDOW_HEIGHT, max_height),
    )


class ContourMainView(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._presenter: Any | None = None
        self._widget = PolygonExtractionWidget(self)
        self.setCentralWidget(self._widget)
        add_theme_menu(self, initial_theme="dark", on_theme_changed=apply_app_theme)
        self._help_menu = self.menuBar().addMenu(self._widget.help_menu_title())
        self._widget.attach_help_menu(self._help_menu)
        _try_apply_app_icon(self)

    @property
    def widget(self) -> PolygonExtractionWidget:
        return self._widget

    def set_presenter(self, presenter: Any) -> None:
        self._presenter = presenter

    def set_window_title(self, title: str) -> None:
        self.setWindowTitle(title)

    def resize_window(self, width: int, height: int) -> None:
        screen = self.screen()
        if screen is None:
            self.resize(width, height)
            return
        available = screen.availableGeometry()
        self.resize(
            _bounded_initial_window_size(
                width,
                height,
                available.width(),
                available.height(),
            )
        )

    def set_ui_language(self, language: str) -> None:
        self._widget.set_ui_language(language)
        self._help_menu.setTitle(self._widget.help_menu_title())

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

    def closeEvent(self, event: QCloseEvent) -> None:
        if hasattr(self._widget, "confirm_ok_to_leave_current_vectors") and not self._widget.confirm_ok_to_leave_current_vectors():
            event.ignore()
            return
        super().closeEvent(event)


class ContourStandaloneWindow(ContourMainView):
    """Backward-compatible alias for the standalone main window."""
