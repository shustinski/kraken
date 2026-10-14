"""main_window: responsive preview label."""
from __future__ import annotations
import os
from pathlib import Path
from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QListWidget,
    QSizePolicy,
    QWidget,
)
from neuralimage.lib.runtime_paths import resolve_internal_path
from neuralimage.lib.shared_styles import load_stylesheet

from neuralimage.view.main_window_parts.module_state import PREVIEW_MINIMUM_SIZE

def _main_window_qsettings() -> QSettings:
    root = os.getenv('NEURALIMAGE_SETTINGS_DIR')
    if root:
        settings_root = Path(root)
        settings_root.mkdir(parents=True, exist_ok=True)
        return QSettings(
            str(settings_root / 'NeuralImage_MainWindow.ini'),
            QSettings.Format.IniFormat,
        )
    return QSettings('NeuralImage', 'MainWindow')

def _load_persisted_ui_mode() -> str:
    settings = _main_window_qsettings()
    value = settings.value('ui_mode', 'simple', type=str)
    settings.sync()
    return 'advanced' if value == 'advanced' else 'simple'

def _save_persisted_ui_mode(mode: str) -> None:
    settings = _main_window_qsettings()
    settings.setValue('ui_mode', 'advanced' if mode == 'advanced' else 'simple')
    settings.sync()

class ResponsivePreviewLabel(QLabel):
    """A centered preview that grows with its layout and keeps its source pixmap."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._source_pixmap = QPixmap()
        self.setMinimumSize(PREVIEW_MINIMUM_SIZE, PREVIEW_MINIMUM_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = QPixmap(pixmap)
        self._refresh_pixmap()

    def clear(self) -> None:
        self._source_pixmap = QPixmap()
        super().clear()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        super().setPixmap(
            self._source_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

def _modal_dialogs_enabled() -> bool:
    disabled_flag = str(os.getenv('NEURALIMAGE_DISABLE_MODAL_DIALOGS', '') or '').strip().lower()
    if disabled_flag in {'1', 'true', 'yes', 'on'}:
        return False
    return not bool(os.getenv('PYTEST_CURRENT_TEST'))

def load_qss_from_resource(qss_path: str):
    return load_stylesheet(qss_path)

def _load_menu_icon() -> QIcon:
    return QIcon(str(resolve_internal_path('settings_icon.png')))

class TaskQueueListWidget(QListWidget):
    rows_reordered: pyqtSignal = pyqtSignal(int, int)

    def dropEvent(self, event) -> None:
        current_item = self.currentItem()
        source_row = self.row(current_item) if current_item is not None else -1
        super().dropEvent(event)
        if current_item is None:
            return
        target_row = self.row(current_item)
        if source_row >= 0 and target_row >= 0 and source_row != target_row:
            self.rows_reordered.emit(source_row, target_row)
