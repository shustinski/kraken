"""main_presenter: rare patch editor preparation."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable
from PyQt6.QtCore import QObject, QThread
from PyQt6 import QtCore, QtWidgets
from neuralimage.lib.message_bus import AbstractMessageBus
import neuralimage.presenter.dialogs as presenter_dialogs

class _ValidationGradientPluginWindow(QtWidgets.QMainWindow):
    """Host one lite plugin widget as an independent top-level window."""

    def __init__(self, plugin, widget, title: str, on_closed: Callable[[], None], parent=None):
        super().__init__(parent)
        self._plugin = plugin
        self._on_closed = on_closed
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setWindowTitle(str(title))
        self.setCentralWidget(widget)
        self.resize(1400, 900)

    def closeEvent(self, event) -> None:
        try:
            if self._plugin is not None:
                self._plugin.shutdown()
        finally:
            self._plugin = None
            try:
                self._on_closed()
            finally:
                super().closeEvent(event)

def _format_auto_answer_button_text(text: str, seconds_left: int) -> str:
    return presenter_dialogs.format_auto_answer_button_text(text, seconds_left)

def _tk_filedialog(kind: str, filetypes=None) -> str | None:
    return presenter_dialogs.choose_path_dialog(kind, filetypes)

class SampleCountSignals(QObject):
    calculated = QtCore.pyqtSignal(int, str, object, int)
    failed = QtCore.pyqtSignal(int, str)

def _should_schedule_startup_update_check() -> bool:
    disabled_flag = str(os.getenv('NEURALIMAGE_DISABLE_AUTO_UPDATE_CHECK', '') or '').strip().lower()
    if disabled_flag in {'1', 'true', 'yes', 'on'}:
        return False
    return not bool(os.getenv('PYTEST_CURRENT_TEST'))

class RarePatchEditorPreparationThread(QThread):
    prepared = QtCore.pyqtSignal(str, str, str)

    def __init__(
        self,
        *,
        sample_folder: Path,
        label_folder: Path,
        message_bus: AbstractMessageBus,
    ) -> None:
        super().__init__()
        self._sample_folder = Path(sample_folder)
        self._label_folder = Path(label_folder)
        self._message_bus = message_bus

    def run(self) -> None:
        from neuralimage.lib.rare_patch_masks import (
            collect_matching_sample_label_pairs,
            prepare_label_folder_for_rare_patch_editor,
        )

        try:
            resolved_label_folder, error_message = prepare_label_folder_for_rare_patch_editor(
                self._label_folder,
                log_callback=lambda message: self._message_bus.publish('logging', message),
            )
            if error_message is None:
                _pairs, error_message = collect_matching_sample_label_pairs(
                    self._sample_folder,
                    resolved_label_folder,
                )
        except Exception as exc:
            resolved_label_folder = self._label_folder
            error_message = f'Не удалось подготовить метки для редактора редких областей: {exc}'

        self.prepared.emit(
            str(self._sample_folder),
            str(resolved_label_folder),
            '' if error_message is None else str(error_message),
        )
