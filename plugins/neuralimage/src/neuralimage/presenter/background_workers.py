from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6 import QtCore
from PyQt6.QtCore import QThread

from neuralimage.lib.data_interfaces import RecognitionParameters, TrainingParameters, WorkMode
from neuralimage.lib.message_bus import AbstractMessageBus
from neuralimage.lib.update_checker import ReleaseInfo, download_update_installer, fetch_update_info
from neuralimage.model.general_neural_handler import GeneralNeuralHandler


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


class GeneralNeuralHandlerThread(QThread):
    ask = QtCore.pyqtSignal(str, str, bool, int)
    answer = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        work_mode: WorkMode,
        message_bus: AbstractMessageBus,
        recognition_parameters: RecognitionParameters | None = None,
        tranining_parameters: TrainingParameters | None = None,
        callback: Callable[..., None] | None = None,
    ):
        super().__init__()
        self._last_answer = False
        self._waiting_for_answer = False
        self.main_logic = GeneralNeuralHandler(
            work_mode=work_mode,
            recogniton_parameters=recognition_parameters,
            tranining_parameters=tranining_parameters,
            question_module=self.check,
            message_bus=message_bus,
        )
        self.answer.connect(self._store_answer)

    def run(self):
        self.main_logic.start()

    def check(self, text, theme, default_answer: bool = False, timeout_seconds: int | None = None):
        self._last_answer = bool(default_answer)
        self._waiting_for_answer = True
        self.ask.emit(text, theme, bool(default_answer), max(0, int(timeout_seconds or 0)))
        loop = QtCore.QEventLoop()

        def _quit_loop(_value: bool) -> None:
            if loop.isRunning():
                loop.quit()

        self.answer.connect(_quit_loop)
        try:
            loop.exec()
            return self._last_answer
        finally:
            self._waiting_for_answer = False
            try:
                self.answer.disconnect(_quit_loop)
            except TypeError:
                pass

    def stop(self):
        if self._waiting_for_answer:
            self.answer.emit(False)
        self.main_logic.stop_execution()

    @QtCore.pyqtSlot(bool)
    def _store_answer(self, val: bool):
        self._last_answer = val


class AppUpdateCheckThread(QThread):
    checked = QtCore.pyqtSignal(object)

    def __init__(self, *, manifest_url: str, channel: str) -> None:
        super().__init__()
        self._manifest_url = str(manifest_url).strip()
        self._channel = str(channel or '').strip().lower()

    def run(self) -> None:
        self.checked.emit(fetch_update_info(self._manifest_url, expected_channel=self._channel))


class AppUpdateDownloadThread(QThread):
    finished_download = QtCore.pyqtSignal(str, str)
    failed_download = QtCore.pyqtSignal(str)

    def __init__(self, *, release: ReleaseInfo) -> None:
        super().__init__()
        self._release = release

    def run(self) -> None:
        try:
            installer_path = download_update_installer(self._release)
        except Exception as exc:
            self.failed_download.emit(str(exc))
            return
        self.finished_download.emit(str(installer_path), self._release.version)
