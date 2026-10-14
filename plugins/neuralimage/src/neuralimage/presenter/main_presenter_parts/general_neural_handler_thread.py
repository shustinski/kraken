"""main_presenter: general neural handler thread."""
from __future__ import annotations
from typing import Callable
from neuralimage.infrastructure.dataset_source import DatasetStreamSource
from PyQt6.QtCore import QThread
from PyQt6 import QtCore
from neuralimage.lib.data_interfaces import (
    WorkMode,
    TrainingParameters,
    RecognitionParameters,
)
from neuralimage.lib.message_bus import AbstractMessageBus

class GeneralNeuralHandlerThread(QThread):
    ask = QtCore.pyqtSignal(str, str, bool, int)  # text, title, default answer, timeout
    answer = QtCore.pyqtSignal(bool)

    def __init__(self, work_mode: WorkMode,
                 message_bus: AbstractMessageBus,
                 recognition_parameters: RecognitionParameters | None = None,
                 tranining_parameters: TrainingParameters | None = None,
                 callback: Callable[..., None] | None = None, dataset_source: DatasetStreamSource | None = None):
        super().__init__()
        self._last_answer = False
        self._waiting_for_answer = False
        from neuralimage.model.general_neural_handler import GeneralNeuralHandler

        self.main_logic = GeneralNeuralHandler(work_mode=work_mode,
                                               recogniton_parameters=recognition_parameters,
                                               tranining_parameters=tranining_parameters,
                                               question_module=self.check,
                                               message_bus=message_bus, dataset_source=dataset_source)
        self.answer.connect(self._store_answer)

    def run(self):
        self.main_logic.start()

    def check(self, text, theme, default_answer: bool = False, timeout_seconds: int | None = None):
        self._last_answer = bool(default_answer)
        self._waiting_for_answer = True
        self.ask.emit(
            text,
            theme,
            bool(default_answer),
            max(0, int(timeout_seconds or 0)),
        )
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

    def pause(self):
        if self._waiting_for_answer:
            self.answer.emit(False)
        self.main_logic.pause_execution()

    @QtCore.pyqtSlot(bool)
    def _store_answer(self, val: bool):
        self._last_answer = val
