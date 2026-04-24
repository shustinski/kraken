from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class BatchWorkerSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(str, str)
    finished = pyqtSignal()
    log = pyqtSignal(str)
