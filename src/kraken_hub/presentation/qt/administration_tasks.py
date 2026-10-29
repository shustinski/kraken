"""One retained administration task; Qt receivers never outlive their owner."""
from PyQt6.QtCore import QEvent, QObject, pyqtSlot
from PyQt6.QtWidgets import QApplication

from kraken_hub.presentation.qt.parts.layer_create_thread import _BackgroundCallThread

_RUNNING = set()


def _wait_for_tasks():
    for worker in tuple(_RUNNING):
        worker.wait()


def _release(worker):
    _RUNNING.discard(worker)


class AdministrationTasks(QObject):
    def __init__(self, owner, status, set_busy):
        super().__init__(owner)
        self.status, self.set_busy = status, set_busy
        self._worker = None
        self._result = None
        self._error = None
        self._success = self._failure = None
        app = QApplication.instance()
        if not app.property("krakenAdministrationWaitConnected"):
            app.aboutToQuit.connect(_wait_for_tasks)
            app.setProperty("krakenAdministrationWaitConnected", True)

    @property
    def busy(self):
        return self._worker is not None

    def start(self, description, call, success, failure):
        if self.busy:
            return False
        self._success, self._failure = success, failure
        self._description = description
        self._result, self._error = None, None
        # The callable must capture service/input values only, never widgets.
        worker = _BackgroundCallThread(call)
        self._worker = worker
        _RUNNING.add(worker)
        worker.succeeded.connect(self._received)
        worker.failed.connect(self._failed)
        worker.finished.connect(self._finished)
        worker.finished.connect(lambda retained=worker: _release(retained))
        worker.finished.connect(worker.deleteLater)
        self._watched = (self.parent(), self.parent().window())
        for watched in self._watched:
            watched.installEventFilter(self)
        self.status.setText(description)
        self.set_busy(True)
        worker.start()
        return True

    @pyqtSlot(object)
    def _received(self, value):
        self._result = value

    @pyqtSlot(str)
    def _failed(self, message):
        self._error = message

    @pyqtSlot()
    def _finished(self):
        result, error = self._result, self._error
        success, failure = self._success, self._failure
        self._worker = None
        from PyQt6 import sip
        for watched in self._watched:
            if not sip.isdeleted(watched):
                watched.removeEventFilter(self)
        self._watched = ()
        self._success = self._failure = self._result = self._error = None
        self.set_busy(False)
        self.status.setText("Операция завершена." if error is None else f"Ошибка: {error}")
        try:
            if error is None:
                try:
                    success(result)
                except Exception as exc:
                    if not sip.isdeleted(self):
                        self.status.setText(f"Ошибка обработки результата: {exc}")
                        failure(str(exc))
            else:
                failure(error)
        except Exception as exc:
            if not sip.isdeleted(self):
                self.status.setText(f"Ошибка отображения результата: {exc}")

    def eventFilter(self, watched, event):
        if self.busy and event.type() == QEvent.Type.Close:
            self.status.setText(self._description + "\nДождитесь завершения перед закрытием окна.")
            event.ignore()
            return True
        return super().eventFilter(watched, event)
