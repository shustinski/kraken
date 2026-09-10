import pytest
from PyQt6.QtCore import QCoreApplication, QObject, pyqtSignal

from contour.adapters.qt.signal_connection import connect_queued

pytestmark = pytest.mark.fast


def test_queued_connection_waits_for_event_loop() -> None:
    class Sender(QObject):
        changed = pyqtSignal(int)

    sender = Sender()
    received: list[int] = []
    connect_queued(sender.changed, received.append)
    sender.changed.emit(42)
    assert received == []
    QCoreApplication.processEvents()
    assert received == [42]
