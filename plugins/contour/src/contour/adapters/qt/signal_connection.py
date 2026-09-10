"""Typed access to the queued-connection overload missing from PyQt6 stubs."""

from collections.abc import Callable
from typing import cast

from PyQt6.QtCore import QMetaObject, Qt, pyqtBoundSignal


def connect_queued(signal: pyqtBoundSignal, slot: Callable[..., object]) -> QMetaObject.Connection:
    # PyQt6's runtime connect(slot, type, no_receiver_check=False) supports this
    # overload, although QtCore.pyi currently declares only connect(slot).
    connect = cast(Callable[[Callable[..., object], Qt.ConnectionType], QMetaObject.Connection], signal.connect)
    return connect(slot, Qt.ConnectionType.QueuedConnection)
