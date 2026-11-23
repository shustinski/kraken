"""A queued remote wake may arrive while the Qt shell is being destroyed."""

from PyQt6 import sip
from PyQt6.QtCore import QTimer

from kraken_hub.presentation.qt.parts.poll_agent_jobs import PollAgentJobsMixin
from kraken_hub.presentation.qt.parts.event_matches_node import _RemoteSignalBridge


def test_remote_wake_after_timer_deletion_is_ignored(qtbot):
    del qtbot
    controller = PollAgentJobsMixin()
    controller._sync_timer = QTimer()
    sip.delete(controller._sync_timer)

    controller._on_remote_wake(None)


def test_remote_signal_after_bridge_deletion_is_ignored(qtbot):
    del qtbot
    controller = PollAgentJobsMixin()
    controller._remote_bridge = _RemoteSignalBridge()
    sip.delete(controller._remote_bridge)

    controller._emit_remote_signal("wakeReceived", None)
