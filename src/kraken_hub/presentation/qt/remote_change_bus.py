"""Qt delivery of addressed changes from the remote project stream."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


_PREFIX_TOPICS = (
    ("analysis_", "analysis"),
    ("job.", "jobs"),
    ("pluginjob", "jobs"),
    ("layer", "layers"),
    ("workorder", "work"),
    ("work_order", "work"),
    ("pipeline", "pipeline"),
    ("artifact", "artifacts"),
    ("stitch", "stitch"),
    ("multilayer", "stitch"),
    ("tiles.", "tiles"),
    ("project", "catalog"),
)


def topics_for_event(event_type: str) -> frozenset[str]:
    name = event_type.lower()
    for prefix, topic in _PREFIX_TOPICS:
        if name.startswith(prefix):
            return frozenset({topic})
    return frozenset({"project"})


class RemoteChangeBus(QObject):
    changed = pyqtSignal(str, object, object)
    statusChanged = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.status = "offline"
