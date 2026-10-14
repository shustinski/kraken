"""test_hub_manager_app: event."""
from __future__ import annotations
from datetime import UTC, datetime
from types import SimpleNamespace
import pytest
from PyQt6.QtWidgets import (
    QApplication,
)

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])

def _event(
    event_type: str,
    payload: dict[str, object],
    *,
    minute: int = 0,
    event_id: str = "event-1",
    stream_id: str = "project:project-1",
):
    return SimpleNamespace(
        event_id=event_id,
        stream_id=stream_id,
        event_type=event_type,
        payload=payload,
        recorded_at=datetime(2026, 7, 28, 10, minute, tzinfo=UTC),
        actor=SimpleNamespace(display_name="Оператор"),
    )
