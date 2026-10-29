"""Batch work keeps one history line. A single-file edit stays visible."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from kraken_manager.domain.file_change_history import (
    BATCH_FILE_CHANGE,
    events_for_file_change_history,
)


def _event(event_type: str, payload: dict, event_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        event_type=event_type,
        payload=payload,
        recorded_at=datetime(2026, 10, 27, tzinfo=UTC),
        actor=SimpleNamespace(display_name="Оператор"),
        event_id=event_id,
        correlation_id="",
    )


def test_batch_import_does_not_append_one_history_line_per_file() -> None:
    events = [
        _event(
            "PluginResultImported",
            {"plugin_job_id": "job-1"},
            "batch",
        ),
        *[
            _event(
                "ArtifactVersionCreated",
                {"file_history": "batch", "batch_key": "job-1", "filename": f"{index}.png"},
                f"file-{index}",
            )
            for index in range(3)
        ],
    ]

    visible = events_for_file_change_history(events)
    assert [event.event_type for event in visible] == ["PluginResultImported"]


def test_layer_load_without_a_batch_event_keeps_one_batch_line() -> None:
    events = [
        _event(
            "ArtifactSeriesCreated",
            {"file_history": "batch", "batch_key": "import-1"},
            "series-1",
        ),
        _event(
            "ArtifactVersionCreated",
            {"file_history": "batch", "batch_key": "import-1"},
            "version-1",
        ),
        _event(
            "ArtifactVersionCreated",
            {"file_history": "batch", "batch_key": "import-1"},
            "version-2",
        ),
    ]

    visible = events_for_file_change_history(events)
    assert len(visible) == 1
    assert visible[0].event_type == BATCH_FILE_CHANGE
    assert visible[0].payload["file_count"] == 2
    assert visible[0].event_id == "batch-file-change:import-1"


def test_interactive_single_file_edit_stays_in_history() -> None:
    interactive = _event(
        "ArtifactVersionCreated",
        {"filename": "edited.png"},
        "edit-1",
    )
    batch = _event(
        "ArtifactVersionCreated",
        {"file_history": "batch", "batch_key": "vectorize-1"},
        "batch-file",
    )

    visible = events_for_file_change_history([interactive, batch])
    types = [event.event_type for event in visible]
    assert types.count("ArtifactVersionCreated") == 1
    assert types.count(BATCH_FILE_CHANGE) == 1
    kept = next(event for event in visible if event.event_type == "ArtifactVersionCreated")
    assert kept.payload["filename"] == "edited.png"
