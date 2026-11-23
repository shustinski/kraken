"""Addressed WebSocket wakes delivered to Qt consumers."""

import pytest

pytest.importorskip("PyQt6")

from kraken_hub.presentation.qt.remote_change_bus import topics_for_event


@pytest.mark.parametrize(("event_type", "expected"), [
    ("analysis_partition.imported", "analysis"),
    ("analysis_map.failed", "analysis"),
    ("job.state_changed", "jobs"),
    ("PluginJobStateChanged", "jobs"),
    ("LayerCreated", "layers"),
    ("WorkOrderAssigned", "work"),
    ("pipeline.started", "pipeline"),
    ("ArtifactImported", "artifacts"),
    ("stitch.completed", "stitch"),
    ("multilayer.updated", "stitch"),
    ("tiles.ready", "tiles"),
    ("ProjectRenamed", "catalog"),
    ("UnexpectedChange", "project"),
])
def test_event_topic(event_type, expected):
    assert expected in topics_for_event(event_type)
