"""test_hub_manager_app: test property helpers filter."""
from __future__ import annotations
from pathlib import Path
from kraken_hub.manager_app import (
    _count_regular_files,
    _event_belongs_to_layer,
    _event_matches_node,
    _history_entries,
)
from kraken_manager.presentation.qt import (
    PipelineNode,
)

from test_hub_manager_app_parts.event import _event

def test_property_helpers_filter_history_and_deduplicate_files(tmp_path: Path) -> None:
    first = tmp_path / "first"
    nested = first / "nested"
    nested.mkdir(parents=True)
    (first / "a.png").write_bytes(b"a")
    (nested / "b.cif").write_bytes(b"b")
    missing = tmp_path / "missing"

    assert _count_regular_files((first, first / "a.png", missing)) == 2
    assert _count_regular_files((missing,)) is None

    layer_event = _event("LayerRenamed", {"layer": {"id": "layer-1"}})
    job_event = _event(
        "PluginJobCreated",
        {
            "plugin_job_id": "job-1",
            "manifest": {
                "layer_id": "layer-1",
                "target_representation_id": "representation-1",
            },
            "job": {"id": "job-1", "layer_id": "layer-1"},
        },
        minute=1,
        event_id="event-2",
        stream_id="plugin-job:job-1",
    )
    unrelated = _event("LayerCreated", {"layer_id": "layer-2"}, minute=2)
    representation_event = _event(
        "RepresentationNoteUpdated",
        {"representation": {"id": "representation-1", "layer_id": "layer-1"}},
        minute=3,
        event_id="event-3",
        stream_id="layer:layer-1",
    )
    node = PipelineNode(
        "representation-1",
        "Исходники",
        "source",
        representation_id="representation-1",
    )

    assert _event_belongs_to_layer(layer_event, "layer-1")
    assert _event_belongs_to_layer(job_event, "layer-1")
    assert not _event_belongs_to_layer(unrelated, "layer-1")
    assert _event_matches_node(job_event, node)
    assert _event_matches_node(representation_event, node)
    assert not _event_matches_node(unrelated, node)
    assert [item.event_type for item in _history_entries((layer_event, job_event))] == [
        "PluginJobCreated",
        "LayerRenamed",
    ]
