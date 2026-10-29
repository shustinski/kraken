"""test_hub_manager_app: test controller builds derived."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from kraken_hub.presentation.qt.manager_app import (
    DesktopController,
)
from kraken_manager.presentation.qt import (
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
)

from test_hub_manager_app_parts.event import _event

def test_controller_builds_derived_job_karakal_and_virtual_properties(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.bin").write_bytes(b"result")
    run = SimpleNamespace(
        run_id="run-1",
        path=str(output),
        kind=SimpleNamespace(value="result"),
        state=SimpleNamespace(value="succeeded"),
        plugin_id="neuralimage",
        operation="recognize",
        created_at="2026-07-28T10:00:00+00:00",
        provenance={"notes": "Готово"},
    )
    events = (
        _event(
            "DerivedRunStartedV1",
            {"layer_id": "layer-1", "run_id": "run-1"},
            event_id="event-run-start",
            stream_id="derived-run:run-1",
        ),
        _event(
            "PluginJobCreated",
            {
                "plugin_job_id": "job-1",
                "manifest": {
                    "layer_id": "layer-1",
                    "capability": "dataset.model.train.v1",
                    "parameters": {"epochs": 5},
                },
                "job": {"id": "job-1", "layer_id": "layer-1", "progress": 20},
            },
            minute=1,
            event_id="event-job",
            stream_id="plugin-job:job-1",
        ),
        _event(
            "KarakalAnalysisPublished",
            {"layer_id": "layer-1", "run_id": "karakal-1", "report": {"ok": True}},
            minute=2,
            event_id="event-karakal",
            stream_id="karakal:layer-1",
        ),
    )
    derived = PipelineNode(
        "workspace-output:run-1",
        "Результат NeuralImage",
        "model",
        details={"run_id": "run-1"},
    )
    job = PipelineNode(
        "job-1",
        "Обучение",
        "job",
        details={"job_id": "job-1"},
    )
    karakal = PipelineNode(
        "karakal:karakal-1",
        "Karakal",
        "karakal",
        details={"отчёт": {"ok": True}},
    )
    missing = PipelineNode(
        "source-1:missing-cif",
        "CIF не получен",
        "missing",
        "Результат отсутствует",
    )
    internal = (derived, job, karakal)
    snapshot = LayerPipelineSnapshot(
        "project-1",
        "layer-1",
        (
            PipelineLane(
                "source-1",
                "Источник",
                (PipelineNode("source-1", "Источник", "source"), *internal, missing),
                (),
            ),
        ),
    )

    class Service:
        @staticmethod
        def history(_project_id):
            return events

        @staticmethod
        def list_representations(_project_id, _layer_id):
            return ()

        @staticmethod
        def list_derived_runs(_project_id, _layer_id):
            return (run,)

        @staticmethod
        def layer_file_binding(_project_id, _layer_id):
            return None

    controller = object.__new__(DesktopController)
    controller.service = Service()
    controller._project_id = "project-1"
    controller._pipeline_snapshot = lambda *_args: snapshot
    snapshots = []
    controller._open_properties = snapshots.append

    controller._show_node_properties("layer-1", derived)
    assert dict(snapshots[-1].properties)["Количество файлов"] == 1
    assert dict(snapshots[-1].properties)["Примечание"] == "Готово"

    controller._show_node_properties("layer-1", job)
    assert "Capability" not in dict(snapshots[-1].properties)
    assert dict(snapshots[-1].properties)["Параметры"] == {"epochs": 5}

    controller._show_node_properties("layer-1", karakal)
    assert snapshots[-1].history[0].event_type == "KarakalAnalysisPublished"

    controller._show_node_properties("layer-1", missing)
    assert snapshots[-1].history == ()

    blackbox = PipelineNode("source-1:blackbox", "Чёрный ящик", "blackbox")
    controller._show_node_properties("layer-1", blackbox)
    blackbox_properties = dict(snapshots[-1].properties)
    assert blackbox_properties["Количество скрытых этапов"] == 3
    assert len(snapshots[-1].history) == 3
