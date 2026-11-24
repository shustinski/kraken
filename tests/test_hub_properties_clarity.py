"""User-facing property cards retain useful data without exposing transport fields."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
import pytest
from PyQt6.QtCore import QTimer

from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_manager.presentation.qt import LayerPipelineSnapshot, PipelineLane, PipelineNode
from kraken_manager.presentation.qt.layer_management import ObjectPropertiesDialog, ObjectPropertiesSnapshot


def _job_snapshot(*, state: str, applied: dict | None = None):
    created = SimpleNamespace(
        event_type="PluginJobCreated",
        event_id="created",
        recorded_at=datetime(2026, 7, 28, 10, tzinfo=UTC),
        actor=SimpleNamespace(display_name="Mark"),
        program=SimpleNamespace(name="contour", version="1.0"),
        payload={
            "plugin_job_id": "job-1",
            "manifest": {
                "layer_id": "layer-1",
                "capability": "frames.vectorize.v1",
                "parameters": {"project_width": 70},
            },
            "job": {"id": "job-1", "layer_id": "layer-1", "state": state, "progress": 0},
        },
    )
    events = [created]
    if applied is not None:
        events.append(
            SimpleNamespace(
                event_type="PluginResultImported",
                event_id="result",
                recorded_at=datetime(2026, 7, 28, 11, tzinfo=UTC),
                actor=SimpleNamespace(display_name="Mark"),
                program=SimpleNamespace(name="contour", version="1.0"),
                payload={
                    "plugin_job_id": "job-1",
                    "job": {"id": "job-1", "layer_id": "layer-1", "state": state},
                    "result": {"plugin_name": "contour", "parameters_applied": applied},
                },
            )
        )
    node = PipelineNode(
        "job-1",
        "Векторизация изображений",
        "job",
        "Ожидает действий пользователя",
        state=state,
        details={"job_id": "job-1", "capability": "frames.vectorize.v1", "параметры": applied or {"project_width": 70}},
    )
    controller = object.__new__(DesktopController)
    controller.service = SimpleNamespace(
        history=lambda *_args, **_kwargs: tuple(events),
        list_representations=lambda *_: (),
        list_derived_runs=lambda *_: (),
    )
    controller._project_id = "project-1"
    controller.shell = None
    controller._pipeline_snapshot = lambda *_: LayerPipelineSnapshot(
        "project-1", "layer-1", (PipelineLane("source", "Источник", (node,), ()),)
    )
    snapshots = []
    controller._open_properties = snapshots.append
    controller._show_node_properties("layer-1", node)
    return snapshots[0]


def test_pending_contour_properties_show_no_unapplied_settings():
    snapshot = _job_snapshot(state="waiting_for_user")
    assert [name for name, _ in snapshot.properties] == [
        "Кто добавил",
        "Когда добавлен",
        "Плагин",
        "Состояние",
        "Параметры",
    ]
    values = dict(snapshot.properties)
    assert values["Плагин"] == "Contour"
    assert values["Состояние"] == "Ожидает действий пользователя"
    assert values["Параметры"] == "Параметры распознавания появятся после сохранения результата"
    assert snapshot.actions == ()


def test_completed_contour_properties_use_applied_settings():
    applied = {
        "format": "contour-recognition-settings",
        "version": 2,
        "recognition_settings": {"min_area": 42},
        "pipeline": ["metal"],
    }
    snapshot = _job_snapshot(state="succeeded", applied=applied)
    assert dict(snapshot.properties)["Параметры"] == {"recognition_settings": {"min_area": 42}, "pipeline": ["metal"]}
    assert len(snapshot.actions) == 1


def test_completed_contour_exports_original_applied_settings(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    import json

    applied = {
        "format": "contour-recognition-settings",
        "version": 2,
        "recognition_settings": {"min_area": 42},
        "pipeline": ["metal"],
    }
    snapshot = _job_snapshot(state="succeeded", applied=applied)
    target = tmp_path / "contour.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(target), "JSON (*.json)"))
    snapshot.actions[0][1]()
    assert json.loads(target.read_text(encoding="utf-8")) == applied


def test_property_table_formats_dates_and_hides_empty_rows(qtbot):
    snapshot = ObjectPropertiesSnapshot(
        "Объект",
        "job",
        (
            ("Кто добавил", "Mark"),
            ("Когда добавлен", "2026-07-28T10:15:15+00:00"),
            ("Примечание", None),
            ("Состояние", "waiting_for_user"),
        ),
    )
    dialog = ObjectPropertiesDialog(snapshot)
    qtbot.addWidget(dialog)
    rows = {
        dialog.properties_table.item(row, 0).text(): dialog.properties_table.item(row, 1).text()
        for row in range(dialog.properties_table.rowCount())
    }
    assert "Примечание" not in rows
    assert rows["Когда добавлен"].count(":") == 2
    assert rows["Когда добавлен"].startswith("28.07.2026 ")
    assert rows["Состояние"] == "Ожидает действий пользователя"


@pytest.mark.parametrize(
    ("name", "raw", "expected"),
    [
        ("Ориентация", "y_down", "Y вниз"),
        ("Тип слоя", "metal", "Металл"),
        ("Состояние", "active", "Активен"),
        ("Проверка", "checked", "Проверено"),
        ("Режим хранения", "managed_copy", "Файлы в проекте"),
        ("Размер", 1024, "1.0 КБ"),
        ("Создан", "2026-07-28T10:15:15+00:00", "28.07.2026 "),
    ],
)
def test_other_property_kinds_have_readable_values(qtbot, name, raw, expected):
    dialog = ObjectPropertiesDialog(ObjectPropertiesSnapshot("Объект", "project", ((name, raw),)))
    qtbot.addWidget(dialog)
    assert dialog.properties_table.item(0, 1).text().startswith(expected)


def test_node_properties_profile_finishes_while_modal_dialog_is_open(qtbot, tmp_path, monkeypatch):
    from kraken_core.frame_matrix import profiling
    from kraken_hub.presentation.qt.parts import open_properties

    (tmp_path / "kraken.ini").write_text("[profiling]\nnode_properties = true\n", encoding="utf-8")
    profiling.configure(tmp_path)
    profile = profiling.begin_region("node_properties", label="job")
    assert profile is not None
    profile.checkpoint("data_ready")

    real_dialog = open_properties.ObjectPropertiesDialog

    def dialog_factory(*args):
        dialog = real_dialog(*args)
        QTimer.singleShot(100, dialog.accept)
        return dialog

    monkeypatch.setattr(open_properties, "ObjectPropertiesDialog", dialog_factory)

    controller = object.__new__(DesktopController)
    controller._layer_dialog = None
    controller.shell = None
    controller._open_node_properties(ObjectPropertiesSnapshot("Задание", "job", (("Название", "Задание"),)), profile)
    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert "[node_properties] job" in report
    assert "data_ready:" in report
    assert "dialog_created:" in report
    assert "first_event_loop:" in report
    assert "ready=yes" in report
    assert profile.finished


def test_node_properties_profile_is_opt_in(tmp_path):
    from kraken_core.frame_matrix import profiling

    (tmp_path / "kraken.ini").write_text("[profiling]\nnode_properties = false\n", encoding="utf-8")
    profiling.configure(tmp_path)
    assert profiling.begin_region("node_properties") is None
    assert not (tmp_path / "profiling.log").exists()


def test_project_properties_show_storage_without_revision():
    project = SimpleNamespace(
        id="project-1",
        name="Test",
        width=70,
        height=2,
        orientation=SimpleNamespace(value="y_down"),
        storage_profile="local-filesystem",
        state=SimpleNamespace(value="active"),
        revision=7,
        created_at=datetime(2026, 7, 28, 10, tzinfo=UTC),
    )
    controller = object.__new__(DesktopController)
    controller.service = SimpleNamespace(get_project=lambda *_: project, history=lambda *_: ())
    controller._scope_content = lambda *_: {}
    snapshots = []
    controller._open_properties = snapshots.append
    controller.show_project_properties(SimpleNamespace(project_id=project.id))
    values = dict(snapshots[0].properties)
    assert values["Хранение"] == "Локально"
    assert "Ревизия" not in values
    assert values["Создан"] == "2026-07-28T10:00:00+00:00"


def test_file_properties_keep_versions_and_export_action():
    series = SimpleNamespace(id="series-1", project_id="project-1", name="CIF", scope=SimpleNamespace(value="frame"))
    version = SimpleNamespace(
        id="version-1",
        filename="frame.cif",
        size_bytes=32,
        sha256="abc",
        blob=b"data",
        external=None,
        author_principal_id="author-1",
        created_at=datetime(2026, 7, 28, 10, tzinfo=UTC),
        parent_version_id=None,
        tool_name="Contour",
        tool_version="2",
        parameters={"threshold": 0.5},
    )
    controller = object.__new__(DesktopController)
    controller.service = SimpleNamespace(
        list_principals=lambda **_: (SimpleNamespace(id="author-1", display_name="Mark"),),
        artifact_versions=lambda *_: (version,),
        history=lambda *_args, **_kwargs: (),
    )
    snapshots = []
    controller._open_properties = snapshots.append
    controller._show_frame_file_properties("Вектор", series, version)
    snapshot = snapshots[0]
    values = dict(snapshot.properties)
    assert values["Параметры создания"] == {"threshold": 0.5}
    assert "MIME-тип" not in values
    assert snapshot.versions[0]["filename"] == "frame.cif"
    assert snapshot.version_actions[0][0] == "Выгрузить"
