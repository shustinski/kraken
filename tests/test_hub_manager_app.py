from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QWidget,
)

from kraken_hub import windows_credentials
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.manager_app import (
    DesktopController,
    _count_regular_files,
    _development_session,
    _event_belongs_to_layer,
    _event_matches_node,
    _history_entries,
    _login,
    _my_work_panel,
    _performer_panel,
    _statistics_panel,
)
from kraken_hub.statistics_widgets import MetricChartWidget
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import LayerType, RepresentationKind
from kraken_manager.infrastructure.reports import ActivityRecord
from kraken_manager.presentation.qt import (
    FrameSelection,
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
    ProjectListItem,
    ProjectManagerShell,
)
from kraken_manager.presentation.qt.widgets import ClickableLabel


def test_matrix_context_menu_exports_and_opens_file_properties(
    qapp,
    monkeypatch,
) -> None:
    controller = object.__new__(DesktopController)
    controller.shell = QWidget()
    series = SimpleNamespace(name="frame.cif")
    version = SimpleNamespace(filename="frame.cif")
    controller._active_frame_files = lambda _x, _y: (
        ("CIF", series, version),
    )
    exported = []
    opened = []
    controller._export_selected_frame_files = exported.append
    controller._show_frame_file_properties = (
        lambda *values: opened.append(values)
    )
    context = SimpleNamespace(
        x=2,
        y=3,
        selection=FrameSelection.single(2, 3),
    )

    monkeypatch.setattr(
        QMenu,
        "exec",
        lambda menu, *_args: next(
            action
            for action in menu.actions()
            if action.text().startswith("Выгрузить файлы")
        ),
    )
    controller._show_matrix_context_menu(context, controller.shell.pos())
    assert exported == [context]

    def select_file_properties(menu: QMenu, *_args):
        submenu = next(
            action.menu()
            for action in menu.actions()
            if action.text() == "Свойства файла"
        )
        return submenu.actions()[0]

    monkeypatch.setattr(QMenu, "exec", select_file_properties)
    controller._show_matrix_context_menu(context, controller.shell.pos())
    assert opened == [("CIF", series, version)]


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


def test_performer_dialog_uses_color_picker(qapp, monkeypatch) -> None:
    created: list[dict[str, str]] = []

    class Service:
        @staticmethod
        def list_performers():
            return ()

        @staticmethod
        def create_manual_performer(**kwargs):
            created.append(kwargs)

    monkeypatch.setattr(
        QColorDialog,
        "getColor",
        lambda *_args: QColor("#123ABC"),
    )

    def complete_dialog(dialog: QDialog):
        name = dialog.findChild(QLineEdit)
        color = dialog.findChild(QPushButton, "performerColorPicker")
        assert name is not None
        assert color is not None
        assert color.text() == "#60A5FA"
        name.setText("Reviewer")
        color.click()
        assert color.text() == "#123ABC"
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", complete_dialog)
    panel = _performer_panel(Service())
    add_button = panel.findChild(QPushButton)
    assert add_button is not None
    add_button.click()

    assert created == [{"name": "Reviewer", "color": "#123ABC"}]


def test_performer_table_supports_color_editing_and_context_actions(qapp, monkeypatch) -> None:
    performers = [
        SimpleNamespace(
            id="performer-1",
            name="Reviewer",
            color="#60A5FA",
            principal_id=None,
        )
    ]
    updates: list[dict[str, str]] = []
    archived: list[str] = []

    class Service:
        @staticmethod
        def list_performers():
            return tuple(performers)

        @staticmethod
        def update_performer(**kwargs):
            updates.append(kwargs)
            current = performers[0]
            performers[0] = SimpleNamespace(
                id=current.id,
                name=kwargs["name"],
                color=kwargs["color"],
                principal_id=current.principal_id,
            )

        @staticmethod
        def archive_performer(performer_id):
            archived.append(performer_id)
            performers.clear()

    monkeypatch.setattr(QColorDialog, "getColor", lambda *_args: QColor("#123ABC"))
    panel = _performer_panel(Service())
    table = panel.findChild(QTableWidget)
    assert table is not None
    panel.show()
    qapp.processEvents()
    assert table.item(0, 1).text() == ""
    assert table.item(0, 1).data(Qt.ItemDataRole.UserRole) == "#60A5FA"
    color_center = table.visualItemRect(table.item(0, 1)).center()
    assert table.viewport().grab().toImage().pixelColor(color_center).name().upper() == "#60A5FA"

    table.cellClicked.emit(0, 1)
    assert updates[-1] == {
        "performer_id": "performer-1",
        "name": "Reviewer",
        "color": "#123ABC",
    }
    assert table.item(0, 1).data(Qt.ItemDataRole.UserRole) == "#123ABC"

    def complete_edit(dialog: QDialog):
        name = dialog.findChild(QLineEdit)
        assert name is not None
        name.setText("Edited reviewer")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", complete_edit)
    requested_action = {"text": "Изменить"}
    monkeypatch.setattr(
        QMenu,
        "exec",
        lambda menu, *_args: next(action for action in menu.actions() if action.text() == requested_action["text"]),
    )
    qapp.processEvents()
    position = table.visualItemRect(table.item(0, 0)).center()
    table.customContextMenuRequested.emit(position)
    assert updates[-1]["name"] == "Edited reviewer"

    requested_action["text"] = "Удалить"
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    table.customContextMenuRequested.emit(position)
    assert archived == ["performer-1"]
    assert table.rowCount() == 0


def test_statistics_panel_has_human_readable_project_filtered_charts(qapp) -> None:
    now = datetime.now(UTC)
    projects = (
        SimpleNamespace(
            id="project-active",
            name="Active project",
            state=SimpleNamespace(value="active"),
        ),
        SimpleNamespace(
            id="project-archived",
            name="Old project",
            state=SimpleNamespace(value="archived"),
        ),
    )
    records = (
        ActivityRecord("one", now, "artifact.imported", "project-active", bytes_count=1024),
        ActivityRecord("two", now, "artifact.imported", "project-archived", bytes_count=2048),
    )

    class Service:
        @staticmethod
        def list_projects(*, include_archived=False):
            assert include_archived
            return projects

        @staticmethod
        def activity_records():
            return records

    panel = _statistics_panel(Service())
    tabs = panel.findChild(QTabWidget, "statisticsTabs")
    table = panel.findChild(QTableWidget, "statisticsSummary")
    project = panel.findChild(QComboBox, "statisticsProjectFilter")
    assert tabs is not None
    assert table is not None
    assert project is not None
    assert tuple(tabs.tabText(index) for index in range(tabs.count())) == (
        "Сводка",
        "По дням",
        "По неделям",
        "По месяцам",
        "По годам",
    )
    assert table.rowCount() == 18
    assert table.item(0, 0).text() == "Импортировано файлов"
    assert table.item(0, 1).text() == "2"
    assert project.itemText(0) == "Все проекты"
    assert "архивный" in project.itemText(2)

    charts = panel.findChildren(MetricChartWidget)
    assert len(charts) == 72
    daily_imports = panel.findChild(MetricChartWidget, "statisticsChart_day_imported_files")
    assert daily_imports is not None
    assert daily_imports.total_text == "2"
    assert daily_imports.point_count >= 30

    project.setCurrentIndex(1)
    calculate = next(button for button in panel.findChildren(QPushButton) if button.text() == "Рассчитать")
    calculate.click()
    assert table.item(0, 1).text() == "1"
    assert daily_imports.total_text == "1"

    panel.show()
    tabs.setCurrentIndex(1)
    qapp.processEvents()
    assert not daily_imports.grab().isNull()
    panel.close()


def test_delete_project_confirmation_calls_cache_only_service(qapp, monkeypatch, tmp_path: Path) -> None:
    project = SimpleNamespace(id="project-1", name="Preserve files")
    source = tmp_path / "source" / project.name
    derived = tmp_path / "derived" / project.name
    source.mkdir(parents=True)
    derived.mkdir(parents=True)
    deleted = []

    class Service:
        @staticmethod
        def get_project(_project_id):
            return project

        @staticmethod
        def project_workspace(_project_id):
            return SimpleNamespace(
                source_project_dir=str(source),
                derived_project_dir=str(derived),
            )

        @staticmethod
        def delete_project(**kwargs):
            deleted.append(kwargs)

    shell = ProjectManagerShell()
    controller = object.__new__(DesktopController)
    controller.shell = shell
    controller.service = Service()
    controller.session = SimpleNamespace(principal=SimpleNamespace(id="principal-1"))
    controller.thumbnail_store_uri = ""
    controller._project_id = None
    controller._workspace = None
    controller._layer_dialog = None
    refreshed = []
    controller.refresh_projects = lambda: refreshed.append(True)
    controller._error = pytest.fail

    def confirm(dialog: QInputDialog):
        assert str(source) in dialog.labelText()
        assert str(derived) in dialog.labelText()
        dialog.setTextValue(project.name)
        return QDialog.DialogCode.Accepted

    messages = []
    monkeypatch.setattr(QInputDialog, "exec", confirm)
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *_args: messages.append(True),
    )

    controller.delete_project(
        ProjectListItem(project.id, project.name, 2, 2, "Local")
    )

    assert deleted == [
        {
            "principal": controller.session.principal,
            "project": project,
            "confirmation_name": project.name,
        }
    ]
    assert refreshed == [True]
    assert messages == [True]
    assert source.is_dir()
    assert derived.is_dir()


def test_controller_builds_layer_and_representation_properties(tmp_path: Path) -> None:
    image_directory = tmp_path / "images"
    image_directory.mkdir()
    (image_directory / "0.jpg").write_bytes(b"image")
    layer = SimpleNamespace(
        id="layer-1",
        project_id="project-1",
        name="Metal 1",
        type=SimpleNamespace(value="metal"),
        order=2,
        state=SimpleNamespace(value="active"),
        revision=3,
        created_at=datetime(2026, 7, 28, 10, 0, tzinfo=UTC),
    )
    binding = SimpleNamespace(
        image_directory=str(image_directory),
        ssc_directory="",
        prv_directory="",
        aux_directory="",
        import_root=str(tmp_path),
        mode=SimpleNamespace(value="managed_copy"),
        frame_positions={"0.jpg": 1},
        conversion=SimpleNamespace(target_format="jpg", jpeg_quality=95),
    )
    representation = SimpleNamespace(
        id="representation-1",
        project_id="project-1",
        layer_id="layer-1",
        name="Исходные изображения",
        kind=SimpleNamespace(value="image"),
        purpose=SimpleNamespace(value="source"),
        note="Проверено",
        source=str(image_directory),
        source_image_representation_id=None,
        active=True,
        state=SimpleNamespace(value="active"),
        revision=1,
        created_at=datetime(2026, 7, 28, 10, 1, tzinfo=UTC),
    )
    events = (
        _event("LayerCreated", {"layer_id": "layer-1"}),
        _event(
            "RepresentationCreated",
            {
                "layer_id": "layer-1",
                "representation_id": "representation-1",
                "note": "Проверено",
            },
            minute=1,
            event_id="event-2",
            stream_id="layer:layer-1",
        ),
    )

    class Service:
        @staticmethod
        def list_layers(_project_id):
            return (layer,)

        @staticmethod
        def list_representations(_project_id, _layer_id):
            return (representation,)

        @staticmethod
        def layer_file_binding(_project_id, _layer_id):
            return binding

        @staticmethod
        def history(_project_id):
            return events

        @staticmethod
        def frame_cells(_project_id, _layer_id, _representation_id):
            return (SimpleNamespace(frame_id="frame-1"),)

        @staticmethod
        def list_derived_runs(_project_id, _layer_id):
            return ()

    controller = object.__new__(DesktopController)
    controller.service = Service()
    controller._project_id = "project-1"
    controller._pipeline_snapshot = lambda *_args: LayerPipelineSnapshot(
        "project-1",
        "layer-1",
        (
            PipelineLane(
                "representation-1",
                "Исходные изображения",
                (
                    PipelineNode(
                        "representation-1",
                        "Исходные изображения",
                        "source",
                        representation_id="representation-1",
                    ),
                ),
                (),
            ),
        ),
    )
    snapshots = []
    controller._open_properties = snapshots.append

    controller._show_layer_properties("layer-1")
    layer_properties = dict(snapshots[-1].properties)
    assert layer_properties["Путь"] == str(image_directory)
    assert layer_properties["Количество файлов"] == 1
    assert layer_properties["Кто добавил"] == "Оператор"
    assert len(snapshots[-1].history) == 2

    node = PipelineNode(
        "representation-1",
        "Исходные изображения",
        "source",
        representation_id="representation-1",
    )
    controller._show_node_properties("layer-1", node)
    representation_properties = dict(snapshots[-1].properties)
    assert representation_properties["Примечание"] == "Проверено"
    assert representation_properties["Количество файлов"] == 1
    assert representation_properties["Ревизия"] == 1
    assert [item.event_type for item in snapshots[-1].history] == [
        "RepresentationCreated"
    ]


def test_frame_card_hides_mutating_actions_without_permissions() -> None:
    class Combo:
        @staticmethod
        def currentData():
            return None

    class Matrix:
        @staticmethod
        def selected_coordinates(*, maximum):
            assert maximum == 2
            return ((1, 1),)

        @staticmethod
        def cell_data(_x, _y):
            return SimpleNamespace(
                payload={"frame_id": "frame-1"},
                status="ready",
                performer_initials="",
            )

    class Service:
        permissions = frozenset()

        @staticmethod
        def get_project(_project_id, *, as_of=None):
            del as_of
            return SimpleNamespace(name="Project", width=1)

        @classmethod
        def project_permissions(cls, _project_id, _principal):
            return cls.permissions

        @staticmethod
        def list_layers(_project_id):
            return (SimpleNamespace(id="layer-1", name="Layer"),)

        @staticmethod
        def list_representations(_project_id, _layer_id):
            return ()

        @staticmethod
        def history(_project_id):
            return ()

        @staticmethod
        def list_principals(*, include_inactive):
            assert include_inactive
            return ()

        @staticmethod
        def list_artifact_series(
            _project_id,
            *,
            layer_id,
            include_archived,
        ):
            assert layer_id == "layer-1"
            assert include_archived
            return ()

        @staticmethod
        def list_notes(_project_id, *, layer_id, frame_id):
            assert (layer_id, frame_id) == ("layer-1", "frame-1")
            return ()

    workspace = SimpleNamespace(
        _selected_layer_id="layer-1",
        matrix_view=Matrix(),
        image_representation_combo=Combo(),
        vector_representation_combo=Combo(),
    )
    controller = object.__new__(DesktopController)
    controller.service = Service()
    controller.session = SimpleNamespace(principal=SimpleNamespace(id="viewer"))
    controller._workspace = workspace
    controller._project_id = "project-1"
    snapshots = []
    controller._open_properties = snapshots.append

    controller.show_selected_frame()
    viewer_snapshot = snapshots[-1]
    assert viewer_snapshot.actions == ()
    assert viewer_snapshot.file_actions == ()
    assert [label for label, _callback in viewer_snapshot.version_actions] == [
        "Экспортировать / открыть",
        "Проверить внешний файл",
    ]

    Service.permissions = frozenset(
        {Permission.ADD_NOTE, Permission.IMPORT_ARTIFACT}
    )
    controller.show_selected_frame()
    editor_snapshot = snapshots[-1]
    assert len(editor_snapshot.actions) == 2
    assert len(editor_snapshot.file_actions) == 4
    assert [label for label, _callback in editor_snapshot.version_actions] == [
        "Активировать",
        "Экспортировать / открыть",
        "Проверить внешний файл",
    ]


def test_my_work_exposes_review_and_agent_recovery_actions(qapp) -> None:
    now = datetime.now(UTC)
    batch = SimpleNamespace(
        id="batch-1",
        project_id="project-1",
        layer_id="layer-1",
        assignee_id="performer-1",
        state=SimpleNamespace(value="awaiting_acceptance"),
        due_at=None,
        items=(object(),),
    )
    recovery_job = SimpleNamespace(
        id="job-1",
        project_id="project-1",
        layer_id="layer-1",
        state=SimpleNamespace(value="recovery_required"),
        updated_at=now,
        progress=0.5,
        error="agent stopped",
    )
    partial_job = SimpleNamespace(
        id="job-2",
        project_id="project-1",
        layer_id="layer-1",
        state=SimpleNamespace(value="partial"),
        updated_at=now,
        progress=0.75,
        error="",
    )

    class Service:
        @staticmethod
        def list_projects(*, include_archived):
            assert include_archived
            return (SimpleNamespace(id="project-1", name="Project"),)

        @staticmethod
        def project_permissions(_project_id, _principal):
            return frozenset(
                {
                    Permission.ASSIGN_WORK,
                    Permission.ACCEPT_REVIEW,
                    Permission.MANAGE_REVIEW,
                    Permission.RUN_PLUGIN,
                }
            )

        @staticmethod
        def list_layers(_project_id, *, include_archived):
            assert include_archived
            return (SimpleNamespace(id="layer-1", name="Layer"),)

        @staticmethod
        def list_performers(*, include_archived):
            assert include_archived
            return (SimpleNamespace(id="performer-1", name="Worker"),)

        @staticmethod
        def review_batches():
            return (batch,)

        @staticmethod
        def plugin_jobs():
            return (recovery_job, partial_job)

    controller = SimpleNamespace(
        cancel_agent_job=lambda _job: None,
        retry_agent_job=lambda _job: None,
        import_partial_agent_job=lambda _job: None,
        my_work_refresh=None,
    )
    panel = _my_work_panel(
        Service(),
        SimpleNamespace(principal=SimpleNamespace(id="owner")),
        controller,
    )
    table = panel.findChild(QTableWidget)
    assert table is not None
    assert table.rowCount() == 3
    assert table.item(0, 4).text() == "Ожидает принятия"
    assert table.item(1, 4).text() == "Требуется восстановление"
    assert table.item(2, 4).text() == "Частичный результат"
    assert {
        button.text()
        for button in table.cellWidget(0, 7).findChildren(QPushButton)
    } == {"Повторно выгрузить", "Принять", "На доработку", "Отменить"}
    assert {
        button.text()
        for button in table.cellWidget(1, 7).findChildren(QPushButton)
    } == {"Повторить", "Отменить"}
    assert {
        button.text()
        for button in table.cellWidget(2, 7).findChildren(QPushButton)
    } == {"Импортировать", "Отменить"}
    assert controller.my_work_refresh is not None


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
    assert dict(snapshots[-1].properties)["Capability"] == "dataset.model.train.v1"

    controller._show_node_properties("layer-1", karakal)
    assert snapshots[-1].history[0].event_type == "KarakalAnalysisPublished"

    controller._show_node_properties("layer-1", missing)
    assert snapshots[-1].history == ()

    blackbox = PipelineNode("source-1:blackbox", "Чёрный ящик", "blackbox")
    controller._show_node_properties("layer-1", blackbox)
    blackbox_properties = dict(snapshots[-1].properties)
    assert blackbox_properties["Количество скрытых этапов"] == 3
    assert len(snapshots[-1].history) == 3


def test_login_creates_first_account_in_dialog(qapp, monkeypatch, tmp_path) -> None:
    service = EmbeddedProjectService(tmp_path)
    saved = []
    monkeypatch.setattr(windows_credentials, "credentials_available", lambda: False)
    monkeypatch.setattr(windows_credentials, "save_credentials", lambda *values: saved.append(values))

    def complete_dialog(dialog: QDialog) -> QDialog.DialogCode:
        assert dialog.objectName() == "initialAccountDialog"
        values = {
            "initialAccountUsername": "operator",
            "initialAccountDisplayName": "Оператор",
            "initialAccountPassword": "",
            "initialAccountPasswordConfirmation": "",
        }
        for object_name, value in values.items():
            field = dialog.findChild(QLineEdit, object_name)
            assert field is not None
            field.setText(value)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", complete_dialog)

    session = _login(None, service)

    assert session is not None
    assert service.has_accounts
    assert session.principal.subject == "operator"
    assert service.login("operator", "") is not None
    assert saved == [("operator", "")]


def test_development_session_creates_and_reuses_vscode_account(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KRAKEN_DEV_AUTO_LOGIN", "1")
    monkeypatch.setenv("KRAKEN_DEV_USERNAME", "vscode")
    monkeypatch.setenv("KRAKEN_DEV_PASSWORD", "")
    service = EmbeddedProjectService(tmp_path)

    created = _development_session(service)
    reopened = _development_session(EmbeddedProjectService(tmp_path))

    assert created is not None and reopened is not None
    assert created.principal.id == reopened.principal.id
    assert created.principal.subject == "vscode"


def test_login_autofills_only_after_windows_verification(qapp, monkeypatch, tmp_path) -> None:
    service = EmbeddedProjectService(tmp_path)
    service.create_initial_account("operator", "Operator", "secret")
    verification_windows = []
    saved = []
    monkeypatch.setattr(windows_credentials, "credentials_available", lambda: True)
    monkeypatch.setattr(windows_credentials, "load_credentials", lambda: ("operator", "secret"))
    monkeypatch.setattr(
        windows_credentials,
        "verify_windows_identity",
        lambda window: verification_windows.append(window) or True,
    )
    monkeypatch.setattr(windows_credentials, "save_credentials", lambda *values: saved.append(values))

    def accept_autofilled_dialog(dialog: QDialog) -> QDialog.DialogCode:
        assert dialog.objectName() == "loginDialog"
        username = dialog.findChild(QLineEdit, "loginUsername")
        password = dialog.findChild(QLineEdit, "loginPassword")
        assert username is not None and username.text() == "operator"
        assert password is not None and password.text() == "secret"
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", accept_autofilled_dialog)

    session = _login(None, service)

    assert session is not None
    assert verification_windows and verification_windows[0] != 0
    assert saved == [("operator", "secret")]


def test_login_does_not_autofill_when_windows_verification_is_cancelled(qapp, monkeypatch, tmp_path) -> None:
    service = EmbeddedProjectService(tmp_path)
    service.create_initial_account("operator", "Operator", "secret")
    monkeypatch.setattr(windows_credentials, "credentials_available", lambda: True)
    monkeypatch.setattr(windows_credentials, "verify_windows_identity", lambda _window: False)
    monkeypatch.setattr(
        windows_credentials,
        "load_credentials",
        lambda: pytest.fail("credentials must not be loaded before verification"),
    )

    def cancel_empty_dialog(dialog: QDialog) -> QDialog.DialogCode:
        username = dialog.findChild(QLineEdit, "loginUsername")
        password = dialog.findChild(QLineEdit, "loginPassword")
        assert username is not None and username.text() == ""
        assert password is not None and password.text() == ""
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", cancel_empty_dialog)

    assert _login(None, service) is None


def test_image_representation_source_picker_fills_selected_folder(qapp, monkeypatch) -> None:
    class ServiceStub:
        @staticmethod
        def get_project(_project_id):
            return object()

        @staticmethod
        def list_layers(_project_id):
            return (type("LayerStub", (), {"id": "layer-1"})(),)

    controller = object.__new__(DesktopController)
    controller.service = ServiceStub()
    workspace = QWidget()
    workspace._selected_layer_id = "layer-1"
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args: "C:/images",
    )

    def use_picker_and_cancel(dialog: QDialog) -> QDialog.DialogCode:
        picker = dialog.findChild(ClickableLabel, "representationSourceFolderPicker")
        source = dialog.findChild(QLineEdit, "representationSource")
        assert picker is not None and source is not None
        assert picker.minimumWidth() > 0
        assert picker.text() == "Выбрать папку…"
        picker.clicked.emit()
        assert source.text() == "C:/images"
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", use_picker_and_cancel)

    controller._add_representation(workspace, "project-1", RepresentationKind.IMAGE)


def test_remote_project_can_create_logical_layer(qapp, monkeypatch) -> None:
    project = SimpleNamespace(id="project-1")
    created = SimpleNamespace(id="layer-1")
    calls = []

    class ServiceStub:
        @staticmethod
        def get_project(_project_id):
            return project

        @staticmethod
        def project_workspace(_project_id):
            return None

        @staticmethod
        def is_remote_project(_project_id):
            return True

        @staticmethod
        def list_layers(_project_id):
            return ()

        @staticmethod
        def create_layer(**kwargs):
            calls.append(kwargs)
            return created

    controller = object.__new__(DesktopController)
    controller.service = ServiceStub()
    controller.session = SimpleNamespace(principal=object())
    controller.shell = QWidget()
    shown = []
    controller._show_created_layer = lambda *args: shown.append(args)
    workspace = QWidget()

    monkeypatch.setattr(QInputDialog, "getText", lambda *_args, **_kwargs: ("Metal", True))
    monkeypatch.setattr(
        QInputDialog,
        "getItem",
        lambda *_args, **_kwargs: (_args[3][0], True),
    )

    controller._add_layer(workspace, project.id)

    assert calls[0]["layer_type"] is LayerType.METAL
    assert calls[0]["order"] == 0
    assert shown == [(workspace, project.id, created)]


def test_external_cif_import_uses_source_from_clicked_pipeline_lane(monkeypatch) -> None:
    controller = object.__new__(DesktopController)
    controller._workspace = object()
    controller._project_id = "project-1"
    missing = PipelineNode("missing-cif", "CIF не получен", "missing")
    controller._pipeline_snapshot = lambda *_args: LayerPipelineSnapshot(
        "project-1",
        "layer-1",
        (
            PipelineLane(
                "source-representation-2",
                "Source 2",
                (
                    PipelineNode("source-representation-2", "Source 2", "source"),
                    missing,
                ),
                (("source-representation-2", "missing-cif"),),
            ),
        ),
    )
    calls = []
    monkeypatch.setattr(
        controller,
        "_add_representation",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    controller._layer_manager_action("layer-1", missing, "add_external_vector")

    assert calls
    assert calls[0][0][2] is RepresentationKind.VECTOR
    assert calls[0][1]["source_image_id"] == "source-representation-2"


def test_created_layer_is_selected_and_its_pipeline_is_loaded(monkeypatch) -> None:
    controller = object.__new__(DesktopController)
    layer = SimpleNamespace(id="layer-created")
    layer_item = SimpleNamespace(layer_id="layer-created")
    selected = []
    loaded = []

    class ServiceStub:
        @staticmethod
        def get_project(project_id):
            return SimpleNamespace(id=project_id)

    class LayerModelStub:
        @staticmethod
        def layer_by_id(layer_id):
            return layer_item if layer_id == "layer-created" else None

    class TabsStub:
        current = 0

        @staticmethod
        def count():
            return 2

        @staticmethod
        def tabData(index):
            return ("layer-old", "layer-created")[index]

        @classmethod
        def currentIndex(cls):
            return cls.current

        @classmethod
        def setCurrentIndex(cls, index):
            cls.current = index

    workspace = SimpleNamespace(
        layer_model=LayerModelStub(),
        layer_tabs=TabsStub(),
    )
    controller.service = ServiceStub()
    monkeypatch.setattr(
        controller,
        "_load_layers",
        lambda current_workspace, project: loaded.append(
            (current_workspace, project.id)
        ),
    )
    monkeypatch.setattr(
        controller,
        "_select_layer",
        lambda current_workspace, project_id, item: selected.append(
            (current_workspace, project_id, item)
        ),
    )

    controller._show_created_layer(workspace, "project-1", layer)

    assert loaded == [(workspace, "project-1")]
    assert TabsStub.current == 1
    assert selected == [(workspace, "project-1", layer_item)]


def test_contour_vectorize_receives_staged_base_layer_path(tmp_path: Path) -> None:
    representation = SimpleNamespace(
        id="binary-representation-1",
        kind=RepresentationKind.IMAGE,
        source="managed-import",
    )

    class ServiceStub:
        data_dir = tmp_path

        @staticmethod
        def list_representations(_project_id, _layer_id):
            return (representation,)

        @staticmethod
        def frame_cells(_project_id, _layer_id, _representation_id):
            return (
                SimpleNamespace(x=3, y=4, sha256="a" * 64),
            )

        @staticmethod
        def read_project_blob(_project_id, _sha256):
            return b"image"

    controller = object.__new__(DesktopController)
    controller.service = ServiceStub()
    controller._project_id = "project-1"
    node = PipelineNode(
        "binary-representation-1",
        "Binary",
        "binary",
        representation_id="binary-representation-1",
    )

    arguments, parameters = controller._contour_launch_arguments(
        layer_id="layer-1",
        node=node,
        action="vectorize",
        source_representation_id="source-representation-1",
    )

    input_directory = Path(arguments[arguments.index("--input-dir") + 1])
    output_directory = Path(arguments[arguments.index("--output-dir") + 1])
    assert input_directory.is_dir()
    assert (input_directory / "3_4.png").read_bytes() == b"image"
    assert output_directory.is_dir()
    assert parameters["input_representation_id"] == "binary-representation-1"
