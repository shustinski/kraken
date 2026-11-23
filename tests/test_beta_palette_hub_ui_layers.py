"""Палитра W3 — слои, варианты изображений, контекст матрицы (live UI). S11."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QContextMenuEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QWidget,
)

from kraken_core.plugins import build_plugin_inventory, scan_plugin_directory
from kraken_hub.composition import DesktopSession, EmbeddedProjectService
from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
from kraken_hub.presentation.qt import preferences_dialog
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_manager.presentation.qt import ProjectManagerShell
from kraken_manager.presentation.qt.layer_creation import LayerCreationDialog
from tests.test_beta_palette_hub_ui_project_cycle import _isolate_settings, _ui_config
from tests.test_live_ui_workflow import accept_dialog, click, modal, node_menu, select_catalog_project

SOURCE = Path(r"D:\code\kraken\.test_runtime\palette-source")


def _png_folder(root: Path, names: tuple[str, ...]) -> Path:
    root.mkdir(parents=True)
    for name in names:
        shutil.copy(SOURCE / name, root / name)
    return root


def _jpeg(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (8, 8), "white").save(path, format="JPEG")


@pytest.fixture
def hub(monkeypatch, tmp_path, qtbot):
    config = _ui_config()
    _isolate_settings(monkeypatch, tmp_path)
    remote = RemoteServerProjectService.authenticate_local(
        config["server_url"], username=config["username"], password=config["password"], data_dir=tmp_path / "client",
    )
    local = EmbeddedProjectService(tmp_path / "local")
    local.identities.save(remote.auth.principal)
    service = DualCatalogService(local, remote)
    account = preferences_dialog.ProgramAccount(
        "palette-s11", "local", config["username"], config["username"], config["password"], True,
    )
    monkeypatch.setattr(preferences_dialog, "program_server_url", lambda *_args, **_kwargs: config["server_url"])
    monkeypatch.setattr(preferences_dialog, "default_program_account", lambda *_args, **_kwargs: account)
    shell = ProjectManagerShell()
    qtbot.addWidget(shell)
    session = DesktopSession(remote.auth.access_token, remote.auth.principal, "")
    controller = DesktopController(
        shell, service, session, thumbnail_store_uri="memory://",
        plugin_items=build_plugin_inventory(scan_plugin_directory(Path(__file__).resolve().parents[1] / "plugins")),
    )
    controller._agent_timer.stop()
    shell.show()
    messages: list[str] = []
    timer = QTimer(shell)
    timer.setInterval(20)

    def close_messages():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            messages.append(dialog.text())
            button = dialog.button(QMessageBox.StandardButton.Yes) or dialog.button(QMessageBox.StandardButton.Ok)
            if button is not None:
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            return
        if (
            dialog is not None
            and dialog.objectName() == "representationDialog"
            and any("не пройдена" in text for text in messages)
        ):
            dialog.reject()

    timer.timeout.connect(close_messages)
    timer.start()
    try:
        yield type("Hub", (), {"controller": controller, "remote": remote, "messages": messages, "tmp_path": tmp_path})()
    finally:
        remote.stop_sync()
        shell.close_guard = None
        shell.close()
        QApplication.processEvents()
        timer.stop()
        controller._agent_timer.stop()
        for project in remote.list_projects(include_archived=True):
            if str(project.name).startswith("palette-s11") and project.state.value != "archived":
                remote.archive_project(
                    principal=remote.auth.principal, project=project, idempotency_key=str(uuid4()),
                )


def _create_server_project(controller, name: str) -> None:
    def fill(dialog):
        dialog.findChild(QLineEdit, "projectName").setText(name)
        profiles = dialog.findChild(QComboBox)
        profiles.setCurrentIndex(profiles.findText("Сервер"))
        accept_dialog(dialog)

    modal(lambda: click(controller.catalog_page, "Создать проект"), fill)


def _layers_dialog(controller, qtbot):
    controller.shell.layers_action.trigger()
    qtbot.waitUntil(lambda: controller._layer_dialog is not None and controller._layer_dialog.isVisible(), timeout=15000)
    return controller._layer_dialog


def _add_layer(controller, qtbot, folder: Path, layer_name: str) -> None:
    dialog = _layers_dialog(controller, qtbot)

    def fill(layer_dialog):
        assert isinstance(layer_dialog, LayerCreationDialog)
        layer_dialog.name_edit.setText(layer_name)
        layer_dialog.disk_source.edit.setText(str(folder))
        QTest.mouseClick(layer_dialog.scan_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: layer_dialog.scan_result is not None and layer_dialog.scan_result.ready, timeout=30000)
        layer_dialog.tabs.setCurrentWidget(layer_dialog.manual_tab)
        layer_dialog.manual_images.edit.setText(str(folder))
        accept_dialog(layer_dialog)

    modal(lambda: QTest.mouseClick(dialog.add_layer_button, Qt.MouseButton.LeftButton), fill)
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=60000)


def _project(remote, name: str):
    return next(project for project in remote.list_projects(include_archived=True) if project.name == name)


def _event_type(event) -> str:
    return str(getattr(event, "event_type", None) or (event.get("event_type") if isinstance(event, dict) else ""))


def _popup(action, handle) -> None:
    errors: list[BaseException] = []
    timer = QTimer()
    timer.setInterval(20)

    def poll():
        menu = QApplication.activePopupWidget()
        if not isinstance(menu, QMenu):
            return
        timer.stop()
        try:
            handle(menu)
        except Exception as exc:
            errors.append(exc)
            menu.close()

    timer.timeout.connect(poll)
    timer.start()
    try:
        action()
    finally:
        timer.stop()
    if errors:
        raise errors[0]


def _click_menu(menu: QMenu, label: str) -> None:
    action = next(item for item in menu.actions() if item.text() == label)
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())


def test_s11_two_projects_two_layers_each(hub, qtbot):
    controller = hub.controller
    left = f"palette-s11a-{uuid4().hex[:8]}"
    right = f"palette-s11b-{uuid4().hex[:8]}"
    _create_server_project(controller, left)
    _create_server_project(controller, right)
    folders = {
        (left, "Металл"): _png_folder(hub.tmp_path / "a1", ("001.png", "002.png")),
        (left, "Диэлектрик"): _png_folder(hub.tmp_path / "a2", ("003.png", "004.png")),
        (right, "Металл"): _png_folder(hub.tmp_path / "b1", ("001.png", "002.png")),
        (right, "Диэлектрик"): _png_folder(hub.tmp_path / "b2", ("003.png", "004.png")),
    }
    for name in (left, right):
        select_catalog_project(controller, name, open_workspace=True)
        qtbot.waitUntil(lambda: controller._workspace is not None, timeout=15000)
        for layer_name in ("Металл", "Диэлектрик"):
            _add_layer(controller, qtbot, folders[(name, layer_name)], layer_name)
        assert controller._workspace.layer_tabs.count() == 2
        project = _project(hub.remote, name)
        assert {layer.name for layer in hub.remote.list_layers(project.id)} == {"Металл", "Диэлектрик"}

    dialog = _layers_dialog(controller, qtbot)
    listing = dialog.layer_list
    before = listing.layer_ids()
    moved = listing.takeItem(0)
    listing.insertItem(listing.count(), moved)
    desired = tuple(listing.layer_ids())
    assert desired != before
    listing.orderChanged.emit(desired)
    qtbot.waitUntil(
        lambda: any(_event_type(event) == "LayersReordered" for event in hub.remote.history(controller._project_id)),
        timeout=15000,
    )


def test_s11_two_variants_switch_primary_and_context_menu(hub, qtbot):
    controller = hub.controller
    name = f"palette-s11a-{uuid4().hex[:8]}"
    _create_server_project(controller, name)
    select_catalog_project(controller, name, open_workspace=True)
    qtbot.waitUntil(lambda: controller._workspace is not None, timeout=15000)
    _add_layer(controller, qtbot, _png_folder(hub.tmp_path / "v1", ("001.png", "002.png")), "Металл")
    extra = _png_folder(hub.tmp_path / "v2", ("003.png", "004.png"))

    def fill_variant(form):
        form.findChild(QLineEdit, "representationName").setText("Второй вариант")
        picker = form.findChild(QWidget, "representationSourceFolderPicker")
        picker.setProperty("sourceDirectory", str(extra))
        accept_dialog(form)

    modal(controller._workspace.add_image_representation_button.click, fill_variant)
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=60000)
    project = _project(hub.remote, name)
    layer = hub.remote.list_layers(project.id)[0]
    images = [item for item in hub.remote.list_representations(project.id, layer.id)]
    assert len(images) >= 2
    second = next(item for item in images if item.name == "Второй вариант")
    from tests.test_live_server_plugin_chain import _select_representation_in_graph

    _select_representation_in_graph(controller.shell, controller, second.id, qtbot)
    assert str(controller._workspace.image_representation_combo.currentData()) == str(second.id)

    labels: list[tuple[str, bool]] = []

    def capture(menu):
        labels.extend((action.text(), action.isEnabled()) for action in menu.actions() if action.text())
        menu.close()

    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameSelection

    view = controller._workspace.matrix_view
    view.set_selection(FrameSelection.single(1, 1))
    view.ensureVisible(view.scene_rect_for_frame(1, 1))
    point = view.mapFromScene(view.scene_rect_for_frame(1, 1).center())
    event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, point, view.viewport().mapToGlobal(point))
    _popup(lambda: QApplication.sendEvent(view.viewport(), event), capture)
    texts = [label for label, _enabled in labels]
    assert any(label.startswith("Выгрузить файлы") for label in texts), texts
    assert "Выдать работу…" in texts
    assert "Свойства кадра…" in texts
    vector = next(((label, enabled) for label, enabled in labels if label == "Свойства вектора…"), None)
    assert vector is None or vector[1] is False


def test_s11_lifecycle_rename_note_activate_archive_deactivate(hub, qtbot):
    controller = hub.controller
    name = f"palette-s11a-{uuid4().hex[:8]}"
    _create_server_project(controller, name)
    select_catalog_project(controller, name, open_workspace=True)
    qtbot.waitUntil(lambda: controller._workspace is not None, timeout=15000)
    _add_layer(controller, qtbot, _png_folder(hub.tmp_path / "life", ("001.png", "002.png")), "Металл")
    project = _project(hub.remote, name)
    layer = hub.remote.list_layers(project.id)[0]
    representation = hub.remote.list_representations(project.id, layer.id)[0]
    _layers_dialog(controller, qtbot)
    renamed = "Вариант переименован"

    def fill_rename(form):
        form.findChild(QLineEdit).setText(renamed)
        form.accept()

    modal(
        lambda: node_menu(controller, "source", "Переименовать…", representation_id=str(representation.id)),
        fill_rename,
    )
    qtbot.waitUntil(
        lambda: hub.remote.list_representations(project.id, layer.id)[0].name == renamed,
        timeout=15000,
    )

    def fill_note(form):
        editor = form.findChild(QTextEdit) or form.findChild(QPlainTextEdit)
        editor.setPlainText("заметка палитры")
        form.accept()

    modal(
        lambda: node_menu(controller, "source", "Изменить заметку…", representation_id=str(representation.id)),
        fill_note,
    )
    qtbot.waitUntil(
        lambda: "заметка палитры" in (hub.remote.list_representations(project.id, layer.id)[0].note or ""),
        timeout=15000,
    )
    node_menu(controller, "source", "Деактивировать", representation_id=str(representation.id))
    qtbot.waitUntil(lambda: hub.remote.list_representations(project.id, layer.id)[0].active is False, timeout=15000)
    node_menu(controller, "source", "Активировать", representation_id=str(representation.id))
    qtbot.waitUntil(lambda: hub.remote.list_representations(project.id, layer.id)[0].active is True, timeout=15000)
    node_menu(controller, "source", "Архивировать", representation_id=str(representation.id))

    def recorded() -> set[str]:
        return {_event_type(event) for event in hub.remote.history(project.id)}

    qtbot.waitUntil(lambda: "RepresentationArchived" in recorded(), timeout=15000)
    assert {"RepresentationRenamed", "RepresentationNoteUpdated", "RepresentationDeactivated", "RepresentationActivated"} <= recorded()


def test_s11_use_then_delete_layer_project_stays_readable(hub, qtbot):
    controller = hub.controller
    name = f"palette-s11a-{uuid4().hex[:8]}"
    _create_server_project(controller, name)
    select_catalog_project(controller, name, open_workspace=True)
    qtbot.waitUntil(lambda: controller._workspace is not None, timeout=15000)
    _add_layer(controller, qtbot, _png_folder(hub.tmp_path / "d1", ("001.png", "002.png")), "Металл")
    _add_layer(controller, qtbot, _png_folder(hub.tmp_path / "d2", ("003.png", "004.png")), "Диэлектрик")
    dialog = _layers_dialog(controller, qtbot)
    listing = dialog.layer_list

    def open_delete():
        item = next(listing.item(row) for row in range(listing.count()) if listing.item(row).text() == "Металл")
        listing.customContextMenuRequested.emit(listing.visualItemRect(item).center())

    def fill_name(form, value):
        form.findChild(QLineEdit).setText(value)
        form.accept()

    modal(lambda: _popup(open_delete, lambda menu: _click_menu(menu, "Удалить слой…")), lambda form: fill_name(form, "Мет"))
    qtbot.waitUntil(lambda: any("не совпадает с названием слоя" in text for text in hub.messages), timeout=5000)
    project = _project(hub.remote, name)
    assert {layer.name for layer in hub.remote.list_layers(project.id)} == {"Металл", "Диэлектрик"}
    modal(lambda: _popup(open_delete, lambda menu: _click_menu(menu, "Удалить слой…")), lambda form: fill_name(form, "Металл"))
    qtbot.waitUntil(
        lambda: {layer.name for layer in hub.remote.list_layers(project.id)} == {"Диэлектрик"},
        timeout=15000,
    )
    select_catalog_project(controller, name, open_workspace=True)
    qtbot.waitUntil(lambda: controller._workspace is not None and controller._workspace.layer_tabs.count() == 1, timeout=15000)


def test_s11_mixed_png_jpg_scan_blocked(hub, qtbot):
    controller = hub.controller
    name = f"palette-s11a-{uuid4().hex[:8]}"
    _create_server_project(controller, name)
    select_catalog_project(controller, name, open_workspace=True)
    qtbot.waitUntil(lambda: controller._workspace is not None, timeout=15000)
    folder = hub.tmp_path / "mixed"
    _png_folder(folder, ("001.png",))
    _jpeg(folder / "002.jpg")
    dialog = _layers_dialog(controller, qtbot)

    def fill(layer_dialog):
        assert isinstance(layer_dialog, LayerCreationDialog)
        layer_dialog.name_edit.setText("Смесь")
        layer_dialog.disk_source.edit.setText(str(folder))
        QTest.mouseClick(layer_dialog.scan_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: layer_dialog.scan_result is not None, timeout=30000)
        assert layer_dialog.scan_result.ready is False
        assert any("одновременно найдены" in issue for issue in layer_dialog.scan_result.issues)
        layer_dialog.reject()

    modal(lambda: QTest.mouseClick(dialog.add_layer_button, Qt.MouseButton.LeftButton), fill)


def test_s11_unsupported_representation_import_rejected(hub, qtbot):
    controller = hub.controller
    name = f"palette-s11a-{uuid4().hex[:8]}"
    _create_server_project(controller, name)
    select_catalog_project(controller, name, open_workspace=True)
    qtbot.waitUntil(lambda: controller._workspace is not None, timeout=15000)
    _add_layer(controller, qtbot, _png_folder(hub.tmp_path / "base", ("001.png", "002.png")), "Металл")
    foreign = hub.tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "001.txt").write_text("не изображение", encoding="utf-8")
    def fill(form):
        form.findChild(QLineEdit, "representationName").setText("Текст")
        form.findChild(QWidget, "representationSourceFolderPicker").setProperty("sourceDirectory", str(foreign))
        accept_dialog(form)

    modal(controller._workspace.add_image_representation_button.click, fill)
    assert any("Импорт заблокирован" in text or "не пройдена" in text for text in hub.messages)
