"""User actions on persisted roles and performers, beyond service-only checks."""
from types import SimpleNamespace

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QFileDialog, QLineEdit, QMessageBox, QPushButton, QTableWidget

from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.presentation.qt.manager_app import DesktopController, _performer_panel
from kraken_manager.domain.identity import Principal, ProjectRole
from kraken_manager.domain.project import GridOrientation
from kraken_manager.presentation.qt.role_management import RoleManagementDialog
from tests.test_work_orders import _service


def message_pump():
    messages = []
    timer = QTimer()
    timer.setInterval(10)

    def dismiss():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            messages.append((dialog.windowTitle(), dialog.text()))
            dialog.accept()

    timer.timeout.connect(dismiss)
    timer.start()
    return timer, messages


def modal_action(qtbot, call, kind, action):
    errors = []

    def drive():
        dialog = QApplication.activeModalWidget()
        try:
            assert isinstance(dialog, kind), repr(dialog)
            assert dialog.isVisible()
            action(dialog)
        except BaseException as exc:
            errors.append(exc)
        finally:
            if isinstance(dialog, QDialog):
                dialog.accept()

    QTimer.singleShot(80, drive)
    call()
    assert not errors, errors


def test_role_buttons_assign_reopen_and_revoke_on_real_service(qtbot, tmp_path):
    service = _service(tmp_path)
    owner = service.create_initial_account("owner", "Owner", "")
    target = Principal.local(subject="reviewer", display_name="Reviewer")
    service.identities.save(target)
    project = service.create_project(principal=owner.principal, name="Role clicks", width=1, height=1,
                                     orientation=GridOrientation.Y_DOWN, idempotency_key="project")
    controller = object.__new__(DesktopController)
    controller.shell = QDialog()
    qtbot.addWidget(controller.shell)
    controller.session = owner
    controller.service = service
    controller._error = lambda message: (_ for _ in ()).throw(AssertionError(message))

    def select_target(dialog):
        row = next(i for i in range(dialog.people.count())
                   if dialog.people.item(i).data(Qt.ItemDataRole.UserRole).id == target.id)
        rect = dialog.people.visualItemRect(dialog.people.item(row))
        QTest.mouseClick(dialog.people.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
        assert dialog._current_principal().id == target.id

    def assign(dialog):
        select_target(dialog)
        QTest.mouseClick(dialog.assign_button, Qt.MouseButton.LeftButton)
        assert ProjectRole.VIEWER in service.project_roles(project.id, target.id)

    modal_action(qtbot, lambda: controller.manage_project_participants(SimpleNamespace(project_id=project.id)),
                 RoleManagementDialog, assign)
    controller.service = EmbeddedProjectService(service.data_dir)
    assert ProjectRole.VIEWER in controller.service.project_roles(project.id, target.id)

    # A real viewer pressing Assign must see the authorization error, without a write.
    controller.session = SimpleNamespace(principal=target)
    timer, messages = message_pump()

    def denied(dialog):
        row = next(i for i in range(dialog.people.count())
                   if dialog.people.item(i).data(Qt.ItemDataRole.UserRole).id == owner.principal.id)
        rect = dialog.people.visualItemRect(dialog.people.item(row))
        QTest.mouseClick(dialog.people.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
        assert dialog.assign_button.isEnabled()
        QTest.mouseClick(dialog.assign_button, Qt.MouseButton.LeftButton)

    try:
        modal_action(qtbot, lambda: controller.manage_project_participants(SimpleNamespace(project_id=project.id)),
                     RoleManagementDialog, denied)
        assert messages and messages[-1][0] == "Недостаточно прав"
        assert ProjectRole.VIEWER not in controller.service.project_roles(project.id, owner.principal.id)
    finally:
        timer.stop()
        controller.session = owner

    def revoke(dialog):
        select_target(dialog)
        assert dialog.revoke_button.isEnabled()
        QTest.mouseClick(dialog.revoke_button, Qt.MouseButton.LeftButton)
        assert ProjectRole.VIEWER not in controller.service.project_roles(project.id, target.id)

    modal_action(qtbot, lambda: controller.manage_project_participants(SimpleNamespace(project_id=project.id)),
                 RoleManagementDialog, revoke)
    assert ProjectRole.VIEWER not in EmbeddedProjectService(service.data_dir).project_roles(project.id, target.id)


def test_performer_add_button_persists_and_reopens(qtbot, tmp_path):
    service = _service(tmp_path)
    panel = _performer_panel(service)
    qtbot.addWidget(panel)
    panel.show()
    add = next(b for b in panel.findChildren(QPushButton) if b.text() == "Добавить исполнителя")

    timer, messages = message_pump()

    def empty(dialog):
        buttons = dialog.findChild(QDialogButtonBox)
        QTest.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

    try:
        modal_action(qtbot, lambda: QTest.mouseClick(add, Qt.MouseButton.LeftButton), QDialog, empty)
        assert messages and messages[-1][0] == "Не удалось создать исполнителя"
        assert not service.list_performers()
    finally:
        timer.stop()

    def enter(dialog):
        dialog.findChild(QLineEdit).setText("Qt reviewer")
        buttons = dialog.findChild(QDialogButtonBox)
        QTest.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

    modal_action(qtbot, lambda: QTest.mouseClick(add, Qt.MouseButton.LeftButton), QDialog, enter)
    panel.close()
    reopened = EmbeddedProjectService(service.data_dir)
    assert [p.name for p in reopened.list_performers()] == ["Qt reviewer"]
    panel = _performer_panel(reopened)
    qtbot.addWidget(panel)
    panel.show()
    assert panel.findChild(QTableWidget).item(0, 0).text() == "Qt reviewer"
    panel.close()


def test_integrity_button_reports_clean_and_corrupt_blob_in_real_messagebox(qtbot, tmp_path):
    from kraken_hub.presentation.qt.parts.administration_panel import _administration_panel
    from tests.test_work_orders import _project_with_frames

    service = _service(tmp_path)
    owner, _, project, layer, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    panel = _administration_panel(service, owner)
    qtbot.addWidget(panel)
    panel.show()
    timer, messages = message_pump()
    try:
        qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
        button = next(b for b in panel.findChildren(QPushButton) if b.text() == "Проверить целостность")
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: len(messages) == 1)
        qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
        assert messages[0][0] == "Проверка завершена"
        assert "Ошибок нет" in messages[0][1]
        digest = service.frame_cells(project.id, layer.id, images.id)[0].sha256
        paths = list(service.catalog_root.rglob(digest))
        assert len(paths) == 1
        paths[0].write_bytes(b"corrupted own fixture")
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: len(messages) == 2)
        qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
        assert messages[1][0] == "Обнаружены ошибки"
        assert "Ошибок нет" not in messages[1][1]
    finally:
        timer.stop()
        panel.close()


def test_issue_work_dialog_validates_destination_and_persists_package(qtbot, tmp_path, monkeypatch):
    from pathlib import Path
    from kraken_manager.presentation.qt.pages import ProjectWorkspacePage
    from tests.test_work_orders import _project_with_frames

    service = _service(tmp_path)
    owner, _, project, layer, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    controller = object.__new__(DesktopController)
    controller.shell = QDialog()
    qtbot.addWidget(controller.shell)
    controller.service, controller.session, controller._project_id = service, owner, str(project.id)
    workspace = ProjectWorkspacePage()
    qtbot.addWidget(workspace)
    workspace._selected_layer_id = str(layer.id)
    workspace.image_representation_combo.addItem("Images", str(images.id))
    controller._workspace = workspace
    controller.my_work_refresh = None
    timer, messages = message_pump()
    destination = tmp_path / "packages"
    destination.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_: str(destination))

    def submit(dialog, choose):
        if choose:
            button = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Выбрать…")
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        buttons = dialog.findChild(QDialogButtonBox)
        QTest.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

    try:
        modal_action(qtbot, lambda: controller.issue_work_dialog(preset_kind="vectorize"), QDialog,
                     lambda d: submit(d, False))
        assert messages[-1][0] == "Каталог не выбран"
        assert not service.work_orders()
        modal_action(qtbot, lambda: controller.issue_work_dialog(preset_kind="vectorize"), QDialog,
                     lambda d: submit(d, True))
        assert messages[-1][0] == "Работа выдана"
        orders = EmbeddedProjectService(service.data_dir).work_orders()
        assert len(orders) == 1 and orders[0].frame_count == 2
        package = Path(orders[0].destination)
        assert (package / "images/1_1.png").read_bytes() == b"image-a"
        assert (package / "images/2_1.png").read_bytes() == b"image-b"
    finally:
        timer.stop()


def test_review_return_preview_import_accept_and_reopen(qtbot, tmp_path, monkeypatch):
    from tests.test_work_orders import _project_with_frames

    service = _service(tmp_path)
    owner, performer, project, layer, images, vectors = _project_with_frames(service, tmp_path / "source")
    batch = service.create_review_batch(principal=owner.principal, project_id=project.id, layer_id=layer.id,
        image_representation_id=images.id, vector_representation_id=vectors.id, coordinates=((1, 1),),
        assignee_id=performer.id, instructions="Qt return", due_at=None, idempotency_key="review")
    package = tmp_path / "review-package"
    service.export_review_batch(principal=owner.principal, batch=batch, destination=package, idempotency_key="export")
    changed = b"DS 1 1 1;\nP 0 0 1 0 1 1;\nDF;\nE\n"
    next(package.rglob("*.cif")).write_bytes(changed)
    controller = object.__new__(DesktopController)
    controller.shell = QDialog()
    qtbot.addWidget(controller.shell)
    controller.service, controller.session = service, owner
    controller._workspace, controller._project_id = None, str(project.id)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_: str(package))
    notices = []
    timer = QTimer()
    timer.setInterval(10)

    def decisions():
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, QMessageBox):
            return
        notices.append(dialog.windowTitle())
        if dialog.windowTitle() == "Изменённые файлы загружены":
            button = next(b for b in dialog.buttons() if b.text() == "Принять")
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        elif dialog.windowTitle() == "Подтверждение":
            QTest.mouseClick(dialog.button(QMessageBox.StandardButton.Yes), Qt.MouseButton.LeftButton)
        else:
            dialog.accept()

    timer.timeout.connect(decisions)
    timer.start()

    def import_preview(dialog):
        assert dialog.windowTitle() == "Предварительная проверка"
        table = dialog.findChild(QTableWidget)
        assert table.item(0, 1).text() == "Изменён"
        button = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Загрузить изменённые файлы")
        assert button.isEnabled()
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    try:
        modal_action(qtbot, controller.load_review_return, QDialog, import_preview)
        assert notices == ["Изменённые файлы загружены", "Подтверждение"]
        reopened = EmbeddedProjectService(service.data_dir)
        assert next(b for b in reopened.review_batches() if b.id == batch.id).state.value == "completed"
        cell = next(c for c in reopened.frame_cells(project.id, layer.id, vectors.id) if c.x == 1)
        assert reopened.read_project_blob(project.id, cell.sha256) == changed
    finally:
        timer.stop()


def test_statistics_calculate_and_export_buttons_use_real_history(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QComboBox
    from openpyxl import load_workbook
    from kraken_hub.presentation.qt.parts.statistics_panel import _statistics_panel
    from tests.test_work_orders import _project_with_frames

    service = _service(tmp_path)
    _, _, project, _, _, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    panel = _statistics_panel(service, async_reads=True)
    qtbot.addWidget(panel)
    panel.show()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda _p, _t, name, filt: (str(tmp_path / name), filt))
    timer, messages = message_pump()

    def click(text):
        button = next(b for b in panel.findChildren(QPushButton) if b.text() == text)
        assert button.isVisible() and button.isEnabled()
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: panel._report_worker is None, timeout=10000)

    try:
        qtbot.waitUntil(lambda: panel._report_worker is None, timeout=10000)
        combo = panel.findChild(QComboBox, "statisticsProjectFilter")
        combo.setCurrentIndex(combo.findData(str(project.id)))
        click("Рассчитать")
        assert panel.findChild(QTableWidget, "statisticsSummary").item(0, 1).text() == "2"
        click("Экспорт CSV")
        click("Экспорт XLSX")
        assert (tmp_path / "kraken-report.csv").stat().st_size > 100
        workbook = load_workbook(tmp_path / "kraken-report.xlsx", read_only=True)
        assert workbook.sheetnames
        workbook.close()
        assert not messages
    finally:
        qtbot.waitUntil(lambda: panel._report_worker is None, timeout=10000)
        timer.stop()
        panel.close()
