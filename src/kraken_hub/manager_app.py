"""PyQt Desktop composition for projects, matrix and legacy plugin launcher."""

from __future__ import annotations

import os
import sys
from uuid import uuid4

from kraken_core.plugins import PluginInventoryItem
from kraken_core.qt import configure_application_identity
from kraken_core.styles import load_shared_stylesheet
from kraken_manager.domain.project import GridOrientation as DomainOrientation
from kraken_manager.domain.project import LayerType, RepresentationKind
from kraken_manager.application.imports import ImportMappingMode
from kraken_manager.presentation.qt import ProjectManagerShell
from kraken_manager.presentation.qt.models import LayerListItem, ProjectListItem
from kraken_manager.presentation.qt.widgets import ClickableLabel, FrameCellData, GridDimensionsWidget

from . import windows_credentials
from .composition import DesktopSession, EmbeddedProjectService


def _remember_credentials(username: str, password: str) -> None:
    try:
        windows_credentials.save_credentials(username, password)
    except Exception:
        pass


def _autofill_credentials(dialog, username, password) -> None:
    try:
        if not windows_credentials.credentials_available():
            return
        if not windows_credentials.verify_windows_identity(int(dialog.winId())):
            return
        remembered = windows_credentials.load_credentials()
        if remembered is None:
            return
    except Exception:
        return
    username.setText(remembered[0])
    password.setText(remembered[1])


def _development_session(service: EmbeddedProjectService) -> DesktopSession | None:
    """Return the explicit VS Code development session without showing login UI."""
    if getattr(sys, "frozen", False) or os.environ.get("KRAKEN_DEV_AUTO_LOGIN") != "1":
        return None
    username = os.environ.get("KRAKEN_DEV_USERNAME", "vscode")
    password = os.environ.get("KRAKEN_DEV_PASSWORD", "")
    if not service.has_accounts:
        return service.create_initial_account(
            username=username,
            display_name=os.environ.get("KRAKEN_DEV_DISPLAY_NAME", "VS Code Developer"),
            password=password,
        )
    session = service.login(username, password)
    if session is None:
        raise RuntimeError(
            "Dev auto-login account does not match the configured data directory; "
            "use a clean KRAKEN_DATA_DIR or update KRAKEN_DEV credentials"
        )
    return session


def _login(parent, service: EmbeddedProjectService) -> DesktopSession | None:
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QMessageBox, QVBoxLayout

    if not service.has_accounts:
        dialog = QDialog(parent)
        dialog.setObjectName("initialAccountDialog")
        dialog.setWindowTitle("Создание аккаунта Kraken")
        layout = QVBoxLayout(dialog)
        intro = QLabel("Создайте первый локальный аккаунт для этой рабочей станции.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        username = QLineEdit()
        username.setObjectName("initialAccountUsername")
        username.setPlaceholderText("admin")
        display_name = QLineEdit()
        display_name.setObjectName("initialAccountDisplayName")
        password = QLineEdit()
        password.setObjectName("initialAccountPassword")
        password.setEchoMode(QLineEdit.EchoMode.Password)
        confirmation = QLineEdit()
        confirmation.setObjectName("initialAccountPasswordConfirmation")
        confirmation.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Имя пользователя", username)
        form.addRow("Отображаемое имя", display_name)
        form.addRow("Пароль", password)
        form.addRow("Повторите пароль", confirmation)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Создать аккаунт")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        username.setFocus()
        while dialog.exec() == QDialog.DialogCode.Accepted:
            if not username.text().strip() or any(char.isspace() for char in username.text().strip()):
                QMessageBox.warning(
                    dialog,
                    "Не удалось создать аккаунт",
                    "Введите имя пользователя без пробелов.",
                )
                username.setFocus()
                continue
            if not display_name.text().strip():
                QMessageBox.warning(dialog, "Не удалось создать аккаунт", "Введите отображаемое имя.")
                display_name.setFocus()
                continue
            if password.text() != confirmation.text():
                QMessageBox.warning(dialog, "Не удалось создать аккаунт", "Пароли не совпадают.")
                password.clear()
                confirmation.clear()
                password.setFocus()
                continue
            try:
                session = service.create_initial_account(
                    username=username.text(),
                    display_name=display_name.text(),
                    password=password.text(),
                )
            except Exception as exc:
                QMessageBox.warning(dialog, "Не удалось создать аккаунт", str(exc))
                continue
            _remember_credentials(username.text().strip(), password.text())
            return session
        return None
    dialog = QDialog(parent)
    dialog.setObjectName("loginDialog")
    dialog.setWindowTitle("Вход в Kraken")
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("Локальная сессия рабочей станции"))
    form = QFormLayout()
    username = QLineEdit()
    username.setObjectName("loginUsername")
    password = QLineEdit()
    password.setObjectName("loginPassword")
    password.setEchoMode(QLineEdit.EchoMode.Password)
    form.addRow("Имя пользователя", username)
    form.addRow("Пароль", password)
    layout.addLayout(form)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    _autofill_credentials(dialog, username, password)
    while dialog.exec() == QDialog.DialogCode.Accepted:
        entered_username = username.text()
        entered_password = password.text()
        session = service.login(entered_username, entered_password)
        password.clear()
        if session is not None:
            _remember_credentials(entered_username.strip(), entered_password)
            return session
        QMessageBox.warning(dialog, "Kraken", "Неверные учётные данные или аккаунт временно заблокирован.")
    return None


class DesktopController:
    def __init__(self, shell: ProjectManagerShell, service: EmbeddedProjectService, session: DesktopSession) -> None:
        self.shell = shell
        self.service = service
        self.session = session
        self.catalog_page = shell.page("projects")
        assert self.catalog_page is not None
        self.catalog_page.createRequested.connect(self.create_project)
        self.catalog_page.refreshRequested.connect(self.refresh_projects)
        self.catalog_page.projectActivated.connect(self.open_project)
        self.catalog_page.renameRequested.connect(self.rename_project)
        self.catalog_page.archiveRequested.connect(self.archive_project)
        self.catalog_page.restoreRequested.connect(self.restore_project)
        self.refresh_projects()

    def refresh_projects(self) -> None:
        items = [
            ProjectListItem(
                project_id=str(project.id),
                name=project.name,
                width=project.width,
                height=project.height,
                storage_label="Локальный файл",
                status=project.state.value,
                archived=project.state.value == "archived",
                metadata={"orientation": project.orientation.value, "revision": project.revision},
            )
            for project in self.service.list_projects(
                include_archived=self.catalog_page.show_archived_check.isChecked()
            )
        ]
        self.catalog_page.project_model.replace_items(items)

    def rename_project(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        from PyQt6.QtWidgets import QInputDialog

        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return
        name, accepted = QInputDialog.getText(
            self.shell, "Переименовать проект", "Новое имя", text=project.name
        )
        if not accepted:
            return
        try:
            self.service.rename_project(
                principal=self.session.principal,
                project=project,
                name=name,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self.refresh_projects()

    def archive_project(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        from PyQt6.QtWidgets import QMessageBox

        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return
        answer = QMessageBox.question(
            self.shell,
            "Архивировать проект",
            f"Перевести «{project.name}» в read-only архив?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.archive_project(
                principal=self.session.principal,
                project=project,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self.refresh_projects()

    def restore_project(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return
        try:
            self.service.restore_project(
                principal=self.session.principal,
                project=project,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self.refresh_projects()

    def create_project(self) -> None:
        from PyQt6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QVBoxLayout,
        )

        dialog = QDialog(self.shell)
        dialog.setWindowTitle("Новый локальный проект")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        name = QLineEdit()
        name.setObjectName("projectName")
        form.addRow("Имя проекта", name)
        layout.addLayout(form)
        dimensions = GridDimensionsWidget(maximum_frames=self.service.profile.capabilities.max_frames or 100_000)
        layout.addWidget(dimensions)
        layout.addWidget(QLabel("Хранилище: локальная файловая система · single-writer"))
        template = QCheckBox("Создать шаблон из четырёх типов слоёв")
        layout.addWidget(template)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        while dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                size = dimensions.validated_dimensions()
                if not name.text().strip():
                    raise ValueError("Введите имя проекта")
                project = self.service.create_project(
                    principal=self.session.principal,
                    name=name.text(),
                    width=size.width,
                    height=size.height,
                    orientation=DomainOrientation(size.orientation.value),
                    idempotency_key=str(uuid4()),
                    layer_template=template.isChecked(),
                )
            except Exception as exc:
                QMessageBox.warning(dialog, "Не удалось создать проект", str(exc))
                continue
            self.refresh_projects()
            self.open_project(
                ProjectListItem(
                    str(project.id),
                    project.name,
                    project.width,
                    project.height,
                    "Локальный файл",
                    metadata={"orientation": project.orientation.value, "revision": project.revision},
                )
            )
            break

    def open_project(self, item: ProjectListItem) -> None:
        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            self.refresh_projects()
            return
        workspace = self.shell.open_project_workspace()
        workspace.set_project_title(project.name)
        workspace.matrix_view.set_matrix_size(project.width, project.height, project.orientation.value)
        self._load_layers(workspace, project)

        def add_layer() -> None:
            self._add_layer(workspace, project.id)

        workspace.addLayerRequested.connect(add_layer)
        workspace.layerActivated.connect(
            lambda layer_item: self._select_layer(workspace, project.id, layer_item)
        )
        workspace.addImageRepresentationRequested.connect(
            lambda: self._add_representation(workspace, project.id, RepresentationKind.IMAGE)
        )
        workspace.addVectorRepresentationRequested.connect(
            lambda: self._add_representation(workspace, project.id, RepresentationKind.VECTOR)
        )
        workspace.imageRepresentationChanged.connect(
            lambda identifier: self._activate_representation(workspace, project.id, identifier)
        )
        workspace.vectorRepresentationChanged.connect(
            lambda identifier: self._activate_representation(workspace, project.id, identifier)
        )
        layers = self.service.list_layers(project.id)
        if layers:
            first = workspace.layer_model.layer_by_id(str(layers[0].id))
            if first is not None:
                workspace.layer_list.setCurrentIndex(workspace.layer_model.index(0, 0))
                self._select_layer(workspace, project.id, first)

    def _load_layers(self, workspace, project) -> None:
        colors = {
            "metal": "#60A5FA",
            "contact": "#F59E0B",
            "gate": "#A78BFA",
            "diffusion": "#34D399",
        }
        workspace.layer_model.replace_items(
            LayerListItem(str(layer.id), layer.name, layer.type.value, colors[layer.type.value])
            for layer in self.service.list_layers(project.id)
        )

    def _select_layer(self, workspace, project_id, item: LayerListItem) -> None:
        workspace._selected_layer_id = item.layer_id
        self._load_representations(workspace, project_id, item.layer_id)

    def _load_representations(self, workspace, project_id, layer_id) -> None:
        representations = self.service.list_representations(project_id, layer_id)
        workspace.set_representations(
            images=[
                (str(item.id), item.name)
                for item in sorted(
                    (value for value in representations if value.kind is RepresentationKind.IMAGE),
                    key=lambda value: (not value.active, value.name.casefold()),
                )
            ],
            vectors=[
                (str(item.id), item.name)
                for item in sorted(
                    (value for value in representations if value.kind is RepresentationKind.VECTOR),
                    key=lambda value: (not value.active, value.name.casefold()),
                )
            ],
        )
        self._refresh_matrix(workspace, project_id)

    def _activate_representation(self, workspace, project_id, representation_id: str) -> None:
        if not representation_id:
            self._refresh_matrix(workspace, project_id)
            return
        layer_id = getattr(workspace, "_selected_layer_id", None)
        project = self.service.get_project(project_id)
        if project is None or layer_id is None:
            return
        layer = next(
            (item for item in self.service.list_layers(project.id) if str(item.id) == str(layer_id)),
            None,
        )
        representation = next(
            (
                item
                for item in self.service.list_representations(project.id, layer_id)
                if str(item.id) == str(representation_id)
            ),
            None,
        )
        if layer is None or representation is None:
            self._refresh_matrix(workspace, project_id)
            return
        if representation.active:
            self._refresh_matrix(workspace, project_id)
            return
        try:
            self.service.activate_representation(
                principal=self.session.principal,
                project=project,
                layer=layer,
                representation=representation,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self._load_representations(workspace, project_id, layer_id)

    def _refresh_matrix(self, workspace, project_id) -> None:
        layer_id = getattr(workspace, "_selected_layer_id", None)
        if layer_id is None:
            workspace.matrix_view.clear_cells()
            return
        representation_ids = (
            workspace.image_representation_combo.currentData(),
            workspace.vector_representation_combo.currentData(),
        )
        priority = {
            "empty": 0,
            "image_ready": 1,
            "processing": 2,
            "vectorized": 3,
            "in_review": 4,
            "returned_unchanged": 5,
            "returned_changed": 6,
            "approved": 7,
            "changes_requested": 8,
            "conflict": 9,
            "error": 10,
        }
        cells: dict[tuple[int, int], FrameCellData] = {}
        for representation_id in representation_ids:
            if not representation_id:
                continue
            for item in self.service.frame_cells(
                project_id, layer_id, str(representation_id)
            ):
                key = (item.x, item.y)
                current = cells.get(key)
                if current is not None and priority.get(current.status, 0) > priority.get(item.status, 0):
                    continue
                cells[key] = FrameCellData(
                    item.x,
                    item.y,
                    status=item.status,
                    label=f"{item.x},{item.y}",
                    tooltip=(
                        f"Кадр ({item.x}, {item.y})\nСтатус: {item.status}\n"
                        f"SHA-256: {item.sha256}\nВерсия: {item.artifact_version_id}"
                    ),
                    payload={
                        "frame_id": item.frame_id,
                        "artifact_version_id": item.artifact_version_id,
                    },
                )
        workspace.matrix_view.set_cells(cells.values())

    def _add_representation(self, workspace, project_id, kind: RepresentationKind) -> None:
        from PyQt6.QtWidgets import (
            QCheckBox,
            QDialog,
            QDialogButtonBox,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLineEdit,
            QMessageBox,
            QSizePolicy,
            QWidget,
        )

        layer_id = getattr(workspace, "_selected_layer_id", None)
        if layer_id is None:
            QMessageBox.information(workspace, "Kraken", "Сначала выберите слой.")
            return
        project = self.service.get_project(project_id)
        layer = next((item for item in self.service.list_layers(project_id) if str(item.id) == layer_id), None)
        if project is None or layer is None:
            self._error("Слой или проект больше не доступен")
            return
        label = "изображение" if kind is RepresentationKind.IMAGE else "вектор"
        dialog = QDialog(workspace)
        dialog.setWindowTitle(f"Добавить {label}")
        form = QFormLayout(dialog)
        name = QLineEdit()
        name.setObjectName("representationName")
        note = QLineEdit()
        note.setObjectName("representationNote")
        source = QLineEdit()
        source.setObjectName("representationSource")
        mapping_mode = QComboBox()
        mapping_mode.addItem("Координаты <x>_<y>", ImportMappingMode.XY_FILENAME.value)
        mapping_mode.addItem("Числовой суффикс (row-major)", ImportMappingMode.ROW_MAJOR_SUFFIX.value)
        mapping_mode.addItem("Регулярное выражение (?P<x>…)(?P<y>…)", ImportMappingMode.REGEX.value)
        mapping_regex = QLineEdit()
        mapping_regex.setPlaceholderText(r"frame_(?P<x>\d+)_(?P<y>\d+)")
        active = QCheckBox("Сделать активным")
        active.setChecked(True)
        form.addRow("Название", name)
        form.addRow("Примечание", note)
        if kind is RepresentationKind.IMAGE:
            source_row = QWidget()
            source_layout = QHBoxLayout(source_row)
            source_layout.setContentsMargins(0, 0, 0, 0)
            source_layout.addWidget(source, 1)
            source_picker = ClickableLabel("Выбрать папку…")
            source_picker.setObjectName("representationSourceFolderPicker")
            source_picker.setToolTip("Открыть окно выбора папки с изображениями")
            source_picker.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            source_picker.setMinimumWidth(
                source_picker.fontMetrics().horizontalAdvance(source_picker.text()) + 8
            )
            source_picker.setStyleSheet("color: #60A5FA; text-decoration: underline;")
            source_layout.addWidget(source_picker)

            def choose_source_folder() -> None:
                directory = QFileDialog.getExistingDirectory(
                    dialog,
                    "Выберите папку с изображениями",
                    source.text().strip(),
                )
                if directory:
                    source.setText(directory)

            source_picker.clicked.connect(choose_source_folder)
            form.addRow("Источник", source_row)
        else:
            form.addRow("Источник", source)
        form.addRow("Сопоставление кадров", mapping_mode)
        form.addRow("Regex", mapping_regex)
        form.addRow(active)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        while dialog.exec() == QDialog.DialogCode.Accepted:
            if not name.text().strip():
                QMessageBox.warning(dialog, f"Не удалось добавить {label}", "Введите название.")
                name.setFocus()
                continue
            import_plan = None
            source_directory = source.text().strip()
            if source_directory:
                try:
                    import_plan = self.service.plan_import_directory(
                        project=project,
                        directory=source_directory,
                        mode=ImportMappingMode(str(mapping_mode.currentData())),
                        regex=mapping_regex.text().strip() or None,
                    )
                except Exception as exc:
                    QMessageBox.warning(dialog, "Ошибка preflight", str(exc))
                    continue
                issue_text = "\n".join(
                    f"• {issue.message}" for issue in import_plan.issues
                ) or "Ошибок не обнаружено"
                if not import_plan.ready:
                    QMessageBox.warning(
                        dialog,
                        "Импорт заблокирован",
                        f"Preflight не пройден:\n{issue_text}",
                    )
                    continue
                answer = QMessageBox.question(
                    dialog,
                    "Подтверждение managed import",
                    (
                        f"Файлов: {len(import_plan.items)}\n"
                        f"Объём: {import_plan.total_bytes:n} байт\n"
                        f"Пустых кадров: {import_plan.missing_coordinates:n}\n\n"
                        f"{issue_text}\n\nСкопировать файлы в immutable BlobStore?"
                    ),
                )
                if answer != QMessageBox.StandardButton.Yes:
                    continue
            try:
                representation = self.service.create_representation(
                    principal=self.session.principal,
                    project=project,
                    layer=layer,
                    name=name.text(),
                    kind=kind,
                    idempotency_key=str(uuid4()),
                    note=note.text(),
                    source="managed-import" if import_plan is not None else None,
                    active=active.isChecked(),
                )
                if import_plan is not None:
                    imported = self.service.commit_managed_import(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        representation=representation,
                        plan=import_plan,
                        idempotency_key=str(uuid4()),
                    )
                    QMessageBox.information(
                        dialog,
                        "Импорт завершён",
                        f"Создано immutable версий: {len(imported.versions)}",
                    )
            except Exception as exc:
                QMessageBox.warning(dialog, f"Не удалось добавить {label}", str(exc))
                continue
            self._load_representations(workspace, project_id, layer_id)
            combo = (
                workspace.image_representation_combo
                if kind is RepresentationKind.IMAGE
                else workspace.vector_representation_combo
            )
            combo.setCurrentIndex(combo.findData(str(representation.id)))
            return

    def _add_layer(self, workspace, project_id) -> None:
        from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QMessageBox

        project = self.service.get_project(project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return
        dialog = QDialog(self.shell)
        dialog.setWindowTitle("Добавить слой")
        form = QFormLayout(dialog)
        name = QLineEdit()
        layer_type = QComboBox()
        for value in LayerType:
            layer_type.addItem(value.value, value.value)
        form.addRow("Имя", name)
        form.addRow("Тип", layer_type)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            layers = self.service.list_layers(project.id)
            self.service.create_layer(
                principal=self.session.principal,
                project=project,
                name=name.text(),
                layer_type=LayerType(str(layer_type.currentData())),
                order=len(layers) + 1,
                idempotency_key=str(uuid4()),
            )
            latest = self.service.get_project(project.id)
            if latest is not None:
                self._load_layers(workspace, latest)
        except Exception as exc:
            QMessageBox.warning(dialog, "Не удалось добавить слой", str(exc))

    def _error(self, text: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.warning(self.shell, "Kraken", text)


def _plugin_panel(items: list[PluginInventoryItem]):
    from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

    from .app import launch_plugin

    host = QWidget()
    layout = QVBoxLayout(host)
    for inventory in items:
        plugin = inventory.metadata
        row = QFrame()
        row.setObjectName("pluginInventoryRow")
        row_layout = QHBoxLayout(row)
        capabilities = ", ".join(capability.operation for capability in plugin.capabilities) or "standalone only"
        label = QLabel(f"{plugin.display_name} {plugin.version}\n{plugin.description}\n{capabilities}")
        label.setWordWrap(True)
        row_layout.addWidget(label, 1)
        button = QPushButton("Открыть")
        button.setEnabled(plugin.enabled and (inventory.installed or bool(plugin.source_dir)))
        button.clicked.connect(lambda _checked=False, selected=plugin: launch_plugin(selected))
        row_layout.addWidget(button)
        layout.addWidget(row)
    layout.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(host)
    return scroll


def _performer_panel(service: EmbeddedProjectService):
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QHBoxLayout,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    host = QWidget()
    layout = QVBoxLayout(host)
    actions = QHBoxLayout()
    add = QPushButton("Добавить исполнителя")
    refresh_button = QPushButton("Обновить")
    actions.addWidget(add)
    actions.addWidget(refresh_button)
    actions.addStretch(1)
    layout.addLayout(actions)
    table = QTableWidget(0, 3)
    table.setHorizontalHeaderLabels(("Имя", "Цвет", "Учётная запись"))
    table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(table)

    def refresh() -> None:
        values = service.list_performers()
        table.setRowCount(len(values))
        for row, performer in enumerate(values):
            name_item = QTableWidgetItem(performer.name)
            color_item = QTableWidgetItem(performer.color)
            color_item.setBackground(QColor(performer.color))
            account_item = QTableWidgetItem("GitLab" if performer.principal_id is not None else "Ручной")
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, color_item)
            table.setItem(row, 2, account_item)

    def create() -> None:
        dialog = QDialog(host)
        dialog.setWindowTitle("Новый исполнитель")
        form = QFormLayout(dialog)
        name = QLineEdit()
        color = QLineEdit("#60A5FA")
        form.addRow("Имя", name)
        form.addRow("Цвет #RRGGBB", color)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            service.create_manual_performer(name=name.text(), color=color.text())
        except Exception as exc:
            QMessageBox.warning(dialog, "Не удалось создать исполнителя", str(exc))
            return
        refresh()

    add.clicked.connect(create)
    refresh_button.clicked.connect(refresh)
    refresh()
    return host


def _my_work_panel(service: EmbeddedProjectService):
    from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

    host = QWidget()
    layout = QVBoxLayout(host)
    actions = QHBoxLayout()
    refresh_button = QPushButton("Обновить")
    actions.addWidget(refresh_button)
    actions.addStretch(1)
    layout.addLayout(actions)
    table = QTableWidget(0, 6)
    table.setHorizontalHeaderLabels(("Проект", "Слой", "Исполнитель", "Статус", "Срок", "Кадров"))
    table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(table)

    def refresh() -> None:
        batches = service.active_review_batches()
        table.setRowCount(len(batches))
        for row, batch in enumerate(batches):
            values = (
                str(batch.project_id),
                str(batch.layer_id),
                str(batch.assignee_id),
                batch.state.value,
                "" if batch.due_at is None else batch.due_at.astimezone().strftime("%Y-%m-%d %H:%M"),
                str(len(batch.items)),
            )
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))

    refresh_button.clicked.connect(refresh)
    refresh()
    return host


def _statistics_panel(service: EmbeddedProjectService):
    from datetime import UTC, datetime

    from PyQt6.QtCore import QDateTime
    from PyQt6.QtWidgets import (
        QDateTimeEdit,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from kraken_manager.infrastructure.reports import ReportFilters, ReportService

    host = QWidget()
    layout = QVBoxLayout(host)
    controls = QHBoxLayout()
    start = QDateTimeEdit(QDateTime.currentDateTime().addDays(-30))
    end = QDateTimeEdit(QDateTime.currentDateTime())
    start.setCalendarPopup(True)
    end.setCalendarPopup(True)
    refresh_button = QPushButton("Рассчитать")
    csv_button = QPushButton("Экспорт CSV")
    xlsx_button = QPushButton("Экспорт XLSX")
    controls.addWidget(QLabel("С"))
    controls.addWidget(start)
    controls.addWidget(QLabel("По"))
    controls.addWidget(end)
    controls.addWidget(refresh_button)
    controls.addWidget(csv_button)
    controls.addWidget(xlsx_button)
    controls.addStretch(1)
    layout.addLayout(controls)
    table = QTableWidget(0, 2)
    table.setHorizontalHeaderLabels(("Метрика", "Значение"))
    table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(table)
    reports = ReportService()

    def filters() -> ReportFilters:
        start_at = datetime.fromtimestamp(start.dateTime().toSecsSinceEpoch(), UTC)
        end_at = datetime.fromtimestamp(end.dateTime().toSecsSinceEpoch(), UTC)
        return ReportFilters(start_at, end_at)

    def refresh() -> None:
        try:
            metrics = reports.aggregate(service.activity_records(), filters())
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось рассчитать статистику", str(exc))
            return
        values = sorted(metrics.values.items())
        table.setRowCount(len(values))
        for row, (name, value) in enumerate(values):
            table.setItem(row, 0, QTableWidgetItem(name))
            table.setItem(row, 1, QTableWidgetItem(str(value)))

    def export(extension: str) -> None:
        destination, _ = QFileDialog.getSaveFileName(
            host,
            "Экспорт статистики",
            f"kraken-report.{extension}",
            f"{extension.upper()} (*.{extension})",
        )
        if not destination:
            return
        try:
            records = service.activity_records()
            if extension == "csv":
                reports.write_csv(destination, records, filters(), assume_sorted=True)
            else:
                reports.write_xlsx(destination, records, filters(), assume_sorted=True)
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось экспортировать отчёт", str(exc))

    refresh_button.clicked.connect(refresh)
    csv_button.clicked.connect(lambda: export("csv"))
    xlsx_button.clicked.connect(lambda: export("xlsx"))
    refresh()
    return host


def _administration_panel(service: EmbeddedProjectService):
    from PyQt6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QWidget,
    )

    host = QWidget()
    layout = QFormLayout(host)
    capabilities = service.profile.capabilities
    layout.addRow("Каталог", QLabel(str(service.catalog_root)))
    layout.addRow("Metadata backend", QLabel(service.profile.metadata_backend.value))
    layout.addRow("Blob backend", QLabel(service.profile.blob_backend))
    layout.addRow("Режим записи", QLabel("multi-writer" if capabilities.multi_writer else "single-writer"))
    layout.addRow("Максимум кадров", QLabel(str(capabilities.max_frames or "без лимита")))
    project_count = QLabel()
    layout.addRow("Проектов", project_count)
    layout.addRow("Источник истины", QLabel("append-only event log; SQLite — перестраиваемый индекс"))
    projects = QComboBox()
    layout.addRow("Проект для backup", projects)
    actions = QWidget()
    action_layout = QHBoxLayout(actions)
    action_layout.setContentsMargins(0, 0, 0, 0)
    scan_button = QPushButton("Проверить целостность")
    export_button = QPushButton("Создать backup")
    import_button = QPushButton("Восстановить backup")
    action_layout.addWidget(scan_button)
    action_layout.addWidget(export_button)
    action_layout.addWidget(import_button)
    layout.addRow("Операции", actions)

    def refresh_projects() -> None:
        values = service.list_projects(include_archived=True)
        projects.clear()
        for project in values:
            projects.addItem(project.name, str(project.id))
        project_count.setText(str(len(values)))
        export_button.setEnabled(bool(values))

    def scan() -> None:
        result = service.scan_integrity()
        detail = (
            f"Проектов: {result.projects}\nСобытий: {result.events}\nBlob-объектов: {result.blobs}"
        )
        if result.valid:
            QMessageBox.information(host, "Проверка завершена", detail + "\n\nОшибок нет.")
        else:
            QMessageBox.warning(host, "Обнаружены ошибки", detail + "\n\n" + "\n".join(result.errors))

    def export_backup() -> None:
        project_id = str(projects.currentData() or "")
        destination = QFileDialog.getExistingDirectory(host, "Каталог для backup")
        if not project_id or not destination:
            return
        target = os.path.join(destination, f"kraken-backup-{project_id}")
        try:
            manifest = service.export_backup(project_id, target)
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось создать backup", str(exc))
            return
        QMessageBox.information(
            host,
            "Backup создан",
            f"Bundle {manifest.bundle_id}\nСобытий: {manifest.event_count}\n{target}",
        )

    def import_backup() -> None:
        source = QFileDialog.getExistingDirectory(host, "Выберите Kraken migration bundle")
        if not source:
            return
        try:
            project = service.import_backup(source)
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось восстановить backup", str(exc))
            return
        refresh_projects()
        QMessageBox.information(host, "Backup восстановлен", f"Проект: {project.name}")

    scan_button.clicked.connect(scan)
    export_button.clicked.connect(export_backup)
    import_button.clicked.connect(import_backup)
    refresh_projects()
    return host


def run_manager_gui(items: list[PluginInventoryItem], *, update_url: str = "") -> int:
    del update_url  # Update wiring remains available in the legacy feature flag.
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    configure_application_identity(app, app_id="Kraken.ProjectManager", icon_name="kraken")
    app.setStyleSheet(load_shared_stylesheet("dark_modern.qss"))
    service = EmbeddedProjectService()
    session = _development_session(service) or _login(None, service)
    if session is None:
        return 1
    shell = ProjectManagerShell()
    shell.set_session_summary(f"{session.principal.display_name}\nЛокальная сессия")
    plugins_page = shell.page("plugins")
    if plugins_page is not None:
        plugins_page.set_content(_plugin_panel(items))
    performers_page = shell.page("performers")
    if performers_page is not None:
        performers_page.set_content(_performer_panel(service))
    my_work_page = shell.page("my_work")
    if my_work_page is not None:
        my_work_page.set_content(_my_work_panel(service))
    statistics_page = shell.page("statistics")
    if statistics_page is not None:
        statistics_page.set_content(_statistics_panel(service))
    administration_page = shell.page("administration")
    if administration_page is not None:
        administration_page.set_content(_administration_panel(service))
    shell._desktop_controller = DesktopController(shell, service, session)  # keep Qt slots alive
    shell.show()
    return app.exec()


__all__ = ["DesktopController", "run_manager_gui"]
