"""PyQt Desktop composition for projects, matrix and legacy plugin launcher."""

from __future__ import annotations

import os
import sys
import logging
import shutil
from pathlib import Path
from uuid import uuid4

from kraken_core.frame_matrix import MatrixSession, ThumbnailStoreFactory
from kraken_core.frame_matrix.qt import FrameMatrixWidget
from kraken_core.external_model import ExternalModelLink
from kraken_core.plugins import PluginInventoryItem
from kraken_core.qt import configure_application_identity
from kraken_core.styles import load_shared_stylesheet
from kraken_manager.domain.project import GridOrientation as DomainOrientation
from kraken_manager.domain.project import LayerType, RepresentationKind, RepresentationPurpose
from kraken_manager.application.imports import ImportMappingMode
from kraken_manager.presentation.qt import (
    LayerManagerDialog,
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
    ProjectManagerShell,
    ProjectWorkspacePage,
)
from kraken_manager.presentation.qt.models import LayerListItem, ProjectListItem
from kraken_manager.presentation.qt.widgets import ClickableLabel, GridDimensionsWidget

from . import windows_credentials
from .composition import DesktopSession, EmbeddedProjectService
from .matrix_source import KrakenMatrixAssetSource, KrakenMatrixDataSource


class RepresentationDialog:
    """Shared image/vector representation form with a folder-only source picker."""

    def __init__(self, parent, kind: RepresentationKind) -> None:
        from PyQt6.QtWidgets import (
            QCheckBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLineEdit,
            QTextEdit,
        )

        self.kind = kind
        self.dialog = QDialog(parent)
        self.dialog.setObjectName("representationDialog")
        noun = "изображение" if kind is RepresentationKind.IMAGE else "вектор"
        self.dialog.setWindowTitle(f"Добавить {noun}")
        self.name = QLineEdit()
        self.name.setObjectName("representationName")
        self.source_picker = ClickableLabel("Выбрать папку…")
        self.source_picker.setObjectName("representationSourceFolderPicker")
        self.source_picker.setToolTip("Выбрать папку с файлами")
        self.source_picker.setStyleSheet("color: #60A5FA; text-decoration: underline;")
        self.source_picker.setMinimumWidth(
            self.source_picker.fontMetrics().horizontalAdvance(self.source_picker.text()) + 8
        )
        self.source_picker.clicked.connect(self._choose_source_folder)
        # Kept hidden for automation/API compatibility; the visible source
        # control is deliberately only the ClickableLabel.
        self._source_compatibility = QLineEdit(self.dialog)
        self._source_compatibility.setObjectName("representationSource")
        self._source_compatibility.hide()
        self.note = QTextEdit()
        self.note.setObjectName("representationNote")
        self.note.setAcceptRichText(False)
        self.note.setFixedHeight(80)
        self.active = QCheckBox("Сделать активным")
        self.active.setChecked(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.dialog.accept)
        self.buttons.rejected.connect(self.dialog.reject)

        form = QFormLayout(self.dialog)
        form.addRow("Название", self.name)
        form.addRow("Источник", self.source_picker)
        form.addRow(self.active)
        form.addRow("Примечание", self.note)
        form.addRow(self.buttons)

    @property
    def source_directory(self) -> str:
        return str(self.source_picker.property("sourceDirectory") or "")

    def _choose_source_folder(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        directory = QFileDialog.getExistingDirectory(
            self.dialog,
            "Выберите папку с изображениями"
            if self.kind is RepresentationKind.IMAGE
            else "Выберите папку с векторами",
            self.source_directory,
        )
        if directory:
            self.source_picker.setProperty("sourceDirectory", directory)
            self.source_picker.setText(directory)
            self._source_compatibility.setText(directory)

    def exec(self):
        return self.dialog.exec()


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
    def __init__(
        self,
        shell: ProjectManagerShell,
        service: EmbeddedProjectService,
        session: DesktopSession,
        *,
        thumbnail_store_uri: str = "",
        plugin_items: list[PluginInventoryItem] | None = None,
    ) -> None:
        self.shell = shell
        self.service = service
        self.session = session
        self.thumbnail_store_uri = str(thumbnail_store_uri)
        self.plugin_items = {item.metadata.id: item for item in (plugin_items or [])}
        self._workspace = None
        self._project_id = None
        self._layer_dialog: LayerManagerDialog | None = None
        self.catalog_page = shell.page("projects")
        assert self.catalog_page is not None
        self.catalog_page.createRequested.connect(self.create_project)
        self.catalog_page.refreshRequested.connect(self.refresh_projects)
        self.catalog_page.projectActivated.connect(self.open_project)
        self.catalog_page.renameRequested.connect(self.rename_project)
        self.catalog_page.archiveRequested.connect(self.archive_project)
        self.catalog_page.restoreRequested.connect(self.restore_project)
        self.shell.layersRequested.connect(self.open_layer_manager)
        self.shell.cellVisualModeChanged.connect(self._set_matrix_visual_mode)
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
        cache_root = self.service.data_dir / "cache" / "frame-thumbnails"
        store_uri = (
            self.thumbnail_store_uri
            or os.environ.get("KRAKEN_THUMBNAIL_STORE_URI")
            or f"sqlite:///{cache_root.as_posix()}"
        )
        try:
            thumbnail_store = ThumbnailStoreFactory().create(store_uri)
        except Exception:
            thumbnail_store = ThumbnailStoreFactory().create("memory://")
        data_source = KrakenMatrixDataSource(
            self.service,
            project_id=str(project.id),
            matrix_width=project.width,
        )
        asset_source = KrakenMatrixAssetSource(self.service, project_id=str(project.id))
        matrix_view = FrameMatrixWidget(
            project.width,
            project.height,
            project.orientation.value,
            data_source=data_source,
            asset_source=asset_source,
            thumbnail_store=thumbnail_store,
        )
        workspace = self.shell.open_project_workspace(ProjectWorkspacePage(matrix_view=matrix_view))
        self._workspace = workspace
        self._project_id = str(project.id)
        if self._layer_dialog is not None:
            self._layer_dialog.hide()
            self._layer_dialog.deleteLater()
            self._layer_dialog = None
        workspace._matrix_data_source = data_source
        workspace._matrix_asset_source = asset_source
        workspace.set_project_title(project.name)
        workspace.matrix_view.set_session(
            MatrixSession(
                namespace=str(project.id),
                width=project.width,
                height=project.height,
                source_revision=str(project.revision),
                orientation=project.orientation.value,
                generation=0,
            ),
            data_source=data_source,
            asset_source=asset_source,
        )
        workspace.matrix_view.set_visual_modes(*self.shell.visual_modes())
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
                workspace.layer_tabs.setCurrentIndex(0)
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
        workspace.sync_layer_tabs()
        if self._layer_dialog is not None:
            self._layer_dialog.set_layers(
                list(workspace.layer_model.items()),
                str(getattr(workspace, "_selected_layer_id", "")),
            )

    def _select_layer(self, workspace, project_id, item: LayerListItem) -> None:
        workspace._selected_layer_id = item.layer_id
        self._load_representations(workspace, project_id, item.layer_id)
        if self._layer_dialog is not None:
            self._layer_dialog.select_layer(item.layer_id)
            self._layer_dialog.set_pipeline(self._pipeline_snapshot(project_id, item.layer_id))

    def open_layer_manager(self) -> None:
        workspace = self._workspace
        project_id = self._project_id
        if workspace is None or not project_id or self.shell.current_page_key() != "workspace":
            return
        if self._layer_dialog is None:
            dialog = LayerManagerDialog(
                project_id,
                self.shell,
                action_availability=self._action_availability,
            )
            dialog.layerSelected.connect(self._layer_manager_select)
            dialog.orderChanged.connect(self._layer_manager_reorder)
            dialog.representationActivated.connect(self._layer_manager_activate)
            dialog.nodeActionRequested.connect(self._layer_manager_action)
            dialog.layerActionRequested.connect(self._layer_manager_layer_action)
            self._layer_dialog = dialog
        self._layer_dialog.set_layers(
            list(workspace.layer_model.items()),
            str(getattr(workspace, "_selected_layer_id", "")),
        )
        selected = str(getattr(workspace, "_selected_layer_id", ""))
        if selected:
            self._layer_dialog.set_pipeline(self._pipeline_snapshot(project_id, selected))
        self._layer_dialog.show()
        self._layer_dialog.raise_()
        self._layer_dialog.activateWindow()

    def _layer_manager_select(self, layer_id: str) -> None:
        workspace = self._workspace
        if workspace is None:
            return
        item = workspace.layer_model.layer_by_id(layer_id)
        if item is None:
            return
        for index in range(workspace.layer_tabs.count()):
            if str(workspace.layer_tabs.tabData(index)) == layer_id:
                if workspace.layer_tabs.currentIndex() != index:
                    workspace.layer_tabs.setCurrentIndex(index)
                    return
                break
        self._select_layer(workspace, self._project_id, item)

    def _layer_manager_reorder(self, layer_ids: tuple[str, ...]) -> None:
        workspace = self._workspace
        project = self.service.get_project(self._project_id)
        if workspace is None or project is None:
            return
        layers = self.service.list_layers(project.id)
        try:
            self.service.reorder_layers(
                principal=self.session.principal,
                project=project,
                layers=layers,
                layer_ids=layer_ids,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
        latest = self.service.get_project(project.id)
        if latest is not None:
            self._load_layers(workspace, latest)

    def _layer_manager_activate(self, representation_id: str) -> None:
        workspace = self._workspace
        if workspace is None:
            return
        layer_id = str(getattr(workspace, "_selected_layer_id", ""))
        representations = self.service.list_representations(self._project_id, layer_id)
        selected = next(
            (item for item in representations if str(item.id) == str(representation_id)),
            None,
        )
        if (
            selected is not None
            and selected.kind is RepresentationKind.VECTOR
            and selected.source_image_representation_id is not None
        ):
            # A vector is rendered together with its linked raster.  Activating
            # both keeps the hidden compatibility selectors and the matrix data
            # source on the same pair selected in the graph.
            self._activate_representation(
                workspace,
                self._project_id,
                str(selected.source_image_representation_id),
            )
        self._activate_representation(workspace, self._project_id, representation_id)
        if self._layer_dialog is not None and layer_id:
            self._layer_dialog.set_pipeline(self._pipeline_snapshot(self._project_id, layer_id))

    def _pipeline_snapshot(self, project_id, layer_id) -> LayerPipelineSnapshot:
        representations = self.service.list_representations(project_id, layer_id)
        project_history = self.service.history(project_id)
        histories = [
            event
            for event in project_history
            if event.event_type == "PluginJobCreated"
            and str(event.payload.get("manifest", {}).get("layer_id", "")) == str(layer_id)
        ]
        latest_job_events = {}
        for event in project_history:
            job = event.payload.get("job", {})
            if not isinstance(job, dict) and not hasattr(job, "get"):
                continue
            job_id = str(event.payload.get("plugin_job_id", job.get("id", "")))
            if job_id and str(job.get("layer_id", "")) == str(layer_id):
                latest_job_events[job_id] = event
        removed_action_ids = {
            str(event.payload.get("action_event_id", ""))
            for event in project_history
            if event.event_type == "LayerPipelineActionRemoved"
            and str(event.payload.get("layer_id", "")) == str(layer_id)
        }
        action_events = [
            event
            for event in project_history
            if event.event_type == "LayerPipelineActionRequested"
            and str(event.payload.get("layer_id", "")) == str(layer_id)
            and event.event_id not in removed_action_ids
        ]
        sources = [
            value for value in representations
            if value.kind is RepresentationKind.IMAGE and value.purpose is RepresentationPurpose.SOURCE
        ]
        binaries = [
            value for value in representations
            if value.kind is RepresentationKind.IMAGE and value.purpose is RepresentationPurpose.BINARY
        ]
        vectors = [value for value in representations if value.kind is RepresentationKind.VECTOR]
        karakal_run = self.service.latest_karakal_analysis(project_id, layer_id)
        if not sources and binaries:
            sources = binaries[:1]
        lanes: list[PipelineLane] = []
        for source in sources:
            lane_nodes: list[PipelineNode] = [
                PipelineNode(
                    str(source.id),
                    source.name,
                    "source",
                    "Исходные изображения",
                    str(source.id),
                    source.active,
                    details={"источник": source.source or "BlobStore", "создан": source.created_at.isoformat()},
                )
            ]
            lane_edges: list[tuple[str, str]] = []
            lane_binaries = [
                value
                for value in binaries
                if str(value.source_image_representation_id or "") == str(source.id)
                or (
                    value.source_image_representation_id is None
                    and source is sources[0]
                )
            ]
            lane_vectors = [
                value for value in vectors
                if str(value.source_image_representation_id or "") in {str(source.id), *(str(item.id) for item in lane_binaries)}
            ]
            target_ids = {str(item.id) for item in (*lane_binaries, *lane_vectors)}
            for action_event in action_events:
                action_payload = action_event.payload
                action_parameters = dict(action_payload.get("parameters", {}))
                action_source = str(
                    action_parameters.get("source_representation_id")
                    or action_payload.get("node_id", "")
                )
                if action_source != str(source.id):
                    # Backward-compatible fallback for old layer-level actions
                    # that were recorded before source association existed.
                    if action_source or source is not sources[0]:
                        continue
                action_id = f"action:{action_event.event_id}"
                lane_nodes.append(
                    PipelineNode(
                        action_id,
                        str(action_payload.get("action", "Действие")),
                        "job",
                        str(action_payload.get("state", "launched")),
                        state=str(action_payload.get("state", "")),
                        details={
                            "время": action_event.recorded_at.isoformat(),
                            "автор": action_event.actor.display_name,
                            "программа": action_payload.get("plugin_id", ""),
                            "capability": action_payload.get("capability", ""),
                            "режим": action_payload.get("mode", ""),
                            "параметры": action_parameters,
                            "pipeline_event_id": action_event.event_id,
                            "deletable": True,
                        },
                    )
                )
                lane_edges.append((str(source.id), action_id))
            for event in histories:
                manifest = dict(event.payload.get("manifest", {}))
                target_id = str(manifest.get("target_representation_id", ""))
                parameters = dict(manifest.get("parameters", {}))
                declared_source = str(parameters.get("source_representation_id", ""))
                if declared_source and declared_source != str(source.id):
                    continue
                if target_id not in target_ids and source is not sources[0]:
                    continue
                job_id = str(event.payload.get("plugin_job_id", event.event_id))
                current_event = latest_job_events.get(job_id, event)
                job = dict(current_event.payload.get("job", event.payload.get("job", {})))
                capability = str(manifest.get("capability", "Задание"))
                program = getattr(current_event, "program", None)
                output_id = target_id
                output_kind = ""
                output_title = ""
                if capability == "frames.dataset.prepare.v1":
                    output_id = f"{job_id}:dataset"
                    output_kind = "dataset"
                    output_title = "Выборка Contour"
                elif capability == "dataset.model.train.v1":
                    output_id = f"{job_id}:model"
                    output_kind = "model"
                    output_title = "Модель NeuralImage"
                lane_nodes.append(
                    PipelineNode(
                        job_id,
                        capability,
                        "job",
                        str(job.get("state", "queued")),
                        state=str(job.get("state", "")),
                        details={
                            "время": current_event.recorded_at.isoformat(),
                            "автор": current_event.actor.display_name,
                            "программа": (
                                ""
                                if program is None
                                else f"{program.name} {program.version or ''}".strip()
                            ),
                            "состояние": str(job.get("state", "queued")),
                            "прогресс": job.get("progress", 0),
                            "ошибка": job.get("error") or "",
                            "параметры": parameters,
                        },
                    )
                )
                lane_edges.append((str(source.id), job_id))
                if output_kind:
                    lane_nodes.append(
                        PipelineNode(
                            output_id,
                            output_title,
                            output_kind,
                            str(job.get("state", "")),
                            state=str(job.get("state", "")),
                            details={"job_id": job_id, "capability": capability},
                        )
                    )
                if output_id:
                    lane_edges.append((job_id, output_id))
            for value in lane_binaries:
                lane_nodes.append(
                    PipelineNode(
                        str(value.id), value.name, "binary", "Бинарные изображения",
                        str(value.id), value.active, details={"создан": value.created_at.isoformat()},
                    )
                )
                if not any(target == str(value.id) for _, target in lane_edges):
                    lane_edges.append((str(source.id), str(value.id)))
            for value in lane_vectors:
                lane_nodes.append(
                    PipelineNode(
                        str(value.id), value.name, "vector", "CIF",
                        str(value.id), value.active, details={"создан": value.created_at.isoformat()},
                    )
                )
                if not any(target == str(value.id) for _, target in lane_edges):
                    lane_edges.append((str(value.source_image_representation_id or source.id), str(value.id)))
            if karakal_run is not None and source is sources[0]:
                karakal_id = f"karakal:{karakal_run.run_id}"
                lane_nodes.append(
                    PipelineNode(
                        karakal_id,
                        f"Karakal · публикация {karakal_run.publication_sequence}",
                        "karakal",
                        f"{len(karakal_run.frame_confidence)} кадров",
                        details={
                            "время": karakal_run.created_at,
                            "версия": karakal_run.plugin_version,
                            "параметры": karakal_run.parameters,
                            "отчёт": karakal_run.report,
                        },
                    )
                )
                for value in lane_binaries:
                    lane_edges.append((str(value.id), karakal_id))
            if not lane_vectors:
                missing_id = f"{source.id}:missing-cif"
                lane_nodes.append(PipelineNode(missing_id, "CIF не получен", "missing", "Результат отсутствует"))
                lane_edges.append((str(source.id), missing_id))
            lanes.append(PipelineLane(str(source.id), source.name, tuple(lane_nodes), tuple(lane_edges)))
        return LayerPipelineSnapshot(str(project_id), str(layer_id), tuple(lanes))

    def _layer_manager_action(self, _layer_id: str, _node: PipelineNode, action: str) -> None:
        if action == "archive_representation":
            self._archive_layer_representation(_layer_id, _node)
            return
        if action == "add_external_vector":
            workspace = self._workspace
            if workspace is None:
                return
            snapshot = self._pipeline_snapshot(self._project_id, _layer_id)
            source_representation_id = next(
                (
                    lane.lane_id
                    for lane in snapshot.lanes
                    if any(node.node_id == _node.node_id for node in lane.nodes)
                ),
                "",
            )
            if not source_representation_id:
                self._error("Не удалось определить исходный слой изображений для CIF")
                return
            self._add_representation(
                workspace,
                self._project_id,
                RepresentationKind.VECTOR,
                source_image_id=source_representation_id,
            )
            return
        if action == "delete_pipeline_step":
            action_event_id = str(_node.details.get("pipeline_event_id", ""))
            if not action_event_id:
                self._error("Этот шаг является частью неизменяемой истории задания")
                return
            try:
                self.service.remove_layer_pipeline_action(
                    principal=self.session.principal,
                    project_id=self._project_id,
                    layer_id=_layer_id,
                    action_event_id=action_event_id,
                )
            except (OSError, ValueError) as exc:
                self._error(str(exc))
                return
            if self._layer_dialog is not None:
                self._layer_dialog.set_pipeline(
                    self._pipeline_snapshot(self._project_id, _layer_id)
                )
            return
        available, reason = self._action_availability(action)
        if not available:
            self._error(reason)
            return
        plugin_id = "contour" if action in {"prepare_dataset", "vectorize"} else "neuralimage"
        _plugin, capability, mode = self._action_requirement(action)
        parameters: dict[str, object] = {}
        snapshot = self._pipeline_snapshot(self._project_id, _layer_id)
        source_representation_id = next(
            (
                lane.lane_id
                for lane in snapshot.lanes
                if any(node.node_id == _node.node_id for node in lane.nodes)
            ),
            "",
        )
        if source_representation_id:
            parameters["source_representation_id"] = source_representation_id
        if action == "recognize_external":
            from PyQt6.QtCore import QSettings
            from PyQt6.QtWidgets import QFileDialog

            settings = QSettings("Kraken", "KrakenHub")
            key_root = f"external-model/{self._project_id}/{_layer_id}"
            key = f"{key_root}/path"
            model_path = str(settings.value(key, "") or "")
            linked_now = False
            if not model_path or not Path(model_path).is_file():
                model_path, _selected_filter = QFileDialog.getOpenFileName(
                    self.shell,
                    "Выберите готовую модель NeuralImage",
                    "",
                    "Модели (*.onnx *.pt *.pth *.h5 *.keras);;Все файлы (*)",
                )
                linked_now = bool(model_path)
            if not model_path:
                return
            try:
                stored_hash = str(settings.value(f"{key_root}/sha256", "") or "")
                stored_size = int(settings.value(f"{key_root}/size", 0) or 0)
                link = (
                    ExternalModelLink(str(Path(model_path).resolve()), stored_size, stored_hash)
                    if not linked_now and stored_size > 0 and len(stored_hash) == 64
                    else ExternalModelLink.observe(model_path)
                )
                stage_root = self.service.data_dir / "agent-staging" / str(uuid4())
                staged = link.stage(stage_root)
            except (OSError, ValueError) as exc:
                self._error(f"Не удалось подготовить внешнюю модель: {exc}")
                return
            if linked_now or not stored_hash:
                settings.setValue(key, link.path)
                settings.setValue(f"{key_root}/size", link.size)
                settings.setValue(f"{key_root}/sha256", link.observed_sha256)
            parameters = {
                "external_model_path": link.path,
                "observed_size": link.size,
                "observed_sha256": link.observed_sha256,
                "staged_relative_path": staged.relative_path,
                "used_sha256": staged.used_sha256,
                "changed_since_observation": staged.changed_since_observation,
            }
        launch_arguments: tuple[str, ...] = ()
        if plugin_id == "contour":
            try:
                launch_arguments, contour_parameters = (
                    self._contour_launch_arguments(
                        layer_id=_layer_id,
                        node=_node,
                        action=action,
                        source_representation_id=source_representation_id,
                    )
                )
            except (OSError, ValueError) as exc:
                self._error(f"Не удалось подготовить данные для Contour: {exc}")
                return
            parameters.update(contour_parameters)
        self.service.record_layer_pipeline_action(
            principal=self.session.principal,
            project_id=self._project_id,
            layer_id=_layer_id,
            action=action,
            node_id=_node.node_id,
            plugin_id=plugin_id,
            capability=capability,
            mode=mode,
            parameters=parameters,
        )
        self._launch_managed_plugin(plugin_id, arguments=launch_arguments)
        if self._layer_dialog is not None:
            self._layer_dialog.set_pipeline(self._pipeline_snapshot(self._project_id, _layer_id))

    def _layer_manager_layer_action(self, _layer_id: str, action: str) -> None:
        if action == "add_image_representation":
            workspace = self._workspace
            if workspace is None:
                return
            item = workspace.layer_model.layer_by_id(_layer_id)
            if item is None:
                self._error("Слой больше не доступен")
                return
            self._select_layer(workspace, self._project_id, item)
            self._add_representation(
                workspace,
                self._project_id,
                RepresentationKind.IMAGE,
            )
            return
        if action == "karakal":
            available, reason = self._action_availability(action)
            if not available:
                self._error(reason)
                return
            _plugin, capability, mode = self._action_requirement(action)
            self.service.record_layer_pipeline_action(
                principal=self.session.principal,
                project_id=self._project_id,
                layer_id=_layer_id,
                action=action,
                node_id=_layer_id,
                plugin_id="karakal",
                capability=capability,
                mode=mode,
            )
            if self._layer_dialog is not None:
                self._layer_dialog.set_pipeline(
                    self._pipeline_snapshot(self._project_id, _layer_id)
                )
            self._launch_karakal(_layer_id)

    @staticmethod
    def _action_requirement(action: str) -> tuple[str, str, str]:
        return {
            "prepare_dataset": ("contour", "frames.dataset.prepare.v1", "interactive"),
            "vectorize": ("contour", "frames.vectorize.v1", "interactive"),
            "train": ("neuralimage", "dataset.model.train.v1", "interactive"),
            "recognize": ("neuralimage", "frames.binary-segment.v2", "headless"),
            "recognize_external": ("neuralimage", "frames.binary-segment.v2", "headless"),
            "karakal": ("karakal", "layer.confidence.analyze.v1", "interactive"),
        }.get(action, ("", "", ""))

    def _action_availability(self, action: str) -> tuple[bool, str]:
        if action in {
            "add_external_vector",
            "add_image_representation",
            "archive_representation",
            "delete_pipeline_step",
        }:
            return True, ""
        requirement = self._action_requirement(action)
        if not requirement[0]:
            return False, f"Неизвестное действие: {action}"
        plugin_id, operation, mode = requirement
        inventory = self.plugin_items.get(plugin_id)
        if inventory is None or not inventory.installed or not inventory.metadata.enabled:
            return False, f"Плагин {plugin_id} не установлен или отключён"
        capability = next(
            (
                item
                for item in inventory.metadata.capabilities
                if item.operation == operation and mode in item.modes
            ),
            None,
        )
        if capability is None:
            return False, f"Плагин {plugin_id} не поддерживает {operation} в режиме {mode}"
        return True, ""

    def _contour_launch_arguments(
        self,
        *,
        layer_id: str,
        node: PipelineNode,
        action: str,
        source_representation_id: str,
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        representations = self.service.list_representations(
            self._project_id, layer_id
        )
        input_representation_id = str(
            node.representation_id or source_representation_id
        )
        representation = next(
            (
                item
                for item in representations
                if str(item.id) == input_representation_id
            ),
            None,
        )
        if representation is None or representation.kind is not RepresentationKind.IMAGE:
            raise ValueError("выбранный базовый слой изображений не найден")

        stage = (
            self.service.data_dir
            / "agent-staging"
            / f"contour-{action}-{uuid4()}"
        ).resolve()
        stage.mkdir(parents=True, exist_ok=False)
        source_path = Path(str(representation.source or "")).expanduser()
        if source_path.is_absolute() and source_path.is_dir():
            input_directory = source_path.resolve()
        else:
            input_directory = stage / "inputs"
            input_directory.mkdir()
            cells = self.service.frame_cells(
                self._project_id,
                layer_id,
                representation.id,
            )
            for cell in cells:
                if not cell.sha256:
                    continue
                destination = input_directory / f"{cell.x}_{cell.y}.png"
                with destination.open("xb") as stream:
                    stream.write(
                        self.service.read_project_blob(
                            self._project_id, cell.sha256
                        )
                    )
            if not any(input_directory.iterdir()):
                raise ValueError("в базовом слое нет доступных изображений")

        destination = stage / (
            "dataset" if action == "prepare_dataset" else "vectors"
        )
        destination.mkdir()
        destination_option = (
            "--dataset-dir"
            if action == "prepare_dataset"
            else "--output-dir"
        )
        return (
            (
                "--input-dir",
                str(input_directory),
                destination_option,
                str(destination),
            ),
            {
                "input_representation_id": str(representation.id),
                "input_directory": str(input_directory),
                "result_directory": str(destination),
            },
        )

    def _launch_managed_plugin(
        self,
        plugin_id: str,
        *,
        arguments: tuple[str, ...] = (),
    ) -> None:
        inventory = self.plugin_items.get(plugin_id)
        if inventory is None or not inventory.installed or not inventory.metadata.enabled:
            self._error(f"Плагин {plugin_id} не установлен или отключён")
            return
        from .app import launch_plugin
        launch_plugin(inventory.metadata, arguments=arguments)

    def _launch_karakal(self, layer_id: str) -> None:
        inventory = self.plugin_items.get("karakal")
        if inventory is None:
            self._error("Плагин karakal не установлен")
            return
        try:
            from PyQt6.QtCore import QSettings, Qt, QUrl
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtWidgets import QDialog, QVBoxLayout
            from karakal.core.domain import FolderSpec
            from karakal.plugin.plugin import KarakalPlugin
        except ImportError as exc:
            self._error(f"Не удалось загрузить Karakal: {exc}")
            return

        controller = self
        project_id = str(self._project_id)
        frame_key_map: dict[str, str] = {}
        project = self.service.get_project(project_id)
        if project is not None:
            for representation in self.service.list_representations(project_id, layer_id):
                if not representation.source or not Path(representation.source).is_dir():
                    continue
                paths = sorted(
                    (
                        path
                        for path in Path(representation.source).iterdir()
                        if path.is_file()
                    ),
                    key=self.service._natural_path_key,
                )
                for index, path in enumerate(paths[: project.width * project.height]):
                    frame_id = str(
                        project.frame_id_at(
                            index % project.width + 1,
                            index // project.width + 1,
                        )
                    )
                    for key in (path.name, path.stem, str(path), frame_id):
                        frame_key_map[key] = frame_id

        class KrakenHost:
            def settings(self):
                return QSettings("Kraken", "Karakal")

            def logger(self):
                return logging.getLogger("kraken.karakal")

            def task_runner(self):
                return None

            def open_path(self, path: Path) -> None:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

            def publish_quality(self, payload: dict) -> None:
                confidences = {
                    frame_key_map.get(
                        str(item.get("frame_key", "")),
                        str(item.get("frame_key", "")),
                    ): float(item.get("confidence", 0.0))
                    for item in payload.get("frames", ())
                    if str(item.get("frame_key", "")).strip()
                }
                controller.service.publish_karakal_analysis(
                    principal=controller.session.principal,
                    project_id=project_id,
                    layer_id=layer_id,
                    frame_confidence=confidences,
                    report=dict(payload),
                    parameters=dict(payload.get("parameters", {})),
                    plugin_version=inventory.metadata.version,
                    idempotency_key=f"karakal:{uuid4()}",
                )
                workspace = controller._workspace
                if workspace is not None:
                    controller._refresh_matrix(workspace, project_id)
                if controller._layer_dialog is not None:
                    controller._layer_dialog.set_pipeline(
                        controller._pipeline_snapshot(project_id, layer_id)
                    )

        plugin = KarakalPlugin()
        dialog = QDialog(self.shell)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setModal(False)
        dialog.setWindowTitle(f"Karakal · {inventory.metadata.version}")
        dialog.resize(1380, 860)
        layout = QVBoxLayout(dialog)
        widget = plugin.create_widget(KrakenHost(), dialog)
        layout.addWidget(widget)

        # External directory representations can be transferred immediately.
        # Managed BlobStore representations remain available through Agent
        # staging and are not exposed as arbitrary host filesystem paths.
        representations = self.service.list_representations(project_id, layer_id)
        karakal_stage = (
            self.service.data_dir / "agent-staging" / f"karakal-{uuid4()}"
        ).resolve()
        karakal_stage.mkdir(parents=True, exist_ok=False)
        staged_folders: dict[str, Path | None] = {}

        def representation_folder(representation) -> Path | None:
            identifier = str(representation.id)
            if identifier in staged_folders:
                return staged_folders[identifier]
            if representation.source and Path(representation.source).is_dir():
                result = Path(representation.source).resolve()
                staged_folders[identifier] = result
                return result
            cells = self.service.frame_cells(
                project_id,
                layer_id,
                representation.id,
            )
            if not cells:
                staged_folders[identifier] = None
                return None
            folder = karakal_stage / identifier
            folder.mkdir()
            for cell in cells:
                if not cell.sha256:
                    continue
                destination = folder / f"{cell.x}_{cell.y}.png"
                with destination.open("xb") as stream:
                    stream.write(self.service.read_project_blob(project_id, cell.sha256))
            result = folder if any(folder.iterdir()) else None
            staged_folders[identifier] = result
            return result

        source = next(
            (
                item
                for item in representations
                if item.purpose is RepresentationPurpose.SOURCE
                and representation_folder(item) is not None
            ),
            None,
        )
        if source is not None:
            source_path = representation_folder(source)
            assert source_path is not None
            widget._presenter._original_folder = FolderSpec(
                path=source_path,
                label=source.name,
            )
        for binary in representations:
            binary_path = (
                representation_folder(binary)
                if binary.purpose is RepresentationPurpose.BINARY
                else None
            )
            if binary_path is not None:
                widget._presenter._append_folder_item(
                    binary_path,
                    checked=True,
                )
        widget._presenter._update_source_labels()
        widget._presenter._refresh_folder_rows()
        widget._presenter._sync_action_buttons()

        windows = getattr(self, "_plugin_windows", None)
        if windows is None:
            windows = []
            self._plugin_windows = windows
        windows.append((dialog, plugin))

        def dispose() -> None:
            plugin.shutdown()
            if (dialog, plugin) in windows:
                windows.remove((dialog, plugin))
            staging_root = (self.service.data_dir / "agent-staging").resolve()
            try:
                karakal_stage.relative_to(staging_root)
            except ValueError:
                return
            if karakal_stage.is_dir() and not karakal_stage.is_symlink():
                shutil.rmtree(karakal_stage)

        dialog.finished.connect(lambda _result: dispose())
        dialog.show()

    def _set_matrix_visual_mode(self, _channel: str, _mode: str) -> None:
        workspace = self._workspace
        if workspace is not None:
            workspace.matrix_view.set_visual_modes(*self.shell.visual_modes())

    def _load_representations(self, workspace, project_id, layer_id) -> None:
        representations = self.service.list_representations(project_id, layer_id)
        images = sorted(
            (value for value in representations if value.kind is RepresentationKind.IMAGE),
            key=lambda value: (not value.active, value.name.casefold()),
        )
        current_image_id = str(workspace.image_representation_combo.currentData() or "")
        if not any(str(item.id) == current_image_id for item in images):
            current_image_id = str(images[0].id) if images else ""
        workspace.set_representations(
            images=[
                (str(item.id), item.name)
                for item in images
            ],
            vectors=[
                (str(item.id), item.name)
                for item in sorted(
                    (
                        value
                        for value in representations
                        if value.kind is RepresentationKind.VECTOR
                        and str(value.source_image_representation_id or "") == current_image_id
                    ),
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
        representation_ids = tuple(
            str(value)
            for value in (
                workspace.image_representation_combo.currentData(),
                workspace.vector_representation_combo.currentData(),
            )
            if value
        )
        data_source = getattr(workspace, "_matrix_data_source", None)
        if data_source is None or not isinstance(workspace.matrix_view, FrameMatrixWidget):
            return
        data_source.set_context(
            layer_id=str(layer_id),
            representation_ids=representation_ids,
            matrix_width=workspace.matrix_view.matrix_width,
        )
        project = self.service.get_project(project_id)
        if project is None:
            workspace.matrix_view.clear_cells()
            return
        previous = workspace.matrix_view.session()
        generation = 1 if previous is None else previous.generation + 1
        workspace.matrix_view.set_session(
            MatrixSession(
                namespace=str(project.id),
                width=project.width,
                height=project.height,
                source_revision=f"{project.revision}:{layer_id}:{','.join(representation_ids)}",
                orientation=project.orientation.value,
                generation=generation,
            ),
            data_source=data_source,
            asset_source=getattr(workspace, "_matrix_asset_source", None),
        )

    def _archive_layer_representation(self, layer_id: str, node: PipelineNode) -> None:
        from PyQt6.QtWidgets import QMessageBox

        workspace = self._workspace
        project = self.service.get_project(self._project_id)
        layer = next(
            (
                item
                for item in self.service.list_layers(self._project_id)
                if str(item.id) == str(layer_id)
            ),
            None,
        )
        representation_id = str(node.representation_id or node.node_id)
        representation = next(
            (
                item
                for item in self.service.list_representations(
                    self._project_id, layer_id
                )
                if str(item.id) == representation_id
            ),
            None,
        )
        if workspace is None or project is None or layer is None or representation is None:
            self._error("Слой изображений больше не доступен")
            return
        answer = QMessageBox.question(
            self._layer_dialog or workspace,
            "Удалить слой изображений",
            (
                f"Удалить «{representation.name}» из проекта?\n\n"
                "Связанная история обработки останется в журнале проекта."
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.archive_representation(
                principal=self.session.principal,
                project=project,
                layer=layer,
                representation=representation,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self._load_representations(workspace, self._project_id, layer_id)
        if self._layer_dialog is not None:
            self._layer_dialog.set_pipeline(
                self._pipeline_snapshot(self._project_id, layer_id)
            )

    def _add_representation(
        self,
        workspace,
        project_id,
        kind: RepresentationKind,
        *,
        source_image_id: str | None = None,
    ) -> None:
        from PyQt6.QtWidgets import QDialog, QMessageBox

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
        if kind is RepresentationKind.VECTOR and not source_image_id:
            source_image_id = str(
                workspace.image_representation_combo.currentData() or ""
            )
        if kind is RepresentationKind.IMAGE:
            source_image_id = None
        if kind is RepresentationKind.VECTOR and not source_image_id:
            QMessageBox.information(
                workspace, "Kraken", "Сначала добавьте и выберите слой изображения."
            )
            return
        form = RepresentationDialog(workspace, kind)
        while form.exec() == QDialog.DialogCode.Accepted:
            if not form.name.text().strip():
                QMessageBox.warning(
                    form.dialog, f"Не удалось добавить {label}", "Введите название."
                )
                form.name.setFocus()
                continue
            import_plan = None
            source_directory = form.source_directory
            if source_directory:
                try:
                    import_plan = self.service.plan_import_directory(
                        project=project,
                        directory=source_directory,
                        mode=ImportMappingMode.ROW_MAJOR_SUFFIX,
                    )
                except Exception as exc:
                    QMessageBox.warning(form.dialog, "Ошибка проверки файлов", str(exc))
                    continue
                issue_text = "\n".join(
                    f"• {issue.message}" for issue in import_plan.issues
                ) or "Ошибок не обнаружено"
                if not import_plan.ready:
                    QMessageBox.warning(
                        form.dialog,
                        "Импорт заблокирован",
                        f"Preflight не пройден:\n{issue_text}",
                    )
                    continue
                answer = QMessageBox.question(
                    form.dialog,
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
                    name=form.name.text(),
                    kind=kind,
                    idempotency_key=str(uuid4()),
                    note=form.note.toPlainText(),
                    source="managed-import" if import_plan is not None else None,
                    source_image_representation_id=source_image_id,
                    active=form.active.isChecked(),
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
                        form.dialog,
                        "Импорт завершён",
                        f"Создано immutable версий: {len(imported.versions)}",
                    )
                    if import_plan.missing_coordinates:
                        QMessageBox.warning(
                            form.dialog,
                            "Неполный набор файлов",
                            (
                                f"Не найдено файлов для кадров: "
                                f"{import_plan.missing_coordinates:n}. "
                                "Соответствующие ячейки отмечены красным."
                            ),
                        )
            except Exception as exc:
                QMessageBox.warning(form.dialog, f"Не удалось добавить {label}", str(exc))
                continue
            self._load_representations(workspace, project_id, layer_id)
            combo = (
                workspace.image_representation_combo
                if kind is RepresentationKind.IMAGE
                else workspace.vector_representation_combo
            )
            combo.setCurrentIndex(combo.findData(str(representation.id)))
            if self._layer_dialog is not None:
                self._layer_dialog.set_pipeline(
                    self._pipeline_snapshot(project_id, layer_id)
                )
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


def run_manager_gui(
    items: list[PluginInventoryItem],
    *,
    update_url: str = "",
    thumbnail_store_uri: str = "",
) -> int:
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
    shell._desktop_controller = DesktopController(  # keep Qt slots alive
        shell,
        service,
        session,
        thumbnail_store_uri=thumbnail_store_uri,
        plugin_items=items,
    )
    shell.show()
    return app.exec()


__all__ = ["DesktopController", "run_manager_gui"]
