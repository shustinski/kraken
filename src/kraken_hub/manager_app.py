"""PyQt Desktop composition for projects, matrix and legacy plugin launcher."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
import os
import sys
import logging
import shutil
import hashlib
from pathlib import Path
from threading import Event
from uuid import uuid4

from PyQt6.QtCore import QThread, QTimer, pyqtSignal

from kraken_core.frame_matrix import (
    MatrixSession,
    StoreNamespace,
    StorePolicy,
    ThumbnailStoreFactory,
)
from kraken_core.frame_matrix.qt import FrameMatrixWidget
from kraken_core.external_model import ExternalModelLink
from kraken_core.plugins import PluginInventoryItem
from kraken_core.plugin_protocol import (
    PluginFrameInput,
    PluginJobManifest,
    PluginResultManifest,
    WorkspacePluginContextV1,
    WorkspacePluginResultV1,
)
from kraken_core.qt import configure_application_identity
from kraken_core.styles import load_shared_stylesheet
from kraken_manager.domain.project import GridOrientation as DomainOrientation
from kraken_manager.domain.project import LayerType, RepresentationKind, RepresentationPurpose
from kraken_manager.domain.artifacts import ArtifactScope, ArtifactSeries, ArtifactVersion
from kraken_manager.domain.identity import Permission, Performer, ProjectRole
from kraken_manager.application.imports import ImportMappingMode
from kraken_manager.workspace import DerivedRunKind
from kraken_manager.infrastructure.workspace_files import validate_workspace_roots
from kraken_manager.infrastructure.plugin import AgentPluginGateway
from kraken_manager.presentation.qt import (
    LayerCreationDialog,
    LayerManagerDialog,
    LayerPipelineSnapshot,
    ObjectHistoryEntry,
    ObjectPropertiesDialog,
    ObjectPropertiesSnapshot,
    PipelineLane,
    PipelineNode,
    ProjectManagerShell,
    ProjectWorkspacePage,
)
from kraken_manager.presentation.qt.models import LayerListItem, ProjectListItem
from kraken_manager.presentation.qt.widgets import ClickableLabel, GridDimensionsWidget

from . import windows_credentials
from .composition import DesktopSession, EmbeddedProjectService
from .dual_catalog import DualCatalogService, build_workspace_service
from .matrix_source import KrakenMatrixAssetSource, KrakenMatrixDataSource
from .agent_runtime import LocalAgentRuntime
from .remote_client import load_remote_auth_from_env
from .workspace_service import REMOTE_STORAGE_PROFILE, ProjectEventWake


def _plain_value(value):
    if is_dataclass(value):
        return _plain_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_plain_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _event_payload(event) -> dict[str, object]:
    payload = getattr(event, "payload", {})
    return _plain_value(payload) if isinstance(payload, Mapping) else {}


def _nested_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _event_belongs_to_layer(event, layer_id: str) -> bool:
    identifier = str(layer_id)
    payload = _event_payload(event)
    candidates = {
        str(payload.get("layer_id", "")),
        str(_nested_mapping(payload, "layer").get("id", "")),
        str(_nested_mapping(payload, "job").get("layer_id", "")),
        str(_nested_mapping(payload, "manifest").get("layer_id", "")),
    }
    if identifier in candidates:
        return True
    if identifier in {str(value) for value in payload.get("layer_ids", ())}:
        return True
    stream_id = str(getattr(event, "stream_id", ""))
    return stream_id in {
        f"layer:{identifier}",
        f"layer-files:{identifier}",
        f"layer-pipeline:{identifier}",
        f"karakal:{identifier}",
    }


def _node_identifiers(node: PipelineNode) -> set[str]:
    identifiers = {str(node.node_id), str(node.representation_id or "")}
    for key in ("job_id", "run_id", "pipeline_event_id"):
        value = node.details.get(key)
        if value:
            identifiers.add(str(value))
    for prefix in ("action:", "workspace-job:", "workspace-output:", "karakal:"):
        if str(node.node_id).startswith(prefix):
            identifiers.add(str(node.node_id)[len(prefix):])
    if ":" in str(node.node_id):
        identifiers.add(str(node.node_id).split(":", 1)[0])
    return {value for value in identifiers if value}


def _event_matches_node(event, node: PipelineNode) -> bool:
    identifiers = _node_identifiers(node)
    if not identifiers or node.kind in {"missing", "blackbox"}:
        return False
    payload = _event_payload(event)
    manifest = _nested_mapping(payload, "manifest")
    parameters = _nested_mapping(manifest, "parameters")
    action_parameters = _nested_mapping(payload, "parameters")
    job = _nested_mapping(payload, "job")
    representation = _nested_mapping(payload, "representation")
    candidates = {
        str(getattr(event, "event_id", "")),
        str(payload.get("representation_id", "")),
        str(payload.get("plugin_job_id", "")),
        str(payload.get("run_id", "")),
        str(payload.get("action_event_id", "")),
        str(payload.get("node_id", "")),
        str(job.get("id", "")),
        str(job.get("target_representation_id", "")),
        str(manifest.get("target_representation_id", "")),
        str(manifest.get("source_representation_id", "")),
        str(parameters.get("source_representation_id", "")),
        str(action_parameters.get("source_representation_id", "")),
        str(payload.get("source_image_representation_id", "")),
        str(representation.get("id", "")),
        str(representation.get("source_image_representation_id", "")),
    }
    deactivated = {
        str(value) for value in payload.get("deactivated_representation_ids", ())
    }
    for value in payload.get("deactivated", ()):
        if isinstance(value, Mapping):
            deactivated.add(str(value.get("id", "")))
    if identifiers.intersection(candidates | deactivated):
        return True
    stream_id = str(getattr(event, "stream_id", ""))
    return any(stream_id.endswith(f":{identifier}") for identifier in identifiers)


def _history_entries(events) -> tuple[ObjectHistoryEntry, ...]:
    ordered = sorted(
        events,
        key=lambda event: (
            getattr(event, "recorded_at", None),
            str(getattr(event, "event_id", "")),
        ),
        reverse=True,
    )
    return tuple(
        ObjectHistoryEntry(
            recorded_at=getattr(event, "recorded_at").isoformat(),
            actor=str(getattr(getattr(event, "actor", None), "display_name", "") or "—"),
            event_type=str(getattr(event, "event_type", "")),
            payload=_event_payload(event),
        )
        for event in ordered
    )


def _creator(events) -> tuple[str, str]:
    ordered = sorted(
        events,
        key=lambda event: (
            getattr(event, "recorded_at", None),
            str(getattr(event, "event_id", "")),
        ),
    )
    if not ordered:
        return "—", "—"
    event = ordered[0]
    actor = str(getattr(getattr(event, "actor", None), "display_name", "") or "—")
    recorded_at = getattr(event, "recorded_at", None)
    return actor, "—" if recorded_at is None else recorded_at.isoformat()


def _count_regular_files(paths) -> int | None:
    candidates = [Path(str(value)) for value in paths if str(value or "").strip()]
    if not candidates:
        return None
    seen: set[str] = set()
    accessible = False
    for candidate in candidates:
        try:
            if candidate.is_file():
                accessible = True
                seen.add(str(candidate.resolve()).casefold())
                continue
            if not candidate.is_dir():
                continue
            accessible = True
            for path in candidate.rglob("*"):
                try:
                    if path.is_file():
                        seen.add(str(path.resolve()).casefold())
                except OSError:
                    continue
        except OSError:
            continue
    return len(seen) if accessible else None


class _LayerCreateThread(QThread):
    progress = pyqtSignal(int, int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function, *, cancelled: Event, parent=None) -> None:
        super().__init__(parent)
        self._function = function
        self.cancelled = cancelled

    def run(self) -> None:
        try:
            result = self._function(
                progress=lambda done, total, label: self.progress.emit(done, total, label),
                cancelled=self.cancelled,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)


class _BackgroundCallThread(QThread):
    """Run one blocking service call without occupying the Qt event loop."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function, *, parent=None) -> None:
        super().__init__(parent)
        self._function = function

    def run(self) -> None:
        try:
            result = self._function()
        except Exception as exc:  # noqa: BLE001 - Qt boundary reports service failures
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)


def _volume_key(path: Path) -> str:
    text = str(path)
    if text.startswith("\\\\"):
        parts = [part for part in text.split("\\") if part]
        return "\\\\" + "\\".join(parts[:2]).casefold()
    return path.anchor.casefold()


def _configure_workspace_roots(parent, service, *, force: bool = False) -> tuple[str, str] | None:
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    settings = QSettings("Kraken", "KrakenHub")
    source_value = str(settings.value("workspace/source-root", "") or "")
    derived_value = str(settings.value("workspace/derived-root", "") or "")
    if not force and source_value and derived_value:
        try:
            source, derived = validate_workspace_roots(source_value, derived_value)
            return str(source), str(derived)
        except Exception:
            pass

    dialog = QDialog(parent)
    dialog.setObjectName("workspaceRootsDialog")
    dialog.setWindowTitle("Хранилища проектов")
    dialog.setMinimumWidth(680)
    root_layout = QVBoxLayout(dialog)
    hint = QLabel(
        "Выберите два корневых каталога. В каждом новом проекте Kraken создаст "
        "одноимённую папку; изменение настроек не переносит существующие проекты.",
        dialog,
    )
    hint.setWordWrap(True)
    root_layout.addWidget(hint)
    form = QFormLayout()

    def directory_row(value: str, object_name: str, title: str):
        host = QWidget(dialog)
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(value, host)
        edit.setObjectName(object_name)
        button = QPushButton("Обзор…", host)
        button.clicked.connect(
            lambda: (
                (selected := QFileDialog.getExistingDirectory(dialog, title, edit.text().strip()))
                and edit.setText(selected)
            )
        )
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return host, edit

    source_host, source_edit = directory_row(
        source_value or str(service.default_source_root),
        "workspaceSourceRoot",
        "Корень исходных данных",
    )
    derived_host, derived_edit = directory_row(
        derived_value or str(service.default_derived_root),
        "workspaceDerivedRoot",
        "Корень производных данных",
    )
    form.addRow("Исходные данные", source_host)
    form.addRow("Производные данные", derived_host)
    root_layout.addLayout(form)
    error_label = QLabel(dialog)
    error_label.setObjectName("workspaceRootsError")
    error_label.setWordWrap(True)
    error_label.setStyleSheet("color:#fca5a5;")
    root_layout.addWidget(error_label)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        dialog,
    )
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Сохранить")
    buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
    buttons.rejected.connect(dialog.reject)

    def accept() -> None:
        try:
            source, derived = validate_workspace_roots(source_edit.text(), derived_edit.text())
        except Exception as exc:
            error_label.setText(str(exc))
            return
        if _volume_key(source) == _volume_key(derived):
            answer = QMessageBox.warning(
                dialog,
                "Один том",
                "Оба каталога находятся на одном томе или UNC-ресурсе. Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        settings.setValue("workspace/source-root", str(source))
        settings.setValue("workspace/derived-root", str(derived))
        dialog.accept()

    buttons.accepted.connect(accept)
    root_layout.addWidget(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return (
        str(settings.value("workspace/source-root", "")),
        str(settings.value("workspace/derived-root", "")),
    )


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
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
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
        service: EmbeddedProjectService | DualCatalogService,
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
        self.agent_runtime = LocalAgentRuntime(
            self.service.data_dir / "agent",
            tuple(plugin_items or ()),
        )
        self._agent_gateway: AgentPluginGateway | None = None
        self._agent_poll_error = ""
        self.my_work_refresh: Callable[[], None] | None = None
        self._agent_timer = QTimer(self.shell)
        self._agent_timer.setInterval(2000)
        self._agent_timer.timeout.connect(self._poll_agent_jobs)
        self._agent_timer.start()
        QTimer.singleShot(0, self._resume_agent_jobs)
        self.shell.close_guard = self._allow_close
        self._workspace = None
        self._project_id = None
        self._layer_dialog: LayerManagerDialog | None = None
        self._background_workers: set[QThread] = set()
        self._source_preparation_keys: set[tuple[str, str]] = set()
        self._pending_wakes: list[ProjectEventWake] = []
        self._sync_timer = QTimer(self.shell)
        self._sync_timer.setSingleShot(True)
        self._sync_timer.timeout.connect(self._drain_live_wakes)
        self.catalog_page = shell.page("projects")
        assert self.catalog_page is not None
        self.catalog_page.createRequested.connect(self.create_project)
        self.catalog_page.refreshRequested.connect(self.refresh_projects)
        self.catalog_page.projectActivated.connect(self.open_project)
        self.catalog_page.renameRequested.connect(self.rename_project)
        self.catalog_page.archiveRequested.connect(self.archive_project)
        self.catalog_page.restoreRequested.connect(self.restore_project)
        self.catalog_page.deleteRequested.connect(self.delete_project)
        self.catalog_page.participantsRequested.connect(self.manage_project_participants)
        self.catalog_page.propertiesRequested.connect(self.show_project_properties)
        self.catalog_page.selectionChanged.connect(self._sync_project_permissions)
        self.shell.layersRequested.connect(self.open_layer_manager)
        self.shell.cellVisualModeChanged.connect(self._set_matrix_visual_mode)
        self.shell.reviewReturnRequested.connect(self.load_review_return)
        self.shell.framePropertiesRequested.connect(self.show_selected_frame)
        self._wire_live_sync()
        self.refresh_projects()

    def _wire_live_sync(self) -> None:
        add_wake = getattr(self.service, "add_wake_handler", None)
        add_catalog = getattr(self.service, "add_catalog_handler", None)
        start_sync = getattr(self.service, "start_sync", None)
        if callable(add_wake):
            add_wake(self._on_remote_wake)
        if callable(add_catalog):
            add_catalog(lambda: self._on_remote_wake(None))
        if callable(start_sync):
            try:
                start_sync()
            except Exception:
                logging.getLogger(__name__).exception("Failed to start remote live sync")

    def _on_remote_wake(self, wake: ProjectEventWake | None) -> None:
        if wake is not None:
            self._pending_wakes.append(wake)
        if not self._sync_timer.isActive():
            self._sync_timer.start(50)

    def _drain_live_wakes(self) -> None:
        wakes = list(self._pending_wakes)
        self._pending_wakes.clear()
        catalog_changed = not wakes or any(
            wake.event_type.startswith("Project") or wake.event_type.startswith("project.")
            for wake in wakes
        )
        if catalog_changed:
            self.refresh_projects()
        if self._project_id is None or self._workspace is None:
            return
        relevant = [
            wake
            for wake in wakes
            if wake is not None and wake.project_id == self._project_id
        ]
        if not wakes or relevant:
            project = self.service.get_project(self._project_id)
            if project is None:
                return
            self._workspace.set_project_title(project.name)
            matrix = getattr(self._workspace, "matrix_view", None)
            if matrix is not None:
                session = getattr(matrix, "session", None)
                if session is not None:
                    matrix.set_session(
                        MatrixSession(
                            namespace=str(project.id),
                            width=project.width,
                            height=project.height,
                            source_revision=str(project.revision),
                            orientation=project.orientation.value,
                            generation=getattr(session, "generation", 0) + 1,
                        ),
                        data_source=getattr(self._workspace, "_matrix_data_source", None),
                        asset_source=getattr(self._workspace, "_matrix_asset_source", None),
                    )
            self._load_layers(self._workspace, project)

    def _sync_project_permissions(self, item: ProjectListItem | None) -> None:
        permissions = (
            frozenset()
            if item is None
            else frozenset(
                permission.value
                for permission in self.service.project_permissions(
                    item.project_id,
                    self.session.principal,
                )
            )
        )
        self.catalog_page.set_project_permissions(permissions)

    def _has_permission(self, permission: Permission) -> bool:
        if self._project_id is None:
            return False
        return permission in self.service.project_permissions(
            self._project_id,
            self.session.principal,
        )

    def _resume_agent_jobs(self) -> None:
        active = next(
            (
                job
                for job in self.service.plugin_jobs()
                if job.state.value not in {"succeeded", "failed", "cancelled"}
            ),
            None,
        )
        if active is None:
            return
        try:
            self._gateway_for_job(active)
            self._poll_agent_jobs()
        except Exception as exc:
            self.shell.statusBar().showMessage(
                f"Kraken Agent: {exc}",
                10000,
            )

    def _poll_agent_jobs(self) -> None:
        gateway = self._agent_gateway
        if gateway is None:
            return
        try:
            jobs = self.service.synchronize_plugin_jobs(
                principal=self.session.principal,
                gateway=gateway,
            )
            for job in jobs:
                if job.state.value != "importing":
                    continue
                publications = gateway.get_publications(job.id)
                if publications:
                    imported = self.service.import_agent_publications(
                        principal=self.session.principal,
                        publications=publications,
                        staging_root=self.service.data_dir / "agent" / "staging",
                    )
                else:
                    payload = gateway.get_result(job.id)
                    imported = self.service.import_agent_result(
                        principal=self.session.principal,
                        result_payload=payload,
                        staging_root=self.service.data_dir / "agent" / "staging",
                    )
                if not imported.requires_partial_confirmation:
                    gateway.complete_import(job.id)
                    self._refresh_matrix()
            if self.my_work_refresh is not None:
                self.my_work_refresh()
            self._agent_poll_error = ""
        except Exception as exc:
            message = str(exc)
            if message != self._agent_poll_error:
                self.shell.statusBar().showMessage(
                    f"Kraken Agent: {message}",
                    10000,
                )
                self._agent_poll_error = message

    def _allow_close(self) -> bool:
        from PyQt6.QtWidgets import QMessageBox

        if any(
            worker.isRunning()
            for worker in getattr(self, "_background_workers", ())
        ):
            QMessageBox.information(
                self.shell,
                "Сохранение файлов ещё не завершено",
                "Дождитесь окончания сохранения файлов. Выход отменён, чтобы данные не были повреждены.",
            )
            return False
        active = tuple(
            job
            for job in self.service.plugin_jobs()
            if job.state.value not in {"succeeded", "failed", "cancelled"}
        )
        if active:
            QMessageBox.information(
                self.shell,
                "Есть активные задания",
                "Дождитесь завершения заданий Kraken Agent или отмените их во вкладке «Моя работа». "
                "Выход отменён.",
            )
            return False
        try:
            self.agent_runtime.shutdown()
        except Exception as exc:
            QMessageBox.warning(
                self.shell,
                "Kraken Agent",
                f"Не удалось корректно остановить Agent: {exc}",
            )
            return False
        return True

    def _gateway_for_job(self, job) -> AgentPluginGateway:
        project = self.service.get_project(job.project_id)
        if project is None:
            raise ValueError("Проект задания больше не доступен")
        capabilities = self.agent_runtime.ensure_started()
        coordinate_by_frame = {
            str(coordinate.frame_id(project.id)): (coordinate.x, coordinate.y)
            for coordinate in job.selection.iter_coordinates()
        }

        def version_path(version_id):
            return self.service.managed_artifact_path(project.id, version_id)

        def frame_coordinate(frame_id):
            coordinate = coordinate_by_frame.get(str(frame_id))
            if coordinate is None:
                raise ValueError(f"Кадр {frame_id} не входит в задание")
            return coordinate

        def media_type(version_id):
            version = self.service.artifact_version(project.id, version_id)
            if version is None:
                raise ValueError(f"Версия {version_id} больше не доступна")
            return version.media_type

        gateway = AgentPluginGateway(
            base_url=self.agent_runtime.base_url,
            token=self.agent_runtime.token,
            staging_root=self.service.data_dir / "agent" / "staging",
            source_for_version=version_path,
            coordinate_for_frame=frame_coordinate,
            media_type_for_version=media_type,
            capabilities=capabilities,
            v2_capabilities=frozenset(
                operation
                for operation, protocol
                in self.agent_runtime.protocol_by_capability.items()
                if protocol == "2.0"
            ),
        )
        self._agent_gateway = gateway
        return gateway

    def cancel_agent_job(self, job) -> None:
        gateway = self._gateway_for_job(job)
        self.service.cancel_plugin_job(
            principal=self.session.principal,
            gateway=gateway,
            job=job,
            idempotency_key=str(uuid4()),
        )
        if self.my_work_refresh is not None:
            self.my_work_refresh()

    def retry_agent_job(self, job) -> None:
        gateway = self._gateway_for_job(job)
        self.service.retry_plugin_job(
            principal=self.session.principal,
            gateway=gateway,
            job=job,
            idempotency_key=str(uuid4()),
        )
        if self.my_work_refresh is not None:
            self.my_work_refresh()

    def import_partial_agent_job(self, job) -> None:
        gateway = self._gateway_for_job(job)
        gateway.confirm_partial(job.id)
        self._poll_agent_jobs()

    def manage_project_participants(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        from PyQt6.QtWidgets import (
            QCheckBox,
            QDialog,
            QDialogButtonBox,
            QMessageBox,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
        )

        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен.")
            return
        project_principals = getattr(
            self.service,
            "list_project_principals",
            None,
        )
        principals = (
            project_principals(project.id)
            if callable(project_principals)
            else self.service.list_principals()
        )
        roles = (
            ProjectRole.OWNER,
            ProjectRole.MANAGER,
            ProjectRole.CONTRIBUTOR,
            ProjectRole.REVIEWER,
            ProjectRole.VIEWER,
        )
        role_labels = {
            ProjectRole.OWNER: "Владелец",
            ProjectRole.MANAGER: "Руководитель",
            ProjectRole.CONTRIBUTOR: "Участник",
            ProjectRole.REVIEWER: "Проверяющий",
            ProjectRole.VIEWER: "Наблюдатель",
        }
        dialog = QDialog(self.shell)
        dialog.setWindowTitle(f"Участники и роли — {project.name}")
        dialog.resize(800, 420)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(len(principals), 1 + len(roles), dialog)
        table.setHorizontalHeaderLabels(
            ("Участник", *(role_labels[role] for role in roles))
        )
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)

        def change(principal, role: ProjectRole, enabled: bool, checkbox: QCheckBox) -> None:
            try:
                revision = self.service.project_role_revision(project.id, principal.id)
                if enabled:
                    self.service.assign_project_role(
                        principal=self.session.principal,
                        project=project,
                        target_principal_id=principal.id,
                        role=role,
                        expected_revision=revision,
                        idempotency_key=str(uuid4()),
                    )
                else:
                    self.service.revoke_project_role(
                        principal=self.session.principal,
                        project=project,
                        target_principal_id=principal.id,
                        role=role,
                        expected_revision=revision,
                        idempotency_key=str(uuid4()),
                    )
            except Exception as exc:
                checkbox.blockSignals(True)
                checkbox.setChecked(not enabled)
                checkbox.blockSignals(False)
                QMessageBox.warning(dialog, "Не удалось изменить роль", str(exc))

        for row, principal in enumerate(principals):
            label = principal.display_name
            if principal.email:
                label += f"\n{principal.email}"
            table.setItem(row, 0, QTableWidgetItem(label))
            assigned = self.service.project_roles(project.id, principal.id)
            for column, role in enumerate(roles, start=1):
                checkbox = QCheckBox(table)
                checkbox.setChecked(role in assigned)
                checkbox.toggled.connect(
                    lambda enabled, value=principal, selected=role, control=checkbox: change(
                        value,
                        selected,
                        enabled,
                        control,
                    )
                )
                table.setCellWidget(row, column, checkbox)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def refresh_projects(self) -> None:
        label_for = getattr(self.service, "project_storage_label", None)
        items = [
            ProjectListItem(
                project_id=str(project.id),
                name=project.name,
                width=project.width,
                height=project.height,
                storage_label=(
                    label_for(project.id)
                    if callable(label_for)
                    else "Локальный файл"
                ),
                status=project.state.value,
                archived=project.state.value == "archived",
                metadata={
                    "orientation": project.orientation.value,
                    "revision": project.revision,
                    "storage_profile": project.storage_profile,
                },
            )
            for project in self.service.list_projects(
                include_archived=self.catalog_page.show_archived_check.isChecked()
            )
        ]
        self.catalog_page.project_model.replace_items(items)

    def rename_project(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        from PyQt6.QtWidgets import QInputDialog, QMessageBox

        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен.")
            self.refresh_projects()
            return
        name, accepted = QInputDialog.getText(
            self.shell,
            "Переименовать проект",
            "Новое название:",
            text=project.name,
        )
        if not accepted or name.strip() == project.name:
            return
        is_remote = bool(getattr(self.service, "is_remote_project", lambda _id: False)(project.id))
        if not is_remote:
            answer = QMessageBox.question(
                self.shell,
                "Подтверждение переименования",
                "Будут согласованно переименованы папки исходных и производных данных. Продолжить?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            renamed = self.service.rename_project(
                principal=self.session.principal,
                project=project,
                name=name,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self.refresh_projects()
        if self._project_id == str(renamed.id) and self._workspace is not None:
            self._workspace.set_project_title(renamed.name)

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

    def _clear_configured_thumbnail_cache(self, project_id: str) -> None:
        store_uri = self.thumbnail_store_uri or os.environ.get("KRAKEN_THUMBNAIL_STORE_URI")
        if not store_uri:
            return
        store = ThumbnailStoreFactory().create(store_uri)
        namespace = StoreNamespace(
            plugin="matrix",
            project=str(project_id),
            generation="v1",
        )
        try:
            store.open(namespace, StorePolicy())
            store.clear_namespace()
        finally:
            store.close()
        root = getattr(store, "root", None)
        if root is None:
            return
        safe_root = Path(root).resolve()
        namespace_root = (safe_root / namespace.digest()).resolve()
        if (
            namespace_root.parent == safe_root
            and namespace_root.is_dir()
            and not namespace_root.is_symlink()
        ):
            shutil.rmtree(namespace_root)

    def delete_project(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        from PyQt6.QtCore import QSettings
        from PyQt6.QtWidgets import QDialog, QInputDialog, QLineEdit, QMessageBox

        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            self.refresh_projects()
            return
        binding = self.service.project_workspace(project.id)
        preserved_paths = (
            ()
            if binding is None
            else (binding.source_project_dir, binding.derived_project_dir)
        )
        dialog = QInputDialog(self.shell)
        dialog.setObjectName("deleteProjectConfirmationDialog")
        dialog.setWindowTitle("Удалить проект")
        dialog.setLabelText(
            "Проект будет удалён из Kraken вместе с его метаданными, историей "
            "и файлами кэша.\n\n"
            "Папки с изображениями слоёв, результатами нейросети и векторами "
            "останутся без изменений:\n"
            + ("\n".join(preserved_paths) if preserved_paths else "Папки проекта не зарегистрированы.")
            + f"\n\nДля подтверждения введите: {project.name}"
        )
        dialog.setInputMode(QInputDialog.InputMode.TextInput)
        dialog.setTextEchoMode(QLineEdit.EchoMode.Normal)
        dialog.setOkButtonText("Удалить из Kraken")
        dialog.setCancelButtonText("Отмена")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if dialog.textValue() != project.name:
            QMessageBox.warning(
                self.shell,
                "Удаление не подтверждено",
                "Введённое имя не совпадает с названием проекта.",
            )
            return

        if self._project_id == str(project.id) and self._workspace is not None:
            try:
                self._workspace.matrix_view.clear_thumbnail_cache()
                self._workspace.matrix_view.close()
            finally:
                self._workspace = None
                self._project_id = None
                if self._layer_dialog is not None:
                    self._layer_dialog.hide()
                    self._layer_dialog.deleteLater()
                    self._layer_dialog = None
        try:
            self._clear_configured_thumbnail_cache(str(project.id))
            self.service.delete_project(
                principal=self.session.principal,
                project=project,
                confirmation_name=dialog.textValue(),
            )
        except Exception as exc:
            self._error(str(exc))
            return

        settings = QSettings("Kraken", "KrakenHub")
        settings.remove(f"external-model/{project.id}")
        settings.remove(f"layer-manager/{project.id}")
        self.shell.show_page("projects")
        self.refresh_projects()
        QMessageBox.information(
            self.shell,
            "Проект удалён",
            "Проект и его кэш удалены из Kraken. Файлы в папках проекта сохранены.",
        )

    def create_project(self) -> None:
        from PyQt6.QtWidgets import (
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QVBoxLayout,
        )

        profiles = list(
            getattr(self.service, "list_storage_profiles", lambda: (self.service.profile,))()
        )
        dialog = QDialog(self.shell)
        dialog.setWindowTitle("Новый проект")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        name = QLineEdit()
        name.setObjectName("projectName")
        form.addRow("Имя проекта", name)
        profile_box = QComboBox(dialog)
        for profile in profiles:
            label = profile.name
            if profile.id == REMOTE_STORAGE_PROFILE.id:
                label = f"{profile.name} (общая БД)"
            profile_box.addItem(label, profile.id)
        form.addRow("Хранилище", profile_box)
        layout.addLayout(form)
        dimensions = GridDimensionsWidget(
            maximum_frames=max(
                (profile.capabilities.max_frames or 100_000 for profile in profiles),
                default=100_000,
            )
        )
        layout.addWidget(dimensions)
        storage = QLabel("")
        storage.setWordWrap(True)
        layout.addWidget(storage)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Создать проект")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        source_root = None
        derived_root = None

        def _update_storage_hint(_index: int = 0) -> None:
            nonlocal source_root, derived_root
            profile_id = str(profile_box.currentData() or "")
            if profile_id == REMOTE_STORAGE_PROFILE.id:
                source_root = None
                derived_root = None
                storage.setText(
                    "Проект будет создан на сервере PostgreSQL через kraken-server.\n"
                    "Требуется GitLab-токен (KRAKEN_GITLAB_TOKEN) и KRAKEN_SERVER_URL."
                )
                return
            roots = _configure_workspace_roots(self.shell, self.service)
            if roots is None:
                storage.setText("Выберите локальные каталоги исходных и производных данных.")
                source_root = None
                derived_root = None
                return
            source_root, derived_root = roots
            storage.setText(
                f"Исходные данные: {source_root}\n"
                f"Производные данные: {derived_root}"
            )

        profile_box.currentIndexChanged.connect(_update_storage_hint)
        _update_storage_hint()

        while dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                size = dimensions.validated_dimensions()
                if not name.text().strip():
                    raise ValueError("Введите имя проекта")
                profile_id = str(profile_box.currentData() or self.service.profile.id)
                if profile_id != REMOTE_STORAGE_PROFILE.id and (
                    source_root is None or derived_root is None
                ):
                    _update_storage_hint()
                    if source_root is None or derived_root is None:
                        raise ValueError("Выберите каталоги хранилища для локального проекта")
                if profile_id == REMOTE_STORAGE_PROFILE.id:
                    auth = load_remote_auth_from_env()
                    if auth is None:
                        raise ValueError(
                            "Для создания shared-проекта задайте KRAKEN_SERVER_URL и "
                            "KRAKEN_GITLAB_TOKEN (GitLab access token)."
                        )
                project = self.service.create_project(
                    principal=self.session.principal,
                    name=name.text(),
                    width=size.width,
                    height=size.height,
                    orientation=DomainOrientation(size.orientation.value),
                    idempotency_key=str(uuid4()),
                    layer_template=False,
                    source_root=source_root,
                    derived_root=derived_root,
                    storage_profile_id=profile_id,
                )
            except Exception as exc:
                QMessageBox.warning(dialog, "Не удалось создать проект", str(exc))
                continue
            self.refresh_projects()
            label = getattr(self.service, "project_storage_label", lambda _id: "Локальный файл")
            self.open_project(
                ProjectListItem(
                    str(project.id),
                    project.name,
                    project.width,
                    project.height,
                    label(project.id),
                    metadata={
                        "orientation": project.orientation.value,
                        "revision": project.revision,
                        "storage_profile": project.storage_profile,
                    },
                )
            )
            break

    def open_project(self, item: ProjectListItem) -> None:
        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            self.refresh_projects()
            return
        is_remote = bool(getattr(self.service, "is_remote_project", lambda _id: False)(project.id))
        workspace_binding = self.service.project_workspace(project.id)
        if workspace_binding is None and not is_remote:
            self._error(
                "Этот проект создан до введения двухдискового хранилища. "
                "Автоматическая миграция отключена; создайте новый проект."
            )
            return
        if workspace_binding is not None:
            unavailable = [
                path
                for path in (
                    workspace_binding.source_project_dir,
                    workspace_binding.derived_project_dir,
                )
                if not Path(path).is_dir()
            ]
            if unavailable:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self.shell,
                    "Хранилище недоступно",
                    "Метаданные и история доступны только для просмотра. "
                    "Файловые операции и плагины заблокированы.\n\n"
                    + "\n".join(unavailable),
                )
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
        if self._project_id is not None:
            unsubscribe = getattr(self.service, "unsubscribe_project", None)
            if callable(unsubscribe):
                unsubscribe(self._project_id)
        self._project_id = str(project.id)
        subscribe = getattr(self.service, "subscribe_project", None)
        if is_remote and callable(subscribe):
            subscribe(project.id)
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
        workspace.matrix_view.contextMenuRequested.connect(
            self._show_matrix_context_menu
        )
        permissions = self.service.project_permissions(
            project.id,
            self.session.principal if not is_remote else getattr(
                getattr(self.service, "remote", None),
                "auth",
                type("A", (), {"principal": self.session.principal})(),
            ).principal,
        )
        self.shell.review_return_action.setVisible(
            Permission.RETURN_REVIEW in permissions
        )
        layers = self.service.list_layers(project.id)
        if layers:
            first = workspace.layer_model.layer_by_id(str(layers[0].id))
            if first is not None:
                workspace.layer_tabs.setCurrentIndex(0)
                self._select_layer(workspace, project.id, first)

    def send_selection_for_review(self) -> None:
        from datetime import UTC

        from PyQt6.QtCore import QDateTime
        from PyQt6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDateTimeEdit,
            QDialog,
            QDialogButtonBox,
            QFileDialog,
            QFormLayout,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        workspace = self._workspace
        project_id = str(self._project_id or "")
        layer_id = str(getattr(workspace, "_selected_layer_id", "") if workspace else "")
        if workspace is None or not project_id or not layer_id:
            self._error("Сначала откройте проект и выберите слой.")
            return
        coordinates = workspace.matrix_view.selected_coordinates(
            maximum=self.service.profile.capabilities.max_frames or 100_000
        )
        if not coordinates:
            self._error("Выберите хотя бы один кадр.")
            return
        image_id = str(workspace.image_representation_combo.currentData() or "")
        vector_id = str(workspace.vector_representation_combo.currentData() or "")
        if not image_id or not vector_id:
            self._error("Выберите связанные репрезентации изображения и CIF.")
            return
        performers = self.service.list_performers()
        if not performers:
            self._error("Сначала добавьте исполнителя в разделе «Исполнители».")
            return

        dialog = QDialog(self.shell)
        dialog.setWindowTitle("Отправить на проверку")
        root = QVBoxLayout(dialog)
        form = QFormLayout()
        assignee = QComboBox(dialog)
        for performer in performers:
            assignee.addItem(performer.name, str(performer.id))
        instructions = QLineEdit(dialog)
        instructions.setPlaceholderText("Что необходимо проверить")
        due_enabled = QCheckBox("Установить срок", dialog)
        due = QDateTimeEdit(QDateTime.currentDateTime().addDays(7), dialog)
        due.setCalendarPopup(True)
        due.setDisplayFormat("dd.MM.yyyy HH:mm")
        due.setEnabled(False)
        due_enabled.toggled.connect(due.setEnabled)
        directory_row = QWidget(dialog)
        directory_layout = QFormLayout(directory_row)
        directory_layout.setContentsMargins(0, 0, 0, 0)
        destination_root = QLineEdit(dialog)
        destination_root.setReadOnly(True)
        choose = QPushButton("Выбрать…", dialog)
        choose.clicked.connect(
            lambda: destination_root.setText(
                QFileDialog.getExistingDirectory(
                    dialog,
                    "Каталог для пакета проверки",
                    destination_root.text(),
                )
                or destination_root.text()
            )
        )
        directory_layout.addRow(destination_root, choose)
        form.addRow("Исполнитель", assignee)
        form.addRow("Инструкции", instructions)
        form.addRow(due_enabled, due)
        form.addRow("Каталог назначения", directory_row)
        root.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dialog,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Выгрузить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        root.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if not destination_root.text():
            QMessageBox.warning(dialog, "Каталог не выбран", "Выберите каталог назначения.")
            return
        project = self.service.get_project(project_id)
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd-HHmmss")
        safe_project = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in (project.name if project is not None else "project")
        ).strip("_") or "project"
        destination = Path(destination_root.text()) / f"Проверка_{safe_project}_{timestamp}"
        try:
            deadline = (
                due.dateTime().toPyDateTime().astimezone(UTC)
                if due_enabled.isChecked()
                else None
            )
            batch = self.service.create_review_batch(
                principal=self.session.principal,
                project_id=project_id,
                layer_id=layer_id,
                image_representation_id=image_id,
                vector_representation_id=vector_id,
                coordinates=coordinates,
                assignee_id=str(assignee.currentData()),
                instructions=instructions.text(),
                due_at=deadline,
                idempotency_key=str(uuid4()),
            )
            issued = self.service.export_review_batch(
                principal=self.session.principal,
                batch=batch,
                destination=destination,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            QMessageBox.warning(dialog, "Не удалось отправить на проверку", str(exc))
            return
        QMessageBox.information(
            self.shell,
            "Пакет проверки создан",
            f"Кадров: {len(issued.items)}\nКаталог: {destination}",
        )

    def load_review_return(self) -> None:
        from PyQt6.QtWidgets import (
            QFileDialog,
            QMessageBox,
            QTableWidget,
            QTableWidgetItem,
        )

        source = QFileDialog.getExistingDirectory(
            self.shell,
            "Выберите папку с проверенными файлами",
        )
        if not source:
            return
        try:
            batch, plan = self.service.review_return_preflight(
                principal=self.session.principal,
                source=source,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            QMessageBox.warning(self.shell, "Импорт заблокирован", str(exc))
            return

        labels = {
            "changed": "Изменён",
            "unchanged": "Без изменений",
            "missing": "Отсутствует",
            "extra": "Лишний",
            "duplicate": "Дубликат",
            "invalid": "Повреждён",
            "stale_base_conflict": "Конфликт исходной версии",
        }
        preview = QMessageBox(self.shell)
        preview.setWindowTitle("Предварительная проверка")
        preview.setIcon(QMessageBox.Icon.Information)
        counts: dict[str, int] = {}
        for item in plan.report.items:
            key = item.status.value
            counts[key] = counts.get(key, 0) + 1
        preview.setText(
            "\n".join(
                f"{labels.get(key, key)}: {value}"
                for key, value in counts.items()
            )
            or "Файлы не найдены."
        )
        details = QTableWidget(len(plan.report.items), 2, preview)
        details.setHorizontalHeaderLabels(("Файл", "Результат"))
        for row, item in enumerate(plan.report.items):
            details.setItem(row, 0, QTableWidgetItem(item.relative_path))
            details.setItem(row, 1, QTableWidgetItem(labels.get(item.status.value, item.status.value)))
        details.resizeColumnsToContents()
        preview.layout().addWidget(
            details,
            preview.layout().rowCount(),
            0,
            1,
            preview.layout().columnCount(),
        )
        import_button = preview.addButton("Загрузить изменённые файлы", QMessageBox.ButtonRole.AcceptRole)
        preview.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        import_button.setEnabled(plan.report.can_commit)
        preview.exec()
        if preview.clickedButton() is not import_button:
            return
        try:
            result = self.service.commit_review_return(
                principal=self.session.principal,
                batch=batch,
                source=source,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            QMessageBox.warning(self.shell, "Не удалось загрузить результат", str(exc))
            return
        if not result.candidate_versions:
            QMessageBox.information(
                self.shell,
                "Проверка загружена",
                "Изменённых CIF-файлов нет; результат сохранён.",
            )
            self._refresh_matrix()
            return
        decision = QMessageBox(self.shell)
        decision.setWindowTitle("Изменённые файлы загружены")
        decision.setText(
            f"Загружено изменённых файлов: {len(result.candidate_versions)}.\n"
            "Использовать их в проекте или вернуть на доработку?"
        )
        accept_button = decision.addButton("Принять", QMessageBox.ButtonRole.AcceptRole)
        changes_button = decision.addButton("На доработку", QMessageBox.ButtonRole.DestructiveRole)
        decision.addButton("Позже", QMessageBox.ButtonRole.RejectRole)
        decision.exec()
        try:
            if decision.clickedButton() is accept_button:
                answer = QMessageBox.question(
                    self.shell,
                    "Подтверждение",
                    "Использовать загруженные файлы как текущие?",
                )
                if answer == QMessageBox.StandardButton.Yes:
                    self.service.accept_review(
                        principal=self.session.principal,
                        batch=result.batch,
                        candidate_version_ids=tuple(
                            version.id for version in result.candidate_versions
                        ),
                        idempotency_key=str(uuid4()),
                    )
            elif decision.clickedButton() is changes_button:
                self._request_review_changes(result.batch)
        except Exception as exc:
            QMessageBox.warning(self.shell, "Не удалось изменить состояние проверки", str(exc))
        self._refresh_matrix()

    def _request_review_changes(self, batch) -> None:
        from PyQt6.QtWidgets import QInputDialog

        reason, accepted = QInputDialog.getMultiLineText(
            self.shell,
            "На доработку",
            "Причина (обязательно):",
        )
        if not accepted:
            return
        if not reason.strip():
            raise ValueError("Укажите причину возврата на доработку.")
        self.service.request_review_changes(
            principal=self.session.principal,
            batch=batch,
            reason=reason,
            idempotency_key=str(uuid4()),
        )

    def _refresh_matrix(self) -> None:
        workspace = self._workspace
        project_id = str(self._project_id or "")
        layer_id = str(getattr(workspace, "_selected_layer_id", "") if workspace else "")
        if workspace is None or not project_id or not layer_id:
            return
        self._load_representations(workspace, project_id, layer_id)

    def _active_frame_files(
        self,
        x: int,
        y: int,
    ) -> tuple[tuple[str, ArtifactSeries, ArtifactVersion], ...]:
        workspace = self._workspace
        project_id = str(self._project_id or "")
        layer_id = str(
            getattr(workspace, "_selected_layer_id", "") if workspace else ""
        )
        project = self.service.get_project(project_id) if project_id else None
        if workspace is None or project is None or not layer_id:
            return ()
        frame_id = str(project.frame_id_at(int(x), int(y)))
        representation_ids = tuple(
            identifier
            for identifier in (
                str(workspace.image_representation_combo.currentData() or ""),
                str(workspace.vector_representation_combo.currentData() or ""),
            )
            if identifier
        )
        representations = {
            str(representation.id): representation.name
            for representation in self.service.list_representations(
                project.id,
                layer_id,
            )
            if str(representation.id) in representation_ids
        }
        files: list[tuple[str, ArtifactSeries, ArtifactVersion]] = []
        for representation_id in representation_ids:
            series = next(
                (
                    candidate
                    for candidate in self.service.list_artifact_series(
                        project.id,
                        layer_id=layer_id,
                        representation_id=representation_id,
                    )
                    if str(candidate.frame_id or "") == frame_id
                    and not candidate.archived
                ),
                None,
            )
            if series is None:
                continue
            version = self.service.active_artifact_version(project.id, series.id)
            if version is not None:
                files.append(
                    (
                        representations.get(representation_id, series.name),
                        series,
                        version,
                    )
                )
        return tuple(files)

    def _show_frame_file_properties(
        self,
        label: str,
        series: ArtifactSeries,
        active: ArtifactVersion,
    ) -> None:
        project_id = series.project_id
        principals = {
            str(principal.id): principal.display_name
            for principal in self.service.list_principals(include_inactive=True)
        }
        versions = self.service.artifact_versions(project_id, series.id)
        history = tuple(
            event
            for event in self.service.history(project_id)
            if str(series.id) in str(_event_payload(event))
        )

        def export_version(row: Mapping[str, object]) -> None:
            from PyQt6.QtWidgets import QFileDialog, QMessageBox

            version = row.get("_version")
            if not isinstance(version, ArtifactVersion):
                return
            destination, _selected_filter = QFileDialog.getSaveFileName(
                self.shell,
                "Выгрузить файл",
                version.filename,
            )
            if not destination:
                return
            try:
                self.service.export_artifact_version(
                    project_id,
                    version,
                    destination,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось выгрузить файл",
                    str(exc),
                )

        version_rows = tuple(
            {
                "_version": version,
                "filename": version.filename,
                "size": f"{version.size_bytes:,}".replace(",", " ") + " байт",
                "sha256": version.sha256,
                "author": principals.get(
                    str(version.author_principal_id),
                    str(version.author_principal_id),
                ),
                "created_at": version.created_at.isoformat(),
                "parent": (
                    "—"
                    if version.parent_version_id is None
                    else str(version.parent_version_id)
                ),
                "tool": " ".join(
                    value
                    for value in (version.tool_name, version.tool_version)
                    if value
                )
                or "—",
                "provenance": dict(version.parameters),
                "active": version.id == active.id,
            }
            for version in versions
        )
        self._open_properties(
            ObjectPropertiesSnapshot(
                title=f"{label}: {active.filename}",
                object_kind="artifact-version",
                properties=(
                    ("Логическое название", series.name),
                    ("Идентификатор серии", str(series.id)),
                    ("Область", series.scope.value),
                    ("Имя файла", active.filename),
                    ("MIME-тип", active.media_type),
                    ("Размер", active.size_bytes),
                    ("SHA-256", active.sha256),
                    (
                        "Хранение",
                        "В базе" if active.blob is not None else "Внешняя ссылка",
                    ),
                    (
                        "Внешний путь",
                        None if active.external is None else active.external.uri,
                    ),
                    ("Автор", principals.get(str(active.author_principal_id))),
                    ("Создан", active.created_at.isoformat()),
                    (
                        "Родительская версия",
                        None
                        if active.parent_version_id is None
                        else str(active.parent_version_id),
                    ),
                    ("Инструмент", active.tool_name),
                    ("Версия инструмента", active.tool_version),
                    ("Provenance", dict(active.parameters)),
                ),
                history=_history_entries(history),
                versions=version_rows,
                version_actions=(("Выгрузить", export_version),),
            )
        )

    def _export_selected_frame_files(self, context) -> None:
        from datetime import UTC, datetime

        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        parent = QFileDialog.getExistingDirectory(
            self.shell,
            "Каталог для выгрузки файлов",
        )
        if not parent:
            return
        coordinates = tuple(context.selection.coordinates(maximum=10_000))
        destination = Path(parent) / (
            "Кадры_"
            + datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S-")
            + uuid4().hex[:6]
        )
        exported = 0
        try:
            destination.mkdir(parents=False, exist_ok=False)
            for x, y in coordinates:
                files = self._active_frame_files(x, y)
                if not files:
                    continue
                frame_directory = destination / f"{x}_{y}"
                frame_directory.mkdir(exist_ok=False)
                used_names: set[str] = set()
                for label, _series, version in files:
                    filename = version.filename
                    if filename.casefold() in used_names:
                        prefix = "".join(
                            character
                            if character.isalnum() or character in "-_"
                            else "_"
                            for character in label
                        ).strip("_") or "file"
                        filename = f"{prefix}_{filename}"
                    used_names.add(filename.casefold())
                    self.service.export_artifact_version(
                        self._project_id,
                        version,
                        frame_directory / filename,
                    )
                    exported += 1
            if exported == 0:
                destination.rmdir()
                QMessageBox.information(
                    self.shell,
                    "Нет файлов",
                    "У выбранных кадров нет активных файлов для выгрузки.",
                )
                return
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(
                self.shell,
                "Не удалось выгрузить файлы",
                str(exc),
            )
            return
        QMessageBox.information(
            self.shell,
            "Файлы выгружены",
            f"Файлов: {exported}\nКаталог: {destination}",
        )

    def _show_matrix_context_menu(self, context, global_position) -> None:
        from PyQt6.QtWidgets import QMenu

        menu = QMenu(self.shell)
        selected_count = len(tuple(context.selection.coordinates(maximum=10_000)))
        export_action = menu.addAction(
            "Выгрузить файлы кадра…"
            if selected_count == 1
            else f"Выгрузить файлы выбранных кадров ({selected_count})…"
        )
        review_action = None
        if (
            selected_count
            and getattr(self, "_project_id", None) is not None
            and self._has_permission(Permission.ASSIGN_WORK)
        ):
            review_action = menu.addAction("Отправить выбранное на проверку…")
        vector_properties = None
        vector_file = None
        frame_properties = None
        if selected_count == 1:
            workspace = getattr(self, "_workspace", None)
            vector_representation_id = (
                str(workspace.vector_representation_combo.currentData() or "")
                if workspace is not None
                else ""
            )
            vector_file = next(
                (
                    item
                    for item in self._active_frame_files(context.x, context.y)
                    if str(item[1].representation_id or "")
                    == vector_representation_id
                ),
                None,
            )
            if vector_file is not None:
                vector_properties = menu.addAction("Свойства вектора…")
            frame_properties = menu.addAction("Свойства кадра…")
        selected = menu.exec(global_position)
        if selected is export_action:
            self._export_selected_frame_files(context)
        elif review_action is not None and selected is review_action:
            self.send_selection_for_review()
        elif frame_properties is not None and selected is frame_properties:
            workspace = self._workspace
            if workspace is not None:
                workspace.matrix_view.set_selection(
                    type(context.selection).single(context.x, context.y)
                )
            self.show_selected_frame()
        elif vector_properties is not None and selected is vector_properties:
            assert vector_file is not None
            self._show_frame_file_properties(*vector_file)

    def show_selected_frame(self) -> None:
        workspace = self._workspace
        project_id = str(self._project_id or "")
        if workspace is None or not project_id:
            return
        coordinates = workspace.matrix_view.selected_coordinates(maximum=2)
        if len(coordinates) != 1:
            self._error("Для просмотра карточки выберите ровно один кадр.")
            return
        x, y = coordinates[0]
        cell = workspace.matrix_view.cell_data(x, y)
        payload = cell.payload if isinstance(cell.payload, Mapping) else {}
        frame_id = str(payload.get("frame_id") or payload.get("key") or "")
        project = self.service.get_project(project_id)
        permissions = self.service.project_permissions(
            project_id,
            self.session.principal,
        )
        layer_id = str(getattr(workspace, "_selected_layer_id", ""))
        layer = next(
            (item for item in self.service.list_layers(project_id) if str(item.id) == layer_id),
            None,
        )
        representations = self.service.list_representations(project_id, layer_id)
        image_id = str(workspace.image_representation_combo.currentData() or "")
        vector_id = str(workspace.vector_representation_combo.currentData() or "")
        image = next((item for item in representations if str(item.id) == image_id), None)
        vector = next((item for item in representations if str(item.id) == vector_id), None)
        events = tuple(
            event
            for event in self.service.history(project_id)
            if frame_id and frame_id in str(_event_payload(event))
        )
        principals = {
            str(principal.id): principal.display_name
            for principal in self.service.list_principals(include_inactive=True)
        }
        series_values = tuple(
            series
            for series in self.service.list_artifact_series(
                project_id,
                layer_id=layer_id,
                include_archived=True,
            )
            if str(series.frame_id or "") == frame_id
        )
        notes = self.service.list_notes(
            project_id,
            layer_id=layer_id,
            frame_id=frame_id,
        )
        files: list[Mapping[str, object]] = []
        versions: list[Mapping[str, object]] = []
        for series in series_values:
            active = self.service.active_artifact_version(project_id, series.id)
            files.append(
                {
                    "_series": series,
                    "name": series.name,
                    "scope": series.scope.value,
                    "active_version": None if active is None else str(active.id),
                    "archived": series.archived,
                }
            )
            for version in self.service.artifact_versions(project_id, series.id):
                if version.size_bytes < 1024:
                    size = f"{version.size_bytes} байт"
                elif version.size_bytes < 1024 * 1024:
                    size = f"{version.size_bytes / 1024:.1f} КБ"
                else:
                    size = f"{version.size_bytes / (1024 * 1024):.1f} МБ"
                versions.append(
                    {
                        "_version": version,
                        "filename": version.filename,
                        "size": size,
                        "sha256": version.sha256,
                        "author": principals.get(
                            str(version.author_principal_id),
                            str(version.author_principal_id),
                        ),
                        "created_at": version.created_at.isoformat(),
                        "parent": (
                            None
                            if version.parent_version_id is None
                            else str(version.parent_version_id)
                        ),
                        "tool": " ".join(
                            value
                            for value in (version.tool_name, version.tool_version)
                            if value
                        ),
                        "provenance": dict(version.parameters),
                        "active": active is not None and active.id == version.id,
                    }
                )

        def add_note() -> None:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox

            body, accepted = QInputDialog.getMultiLineText(
                self.shell,
                "Новая заметка",
                "Текст:",
            )
            if not accepted:
                return
            try:
                self.service.create_note(
                    principal=self.session.principal,
                    project_id=project_id,
                    layer_id=layer_id,
                    frame_id=frame_id,
                    body=body,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось создать заметку", str(exc))
                return
            QMessageBox.information(
                self.shell,
                "Заметка создана",
                "Заметка сохранена новой неизменяемой ревизией. Переоткройте карточку.",
            )

        def revise_latest_note() -> None:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox

            if not notes:
                QMessageBox.information(self.shell, "Нет заметок", "Сначала создайте заметку.")
                return
            current = max(notes, key=lambda value: (value.recorded_at, value.revision))
            body, accepted = QInputDialog.getMultiLineText(
                self.shell,
                "Изменить заметку",
                "Новая ревизия:",
                text=current.body,
            )
            if not accepted:
                return
            try:
                self.service.revise_note(
                    principal=self.session.principal,
                    note=current,
                    body=body,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось изменить заметку", str(exc))
                return
            QMessageBox.information(
                self.shell,
                "Ревизия сохранена",
                "Предыдущая ревизия осталась в истории.",
            )

        def add_version(row: Mapping[str, object], *, external: bool) -> None:
            from PyQt6.QtWidgets import QFileDialog, QMessageBox

            series = row.get("_series")
            if series is None:
                return
            source, _selected_filter = QFileDialog.getOpenFileName(
                self.shell,
                "Выберите новую версию файла",
            )
            if not source:
                return
            try:
                method = (
                    self.service.add_external_artifact_version
                    if external
                    else self.service.add_managed_artifact_version
                )
                method(
                    principal=self.session.principal,
                    project_id=project_id,
                    series_id=series.id,
                    source=source,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось добавить версию", str(exc))
                return
            QMessageBox.information(
                self.shell,
                "Версия добавлена",
                "Новая неизменяемая версия сохранена. Переоткройте карточку.",
            )

        def rename_series(row: Mapping[str, object]) -> None:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox

            series = row.get("_series")
            if series is None:
                return
            name, accepted = QInputDialog.getText(
                self.shell,
                "Переименовать файл",
                "Логическое название:",
                text=series.name,
            )
            if not accepted:
                return
            try:
                self.service.rename_artifact_series(
                    principal=self.session.principal,
                    series=series,
                    name=name,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось переименовать файл", str(exc))

        def archive_series(row: Mapping[str, object]) -> None:
            from PyQt6.QtWidgets import QMessageBox

            series = row.get("_series")
            if series is None:
                return
            if QMessageBox.question(
                self.shell,
                "Архивировать файл",
                f"Архивировать логическую серию «{series.name}»?",
            ) != QMessageBox.StandardButton.Yes:
                return
            try:
                self.service.archive_artifact_series(
                    principal=self.session.principal,
                    series=series,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось архивировать файл", str(exc))

        def activate_version(row: Mapping[str, object]) -> None:
            from PyQt6.QtWidgets import QMessageBox

            version = row.get("_version")
            if version is None or bool(row.get("active")):
                return
            if QMessageBox.question(
                self.shell,
                "Активировать прежнюю версию",
                "Сделать выбранную неизменяемую версию активной? История не будет переписана.",
            ) != QMessageBox.StandardButton.Yes:
                return
            try:
                self.service.activate_artifact_version(
                    principal=self.session.principal,
                    project_id=project_id,
                    series_id=version.series_id,
                    version_id=version.id,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось активировать версию", str(exc))

        def export_or_open_version(row: Mapping[str, object]) -> None:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtWidgets import QFileDialog, QMessageBox

            version = row.get("_version")
            if version is None:
                return
            if version.external is not None:
                QDesktopServices.openUrl(QUrl(version.external.uri))
                return
            destination, _selected_filter = QFileDialog.getSaveFileName(
                self.shell,
                "Экспортировать файл",
                version.filename,
            )
            if not destination:
                return
            try:
                self.service.export_managed_artifact(
                    project_id,
                    version,
                    destination,
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось экспортировать файл", str(exc))

        def check_external(row: Mapping[str, object]) -> None:
            from PyQt6.QtWidgets import QMessageBox

            version = row.get("_version")
            if version is None or version.external is None:
                QMessageBox.information(
                    self.shell,
                    "Управляемый файл",
                    "Эта версия хранится в базе и не может измениться снаружи.",
                )
                return
            try:
                changed = self.service.external_artifact_changed(version)
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось проверить файл", str(exc))
                return
            QMessageBox.information(
                self.shell,
                "Проверка внешнего файла",
                "Файл изменён или недоступен." if changed else "Файл не изменился.",
            )
        properties = (
            ("Координаты", f"X={x}, Y={y}"),
            ("Номер кадра", (y - 1) * (project.width if project is not None else 1) + x),
            ("Идентификатор кадра", frame_id),
            ("Проект", None if project is None else project.name),
            ("Слой", None if layer is None else layer.name),
            ("Статус", cell.status),
            ("Исполнитель", cell.performer_initials or "—"),
            ("Качество", payload.get("quality")),
            ("Проверка", payload.get("review_status")),
            ("Изображение", None if image is None else image.name),
            ("Версия изображения", payload.get("asset_revision")),
            ("SHA-256 изображения", payload.get("asset_sha256")),
            ("CIF", None if vector is None else vector.name),
            ("Активная версия CIF", payload.get("artifact_version_id")),
            ("Изменён", payload.get("modified_at")),
            (
                "Отсутствующие данные",
                "CIF или изображение отсутствует" if payload.get("missing") else "Нет",
            ),
        )

        def temporal(entry: ObjectHistoryEntry):
            from datetime import datetime

            moment = datetime.fromisoformat(entry.recorded_at)
            historical_project = self.service.get_project(
                project_id,
                as_of=moment,
            )
            if historical_project is None or layer is None:
                return None
            historical_cells = []
            for representation in (image, vector):
                if representation is None:
                    continue
                historical_cells.extend(
                    candidate
                    for candidate in self.service.frame_cells(
                        project_id,
                        layer.id,
                        representation.id,
                        as_of=moment,
                    )
                    if str(candidate.frame_id) == frame_id
                )
            historical = historical_cells[-1] if historical_cells else None
            return ObjectPropertiesSnapshot(
                title=f"Кадр ({x}, {y}) — на {entry.recorded_at}",
                object_kind="frame-temporal",
                properties=(
                    ("Координаты", f"X={x}, Y={y}"),
                    ("Идентификатор кадра", frame_id),
                    ("Проект", historical_project.name),
                    ("Слой", layer.name),
                    (
                        "Статус",
                        "Нет данных" if historical is None else historical.status,
                    ),
                    (
                        "SHA-256",
                        None if historical is None else historical.sha256,
                    ),
                    (
                        "Версия",
                        None
                        if historical is None
                        else historical.artifact_version_id,
                    ),
                ),
                history=tuple(
                    history
                    for history in _history_entries(events)
                    if history.recorded_at <= entry.recorded_at
                ),
            )

        self._open_properties(
            ObjectPropertiesSnapshot(
                title=f"Кадр ({x}, {y})",
                object_kind="frame",
                properties=properties,
                history=_history_entries(events),
                temporal_loader=temporal,
                notes=tuple(
                    {
                        "revision": note.revision,
                        "body": note.body,
                        "author": principals.get(
                            str(note.author_principal_id),
                            str(note.author_principal_id),
                        ),
                        "recorded_at": note.recorded_at.isoformat(),
                    }
                    for note in notes
                ),
                files=tuple(files),
                versions=tuple(versions),
                actions=(
                    (
                        ("Добавить заметку", add_note),
                        ("Изменить последнюю заметку", revise_latest_note),
                    )
                    if Permission.ADD_NOTE in permissions
                    else ()
                ),
                file_actions=(
                    (
                        (
                            "Добавить версию в базу",
                            lambda row: add_version(row, external=False),
                        ),
                        (
                            "Добавить внешнюю версию",
                            lambda row: add_version(row, external=True),
                        ),
                        ("Переименовать", rename_series),
                        ("Архивировать", archive_series),
                    )
                    if Permission.IMPORT_ARTIFACT in permissions
                    else ()
                ),
                version_actions=(
                    (
                        (("Активировать", activate_version),)
                        if Permission.IMPORT_ARTIFACT in permissions
                        else ()
                    )
                    + (
                        ("Экспортировать / открыть", export_or_open_version),
                        ("Проверить внешний файл", check_external),
                    )
                ),
            )
        )

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

    def _show_created_layer(self, workspace, project_id, layer) -> None:
        """Reload and select a layer so its registered image source is visible."""

        latest = self.service.get_project(project_id)
        if latest is None:
            return
        self._load_layers(workspace, latest)
        item = workspace.layer_model.layer_by_id(str(layer.id))
        if item is None:
            return
        for index in range(workspace.layer_tabs.count()):
            if str(workspace.layer_tabs.tabData(index)) == str(layer.id):
                if workspace.layer_tabs.currentIndex() != index:
                    workspace.layer_tabs.setCurrentIndex(index)
                break
        self._select_layer(workspace, project_id, item)

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
            dialog.nodePropertiesRequested.connect(self._show_node_properties)
            dialog.layerPropertiesRequested.connect(self._show_layer_properties)
            dialog.addLayerRequested.connect(
                lambda: self._add_layer(workspace, project_id)
            )
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
        workspace_runs = self.service.list_derived_runs(project_id, layer_id)
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
                    details={
                        "источник": (
                            "Файлы сохранены в проекте"
                            if source.source == "managed-import" or not source.source
                            else source.source
                        ),
                        "создан": source.created_at.isoformat(),
                    },
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
            if source is sources[0]:
                for run in workspace_runs:
                    job_id = f"workspace-job:{run.run_id}"
                    lane_nodes.append(
                        PipelineNode(
                            job_id,
                            f"{run.plugin_id} · {run.operation}",
                            "job",
                            run.state.value,
                            state=run.state.value,
                            details={
                                "run_id": run.run_id,
                                "path": run.path,
                                "operation": run.operation,
                                "plugin": run.plugin_id,
                                "создан": run.created_at,
                            },
                        )
                    )
                    lane_edges.append((str(source.id), job_id))
                    if run.state.value != "succeeded":
                        continue
                    output_id = f"workspace-output:{run.run_id}"
                    output_kind = {
                        DerivedRunKind.DATASET: "dataset",
                        DerivedRunKind.RESULT: "model",
                        DerivedRunKind.VECTOR: "vector",
                    }[run.kind]
                    output_title = {
                        DerivedRunKind.DATASET: "Выборка Contour",
                        DerivedRunKind.RESULT: "Результат NeuralImage",
                        DerivedRunKind.VECTOR: "Векторы Contour",
                    }[run.kind]
                    lane_nodes.append(
                        PipelineNode(
                            output_id,
                            output_title,
                            output_kind,
                            run.state.value,
                            state=run.state.value,
                            details={
                                "run_id": run.run_id,
                                "path": run.path,
                                "operation": run.operation,
                                "plugin": run.plugin_id,
                            },
                        )
                    )
                    lane_edges.append((job_id, output_id))
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

    def _open_properties(self, snapshot: ObjectPropertiesSnapshot) -> None:
        ObjectPropertiesDialog(
            snapshot,
            self._layer_dialog or self.shell,
        ).exec()

    def _scope_content(
        self,
        project_id,
        *,
        layer_id=None,
    ) -> dict[str, tuple]:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

        permission_reader = getattr(self.service, "project_permissions", None)
        permissions = (
            frozenset(Permission)
            if not callable(permission_reader)
            else permission_reader(project_id, self.session.principal)
        )
        list_notes = getattr(self.service, "list_notes", None)
        notes = (
            ()
            if not callable(list_notes)
            else list_notes(project_id, layer_id=layer_id)
        )
        list_principals = getattr(self.service, "list_principals", None)
        principals = {
            str(principal.id): principal.display_name
            for principal in (
                ()
                if not callable(list_principals)
                else list_principals(include_inactive=True)
            )
        }
        wanted_scopes = (
            {
                ArtifactScope.PROJECT_ATTACHMENT,
                ArtifactScope.PROJECT_EXTERNAL_LINK,
            }
            if layer_id is None
            else {
                ArtifactScope.LAYER_ATTACHMENT,
                ArtifactScope.LAYER_EXTERNAL_LINK,
            }
        )
        list_series = getattr(self.service, "list_artifact_series", None)
        available_series = (
            ()
            if not callable(list_series)
            else list_series(
                project_id,
                layer_id=layer_id,
                include_archived=True,
            )
        )
        series_items = tuple(
            series
            for series in available_series
            if series.scope in wanted_scopes
            and (
                (layer_id is None and series.layer_id is None)
                or str(series.layer_id) == str(layer_id)
            )
        )
        files: list[Mapping[str, object]] = []
        versions: list[Mapping[str, object]] = []
        for series in series_items:
            active = self.service.active_artifact_version(project_id, series.id)
            files.append(
                {
                    "name": series.name,
                    "scope": (
                        "Вложение"
                        if series.scope
                        in {
                            ArtifactScope.PROJECT_ATTACHMENT,
                            ArtifactScope.LAYER_ATTACHMENT,
                        }
                        else "Внешняя ссылка"
                    ),
                    "active_version": (
                        "—" if active is None else active.filename
                    ),
                    "archived": series.archived,
                    "_series": series,
                }
            )
            for version in self.service.artifact_versions(project_id, series.id):
                versions.append(
                    {
                        "filename": version.filename,
                        "size": f"{version.size_bytes:,}".replace(",", " ") + " байт",
                        "sha256": version.sha256,
                        "author": principals.get(
                            str(version.author_principal_id),
                            str(version.author_principal_id),
                        ),
                        "created_at": version.created_at.isoformat(),
                        "parent": (
                            "—"
                            if version.parent_version_id is None
                            else str(version.parent_version_id)
                        ),
                        "tool": " ".join(
                            value
                            for value in (version.tool_name, version.tool_version)
                            if value
                        )
                        or "—",
                        "provenance": dict(version.parameters),
                        "active": active is not None and active.id == version.id,
                        "_version": version,
                    }
                )

        def add_note() -> None:
            body, accepted = QInputDialog.getMultiLineText(
                self.shell,
                "Новая заметка",
                "Текст:",
            )
            if not accepted or not body.strip():
                return
            try:
                self.service.create_note(
                    principal=self.session.principal,
                    project_id=project_id,
                    layer_id=layer_id,
                    body=body,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось добавить заметку",
                    str(exc),
                )

        def revise_note() -> None:
            if not notes:
                QMessageBox.information(
                    self.shell,
                    "Нет заметок",
                    "Сначала добавьте заметку.",
                )
                return
            note = max(notes, key=lambda item: (item.recorded_at, item.revision))
            body, accepted = QInputDialog.getMultiLineText(
                self.shell,
                "Новая ревизия заметки",
                "Текст:",
                text=note.body,
            )
            if not accepted or not body.strip() or body.strip() == note.body:
                return
            try:
                self.service.revise_note(
                    principal=self.session.principal,
                    note=note,
                    body=body,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось изменить заметку",
                    str(exc),
                )

        def create_file_series(*, external: bool) -> None:
            source, _selected_filter = QFileDialog.getOpenFileName(
                self.shell,
                "Выберите файл",
            )
            if not source:
                return
            path = Path(source)
            scope = (
                ArtifactScope.PROJECT_EXTERNAL_LINK
                if layer_id is None and external
                else ArtifactScope.PROJECT_ATTACHMENT
                if layer_id is None
                else ArtifactScope.LAYER_EXTERNAL_LINK
                if external
                else ArtifactScope.LAYER_ATTACHMENT
            )
            try:
                series = self.service.create_artifact_series(
                    principal=self.session.principal,
                    project_id=project_id,
                    layer_id=layer_id,
                    scope=scope,
                    name=path.name,
                    idempotency_key=str(uuid4()),
                )
                add = (
                    self.service.add_external_artifact_version
                    if external
                    else self.service.add_managed_artifact_version
                )
                add(
                    principal=self.session.principal,
                    project_id=project_id,
                    series_id=series.id,
                    source=path,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось добавить файл",
                    str(exc),
                )

        def add_version(row: Mapping[str, object], *, external: bool) -> None:
            series = row.get("_series")
            if series is None:
                return
            source, _selected_filter = QFileDialog.getOpenFileName(
                self.shell,
                "Выберите новую версию",
            )
            if not source:
                return
            active = self.service.active_artifact_version(project_id, series.id)
            try:
                add = (
                    self.service.add_external_artifact_version
                    if external
                    else self.service.add_managed_artifact_version
                )
                add(
                    principal=self.session.principal,
                    project_id=project_id,
                    series_id=series.id,
                    source=source,
                    parent_version_id=None if active is None else active.id,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось добавить версию",
                    str(exc),
                )

        def rename_series(row: Mapping[str, object]) -> None:
            series = row.get("_series")
            if series is None:
                return
            name, accepted = QInputDialog.getText(
                self.shell,
                "Переименовать файл",
                "Название:",
                text=series.name,
            )
            if not accepted or not name.strip() or name.strip() == series.name:
                return
            try:
                self.service.rename_artifact_series(
                    principal=self.session.principal,
                    series=series,
                    name=name,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось переименовать файл",
                    str(exc),
                )

        def archive_series(row: Mapping[str, object]) -> None:
            series = row.get("_series")
            if series is None or series.archived:
                return
            if QMessageBox.question(
                self.shell,
                "Архивировать файл",
                f"Архивировать логический файл «{series.name}»?",
            ) != QMessageBox.StandardButton.Yes:
                return
            try:
                self.service.archive_artifact_series(
                    principal=self.session.principal,
                    series=series,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось архивировать файл",
                    str(exc),
                )

        def activate_version(row: Mapping[str, object]) -> None:
            version = row.get("_version")
            if version is None or bool(row.get("active")):
                return
            if QMessageBox.question(
                self.shell,
                "Активировать версию",
                "Сделать эту неизменяемую версию активной?",
            ) != QMessageBox.StandardButton.Yes:
                return
            try:
                self.service.activate_artifact_version(
                    principal=self.session.principal,
                    project_id=project_id,
                    series_id=version.series_id,
                    version_id=version.id,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось активировать версию",
                    str(exc),
                )

        def export_or_open(row: Mapping[str, object]) -> None:
            version = row.get("_version")
            if version is None:
                return
            if version.external is not None:
                QDesktopServices.openUrl(QUrl(version.external.uri))
                return
            destination, _selected_filter = QFileDialog.getSaveFileName(
                self.shell,
                "Экспортировать файл",
                version.filename,
            )
            if not destination:
                return
            try:
                self.service.export_managed_artifact(
                    project_id,
                    version,
                    destination,
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось экспортировать файл",
                    str(exc),
                )

        def check_external(row: Mapping[str, object]) -> None:
            version = row.get("_version")
            if version is None or version.external is None:
                QMessageBox.information(
                    self.shell,
                    "Управляемый файл",
                    "Эта версия хранится в базе и не изменяется снаружи.",
                )
                return
            try:
                changed = self.service.external_artifact_changed(version)
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось проверить файл",
                    str(exc),
                )
                return
            QMessageBox.information(
                self.shell,
                "Проверка внешнего файла",
                "Файл изменён или недоступен." if changed else "Файл не изменился.",
            )

        general_actions: list[tuple[str, Callable[[], None]]] = []
        if Permission.ADD_NOTE in permissions:
            general_actions.extend(
                (
                    ("Добавить заметку", add_note),
                    ("Изменить последнюю заметку", revise_note),
                )
            )
        if Permission.IMPORT_ARTIFACT in permissions:
            general_actions.extend(
                (
                    (
                        "Добавить файл в базу",
                        lambda: create_file_series(external=False),
                    ),
                    (
                        "Добавить внешнюю ссылку",
                        lambda: create_file_series(external=True),
                    ),
                )
            )
        file_actions = (
            (
                (
                    "Добавить версию в базу",
                    lambda row: add_version(row, external=False),
                ),
                (
                    "Добавить внешнюю версию",
                    lambda row: add_version(row, external=True),
                ),
                ("Переименовать", rename_series),
                ("Архивировать", archive_series),
            )
            if Permission.IMPORT_ARTIFACT in permissions
            else ()
        )
        version_actions = (
            (
                ("Сделать активной", activate_version),
                ("Открыть / экспортировать", export_or_open),
                ("Проверить внешний файл", check_external),
            )
            if Permission.IMPORT_ARTIFACT in permissions
            else (("Открыть / экспортировать", export_or_open),)
        )
        return {
            "notes": tuple(
                {
                    "revision": note.revision,
                    "body": note.body,
                    "author": principals.get(
                        str(note.author_principal_id),
                        str(note.author_principal_id),
                    ),
                    "recorded_at": note.recorded_at.isoformat(),
                }
                for note in notes
            ),
            "files": tuple(files),
            "versions": tuple(versions),
            "actions": tuple(general_actions),
            "file_actions": file_actions,
            "version_actions": version_actions,
        }

    def show_project_properties(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return
        events = tuple(self.service.history(project.id))
        content = self._scope_content(project.id)

        def properties_for(value) -> tuple[tuple[str, object], ...]:
            return (
                ("Название", value.name),
                ("Идентификатор", str(value.id)),
                ("Размер матрицы", f"{value.width} × {value.height}"),
                ("Ориентация", value.orientation.value),
                ("Профиль хранения", value.storage_profile),
                ("Состояние", value.state.value),
                ("Ревизия", value.revision),
                ("Создан", value.created_at.isoformat()),
            )

        def temporal(entry: ObjectHistoryEntry):
            from datetime import datetime

            moment = datetime.fromisoformat(entry.recorded_at)
            value = self.service.get_project(project.id, as_of=moment)
            if value is None:
                return None
            return ObjectPropertiesSnapshot(
                title=f"{value.name} — на {entry.recorded_at}",
                object_kind="project-temporal",
                properties=properties_for(value),
                history=tuple(
                    history
                    for history in _history_entries(events)
                    if history.recorded_at <= entry.recorded_at
                ),
            )

        self._open_properties(
            ObjectPropertiesSnapshot(
                title=project.name,
                object_kind="project",
                properties=properties_for(project),
                history=_history_entries(events),
                temporal_loader=temporal,
                **content,
            )
        )

    def _show_layer_properties(self, layer_id: str) -> None:
        layer = next(
            (
                item
                for item in self.service.list_layers(self._project_id)
                if str(item.id) == str(layer_id)
            ),
            None,
        )
        if layer is None:
            self._error("Слой больше не доступен")
            return
        binding = self.service.layer_file_binding(self._project_id, layer_id)
        events = tuple(
            event
            for event in self.service.history(self._project_id)
            if _event_belongs_to_layer(event, layer_id)
        )
        created_events = tuple(
            event
            for event in events
            if str(getattr(event, "event_type", "")) == "LayerCreated"
            and str(_event_payload(event).get("layer_id", "")) == str(layer_id)
        )
        actor, created_at = _creator(created_events or events)
        paths = (
            ()
            if binding is None
            else (
                binding.image_directory,
                binding.ssc_directory,
                binding.prv_directory,
                binding.aux_directory,
            )
        )
        properties: list[tuple[str, object]] = [
            ("Название", layer.name),
            ("Тип объекта", "Слой"),
            ("Идентификатор", str(layer.id)),
            ("Идентификатор проекта", str(layer.project_id)),
            ("Тип слоя", layer.type.value),
            ("Порядок", layer.order),
            ("Состояние", layer.state.value),
            ("Ревизия", layer.revision),
            (
                "Путь",
                None if binding is None else binding.image_directory,
            ),
            ("Примечание", None),
            ("Количество файлов", _count_regular_files(paths)),
            ("Кто добавил", actor),
            ("Когда добавлен", created_at if created_at != "—" else layer.created_at.isoformat()),
        ]
        if binding is not None:
            properties.extend(
                (
                    ("Режим хранения", binding.mode.value),
                    ("Каталог изображений", binding.image_directory),
                    ("Каталог SSC", binding.ssc_directory),
                    ("Каталог PRV", binding.prv_directory),
                    ("Вспомогательный каталог", binding.aux_directory),
                    ("Корень импорта", binding.import_root),
                    ("Известно кадров", len(binding.frame_positions)),
                    ("Позиции кадров", binding.frame_positions),
                    ("Параметры преобразования", _plain_value(binding.conversion)),
                )
            )
        content = self._scope_content(
            self._project_id,
            layer_id=layer.id,
        )

        def temporal(entry: ObjectHistoryEntry):
            from datetime import datetime

            moment = datetime.fromisoformat(entry.recorded_at)
            value = next(
                (
                    candidate
                    for candidate in self.service.list_layers(
                        self._project_id,
                        as_of=moment,
                        include_archived=True,
                    )
                    if candidate.id == layer.id
                ),
                None,
            )
            if value is None:
                return None
            return ObjectPropertiesSnapshot(
                title=f"{value.name} — на {entry.recorded_at}",
                object_kind="layer-temporal",
                properties=(
                    ("Название", value.name),
                    ("Идентификатор", str(value.id)),
                    ("Тип слоя", value.type.value),
                    ("Порядок", value.order),
                    ("Состояние", value.state.value),
                    ("Ревизия", value.revision),
                    ("Создан", value.created_at.isoformat()),
                ),
                history=tuple(
                    history
                    for history in _history_entries(events)
                    if history.recorded_at <= entry.recorded_at
                ),
            )

        self._open_properties(
            ObjectPropertiesSnapshot(
                title=layer.name,
                object_kind="layer",
                properties=tuple(properties),
                history=_history_entries(events),
                temporal_loader=temporal,
                **content,
            )
        )

    def _representation_file_count(self, layer_id: str, representation) -> int | None:
        try:
            cells = self.service.frame_cells(
                self._project_id,
                layer_id,
                representation.id,
            )
        except (OSError, ValueError):
            cells = ()
        if cells:
            return len({str(item.frame_id) for item in cells})
        binding = self.service.layer_file_binding(self._project_id, layer_id)
        if (
            binding is not None
            and representation.source
            and str(Path(representation.source)).casefold()
            == str(Path(binding.image_directory)).casefold()
            and binding.frame_positions
        ):
            return len(binding.frame_positions)
        return _count_regular_files((representation.source,))

    def _show_node_properties(self, layer_id: str, node: PipelineNode) -> None:
        project_history = tuple(self.service.history(self._project_id))
        snapshot = self._pipeline_snapshot(self._project_id, layer_id)
        if node.kind == "blackbox":
            lane_id = str(node.node_id).removesuffix(":blackbox")
            lane = next((item for item in snapshot.lanes if item.lane_id == lane_id), None)
            hidden = (
                ()
                if lane is None
                else tuple(
                    item
                    for item in lane.nodes
                    if item.kind not in {"source", "vector", "missing"}
                )
            )
            hidden_events = tuple(
                event
                for event in project_history
                if any(_event_matches_node(event, item) for item in hidden)
            )
            self._open_properties(
                ObjectPropertiesSnapshot(
                    title=node.title,
                    object_kind=node.kind,
                    properties=(
                        ("Название", node.title),
                        ("Тип объекта", "Сгруппированные этапы"),
                        ("Идентификатор", node.node_id),
                        ("Путь", None),
                        ("Примечание", node.subtitle),
                        ("Количество файлов", None),
                        ("Кто добавил", None),
                        ("Когда добавлен", None),
                        ("Количество скрытых этапов", len(hidden)),
                        (
                            "Скрытые этапы",
                            [
                                {
                                    "id": item.node_id,
                                    "name": item.title,
                                    "kind": item.kind,
                                    "state": item.state,
                                }
                                for item in hidden
                            ],
                        ),
                    ),
                    history=_history_entries(hidden_events),
                )
            )
            return

        events = tuple(
            event for event in project_history if _event_matches_node(event, node)
        )
        actor, created_at = _creator(events)
        representations = self.service.list_representations(self._project_id, layer_id)
        representation = next(
            (
                item
                for item in representations
                if str(item.id) in {str(node.representation_id), str(node.node_id)}
            ),
            None,
        )
        run_id = str(node.details.get("run_id", ""))
        if not run_id:
            for prefix in ("workspace-job:", "workspace-output:"):
                if str(node.node_id).startswith(prefix):
                    run_id = str(node.node_id)[len(prefix):]
                    break
        derived_run = next(
            (
                item
                for item in self.service.list_derived_runs(self._project_id, layer_id)
                if item.run_id == run_id
            ),
            None,
        )
        kind_labels = {
            "source": "Исходное представление",
            "binary": "Бинарное представление",
            "vector": "Векторное представление",
            "dataset": "Выборка",
            "model": "Модель",
            "job": "Задание",
            "karakal": "Публикация Karakal",
            "missing": "Отсутствующий результат",
        }
        path: object = None
        note: object = None
        file_count: object = None
        properties: list[tuple[str, object]] = [
            ("Название", node.title),
            ("Тип объекта", kind_labels.get(node.kind, node.kind)),
            ("Идентификатор", node.node_id),
            ("Идентификатор слоя", str(layer_id)),
            ("Состояние", node.state or node.subtitle),
        ]
        if representation is not None:
            path = representation.source
            note = representation.note
            file_count = self._representation_file_count(layer_id, representation)
            properties.extend(
                (
                    ("Идентификатор представления", str(representation.id)),
                    ("Идентификатор проекта", str(representation.project_id)),
                    ("Вид представления", representation.kind.value),
                    ("Назначение", representation.purpose.value),
                    (
                        "Исходное представление",
                        None
                        if representation.source_image_representation_id is None
                        else str(representation.source_image_representation_id),
                    ),
                    ("Активно", representation.active),
                    ("Состояние объекта", representation.state.value),
                    ("Ревизия", representation.revision),
                    ("Дата объекта", representation.created_at.isoformat()),
                )
            )
        elif derived_run is not None:
            path = derived_run.path
            note = (
                derived_run.provenance.get("notes")
                or derived_run.provenance.get("note")
            )
            file_count = _count_regular_files((derived_run.path,))
            properties.extend(
                (
                    ("Идентификатор запуска", derived_run.run_id),
                    ("Тип результата", derived_run.kind.value),
                    ("Состояние запуска", derived_run.state.value),
                    ("Плагин", derived_run.plugin_id),
                    ("Операция", derived_run.operation),
                    ("Дата запуска", derived_run.created_at),
                    ("Provenance", derived_run.provenance),
                )
            )
        else:
            creation = next(
                (
                    event
                    for event in reversed(events)
                    if str(getattr(event, "event_type", ""))
                    in {
                        "PluginJobCreated",
                        "LayerPipelineActionRequested",
                        "KarakalAnalysisPublished",
                    }
                ),
                None,
            )
            latest = events[-1] if events else creation
            creation_payload = {} if creation is None else _event_payload(creation)
            latest_payload = {} if latest is None else _event_payload(latest)
            manifest = _nested_mapping(creation_payload, "manifest")
            job = _nested_mapping(latest_payload, "job")
            if node.kind == "karakal":
                properties.extend(
                    (
                        ("Идентификатор публикации", creation_payload.get("run_id")),
                        (
                            "Номер публикации",
                            creation_payload.get("publication_sequence"),
                        ),
                        (
                            "Версия плагина",
                            creation_payload.get("plugin_version")
                            or node.details.get("версия"),
                        ),
                        (
                            "Параметры",
                            creation_payload.get("parameters")
                            or node.details.get("параметры"),
                        ),
                        (
                            "Отчёт",
                            creation_payload.get("report")
                            or node.details.get("отчёт"),
                        ),
                    )
                )
            elif manifest or job:
                properties.extend(
                    (
                        (
                            "Идентификатор задания",
                            creation_payload.get("plugin_job_id")
                            or job.get("id")
                            or node.details.get("job_id"),
                        ),
                        ("Capability", manifest.get("capability") or node.details.get("capability")),
                        ("Плагин", manifest.get("plugin_name") or node.details.get("программа")),
                        ("Параметры", manifest.get("parameters") or node.details.get("параметры")),
                        ("Прогресс", job.get("progress") or node.details.get("прогресс")),
                        ("Ошибка", job.get("error") or node.details.get("ошибка")),
                    )
                )
            elif creation is not None:
                properties.extend(
                    (
                        ("Событие создания", getattr(creation, "event_type", "")),
                        ("Плагин", creation_payload.get("plugin_id")),
                        ("Capability", creation_payload.get("capability")),
                        ("Режим", creation_payload.get("mode")),
                        ("Параметры", creation_payload.get("parameters")),
                        ("Параметры события", creation_payload),
                    )
                )
        properties.extend(
            (
                ("Путь", path),
                ("Примечание", note),
                ("Количество файлов", file_count),
                ("Кто добавил", actor),
                ("Когда добавлен", created_at),
                ("Дополнительные свойства", node.details),
            )
        )
        self._open_properties(
            ObjectPropertiesSnapshot(
                title=node.title,
                object_kind=node.kind,
                properties=tuple(properties),
                history=_history_entries(events),
            )
        )

    def _layer_manager_action(self, _layer_id: str, _node: PipelineNode, action: str) -> None:
        if action == "archive_representation":
            self._archive_layer_representation(_layer_id, _node)
            return
        if action in {
            "rename_representation",
            "edit_representation_note",
            "activate_representation",
            "deactivate_representation",
        }:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox

            workspace = self._workspace
            project = self.service.get_project(self._project_id)
            layer = next(
                (
                    item
                    for item in self.service.list_layers(self._project_id)
                    if str(item.id) == str(_layer_id)
                ),
                None,
            )
            representation_id = str(_node.representation_id or _node.node_id)
            representation = next(
                (
                    item
                    for item in self.service.list_representations(
                        self._project_id,
                        _layer_id,
                    )
                    if str(item.id) == representation_id
                ),
                None,
            )
            if workspace is None or project is None or layer is None or representation is None:
                self._error("Репрезентация больше не доступна.")
                return
            try:
                if action == "rename_representation":
                    value, accepted = QInputDialog.getText(
                        self.shell,
                        "Переименовать репрезентацию",
                        "Новое название:",
                        text=representation.name,
                    )
                    if not accepted or value.strip() == representation.name:
                        return
                    self.service.rename_representation(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        representation=representation,
                        name=value,
                        idempotency_key=str(uuid4()),
                    )
                elif action == "edit_representation_note":
                    value, accepted = QInputDialog.getMultiLineText(
                        self.shell,
                        "Заметка репрезентации",
                        "Заметка:",
                        text=representation.note,
                    )
                    if not accepted or value.strip() == representation.note:
                        return
                    self.service.update_representation_note(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        representation=representation,
                        note=value,
                        idempotency_key=str(uuid4()),
                    )
                elif action == "activate_representation":
                    self.service.activate_representation(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        representation=representation,
                        idempotency_key=str(uuid4()),
                    )
                else:
                    answer = QMessageBox.question(
                        self.shell,
                        "Деактивировать репрезентацию",
                        f"Деактивировать «{representation.name}»?",
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        return
                    self.service.deactivate_representation(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        representation=representation,
                        idempotency_key=str(uuid4()),
                    )
            except Exception as exc:
                self._error(str(exc))
                return
            self._load_representations(workspace, self._project_id, _layer_id)
            if self._layer_dialog is not None:
                self._layer_dialog.set_pipeline(
                    self._pipeline_snapshot(self._project_id, _layer_id)
                )
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
            if not self._ensure_agent_source_ready(
                layer_id=str(_layer_id),
                source_representation_id=source_representation_id,
                retry=lambda: self._layer_manager_action(_layer_id, _node, action),
            ):
                return
        if action in {"recognize", "recognize_external"}:
            from PyQt6.QtCore import QSettings
            from PyQt6.QtWidgets import QFileDialog

            settings = QSettings("Kraken", "KrakenHub")
            key_root = f"external-model/{self._project_id}/{_layer_id}"
            key = f"{key_root}/path"
            model_path = ""
            if action == "recognize":
                result_directory = Path(str(_node.details.get("path", "")))
                model_path = str(
                    next(
                        (
                            path
                            for path in result_directory.rglob("*")
                            if path.is_file()
                            and path.suffix.casefold()
                            in {".onnx", ".pt", ".pth", ".h5", ".keras"}
                        ),
                        "",
                    )
                )
            else:
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
                stage_root = (
                    self.service.data_dir
                    / "agent-staging"
                    / f"{self._project_id}-{uuid4()}"
                )
                staged = link.stage(stage_root)
            except (OSError, ValueError) as exc:
                self._error(f"Не удалось подготовить внешнюю модель: {exc}")
                return
            if action == "recognize_external" and (linked_now or not stored_hash):
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
        try:
            job = self._submit_agent_action(
                layer_id=str(_layer_id),
                source_representation_id=source_representation_id,
                capability=capability,
                parameters=parameters,
            )
        except Exception as exc:
            self._error(f"Не удалось запустить задание: {exc}")
            return
        self.service.record_layer_pipeline_action(
            principal=self.session.principal,
            project_id=self._project_id,
            layer_id=_layer_id,
            action=action,
            node_id=_node.node_id,
            plugin_id=plugin_id,
            capability=capability,
            mode=mode,
            parameters={**parameters, "plugin_job_id": str(job.id)},
        )
        if self._layer_dialog is not None:
            self._layer_dialog.set_pipeline(
                self._pipeline_snapshot(self._project_id, _layer_id)
            )
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            self.shell,
            "Задание создано",
            f"Задание {job.id} передано Kraken Agent и продолжит работу независимо от окна проекта.",
        )

    def _layer_manager_layer_action(self, _layer_id: str, action: str) -> None:
        if action in {"rename_layer", "archive_layer"}:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox

            workspace = self._workspace
            project = self.service.get_project(self._project_id)
            if workspace is None or project is None:
                return
            layer = next(
                (
                    item
                    for item in self.service.list_layers(project.id)
                    if str(item.id) == str(_layer_id)
                ),
                None,
            )
            if layer is None:
                self._error("Слой больше не доступен.")
                return
            try:
                if action == "rename_layer":
                    name, accepted = QInputDialog.getText(
                        self.shell,
                        "Переименовать слой",
                        "Новое название:",
                        text=layer.name,
                    )
                    if not accepted or name.strip() == layer.name:
                        return
                    self.service.rename_layer(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        name=name,
                        idempotency_key=str(uuid4()),
                    )
                else:
                    answer = QMessageBox.question(
                        self.shell,
                        "Архивировать слой",
                        f"Архивировать слой «{layer.name}» без удаления файлов?",
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        return
                    self.service.archive_layer(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        idempotency_key=str(uuid4()),
                    )
            except Exception as exc:
                self._error(str(exc))
                return
            latest = self.service.get_project(project.id)
            if latest is not None:
                self._load_layers(workspace, latest)
            return
        if action == "delete_layer":
            from PyQt6.QtWidgets import (
                QDialog,
                QInputDialog,
                QLineEdit,
                QMessageBox,
            )

            workspace = self._workspace
            project = self.service.get_project(self._project_id)
            if workspace is None or project is None:
                return
            layer = next(
                (
                    item
                    for item in self.service.list_layers(project.id)
                    if str(item.id) == str(_layer_id)
                ),
                None,
            )
            if layer is None:
                self._error("Слой больше не доступен")
                return
            confirmation_dialog = QInputDialog(self.shell)
            confirmation_dialog.setWindowTitle("Удалить слой")
            confirmation_dialog.setLabelText(
                (
                    "Управляемые файлы будут перемещены в корзину Kraken, "
                    "внешние папки останутся без изменений.\n\n"
                    f"Для подтверждения введите: {layer.name}"
                )
            )
            confirmation_dialog.setInputMode(QInputDialog.InputMode.TextInput)
            confirmation_dialog.setTextEchoMode(QLineEdit.EchoMode.Normal)
            confirmation_dialog.setOkButtonText("Удалить")
            confirmation_dialog.setCancelButtonText("Отмена")
            if confirmation_dialog.exec() != QDialog.DialogCode.Accepted:
                return
            confirmation = confirmation_dialog.textValue()
            if confirmation != layer.name:
                QMessageBox.warning(
                    self.shell,
                    "Удаление не подтверждено",
                    "Введённое имя не совпадает с названием слоя.",
                )
                return
            try:
                self.service.delete_layer(
                    principal=self.session.principal,
                    project=project,
                    layer=layer,
                    confirmation_name=confirmation,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                self._error(str(exc))
                return
            latest = self.service.get_project(project.id)
            if latest is not None:
                self._load_layers(workspace, latest)
            QMessageBox.information(
                self.shell,
                "Слой удалён",
                "Метаданные заархивированы, управляемые файлы удаляются из корзины в фоне.",
            )
            return
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
            workspace = self._workspace
            source_id = (
                ""
                if workspace is None
                else str(workspace.image_representation_combo.currentData() or "")
            )
            if source_id and not self._ensure_agent_source_ready(
                layer_id=str(_layer_id),
                source_representation_id=source_id,
                retry=lambda: self._layer_manager_layer_action(_layer_id, action),
            ):
                return
            try:
                job = self._submit_agent_action(
                    layer_id=str(_layer_id),
                    source_representation_id=source_id,
                    capability=capability,
                    parameters={},
                )
                self.service.record_layer_pipeline_action(
                    principal=self.session.principal,
                    project_id=self._project_id,
                    layer_id=_layer_id,
                    action=action,
                    node_id=_layer_id,
                    plugin_id="karakal",
                    capability=capability,
                    mode=mode,
                    parameters={"plugin_job_id": str(job.id)},
                )
            except Exception as exc:
                self._error(f"Не удалось запустить задание: {exc}")
                return
            if self._layer_dialog is not None:
                self._layer_dialog.set_pipeline(
                    self._pipeline_snapshot(self._project_id, _layer_id)
                )
            return

    def _ensure_agent_source_ready(
        self,
        *,
        layer_id: str,
        source_representation_id: str,
        retry: Callable[[], None],
    ) -> bool:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog

        project = self.service.get_project(self._project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return False
        if self.service.frame_cells(
            project.id,
            layer_id,
            source_representation_id,
        ):
            return True
        layer = next(
            (
                item
                for item in self.service.list_layers(project.id)
                if str(item.id) == str(layer_id)
            ),
            None,
        )
        representation = next(
            (
                item
                for item in self.service.list_representations(project.id, layer_id)
                if str(item.id) == str(source_representation_id)
            ),
            None,
        )
        prepare = getattr(self.service, "materialize_representation_inputs", None)
        if layer is None or representation is None or not callable(prepare):
            self._error("Не удалось подготовить изображения для задания")
            return False
        key = (str(project.id), str(representation.id))
        pending = getattr(self, "_source_preparation_keys", None)
        if pending is None:
            pending = set()
            self._source_preparation_keys = pending
        if key in pending:
            self.shell.statusBar().showMessage(
                "Изображения уже подготавливаются для задания…",
                5000,
            )
            return False
        pending.add(key)
        progress_dialog = QProgressDialog(
            "Изображения сохраняются для запуска задания…",
            "",
            0,
            0,
            self.shell,
        )
        progress_dialog.setObjectName("agentSourcePreparationProgress")
        progress_dialog.setWindowTitle("Подготовка задания")
        progress_dialog.setWindowModality(Qt.WindowModality.NonModal)
        progress_dialog.setCancelButton(None)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)

        def perform_preparation():
            return prepare(
                principal=self.session.principal,
                project=project,
                layer=layer,
                representation=representation,
            )

        worker = _BackgroundCallThread(perform_preparation, parent=self.shell)
        workers = getattr(self, "_background_workers", None)
        if workers is None:
            workers = set()
            self._background_workers = workers
        workers.add(worker)

        def preparation_succeeded(cells: object) -> None:
            progress_dialog.close()
            pending.discard(key)
            if not cells:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось подготовить задание",
                    "В каталоге исходных изображений не найдено кадров проекта.",
                )
                return
            self.shell.statusBar().showMessage(
                "Изображения подготовлены. Задание запускается…",
                5000,
            )
            QTimer.singleShot(0, retry)

        def preparation_failed(message: str) -> None:
            progress_dialog.close()
            pending.discard(key)
            QMessageBox.warning(
                self.shell,
                "Не удалось подготовить задание",
                message,
            )

        def release_worker() -> None:
            workers.discard(worker)

        worker.succeeded.connect(preparation_succeeded)
        worker.failed.connect(preparation_failed)
        worker.finished.connect(release_worker)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        progress_dialog.show()
        return False

    def _submit_agent_action(
        self,
        *,
        layer_id: str,
        source_representation_id: str,
        capability: str,
        parameters: Mapping[str, object],
    ):
        workspace = self._workspace
        project = self.service.get_project(self._project_id)
        if workspace is None or project is None or not source_representation_id:
            raise ValueError("Не выбраны исходные изображения")
        capabilities = self.agent_runtime.ensure_started()
        if capability not in capabilities:
            raise ValueError(f"Установленные плагины не поддерживают {capability}")
        coordinates = workspace.matrix_view.selected_coordinates(
            maximum=self.service.profile.capabilities.max_frames or 100_000
        )
        if not coordinates:
            coordinates = tuple(
                (cell.x, cell.y)
                for cell in self.service.frame_cells(
                    project.id,
                    layer_id,
                    source_representation_id,
                )
            )
        if not coordinates:
            raise ValueError("В выбранном слое нет изображений для запуска задания")
        coordinate_by_frame = {
            str(project.frame_id_at(x, y)): (x, y)
            for x, y in coordinates
        }
        if capability == "frames.vectorize.v1":
            target_id = str(
                workspace.vector_representation_combo.currentData() or ""
            )
            if not target_id:
                raise ValueError(
                    "Сначала создайте слой CIF, связанный с исходными изображениями"
                )
        else:
            target_id = source_representation_id

        def version_path(version_id):
            return self.service.managed_artifact_path(project.id, version_id)

        def frame_coordinate(frame_id):
            coordinate = coordinate_by_frame.get(str(frame_id))
            if coordinate is None:
                raise ValueError(f"Кадр {frame_id} не входит в выбранный набор")
            return coordinate

        def media_type(version_id):
            version = self.service.artifact_version(project.id, version_id)
            if version is None:
                raise ValueError(f"Версия {version_id} больше не доступна")
            return version.media_type

        gateway = AgentPluginGateway(
            base_url=self.agent_runtime.base_url,
            token=self.agent_runtime.token,
            staging_root=self.service.data_dir / "agent" / "staging",
            source_for_version=version_path,
            coordinate_for_frame=frame_coordinate,
            media_type_for_version=media_type,
            capabilities=capabilities,
            v2_capabilities=frozenset(
                operation
                for operation, protocol in self.agent_runtime.protocol_by_capability.items()
                if protocol == "2.0"
            ),
        )
        self._agent_gateway = gateway
        return self.service.submit_plugin_job(
            principal=self.session.principal,
            gateway=gateway,
            project_id=project.id,
            layer_id=layer_id,
            source_representation_id=source_representation_id,
            target_representation_id=target_id,
            coordinates=coordinates,
            capability=capability,
            parameters=parameters,
            idempotency_key=str(uuid4()),
        )

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
            "rename_representation",
            "edit_representation_note",
            "activate_representation",
            "deactivate_representation",
            "delete_pipeline_step",
            "delete_layer",
            "rename_layer",
            "archive_layer",
        }:
            if not self._has_permission(Permission.MANAGE_STRUCTURE):
                return False, "Недостаточно прав для изменения структуры проекта"
            return True, ""
        if not self._has_permission(Permission.RUN_PLUGIN):
            return False, "Недостаточно прав для запуска плагинов"
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

        if not all(
            hasattr(self.service, name)
            for name in (
                "list_layers",
                "project_workspace",
                "layer_file_binding",
                "begin_derived_run",
            )
        ):
            stage = (
                self.service.data_dir
                / "agent-staging"
                / f"{self._project_id}-contour-{action}-{uuid4()}"
            ).resolve()
            stage.mkdir(parents=True, exist_ok=False)
            source_path = Path(str(representation.source or "")).expanduser()
            if source_path.is_absolute() and source_path.is_dir():
                input_directory = source_path.resolve()
            else:
                input_directory = stage / "inputs"
                input_directory.mkdir()
                for cell in self.service.frame_cells(
                    self._project_id,
                    layer_id,
                    representation.id,
                ):
                    if not cell.sha256:
                        continue
                    destination = input_directory / f"{cell.x}_{cell.y}.png"
                    with destination.open("xb") as stream:
                        stream.write(
                            self.service.read_project_blob(
                                self._project_id,
                                cell.sha256,
                            )
                        )
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

        layer = next(
            (
                value
                for value in self.service.list_layers(self._project_id)
                if str(value.id) == str(layer_id)
            ),
            None,
        )
        if layer is None:
            raise ValueError("слой больше не доступен")
        workspace = self.service.project_workspace(self._project_id)
        binding = self.service.layer_file_binding(self._project_id, layer_id)
        if workspace is None or binding is None or not workspace.available:
            raise ValueError("двухдисковое хранилище слоя недоступно")
        source_path = Path(str(representation.source or "")).expanduser()
        if source_path.is_absolute() and source_path.is_dir():
            input_directory = source_path.resolve()
        else:
            input_directory = (
                Path(workspace.derived_project_dir)
                / ".plugin-inputs"
                / f"contour-{action}-{uuid4()}"
            )
            input_directory.mkdir(parents=True)
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

        operation = self._action_requirement(action)[1]
        run = self.service.begin_derived_run(
            project_id=self._project_id,
            layer_id=layer_id,
            layer_name=layer.name,
            kind=(
                DerivedRunKind.DATASET
                if action == "prepare_dataset"
                else DerivedRunKind.VECTOR
            ),
            plugin_id="contour",
            operation=operation,
            principal=self.session.principal,
        )
        destination = Path(run.path)
        if action == "prepare_dataset":
            (destination / "images").mkdir(exist_ok=True)
            (destination / "cif").mkdir(exist_ok=True)
        result_manifest = destination / ".kraken-result.json"
        context_path = destination / ".kraken-context.json"
        input_directories = {"images": str(input_directory)}
        if binding.ssc_directory:
            input_directories["ssc"] = binding.ssc_directory
        if binding.prv_directory:
            input_directories["prv"] = binding.prv_directory
        current_project = self.service.get_project(self._project_id)
        if current_project is None:
            raise ValueError("проект больше не доступен")
        context = WorkspacePluginContextV1(
            project_id=str(self._project_id),
            project_name=current_project.name,
            layer_id=str(layer_id),
            layer_name=layer.name,
            operation=operation,
            plugin_id="contour",
            run_id=run.run_id,
            input_directories=input_directories,
            proposed_output_directory=str(destination),
            result_manifest_path=str(result_manifest),
        )
        context.write(context_path)
        return (
            (
                "--kraken-workspace-context",
                str(context_path),
            ),
            {
                "input_representation_id": str(representation.id),
                "input_directory": str(input_directory),
                "result_directory": str(destination),
                "workspace_run_id": run.run_id,
                "workspace_result_manifest": str(result_manifest),
            },
        )

    def _neural_train_launch_arguments(
        self,
        *,
        layer_id: str,
        node: PipelineNode,
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        dataset = Path(str(node.details.get("path", ""))).expanduser()
        images = dataset / "images"
        cif = dataset / "cif"
        if not dataset.is_absolute() or not images.is_dir() or not cif.is_dir():
            raise ValueError(
                "выбранная опубликованная выборка не содержит папки images и cif"
            )
        project = self.service.get_project(self._project_id)
        layer = next(
            (
                value
                for value in self.service.list_layers(self._project_id)
                if str(value.id) == str(layer_id)
            ),
            None,
        )
        if project is None or layer is None:
            raise ValueError("проект или слой больше не доступен")
        workspace = self.service.project_workspace(project.id)
        if workspace is None or not workspace.available:
            raise ValueError("двухдисковое хранилище проекта недоступно")
        operation = self._action_requirement("train")[1]
        run = self.service.begin_derived_run(
            project_id=project.id,
            layer_id=layer.id,
            layer_name=layer.name,
            kind=DerivedRunKind.RESULT,
            plugin_id="neuralimage",
            operation=operation,
            principal=self.session.principal,
        )
        destination = Path(run.path)
        result_manifest = destination / ".kraken-result.json"
        context_path = destination / ".kraken-context.json"
        WorkspacePluginContextV1(
            project_id=str(project.id),
            project_name=project.name,
            layer_id=str(layer.id),
            layer_name=layer.name,
            operation=operation,
            plugin_id="neuralimage",
            run_id=run.run_id,
            input_directories={
                "images": str(images.resolve()),
                "cif": str(cif.resolve()),
            },
            proposed_output_directory=str(destination),
            result_manifest_path=str(result_manifest),
        ).write(context_path)
        return (
            ("--kraken-workspace-context", str(context_path)),
            {
                "dataset_run_id": str(node.details.get("run_id", "")),
                "dataset_directory": str(dataset),
                "result_directory": str(destination),
                "workspace_run_id": run.run_id,
                "workspace_result_manifest": str(result_manifest),
            },
        )

    def _neural_recognition_launch_arguments(
        self,
        *,
        layer_id: str,
        source_representation_id: str,
        stage_root: Path,
        staged_model,
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        project = self.service.get_project(self._project_id)
        layer = next(
            (
                value
                for value in self.service.list_layers(self._project_id)
                if str(value.id) == str(layer_id)
            ),
            None,
        )
        binding = self.service.layer_file_binding(self._project_id, layer_id)
        representation = next(
            (
                value
                for value in self.service.list_representations(
                    self._project_id, layer_id
                )
                if str(value.id) == str(source_representation_id)
            ),
            None,
        )
        if project is None or layer is None or binding is None or representation is None:
            raise ValueError("проект, слой или исходные изображения недоступны")
        image_directory = Path(str(representation.source or binding.image_directory))
        if not image_directory.is_dir():
            raise ValueError("папка исходных изображений недоступна")
        input_stage = stage_root / "inputs"
        input_stage.mkdir(exist_ok=False)
        inputs: list[PluginFrameInput] = []
        frame_positions_by_id: dict[str, int] = {}
        image_paths = sorted(
            (
                path
                for path in image_directory.iterdir()
                if path.is_file()
                and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp"}
            ),
            key=self.service._natural_path_key,
        )
        for path in image_paths:
            position = binding.frame_positions.get(path.name)
            if position is None:
                continue
            visual_row, column = divmod(position - 1, project.width)
            x = column + 1
            y = (
                visual_row + 1
                if project.orientation is DomainOrientation.Y_DOWN
                else project.height - visual_row
            )
            if path.suffix.casefold() == ".bmp":
                from PIL import Image, ImageOps

                destination = input_stage / f"{path.stem}.png"
                with Image.open(path) as opened:
                    ImageOps.exif_transpose(opened).save(destination, format="PNG")
            else:
                destination = input_stage / path.name
                shutil.copy2(path, destination)
            digest_builder = hashlib.sha256()
            with destination.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest_builder.update(chunk)
            digest = digest_builder.hexdigest()
            frame_id = str(project.frame_id_at(x, y))
            # Layer bindings use Kraken's one-based matrix slots internally,
            # while microscope and plugin filenames are always zero-based.
            frame_positions_by_id[frame_id] = position - 1
            inputs.append(
                PluginFrameInput(
                    frame_id=frame_id,
                    x=x,
                    y=y,
                    artifact_version_id=str(uuid4()),
                    sha256=digest,
                    media_type=(
                        "image/jpeg"
                        if destination.suffix.casefold() in {".jpg", ".jpeg"}
                        else "image/png"
                        if destination.suffix.casefold() == ".png"
                        else "image/bmp"
                    ),
                    relative_path=f"inputs/{destination.name}",
                )
            )
        if not inputs:
            raise ValueError("не найдено изображений с валидной нумерацией кадров")
        operation = self._action_requirement("recognize")[1]
        run = self.service.begin_derived_run(
            project_id=project.id,
            layer_id=layer.id,
            layer_name=layer.name,
            kind=DerivedRunKind.RESULT,
            plugin_id="neuralimage",
            operation=operation,
            principal=self.session.principal,
        )
        job_id = str(uuid4())
        job_manifest = stage_root / "job.json"
        result_manifest = stage_root / "result.json"
        manifest = PluginJobManifest(
            job_id=job_id,
            operation=operation,
            project_id=str(project.id),
            layer_id=str(layer.id),
            actor_id=str(self.session.principal.id),
            target_representation_id=str(representation.id),
            inputs=tuple(inputs),
            parameters={
                "model_relative_path": staged_model.relative_path,
                "model_sha256": staged_model.used_sha256,
                "model_version": f"sha256:{staged_model.used_sha256[:12]}",
                "lossless_binary_png": True,
                "frame_positions": frame_positions_by_id,
            },
        )
        with job_manifest.open("x", encoding="utf-8") as stream:
            stream.write(manifest.to_json())
        return (
            (
                "--kraken-job-manifest",
                str(job_manifest),
                "--kraken-result-manifest",
                str(result_manifest),
                "--kraken-staging-root",
                str(stage_root),
            ),
            {
                "workspace_run_id": run.run_id,
                "agent_staging_root": str(stage_root),
                "agent_result_manifest": str(result_manifest),
                "frame_count": len(inputs),
            },
        )

    def _launch_managed_plugin(
        self,
        plugin_id: str,
        *,
        arguments: tuple[str, ...] = (),
    ):
        inventory = self.plugin_items.get(plugin_id)
        if inventory is None or not inventory.installed or not inventory.metadata.enabled:
            self._error(f"Плагин {plugin_id} не установлен или отключён")
            return None
        from .app import launch_plugin
        return launch_plugin(inventory.metadata, arguments=arguments)

    def _monitor_workspace_result(
        self,
        process,
        *,
        result_manifest: str,
        workspace_run_id: str,
        layer_id: str,
    ) -> None:
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QMessageBox

        timer = QTimer(self.shell)
        timer.setInterval(750)

        def poll() -> None:
            if process.poll() is None:
                return
            timer.stop()
            timer.deleteLater()
            manifest_path = Path(result_manifest)
            if not manifest_path.is_file():
                return_code = process.poll()
                message = (
                    "Плагин закрыт без публикации результата"
                    if return_code == 0
                    else f"Плагин завершился с кодом {return_code} и не опубликовал результат"
                )
                try:
                    self.service.fail_derived_run(
                        principal=self.session.principal,
                        project_id=self._project_id,
                        run_id=workspace_run_id,
                        error=message,
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Could not mark workspace plugin run as failed"
                    )
                QMessageBox.warning(
                    self.shell,
                    "Плагин не вернул результат",
                    message,
                )
                return
            try:
                result = WorkspacePluginResultV1.read(manifest_path)
                project = self.service.get_project(self._project_id)
                if project is None:
                    raise ValueError("проект больше не доступен")
                layer = next(
                    (
                        value
                        for value in self.service.list_layers(project.id)
                        if str(value.id) == str(layer_id)
                    ),
                    None,
                )
                if layer is None:
                    raise ValueError("слой больше не доступен")
                published, _representation = (
                    self.service.publish_workspace_plugin_result(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        result=result,
                    )
                )
            except Exception as exc:
                try:
                    self.service.fail_derived_run(
                        principal=self.session.principal,
                        project_id=self._project_id,
                        run_id=workspace_run_id,
                        error=str(exc),
                    )
                except Exception:
                    pass
                QMessageBox.warning(
                    self.shell,
                    "Не удалось вернуть результат плагина",
                    str(exc),
                )
                return
            workspace = self._workspace
            if workspace is not None:
                latest = self.service.get_project(self._project_id)
                if latest is not None:
                    self._load_layers(workspace, latest)
                item = workspace.layer_model.layer_by_id(layer_id)
                if item is not None:
                    self._select_layer(workspace, self._project_id, item)
            QMessageBox.information(
                self.shell,
                "Результат опубликован",
                f"Запуск {published.run_id[:8]} сохранён во втором хранилище.",
            )

        timer.timeout.connect(poll)
        timer.start()

    def _monitor_agent_result(
        self,
        process,
        *,
        result_manifest: str,
        staging_root: str,
        workspace_run_id: str,
        layer_id: str,
    ) -> None:
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QMessageBox

        timer = QTimer(self.shell)
        timer.setInterval(750)

        def poll() -> None:
            if process.poll() is None:
                return
            timer.stop()
            timer.deleteLater()
            root = Path(staging_root)
            try:
                result = PluginResultManifest.from_json(
                    Path(result_manifest).read_text(encoding="utf-8")
                )
                if result.outcome != "succeeded":
                    raise ValueError(
                        "; ".join(result.errors)
                        or f"NeuralImage завершился со статусом {result.outcome}"
                    )
                project = self.service.get_project(self._project_id)
                layer = next(
                    (
                        value
                        for value in self.service.list_layers(self._project_id)
                        if str(value.id) == str(layer_id)
                    ),
                    None,
                )
                if project is None or layer is None:
                    raise ValueError("проект или слой больше не доступен")
                workspace_result = WorkspacePluginResultV1(
                    run_id=workspace_run_id,
                    plugin_id="neuralimage",
                    operation=self._action_requirement("recognize")[1],
                    outcome="succeeded",
                    output_directory=str(root / "outputs"),
                    provenance={
                        "plugin_version": result.plugin_version,
                        "output_count": len(result.outputs),
                    },
                )
                published, _representation = (
                    self.service.publish_workspace_plugin_result(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        result=workspace_result,
                    )
                )
            except Exception as exc:
                try:
                    self.service.fail_derived_run(
                        principal=self.session.principal,
                        project_id=self._project_id,
                        run_id=workspace_run_id,
                        error=str(exc),
                    )
                except Exception:
                    pass
                QMessageBox.warning(
                    self.shell,
                    "Распознавание NeuralImage не опубликовано",
                    str(exc),
                )
                return
            finally:
                shutil.rmtree(root, ignore_errors=True)
            workspace = self._workspace
            if workspace is not None:
                latest = self.service.get_project(self._project_id)
                if latest is not None:
                    self._load_layers(workspace, latest)
                item = workspace.layer_model.layer_by_id(layer_id)
                if item is not None:
                    self._select_layer(workspace, self._project_id, item)
            QMessageBox.information(
                self.shell,
                "Распознавание опубликовано",
                f"Запуск {published.run_id[:8]} сохранён как бинарное представление.",
            )

        timer.timeout.connect(poll)
        timer.start()

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
            self.service.data_dir / "agent-staging" / f"{project_id}-karakal-{uuid4()}"
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
        current_vector_id = str(workspace.vector_representation_combo.currentData() or "")
        vectors = sorted(
            (
                value
                for value in representations
                if value.kind is RepresentationKind.VECTOR
                and str(value.source_image_representation_id or "") == current_image_id
            ),
            key=lambda value: (not value.active, value.name.casefold()),
        )
        if not any(str(item.id) == current_vector_id for item in vectors):
            current_vector_id = str(vectors[0].id) if vectors else ""
        workspace.set_representations(
            images=[
                (str(item.id), item.name)
                for item in images
            ],
            vectors=[
                (str(item.id), item.name)
                for item in vectors
            ],
            selected_image_id=current_image_id,
            selected_vector_id=current_vector_id,
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
        if representation.kind is RepresentationKind.VECTOR:
            source_image_id = str(representation.source_image_representation_id or "")
            image_index = workspace.image_representation_combo.findData(source_image_id)
            if image_index >= 0:
                workspace.image_representation_combo.blockSignals(True)
                workspace.image_representation_combo.setCurrentIndex(image_index)
                workspace.image_representation_combo.blockSignals(False)
            self._load_representations(workspace, project_id, layer_id)
            vector_index = workspace.vector_representation_combo.findData(
                str(representation.id)
            )
            if vector_index >= 0:
                workspace.vector_representation_combo.blockSignals(True)
                workspace.vector_representation_combo.setCurrentIndex(vector_index)
                workspace.vector_representation_combo.blockSignals(False)
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

    def _start_representation_import(
        self,
        *,
        workspace,
        project_id,
        layer_id: str,
        project,
        layer,
        representation,
        import_plan,
        kind: RepresentationKind,
    ) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog

        progress_dialog = QProgressDialog(
            "Файлы сохраняются в проекте в фоновом режиме…",
            "",
            0,
            0,
            self.shell,
        )
        progress_dialog.setObjectName("representationImportProgress")
        progress_dialog.setWindowTitle("Сохранение файлов")
        progress_dialog.setWindowModality(Qt.WindowModality.NonModal)
        progress_dialog.setCancelButton(None)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        operation_id = str(uuid4())

        def perform_import():
            return self.service.commit_managed_import(
                principal=self.session.principal,
                project=project,
                layer=layer,
                representation=representation,
                plan=import_plan,
                idempotency_key=operation_id,
            )

        worker = _BackgroundCallThread(perform_import, parent=self.shell)
        workers = getattr(self, "_background_workers", None)
        if workers is None:
            workers = set()
            self._background_workers = workers
        workers.add(worker)

        def refresh_created_representation() -> None:
            if (
                self._workspace is not workspace
                or self._project_id != str(project_id)
                or getattr(workspace, "_selected_layer_id", None) != layer_id
            ):
                return
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

        def import_succeeded(result: object) -> None:
            progress_dialog.close()
            refresh_created_representation()
            saved_count = len(result.versions)
            message = f"Сохранено файлов: {saved_count:n}."
            if import_plan.missing_coordinates:
                message += (
                    f" Не найдено файлов для кадров: "
                    f"{import_plan.missing_coordinates:n}."
                )
            self.shell.statusBar().showMessage(message, 15000)

        def import_failed(message: str) -> None:
            progress_dialog.close()
            refresh_created_representation()
            QMessageBox.warning(
                self.shell,
                "Не удалось сохранить файлы",
                (
                    "Слой изображений создан, но не все файлы удалось сохранить.\n\n"
                    f"{message}"
                ),
            )

        def release_worker() -> None:
            workers.discard(worker)

        worker.succeeded.connect(import_succeeded)
        worker.failed.connect(import_failed)
        worker.finished.connect(release_worker)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        progress_dialog.show()

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
                        f"Проверка файлов не пройдена:\n{issue_text}",
                    )
                    continue
                answer = QMessageBox.question(
                    form.dialog,
                    "Подтверждение импорта файлов",
                    (
                        f"Файлов: {len(import_plan.items)}\n"
                        f"Объём: {import_plan.total_bytes:n} байт\n"
                        f"Пустых кадров: {import_plan.missing_coordinates:n}\n\n"
                        f"{issue_text}\n\nЗаписать файлы в базу?"
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
            except Exception as exc:
                QMessageBox.warning(form.dialog, f"Не удалось добавить {label}", str(exc))
                continue

            if import_plan is not None:
                self._start_representation_import(
                    workspace=workspace,
                    project_id=project_id,
                    layer_id=layer_id,
                    project=project,
                    layer=layer,
                    representation=representation,
                    import_plan=import_plan,
                    kind=kind,
                )
                return

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
        from PyQt6.QtWidgets import (
            QDialog,
            QInputDialog,
            QMessageBox,
            QProgressDialog,
        )

        project = self.service.get_project(project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return
        project_binding = self.service.project_workspace(project.id)
        if project_binding is None:
            is_remote = bool(
                getattr(self.service, "is_remote_project", lambda _id: False)(
                    project.id
                )
            )
            if not is_remote:
                self._error("Проект не привязан к двухдисковому хранилищу.")
                return
            name, accepted = QInputDialog.getText(
                self.shell,
                "Новый слой",
                "Название:",
            )
            if not accepted or not name.strip():
                return
            layer_labels = {
                "Металл": LayerType.METAL,
                "Контакт": LayerType.CONTACT,
                "Затвор": LayerType.GATE,
                "Диффузия": LayerType.DIFFUSION,
            }
            layer_label, accepted = QInputDialog.getItem(
                self.shell,
                "Новый слой",
                "Тип:",
                tuple(layer_labels),
                editable=False,
            )
            if not accepted:
                return
            try:
                layer = self.service.create_layer(
                    principal=self.session.principal,
                    project=project,
                    name=name,
                    layer_type=layer_labels[layer_label],
                    order=len(self.service.list_layers(project.id)),
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:  # noqa: BLE001 - Qt boundary reports domain/transport failures
                QMessageBox.warning(
                    self.shell,
                    "Не удалось добавить слой",
                    str(exc),
                )
                return
            self._show_created_layer(workspace, project.id, layer)
            return
        missing = [
            value
            for value in (
                project_binding.source_project_dir,
                project_binding.derived_project_dir,
            )
            if not Path(value).is_dir()
        ]
        if missing:
            self._error(
                "Файловые операции заблокированы: недоступно хранилище.\n\n"
                + "\n".join(missing)
            )
            return
        dialog = LayerCreationDialog(
            maximum_frames=project.frame_count,
            scanner=self.service.scan_layer_source,
            parent=self.shell,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        order = len(self.service.list_layers(project.id)) + 1
        operation_id = str(uuid4())
        if dialog.mode == "manual":
            try:
                layer, _binding, _representation = self.service.create_external_layer(
                    principal=self.session.principal,
                    project=project,
                    name=dialog.layer_name,
                    layer_type=dialog.layer_type,
                    order=order,
                    image_directory=dialog.manual_images.text(),
                    ssc_directory=dialog.manual_ssc.text() or None,
                    prv_directory=dialog.manual_prv.text() or None,
                    idempotency_key=operation_id,
                )
            except Exception as exc:
                QMessageBox.warning(dialog, "Не удалось добавить слой", str(exc))
                return
            self._show_created_layer(workspace, project.id, layer)
            QMessageBox.information(
                self.shell,
                "Слой создан",
                "Внешние каталоги проверены и привязаны. Файлы не копировались.",
            )
            return

        scan = dialog.scan_result
        if scan is None:
            self._error("Результат сканирования потерян. Выполните сканирование ещё раз.")
            return
        conversion = dialog.conversion_settings()
        cancelled = Event()
        progress_dialog = QProgressDialog(
            "Подготовка импорта…",
            "Отменить",
            0,
            max(1, scan.total_files),
            self.shell,
        )
        progress_dialog.setObjectName("layerImportProgress")
        progress_dialog.setWindowTitle("Импорт слоя")
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.canceled.connect(cancelled.set)

        def perform(*, progress, cancelled):
            return self.service.create_layer_from_disk(
                principal=self.session.principal,
                project=project,
                name=dialog.layer_name,
                layer_type=dialog.layer_type,
                order=order,
                scan=scan,
                conversion=conversion,
                idempotency_key=operation_id,
                progress=progress,
                cancelled=cancelled,
            )

        worker = _LayerCreateThread(perform, cancelled=cancelled, parent=self.shell)

        def update_progress(done: int, total: int, label: str) -> None:
            progress_dialog.setMaximum(max(1, total))
            progress_dialog.setValue(done)
            progress_dialog.setLabelText(
                f"Скопировано и обработано: {done:n} из {total:n}\n{label}"
            )

        def succeeded(result: object) -> None:
            progress_dialog.setValue(progress_dialog.maximum())
            progress_dialog.close()
            layer, _binding, _representation = result
            self._show_created_layer(workspace, project.id, layer)
            QMessageBox.information(
                self.shell,
                "Импорт завершён",
                f"Слой «{dialog.layer_name}» создан атомарно. "
                "Исходная папка не изменена.",
            )

        def failed(message: str) -> None:
            progress_dialog.close()
            if cancelled.is_set():
                QMessageBox.information(
                    self.shell,
                    "Импорт отменён",
                    "Слой не создан, частичные файлы удалены.",
                )
            else:
                QMessageBox.warning(self.shell, "Не удалось импортировать слой", message)

        worker.progress.connect(update_progress)
        worker.succeeded.connect(succeeded)
        worker.failed.connect(failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        progress_dialog.show()

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
    from PyQt6.QtCore import QModelIndex, Qt
    from PyQt6.QtGui import QColor, QPainter
    from PyQt6.QtWidgets import (
        QColorDialog,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QHBoxLayout,
        QLineEdit,
        QMenu,
        QMessageBox,
        QPushButton,
        QStyledItemDelegate,
        QStyleOptionViewItem,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    class ColorSwatchDelegate(QStyledItemDelegate):
        def paint(
            self,
            painter: QPainter | None,
            option: QStyleOptionViewItem,
            index: QModelIndex,
        ) -> None:
            super().paint(painter, option, index)
            if painter is None:
                return
            color = QColor(str(index.data(Qt.ItemDataRole.UserRole) or ""))
            if not color.isValid():
                return
            painter.save()
            swatch = option.rect.adjusted(8, 5, -8, -5)
            painter.setPen(QColor("#64748B"))
            painter.setBrush(color)
            painter.drawRoundedRect(swatch, 4, 4)
            painter.restore()

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
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    table.setItemDelegateForColumn(1, ColorSwatchDelegate(table))
    layout.addWidget(table)
    performers: list[Performer] = []

    def refresh() -> None:
        nonlocal performers
        performers = list(service.list_performers())
        table.setRowCount(len(performers))
        for row, performer in enumerate(performers):
            name_item = QTableWidgetItem(performer.name)
            color_item = QTableWidgetItem()
            color_item.setData(Qt.ItemDataRole.UserRole, performer.color)
            color_item.setToolTip("Нажмите, чтобы изменить цвет")
            account_item = QTableWidgetItem("GitLab" if performer.principal_id is not None else "Ручной")
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, color_item)
            table.setItem(row, 2, account_item)

    def performer_dialog(
        *,
        title: str,
        initial_name: str = "",
        initial_color: str = "#60A5FA",
    ) -> tuple[str, str] | None:
        dialog = QDialog(host)
        dialog.setWindowTitle(title)
        form = QFormLayout(dialog)
        name = QLineEdit(initial_name)
        selected_color = QColor(initial_color)
        color = QPushButton()
        color.setObjectName("performerColorPicker")
        color.setToolTip("Выбрать цвет исполнителя")

        def update_color_button() -> None:
            color_value = selected_color.name(QColor.NameFormat.HexRgb).upper()
            text_color = "#111827" if selected_color.lightnessF() > 0.65 else "#FFFFFF"
            color.setText(color_value)
            color.setStyleSheet(f"background-color: {color_value}; color: {text_color};")

        def choose_color() -> None:
            nonlocal selected_color
            chosen_color = QColorDialog.getColor(selected_color, dialog, "Цвет исполнителя")
            if not chosen_color.isValid():
                return
            selected_color = chosen_color
            update_color_button()

        color.clicked.connect(choose_color)
        update_color_button()
        form.addRow("Имя", name)
        form.addRow("Цвет", color)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return name.text(), selected_color.name(QColor.NameFormat.HexRgb).upper()

    def create() -> None:
        values = performer_dialog(title="Новый исполнитель")
        if values is None:
            return
        try:
            service.create_manual_performer(name=values[0], color=values[1])
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось создать исполнителя", str(exc))
            return
        refresh()

    def edit(row: int) -> None:
        if not 0 <= row < len(performers):
            return
        performer = performers[row]
        values = performer_dialog(
            title="Изменить исполнителя",
            initial_name=performer.name,
            initial_color=performer.color,
        )
        if values is None:
            return
        try:
            service.update_performer(
                performer_id=performer.id,
                name=values[0],
                color=values[1],
            )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось изменить исполнителя", str(exc))
            return
        refresh()

    def change_color(row: int, column: int) -> None:
        if column != 1 or not 0 <= row < len(performers):
            return
        performer = performers[row]
        selected_color = QColorDialog.getColor(
            QColor(performer.color),
            host,
            "Цвет исполнителя",
        )
        if not selected_color.isValid():
            return
        try:
            service.update_performer(
                performer_id=performer.id,
                name=performer.name,
                color=selected_color.name(QColor.NameFormat.HexRgb).upper(),
            )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось изменить цвет исполнителя", str(exc))
            return
        refresh()

    def delete(row: int) -> None:
        if not 0 <= row < len(performers):
            return
        performer = performers[row]
        answer = QMessageBox.question(
            host,
            "Удалить исполнителя",
            f"Удалить исполнителя «{performer.name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            service.archive_performer(performer.id)
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось удалить исполнителя", str(exc))
            return
        refresh()

    def context_menu(position) -> None:
        row = table.rowAt(position.y())
        if row < 0:
            return
        table.selectRow(row)
        menu = QMenu(table)
        edit_action = menu.addAction("Изменить")
        delete_action = menu.addAction("Удалить")
        selected_action = menu.exec(table.viewport().mapToGlobal(position))
        if selected_action is edit_action:
            edit(row)
        elif selected_action is delete_action:
            delete(row)

    add.clicked.connect(create)
    refresh_button.clicked.connect(refresh)
    table.cellClicked.connect(change_color)
    table.customContextMenuRequested.connect(context_menu)
    refresh()
    return host


def _my_work_panel(
    service: EmbeddedProjectService,
    session: DesktopSession,
    controller: DesktopController,
):
    from PyQt6.QtWidgets import (
        QFileDialog,
        QHBoxLayout,
        QInputDialog,
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
    refresh_button = QPushButton("Обновить")
    actions.addWidget(refresh_button)
    actions.addStretch(1)
    layout.addLayout(actions)
    table = QTableWidget(0, 8)
    table.setHorizontalHeaderLabels(
        (
            "Тип",
            "Проект",
            "Слой",
            "Исполнитель",
            "Состояние",
            "Срок / обновление",
            "Кадры / прогресс",
            "Действия",
        )
    )
    table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(table)

    review_states = {
        "draft": "Черновик",
        "issued": "Выдано на проверку",
        "partially_returned": "Возвращено частично",
        "awaiting_acceptance": "Ожидает принятия",
        "completed": "Завершено",
        "changes_requested": "На доработке",
        "cancelled": "Отменено",
    }
    job_states = {
        "queued": "В очереди",
        "staging": "Подготовка файлов",
        "running": "Выполняется",
        "waiting_for_user": "Ожидает пользователя",
        "importing": "Импорт результата",
        "partial": "Частичный результат",
        "succeeded": "Завершено",
        "failed": "Ошибка",
        "cancelled": "Отменено",
        "awaiting_authorization": "Ожидает разрешения",
        "recovery_required": "Требуется восстановление",
    }

    def repeat_export(batch) -> None:
        parent = QFileDialog.getExistingDirectory(host, "Каталог для повторной выгрузки")
        if not parent:
            return
        destination = Path(parent) / f"Проверка_{batch.id}_{uuid4().hex[:8]}"
        try:
            service.export_review_batch(
                principal=session.principal,
                batch=batch,
                destination=destination,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось выгрузить пакет", str(exc))
            return
        QMessageBox.information(host, "Пакет выгружен", str(destination))
        refresh()

    def accept(batch) -> None:
        identifiers = service.review_candidate_version_ids(batch)
        if not identifiers:
            QMessageBox.information(
                host,
                "Нет файлов для подтверждения",
                "Для этого задания нет файлов, ожидающих подтверждения.",
            )
            return
        if QMessageBox.question(
            host,
            "Принять результат",
            f"Использовать загруженные файлы как текущие ({len(identifiers)})?",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            service.accept_review(
                principal=session.principal,
                batch=batch,
                candidate_version_ids=identifiers,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось принять результат", str(exc))
        refresh()

    def request_changes(batch) -> None:
        reason, accepted = QInputDialog.getMultiLineText(
            host,
            "На доработку",
            "Причина (обязательно):",
        )
        if not accepted:
            return
        if not reason.strip():
            QMessageBox.warning(host, "Причина не указана", "Укажите причину возврата.")
            return
        try:
            service.request_review_changes(
                principal=session.principal,
                batch=batch,
                reason=reason,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось вернуть на доработку", str(exc))
        refresh()

    def cancel(batch) -> None:
        if QMessageBox.question(
            host,
            "Отменить проверку",
            "Отменить это задание на проверку?",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            service.cancel_review_batch(
                principal=session.principal,
                batch=batch,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось отменить проверку", str(exc))
        refresh()

    def cancel_job(job) -> None:
        if QMessageBox.question(
            host,
            "Отменить задание",
            "Отменить выполнение задания плагина?",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            controller.cancel_agent_job(job)
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось отменить задание", str(exc))
        refresh()

    def retry_job(job) -> None:
        try:
            controller.retry_agent_job(job)
        except Exception as exc:
            QMessageBox.warning(
                host,
                "Не удалось повторить задание",
                str(exc),
            )
        refresh()

    def import_partial_job(job) -> None:
        if QMessageBox.question(
            host,
            "Импортировать частичный результат",
            "Импортировать доступные результаты задания?",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            controller.import_partial_agent_job(job)
        except Exception as exc:
            QMessageBox.warning(
                host,
                "Не удалось импортировать результат",
                str(exc),
            )
        refresh()

    def refresh() -> None:
        projects = {
            str(project.id): project.name
            for project in service.list_projects(include_archived=True)
        }
        project_permissions = {
            project_id: service.project_permissions(
                project_id,
                session.principal,
            )
            for project_id in projects
        }
        layers = {
            str(layer.id): layer.name
            for project_id in projects
            for layer in service.list_layers(project_id, include_archived=True)
        }
        performers = {
            str(performer.id): performer.name
            for performer in service.list_performers(include_archived=True)
        }
        batches = service.review_batches()
        jobs = service.plugin_jobs()
        table.setRowCount(len(batches) + len(jobs))
        for row, batch in enumerate(batches):
            values = (
                "Проверка",
                projects.get(str(batch.project_id), str(batch.project_id)),
                layers.get(str(batch.layer_id), str(batch.layer_id)),
                performers.get(str(batch.assignee_id), str(batch.assignee_id)),
                review_states.get(batch.state.value, batch.state.value),
                "" if batch.due_at is None else batch.due_at.astimezone().strftime("%d.%m.%Y %H:%M"),
                str(len(batch.items)),
            )
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))
            action_host = QWidget(table)
            action_layout = QHBoxLayout(action_host)
            action_layout.setContentsMargins(0, 0, 0, 0)
            permissions = project_permissions.get(str(batch.project_id), frozenset())
            if Permission.ASSIGN_WORK in permissions:
                repeat = QPushButton("Повторно выгрузить", action_host)
                repeat.setEnabled(
                    batch.state.value not in {"completed", "cancelled"}
                )
                repeat.clicked.connect(
                    lambda _checked=False, value=batch: repeat_export(value)
                )
                action_layout.addWidget(repeat)
            if (
                batch.state.value == "awaiting_acceptance"
                and Permission.ACCEPT_REVIEW in permissions
            ):
                accept_button = QPushButton("Принять", action_host)
                accept_button.clicked.connect(
                    lambda _checked=False, value=batch: accept(value)
                )
                action_layout.addWidget(accept_button)
                changes_button = QPushButton("На доработку", action_host)
                changes_button.clicked.connect(
                    lambda _checked=False, value=batch: request_changes(value)
                )
                action_layout.addWidget(changes_button)
            if (
                batch.state.value not in {"completed", "cancelled"}
                and Permission.MANAGE_REVIEW in permissions
            ):
                cancel_button = QPushButton("Отменить", action_host)
                cancel_button.clicked.connect(
                    lambda _checked=False, value=batch: cancel(value)
                )
                action_layout.addWidget(cancel_button)
            table.setCellWidget(row, 7, action_host)
        for offset, job in enumerate(jobs, start=len(batches)):
            values = (
                "Плагин",
                projects.get(str(job.project_id), str(job.project_id)),
                layers.get(str(job.layer_id), str(job.layer_id)),
                "—",
                job_states.get(job.state.value, job.state.value),
                job.updated_at.astimezone().strftime("%d.%m.%Y %H:%M"),
                f"{job.progress * 100:.0f} %",
            )
            for column, value in enumerate(values):
                table.setItem(offset, column, QTableWidgetItem(value))
            action_host = QWidget(table)
            action_layout = QHBoxLayout(action_host)
            action_layout.setContentsMargins(0, 0, 0, 0)
            permissions = project_permissions.get(str(job.project_id), frozenset())
            if (
                job.state.value == "recovery_required"
                and Permission.RUN_PLUGIN in permissions
            ):
                retry = QPushButton("Повторить", action_host)
                retry.clicked.connect(
                    lambda _checked=False, value=job: retry_job(value)
                )
                action_layout.addWidget(retry)
            if (
                job.state.value == "partial"
                and Permission.RUN_PLUGIN in permissions
            ):
                partial = QPushButton("Импортировать", action_host)
                partial.clicked.connect(
                    lambda _checked=False, value=job: import_partial_job(value)
                )
                action_layout.addWidget(partial)
            if (
                job.state.value not in {"succeeded", "failed", "cancelled"}
                and Permission.RUN_PLUGIN in permissions
            ):
                cancel_button = QPushButton("Отменить", action_host)
                cancel_button.clicked.connect(
                    lambda _checked=False, value=job: cancel_job(value)
                )
                action_layout.addWidget(cancel_button)
            if job.error:
                action_host.setToolTip(job.error)
            table.setCellWidget(offset, 7, action_host)

    refresh_button.clicked.connect(refresh)
    controller.my_work_refresh = refresh
    refresh()
    return host


def _statistics_panel(service: EmbeddedProjectService):
    from datetime import UTC, datetime

    from PyQt6.QtCore import QDateTime, Qt
    from PyQt6.QtWidgets import (
        QComboBox,
        QDateTimeEdit,
        QFileDialog,
        QGridLayout,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from kraken_manager.infrastructure.reports import (
        METRIC_DEFINITIONS,
        ReportFilters,
        ReportGranularity,
        ReportService,
        present_metrics,
    )

    from .statistics_widgets import MetricChartWidget

    host = QWidget()
    layout = QVBoxLayout(host)
    controls = QHBoxLayout()
    project = QComboBox()
    project.setObjectName("statisticsProjectFilter")
    project.addItem("Все проекты", None)
    for item in sorted(
        service.list_projects(include_archived=True),
        key=lambda value: (value.name.casefold(), str(value.id)),
    ):
        archived = item.state.value == "archived"
        project.addItem(
            f"{item.name} (архивный)" if archived else item.name,
            str(item.id),
        )
    start = QDateTimeEdit(QDateTime.currentDateTime().addDays(-30))
    end = QDateTimeEdit(QDateTime.currentDateTime())
    start.setObjectName("statisticsStart")
    end.setObjectName("statisticsEnd")
    start.setDisplayFormat("dd.MM.yyyy HH:mm")
    end.setDisplayFormat("dd.MM.yyyy HH:mm")
    start.setCalendarPopup(True)
    end.setCalendarPopup(True)
    refresh_button = QPushButton("Рассчитать")
    csv_button = QPushButton("Экспорт CSV")
    xlsx_button = QPushButton("Экспорт XLSX")
    controls.addWidget(QLabel("Проект"))
    controls.addWidget(project)
    controls.addWidget(QLabel("С"))
    controls.addWidget(start)
    controls.addWidget(QLabel("По"))
    controls.addWidget(end)
    controls.addWidget(refresh_button)
    controls.addWidget(csv_button)
    controls.addWidget(xlsx_button)
    controls.addStretch(1)
    layout.addLayout(controls)

    tabs = QTabWidget()
    tabs.setObjectName("statisticsTabs")
    table = QTableWidget(0, 2)
    table.setObjectName("statisticsSummary")
    table.setHorizontalHeaderLabels(("Показатель", "Значение"))
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    tabs.addTab(table, "Сводка")

    granularities = (
        (ReportGranularity.DAY, "По дням"),
        (ReportGranularity.WEEK, "По неделям"),
        (ReportGranularity.MONTH, "По месяцам"),
        (ReportGranularity.YEAR, "По годам"),
    )
    charts: dict[ReportGranularity, dict[str, MetricChartWidget]] = {}
    for granularity, title in granularities:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        chart_host = QWidget()
        chart_layout = QGridLayout(chart_host)
        chart_layout.setContentsMargins(8, 8, 8, 8)
        chart_layout.setSpacing(10)
        granularity_charts: dict[str, MetricChartWidget] = {}
        for index, definition in enumerate(METRIC_DEFINITIONS):
            chart = MetricChartWidget(definition)
            chart.setObjectName(f"statisticsChart_{granularity.value}_{definition.key}")
            chart_layout.addWidget(chart, index // 2, index % 2)
            granularity_charts[definition.key] = chart
        chart_layout.setColumnStretch(0, 1)
        chart_layout.setColumnStretch(1, 1)
        scroll.setWidget(chart_host)
        tabs.addTab(scroll, title)
        charts[granularity] = granularity_charts
    layout.addWidget(tabs)
    reports = ReportService()
    local_timezone = datetime.now().astimezone().tzinfo or UTC

    def filters() -> ReportFilters:
        start_at = datetime.fromtimestamp(start.dateTime().toSecsSinceEpoch(), UTC).replace(
            second=0,
            microsecond=0,
        )
        end_at = datetime.fromtimestamp(end.dateTime().toSecsSinceEpoch(), UTC).replace(
            second=59,
            microsecond=999999,
        )
        selected_project = project.currentData()
        project_ids = frozenset((str(selected_project),)) if selected_project else frozenset()
        return ReportFilters(start_at, end_at, project_ids=project_ids)

    def refresh() -> None:
        try:
            records = service.activity_records()
            selected_filters = filters()
            metrics = reports.aggregate(records, selected_filters)
            series_by_granularity = {
                granularity: reports.aggregate_series(
                    records,
                    selected_filters,
                    granularity,
                    timezone=local_timezone,
                )
                for granularity, _title in granularities
            }
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось рассчитать статистику", str(exc))
            return
        presented = present_metrics(metrics.values)
        table.setRowCount(len(presented))
        for row, metric in enumerate(presented):
            name_item = QTableWidgetItem(metric.definition.label)
            name_item.setToolTip(metric.definition.description)
            value_item = QTableWidgetItem(metric.formatted_value)
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, value_item)
        for granularity, series in series_by_granularity.items():
            for definition in METRIC_DEFINITIONS:
                charts[granularity][definition.key].set_series(
                    series.buckets,
                    metrics.values.get(definition.key, 0),
                )

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
                reports.write_xlsx(
                    destination,
                    records,
                    filters(),
                    assume_sorted=True,
                    include_series=True,
                    timezone=local_timezone,
                )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось экспортировать отчёт", str(exc))

    refresh_button.clicked.connect(refresh)
    csv_button.clicked.connect(lambda: export("csv"))
    xlsx_button.clicked.connect(lambda: export("xlsx"))
    refresh()
    return host


def _administration_panel(
    service: EmbeddedProjectService,
    session: DesktopSession,
):
    from PyQt6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QWidget,
    )

    host = QWidget()
    layout = QFormLayout(host)
    capabilities = service.profile.capabilities
    layout.addRow("Каталог", QLabel(str(service.catalog_root)))
    layout.addRow("Хранилище метаданных", QLabel(service.profile.metadata_backend.value))
    layout.addRow("Хранилище файлов", QLabel(service.profile.blob_backend))
    layout.addRow("Режим записи", QLabel("multi-writer" if capabilities.multi_writer else "single-writer"))
    layout.addRow("Максимум кадров", QLabel(str(capabilities.max_frames or "без лимита")))
    project_count = QLabel()
    layout.addRow("Проектов", project_count)
    layout.addRow("Источник истины", QLabel("append-only event log; SQLite — перестраиваемый индекс"))
    projects = QComboBox()
    layout.addRow("Проект для резервной копии", projects)
    actions = QWidget()
    action_layout = QHBoxLayout(actions)
    action_layout.setContentsMargins(0, 0, 0, 0)
    scan_button = QPushButton("Проверить целостность")
    roots_button = QPushButton("Настроить хранилища…")
    export_button = QPushButton("Создать резервную копию")
    import_button = QPushButton("Восстановить резервную копию")
    attach_button = QPushButton("Подключить проект из локального каталога…")
    action_layout.addWidget(scan_button)
    action_layout.addWidget(roots_button)
    action_layout.addWidget(export_button)
    action_layout.addWidget(import_button)
    action_layout.addWidget(attach_button)
    layout.addRow("Операции", actions)
    audit_table = QTableWidget(0, 4)
    audit_table.setHorizontalHeaderLabels(
        ("Время", "Проект", "Действие", "Кто")
    )
    audit_table.horizontalHeader().setStretchLastSection(True)
    layout.addRow("Журнал аудита", audit_table)

    def refresh_projects() -> None:
        values = service.list_projects(include_archived=True)
        projects.clear()
        for project in values:
            projects.addItem(project.name, str(project.id))
        project_count.setText(str(len(values)))
        export_button.setEnabled(bool(values))
        event_labels = {
            "ProjectCreated": "Создан проект",
            "ProjectRenamed": "Переименован проект",
            "ProjectArchived": "Проект архивирован",
            "ProjectRestored": "Проект восстановлен",
            "LayerCreated": "Создан слой",
            "LayerRenamed": "Переименован слой",
            "LayerArchived": "Слой архивирован",
            "RepresentationCreated": "Создана репрезентация",
            "RepresentationRenamed": "Переименована репрезентация",
            "RepresentationActivated": "Репрезентация активирована",
            "RepresentationDeactivated": "Репрезентация деактивирована",
            "RepresentationArchived": "Репрезентация архивирована",
            "ReviewBatchCreated": "Создано задание на проверку",
            "ReviewBatchIssued": "Задание выдано",
            "ReviewBatchReexported": "Пакет выдан повторно",
            "ReviewReturnCommitted": "Результат проверки загружен",
            "ReviewBatchAccepted": "Результат проверки принят",
            "ReviewChangesRequested": "Запрошена доработка",
            "ReviewBatchCancelled": "Проверка отменена",
            "ProjectRoleAssigned": "Назначена роль",
            "ProjectRoleRevoked": "Отозвана роль",
        }
        events = sorted(
            (
                (project, event)
                for project in values
                for event in service.history(project.id)
            ),
            key=lambda value: value[1].recorded_at,
            reverse=True,
        )[:500]
        audit_table.setRowCount(len(events))
        for row, (project, event) in enumerate(events):
            actor = getattr(event, "actor", None)
            values_row = (
                event.recorded_at.astimezone().strftime("%d.%m.%Y %H:%M:%S"),
                project.name,
                event_labels.get(event.event_type, event.event_type),
                getattr(actor, "display_name", "") or str(
                    getattr(actor, "principal_id", "")
                ),
            )
            for column, value in enumerate(values_row):
                audit_table.setItem(row, column, QTableWidgetItem(str(value)))

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
        destination = QFileDialog.getExistingDirectory(host, "Каталог для резервной копии")
        if not project_id or not destination:
            return
        target = os.path.join(destination, f"kraken-backup-{project_id}")
        try:
            manifest = service.export_backup(
                project_id,
                target,
                principal=session.principal,
            )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось создать резервную копию", str(exc))
            return
        QMessageBox.information(
            host,
            "Резервная копия создана",
            f"Идентификатор: {manifest.bundle_id}\nСобытий: {manifest.event_count}\n{target}",
        )

    def import_backup() -> None:
        source = QFileDialog.getExistingDirectory(host, "Выберите резервную копию Kraken")
        if not source:
            return
        take_ownership = QMessageBox.question(
            host,
            "Владение проектом",
            "Если на этой рабочей станции нет доступного владельца проекта, принять владение?",
        ) == QMessageBox.StandardButton.Yes
        try:
            project = service.import_backup(
                source,
                principal=session.principal,
                take_ownership=take_ownership,
            )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось восстановить резервную копию", str(exc))
            return
        refresh_projects()
        QMessageBox.information(host, "Резервная копия восстановлена", f"Проект: {project.name}")

    def attach_project() -> None:
        source = QFileDialog.getExistingDirectory(
            host,
            "Выберите локальный каталог проекта Kraken",
            str(service.catalog_root),
        )
        if not source:
            return
        take_ownership = QMessageBox.question(
            host,
            "Владение проектом",
            "Если локальный владелец недоступен, принять владение проектом?",
        ) == QMessageBox.StandardButton.Yes
        try:
            project = service.attach_project(
                source,
                principal=session.principal,
                take_ownership=take_ownership,
            )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось подключить проект", str(exc))
            return
        refresh_projects()
        QMessageBox.information(host, "Проект подключён", project.name)

    scan_button.clicked.connect(scan)
    roots_button.clicked.connect(
        lambda: _configure_workspace_roots(host, service, force=True)
    )
    export_button.clicked.connect(export_backup)
    import_button.clicked.connect(import_backup)
    attach_button.clicked.connect(attach_project)
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
    service = build_workspace_service()
    local = service.local if isinstance(service, DualCatalogService) else service
    session = _development_session(local) or _login(None, local)
    if session is None:
        return 1
    if isinstance(service, DualCatalogService) and service.remote is not None:
        # Shared mutations use the GitLab principal carried by the remote client.
        session_summary = (
            f"{session.principal.display_name}\n"
            f"Локальная сессия + сервер {service.remote.base_url}"
        )
    else:
        session_summary = f"{session.principal.display_name}\nЛокальная сессия"
    shell = ProjectManagerShell()
    shell.set_session_summary(session_summary)
    shell._desktop_controller = DesktopController(  # keep Qt slots alive
        shell,
        service,
        session,
        thumbnail_store_uri=thumbnail_store_uri,
        plugin_items=items,
    )
    controller = shell._desktop_controller
    plugins_page = shell.page("plugins")
    if plugins_page is not None:
        plugins_page.set_content(_plugin_panel(items))
    performers_page = shell.page("performers")
    if performers_page is not None:
        performers_page.set_content(_performer_panel(local))
    my_work_page = shell.page("my_work")
    if my_work_page is not None:
        my_work_page.set_content(_my_work_panel(local, session, controller))
    statistics_page = shell.page("statistics")
    if statistics_page is not None:
        statistics_page.set_content(_statistics_panel(local))
    administration_page = shell.page("administration")
    if administration_page is not None:
        administration_page.set_content(_administration_panel(local, session))
    shell.show()
    return app.exec()


__all__ = ["DesktopController", "run_manager_gui"]
