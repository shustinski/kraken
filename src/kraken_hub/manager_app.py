"""PyQt Desktop composition for projects, matrix and legacy plugin launcher."""

from __future__ import annotations

from collections.abc import Mapping
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

from PyQt6.QtCore import QThread, pyqtSignal

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
from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose
from kraken_manager.application.imports import ImportMappingMode
from kraken_manager.workspace import DerivedRunKind
from kraken_manager.infrastructure.workspace_files import validate_workspace_roots
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
from .matrix_source import KrakenMatrixAssetSource, KrakenMatrixDataSource


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
        self.catalog_page.deleteRequested.connect(self.delete_project)
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
        self._error(
            "Имя проекта совпадает с физическими папками на двух дисках "
            "и после создания не изменяется."
        )

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
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QVBoxLayout,
        )

        roots = _configure_workspace_roots(self.shell, self.service)
        if roots is None:
            return
        source_root, derived_root = roots
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
        storage = QLabel(
            f"Исходные данные: {source_root}\n"
            f"Производные данные: {derived_root}"
        )
        storage.setWordWrap(True)
        layout.addWidget(storage)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Создать проект")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
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
                    layer_template=False,
                    source_root=source_root,
                    derived_root=derived_root,
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
        workspace_binding = self.service.project_workspace(project.id)
        if workspace_binding is None:
            self._error(
                "Этот проект создан до введения двухдискового хранилища. "
                "Автоматическая миграция отключена; создайте новый проект."
            )
            return
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
        self._open_properties(
            ObjectPropertiesSnapshot(
                title=layer.name,
                object_kind="layer",
                properties=tuple(properties),
                history=_history_entries(events),
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
        elif plugin_id == "neuralimage" and action == "train":
            try:
                launch_arguments, neural_parameters = (
                    self._neural_train_launch_arguments(
                        layer_id=_layer_id,
                        node=_node,
                    )
                )
            except (OSError, ValueError) as exc:
                self._error(f"Не удалось подготовить обучение NeuralImage: {exc}")
                return
            parameters.update(neural_parameters)
        elif plugin_id == "neuralimage" and action in {
            "recognize",
            "recognize_external",
        }:
            try:
                launch_arguments, recognition_parameters = (
                    self._neural_recognition_launch_arguments(
                        layer_id=_layer_id,
                        source_representation_id=source_representation_id,
                        stage_root=stage_root,
                        staged_model=staged,
                    )
                )
            except (OSError, ValueError) as exc:
                self._error(f"Не удалось подготовить распознавание NeuralImage: {exc}")
                return
            parameters.update(recognition_parameters)
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
        process = self._launch_managed_plugin(plugin_id, arguments=launch_arguments)
        if process is not None and parameters.get("workspace_result_manifest"):
            self._monitor_workspace_result(
                process,
                result_manifest=str(parameters["workspace_result_manifest"]),
                workspace_run_id=str(parameters["workspace_run_id"]),
                layer_id=str(_layer_id),
            )
        elif process is not None and parameters.get("agent_result_manifest"):
            self._monitor_agent_result(
                process,
                result_manifest=str(parameters["agent_result_manifest"]),
                staging_root=str(parameters["agent_staging_root"]),
                workspace_run_id=str(parameters["workspace_run_id"]),
                layer_id=str(_layer_id),
            )
        if self._layer_dialog is not None:
            self._layer_dialog.set_pipeline(self._pipeline_snapshot(self._project_id, _layer_id))

    def _layer_manager_layer_action(self, _layer_id: str, action: str) -> None:
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
            "delete_layer",
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
        from PyQt6.QtWidgets import QDialog, QMessageBox, QProgressDialog

        project = self.service.get_project(project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return
        project_binding = self.service.project_workspace(project.id)
        if project_binding is None:
            self._error("Проект не привязан к двухдисковому хранилищу.")
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
    roots_button = QPushButton("Настроить хранилища…")
    export_button = QPushButton("Создать backup")
    import_button = QPushButton("Восстановить backup")
    projects.hide()
    export_button.hide()
    import_button.hide()
    action_layout.addWidget(scan_button)
    action_layout.addWidget(roots_button)
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
    roots_button.clicked.connect(
        lambda: _configure_workspace_roots(host, service, force=True)
    )
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
