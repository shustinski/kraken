"""PyQt Desktop composition for projects, matrix and legacy plugin launcher."""

from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from kraken_core.analysis_protocol import (
    AnalysisArtifactInput,
    AnalysisFrameInput,
    AnalysisParameter,
    AnalysisSourceRole,
)
from kraken_core.analysis_run_protocol import (
    AnalysisRecipe,
    AnalysisRuntimeIdentity,
    AnalysisSourceBinding,
    canonical_json,
    payload_sha256,
)
from kraken_core.frame_matrix import MatrixSession, ThumbnailStoreFactory
from kraken_core.frame_matrix.qt import FrameMatrixWidget
from kraken_core.plugins import PluginInventoryItem
from kraken_core.qt import configure_application_identity
from kraken_core.styles import load_shared_stylesheet
from kraken_manager.domain.project import GridOrientation as DomainOrientation
from kraken_manager.domain.project import LayerType, RepresentationKind
from kraken_manager.application.imports import ImportMappingMode
from kraken_manager.application.analysis_runs import AnalysisRunCoordinator
from kraken_manager.infrastructure.analysis import FilesystemAnalysisStore
from kraken_manager.infrastructure.plugin import AgentAnalysisGateway
from kraken_manager.presentation.qt import ProjectManagerShell, ProjectWorkspacePage
from kraken_manager.presentation.qt.models import LayerListItem, ProjectListItem
from kraken_manager.presentation.qt.widgets import ClickableLabel, GridDimensionsWidget

from . import windows_credentials
from .composition import DesktopSession, EmbeddedProjectService
from .matrix_source import KrakenMatrixAssetSource, KrakenMatrixDataSource


def _installed_version(distribution: str, fallback: str = "not-installed") -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return fallback


def _karakal_runtime_identity() -> AnalysisRuntimeIdentity:
    return AnalysisRuntimeIdentity(
        engine_version=_installed_version("karakal", "1.0.0"),
        engine_build=os.environ.get("KARAKAL_BUILD_HASH", "development"),
        python_version=platform.python_version(),
        numpy_version=_installed_version("numpy"),
        opencv_version=_installed_version("opencv-python", _installed_version("opencv-python-headless")),
        operating_system=f"{platform.system()} {platform.release()}",
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
    ) -> None:
        self.shell = shell
        self.service = service
        self.session = session
        self.thumbnail_store_uri = str(thumbnail_store_uri)
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
        self._configure_analysis(workspace, project.id)

    def _configure_analysis(self, workspace, project_id) -> None:
        from PyQt6.QtCore import QTimer

        store = FilesystemAnalysisStore(self.service.catalog_root, str(project_id))
        workspace._analysis_store = store
        token = os.environ.get("KRAKEN_AGENT_TOKEN", "").strip()
        coordinator = None
        if token:
            gateway = AgentAnalysisGateway(
                base_url=os.environ.get("KRAKEN_AGENT_URL", "http://127.0.0.1:8765"),
                token=token,
                staging_root=Path(
                    os.environ.get(
                        "KRAKEN_AGENT_STAGING_ROOT",
                        str(Path.home() / ".kraken" / "agent" / "staging"),
                    )
                ),
            )
            coordinator = AnalysisRunCoordinator(
                store,
                gateway,
                lambda version_id: self.service.artifact_source_path(project_id, version_id),
            )
        workspace._analysis_coordinator = coordinator
        workspace.analysisRequested.connect(
            lambda configuration: self._start_analysis(workspace, project_id, configuration)
        )
        panel = workspace.analysis_runs_panel
        panel.metricChanged.connect(lambda _key: self._refresh_analysis_panel(workspace))
        panel.run_table.itemSelectionChanged.connect(lambda: self._refresh_analysis_results(workspace))
        panel.retryRequested.connect(lambda run_id: self._analysis_action(workspace, "retry", run_id))
        panel.cancelRequested.connect(lambda run_id: self._analysis_action(workspace, "cancel", run_id))
        panel.repeatRequested.connect(lambda run_id: self._analysis_action(workspace, "repeat", run_id))
        panel.exportRequested.connect(lambda run_id: self._export_analysis(workspace, run_id))
        panel.renderMapRequested.connect(lambda _run_id, _frame_id: self._map_not_configured(workspace))
        timer = QTimer(workspace)
        timer.setInterval(750)
        timer.timeout.connect(lambda: self._poll_analysis(workspace))
        timer.start()
        workspace._analysis_timer = timer
        self._refresh_analysis_panel(workspace)

    def _analysis_frames(self, workspace, project_id, configuration) -> tuple[
        tuple[AnalysisFrameInput, ...], tuple[AnalysisSourceBinding, ...]
    ]:
        layer_id = getattr(workspace, "_selected_layer_id", None)
        if not layer_id:
            raise ValueError("Сначала выберите слой проекта")
        coordinates = workspace.matrix_view.selected_coordinates(maximum=100_000)
        if not coordinates:
            raise ValueError("Выборка не содержит кадров")
        raw_bindings = configuration.get("bindings")
        if not isinstance(raw_bindings, dict):
            raise ValueError("Некорректные привязки моделей A/B/C")
        representations = {
            str(item.id): item
            for item in self.service.list_representations(project_id, layer_id)
        }
        resolved: dict[str, tuple[object, dict[tuple[int, int], object]]] = {}
        source_bindings: list[AnalysisSourceBinding] = []
        for key, identifier_value in raw_bindings.items():
            identifier = str(identifier_value)
            representation = representations.get(identifier)
            if representation is None or representation.kind is not RepresentationKind.IMAGE:
                raise ValueError(f"Модель {key} не является доступным изображением этого слоя")
            cells = {
                (item.x, item.y): item
                for item in self.service.frame_cells(project_id, layer_id, identifier)
            }
            selected_cells = [cells.get(coordinate) for coordinate in coordinates]
            missing = [coordinate for coordinate, cell in zip(coordinates, selected_cells, strict=True) if cell is None]
            if missing:
                preview = ", ".join(f"({x}, {y})" for x, y in missing[:5])
                raise ValueError(f"У модели {key} нет {len(missing)} выбранных кадров: {preview}")
            digests = [str(cell.sha256) for cell in selected_cells if cell is not None]
            source_bindings.append(
                AnalysisSourceBinding(
                    binding_key=str(key),
                    source_id=identifier,
                    source_version=payload_sha256(digests),
                    display_name=representation.name,
                )
            )
            resolved[str(key)] = (representation, cells)

        frames: list[AnalysisFrameInput] = []
        for x, y in coordinates:
            artifacts: list[AnalysisArtifactInput] = []
            frame_id = ""
            for key, (representation, cells) in resolved.items():
                cell = cells[(x, y)]
                if frame_id and frame_id != cell.frame_id:
                    raise ValueError(f"Привязки моделей расходятся по идентификатору кадра ({x}, {y})")
                frame_id = cell.frame_id
                version_item = self.service.get_artifact_version(project_id, cell.artifact_version_id)
                if version_item is None:
                    raise ValueError(f"Версия артефакта для кадра ({x}, {y}) больше недоступна")
                suffix = Path(version_item.filename).suffix.lower()
                artifacts.append(
                    AnalysisArtifactInput(
                        binding_key=key,
                        role=AnalysisSourceRole.MODEL_OUTPUT,
                        artifact_id=str(version_item.series_id),
                        artifact_version_id=str(version_item.id),
                        relative_path=f"inputs/{key}/{frame_id}{suffix}",
                        media_type=version_item.media_type,
                        sha256=version_item.sha256,
                        display_name=representation.name,
                    )
                )
            frames.append(AnalysisFrameInput(frame_id=frame_id, x=x, y=y, artifacts=tuple(artifacts)))
        return tuple(frames), tuple(source_bindings)

    def _start_analysis(self, workspace, project_id, configuration) -> None:
        from PyQt6.QtWidgets import QMessageBox

        coordinator = getattr(workspace, "_analysis_coordinator", None)
        if coordinator is None:
            QMessageBox.warning(
                workspace,
                "Kraken Agent не настроен",
                "Задайте KRAKEN_AGENT_TOKEN, KRAKEN_AGENT_URL и KRAKEN_AGENT_STAGING_ROOT, затем перезапустите Kraken.",
            )
            return
        try:
            frames, source_bindings = self._analysis_frames(workspace, project_id, configuration)
            raw_recipe = configuration.get("recipe")
            raw_parameters = configuration.get("parameters", {})
            if not isinstance(raw_recipe, dict) or not isinstance(raw_parameters, dict):
                raise ValueError("Некорректная конфигурация анализа")
            recipe = AnalysisRecipe.from_payload(raw_recipe)
            parameters = tuple(
                AnalysisParameter(str(key), value)
                for key, value in sorted(raw_parameters.items())
                if isinstance(value, (str, int, float, bool))
            )
            coordinator.start(
                project_id=str(project_id),
                frames=frames,
                source_bindings=source_bindings,
                recipe=recipe,
                runtime=_karakal_runtime_identity(),
                parameters=parameters,
            )
        except Exception as exc:
            QMessageBox.warning(workspace, "Не удалось запустить анализ", str(exc))
            self._refresh_analysis_panel(workspace)
            return
        workspace.analysis_runs_panel.setVisible(True)
        self._refresh_analysis_panel(workspace)

    def _poll_analysis(self, workspace) -> None:
        coordinator = getattr(workspace, "_analysis_coordinator", None)
        store = getattr(workspace, "_analysis_store", None)
        if coordinator is None or store is None:
            return
        changed = False
        for run in store.list_runs():
            if run.state in {"queued", "running"}:
                coordinator.refresh(run.run_id)
                changed = True
        if changed:
            self._refresh_analysis_panel(workspace)

    def _refresh_analysis_panel(self, workspace) -> None:
        store = getattr(workspace, "_analysis_store", None)
        if store is None:
            return
        panel = workspace.analysis_runs_panel
        selected = panel.selected_run_id()
        runs = store.list_runs()
        panel.set_runs(
            {
                "run_id": run.run_id,
                "state": run.state,
                "progress": f"{run.completed_frames + run.failed_frames}/{run.total_frames}",
                "models": ", ".join(
                    f"{item.binding_key}={item.display_name or item.source_id}"
                    for item in run.manifest.source_bindings
                ),
                "recipe": canonical_json(run.manifest.recipe.expression.to_payload()),
                "created_at": "",
            }
            for run in runs
        )
        if runs:
            selected_row = next((index for index, run in enumerate(runs) if run.run_id == selected), 0)
            panel.run_table.selectRow(selected_row)
            active = runs[selected_row]
            progress = round(100 * (active.completed_frames + active.failed_frames) / active.total_frames)
            panel.progress.setValue(progress)
        else:
            panel.progress.setValue(0)
        self._refresh_analysis_results(workspace)

    def _refresh_analysis_results(self, workspace) -> None:
        store = getattr(workspace, "_analysis_store", None)
        panel = workspace.analysis_runs_panel
        run_id = panel.selected_run_id()
        if store is None or not run_id:
            panel.set_results(())
            return
        metric_key = str(panel.metric_combo.currentData() or "xor")
        panel.set_results(dict(row) for row in store.frame_results(run_id, metric_key))

    def _analysis_action(self, workspace, action: str, run_id: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        coordinator = getattr(workspace, "_analysis_coordinator", None)
        if coordinator is None or not run_id:
            return
        try:
            if action == "retry":
                coordinator.retry_failed(run_id)
            elif action == "cancel":
                coordinator.cancel(run_id)
            elif action == "repeat":
                coordinator.repeat(run_id)
        except Exception as exc:
            QMessageBox.warning(workspace, "Операция анализа не выполнена", str(exc))
        self._refresh_analysis_panel(workspace)

    def _export_analysis(self, workspace, run_id: str) -> None:
        import csv
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        store = getattr(workspace, "_analysis_store", None)
        if store is None or not run_id:
            return
        destination, _selected_filter = QFileDialog.getSaveFileName(
            workspace,
            "Экспорт результатов анализа",
            f"karakal-{run_id}.csv",
            "CSV (*.csv)",
        )
        if not destination:
            return
        metric_key = str(workspace.analysis_runs_panel.metric_combo.currentData() or "xor")
        rows = store.frame_results(run_id, metric_key)
        try:
            with Path(destination).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("frame_id", "x", "y", "status", "message", "metric", "raw_value", "goodness", "percentile"))
                for row in rows:
                    writer.writerow(
                        (
                            row["frame_id"], row["x"], row["y"], row["status"], row["message"],
                            metric_key, row["raw_value"], row["goodness"], row["percentile"],
                        )
                    )
        except OSError as exc:
            QMessageBox.warning(workspace, "Не удалось экспортировать анализ", str(exc))

    @staticmethod
    def _map_not_configured(workspace) -> None:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            workspace,
            "Карта расхождений",
            "Ленивый рендеринг карты доступен через Karakal Engine; запуск из Agent будет добавлен отдельной операцией.",
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

    def _select_layer(self, workspace, project_id, item: LayerListItem) -> None:
        workspace._selected_layer_id = item.layer_id
        self._load_representations(workspace, project_id, item.layer_id)

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

    def _add_representation(self, workspace, project_id, kind: RepresentationKind) -> None:
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
        source_image_id = (
            str(workspace.image_representation_combo.currentData() or "")
            if kind is RepresentationKind.VECTOR
            else None
        )
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
    )
    shell.show()
    return app.exec()


__all__ = ["DesktopController", "run_manager_gui"]
