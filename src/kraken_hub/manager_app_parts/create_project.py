"""manager_app: create project."""
from __future__ import annotations
from uuid import uuid4
from kraken_manager.domain.project import GridOrientation as DomainOrientation
from kraken_manager.infrastructure.workspace_files import validate_workspace_roots
from kraken_manager.presentation.qt.models import ProjectListItem
from kraken_manager.presentation.qt.widgets import GridDimensionsWidget
from kraken_hub.workspace_service import REMOTE_STORAGE_PROFILE

from kraken_hub.manager_app_parts.connect_saved_server import (
    _configure_administration_page,
    _server_project_hint,
)


class CreateProjectMixin:
    def create_project(self) -> None:
        from kraken_hub.manager_app import _configure_workspace_roots
        from kraken_core.settings import QSettings
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

        from kraken_hub.preferences_dialog import program_server_url

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
            label = "Сервер" if profile.id == REMOTE_STORAGE_PROFILE.id else "Локальный"
            profile_box.addItem(label, profile.id)
        form.addRow("Хранилище", profile_box)
        settings = QSettings("Kraken", "KrakenHub")
        layout.addLayout(form)
        dimensions = GridDimensionsWidget(maximum_frames=None)
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

        def _saved_workspace_roots() -> tuple[str, str] | None:
            saved_source = str(settings.value("workspace/source-root", "") or "")
            saved_derived = str(settings.value("workspace/derived-root", "") or "")
            if not saved_source or not saved_derived:
                return None
            try:
                source, derived = validate_workspace_roots(saved_source, saved_derived)
            except Exception:
                return None
            return str(source), str(derived)

        def _update_storage_hint(_index: int = 0) -> None:
            nonlocal source_root, derived_root
            profile_id = str(profile_box.currentData() or "")
            is_server = profile_id == REMOTE_STORAGE_PROFILE.id
            if is_server:
                source_root = None
                derived_root = None
                storage.setText(_server_project_hint(settings))
                return
            saved = _saved_workspace_roots()
            if saved is None:
                source_root = None
                derived_root = None
                storage.setText(
                    "Каталоги исходных и производных данных будут запрошены при создании локального проекта."
                )
                return
            source_root, derived_root = saved
            storage.setText(
                f"Исходные данные: {source_root}\n"
                f"Производные данные: {derived_root}"
            )

        if getattr(self.service, "remote", None) is not None or program_server_url(settings):
            server_index = profile_box.findData(REMOTE_STORAGE_PROFILE.id)
            if server_index >= 0:
                profile_box.setCurrentIndex(server_index)
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
                    roots = _configure_workspace_roots(self.shell, self.service)
                    if roots is None:
                        raise ValueError("Выберите каталоги хранилища для локального проекта")
                    source_root, derived_root = roots
                    _update_storage_hint()
                if profile_id == REMOTE_STORAGE_PROFILE.id:
                    self._connect_program_account()
                    _configure_administration_page(
                        self.shell, self.service, self.session
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
