"""manager_app: administration panel."""
from __future__ import annotations
import os
from kraken_hub.composition import DesktopSession, EmbeddedProjectService

def _administration_panel(
    service: EmbeddedProjectService,
    session: DesktopSession,
):
    from kraken_hub.manager_app import _configure_workspace_roots
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
