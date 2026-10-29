"""manager_app: administration panel."""
from __future__ import annotations
import os
from uuid import uuid4
from kraken_hub.composition import DesktopSession, EmbeddedProjectService

def _administration_panel(
    service: EmbeddedProjectService,
    session: DesktopSession,
):
    from kraken_hub.presentation.qt.manager_app import _configure_workspace_roots
    from PyQt6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QTableWidget,
        QTableWidgetItem,
        QWidget,
    )

    host = QWidget()
    from PyQt6.QtCore import Qt
    from kraken_hub.presentation.qt.administration_tasks import AdministrationTasks

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
    action_layout = QGridLayout(actions)
    action_layout.setContentsMargins(0, 0, 0, 0)
    scan_button = QPushButton("Проверить целостность")
    refresh_button = QPushButton("Обновить каталог")
    roots_button = QPushButton("Настроить хранилища…")
    export_button = QPushButton("Создать резервную копию")
    import_button = QPushButton("Восстановить резервную копию")
    attach_button = QPushButton("Подключить проект из локального каталога…")
    for index, button in enumerate((scan_button, refresh_button, export_button, import_button, roots_button, attach_button)):
        action_layout.addWidget(button, index // 2, index % 2)
    layout.addRow("Операции", actions)
    migrate_button = QPushButton("Миграция старых поручений…")
    migrate_button.setObjectName("migrateLegacyWorkOrders")
    layout.addRow("Совместимость", migrate_button)
    audit_table = QTableWidget(0, 4)
    audit_table.setHorizontalHeaderLabels(
        ("Время", "Проект", "Действие", "Кто")
    )
    audit_table.horizontalHeader().setStretchLastSection(True)
    layout.addRow("Журнал аудита", audit_table)
    status = QLabel()
    status.setObjectName("administrationStatus")
    status.setTextFormat(Qt.TextFormat.PlainText)
    status.setWordWrap(True)
    layout.addRow(status)
    progress = QProgressBar()
    progress.setObjectName("administrationProgress")
    progress.setRange(0, 0)
    progress.hide()
    layout.addRow(progress)

    def set_busy(busy):
        for button in (scan_button, refresh_button, roots_button, export_button, import_button, attach_button, migrate_button):
            button.setEnabled(not busy)
        progress.setVisible(busy)
        projects.setEnabled(not busy)
        if not busy:
            export_button.setEnabled(projects.count() > 0)
            migrate_button.setEnabled(projects.count() > 0)

    tasks = AdministrationTasks(host, status, set_busy)
    host._admin_tasks = tasks

    def failure(title):
        return lambda message: QMessageBox.warning(host, title, message)

    def load_projects():
        values = service.list_projects(include_archived=True)
        events = sorted(
            ((project, event) for project in values for event in service.history(project.id)),
            key=lambda value: value[1].recorded_at,
            reverse=True,
        )[:500]
        return values, events

    def show_projects(result) -> None:
        values, events = result
        selected = projects.currentData()
        projects.clear()
        for project in values:
            projects.addItem(project.name, str(project.id))
        index = projects.findData(selected)
        if index >= 0:
            projects.setCurrentIndex(index)
        project_count.setText(str(len(values)))
        export_button.setEnabled(bool(values))
        migrate_button.setEnabled(bool(values))
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
            "WorkOrderImportedFromProjectionV1": "Сохранён снимок старого поручения",
            "LegacyWorkOrdersMigrationRecordedV1": "Миграция старых поручений",
        }
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

    def refresh_projects() -> None:
        tasks.start("Загрузка проектов и журнала…", load_projects, show_projects,
                    failure("Не удалось прочитать каталог проектов"))

    def scan() -> None:
        tasks.start("Проверка целостности…", service.scan_integrity, show_scan,
                    failure("Не удалось проверить целостность"))

    def show_scan(result):
        detail = (
            f"Проектов: {result.projects}\nСобытий: {result.events}\nBlob-объектов: {result.blobs}"
        )
        if result.valid:
            QMessageBox.information(host, "Проверка завершена", detail + "\n\nОшибок нет.")
        else:
            QMessageBox.warning(host, "Обнаружены ошибки", detail + "\n\n" + "\n".join(result.errors))

    def migrate_legacy() -> None:
        from PyQt6 import sip
        from kraken_hub.presentation.qt.legacy_work_order_dialog import LegacyWorkOrderMigrationDialog

        current = getattr(host, "_legacy_migration_dialog", None)
        if current is not None and not sip.isdeleted(current):
            current.show()
            current.raise_()
            return
        project_id = str(projects.currentData() or "")
        if not project_id:
            QMessageBox.information(host, "Миграция поручений", "Выберите проект.")
            return
        dialog = LegacyWorkOrderMigrationDialog(service, session.principal, project_id, host)
        host._legacy_migration_dialog = dialog
        dialog.migrated.connect(lambda _result: refresh_projects())
        dialog.show()

    def export_backup() -> None:
        if tasks.busy:
            return
        project_id = str(projects.currentData() or "")
        destination = QFileDialog.getExistingDirectory(host, "Каталог для резервной копии")
        if not project_id or not destination:
            return
        target = os.path.join(destination, f"kraken-backup-{project_id}-{uuid4().hex}")
        principal = session.principal
        tasks.start(
            f"Создание резервной копии…\n{target}",
            lambda: service.export_backup(project_id, target, principal=principal),
            lambda manifest: QMessageBox.information(host, "Резервная копия создана",
                f"Идентификатор: {manifest.bundle_id}\nСобытий: {manifest.event_count}\n{target}"),
            lambda message: QMessageBox.warning(host, "Не удалось создать резервную копию",
                f"{message}\nКаталог копии: {target}"),
        )

    def import_backup() -> None:
        if tasks.busy:
            return
        source = QFileDialog.getExistingDirectory(host, "Выберите резервную копию Kraken")
        if not source:
            return
        take_ownership = QMessageBox.question(
            host,
            "Владение проектом",
            "Если на этой рабочей станции нет доступного владельца проекта, принять владение?",
        ) == QMessageBox.StandardButton.Yes
        principal = session.principal
        tasks.start("Восстановление резервной копии…",
            lambda: service.import_backup(source, principal=principal, take_ownership=take_ownership),
            show_import, failure("Не удалось восстановить резервную копию"))

    def show_import(project):
        refresh_projects()
        QMessageBox.information(host, "Резервная копия восстановлена", f"Проект: {project.name}")

    def attach_project() -> None:
        if tasks.busy:
            return
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
        principal = session.principal
        tasks.start("Подключение проекта…",
            lambda: service.attach_project(source, principal=principal, take_ownership=take_ownership),
            show_attach, failure("Не удалось подключить проект"))

    def show_attach(project):
        refresh_projects()
        QMessageBox.information(host, "Проект подключён", project.name)

    scan_button.clicked.connect(scan)
    refresh_button.clicked.connect(refresh_projects)
    migrate_button.clicked.connect(migrate_legacy)
    roots_button.clicked.connect(
        lambda: _configure_workspace_roots(host, service, force=True)
    )
    export_button.clicked.connect(export_backup)
    import_button.clicked.connect(import_backup)
    attach_button.clicked.connect(attach_project)
    refresh_projects()
    return host
