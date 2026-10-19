"""manager_app: my work panel."""
from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
from uuid import uuid4
from kraken_manager.domain.identity import Permission
from kraken_hub.composition import DesktopSession, EmbeddedProjectService

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

    def delete_job(job) -> None:
        if (
            QMessageBox.question(
                host, "Удалить задание", "Удалить задание из списка и pipeline? Выполнение будет отменено."
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        try:
            controller.delete_agent_job(job)
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось удалить задание", str(exc))
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
            if Permission.RUN_PLUGIN in permissions:
                delete_button = QPushButton("Удалить", action_host)
                delete_button.clicked.connect(lambda _checked=False, value=job: delete_job(value))
                action_layout.addWidget(delete_button)
            table.setCellWidget(offset, 7, action_host)

    refresh_button.clicked.connect(refresh)
    controller.my_work_refresh = refresh
    refresh()
    return host

if TYPE_CHECKING:
    from kraken_hub.manager_app import DesktopController
