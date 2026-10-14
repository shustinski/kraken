"""manager_app: send selection review."""
from __future__ import annotations
from pathlib import Path
from uuid import uuid4


class SendSelectionReviewMixin:
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
            maximum=self.service.profile.capabilities.max_frames
        )
        if not coordinates:
            self._error("Выберите хотя бы один кадр.")
            return
        image_id = str(workspace.image_representation_combo.currentData() or "")
        vector_id = str(workspace.vector_representation_combo.currentData() or "")
        if not image_id or not vector_id:
            self._error("Выберите связанные репрезентации изображения и CIF.")
            return
        performer_reader = getattr(
            self.service, "list_project_performers", None
        )
        performers = (
            performer_reader(project_id)
            if callable(performer_reader)
            else self.service.list_performers()
        )
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
