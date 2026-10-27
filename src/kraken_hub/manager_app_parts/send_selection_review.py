"""manager_app: issue performer work and open it in the matching program."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def _issued_work_message(order) -> str:
    summary = f"{order.kind.title}\nКадров: {order.frame_count}"
    if order.placement.value == "server" or order.transport == "http":
        return summary + "\nПапка на сервере создана автоматически. Доступ к работе — по HTTP."
    return summary + f"\nПапка: {order.destination}"


_KIND_CHOICES = (
    ("Исправление векторов", "vector_review"),
    ("Векторизация", "vectorize"),
    ("Обучение модели", "train"),
    ("Сшивка", "stitch"),
)
_PLACEMENT_CHOICES = (
    ("Внешний накопитель", "external"),
    ("На сервере", "server"),
)


class SendSelectionReviewMixin:
    def send_selection_for_review(self) -> None:
        self.issue_work_dialog(preset_kind="vector_review")

    def issue_work_dialog(self, *, preset_kind: str = "vector_review", layer_id: str = "") -> None:
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
        selected_layer = layer_id or str(getattr(workspace, "_selected_layer_id", "") if workspace else "")
        if workspace is None or not project_id or not selected_layer:
            self._error("Сначала откройте проект и выберите слой.")
            return
        performer_reader = getattr(self.service, "list_project_performers", None)
        performers = performer_reader(project_id) if callable(performer_reader) else self.service.list_performers()
        if not performers:
            self._error("Сначала добавьте исполнителя в разделе «Исполнители».")
            return

        dialog = QDialog(self.shell)
        dialog.setWindowTitle("Выдать работу")
        root = QVBoxLayout(dialog)
        form = QFormLayout()
        kind_box = QComboBox(dialog)
        for label, value in _KIND_CHOICES:
            kind_box.addItem(label, value)
        preset_index = kind_box.findData(preset_kind)
        if preset_index >= 0:
            kind_box.setCurrentIndex(preset_index)
        assignee = QComboBox(dialog)
        for performer in performers:
            assignee.addItem(performer.name, str(performer.id))
        instructions = QLineEdit(dialog)
        instructions.setPlaceholderText("Что нужно сделать")
        due_enabled = QCheckBox("Установить срок", dialog)
        due = QDateTimeEdit(QDateTime.currentDateTime().addDays(7), dialog)
        due.setCalendarPopup(True)
        due.setDisplayFormat("dd.MM.yyyy HH:mm")
        due.setEnabled(False)
        due_enabled.toggled.connect(due.setEnabled)
        placement = QComboBox(dialog)
        for label, value in _PLACEMENT_CHOICES:
            placement.addItem(label, value)
        directory_row = QWidget(dialog)
        directory_layout = QFormLayout(directory_row)
        directory_layout.setContentsMargins(0, 0, 0, 0)
        destination_root = QLineEdit(dialog)
        destination_root.setReadOnly(True)
        choose = QPushButton("Выбрать…", dialog)

        def choose_directory() -> None:
            selected = QFileDialog.getExistingDirectory(dialog, "Каталог на накопителе", destination_root.text())
            if selected:
                destination_root.setText(selected)

        choose.clicked.connect(choose_directory)
        directory_layout.addRow(destination_root, choose)

        def toggle_directory() -> None:
            external = placement.currentData() == "external"
            directory_row.setVisible(external)
            label = form.labelForField(directory_row)
            if label is not None:
                label.setVisible(external)

        placement.currentIndexChanged.connect(lambda _index: toggle_directory())
        launch_now = QCheckBox("Запустить сейчас", dialog)
        form.addRow("Работа", kind_box)
        form.addRow("Исполнитель", assignee)
        form.addRow("Инструкции", instructions)
        form.addRow(due_enabled, due)
        form.addRow("Куда поместить", placement)
        form.addRow("Каталог накопителя", directory_row)
        form.addRow(launch_now)
        root.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Выдать")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        root.addWidget(buttons)
        toggle_directory()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if placement.currentData() == "external" and not destination_root.text():
            QMessageBox.warning(dialog, "Каталог не выбран", "Выберите каталог на внешнем накопителе.")
            return
        image_id = str(workspace.image_representation_combo.currentData() or "")
        vector_id = str(workspace.vector_representation_combo.currentData() or "")
        coordinates = workspace.matrix_view.selected_coordinates(maximum=self.service.profile.capabilities.max_frames)
        if not coordinates and kind_box.currentData() != "train":
            coordinates = tuple(
                (cell.x, cell.y)
                for cell in self.service.frame_cells(project_id, selected_layer, image_id)
                if image_id
            )
        try:
            deadline = due.dateTime().toPyDateTime().astimezone(UTC) if due_enabled.isChecked() else None
            order = self.service.issue_work(
                principal=self.session.principal,
                project_id=project_id,
                layer_id=selected_layer,
                kind=str(kind_box.currentData()),
                placement=str(placement.currentData()),
                assignee_id=str(assignee.currentData()),
                coordinates=tuple(coordinates),
                image_representation_id=image_id,
                vector_representation_id=vector_id,
                instructions=instructions.text(),
                due_at=deadline,
                destination_parent=None if placement.currentData() == "server" else destination_root.text(),
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            QMessageBox.warning(dialog, "Не удалось выдать работу", str(exc))
            return
        refresh = getattr(self, "my_work_refresh", None)
        if launch_now.isChecked():
            self.open_issued_work(order)
        elif refresh is not None:
            refresh()
        QMessageBox.information(self.shell, "Работа выдана", _issued_work_message(order))

    def open_issued_work(self, order) -> None:
        from PyQt6.QtWidgets import QMessageBox

        destination = Path(order.destination)
        if order.transport == "http" and order.kind.value in {"vectorize", "train"}:
            if str(self._project_id) != str(order.project_id):
                self._error("Откройте проект этого задания, чтобы запустить его по HTTP.")
                return
            parameters = {}
            if order.dataset_version_id:
                parameters["dataset_version_id"] = order.dataset_version_id
            try:
                job = self._submit_agent_action(
                    layer_id=str(order.layer_id),
                    source_representation_id=order.image_representation_id or order.vector_representation_id,
                    capability=order.capability,
                    parameters=parameters,
                )
                self.service.mark_work_running(order, job_id=str(job.id), idempotency_key=str(uuid4()), principal=self.session.principal)
            except Exception as exc:
                self._error(str(exc))
                return
            refresh = getattr(self, "my_work_refresh", None)
            if refresh is not None:
                refresh()
            return
        if order.launch_mode == "contour-job" and (destination / "job.json").is_file():
            result_manifest = destination / "result.json"
            if result_manifest.exists():
                self._error("Результат этой папки уже записан.")
                return
            self._launch_managed_plugin(
                "contour",
                arguments=(
                    "--kraken-job-manifest",
                    str(destination / "job.json"),
                    "--kraken-result-manifest",
                    str(result_manifest),
                    "--kraken-staging-root",
                    str(destination),
                ),
            )
            return
        if order.placement.value == "server" or order.transport == "http":
            QMessageBox.information(self.shell, order.kind.title, _issued_work_message(order))
            return
        if order.launch_mode == "neural-dir":
            QMessageBox.information(
                self.shell,
                "Обучение",
                f"Выборка для NeuralImage:\n{destination / 'dataset'}",
            )
            return
        QMessageBox.information(
            self.shell,
            order.kind.title,
            f"Файлы работы:\n{destination}",
        )

    def cancel_issued_work(self, order) -> None:
        from PyQt6.QtWidgets import QMessageBox

        if (
            QMessageBox.question(self.shell, "Отменить работу", "Отменить выданное задание?")
            != QMessageBox.StandardButton.Yes
        ):
            return
        try:
            self.service.cancel_work_order(order, idempotency_key=str(uuid4()), principal=self.session.principal)
        except Exception as exc:
            self._error(str(exc))
            return
        refresh = getattr(self, "my_work_refresh", None)
        if refresh is not None:
            refresh()

    def complete_issued_work(self, order) -> None:
        try:
            self.service.complete_work_order(order, idempotency_key=str(uuid4()), principal=self.session.principal)
        except Exception as exc:
            self._error(str(exc))
            return
        refresh = getattr(self, "my_work_refresh", None)
        if refresh is not None:
            refresh()
