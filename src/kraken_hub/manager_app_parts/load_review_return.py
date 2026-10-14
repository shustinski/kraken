"""manager_app: load review return."""
from __future__ import annotations
from uuid import uuid4


class LoadReviewReturnMixin:
    def load_review_return(self) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QFileDialog,
            QHeaderView,
            QLabel,
            QMessageBox,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
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
        preview = QDialog(self.shell)
        preview.setWindowTitle("Предварительная проверка")
        preview.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        preview.setMinimumSize(560, 320)
        preview.resize(760, 500)
        preview_layout = QVBoxLayout(preview)
        counts: dict[str, int] = {}
        for item in plan.report.items:
            key = item.status.value
            counts[key] = counts.get(key, 0) + 1
        summary = QLabel(
            "\n".join(f"{labels.get(key, key)}: {value}" for key, value in counts.items())
            or "Файлы не найдены.",
            preview,
        )
        preview_layout.addWidget(summary)
        details = QTableWidget(len(plan.report.items), 2, preview)
        details.setHorizontalHeaderLabels(("Файл", "Результат"))
        for row, item in enumerate(plan.report.items):
            details.setItem(row, 0, QTableWidgetItem(item.relative_path))
            details.setItem(
                row,
                1,
                QTableWidgetItem(labels.get(item.status.value, item.status.value)),
            )
        details.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        details.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        preview_layout.addWidget(details, 1)
        buttons = QDialogButtonBox(preview)
        import_button = buttons.addButton(
            "Загрузить изменённые файлы",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        buttons.addButton("Отмена", QDialogButtonBox.ButtonRole.RejectRole)
        import_button.setEnabled(plan.report.can_commit)
        buttons.accepted.connect(preview.accept)
        buttons.rejected.connect(preview.reject)
        preview_layout.addWidget(buttons)
        if preview.exec() != QDialog.DialogCode.Accepted:
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
            self._refresh_current_matrix()
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
        self._refresh_current_matrix()
