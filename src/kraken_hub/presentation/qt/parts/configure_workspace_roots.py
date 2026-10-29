"""manager_app: configure workspace roots."""
from __future__ import annotations
from kraken_manager.infrastructure.workspace_files import validate_workspace_roots

from kraken_hub.presentation.qt.parts.layer_create_thread import _volume_key

def _configure_workspace_roots(parent, service, *, force: bool = False) -> tuple[str, str] | None:
    from kraken_core.settings import QSettings
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
