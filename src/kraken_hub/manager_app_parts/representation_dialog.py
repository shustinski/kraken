"""manager_app: representation dialog."""
from __future__ import annotations
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.presentation.qt.widgets import ClickableLabel

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
