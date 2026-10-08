"""Settings → File suffixes: how mask and confidence files of one frame are named."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..core.frame_naming import DEFAULT_FRAME_NAMING, FrameNaming, validate_frame_naming


class FileNamingDialog(QDialog):
    def __init__(self, translator, naming: FrameNaming, parent=None) -> None:
        super().__init__(parent)
        self._t = translator
        self.setWindowTitle(translator("file_naming.window"))
        root = QVBoxLayout(self)
        intro = QLabel(translator("file_naming.intro"), self)
        intro.setWordWrap(True)
        root.addWidget(intro)
        form = QFormLayout()
        self.mask_edit = QLineEdit(naming.mask_suffix, self)
        self.mask_edit.setPlaceholderText(translator("file_naming.no_suffix"))
        self.confidence_edit = QLineEdit(naming.confidence_suffix, self)
        self.confidence_edit.setPlaceholderText(translator("file_naming.no_suffix"))
        form.addRow(translator("file_naming.mask_suffix"), self.mask_edit)
        form.addRow(translator("file_naming.confidence_suffix"), self.confidence_edit)
        root.addLayout(form)
        self.example = QLabel("", self)
        self.example.setWordWrap(True)
        root.addWidget(self.example)
        note = QLabel(translator("file_naming.apply_note"), self)
        note.setWordWrap(True)
        root.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        reset = QPushButton(translator("file_naming.defaults"), self)
        buttons.addButton(reset, QDialogButtonBox.ButtonRole.ResetRole)
        reset.clicked.connect(self._reset)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.mask_edit.textChanged.connect(self._refresh_example)
        self.confidence_edit.textChanged.connect(self._refresh_example)
        self._refresh_example()
        self.resize(460, 0)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        # Fields get their styled height only when shown: let the dialog grow to fit them.
        self.resize(max(self.width(), 460), max(self.height(), self.sizeHint().height()))

    def naming(self) -> FrameNaming:
        return FrameNaming(mask_suffix=self.mask_edit.text(), confidence_suffix=self.confidence_edit.text()).normalized()

    def _refresh_example(self, *_args) -> None:
        naming = self.naming()
        self.example.setText(
            self._t(
                "file_naming.example",
                mask=f"TEST_01800{naming.mask_suffix}.jpg",
                confidence=f"TEST_01800{naming.confidence_suffix}.jpg",
            )
        )

    def _reset(self) -> None:
        self.mask_edit.setText(DEFAULT_FRAME_NAMING.mask_suffix)
        self.confidence_edit.setText(DEFAULT_FRAME_NAMING.confidence_suffix)

    def _accept(self) -> None:
        problem = validate_frame_naming(self.naming())
        if problem:
            QMessageBox.warning(self, self.windowTitle(), self._t(f"file_naming.invalid_{problem}"))
            return
        self.accept()
