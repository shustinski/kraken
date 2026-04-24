"""Define small reusable Qt widgets used by the validation widget user interface."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from .ui_constants import FOLDER_BUTTON_SIZE


class FolderRowWidget(QWidget):
    """Render one editable row inside the folder manager list."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        path_text: str,
        display_text: str,
        checked: bool,
        confidence_display_text: str,
        confidence_path_text: str,
        can_move_up: bool,
        can_move_down: bool,
        on_checked_changed,
        on_label_changed,
        on_confidence_folder,
        on_clear_confidence_folder,
        on_remove,
        on_move_up,
        on_move_down,
        checkbox_tooltip: str,
        confidence_placeholder: str,
        confidence_tooltip: str,
        confidence_select_tooltip: str,
        confidence_clear_tooltip: str,
        remove_tooltip: str,
        move_up_tooltip: str,
        move_down_tooltip: str,
    ) -> None:
        """Initialize FolderRowWidget."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)

        main_row = QWidget(self)
        main_layout = QHBoxLayout(main_row)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)

        self.checkbox = QCheckBox(self)
        self.checkbox.setChecked(checked)
        self.checkbox.setToolTip(checkbox_tooltip)
        self.checkbox.toggled.connect(on_checked_changed)
        main_layout.addWidget(self.checkbox)

        self.name_edit = QLineEdit(display_text, self)
        self.name_edit.setMinimumWidth(0)
        self.name_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.name_edit.setToolTip(path_text)
        self.name_edit.editingFinished.connect(lambda: on_label_changed(self.name_edit.text().strip()))
        main_layout.addWidget(self.name_edit, stretch=1)

        self.btn_remove = QToolButton(self)
        self.btn_remove.setAutoRaise(False)
        self.btn_remove.setProperty('folderAction', True)
        self.btn_remove.setText('x')
        self.btn_remove.setToolTip(remove_tooltip)
        self.btn_remove.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_remove.clicked.connect(on_remove)
        main_layout.addWidget(self.btn_remove)

        self.btn_up = QToolButton(self)
        self.btn_up.setAutoRaise(False)
        self.btn_up.setProperty('folderAction', True)
        self.btn_up.setText('^')
        self.btn_up.setToolTip(move_up_tooltip)
        self.btn_up.setEnabled(can_move_up)
        self.btn_up.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_up.clicked.connect(on_move_up)
        main_layout.addWidget(self.btn_up)

        self.btn_down = QToolButton(self)
        self.btn_down.setAutoRaise(False)
        self.btn_down.setProperty('folderAction', True)
        self.btn_down.setText('v')
        self.btn_down.setToolTip(move_down_tooltip)
        self.btn_down.setEnabled(can_move_down)
        self.btn_down.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_down.clicked.connect(on_move_down)
        main_layout.addWidget(self.btn_down)

        confidence_row = QWidget(self)
        confidence_layout = QHBoxLayout(confidence_row)
        confidence_layout.setContentsMargins(0, 0, 0, 0)
        confidence_layout.setSpacing(6)

        self.confidence_edit = QLineEdit(confidence_display_text, self)
        self.confidence_edit.setReadOnly(True)
        self.confidence_edit.setMinimumWidth(0)
        self.confidence_edit.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.confidence_edit.setPlaceholderText(confidence_placeholder)
        self.confidence_edit.setToolTip(confidence_tooltip if confidence_path_text else confidence_placeholder)
        confidence_layout.addWidget(self.confidence_edit, stretch=1)

        self.btn_confidence_select = QToolButton(self)
        self.btn_confidence_select.setAutoRaise(False)
        self.btn_confidence_select.setProperty('folderAction', True)
        self.btn_confidence_select.setText('...')
        self.btn_confidence_select.setToolTip(confidence_select_tooltip)
        self.btn_confidence_select.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_confidence_select.clicked.connect(on_confidence_folder)
        confidence_layout.addWidget(self.btn_confidence_select)

        self.btn_confidence_clear = QToolButton(self)
        self.btn_confidence_clear.setAutoRaise(False)
        self.btn_confidence_clear.setProperty('folderAction', True)
        self.btn_confidence_clear.setText('x')
        self.btn_confidence_clear.setToolTip(confidence_clear_tooltip)
        self.btn_confidence_clear.setEnabled(bool(confidence_path_text))
        self.btn_confidence_clear.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_confidence_clear.clicked.connect(on_clear_confidence_folder)
        confidence_layout.addWidget(self.btn_confidence_clear)

        layout.addWidget(main_row)
        layout.addWidget(confidence_row)


