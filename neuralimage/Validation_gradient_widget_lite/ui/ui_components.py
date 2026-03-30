"""Define small reusable Qt widgets used by the validation widget user interface."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit, QSizePolicy, QToolButton, QWidget

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
        can_move_up: bool,
        can_move_down: bool,
        on_checked_changed,
        on_label_changed,
        on_remove,
        on_move_up,
        on_move_down,
        checkbox_tooltip: str,
        remove_tooltip: str,
        move_up_tooltip: str,
        move_down_tooltip: str,
    ) -> None:
        """Initialize FolderRowWidget."""
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)

        self.checkbox = QCheckBox(self)
        self.checkbox.setChecked(checked)
        self.checkbox.setToolTip(checkbox_tooltip)
        self.checkbox.toggled.connect(on_checked_changed)
        layout.addWidget(self.checkbox)

        self.name_edit = QLineEdit(display_text, self)
        self.name_edit.setMinimumWidth(0)
        self.name_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.name_edit.setToolTip(path_text)
        self.name_edit.editingFinished.connect(lambda: on_label_changed(self.name_edit.text().strip()))
        layout.addWidget(self.name_edit, stretch=1)

        self.btn_remove = QToolButton(self)
        self.btn_remove.setAutoRaise(False)
        self.btn_remove.setProperty('folderAction', True)
        self.btn_remove.setText('x')
        self.btn_remove.setToolTip(remove_tooltip)
        self.btn_remove.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_remove.clicked.connect(on_remove)
        layout.addWidget(self.btn_remove)

        self.btn_up = QToolButton(self)
        self.btn_up.setAutoRaise(False)
        self.btn_up.setProperty('folderAction', True)
        self.btn_up.setText('^')
        self.btn_up.setToolTip(move_up_tooltip)
        self.btn_up.setEnabled(can_move_up)
        self.btn_up.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_up.clicked.connect(on_move_up)
        layout.addWidget(self.btn_up)

        self.btn_down = QToolButton(self)
        self.btn_down.setAutoRaise(False)
        self.btn_down.setProperty('folderAction', True)
        self.btn_down.setText('v')
        self.btn_down.setToolTip(move_down_tooltip)
        self.btn_down.setEnabled(can_move_down)
        self.btn_down.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_down.clicked.connect(on_move_down)
        layout.addWidget(self.btn_down)


