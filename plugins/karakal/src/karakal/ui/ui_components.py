"""Define small reusable Qt widgets used by the validation widget user interface."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

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
        confidence_expanded: bool,
        can_move_up: bool,
        can_move_down: bool,
        on_checked_changed,
        on_label_changed,
        on_confidence_folder,
        on_clear_confidence_folder,
        on_confidence_toggle,
        on_remove,
        on_move_up,
        on_move_down,
        checkbox_tooltip: str,
        confidence_placeholder: str,
        confidence_tooltip: str,
        confidence_select_tooltip: str,
        confidence_clear_tooltip: str,
        confidence_expand_tooltip: str,
        confidence_collapse_tooltip: str,
        remove_tooltip: str,
        move_up_tooltip: str,
        move_down_tooltip: str,
        name_tooltip: str = "",
        name_placeholder: str = "",
        confidence_label: str = "",
        original_label: str = "",
        original_display_text: str = "",
        original_path_text: str = "",
        on_original_folder=lambda *_args: None,
        on_clear_original_folder=lambda *_args: None,
        original_placeholder: str = "",
        original_select_tooltip: str = "",
        original_clear_tooltip: str = "",
        frames_per_row: int = 0,
        on_frames_per_row_changed=lambda _value: None,
        frames_per_row_label: str = "",
        frames_per_row_common_text: str = "",
        frames_per_row_tooltip: str = "",
        frames_per_row_maximum: int = 100_000,
    ) -> None:
        """Initialize FolderRowWidget."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        main_row = QWidget(self)
        main_layout = QHBoxLayout(main_row)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        self.checkbox = QCheckBox(self)
        self.checkbox.setChecked(checked)
        self.checkbox.setToolTip(checkbox_tooltip)
        self.checkbox.toggled.connect(on_checked_changed)
        main_layout.addWidget(self.checkbox)

        self.name_edit = QLineEdit(display_text, self)
        self.name_edit.setMinimumWidth(0)
        self.name_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.name_edit.setToolTip(name_tooltip or path_text)
        self.name_edit.setPlaceholderText(name_placeholder)
        self.name_edit.editingFinished.connect(lambda: on_label_changed(self.name_edit.text().strip()))
        main_layout.addWidget(self.name_edit, stretch=1)

        self.btn_confidence_toggle = QToolButton(self)
        self.btn_confidence_toggle.setAutoRaise(False)
        self.btn_confidence_toggle.setProperty('folderAction', True)
        self.btn_confidence_toggle.setCheckable(True)
        self.btn_confidence_toggle.setChecked(bool(confidence_expanded))
        self.btn_confidence_toggle.setText('-' if confidence_expanded else '+')
        self.btn_confidence_toggle.setToolTip(confidence_collapse_tooltip if confidence_expanded else confidence_expand_tooltip)
        self.btn_confidence_toggle.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_confidence_toggle.toggled.connect(lambda expanded: on_confidence_toggle(bool(expanded)))
        main_layout.addWidget(self.btn_confidence_toggle)

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

        self.btn_remove = QToolButton(self)
        self.btn_remove.setAutoRaise(False)
        self.btn_remove.setProperty('folderAction', True)
        self.btn_remove.setText('x')
        self.btn_remove.setToolTip(remove_tooltip)
        self.btn_remove.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_remove.clicked.connect(on_remove)
        main_layout.addWidget(self.btn_remove)

        # Layer settings under the name: label | value | choose | clear, aligned in one grid.
        details = QWidget(self)
        grid = QGridLayout(details)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(1, 1)
        self.confidence_edit, self.btn_confidence_select, self.btn_confidence_clear = self._add_folder_row(
            grid,
            0,
            label=confidence_label,
            display_text=confidence_display_text,
            path_text=confidence_path_text,
            placeholder=confidence_placeholder,
            tooltip=confidence_tooltip,
            select_tooltip=confidence_select_tooltip,
            clear_tooltip=confidence_clear_tooltip,
            on_select=on_confidence_folder,
            on_clear=on_clear_confidence_folder,
        )
        self.original_edit, self.btn_original_select, self.btn_original_clear = self._add_folder_row(
            grid,
            1,
            label=original_label,
            display_text=original_display_text,
            path_text=original_path_text,
            placeholder=original_placeholder,
            tooltip=original_path_text,
            select_tooltip=original_select_tooltip,
            clear_tooltip=original_clear_tooltip,
            on_select=on_original_folder,
            on_clear=on_clear_original_folder,
        )
        grid.addWidget(QLabel(frames_per_row_label, details), 2, 0)
        self.frames_per_row_spin = QSpinBox(details)
        self.frames_per_row_spin.setRange(0, int(frames_per_row_maximum))
        # 0: the layer has no width of its own and uses the common «Frames per row».
        self.frames_per_row_spin.setSpecialValueText(frames_per_row_common_text)
        self.frames_per_row_spin.setValue(max(0, int(frames_per_row)))
        self.frames_per_row_spin.setToolTip(frames_per_row_tooltip)
        self.frames_per_row_spin.setKeyboardTracking(False)
        self.frames_per_row_spin.valueChanged.connect(lambda value: on_frames_per_row_changed(int(value)))
        grid.addWidget(self.frames_per_row_spin, 2, 1, 1, 3, Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(main_row)
        layout.addWidget(details)
        details.setVisible(bool(confidence_expanded))
        self.details = details

    def _add_folder_row(
        self,
        grid: QGridLayout,
        row: int,
        *,
        label: str,
        display_text: str,
        path_text: str,
        placeholder: str,
        tooltip: str,
        select_tooltip: str,
        clear_tooltip: str,
        on_select,
        on_clear,
    ) -> tuple[QLineEdit, QToolButton, QToolButton]:
        """One optional folder of the layer: label, read-only path, choose and clear buttons."""

        parent = grid.parentWidget()
        grid.addWidget(QLabel(label, parent), row, 0)

        edit = QLineEdit(display_text, parent)
        edit.setReadOnly(True)
        edit.setMinimumWidth(0)
        edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        edit.setPlaceholderText(placeholder)
        edit.setToolTip(tooltip if path_text else placeholder)
        grid.addWidget(edit, row, 1)

        btn_select = QToolButton(parent)
        btn_select.setAutoRaise(False)
        btn_select.setProperty('folderAction', True)
        btn_select.setText('...')
        btn_select.setToolTip(select_tooltip)
        btn_select.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        btn_select.clicked.connect(on_select)
        grid.addWidget(btn_select, row, 2)

        btn_clear = QToolButton(parent)
        btn_clear.setAutoRaise(False)
        btn_clear.setProperty('folderAction', True)
        btn_clear.setText('x')
        btn_clear.setToolTip(clear_tooltip)
        btn_clear.setEnabled(bool(path_text))
        btn_clear.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        btn_clear.clicked.connect(on_clear)
        grid.addWidget(btn_clear, row, 3)
        return edit, btn_select, btn_clear
