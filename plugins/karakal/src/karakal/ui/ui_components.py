"""Define small reusable Qt widgets used by the validation widget user interface."""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .toolbar_icons import toolbar_icon
from .ui_constants import FOLDER_BUTTON_SIZE


class _NoWheelCombo(QComboBox):
    """The layer list scrolls under the wheel; a combo inside a row must not catch it."""

    def wheelEvent(self, event) -> None:
        event.ignore()


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
        on_checked_changed,
        on_label_changed,
        on_confidence_folder,
        on_clear_confidence_folder,
        on_confidence_toggle,
        on_remove,
        checkbox_tooltip: str,
        confidence_placeholder: str,
        confidence_tooltip: str,
        confidence_select_tooltip: str,
        confidence_clear_tooltip: str,
        confidence_expand_tooltip: str,
        confidence_collapse_tooltip: str,
        remove_tooltip: str,
        grip_tooltip: str = "",
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
        markup_label: str = "",
        markup_display_text: str = "",
        markup_path_text: str = "",
        on_markup_folder=lambda *_args: None,
        on_clear_markup_folder=lambda *_args: None,
        markup_placeholder: str = "",
        markup_tooltip: str = "",
        markup_select_tooltip: str = "",
        markup_clear_tooltip: str = "",
        frames_per_row: int = 0,
        on_frames_per_row_changed=lambda _value: None,
        frames_per_row_label: str = "",
        frames_per_row_common_text: str = "",
        frames_per_row_tooltip: str = "",
        frames_per_row_maximum: int = 100_000,
        on_activated=lambda: None,
        cell_class: str = "",
        on_cell_class_changed=lambda _value: None,
        cell_class_label: str = "",
        cell_class_options: tuple[tuple[str, str], ...] = (),
        cell_class_tooltip: str = "",
        cell_class_badge_tooltips: dict[str, str] | None = None,
        cell_bank_label: str = "",
        cell_bank_button_text: str = "",
        cell_bank_tooltip: str = "",
        on_open_cell_bank=lambda *_args: None,
    ) -> None:
        """Initialize FolderRowWidget."""
        super().__init__(parent)
        # The selection frame is drawn by the row itself: the list's own highlight is hidden
        # under the row widget and only peeked out at the edges.
        self.setObjectName("folderRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("selected", False)
        self._on_activated = on_activated
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        main_row = QWidget(self)
        main_row.setObjectName("folderRowPart")
        main_layout = QHBoxLayout(main_row)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # Drag handle: the list moves the row (it ignores the press, so the list gets it).
        self.grip = QLabel(self)
        self.grip.setPixmap(toolbar_icon("grip").pixmap(16, 16))
        self.grip.setFixedWidth(14)
        self.grip.setToolTip(grip_tooltip)
        self.grip.setCursor(Qt.CursorShape.OpenHandCursor)
        self.grip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        main_layout.addWidget(self.grip)

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

        # «1» / «0»: which cells the network paints, visible without opening the settings.
        self.cell_class_badge = QLabel(self)
        self.cell_class_badge.setObjectName("cellClassBadge")
        self.cell_class_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cell_class_badge.setFixedSize(18, 18)
        main_layout.addWidget(self.cell_class_badge)
        self._cell_class_badge_tooltips = dict(cell_class_badge_tooltips or {})

        self.btn_confidence_toggle = QToolButton(self)
        self.btn_confidence_toggle.setAutoRaise(False)
        self.btn_confidence_toggle.setProperty('folderAction', True)
        self.btn_confidence_toggle.setCheckable(True)
        self.btn_confidence_toggle.setChecked(bool(confidence_expanded))
        self.btn_confidence_toggle.setIcon(toolbar_icon("settings"))
        self.btn_confidence_toggle.setToolTip(confidence_collapse_tooltip if confidence_expanded else confidence_expand_tooltip)
        self.btn_confidence_toggle.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_confidence_toggle.toggled.connect(lambda expanded: on_confidence_toggle(bool(expanded)))
        main_layout.addWidget(self.btn_confidence_toggle)

        self.btn_remove = QToolButton(self)
        self.btn_remove.setAutoRaise(False)
        self.btn_remove.setProperty('folderAction', True)
        self.btn_remove.setIcon(toolbar_icon("trash"))
        self.btn_remove.setToolTip(remove_tooltip)
        self.btn_remove.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        self.btn_remove.clicked.connect(on_remove)
        main_layout.addWidget(self.btn_remove)

        # Layer settings under the name: label | value | choose | clear, aligned in one grid.
        details = QWidget(self)
        details.setObjectName("folderRowPart")
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
        # Operator masks of this layer: its bank of normal cells is built from them.
        self.markup_edit, self.btn_markup_select, self.btn_markup_clear = self._add_folder_row(
            grid,
            2,
            label=markup_label,
            display_text=markup_display_text,
            path_text=markup_path_text,
            placeholder=markup_placeholder,
            tooltip=markup_tooltip,
            select_tooltip=markup_select_tooltip,
            clear_tooltip=markup_clear_tooltip,
            on_select=on_markup_folder,
            on_clear=on_clear_markup_folder,
        )
        grid.addWidget(QLabel(frames_per_row_label, details), 3, 0)
        self.frames_per_row_spin = QSpinBox(details)
        self.frames_per_row_spin.setRange(0, int(frames_per_row_maximum))
        # 0: the layer has no width of its own and uses the common «Frames per row».
        self.frames_per_row_spin.setSpecialValueText(frames_per_row_common_text)
        self.frames_per_row_spin.setValue(max(0, int(frames_per_row)))
        self.frames_per_row_spin.setToolTip(frames_per_row_tooltip)
        self.frames_per_row_spin.setKeyboardTracking(False)
        self.frames_per_row_spin.valueChanged.connect(lambda value: on_frames_per_row_changed(int(value)))
        grid.addWidget(self.frames_per_row_spin, 3, 1, 1, 3, Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel(cell_class_label, details), 4, 0)
        self.cell_class_combo = _NoWheelCombo(details)
        for value, text in cell_class_options:
            self.cell_class_combo.addItem(text, value)
        index = self.cell_class_combo.findData(str(cell_class or ""))
        self.cell_class_combo.setCurrentIndex(max(0, index))
        self.cell_class_combo.setToolTip(cell_class_tooltip)
        self.cell_class_combo.currentIndexChanged.connect(
            lambda _index: on_cell_class_changed(str(self.cell_class_combo.currentData() or ""))
        )
        grid.addWidget(self.cell_class_combo, 4, 1, 1, 3, Qt.AlignmentFlag.AlignLeft)
        if cell_bank_label:
            grid.addWidget(QLabel(cell_bank_label, details), 5, 0)
            self.btn_cell_bank = QPushButton(cell_bank_button_text, details)
            self.btn_cell_bank.setToolTip(cell_bank_tooltip)
            self.btn_cell_bank.clicked.connect(on_open_cell_bank)
            grid.addWidget(self.btn_cell_bank, 5, 1, 1, 3, Qt.AlignmentFlag.AlignLeft)
        self._show_cell_class_badge(str(cell_class or ""))

        layout.addWidget(main_row)
        layout.addWidget(details)
        details.setVisible(bool(confidence_expanded))
        self.details = details
        # A press anywhere in the row (name, checkbox, buttons) selects the layer.
        for child in self.findChildren(QWidget):
            child.installEventFilter(self)

    def set_cell_class(self, value: str) -> None:
        """Show a class set from outside the row (settings, another control) without echoing it back."""

        index = self.cell_class_combo.findData(str(value or ""))
        if index >= 0 and index != self.cell_class_combo.currentIndex():
            previous = self.cell_class_combo.blockSignals(True)
            self.cell_class_combo.setCurrentIndex(index)
            self.cell_class_combo.blockSignals(previous)
        self._show_cell_class_badge(str(value or ""))

    def _show_cell_class_badge(self, value: str) -> None:
        text = {"ones": "1", "zeros": "0"}.get(value, "")
        self.cell_class_badge.setText(text)
        self.cell_class_badge.setToolTip(self._cell_class_badge_tooltips.get(value, ""))
        self.cell_class_badge.setVisible(bool(text))

    def set_selected(self, selected: bool) -> None:
        if bool(self.property("selected")) == bool(selected):
            return
        self.setProperty("selected", bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress:
            self._on_activated()
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event) -> None:
        self._on_activated()
        super().mousePressEvent(event)

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
        btn_clear.setIcon(toolbar_icon("clear"))
        btn_clear.setToolTip(clear_tooltip)
        btn_clear.setEnabled(bool(path_text))
        btn_clear.setFixedSize(FOLDER_BUTTON_SIZE, FOLDER_BUTTON_SIZE)
        btn_clear.clicked.connect(on_clear)
        grid.addWidget(btn_clear, row, 3)
        return edit, btn_select, btn_clear
