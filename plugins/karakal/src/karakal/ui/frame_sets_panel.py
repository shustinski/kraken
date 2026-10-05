"""Panel that opens under the matrix gradient: drop rules, frame sets and export.

The panel only shows state and reports what the user did; the presenter side
(``app/frame_sets_controller.py``) owns the rules and sets. The badge above the
matrix tells which set the matrix shows while the panel is closed.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.frame_set_export import FRAME_ITEMS
from .i18n import Translator

SLIDER_STEPS = 1000

_COLUMN_STYLE = "QLabel[role='heading'] { font-weight: 700; color: #cfd8e3; }"
_MUTED = "color: #9eacbd;"
_WARN = "color: #ff9b7a; font-weight: 600;"


def _heading(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setProperty("role", "heading")
    label.setStyleSheet("font-weight: 700;")
    return label


def _count_grid() -> QGridLayout:
    """Rows of name | number | ✕; the numbers stand in one column next to the names."""

    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(2)
    grid.setColumnStretch(3, 1)
    return grid


def _count_label(count: int, parent: QWidget) -> QLabel:
    label = QLabel(str(int(count)), parent)
    label.setStyleSheet(_MUTED + " font-family: Consolas, monospace;")
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return label


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            # Hide at once: the old row must not show until the deferred delete runs.
            widget.hide()
            widget.deleteLater()
        elif item.layout() is not None:
            _clear_layout(item.layout())


class FrameSetsSlider(QWidget):
    """Threshold slider placed right under the gradient bar."""

    valueChanged = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setValue(0)
        self.slider.setSingleStep(5)
        self.slider.setPageStep(50)
        layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self.valueChanged.emit)

    def value(self) -> int:
        return int(self.slider.value())

    def set_value(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(value))
        self.slider.blockSignals(False)


class FrameSetsViewBadge(QFrame):
    """Strip above the matrix: which set is shown, with a way back to the whole run."""

    resetRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._i18n = Translator()
        self.setObjectName("frameSetsViewBadge")
        self.setStyleSheet(
            "#frameSetsViewBadge { background: #1d3550; border: 1px solid #3d6a99; border-radius: 4px; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 4, 3)
        layout.setSpacing(8)
        self.text_label = QLabel(self)
        self.text_label.setStyleSheet("color: #e3eefa;")
        self.reset_button = QToolButton(self)
        self.reset_button.setAutoRaise(True)
        self.reset_button.setText("✕ " + self._i18n.tr("frame_sets.badge.reset"))
        layout.addWidget(self.text_label, stretch=1)
        layout.addWidget(self.reset_button)
        self.reset_button.clicked.connect(self.resetRequested.emit)
        self.setVisible(False)

    def show_view(self, name: str, count: int, total: int) -> None:
        self.text_label.setText(
            self._i18n.tr("frame_sets.badge.text", name=name, count=int(count), total=int(total))
        )
        self.setVisible(True)

    def clear(self) -> None:
        self.setVisible(False)


class FrameSetsPanel(QWidget):
    """Drop controls, frame sets and export steps."""

    applyRequested = pyqtSignal()
    invertRequested = pyqtSignal()
    ruleToggled = pyqtSignal(int, bool)
    ruleRemoved = pyqtSignal(int)
    manualCleared = pyqtSignal()
    viewChosen = pyqtSignal(str)
    setRemoved = pyqtSignal(str)
    addSelectedRequested = pyqtSignal(str)
    selectShownRequested = pyqtSignal()
    clearSelectionRequested = pyqtSignal()
    exportOpened = pyqtSignal()
    exportChoicesChanged = pyqtSignal()
    exportBrowseRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    openFolderRequested = pyqtSignal()
    showReportRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._i18n = Translator()
        self._t = self._i18n.tr
        self.setStyleSheet(_COLUMN_STYLE)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 0)
        root.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.threshold_label = QLabel(self)
        self.threshold_label.setWordWrap(True)
        self.threshold_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        controls.addWidget(self.threshold_label, stretch=1)
        self.apply_button = QPushButton(self)
        self.apply_button.setStyleSheet("font-weight: 700;")
        self.invert_button = QPushButton(self._t("frame_sets.invert"), self)
        self.invert_button.setCheckable(True)
        controls.addWidget(self.apply_button)
        controls.addWidget(self.invert_button)
        root.addLayout(controls)
        self.inverted_label = QLabel(self._t("frame_sets.inverted_note"), self)
        self.inverted_label.setStyleSheet(_WARN)
        self.inverted_label.setVisible(False)
        root.addWidget(self.inverted_label)

        self.pages = QStackedWidget(self)
        self.pages.addWidget(self._build_main_page())
        self.pages.addWidget(self._build_export_page())
        root.addWidget(self.pages)

        self.apply_button.clicked.connect(self.applyRequested.emit)
        self.invert_button.clicked.connect(lambda _checked: self.invertRequested.emit())
        self.set_pending_count(0)

    # Layout ------------------------------------------------------------------

    def _column(self, parent: QWidget) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame(parent)
        frame.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(5)
        return frame, layout

    def _build_main_page(self) -> QWidget:
        page = QWidget(self)
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)

        rules_column, rules_layout = self._column(page)
        rules_layout.addWidget(_heading(self._t("frame_sets.rules.title"), rules_column))
        self.rules_box = _count_grid()
        rules_layout.addLayout(self.rules_box)
        rules_layout.addStretch(1)

        sets_column, sets_layout = self._column(page)
        sets_layout.addWidget(_heading(self._t("frame_sets.sets.title"), sets_column))
        self.sets_box = _count_grid()
        sets_layout.addLayout(self.sets_box)
        self.view_note = QLabel(sets_column)
        self.view_note.setWordWrap(True)
        self.view_note.setStyleSheet(_MUTED)
        sets_layout.addWidget(self.view_note)
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)
        target_row = QHBoxLayout()
        self.target_combo = QComboBox(sets_column)
        self.add_selected_button = QPushButton(sets_column)
        target_row.addWidget(self.target_combo, stretch=1)
        target_row.addWidget(self.add_selected_button)
        sets_layout.addLayout(target_row)
        selection_row = QHBoxLayout()
        self.select_shown_button = QPushButton(self._t("frame_sets.select_shown"), sets_column)
        self.clear_selection_button = QPushButton(self._t("frame_sets.clear_selection"), sets_column)
        selection_row.addWidget(self.select_shown_button)
        selection_row.addWidget(self.clear_selection_button)
        selection_row.addStretch(1)
        sets_layout.addLayout(selection_row)
        sets_layout.addStretch(1)

        export_column, export_layout = self._column(page)
        export_layout.addWidget(_heading(self._t("frame_sets.export.heading"), export_column))
        self.open_export_button = QPushButton(self._t("frame_sets.export.open"), export_column)
        self.open_export_button.setStyleSheet("font-weight: 700;")
        export_layout.addWidget(self.open_export_button)
        export_layout.addStretch(1)

        for index, column in enumerate((rules_column, sets_column, export_column)):
            grid.addWidget(column, 0, index)
            grid.setColumnStretch(index, 1)

        self.add_selected_button.clicked.connect(
            lambda: self.addSelectedRequested.emit(str(self.target_combo.currentData() or "new"))
        )
        self.select_shown_button.clicked.connect(self.selectShownRequested.emit)
        self.clear_selection_button.clicked.connect(self.clearSelectionRequested.emit)
        self.open_export_button.clicked.connect(self._open_export)
        return page

    def _build_export_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        top = QHBoxLayout()
        self.back_button = QToolButton(page)
        self.back_button.setText(self._t("frame_sets.export.back"))
        self.export_title = QLabel(page)
        self.export_title.setStyleSheet("font-weight: 700;")
        top.addWidget(self.back_button)
        top.addWidget(self.export_title, stretch=1)
        layout.addLayout(top)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        sets_column, sets_layout = self._column(page)
        sets_layout.addWidget(_heading(self._t("frame_sets.export.step_sets"), sets_column))
        self.export_sets_box = _count_grid()
        sets_layout.addLayout(self.export_sets_box)
        sets_layout.addStretch(1)
        self._export_set_group = QButtonGroup(self)
        self._export_set_group.setExclusive(True)

        items_column, items_layout = self._column(page)
        items_layout.addWidget(_heading(self._t("frame_sets.export.step_items"), items_column))
        self.item_checks: dict[str, QCheckBox] = {}
        for item in FRAME_ITEMS:
            check = QCheckBox(self._t(f"frame_sets.item.{item}"), items_column)
            check.toggled.connect(lambda _checked: self.exportChoicesChanged.emit())
            self.item_checks[item] = check
            items_layout.addWidget(check)
            if item == "errors":
                self.fill_black_check = QCheckBox(self._t("frame_sets.item.fill_black"), items_column)
                self.fill_black_check.setToolTip(self._t("frame_sets.item.fill_black_hint"))
                self.fill_black_check.setStyleSheet("margin-left: 18px;")
                self.fill_black_check.toggled.connect(lambda _checked: self.exportChoicesChanged.emit())
                items_layout.addWidget(self.fill_black_check)
        summary_label = QLabel(self._t("frame_sets.export.summary_items"), items_column)
        summary_label.setStyleSheet(_MUTED + " font-weight: 600;")
        items_layout.addWidget(summary_label)
        self.canvas_check = QCheckBox(self._t("frame_sets.item.canvas"), items_column)
        self.table_check = QCheckBox(self._t("frame_sets.item.table"), items_column)
        for check in (self.canvas_check, self.table_check):
            check.toggled.connect(lambda _checked: self.exportChoicesChanged.emit())
            items_layout.addWidget(check)
        items_layout.addStretch(1)

        target_column, target_layout = self._column(page)
        target_layout.addWidget(_heading(self._t("frame_sets.export.step_target"), target_column))
        folder_row = QHBoxLayout()
        self.folder_label = QLabel(target_column)
        self.folder_label.setWordWrap(True)
        self.browse_button = QPushButton(self._t("frame_sets.export.browse"), target_column)
        folder_row.addWidget(self.folder_label, stretch=1)
        folder_row.addWidget(self.browse_button)
        target_layout.addLayout(folder_row)
        self.tree_view = QPlainTextEdit(target_column)
        self.tree_view.setReadOnly(True)
        self.tree_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.tree_view.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 11px;")
        self.tree_view.setMinimumHeight(120)
        self.tree_view.setMaximumHeight(200)
        target_layout.addWidget(self.tree_view)
        self.export_summary = QLabel(target_column)
        self.export_summary.setStyleSheet("font-weight: 600;")
        target_layout.addWidget(self.export_summary)
        buttons = QHBoxLayout()
        self.cancel_export_button = QPushButton(self._t("common.cancel"), target_column)
        self.run_export_button = QPushButton(self._t("frame_sets.export.run"), target_column)
        self.run_export_button.setStyleSheet("font-weight: 700;")
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_export_button)
        buttons.addWidget(self.run_export_button)
        target_layout.addLayout(buttons)
        self.done_label = QLabel(target_column)
        self.done_label.setWordWrap(True)
        self.done_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.done_label.setStyleSheet("color: #8fe3b0; font-weight: 600;")
        target_layout.addWidget(self.done_label)
        done_buttons = QHBoxLayout()
        self.open_folder_button = QPushButton(self._t("frame_sets.export.open_folder"), target_column)
        self.show_report_button = QPushButton(self._t("frame_sets.export.show_report"), target_column)
        done_buttons.addWidget(self.open_folder_button)
        done_buttons.addWidget(self.show_report_button)
        done_buttons.addStretch(1)
        target_layout.addLayout(done_buttons)
        self.set_export_done(None)

        for index, column in enumerate((sets_column, items_column, target_column)):
            grid.addWidget(column, 0, index)
            grid.setColumnStretch(index, 1 if index < 2 else 2)
        layout.addLayout(grid)

        self.back_button.clicked.connect(self.close_export)
        self.cancel_export_button.clicked.connect(self.close_export)
        self.browse_button.clicked.connect(self.exportBrowseRequested.emit)
        self.run_export_button.clicked.connect(self.exportRequested.emit)
        self.open_folder_button.clicked.connect(self.openFolderRequested.emit)
        self.show_report_button.clicked.connect(self.showReportRequested.emit)
        return page

    # Threshold ---------------------------------------------------------------

    def set_threshold_text(self, text: str) -> None:
        self.threshold_label.setText(text)

    def set_pending_count(self, count: int) -> None:
        self.apply_button.setText(self._t("frame_sets.apply", count=int(count)))
        self.apply_button.setEnabled(int(count) > 0)

    def set_inverted(self, inverted: bool, *, can_invert: bool) -> None:
        self.invert_button.setChecked(bool(inverted))
        self.invert_button.setEnabled(bool(can_invert) or bool(inverted))
        self.inverted_label.setVisible(bool(inverted))

    # Rules -------------------------------------------------------------------

    def set_rules(self, rows: list[tuple[int, str, int, bool]], manual_count: int) -> None:
        """``rows``: (rule id, label, dropped frame count, enabled)."""

        _clear_layout(self.rules_box)
        parent = self.rules_box.parentWidget() or self
        if not rows and not manual_count:
            hint = QLabel(self._t("frame_sets.rules.empty"), parent)
            hint.setWordWrap(True)
            hint.setStyleSheet(_MUTED)
            self.rules_box.addWidget(hint, 0, 0, 1, 4)
            return
        for index, (rule_id, label, count, enabled) in enumerate(rows):
            self._rule_row(parent, index, label, count, enabled, rule_id=rule_id)
        if manual_count:
            self._rule_row(
                parent, len(rows), self._t("frame_sets.rules.manual"), manual_count, True, rule_id=None
            )

    def _rule_row(
        self, parent: QWidget, index: int, label: str, count: int, enabled: bool, *, rule_id: int | None
    ) -> None:
        check = QCheckBox(label, parent)
        check.setChecked(bool(enabled))
        remove = QToolButton(parent)
        remove.setText("✕")
        remove.setToolTip(self._t("frame_sets.rules.remove"))
        self.rules_box.addWidget(check, index, 0)
        self.rules_box.addWidget(_count_label(count, parent), index, 1)
        self.rules_box.addWidget(remove, index, 2)
        if rule_id is None:
            check.setEnabled(False)
            remove.clicked.connect(self.manualCleared.emit)
        else:
            check.toggled.connect(lambda checked, value=rule_id: self.ruleToggled.emit(value, bool(checked)))
            remove.clicked.connect(lambda _checked=False, value=rule_id: self.ruleRemoved.emit(value))

    # Sets --------------------------------------------------------------------

    def set_sets(self, rows: list[tuple[str, str, int, bool]], active_view: str) -> None:
        """``rows``: (view id, name, frame count, removable)."""

        _clear_layout(self.sets_box)
        for button in list(self._view_group.buttons()):
            self._view_group.removeButton(button)
        parent = self.sets_box.parentWidget() or self
        for index, (view_id, name, count, removable) in enumerate(rows):
            radio = QRadioButton(name, parent)
            radio.setChecked(view_id == active_view)
            self._view_group.addButton(radio)
            radio.toggled.connect(
                lambda checked, value=view_id: self.viewChosen.emit(value) if checked else None
            )
            self.sets_box.addWidget(radio, index, 0)
            self.sets_box.addWidget(_count_label(count, parent), index, 1)
            if removable:
                remove = QToolButton(parent)
                remove.setText("✕")
                remove.setToolTip(self._t("frame_sets.sets.remove"))
                remove.clicked.connect(lambda _checked=False, value=view_id: self.setRemoved.emit(value))
                self.sets_box.addWidget(remove, index, 2)

    def set_view_note(self, text: str) -> None:
        self.view_note.setText(text)
        self.view_note.setVisible(bool(text))

    def set_targets(self, targets: list[tuple[str, str]]) -> None:
        current = self.target_combo.currentData()
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItem(self._t("frame_sets.sets.new"), "new")
        for target_id, name in targets:
            self.target_combo.addItem(name, target_id)
        index = self.target_combo.findData(current)
        self.target_combo.setCurrentIndex(max(0, index))
        self.target_combo.blockSignals(False)

    def set_selected_count(self, count: int) -> None:
        self.add_selected_button.setText(self._t("frame_sets.add_selected", count=int(count)))
        self.add_selected_button.setEnabled(int(count) > 0)
        self.clear_selection_button.setEnabled(int(count) > 0)

    # Export ------------------------------------------------------------------

    def export_open(self) -> bool:
        return self.pages.currentIndex() == 1

    def _open_export(self) -> None:
        self.set_export_done(None)
        self.pages.setCurrentIndex(1)
        self.exportOpened.emit()

    def close_export(self) -> None:
        self.pages.setCurrentIndex(0)

    def set_export_title(self, text: str) -> None:
        self.export_title.setText(text)

    def set_export_sets(self, rows: list[tuple[str, str, int]], chosen: str) -> None:
        _clear_layout(self.export_sets_box)
        for button in list(self._export_set_group.buttons()):
            self._export_set_group.removeButton(button)
        parent = self.export_sets_box.parentWidget() or self
        for index, (view_id, name, count) in enumerate(rows):
            radio = QRadioButton(name, parent)
            radio.setProperty("view_id", view_id)
            radio.setChecked(view_id == chosen)
            radio.setEnabled(int(count) > 0)
            self._export_set_group.addButton(radio)
            radio.toggled.connect(lambda checked: self.exportChoicesChanged.emit() if checked else None)
            self.export_sets_box.addWidget(radio, index, 0)
            self.export_sets_box.addWidget(_count_label(count, parent), index, 1)

    def export_set_id(self) -> str:
        button = self._export_set_group.checkedButton()
        return str(button.property("view_id")) if button is not None else ""

    def export_items(self) -> frozenset[str]:
        return frozenset(item for item, check in self.item_checks.items() if check.isChecked())

    def set_export_items(self, items, *, canvas: bool, table: bool, fill_black: bool = False) -> None:
        for item, check in self.item_checks.items():
            check.blockSignals(True)
            check.setChecked(item in set(items))
            check.blockSignals(False)
        for check, value in ((self.canvas_check, canvas), (self.table_check, table), (self.fill_black_check, fill_black)):
            check.blockSignals(True)
            check.setChecked(bool(value))
            check.blockSignals(False)

    def export_fill_black(self) -> bool:
        return bool(self.fill_black_check.isChecked())

    def set_fill_black_available(self, available: bool) -> None:
        """Black frames make sense only for the errors folder and a set smaller than the run."""

        self.fill_black_check.setEnabled(bool(available))

    def export_canvas(self) -> bool:
        return bool(self.canvas_check.isChecked())

    def export_table(self) -> bool:
        return bool(self.table_check.isChecked())

    def set_output_dir(self, path: str) -> None:
        self.folder_label.setText(path)

    def set_export_preview(self, tree: str, summary: str, *, can_run: bool) -> None:
        self.tree_view.setPlainText(tree)
        self.export_summary.setText(summary)
        self.run_export_button.setEnabled(bool(can_run))

    def set_export_done(self, text: str | None) -> None:
        self.done_label.setText(text or "")
        self.done_label.setVisible(bool(text))
        self.open_folder_button.setVisible(bool(text))
        self.show_report_button.setVisible(bool(text))
