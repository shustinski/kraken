"""Export window of the memory map: which files, the user's own text format, main blocks only,
how the bits are turned (the whole field and each block), and a preview of the text."""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.memory_map import SEAM_BITS, SEAM_EMPTY, MemoryMap
from dataclasses import replace

from ..core.memory_map_export import (
    BLOCK_ORDER_FORWARD,
    BLOCK_ORDER_REVERSE,
    LAYOUT_BLOCKS,
    LAYOUT_FIELD,
    ORIENTATIONS,
    MemoryMapExport,
    TxtFormat,
    blocks_format,
    default_symbols,
    field_format,
    memory_map_text,
    oriented_map,
)

SYMBOL_KEYS = ("one", "zero", "empty", "conflict", "defect", "partial", "outside")


class MemoryMapExportDialog(QDialog):
    def __init__(self, translator: Callable[..., str], mmap: MemoryMap, previous: MemoryMapExport | None = None, parent=None) -> None:
        super().__init__(parent)
        self._t = translator
        self._map = mmap
        self.setWindowTitle(self._t("memory_map.export_title"))
        start = previous or MemoryMapExport(folder="", custom=TxtFormat(symbols=default_symbols(mmap.mode)))
        root = QVBoxLayout(self)

        where = QFormLayout()
        folder_row = QWidget(self)
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        self.folder_edit = QLineEdit(start.folder, folder_row)
        browse = QPushButton("…", folder_row)
        browse.clicked.connect(self._browse)
        folder_layout.addWidget(self.folder_edit, stretch=1)
        folder_layout.addWidget(browse)
        where.addRow(self._t("memory_map.export_folder"), folder_row)
        self.name_edit = QLineEdit(start.name, self)
        where.addRow(self._t("memory_map.export_name"), self.name_edit)
        root.addLayout(where)

        files = QGroupBox(self._t("memory_map.export_files"), self)
        files_layout = QVBoxLayout(files)
        self.txt_field = QCheckBox(self._t("memory_map.export_txt_field"), files)
        self.txt_field.setChecked(start.txt_field)
        self.txt_blocks = QCheckBox(self._t("memory_map.export_txt_blocks"), files)
        self.txt_blocks.setChecked(start.txt_blocks)
        self.txt_custom = QCheckBox(self._t("memory_map.export_txt_custom"), files)
        self.txt_custom.setChecked(start.txt_custom)
        for check in (self.txt_field, self.txt_blocks, self.txt_custom):
            files_layout.addWidget(check)

        # The user's own text format, shown when it is chosen.
        self.custom_box = QGroupBox(self._t("memory_map.custom_format"), files)
        custom = QFormLayout(self.custom_box)
        self.layout_combo = QComboBox(self.custom_box)
        self.layout_combo.addItem(self._t("memory_map.layout.field"), LAYOUT_FIELD)
        self.layout_combo.addItem(self._t("memory_map.layout.blocks"), LAYOUT_BLOCKS)
        self.layout_combo.setCurrentIndex(max(0, self.layout_combo.findData(start.custom.layout)))
        custom.addRow(self._t("memory_map.layout"), self.layout_combo)
        symbols_row = QWidget(self.custom_box)
        symbols_layout = QHBoxLayout(symbols_row)
        symbols_layout.setContentsMargins(0, 0, 0, 0)
        self.symbol_edits: dict[str, QLineEdit] = {}
        for key in SYMBOL_KEYS:
            label = QLabel(self._t(f"memory_map.symbol.{key}"), symbols_row)
            edit = QLineEdit(start.custom.symbols.get(key, " "), symbols_row)
            edit.setMaxLength(4)
            edit.setFixedWidth(34)
            self.symbol_edits[key] = edit
            symbols_layout.addWidget(label)
            symbols_layout.addWidget(edit)
        custom.addRow(self._t("memory_map.symbols"), symbols_row)
        self.separator_edit = QLineEdit(start.custom.cell_separator, self.custom_box)
        self.separator_edit.setMaxLength(4)
        self.separator_edit.setFixedWidth(60)
        self.separator_edit.setToolTip(self._t("memory_map.separator_tooltip"))
        custom.addRow(self._t("memory_map.separator"), self.separator_edit)
        self.row_numbers = QCheckBox(self._t("memory_map.row_numbers"), self.custom_box)
        self.row_numbers.setChecked(start.custom.row_numbers)
        self.column_ruler = QCheckBox(self._t("memory_map.column_ruler"), self.custom_box)
        self.column_ruler.setChecked(start.custom.column_ruler)
        self.header_check = QCheckBox(self._t("memory_map.header"), self.custom_box)
        self.header_check.setChecked(start.custom.header)
        for check in (self.row_numbers, self.column_ruler, self.header_check):
            custom.addRow(check)
        files_layout.addWidget(self.custom_box)
        self.txt_custom.toggled.connect(self.custom_box.setVisible)
        self.custom_box.setVisible(start.txt_custom)

        png_row = QWidget(files)
        png_layout = QHBoxLayout(png_row)
        png_layout.setContentsMargins(0, 0, 0, 0)
        self.png = QCheckBox(self._t("memory_map.export_png"), png_row)
        self.png.setChecked(start.png)
        self.png_px = QSpinBox(png_row)
        self.png_px.setRange(1, 8)
        self.png_px.setValue(start.png_cell_px)
        self.png_px.setSuffix(self._t("memory_map.px_per_cell"))
        self.png_blocks = QCheckBox(self._t("memory_map.show_blocks"), png_row)
        self.png_blocks.setChecked(start.png_blocks)
        self.png_frames = QCheckBox(self._t("memory_map.show_frames"), png_row)
        self.png_frames.setChecked(start.png_frames)
        for widget in (self.png, self.png_px, self.png_blocks, self.png_frames):
            png_layout.addWidget(widget)
        png_layout.addStretch(1)
        files_layout.addWidget(png_row)
        self.csv = QCheckBox(self._t("memory_map.export_csv"), files)
        self.csv.setChecked(start.csv)
        self.summary = QCheckBox(self._t("memory_map.export_summary"), files)
        self.summary.setChecked(start.summary)
        files_layout.addWidget(self.csv)
        files_layout.addWidget(self.summary)
        root.addWidget(files)

        turn = QGroupBox(self._t("memory_map.orientation"), self)
        turn_form = QFormLayout(turn)
        self.field_orientation = QComboBox(turn)
        self.block_orientation = QComboBox(turn)
        for combo, value in ((self.field_orientation, start.field_orientation), (self.block_orientation, start.block_orientation)):
            for key in ORIENTATIONS:
                combo.addItem(self._t(f"memory_map.orient.{key}"), key)
            combo.setCurrentIndex(max(0, combo.findData(value)))
        self.field_orientation.setToolTip(self._t("memory_map.orient_field_tooltip"))
        self.block_orientation.setToolTip(self._t("memory_map.orient_block_tooltip"))
        turn_form.addRow(self._t("memory_map.orient_field"), self.field_orientation)
        turn_form.addRow(self._t("memory_map.orient_block"), self.block_orientation)
        self.block_order = QComboBox(turn)
        self.block_order.addItem(self._t("memory_map.block_order.forward"), BLOCK_ORDER_FORWARD)
        self.block_order.addItem(self._t("memory_map.block_order.reverse"), BLOCK_ORDER_REVERSE)
        self.block_order.setCurrentIndex(max(0, self.block_order.findData(start.block_order)))
        turn_form.addRow(self._t("memory_map.block_order"), self.block_order)
        self.number_from = QComboBox(turn)
        self.number_from.addItem("0", 0)
        self.number_from.addItem("1", 1)
        self.number_from.setCurrentIndex(1 if start.number_from == 1 else 0)
        turn_form.addRow(self._t("memory_map.number_from"), self.number_from)
        root.addWidget(turn)

        self.main_only = QCheckBox(self._t("memory_map.main_only"), self)
        self.main_only.setChecked(start.main_only)
        services = sum(1 for b in mmap.blocks if b.service)
        self.main_only.setToolTip(self._t("memory_map.main_only_tooltip", count=services))
        root.addWidget(self.main_only)

        counts = mmap.seam_counts()
        bad = sum(v for k, v in counts.items() if k not in (SEAM_BITS, SEAM_EMPTY))
        if bad:
            warning = QLabel(self._t("memory_map.export_seams_warning", count=bad), self)
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #ffb040;")
            root.addWidget(warning)

        root.addWidget(QLabel(self._t("memory_map.preview"), self))
        self.preview = QPlainTextEdit(self)
        self.preview.setReadOnly(True)
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.preview.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.preview.setMinimumHeight(170)
        root.addWidget(self.preview)
        # The preview follows every change a moment later (turning a big field takes a little while).
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._update_preview)
        for combo in (self.field_orientation, self.block_orientation, self.block_order, self.number_from, self.layout_combo):
            combo.currentIndexChanged.connect(self._preview_timer.start)
        for check in (self.txt_field, self.txt_blocks, self.txt_custom, self.main_only, self.row_numbers, self.column_ruler, self.header_check):
            check.toggled.connect(self._preview_timer.start)
        for edit in (self.separator_edit, *self.symbol_edits.values()):
            edit.textChanged.connect(self._preview_timer.start)
        self._oriented_key = None
        self._oriented = mmap

        self.error_label = QLabel("", self)
        self.error_label.setStyleSheet("color: #ff6b6b;")
        root.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(self._t("memory_map.export_go"))
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._update_preview()

    def _turned(self, choice: MemoryMapExport):
        key = (choice.field_orientation, choice.block_orientation)
        if key != self._oriented_key:
            self._oriented = oriented_map(self._map, *key)
            self._oriented_key = key
        return self._oriented

    def _update_preview(self) -> None:
        """The first lines of the text the export writes: my own format if chosen, else the
        whole field, else by blocks."""

        choice = self.choice()
        common = {"main_only": choice.main_only, "number_from": choice.number_from, "block_order": choice.block_order}
        if choice.txt_custom:
            fmt = replace(choice.custom, main_only=choice.main_only)
        elif choice.txt_blocks and not choice.txt_field:
            fmt = replace(blocks_format(self._map.mode), **common)
        else:
            fmt = replace(field_format(self._map.mode), **common)
        self.preview.setPlainText(memory_map_text(self._turned(choice), fmt, preview=(12, 120)))

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self._t("memory_map.export_folder"), self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)

    def choice(self) -> MemoryMapExport:
        symbols = {key: (edit.text() or " ") for key, edit in self.symbol_edits.items()}
        return MemoryMapExport(
            folder=self.folder_edit.text().strip(),
            name=self.name_edit.text().strip() or "memory_map",
            txt_field=self.txt_field.isChecked(),
            txt_blocks=self.txt_blocks.isChecked(),
            txt_custom=self.txt_custom.isChecked(),
            custom=TxtFormat(
                layout=str(self.layout_combo.currentData() or LAYOUT_FIELD),
                symbols=symbols,
                cell_separator=self.separator_edit.text(),
                row_numbers=self.row_numbers.isChecked(),
                column_ruler=self.column_ruler.isChecked(),
                header=self.header_check.isChecked(),
                number_from=int(self.number_from.currentData() or 0),
                block_order=str(self.block_order.currentData() or BLOCK_ORDER_FORWARD),
            ),
            png=self.png.isChecked(),
            png_cell_px=int(self.png_px.value()),
            png_blocks=self.png_blocks.isChecked(),
            png_frames=self.png_frames.isChecked(),
            csv=self.csv.isChecked(),
            summary=self.summary.isChecked(),
            main_only=self.main_only.isChecked(),
            field_orientation=str(self.field_orientation.currentData() or "none"),
            block_orientation=str(self.block_orientation.currentData() or "none"),
            number_from=int(self.number_from.currentData() or 0),
            block_order=str(self.block_order.currentData() or BLOCK_ORDER_FORWARD),
        )

    def _accept(self) -> None:
        choice = self.choice()
        if not choice.folder:
            self.error_label.setText(self._t("memory_map.export_need_folder"))
            return
        if not any((choice.txt_field, choice.txt_blocks, choice.txt_custom, choice.png, choice.csv, choice.summary)):
            self.error_label.setText(self._t("memory_map.export_need_file"))
            return
        self.accept()
