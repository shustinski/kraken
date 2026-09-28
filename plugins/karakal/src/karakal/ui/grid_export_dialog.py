"""Choices for the grid-defect JPEG export."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..core.features import show_class_conflict
from ..core.grid_error_export import GridErrorExportChoices
from .ui_constants import GRID_INSPECTION_ERROR_TYPE_COLORS, GRID_INSPECTION_ERROR_TYPE_OPTIONS

Translate = Callable[..., str]
_HIDDEN_TYPES = {"conductor_zone"}


class GridErrorExportDialog(QDialog):
    def __init__(
        self,
        translate: Translate,
        *,
        enabled_types: tuple[str, ...],
        layers: tuple[str, ...],
        masks: tuple[tuple[str, str], ...],
        has_selection: bool,
        output_dir: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._t = translate
        self._output_dir = Path(output_dir)
        self.setWindowTitle(self._t("export_errors.title"))
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(self._t("export_errors.types"), self))
        self._type_checks: dict[str, QCheckBox] = {}
        enabled = {str(item) for item in enabled_types}
        hidden = set(_HIDDEN_TYPES)
        if not show_class_conflict():
            hidden.add("class_conflict")
        for label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS:
            if str(error_type) in hidden:
                continue
            row = QWidget(self)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            swatch = QLabel(row)
            color = GRID_INSPECTION_ERROR_TYPE_COLORS.get(str(error_type), "#94a3b8")
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background: {color}; border-radius: 2px;")
            checkbox = QCheckBox(self._t(label_key), row)
            checkbox.setChecked(str(error_type) in enabled)
            row_layout.addWidget(swatch)
            row_layout.addWidget(checkbox, 1)
            layout.addWidget(row)
            self._type_checks[str(error_type)] = checkbox
        self._all_types = QCheckBox(self._t("export_errors.all_types"), self)
        self._all_types.toggled.connect(self._toggle_all_types)
        layout.addWidget(self._all_types)

        color_row = QWidget(self)
        color_layout = QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0)
        self._color_mode = QRadioButton(self._t("export_errors.color"), color_row)
        self._bw_mode = QRadioButton(self._t("export_errors.bw"), color_row)
        self._color_mode.setChecked(True)
        color_layout.addWidget(self._color_mode)
        color_layout.addWidget(self._bw_mode)
        layout.addWidget(color_row)

        layout.addWidget(QLabel(self._t("export_errors.layers"), self))
        self._layer_checks: dict[str, QCheckBox] = {}
        layer_labels = {
            "confidence": "grid_layer.confidence",
            "binary": "grid_layer.binary",
        }
        for layer_key in ("confidence", "binary"):
            if layer_key not in layers and layers:
                continue
            checkbox = QCheckBox(self._t(layer_labels[layer_key]), self)
            checkbox.setChecked(layer_key in layers or not layers)
            self._layer_checks[layer_key] = checkbox
            layout.addWidget(checkbox)

        self._mask_checks: dict[str, QCheckBox] = {}
        if len(masks) > 1:
            layout.addWidget(QLabel(self._t("export_errors.masks"), self))
            for mask_id, mask_label in masks:
                checkbox = QCheckBox(mask_label, self)
                checkbox.setChecked(True)
                self._mask_checks[str(mask_id)] = checkbox
                layout.addWidget(checkbox)

        self._frames_all = QRadioButton(self._t("export_errors.frames_all"), self)
        self._frames_selected = QRadioButton(self._t("export_errors.frames_selected"), self)
        self._frames_all.setChecked(not has_selection)
        self._frames_selected.setChecked(has_selection)
        self._frames_selected.setEnabled(has_selection)
        layout.addWidget(self._frames_all)
        layout.addWidget(self._frames_selected)
        self._skip_empty = QCheckBox(self._t("export_errors.skip_empty"), self)
        self._skip_empty.setChecked(False)
        layout.addWidget(self._skip_empty)

        folder_row = QWidget(self)
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        self._folder_label = QLabel(str(self._output_dir), folder_row)
        self._folder_label.setWordWrap(True)
        browse = QPushButton("…", folder_row)
        browse.clicked.connect(self._browse)
        folder_layout.addWidget(self._folder_label, 1)
        folder_layout.addWidget(browse)
        form = QFormLayout()
        form.addRow(folder_row)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(self._t("export_errors.run"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _toggle_all_types(self, checked: bool) -> None:
        for checkbox in self._type_checks.values():
            checkbox.setChecked(bool(checked))

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, self._t("export_errors.title"), str(self._output_dir))
        if chosen:
            self._output_dir = Path(chosen)
            self._folder_label.setText(str(self._output_dir))

    def choices(self) -> GridErrorExportChoices | None:
        selected = tuple(key for key, checkbox in self._type_checks.items() if checkbox.isChecked())
        if not selected:
            return None
        layers = tuple(key for key, checkbox in self._layer_checks.items() if checkbox.isChecked()) or ("confidence",)
        masks = tuple(key for key, checkbox in self._mask_checks.items() if checkbox.isChecked())
        return GridErrorExportChoices(
            selected_types=selected,
            all_types=self._all_types.isChecked() or len(selected) == len(self._type_checks),
            color_mode="bw" if self._bw_mode.isChecked() else "color",
            layers=layers,
            masks=masks,
            frame_scope="selected" if self._frames_selected.isChecked() else "all",
            skip_empty=self._skip_empty.isChecked(),
            output_dir=self._output_dir,
        )
