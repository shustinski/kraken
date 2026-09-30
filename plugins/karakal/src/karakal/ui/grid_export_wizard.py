"""Step-by-step wizard for exporting grid-cell defect images."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from ..core.features import show_class_conflict
from .ui_constants import GRID_INSPECTION_ERROR_TYPE_COLORS, GRID_INSPECTION_ERROR_TYPE_OPTIONS

Translate = Callable[..., str]
_HIDDEN_TYPES = {"conductor_zone"}


@dataclass(slots=True)
class ExportLayerOffer:
    """One model/source the current mode can export."""

    key: str
    title: str
    error_count: int
    frame_count: int = 0


@dataclass(slots=True)
class GridExportWizardChoices:
    image_format: str = "jpg"
    color_mode: str = "single"  # single | filter
    single_color: tuple[int, int, int] = (255, 255, 255)
    layer_keys: tuple[str, ...] = ()
    selected_types: tuple[str, ...] = ()
    frame_scope: str = "all"
    skip_empty: bool = False
    output_dir: Path = field(default_factory=Path.cwd)


class _FormatPage(QWizardPage):
    def __init__(self, translate: Translate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._t = translate
        self.setTitle(self._t("export_wizard.format_title"))
        self.setSubTitle(self._t("export_wizard.format_hint"))
        layout = QVBoxLayout(self)
        self._buttons: dict[str, QRadioButton] = {}
        for key, label_key in (
            ("jpg", "export_wizard.format_jpg"),
            ("png", "export_wizard.format_png"),
            ("bmp", "export_wizard.format_bmp"),
            ("tiff", "export_wizard.format_tiff"),
        ):
            button = QRadioButton(self._t(label_key), self)
            self._buttons[key] = button
            layout.addWidget(button)
        self._buttons["jpg"].setChecked(True)
        layout.addWidget(QLabel(self._t("export_wizard.format_note"), self))
        layout.addStretch(1)

    def selected_format(self) -> str:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return "jpg"

    def set_format(self, value: str) -> None:
        button = self._buttons.get(str(value or "jpg").lower())
        if button is not None:
            button.setChecked(True)


class _ColorPage(QWizardPage):
    def __init__(self, translate: Translate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._t = translate
        self.setTitle(self._t("export_wizard.color_title"))
        self.setSubTitle(self._t("export_wizard.color_hint"))
        layout = QVBoxLayout(self)
        self._single = QRadioButton(self._t("export_wizard.color_single"), self)
        self._filter = QRadioButton(self._t("export_wizard.color_filter"), self)
        self._single.setChecked(True)
        layout.addWidget(self._single)
        layout.addWidget(self._filter)
        color_row = QHBoxLayout()
        self._color_button = QPushButton(self._t("export_wizard.pick_color"), self)
        self._color_swatch = QLabel(self)
        self._color_swatch.setFixedSize(28, 28)
        self._color = QColor(255, 255, 255)
        self._apply_swatch()
        self._color_button.clicked.connect(self._pick_color)
        self._single.toggled.connect(self._sync_enabled)
        color_row.addWidget(self._color_button)
        color_row.addWidget(self._color_swatch)
        color_row.addStretch(1)
        layout.addLayout(color_row)
        layout.addWidget(QLabel(self._t("export_wizard.color_white_note"), self))
        layout.addStretch(1)
        self._sync_enabled()

    def _apply_swatch(self) -> None:
        self._color_swatch.setStyleSheet(
            f"background: {self._color.name()}; border: 1px solid #64748b; border-radius: 3px;"
        )

    def _pick_color(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, self._t("export_wizard.pick_color"))
        if chosen.isValid():
            self._color = chosen
            self._apply_swatch()

    def _sync_enabled(self) -> None:
        enabled = self._single.isChecked()
        self._color_button.setEnabled(enabled)
        self._color_swatch.setEnabled(enabled)

    def color_mode(self) -> str:
        return "single" if self._single.isChecked() else "filter"

    def single_color(self) -> tuple[int, int, int]:
        return int(self._color.red()), int(self._color.green()), int(self._color.blue())

    def set_color_mode(self, mode: str) -> None:
        if str(mode) == "filter":
            self._filter.setChecked(True)
        else:
            self._single.setChecked(True)
        self._sync_enabled()

    def set_single_color(self, rgb: tuple[int, int, int]) -> None:
        self._color = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        self._apply_swatch()


class _LayersPage(QWizardPage):
    def __init__(
        self,
        translate: Translate,
        layers: Sequence[ExportLayerOffer],
        selected_types: Sequence[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._t = translate
        self.setTitle(self._t("export_wizard.layers_title"))
        self.setSubTitle(self._t("export_wizard.layers_hint"))
        self._layers = tuple(layers)
        self._type_rows = tuple(selected_types)
        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()
        select_all = QPushButton(self._t("export_wizard.select_all"), self)
        clear_all = QPushButton(self._t("export_wizard.clear_all"), self)
        select_all.clicked.connect(lambda: self._set_all(True))
        clear_all.clicked.connect(lambda: self._set_all(False))
        buttons.addWidget(select_all)
        buttons.addWidget(clear_all)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self._checks: dict[str, QCheckBox] = {}
        for layer in self._layers:
            checkbox = QCheckBox(
                self._t(
                    "export_wizard.layer_row",
                    name=layer.title,
                    errors=int(layer.error_count),
                    frames=int(layer.frame_count),
                ),
                self,
            )
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._complete_changed)
            self._checks[layer.key] = checkbox
            layout.addWidget(checkbox)
        layout.addWidget(QLabel(self._t("export_wizard.types_caption"), self))
        if not self._type_rows:
            self._types_warning = QLabel(self._t("export_wizard.types_empty"), self)
            self._types_warning.setStyleSheet("color: #f87171;")
            layout.addWidget(self._types_warning)
        else:
            self._types_warning = None
            for error_type, label in self._type_rows:
                row = QWidget(self)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                swatch = QLabel(row)
                color = GRID_INSPECTION_ERROR_TYPE_COLORS.get(str(error_type), "#94a3b8")
                swatch.setFixedSize(14, 14)
                swatch.setStyleSheet(f"background: {color}; border-radius: 2px;")
                row_layout.addWidget(swatch)
                row_layout.addWidget(QLabel(label, row), 1)
                layout.addWidget(row)
            layout.addWidget(QLabel(self._t("export_wizard.types_note"), self))
        layout.addStretch(1)
        self._complete_changed()

    def _set_all(self, checked: bool) -> None:
        for checkbox in self._checks.values():
            checkbox.setChecked(bool(checked))

    def _complete_changed(self, *_args) -> None:
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        if self._types_warning is not None:
            return False
        return any(checkbox.isChecked() for checkbox in self._checks.values())

    def selected_layer_keys(self) -> tuple[str, ...]:
        return tuple(key for key, checkbox in self._checks.items() if checkbox.isChecked())

    def set_selected_layer_keys(self, keys: Sequence[str]) -> None:
        wanted = {str(item) for item in keys}
        if not wanted:
            return
        for key, checkbox in self._checks.items():
            checkbox.setChecked(key in wanted)


class _DestinationPage(QWizardPage):
    def __init__(
        self,
        translate: Translate,
        *,
        has_selection: bool,
        output_dir: Path,
        preview_builder: Callable[[], str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._t = translate
        self._preview_builder = preview_builder
        self.setTitle(self._t("export_wizard.destination_title"))
        self.setSubTitle(self._t("export_wizard.destination_hint"))
        layout = QVBoxLayout(self)
        folder_row = QHBoxLayout()
        self._folder = QLineEdit(str(output_dir), self)
        browse = QPushButton("…", self)
        browse.clicked.connect(self._browse)
        folder_row.addWidget(self._folder, 1)
        folder_row.addWidget(browse)
        layout.addLayout(folder_row)
        self._frames_all = QRadioButton(self._t("export_errors.frames_all"), self)
        self._frames_selected = QRadioButton(self._t("export_errors.frames_selected"), self)
        self._frames_all.setChecked(not has_selection)
        self._frames_selected.setChecked(has_selection)
        self._frames_selected.setEnabled(has_selection)
        layout.addWidget(self._frames_all)
        layout.addWidget(self._frames_selected)
        self._skip_empty = QCheckBox(self._t("export_errors.skip_empty"), self)
        self._skip_empty.setChecked(False)
        self._skip_empty.toggled.connect(self._refresh_preview)
        self._frames_all.toggled.connect(self._refresh_preview)
        self._frames_selected.toggled.connect(self._refresh_preview)
        layout.addWidget(self._skip_empty)
        layout.addWidget(QLabel(self._t("export_wizard.preview_caption"), self))
        self._preview = QLabel(self)
        self._preview.setWordWrap(True)
        self._preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._preview)
        layout.addWidget(scroll, 1)

    def initializePage(self) -> None:
        self._refresh_preview()

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, self._t("export_wizard.destination_title"), self._folder.text())
        if chosen:
            self._folder.setText(chosen)
            self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        self._preview.setText(self._preview_builder())

    def output_dir(self) -> Path:
        return Path(self._folder.text().strip() or ".")

    def frame_scope(self) -> str:
        return "selected" if self._frames_selected.isChecked() else "all"

    def skip_empty(self) -> bool:
        return bool(self._skip_empty.isChecked())

    def set_output_dir(self, path: Path | str) -> None:
        self._folder.setText(str(path))

    def set_frame_scope(self, scope: str) -> None:
        if scope == "selected" and self._frames_selected.isEnabled():
            self._frames_selected.setChecked(True)
        else:
            self._frames_all.setChecked(True)

    def set_skip_empty(self, value: bool) -> None:
        self._skip_empty.setChecked(bool(value))


class GridErrorExportWizard(QWizard):
    """Five-step export wizard. Types come from the main-window filter, not from here."""

    def __init__(
        self,
        translate: Translate,
        *,
        layers: Sequence[ExportLayerOffer],
        selected_types: Sequence[str],
        has_selection: bool,
        output_dir: Path,
        preview_builder: Callable[["GridErrorExportWizard"], str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._t = translate
        self.setWindowTitle(self._t("export_wizard.title"))
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setButtonText(QWizard.WizardButton.BackButton, self._t("export_wizard.back"))
        self.setButtonText(QWizard.WizardButton.NextButton, self._t("export_wizard.next"))
        self.setButtonText(QWizard.WizardButton.FinishButton, self._t("export_wizard.finish"))
        self.setButtonText(QWizard.WizardButton.CancelButton, self._t("common.cancel"))
        self.setMinimumWidth(560)
        self.setMinimumHeight(480)
        type_rows = self._type_rows(selected_types)
        self._format_page = _FormatPage(translate, self)
        self._color_page = _ColorPage(translate, self)
        self._layers_page = _LayersPage(translate, layers, type_rows, self)
        self._destination_page = _DestinationPage(
            translate,
            has_selection=has_selection,
            output_dir=output_dir,
            preview_builder=lambda: (preview_builder or (lambda _wizard: ""))(self),
            parent=self,
        )
        self.addPage(self._format_page)
        self.addPage(self._color_page)
        self.addPage(self._layers_page)
        self.addPage(self._destination_page)
        self._selected_types = tuple(str(item) for item in selected_types if str(item))

    def _type_rows(self, selected_types: Sequence[str]) -> tuple[tuple[str, str], ...]:
        wanted = {str(item) for item in selected_types}
        hidden = set(_HIDDEN_TYPES)
        if not show_class_conflict():
            hidden.add("class_conflict")
        rows: list[tuple[str, str]] = []
        for label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS:
            if str(error_type) in hidden or str(error_type) not in wanted:
                continue
            rows.append((str(error_type), self._t(label_key)))
        return tuple(rows)

    def choices(self) -> GridExportWizardChoices | None:
        layers = self._layers_page.selected_layer_keys()
        if not layers or not self._selected_types:
            return None
        return GridExportWizardChoices(
            image_format=self._format_page.selected_format(),
            color_mode=self._color_page.color_mode(),
            single_color=self._color_page.single_color(),
            layer_keys=layers,
            selected_types=self._selected_types,
            frame_scope=self._destination_page.frame_scope(),
            skip_empty=self._destination_page.skip_empty(),
            output_dir=self._destination_page.output_dir(),
        )

    def apply_saved_choices(self, payload: dict[str, object] | None) -> None:
        if not payload:
            return
        self._format_page.set_format(str(payload.get("image_format") or "jpg"))
        self._color_page.set_color_mode(str(payload.get("color_mode") or "single"))
        color = payload.get("single_color")
        if isinstance(color, (list, tuple)) and len(color) == 3:
            self._color_page.set_single_color((int(color[0]), int(color[1]), int(color[2])))
        layers = payload.get("layer_keys")
        if isinstance(layers, (list, tuple)):
            self._layers_page.set_selected_layer_keys([str(item) for item in layers])
        self._destination_page.set_frame_scope(str(payload.get("frame_scope") or "all"))
        self._destination_page.set_skip_empty(bool(payload.get("skip_empty", False)))
        output_dir = payload.get("output_dir")
        if output_dir:
            self._destination_page.set_output_dir(Path(str(output_dir)))

    def persistable_choices(self) -> dict[str, object]:
        choices = self.choices()
        if choices is None:
            return {}
        return {
            "image_format": choices.image_format,
            "color_mode": choices.color_mode,
            "single_color": list(choices.single_color),
            "layer_keys": list(choices.layer_keys),
            "frame_scope": choices.frame_scope,
            "skip_empty": choices.skip_empty,
            "output_dir": str(choices.output_dir),
        }
