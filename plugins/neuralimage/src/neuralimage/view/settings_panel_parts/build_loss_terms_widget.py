"""settings_panel: build loss terms widget."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QCheckBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)
from neuralimage.lib.loss_config import (
    DEFAULT_LOSS_TERM_WEIGHTS,
    LOSS_TERM_DISPLAY_NAMES,
    MAX_LOSS_TERM_WEIGHT_SUM,
)
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    create_double_spinbox,
)

from neuralimage.view.settings_panel_parts.module_state import LOSS_FUNCTIONS


class BuildLossTermsWidgetMixin:
    def _build_loss_terms_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.loss_presets_widget = QWidget()
        self.loss_presets_layout = QHBoxLayout(self.loss_presets_widget)
        self.loss_presets_layout.setContentsMargins(0, 0, 0, 0)
        self.loss_presets_layout.setSpacing(8)
        for preset_name in ('conductors', 'contacts'):
            button = QPushButton('')
            button.setCheckable(True)
            button.setProperty("selectionRole", "mode")
            button.setProperty("presetRole", "quickPreset")
            button.clicked.connect(lambda _checked=False, name=preset_name: self._apply_loss_preset(name))
            self.loss_presets_layout.addWidget(button)
            self.loss_preset_buttons[preset_name] = button
        self.loss_presets_layout.addStretch(1)

        for loss_name in LOSS_FUNCTIONS:
            checkbox = QCheckBox('')
            label = QLabel(LOSS_TERM_DISPLAY_NAMES.get(loss_name, loss_name))
            spinbox = create_double_spinbox(
                (0.0, MAX_LOSS_TERM_WEIGHT_SUM),
                step=0.05,
                default_value=0.0,
                decimals=2,
            )
            spinbox.setEnabled(False)

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            row_layout.addWidget(checkbox, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            row_layout.addWidget(label, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            row_layout.addWidget(spinbox, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(row_widget)

            self.loss_term_checkboxes[loss_name] = checkbox
            self.loss_term_spinboxes[loss_name] = spinbox
            self.loss_term_labels[loss_name] = label

            checkbox.toggled.connect(lambda checked, name=loss_name: self._on_loss_term_toggled(name, checked))
            spinbox.valueChanged.connect(lambda _value, name=loss_name: self._on_loss_term_value_changed(name))

        self._build_topograph_loss_rows(layout)

        self.loss_formula_label = QLabel('')
        self.loss_formula_label.setWordWrap(True)
        self.loss_formula_label.setTextFormat(Qt.TextFormat.RichText)
        self.loss_formula_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.loss_formula_label)

        self.set_loss_term_weights(DEFAULT_LOSS_TERM_WEIGHTS)
        return widget
