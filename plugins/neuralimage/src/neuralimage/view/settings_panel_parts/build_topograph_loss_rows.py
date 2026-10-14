"""settings_panel: build topograph loss rows."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QCheckBox,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
)
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    create_double_spinbox,
)

from neuralimage.view.settings_panel_parts.module_state import (
    MAX_TOPOGRAPH_LOSS_WEIGHT,
    TOPOGRAPH_LOSS_DISPLAY_NAME,
)


class BuildTopographLossRowsMixin:
    def _build_topograph_loss_rows(self, layout: QVBoxLayout) -> None:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        self.topograph_enabled_check_box = QCheckBox('')
        self.topograph_term_label = QLabel(TOPOGRAPH_LOSS_DISPLAY_NAME)
        self.topograph_loss_weight_spinbox = create_double_spinbox(
            (0.0, MAX_TOPOGRAPH_LOSS_WEIGHT),
            step=0.05,
            default_value=0.1,
            decimals=2,
        )
        self.topograph_loss_weight_spinbox.setEnabled(False)
        topograph_row = QWidget()
        topograph_row_layout = QHBoxLayout(topograph_row)
        topograph_row_layout.setContentsMargins(0, 0, 0, 0)
        topograph_row_layout.setSpacing(8)
        topograph_row_layout.addWidget(
            self.topograph_enabled_check_box,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        topograph_row_layout.addWidget(
            self.topograph_term_label,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        topograph_row_layout.addWidget(
            self.topograph_loss_weight_spinbox,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        layout.addWidget(topograph_row)

        self.topograph_debug_viz_check_box = QCheckBox('')
        self.topograph_debug_viz_label = QLabel('Show Topograph critical regions')
        self.topograph_debug_viz_check_box.setChecked(False)
        self.topograph_debug_viz_check_box.setEnabled(False)
        debug_row = QWidget()
        debug_row_layout = QHBoxLayout(debug_row)
        debug_row_layout.setContentsMargins(24, 0, 0, 0)
        debug_row_layout.setSpacing(8)
        debug_row_layout.addWidget(
            self.topograph_debug_viz_check_box,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        debug_row_layout.addWidget(
            self.topograph_debug_viz_label,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        debug_row_layout.addStretch(1)
        layout.addWidget(debug_row)

        self.topograph_enabled_check_box.toggled.connect(self._on_topograph_enabled_toggled)
        self.topograph_loss_weight_spinbox.valueChanged.connect(lambda _value: self._sync_loss_controls())
