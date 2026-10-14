"""details_dialog: connect signals."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QSlider,
    QWidget,
)


class ConnectSignalsMixin:
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._pending_initial_fit:
            self._pending_initial_fit = False
            self._schedule_reset_view()

    def closeEvent(self, event) -> None:
        self._stop_detail_worker()
        self._stop_confidence_worker()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:
        if self._subpixel_selection is not None:
            if event.key() == Qt.Key.Key_Right:
                self._step_subpixel_selection(0, +1)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Down:
                self._step_subpixel_selection(+1, 0)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Left:
                self._step_subpixel_selection(0, -1)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Up:
                self._step_subpixel_selection(-1, 0)
                event.accept()
                return
        if event.key() == Qt.Key.Key_Right:
            if self._legacy_step_record(+1):
                event.accept()
                return
        if event.key() == Qt.Key.Key_Down:
            if self._legacy_step_record(+1):
                event.accept()
                return
        if event.key() == Qt.Key.Key_Left:
            if self._legacy_step_record(-1):
                event.accept()
                return
        if event.key() == Qt.Key.Key_Up:
            if self._legacy_step_record(-1):
                event.accept()
                return
        super().keyPressEvent(event)

    # Legacy lite methods retained for compatibility with older detail-dialog flows.
    def _activate_base_hold(self) -> None:
        self._legacy_base_hold_active = True
        if self._hold_preview_mode is None:
            self._hold_preview_mode = "base"
        if self._hold_preview_mode != "confidence_source":
            self._hold_preview_pixmap = QPixmap()
        self._update_layer_states()

    def _deactivate_base_hold(self) -> None:
        self._legacy_base_hold_active = False
        self._hold_preview_mode = None
        self.hold_preview_item.setPixmap(QPixmap())
        self._update_layer_states()

    def _legacy_update_frame_id_value(self) -> None:
        frame_id = "-"
        if self._record.identity is not None and self._record.identity.frame_id is not None:
            frame_id = str(self._record.identity.frame_id)
        self.frame_id_value.setText(frame_id)
        if hasattr(self, "frame_name_value"):
            self.frame_name_value.setText(str(self._record.display_name))

    def _legacy_step_record(self, delta: int) -> bool:
        records = tuple(self._build_result.records or ())
        if len(records) <= 1:
            return False
        current_key = str(self._record.key)
        current_index = next((index for index, row in enumerate(records) if row.key == current_key), -1)
        if current_index < 0:
            return False
        next_index = current_index + int(delta)
        if next_index < 0 or next_index >= len(records):
            return False
        self._record = records[next_index]
        self._tile_selection = None
        self._subpixel_selection = None
        self._session_view_state.pop("tile_selection", None)
        self._session_view_state.pop("subpixel_selection", None)
        self.setWindowTitle(self._t("details.window_title", name=self._record.display_name))
        self._legacy_update_frame_id_value()
        self._load_current_payload(reset_view=False, preserve_selection=True)
        return True

    def _layer_row(self, checkbox: QCheckBox, slider: QSlider, color_button: QPushButton | None = None) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(checkbox)
        layout.addWidget(slider, stretch=1)
        if color_button is not None:
            layout.addWidget(color_button)
        return row

    def _opacity_slider(self, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setRange(0, 100)
        slider.setValue(int(value))
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return slider

    def _color_button(self) -> QPushButton:
        button = QPushButton("", self)
        button.setToolTip(self._t("details.color_button_tooltip"))
        button.setFixedSize(20, 20)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return button

    def _create_navigation_shortcuts(self) -> None:
        for key, delta in (
            (Qt.Key.Key_Right, +1),
            (Qt.Key.Key_Down, +1),
            (Qt.Key.Key_Left, -1),
            (Qt.Key.Key_Up, -1),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda delta=delta: self._legacy_step_record(delta))

    def _connect_signals(self) -> None:
        self.model_a_combo.currentIndexChanged.connect(self._on_model_a_changed)
        self.model_b_combo.currentIndexChanged.connect(self._on_model_pair_changed)
        self.result_kind_combo.currentIndexChanged.connect(self._refresh_scene_from_controls)
        self.layer_view_combo.currentIndexChanged.connect(self._refresh_scene_from_controls)
        self.grayscale_diff_checkbox.toggled.connect(self._refresh_scene_from_controls)
        self.overlay_view.middlePressed.connect(self._activate_context_hold)
        self.overlay_view.middleReleased.connect(self._deactivate_base_hold)
        for widget in (
            self.original_visible,
            self.first_source_visible,
            self.second_source_visible,
            self.result_visible,
        ):
            widget.toggled.connect(self._update_layer_states)
        for widget in (
            self.original_opacity,
            self.first_source_opacity,
            self.second_source_opacity,
            self.result_opacity,
        ):
            widget.valueChanged.connect(self._update_layer_states)
        for widget in (
            self.original_visible,
            self.first_source_visible,
            self.second_source_visible,
            self.result_visible,
        ):
            widget.toggled.connect(self._store_view_settings)
        for widget in (
            self.original_opacity,
            self.first_source_opacity,
            self.second_source_opacity,
            self.result_opacity,
        ):
            widget.valueChanged.connect(self._store_view_settings)
        self.result_kind_combo.currentIndexChanged.connect(self._store_view_settings)
        self.layer_view_combo.currentIndexChanged.connect(self._store_view_settings)
        self.grayscale_diff_checkbox.toggled.connect(self._store_view_settings)
        self.first_mask_color_button.clicked.connect(lambda: self._choose_named_color("first_mask"))
        self.second_mask_color_button.clicked.connect(lambda: self._choose_named_color("second_mask"))
        self.result_mask_color_button.clicked.connect(self._choose_active_result_color)
