"""Details dialog for the extended validation gradient widget."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.analysis_modes import metric_level_key, metric_visual_ratio
from ..core.backend_constants import BCE_SCORE_CAP, MODEL_CONFIDENCE_UNCERTAIN_DELTA, POINT_SUPPORT_THRESHOLD, POLYGON_SUPPORT_THRESHOLD
from ..core.domain import BuildResult, ComparisonMode, FrameRecord
from ..core.subpixel_grid import (
    SubpixelGrid,
    SubpixelGridSpec,
    SubpixelSelection,
    build_subpixel_grid_from_pair,
    subpixel_bounds_for_index,
    subpixel_spec_from_options,
)
from ..core.repository import (
    _boundary_mask,
    _confidence_map_from_probability,
    _confidence_display_map_from_probability,
    _frame_uncertainty_components_from_probability,
    metric_higher_is_better,
    _paint_disk,
    _point_map_from_view,
    _support_weights_from_probability,
    _uncertainty_map_from_probability,
    compute_comparison,
    compute_comparison_score,
)
from ..core.confidence_maps import (
    DEFAULT_CONFIDENCE_BAD_AREA_THRESHOLD,
    build_algorithmic_uncertainty,
    build_model_uncertainty,
    combine_uncertainty_maps,
    confidence_bad_area_intensity,
)
from ..core.tile_grid import TileGridPlan, plan_tile_grid, tile_bounds_for_index
from ..core.workers import DetailConfidenceWorker, DetailPayloadWorker
from ..ui.i18n import Translator
from ..ui.ui_constants import DEFAULT_SUBPIXEL_AGGREGATION

DETAIL_PIXEL_VIEW_THRESHOLD = 32.0


class _OverlayGraphicsView(QGraphicsView):
    """Provide zoom and middle-button preview toggling for the overlay preview."""

    middlePressed = pyqtSignal()
    middleReleased = pyqtSignal()
    leftClicked = pyqtSignal(QPointF)
    viewTransformChanged = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(92, 96, 102))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setMinimumSize(320, 320)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        current_scale = self.transform().m11()
        next_scale = current_scale * factor
        if 0.05 <= next_scale <= 60.0:
            self.scale(factor, factor)
            self.viewTransformChanged.emit()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.middlePressed.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftClicked.emit(self.mapToScene(event.position().toPoint()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.middleReleased.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def fit_to_scene(self) -> None:
        scene = self.scene()
        if scene is None:
            return
        fit_rect = scene.itemsBoundingRect()
        if fit_rect.isNull():
            fit_rect = scene.sceneRect()
        if fit_rect.isNull() or self.viewport().width() <= 4 or self.viewport().height() <= 4:
            return
        fit_rect = fit_rect.adjusted(-2.0, -2.0, 2.0, 2.0)
        self.fitInView(fit_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(fit_rect.center())

    def reset_view(self) -> None:
        self.resetTransform()
        self.fit_to_scene()
        self.viewTransformChanged.emit()


class _ExpandableDetailScoreCard(QWidget):
    """Show the current frame score with click-to-expand details."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.title_label = QLabel(title, self)
        self.title_label.setWordWrap(True)
        self.value_button = QPushButton("-", self)
        self.value_button.setCheckable(True)
        self.value_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.value_button.setMinimumHeight(40)
        self.value_button.setStyleSheet(
            "padding: 8px 12px; border-radius: 10px; "
            "background-color: #2f3844; color: #edf3fb; font-weight: 700; "
            "border: none; text-align: center;"
        )
        self.details_label = QLabel("", self)
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet(
            "padding: 6px 8px; color: #c9d3df; "
            "background-color: #11161d; border-radius: 8px;"
        )
        self.details_label.hide()
        self.value_button.toggled.connect(self._on_toggled)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_button)
        layout.addWidget(self.details_label)

    def _on_toggled(self, checked: bool) -> None:
        self.details_label.setVisible(bool(checked) and bool(self.details_label.text().strip()))

    def set_payload(self, title: str, value_text: str, value_style: str, details: str, tooltip: str = "") -> None:
        self.title_label.setText(str(title))
        self.value_button.setText(str(value_text))
        self.value_button.setStyleSheet(str(value_style) + "; border: none; text-align: center;")
        self.details_label.setText(str(details))
        self.details_label.setVisible(bool(self.value_button.isChecked()) and bool(str(details).strip()))
        self.setToolTip(str(tooltip or ""))
        self.title_label.setToolTip(str(tooltip or ""))
        self.value_button.setToolTip(str(tooltip or ""))
        self.details_label.setToolTip(str(tooltip or ""))


class ExtendFrameDetailsDialog(QDialog):
    """Show standard frame comparisons for two selected models."""

    def __init__(
        self,
        record: FrameRecord,
        build_result: BuildResult,
        preferred_metric_key: str | None = None,
        *,
        session_view_state: dict[str, object] | None = None,
        on_view_state_changed=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._record = record
        self._build_result = build_result
        self._translator = Translator()
        self._t = self._translator.tr
        self._payload: dict[str, object] = {}
        self._overlay_cache: dict[tuple[object, ...], QPixmap] = {}
        self._derived_cache: dict[tuple[object, ...], object] = {}
        self._comparison_score: float | None = None
        self._preferred_metric_key = str(preferred_metric_key or build_result.selected_metric_key or "overall_frame_score")
        self._pending_initial_fit = True
        self._detail_thread: QThread | None = None
        self._detail_worker: DetailPayloadWorker | None = None
        self._confidence_thread: QThread | None = None
        self._confidence_worker: DetailConfidenceWorker | None = None
        self._retired_confidence_workers: list[tuple[DetailConfidenceWorker, QThread]] = []
        self._loading_confidence_model_id: str | None = None
        self._detail_request_generation = 0
        self._confidence_request_generation = 0
        self._payload_loading = False
        self._legacy_base_hold_active = False
        self._hold_preview_mode: str | None = None
        self._hold_preview_pixmap = QPixmap()
        self._session_view_state = session_view_state if session_view_state is not None else {}
        self._on_view_state_changed = on_view_state_changed
        tile_selection_payload = self._session_view_state.get("tile_selection")
        self._tile_selection = None
        self._subpixel_selection: SubpixelSelection | None = None
        subpixel_selection_payload = self._session_view_state.get("subpixel_selection")
        if isinstance(tile_selection_payload, dict):
            row = tile_selection_payload.get("row")
            column = tile_selection_payload.get("column")
            if row is not None and column is not None:
                self._tile_selection = (int(row), int(column))
        elif isinstance(tile_selection_payload, (tuple, list)) and len(tile_selection_payload) >= 2:
            self._tile_selection = (int(tile_selection_payload[0]), int(tile_selection_payload[1]))
        if isinstance(subpixel_selection_payload, dict):
            try:
                spec = None
                spec_rows = subpixel_selection_payload.get("spec_rows")
                spec_columns = subpixel_selection_payload.get("spec_columns")
                if spec_rows is not None and spec_columns is not None:
                    spec = SubpixelGridSpec(
                        rows=int(spec_rows),
                        columns=int(spec_columns),
                        mode=str(subpixel_selection_payload.get("spec_mode") or "grid"),
                        tile_width=int(subpixel_selection_payload.get("spec_tile_width") or 1),
                        tile_height=int(subpixel_selection_payload.get("spec_tile_height") or 1),
                        overlap=int(subpixel_selection_payload.get("spec_overlap") or 0),
                    ).normalized()
                self._subpixel_selection = SubpixelSelection(
                    parent_row=int(subpixel_selection_payload.get("parent_row", 0)),
                    parent_column=int(subpixel_selection_payload.get("parent_column", 0)),
                    sub_row=int(subpixel_selection_payload.get("sub_row", 0)),
                    sub_column=int(subpixel_selection_payload.get("sub_column", 0)),
                    parent_value=float(subpixel_selection_payload.get("parent_value", 0.0)),
                    subpixel_value=float(subpixel_selection_payload.get("subpixel_value", 0.0)),
                    subpixel_confidence=(
                        None if subpixel_selection_payload.get("subpixel_confidence") is None else float(subpixel_selection_payload.get("subpixel_confidence"))
                    ),
                    aggregation=str(subpixel_selection_payload.get("aggregation") or "mean"),
                    metric_key=str(subpixel_selection_payload.get("metric_key") or "overall_frame_score"),
                    spec=spec,
                )
            except Exception:
                self._subpixel_selection = None
        self._restored_result_kind: str | None = None
        self._sticky_result_kind: str | None = None
        self._tile_plan: TileGridPlan | None = None
        self.frame_id_value = QLabel("-", self)
        self.frame_id_value.hide()
        self.resize(1440, 940)
        self.setWindowTitle(self._t("details.window_title", name=record.display_name))
        self._build_ui()
        scene = self.overlay_view.scene()
        assert scene is not None
        self.original_item = scene.addPixmap(QPixmap())
        self.first_source_item = scene.addPixmap(QPixmap())
        self.second_source_item = scene.addPixmap(QPixmap())
        self.result_item = scene.addPixmap(QPixmap())
        self.hold_preview_item = scene.addPixmap(QPixmap())
        self.tile_preview_item = scene.addPixmap(QPixmap())
        self.tile_grid_item = scene.addPixmap(QPixmap())
        self.tile_selection_item = scene.addRect(0.0, 0.0, 0.0, 0.0)
        # Legacy lite aliases retained for backward compatibility.
        self.base_item = self.original_item
        self.first_item = self.first_source_item
        self.second_item = self.second_source_item
        self.base_visible = self.original_visible
        self.first_visible = self.first_source_visible
        self.second_visible = self.second_source_visible
        self.first_color = QComboBox(self)
        self.second_color = QComboBox(self)
        self.first_color.hide()
        self.second_color.hide()
        self.original_item.setZValue(0.0)
        self.first_source_item.setZValue(1.0)
        self.second_source_item.setZValue(2.0)
        self.result_item.setZValue(3.0)
        self.hold_preview_item.setZValue(4.0)
        self.tile_preview_item.setZValue(-1.0)
        self.tile_grid_item.setZValue(3.5)
        self.tile_selection_item.setZValue(4.5)
        selection_pen = QPen(QColor(255, 196, 0, 235), 0.0)
        selection_pen.setCosmetic(True)
        self.tile_selection_item.setPen(selection_pen)
        self.tile_selection_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.tile_selection_item.setVisible(False)
        self.tile_preview_item.setVisible(False)
        self.tile_grid_item.setVisible(False)
        self.overlay_view.viewTransformChanged.connect(self._update_tile_lod)
        self.overlay_view.leftClicked.connect(self._on_view_clicked)
        self._connect_signals()
        self._create_navigation_shortcuts()
        self._restore_view_settings()
        self._populate_model_combos()
        self._load_current_payload(reset_view=True, preserve_selection=False)
        self._legacy_update_frame_id_value()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(splitter)

        viewer_widget = QWidget(splitter)
        viewer_layout = QVBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_toolbar = QHBoxLayout()
        viewer_toolbar.setContentsMargins(0, 0, 0, 0)
        self.zoom_hint_label = QLabel(self._t("details.zoom_hint"), viewer_widget)
        self.zoom_hint_label.setWordWrap(True)
        self.zoom_hint_label.setMinimumWidth(0)
        self.zoom_hint_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.zoom_hint_label.hide()
        self.confidence_loading_label = QLabel("Loading confidence...", viewer_widget)
        self.confidence_loading_label.setVisible(False)
        self.confidence_loading_bar = QProgressBar(viewer_widget)
        self.confidence_loading_bar.setRange(0, 0)
        self.confidence_loading_bar.setFixedWidth(180)
        self.confidence_loading_bar.setTextVisible(False)
        self.confidence_loading_bar.setVisible(False)
        viewer_toolbar.addWidget(self.zoom_hint_label, stretch=1)
        viewer_toolbar.addWidget(self.confidence_loading_label)
        viewer_toolbar.addWidget(self.confidence_loading_bar)
        viewer_layout.addLayout(viewer_toolbar)
        self.overlay_view = _OverlayGraphicsView(viewer_widget)
        self.overlay_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        viewer_layout.addWidget(self.overlay_view, stretch=1)

        controls_scroll = QScrollArea(splitter)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(420)
        controls_host = QWidget(controls_scroll)
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        controls_scroll.setWidget(controls_host)

        frame_summary_group = QGroupBox(self._t("details.frame_info"), controls_host)
        frame_summary_form = QFormLayout(frame_summary_group)
        frame_summary_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.frame_name_value = QLabel(self._record.display_name, frame_summary_group)
        self.frame_name_value.setWordWrap(True)
        self.frame_id_value.show()
        self.subpixel_selection_value = QLabel("-", frame_summary_group)
        self.subpixel_selection_value.setWordWrap(True)
        self.subpixel_selection_value.hide()
        frame_summary_form.addRow(self._t("matrix.preview.frame"), self.frame_name_value)
        frame_summary_form.addRow(self._t("details.frame_id"), self.frame_id_value)
        frame_summary_form.addRow(self._t("details.subpixel_selection"), self.subpixel_selection_value)
        controls_layout.addWidget(frame_summary_group)

        config_group = QGroupBox(self._t("details.display"), controls_host)
        config_form = QFormLayout(config_group)
        config_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.model_a_combo = QComboBox(config_group)
        self.model_a_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.model_b_combo = QComboBox(config_group)
        self.model_b_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.primary_model_title = QLabel(self._t("details.selected_source"), config_group)
        self.secondary_model_title = QLabel(self._t("details.comparison_source"), config_group)
        self.model_a_combo.hide()
        self.model_b_combo.hide()
        self.primary_model_title.hide()
        self.secondary_model_title.hide()
        self.comparison_preset_combo = QComboBox(config_group)
        self.comparison_preset_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.comparison_preset_combo.hide()
        self.comparison_group_combo = QComboBox(config_group)
        self.comparison_group_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.comparison_group_combo.hide()
        self.result_kind_combo = QComboBox(config_group)
        self.result_kind_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.layer_view_combo = QComboBox(config_group)
        self.layer_view_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.layer_view_combo.addItem(self._t("details.layer_view.binary"), "binary")
        self.layer_view_combo.addItem(self._t("details.layer_view.source"), "source")
        self.operation_combo = QComboBox(config_group)
        self.operation_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.operation_combo.addItem(self._t("details.difference"), ComparisonMode.DISAGREEMENT)
        self.operation_combo.addItem(self._t("details.grayscale_difference"), ComparisonMode.GRAYSCALE_DIFF)
        self.operation_combo.setCurrentIndex(self.operation_combo.findData(ComparisonMode.DISAGREEMENT))
        self.operation_combo.hide()
        self.operation_label = QLabel(self._t("details.grayscale_difference"), config_group)
        self.grayscale_diff_checkbox = QCheckBox(config_group)
        config_form.addRow(self._t("details.layer_view"), self.layer_view_combo)
        config_form.addRow(self._t("details.result_overlay"), self.result_kind_combo)
        config_form.addRow(self.operation_label, self.grayscale_diff_checkbox)
        controls_layout.addWidget(config_group)

        layers_group = QGroupBox(self._t("details.layers"), controls_host)
        layers_form = QFormLayout(layers_group)
        layers_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.original_visible = QCheckBox(layers_group)
        self.original_visible.setChecked(True)
        self.original_opacity = self._opacity_slider(100)
        self.first_source_visible = QCheckBox(layers_group)
        self.first_source_visible.setChecked(True)
        self.first_source_opacity = self._opacity_slider(42)
        self.second_source_visible = QCheckBox(layers_group)
        self.second_source_visible.setChecked(True)
        self.second_source_opacity = self._opacity_slider(55)
        self.result_visible = QCheckBox(layers_group)
        self.result_visible.setChecked(True)
        self.result_opacity = self._opacity_slider(78)
        self.original_layer_title = QLabel(self._t("details.original"), layers_group)
        self.first_source_layer_title = QLabel(self._t("details.selected_source"), layers_group)
        self.second_source_layer_title = QLabel(self._t("details.comparison_source"), layers_group)
        self.result_layer_title = QLabel(self._t("details.result_overlay"), layers_group)
        self.first_mask_color_button = self._color_button()
        self.second_mask_color_button = self._color_button()
        self.result_mask_color_button = self._color_button()
        layers_form.addRow(self.original_layer_title, self._layer_row(self.original_visible, self.original_opacity))
        layers_form.addRow(self.first_source_layer_title, self._layer_row(self.first_source_visible, self.first_source_opacity, self.first_mask_color_button))
        layers_form.addRow(self.second_source_layer_title, self._layer_row(self.second_source_visible, self.second_source_opacity, self.second_mask_color_button))
        layers_form.addRow(self.result_layer_title, self._layer_row(self.result_visible, self.result_opacity, self.result_mask_color_button))
        controls_layout.addWidget(layers_group)

        comparison_score_group = QGroupBox(self._t("details.selected_comparison_score"), controls_host)
        comparison_score_layout = QVBoxLayout(comparison_score_group)
        self.comparison_score_card = _ExpandableDetailScoreCard(self._t("details.frame_score"), comparison_score_group)
        comparison_score_layout.addWidget(self.comparison_score_card)
        self.subpixel_score_card = _ExpandableDetailScoreCard(self._t("details.subpixel_score"), comparison_score_group)
        self.subpixel_score_card.hide()
        comparison_score_layout.addWidget(self.subpixel_score_card)
        controls_layout.addWidget(comparison_score_group)
        controls_layout.addStretch(1)

        splitter.addWidget(viewer_widget)
        splitter.addWidget(controls_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([980, 420])

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

    def _populate_model_combos(self) -> None:
        current_a = self.model_a_combo.currentData()
        current_b = self.model_b_combo.currentData()
        for combo in (self.model_a_combo, self.model_b_combo):
            combo.blockSignals(True)
            combo.clear()
            for spec in self._build_result.model_specs:
                combo.addItem(spec.display_name, spec.model_id)
            combo.blockSignals(False)
        if self.model_a_combo.count() <= 0:
            return
        index_a = self.model_a_combo.findData(current_a)
        self.model_a_combo.setCurrentIndex(index_a if index_a >= 0 else 0)
        preferred_b = current_b
        if preferred_b is None or preferred_b == self.model_a_combo.currentData():
            preferred_b = None
            for index in range(self.model_b_combo.count()):
                candidate = self.model_b_combo.itemData(index)
                if candidate != self.model_a_combo.currentData():
                    preferred_b = candidate
                    break
            if preferred_b is None:
                preferred_b = self.model_a_combo.currentData()
        index_b = self.model_b_combo.findData(preferred_b)
        self.model_b_combo.setCurrentIndex(index_b if index_b >= 0 else 0)

    def _schedule_reset_view(self) -> None:
        QTimer.singleShot(0, self.overlay_view.reset_view)

    def _has_ground_truth(self) -> bool:
        return self._payload.get("gt_mask") is not None

    def _has_original(self) -> bool:
        original = self._payload.get("original_gray")
        if original is None:
            return False
        return np.asarray(original).size > 0

    def _model_display_name(self, model_id: str | None) -> str:
        if model_id is None:
            return "-"
        payload_names = self._payload.get("model_display_names") or {}
        if model_id in payload_names:
            return str(payload_names[model_id])
        for spec in self._build_result.model_specs:
            if spec.model_id == model_id:
                return spec.display_name
        return str(model_id)

    def _status_label(self, status: str) -> str:
        return self._t(f"status.{status}")

    def _component_name_label(self, name: str) -> str:
        labels_en = {
            'source': 'Source',
            'formula': 'Formula',
            'definition': 'Definition',
            'value': 'Value',
            'model': 'Model',
            'hot_region_count': 'Hot region count',
            'mean_object_confidence': 'Mean object confidence',
            'mean_object_probability': 'Mean object probability',
            'uncertain_fraction': 'Uncertain fraction',
            'mean_transition_width': 'Mean transition width',
            'transition_width': 'Transition width',
            'object_area_fraction': 'Object area fraction',
            'mean_point_confidence': 'Mean point confidence',
            'mean_point_probability': 'Mean point probability',
            'mean_point_contrast': 'Mean point contrast',
            'point_count': 'Point count',
            'soft_dice': 'Soft Dice',
            'soft_iou': 'Soft IoU',
            'ssim': 'SSIM',
            'dice': 'Dice',
            'iou': 'IoU',
            'hausdorff_distance': 'Hausdorff distance',
            'centroid_distance': 'Centroid distance',
            'mae': 'MAE',
            'rmse': 'RMSE',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_at_r': 'F1@r',
            'mean_localization_error': 'Mean localization error',
            'localization_score': 'Localization score',
            'localization_agreement': 'Localization agreement',
            'count_error': 'Count error',
            'count_agreement': 'Count agreement',
            'connected_component_error': 'Connected-component error',
            'chamfer_score': 'Chamfer score',
            'hausdorff_score': 'Hausdorff score',
        }
        labels_ru = {
            'source': 'Источник',
            'formula': 'Формула',
            'definition': 'Определение',
            'value': 'Значение',
            'model': 'Модель',
            'hot_region_count': 'Число горячих областей',
            'summary_metric': 'Сводная метрика',
            'mean_object_confidence': 'Средняя объектная уверенность',
            'mean_core_confidence': 'Средняя уверенность ядра',
            'mean_boundary_uncertainty': 'Средняя неуверенность границы',
            'mean_weighted_confidence': 'Средняя взвешенная уверенность',
            'mean_object_probability': 'Среднее значение grayscale внутри объекта',
            'uncertain_fraction': 'Доля сомнительных пикселей',
            'object_area_fraction': 'Доля площади объекта',
            'polygon_count': 'Число полигонов',
            'mean_point_confidence': 'Средняя уверенность точек',
            'mean_center_confidence': 'Средняя уверенность в центре точки',
            'mean_local_confidence': 'Средняя локальная уверенность точки',
            'mean_point_probability': 'Среднее значение grayscale по точкам',
            'mean_point_contrast': 'Средний контраст точек',
            'point_count': 'Количество точек',
            'soft_dice': 'Soft Dice',
            'soft_iou': 'Soft IoU',
            'ssim': 'SSIM',
            'dice': 'Dice',
            'iou': 'IoU',
            'hausdorff_distance': 'Расстояние Хаусдорфа',
            'centroid_distance': 'Расстояние между центроидами',
            'mae': 'MAE',
            'rmse': 'RMSE',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_at_r': 'F1@r',
            'mean_localization_error': 'Средняя ошибка локализации',
            'localization_score': 'Оценка локализации',
            'localization_agreement': 'Согласованность локализации',
            'count_error': 'Ошибка количества',
            'count_agreement': 'Согласованность количества',
            'connected_component_error': 'Ошибка числа компонент',
            'chamfer_score': 'Оценка Chamfer',
            'hausdorff_score': 'Оценка Хаусдорфа',
        }
        labels = labels_ru if getattr(self._translator, 'language', 'en') == 'ru' else labels_en
        if name in labels:
            return labels[name]
        return name.replace('_', ' ')

    def _component_value_text(self, value: str) -> str:
        values_en = {
            'labeled frame': 'labeled frame',
            'unlabeled frame': 'unlabeled frame',
            'supervised error map': 'supervised error map',
            'variance/entropy risk map': 'variance / entropy risk map',
            'mean entropy of consensus probability': 'mean entropy of consensus probability',
            'mean variance over model probability maps': 'mean variance over model probability maps',
            'acquisition_score': 'acquisition_score',
        }
        values_ru = {
            'labeled frame': 'размеченный кадр',
            'unlabeled frame': 'неразмеченный кадр',
            'supervised error map': 'supervised-карта ошибки',
            'variance/entropy risk map': 'карта риска по variance / entropy',
            'mean entropy of consensus probability': 'средняя энтропия consensus probability',
            'mean variance over model probability maps': 'средняя дисперсия probability maps моделей',
            'acquisition_score': 'приоритет на разметку',
            'weighted': 'взвешенная',
            'core': 'ядро',
        }
        values = values_ru if getattr(self._translator, 'language', 'en') == 'ru' else values_en
        return values.get(value, value.replace(' vs ', ' против ') if getattr(self._translator, 'language', 'en') == 'ru' else value)

    def _localize_metric_lines(self, lines: list[str]) -> list[str]:
        localized: list[str] = []
        for line in lines:
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            status_prefix = ''
            for status in ("active", "auxiliary", "legacy"):
                prefix = f"{status} "
                if stripped.startswith(prefix):
                    status_prefix = f"{self._status_label(status)} "
                    stripped = stripped[len(prefix):]
                    break
            if ':' in stripped:
                name, value = stripped.split(':', 1)
                name = self._component_name_label(name.strip())
                value = self._component_value_text(value.strip())
                stripped = f"{status_prefix}{name}: {value}"
            else:
                stripped = status_prefix + self._component_value_text(stripped)
            localized.append(indent + stripped)
        return localized

    def _frame_type_label(self) -> str:
        return self._t('frame_type.point') if self._is_point_geometry() else self._t('frame_type.polygon')

    def _result_kind_hint(self, result_kind: str) -> str | None:
        if self._is_point_geometry() and result_kind == "point_matches":
            return self._t("hint.point_matches")
        if self._is_point_geometry() and result_kind == "confidence":
            return self._t("hint.point_confidence")
        if (not self._is_point_geometry()) and result_kind == "boundary":
            return self._t("hint.boundary_difference")
        if (not self._is_point_geometry()) and result_kind == "confidence":
            return self._t("hint.polygon_confidence")
        if result_kind == "confidence_bad_areas":
            return self._t("hint.confidence_bad_areas")
        if result_kind == "confidence_mix":
            return self._t("hint.confidence_mix")
        if result_kind == 'confidence_low_mask':
            return self._t('hint.confidence_low_mask')
        if result_kind == 'confidence_high_mask':
            return self._t('hint.confidence_high_mask')
        if result_kind == 'confidence_final_mask':
            return self._t('hint.confidence_final_mask')
        if result_kind == 'confidence_object_contours':
            return self._t('hint.confidence_object_contours')
        if result_kind == 'confidence_branch_debug':
            return self._t('hint.confidence_branch_debug')
        if result_kind == 'confidence_preprocessed':
            return self._t('hint.confidence_preprocessed_probability')
        if result_kind == 'confidence_boundary_cues':
            return 'Raw boundary cues before thinning'
        if result_kind == 'confidence_thin_barrier':
            return 'Thin barrier map used to stop object growth'
        if result_kind == 'confidence_core_seeds':
            return 'High-confidence object cores used as instance seeds'
        if result_kind == 'confidence_candidate_region':
            return 'Candidate region allowed for object expansion'
        if result_kind == 'confidence_bridge_cuts':
            return 'Weak bridge cuts inserted between conflicting objects'
        if result_kind == 'confidence_barrier_stops':
            return 'Pixels blocked from growth by barriers and bridge cuts'
        return None

    def _selected_score_hint(self) -> str | None:
        preferred_confidence_model = self._preferred_confidence_model_id()
        if preferred_confidence_model is not None:
            return self._t('hint.intra_model_point') if self._is_point_geometry() else self._t('hint.confidence_polygon')
        group_key = self._current_comparison_group()
        if group_key == 'model_model':
            return self._t('hint.model_model_point') if self._is_point_geometry() else self._t('hint.model_model_polygon')
        if group_key == 'model_labeled':
            return self._t('hint.model_labeled_point') if self._is_point_geometry() else self._t('hint.model_labeled_polygon')
        return None

    def _refresh_tooltips(self, score_hint: str | None, overlay_hint: str | None, result_kind: str) -> None:
        score_tooltip = score_hint or ''
        overlay_tooltip = overlay_hint or ''
        self.comparison_score_card.setToolTip(score_tooltip)
        self.comparison_score_card.title_label.setToolTip(score_tooltip)
        self.comparison_score_card.value_button.setToolTip(score_tooltip)
        self.comparison_score_card.details_label.setToolTip(score_tooltip)
        if getattr(self, "subpixel_score_card", None) is not None:
            self.subpixel_score_card.setToolTip(score_tooltip)
            self.subpixel_score_card.title_label.setToolTip(score_tooltip)
            self.subpixel_score_card.value_button.setToolTip(score_tooltip)
            self.subpixel_score_card.details_label.setToolTip(score_tooltip)
        self.result_kind_combo.setToolTip(overlay_tooltip)
        self.result_layer_title.setToolTip(overlay_tooltip)
        self.result_item.setToolTip(overlay_tooltip) if hasattr(self.result_item, 'setToolTip') else None
        if result_kind == 'diff':
            self.grayscale_diff_checkbox.setToolTip(self._t('details.grayscale_difference'))
            self.operation_label.setToolTip(self._t('details.grayscale_difference'))
        else:
            self.grayscale_diff_checkbox.setToolTip('')
            self.operation_label.setToolTip('')

    def _preferred_group_from_metric(self) -> str:
        metric_key = str(self._preferred_metric_key or "")
        if metric_key in {"model_labeled_score", "labeled_best_quality", "labeled_mean_quality"} and self._has_ground_truth():
            return "model_labeled"
        if len(self._build_result.model_specs) >= 2:
            return "model_model"
        if self._has_ground_truth():
            return "model_labeled"
        return "fallback"

    def _auto_model_model_tuple(self) -> tuple[str, str, str]:
        pairwise = self._payload.get("pairwise_model_comparisons") or ()
        if pairwise:
            top = pairwise[0]
            model_a = str(top.get("model_a", ""))
            model_b = str(top.get("model_b", ""))
            if model_a and model_b:
                return f"model_vs_model::{model_a}::{model_b}", f"model:{model_a}", f"model:{model_b}"
        model_ids = [spec.model_id for spec in self._build_result.model_specs]
        if len(model_ids) >= 2:
            return f"model_vs_model::{model_ids[0]}::{model_ids[1]}", f"model:{model_ids[0]}", f"model:{model_ids[1]}"
        if model_ids:
            return f"model_vs_model::{model_ids[0]}::{model_ids[0]}", f"model:{model_ids[0]}", f"model:{model_ids[0]}"
        return "none", "gt", "gt"

    def _auto_model_labeled_tuple(self) -> tuple[str, str, str]:
        model_metrics = self._payload.get("model_metrics") or {}
        if model_metrics:
            best_model_id = max(model_metrics.items(), key=lambda item: float(getattr(item[1], 'quality_score', 0.0)))[0]
            return f"gt_vs_model::{best_model_id}", "gt", f"model:{best_model_id}"
        model_ids = [spec.model_id for spec in self._build_result.model_specs]
        if model_ids:
            return f"gt_vs_model::{model_ids[0]}", "gt", f"model:{model_ids[0]}"
        return "none", "gt", "gt"

    def _available_comparison_groups(self) -> list[tuple[str, str]]:
        groups: list[tuple[str, str]] = []
        if len(self._build_result.model_specs) >= 2:
            groups.append((self._t("metric.group.model_model"), "model_model"))
        if self._has_ground_truth():
            groups.append((self._t("metric.group.model_labeled"), "model_labeled"))
        if not groups:
            groups.append((self._t("details.comparisons"), "fallback"))
        return groups

    def _current_comparison_group(self) -> str:
        return self._preferred_group_from_metric()

    def _refresh_comparison_groups(self, previous_group: str | None) -> None:
        groups = self._available_comparison_groups()
        self.comparison_group_combo.blockSignals(True)
        self.comparison_group_combo.clear()
        for label, key in groups:
            self.comparison_group_combo.addItem(label, key)
        target_group = previous_group
        if target_group is None and groups:
            target_group = groups[0][1]
        selected_index = 0
        for index in range(self.comparison_group_combo.count()):
            if self.comparison_group_combo.itemData(index) == target_group:
                selected_index = index
                break
        if self.comparison_group_combo.count() > 0:
            self.comparison_group_combo.setCurrentIndex(selected_index)
        self.comparison_group_combo.blockSignals(False)

    def _comparison_presets(self, group_key: str | None = None) -> list[tuple[str, str, str, str]]:
        presets: list[tuple[str, str, str, str]] = []
        model_ids = [spec.model_id for spec in self._build_result.model_specs]
        selected_group = group_key or self._current_comparison_group()
        if selected_group == "model_model":
            for first_model_id in model_ids:
                for second_model_id in model_ids:
                    if first_model_id == second_model_id:
                        continue
                    presets.append((
                        f"model_vs_model::{first_model_id}::{second_model_id}",
                        f"{self._model_display_name(first_model_id)} vs {self._model_display_name(second_model_id)}",
                        f"model:{first_model_id}",
                        f"model:{second_model_id}",
                    ))
            return presets
        if selected_group == "model_labeled" and self._has_ground_truth():
            for model_id in model_ids:
                presets.append((
                    f"gt_vs_model::{model_id}",
                    f"{self._t("details.ground_truth")} vs {self._model_display_name(model_id)}",
                    "gt",
                    f"model:{model_id}",
                ))
            return presets
        for first_model_id in model_ids:
            for second_model_id in model_ids:
                if first_model_id == second_model_id:
                    continue
                presets.append((
                    f"model_vs_model::{first_model_id}::{second_model_id}",
                    f"{self._model_display_name(first_model_id)} vs {self._model_display_name(second_model_id)}",
                    f"model:{first_model_id}",
                    f"model:{second_model_id}",
                ))
        return presets

    def _refresh_comparison_presets(self, previous_key: str | None) -> None:
        presets = self._comparison_presets(self._current_comparison_group())
        self.comparison_preset_combo.blockSignals(True)
        self.comparison_preset_combo.clear()
        for key, label, source_a, source_b in presets:
            self.comparison_preset_combo.addItem(label, (key, source_a, source_b))
        target_key = previous_key
        if target_key is None and presets:
            target_key = presets[0][0]
        chosen_index = 0
        for index in range(self.comparison_preset_combo.count()):
            entry = self.comparison_preset_combo.itemData(index)
            if isinstance(entry, tuple) and len(entry) == 3 and entry[0] == target_key:
                chosen_index = index
                break
        if self.comparison_preset_combo.count() > 0:
            self.comparison_preset_combo.setCurrentIndex(chosen_index)
        self.comparison_preset_combo.blockSignals(False)

    def _current_comparison_tuple(self) -> tuple[str, str, str]:
        group_key = self._current_comparison_group()
        if group_key == "model_labeled" and self._has_ground_truth():
            return self._auto_model_labeled_tuple()
        if group_key == "model_model":
            return self._auto_model_model_tuple()
        return self._auto_model_model_tuple()

    def _start_loading_payload(self, *, reset_view: bool, preserve_selection: bool) -> None:
        self._stop_detail_worker()
        self._stop_confidence_worker()
        self._detail_request_generation += 1
        generation = int(self._detail_request_generation)
        self._payload_loading = True
        previous_result_kind = self._selected_result_kind() if preserve_selection else self._sticky_result_kind
        default_model_id = self._preferred_confidence_model_id() or (self._build_result.model_specs[0].model_id if self._build_result.model_specs else None)
        max_side = int(getattr(self._build_result.options, "analysis_max_side", 0) or 0) or None
        self.comparison_score_card.set_payload(
            self._t("details.frame_score"),
            "Loading...",
            self._comparison_score_style(None),
            "Loading frame details...",
        )
        self._refresh_scene(reset_view=reset_view)
        self._refresh_info()
        self._detail_thread = QThread(self)
        self._detail_worker = DetailPayloadWorker(self._record, self._build_result, default_model_id, max_side)
        self._detail_worker.moveToThread(self._detail_thread)
        self._detail_thread.started.connect(self._detail_worker.run)
        self._detail_worker.finished.connect(
            lambda payload, rv=reset_view, ps=preserve_selection, prk=previous_result_kind, g=generation: self._on_payload_loaded(
                payload,
                reset_view=rv,
                preserve_selection=ps,
                previous_result_kind=prk,
                generation=g,
            )
        )
        self._detail_worker.failed.connect(lambda message, g=generation: self._on_detail_worker_failed(message, generation=g, worker_kind="detail"))
        self._detail_worker.finished.connect(self._detail_thread.quit)
        self._detail_worker.failed.connect(self._detail_thread.quit)
        self._detail_thread.finished.connect(self._cleanup_detail_worker)
        self._detail_thread.start()

    def _on_payload_loaded(
        self,
        payload: dict[str, object],
        *,
        reset_view: bool,
        preserve_selection: bool,
        previous_result_kind: str | None,
        generation: int | None = None,
    ) -> None:
        if generation is not None and int(generation) != int(self._detail_request_generation):
            return
        self._payload_loading = False
        self._payload = payload
        self._overlay_cache.clear()
        self._derived_cache.clear()
        self._set_confidence_loading_state(False)
        preferred_kind = previous_result_kind if preserve_selection else (self._sticky_result_kind or self._restored_result_kind)
        self._refresh_result_kind_options(preferred_kind)
        self._restored_result_kind = None
        self._apply_selected_model(self._selected_model_for_current_comparison())
        self._refresh_scene(reset_view=reset_view)
        self._refresh_info()

    def _load_current_payload(self, *, reset_view: bool, preserve_selection: bool) -> None:
        self._start_loading_payload(reset_view=reset_view, preserve_selection=preserve_selection)

    def _default_subpixel_spec(self) -> SubpixelGridSpec:
        source_shape = None
        try:
            base = self._full_base_array()
            if isinstance(base, np.ndarray) and base.ndim == 2 and base.size > 0:
                source_shape = (int(base.shape[0]), int(base.shape[1]))
        except Exception:
            source_shape = None
        return subpixel_spec_from_options(self._build_result.options, source_shape)

    def _subpixel_grid_for_view(self) -> SubpixelGrid | None:
        selection = self._subpixel_selection
        if selection is None:
            return None
        spec = selection.spec or self._default_subpixel_spec()
        mode = self._current_operation_mode()
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        cache_key = (
            "subpixel_grid_view",
            str(self._record.key),
            str(first_key),
            str(second_key),
            str(mode.value),
            int(spec.rows),
            int(spec.columns),
            str(getattr(selection, "aggregation", DEFAULT_SUBPIXEL_AGGREGATION) or DEFAULT_SUBPIXEL_AGGREGATION),
        )
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, SubpixelGrid):
            return cached
        first, second = self._pairwise_comparison_inputs(first_key, second_key, mode)
        if first.ndim != 2 or second.ndim != 2 or first.size == 0 or second.size == 0 or first.shape != second.shape:
            return None
        grid = build_subpixel_grid_from_pair(
            first,
            second,
            spec,
            score_fn=lambda first_tile, second_tile: compute_comparison_score(first_tile, second_tile, mode),
            aggregation=str(getattr(selection, "aggregation", DEFAULT_SUBPIXEL_AGGREGATION) or DEFAULT_SUBPIXEL_AGGREGATION),
            value_kind="risk",
        )
        self._derived_cache[cache_key] = grid
        return grid

    def _step_subpixel_selection(self, delta_row: int, delta_column: int) -> bool:
        selection = self._subpixel_selection
        if selection is None:
            return False
        spec = selection.spec or self._default_subpixel_spec()
        rows = max(1, int(spec.rows))
        columns = max(1, int(spec.columns))
        next_row = min(max(int(selection.sub_row) + int(delta_row), 0), rows - 1)
        next_column = min(max(int(selection.sub_column) + int(delta_column), 0), columns - 1)
        if next_row == int(selection.sub_row) and next_column == int(selection.sub_column):
            return False
        grid = self._subpixel_grid_for_view()
        if grid is not None and grid.spec.normalized() == spec.normalized():
            subpixel_value = float(grid.value_at(next_row, next_column))
            subpixel_confidence = grid.confidence_at(next_row, next_column)
            parent_value = float(grid.aggregate_value(selection.aggregation or DEFAULT_SUBPIXEL_AGGREGATION))
        else:
            subpixel_value = float(selection.subpixel_value)
            subpixel_confidence = selection.subpixel_confidence
            parent_value = float(selection.parent_value)
        self._subpixel_selection = SubpixelSelection(
            parent_row=int(selection.parent_row),
            parent_column=int(selection.parent_column),
            sub_row=int(next_row),
            sub_column=int(next_column),
            parent_value=float(parent_value),
            subpixel_value=float(subpixel_value),
            subpixel_confidence=None if subpixel_confidence is None else float(subpixel_confidence),
            aggregation=str(selection.aggregation or DEFAULT_SUBPIXEL_AGGREGATION),
            metric_key=str(selection.metric_key or self._comparison_score_metric_key()),
            spec=spec.normalized(),
        )
        self._session_view_state["subpixel_selection"] = {
            "parent_row": int(self._subpixel_selection.parent_row),
            "parent_column": int(self._subpixel_selection.parent_column),
            "sub_row": int(self._subpixel_selection.sub_row),
            "sub_column": int(self._subpixel_selection.sub_column),
            "parent_value": float(self._subpixel_selection.parent_value),
            "subpixel_value": float(self._subpixel_selection.subpixel_value),
            "subpixel_confidence": None if self._subpixel_selection.subpixel_confidence is None else float(self._subpixel_selection.subpixel_confidence),
            "aggregation": str(self._subpixel_selection.aggregation),
            "metric_key": str(self._subpixel_selection.metric_key),
            "spec_rows": int(self._subpixel_selection.spec.rows if self._subpixel_selection.spec is not None else spec.rows),
            "spec_columns": int(self._subpixel_selection.spec.columns if self._subpixel_selection.spec is not None else spec.columns),
            "spec_mode": str(self._subpixel_selection.spec.mode if self._subpixel_selection.spec is not None else spec.mode),
            "spec_tile_width": int(self._subpixel_selection.spec.tile_width if self._subpixel_selection.spec is not None else spec.tile_width),
            "spec_tile_height": int(self._subpixel_selection.spec.tile_height if self._subpixel_selection.spec is not None else spec.tile_height),
            "spec_overlap": int(self._subpixel_selection.spec.overlap if self._subpixel_selection.spec is not None else spec.overlap),
        }
        self._overlay_cache.clear()
        self._derived_cache.clear()
        self._refresh_scene(reset_view=False)
        self._refresh_info()
        self._update_tile_lod()
        self._store_view_settings()
        return True

    def _cleanup_detail_worker(self) -> None:
        if self._detail_worker is not None:
            self._detail_worker.deleteLater()
        if self._detail_thread is not None:
            self._detail_thread.deleteLater()
        self._detail_worker = None
        self._detail_thread = None

    def _stop_detail_worker(self) -> None:
        worker = self._detail_worker
        if worker is not None:
            request_cancel = getattr(worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()
        thread = self._detail_thread
        if thread is not None:
            try:
                thread.quit()
            except Exception:
                pass
            if thread.isRunning():
                thread.wait(30000)

    def _cleanup_confidence_worker(
        self,
        worker: DetailConfidenceWorker | None = None,
        thread: QThread | None = None,
        generation: int | None = None,
    ) -> None:
        target_worker = worker or self._confidence_worker
        target_thread = thread or self._confidence_thread
        if target_worker is not None:
            target_worker.deleteLater()
        if target_thread is not None:
            target_thread.deleteLater()
        self._retired_confidence_workers = [
            (retired_worker, retired_thread)
            for retired_worker, retired_thread in self._retired_confidence_workers
            if retired_worker is not target_worker and retired_thread is not target_thread
        ]
        if generation is None or int(generation) == int(self._confidence_request_generation):
            if worker is None or worker is self._confidence_worker:
                self._confidence_worker = None
            if thread is None or thread is self._confidence_thread:
                self._confidence_thread = None
            self._loading_confidence_model_id = None
            self._set_confidence_loading_state(False)

    def _cancel_confidence_worker(self) -> None:
        if self._confidence_worker is not None:
            request_cancel = getattr(self._confidence_worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()

    def _stop_confidence_worker(self) -> None:
        active_pairs: list[tuple[DetailConfidenceWorker | None, QThread | None]] = [
            (self._confidence_worker, self._confidence_thread),
            *list(self._retired_confidence_workers),
        ]
        for worker, _thread in active_pairs:
            if worker is None:
                continue
            request_cancel = getattr(worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()
        for _worker, thread in active_pairs:
            if thread is None:
                continue
            try:
                thread.quit()
            except Exception:
                pass
            if thread.isRunning():
                thread.wait(30000)
        self._retired_confidence_workers.clear()
        self._confidence_worker = None
        self._confidence_thread = None
        self._loading_confidence_model_id = None
        self._set_confidence_loading_state(False)

    def _on_detail_worker_failed(self, message: str, *, generation: int | None = None, worker_kind: str = "detail") -> None:
        if generation is not None:
            active_generation = self._confidence_request_generation if worker_kind == "confidence" else self._detail_request_generation
            if int(generation) != int(active_generation):
                return
        self._payload_loading = False
        self._set_confidence_loading_state(False)
        self.comparison_score_card.set_payload(
            self._t("details.frame_score"),
            "-",
            self._comparison_score_style(None),
            str(message),
        )

    def _confidence_payload_ready(self, model_id: str | None) -> bool:
        if model_id is None:
            return False
        confidence_row = self._confidence_metrics_for_model(model_id)
        if confidence_row is None:
            return False
        if self._is_point_geometry():
            return hasattr(confidence_row, 'mean_point_confidence')
        return hasattr(confidence_row, 'mean_object_confidence')

    def _result_kind_requires_confidence(self, kind: str | None = None) -> bool:
        result_kind = str(kind or self._selected_result_kind())
        return result_kind == "confidence" or result_kind.startswith("confidence_")

    def _start_confidence_loading(self, model_id: str | None) -> None:
        if model_id is None or not self._payload:
            return
        if self._confidence_payload_ready(model_id):
            return
        if self._loading_confidence_model_id == model_id and self._confidence_thread is not None:
            return
        previous_worker = self._confidence_worker
        previous_thread = self._confidence_thread
        self._cancel_confidence_worker()
        if previous_worker is not None and previous_thread is not None:
            self._retired_confidence_workers.append((previous_worker, previous_thread))
        self._confidence_request_generation += 1
        generation = int(self._confidence_request_generation)
        max_side = int(getattr(self._build_result.options, "analysis_max_side", 0) or 0) or None
        thread = QThread(self)
        worker = DetailConfidenceWorker(self._record, self._build_result, model_id, max_side, self._payload)
        self._confidence_thread = thread
        self._confidence_worker = worker
        self._loading_confidence_model_id = model_id
        self._set_confidence_loading_state(True, model_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda row, mid=model_id, g=generation: self._on_confidence_loaded(mid, row, generation=g))
        worker.failed.connect(lambda message, g=generation: self._on_detail_worker_failed(message, generation=g, worker_kind="confidence"))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(lambda w=worker, t=thread, g=generation: self._cleanup_confidence_worker(w, t, g))
        thread.start()

    def _on_confidence_loaded(self, model_id: str | None, confidence_row, *, generation: int | None = None) -> None:
        if generation is not None and int(generation) != int(self._confidence_request_generation):
            return
        if self._loading_confidence_model_id is not None and model_id != self._loading_confidence_model_id:
            return
        if model_id is None or confidence_row is None:
            return
        (self._payload.setdefault("model_confidence", {}))[model_id] = confidence_row
        self._overlay_cache.clear()
        self._derived_cache.clear()
        self._set_confidence_loading_state(False)
        self._refresh_scene(reset_view=False)
        self._refresh_info()

    def _ensure_confidence_ready(self) -> None:
        if not self._result_kind_requires_confidence():
            return
        model_id = self._confidence_model_id()
        if model_id is None or self._payload_loading:
            return
        self._start_confidence_loading(model_id)

    def _apply_selected_model(self, model_id: str | None) -> None:
        probabilities = self._payload.get("model_probabilities") or {}
        masks = self._payload.get("model_masks") or {}
        selected_model_id = model_id if model_id in probabilities else (next(iter(probabilities.keys()), None))
        fallback_prob = np.zeros_like(next(iter(probabilities.values()))) if probabilities else np.zeros((1, 1), dtype=np.float32)
        selected_prob = probabilities.get(selected_model_id, fallback_prob)
        selected_mask = masks.get(selected_model_id, np.asarray(selected_prob >= 0.5, dtype=bool))
        self._payload["selected_model_id"] = selected_model_id
        self._payload["selected_prob"] = np.asarray(selected_prob, dtype=np.float32)
        self._payload["selected_mask"] = np.asarray(selected_mask, dtype=bool)

    def _selected_model_for_current_comparison(self) -> str | None:
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        for source_key in (first_key, second_key):
            if isinstance(source_key, str) and source_key.startswith("model:"):
                return source_key.split(":", 1)[1]
        return self._build_result.model_specs[0].model_id if self._build_result.model_specs else None

    def _preferred_confidence_model_id(self) -> str | None:
        metric_key = str(self._preferred_metric_key or "")
        if '::' not in metric_key:
            return None
        family, model_id = metric_key.split('::', 1)
        if family in {"model_confidence", "model_uncertain_fraction", "model_point_contrast"}:
            return model_id
        return None

    def _confidence_model_id(self) -> str | None:
        preferred_model_id = self._preferred_confidence_model_id()
        if preferred_model_id:
            return preferred_model_id
        return self._selected_model_for_current_comparison()

    def _confidence_metrics_for_model(self, model_id: str | None):
        if model_id is None:
            return None
        return (self._payload.get("model_confidence") or {}).get(model_id)

    def _model_confidence_output_available(self, model_id: str | None) -> bool:
        if model_id is None:
            return False
        availability = self._payload.get("model_confidence_output_available")
        if isinstance(availability, dict) and model_id in availability:
            return bool(availability.get(model_id))
        return bool((getattr(self._record, "model_prob_paths", {}) or {}).get(model_id))

    def _confidence_debug_for_model(self, model_id: str | None):
        confidence_row = self._confidence_metrics_for_model(model_id)
        if confidence_row is None:
            return None
        return getattr(confidence_row, 'debug_data', None)

    def _confidence_debug_mask(self, model_id: str | None, key: str) -> np.ndarray | None:
        debug_data = self._confidence_debug_for_model(model_id)
        if debug_data is None:
            return None
        if key == 'final_mask':
            value = getattr(debug_data, 'merged_mask', None)
        elif key == 'low_mask':
            value = getattr(debug_data, 'low_mask', None)
        elif key == 'high_mask':
            value = getattr(debug_data, 'high_mask', None)
        elif key == 'adaptive_low_mask':
            value = getattr(debug_data, 'adaptive_low_mask', None)
        elif key == 'adaptive_high_mask':
            value = getattr(debug_data, 'adaptive_high_mask', None)
        elif key == 'core_seeds':
            value = getattr(debug_data, 'core_seed_mask', None)
        elif key == 'candidate_region':
            value = getattr(debug_data, 'candidate_region_mask', None)
        elif key == 'thin_barrier':
            value = getattr(debug_data, 'thin_barrier_map', None)
        elif key == 'bridge_cuts':
            value = getattr(debug_data, 'bridge_cut_mask', None)
        elif key == 'barrier_stops':
            value = getattr(debug_data, 'barrier_stop_mask', None)
        else:
            value = (getattr(debug_data, 'branch_masks', {}) or {}).get(key)
        if value is None:
            return None
        return np.asarray(value, dtype=bool)

    def _confidence_debug_float_map(self, model_id: str | None, key: str) -> np.ndarray | None:
        debug_data = self._confidence_debug_for_model(model_id)
        if debug_data is None:
            return None
        value = getattr(debug_data, key, None)
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32)

    def _confidence_object_labels(self, model_id: str | None) -> np.ndarray | None:
        debug_data = self._confidence_debug_for_model(model_id)
        if debug_data is None:
            return None
        value = getattr(debug_data, 'object_labels', None)
        if value is None:
            return None
        labels = np.asarray(value)
        if labels.ndim != 2:
            return None
        return labels.astype(np.int32, copy=False)

    @staticmethod
    def _label_boundary_mask(labels: np.ndarray) -> np.ndarray:
        labels_int = np.asarray(labels, dtype=np.int32)
        positive = labels_int > 0
        if not np.any(positive):
            return np.zeros_like(labels_int, dtype=bool)
        boundary = np.asarray(_boundary_mask(positive), dtype=bool)
        horizontal_diff = (
            (labels_int[:, 1:] != labels_int[:, :-1])
            & (labels_int[:, 1:] > 0)
            & (labels_int[:, :-1] > 0)
        )
        vertical_diff = (
            (labels_int[1:, :] != labels_int[:-1, :])
            & (labels_int[1:, :] > 0)
            & (labels_int[:-1, :] > 0)
        )
        boundary[:, 1:] |= horizontal_diff
        boundary[1:, :] |= vertical_diff
        return boundary

    def _model_threshold(self, model_id: str | None) -> float:
        if model_id is None:
            return 0.5
        for spec in self._build_result.model_specs:
            if spec.model_id == model_id:
                return float(spec.threshold)
        return 0.5

    def _polygon_confidence_render_data(self, model_id: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        cache_key = ("polygon_confidence_render", model_id)
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 3:
            return cached
        if model_id is None:
            return None
        model_probabilities = self._payload.get("model_probabilities") or {}
        model_probability = model_probabilities.get(model_id)
        if model_probability is None:
            return None
        probability = np.clip(np.asarray(model_probability, dtype=np.float32), 0.0, 1.0)
        if probability.ndim != 2 or probability.size == 0:
            return None
        source_mask = None
        model_masks = self._payload.get("model_masks") or {}
        model_mask = model_masks.get(model_id)
        if model_mask is not None:
            source_mask = np.asarray(model_mask, dtype=bool)
        if source_mask is None:
            source_mask = np.asarray(probability >= self._model_threshold(model_id), dtype=bool)
        strong_mask = np.asarray(source_mask, dtype=bool)
        if strong_mask.shape != probability.shape:
            strong_mask = np.zeros_like(probability, dtype=bool)
        support_weights = _support_weights_from_probability(probability, POLYGON_SUPPORT_THRESHOLD)
        support_mask = np.asarray(support_weights > 0.0, dtype=bool)
        if not np.any(support_mask):
            support_mask = strong_mask
        display_confidence = _confidence_display_map_from_probability(
            probability,
            support_mask=support_mask,
            support_threshold=POLYGON_SUPPORT_THRESHOLD,
        )
        result = (
            np.asarray(display_confidence, dtype=np.float32),
            np.asarray(support_weights, dtype=np.float32),
            np.asarray(support_mask, dtype=bool),
        )
        self._derived_cache[cache_key] = result
        return result

    def _confidence_heatmap(self, model_id: str | None, *, uncertainty: bool) -> np.ndarray:
        if model_id is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        probabilities = self._payload.get("model_probabilities") or {}
        model_prob = probabilities.get(model_id)
        if model_prob is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        prob = np.asarray(model_prob, dtype=np.float32)
        tile_rect = self._selection_crop_rect()
        if tile_rect is not None:
            prob = self._crop_ndarray(prob, tile_rect)
        if self._is_point_geometry():
            model_view = (self._payload.get("model_views") or {}).get(model_id)
            result = np.zeros_like(prob, dtype=np.float32)
            if model_view is None:
                return result
            points = tuple(getattr(model_view, 'points', ()))
            x_offset = 0.0
            y_offset = 0.0
            if tile_rect is not None:
                x0 = float(tile_rect.left())
                y0 = float(tile_rect.top())
                x1 = float(tile_rect.right())
                y1 = float(tile_rect.bottom())
                x_offset = x0
                y_offset = y0
                points = tuple(
                    point for point in points
                    if x0 <= float(getattr(point, 'x', 0.0)) <= x1 and y0 <= float(getattr(point, 'y', 0.0)) <= y1
                )
            for point in points:
                x = float(getattr(point, 'x', 0.0)) - x_offset
                y = float(getattr(point, 'y', 0.0)) - y_offset
                px = int(round(x))
                py = int(round(y))
                point_prob = float(prob[py, px]) if 0 <= py < prob.shape[0] and 0 <= px < prob.shape[1] else 0.0
                point_conf = float(2.0 * abs(point_prob - 0.5))
                value = float(1.0 - point_conf) if uncertainty else point_conf
                _paint_disk(result, x, y, max(1.0, float(getattr(point, 'radius', 0.0))) + 1.0, value)
            return np.clip(result, 0.0, 1.0).astype(np.float32)
        render_data = self._polygon_confidence_render_data(model_id)
        if render_data is None:
            return np.zeros_like(prob, dtype=np.float32)
        display_confidence, _support_weights, _support_mask = render_data
        display_confidence = np.asarray(display_confidence, dtype=np.float32)
        return np.clip(1.0 - display_confidence if uncertainty else display_confidence, 0.0, 1.0).astype(np.float32, copy=False)

    def _confidence_bad_area_threshold(self) -> float:
        delta = float(self._payload.get("confidence_uncertainty_delta") or MODEL_CONFIDENCE_UNCERTAIN_DELTA)
        if not np.isfinite(delta):
            return float(DEFAULT_CONFIDENCE_BAD_AREA_THRESHOLD)
        return float(np.clip(1.0 - 2.0 * max(0.0, delta), 0.0, 0.999))

    def _model_output_uncertainty_map(self, model_id: str | None) -> np.ndarray:
        if model_id is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        if not self._model_confidence_output_available(model_id):
            return np.zeros_like(self._base_array(), dtype=np.float32)
        probabilities = self._payload.get("model_probabilities") or {}
        confidence_map = probabilities.get(model_id)
        if confidence_map is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        values = np.asarray(confidence_map, dtype=np.float32)
        tile_rect = self._selection_crop_rect()
        if tile_rect is not None:
            values = np.asarray(self._crop_ndarray(values, tile_rect), dtype=np.float32)
        return build_model_uncertainty(values)

    def _algorithmic_uncertainty_map(self, model_id: str | None) -> np.ndarray:
        if model_id is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        if self._is_point_geometry():
            point_map = self._source_point_map(f"model:{model_id}")
            source = np.asarray(point_map, dtype=np.float32) if point_map is not None else None
        else:
            masks = self._payload.get("model_masks") or {}
            source = masks.get(model_id)
        if source is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        values = np.asarray(source, dtype=np.float32)
        tile_rect = self._selection_crop_rect()
        if tile_rect is not None:
            values = np.asarray(self._crop_ndarray(values, tile_rect), dtype=np.float32)
        return build_algorithmic_uncertainty(values, boundary_radius=float(self._payload.get("boundary_radius") or 1.0) + 2.0)

    def _bad_area_intensity_pixmap(self, intensity: np.ndarray, color: QColor, *, alpha_scale: float = 245.0) -> QPixmap:
        alpha = np.clip(np.asarray(intensity, dtype=np.float32), 0.0, 1.0)
        if alpha.ndim != 2 or alpha.size == 0:
            return QPixmap()
        height, width = alpha.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = np.clip(np.round(alpha * float(alpha_scale)), 0.0, 255.0).astype(np.uint8)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(image)

    def _confidence_bad_areas_pixmap(self, model_id: str | None) -> QPixmap:
        threshold = self._confidence_bad_area_threshold()
        cache_key = (
            "confidence_bad_areas",
            model_id,
            round(float(threshold), 4),
            self._selection_cache_key(),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        uncertainty = self._model_output_uncertainty_map(model_id)
        intensity = confidence_bad_area_intensity(uncertainty, threshold=threshold)
        pixmap = self._bad_area_intensity_pixmap(intensity, QColor(255, 48, 80, 255), alpha_scale=250.0)
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _confidence_mix_pixmap(self, model_id: str | None) -> QPixmap:
        threshold = self._confidence_bad_area_threshold()
        cache_key = (
            "confidence_mix",
            model_id,
            round(float(threshold), 4),
            self._selection_cache_key(),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        algorithmic_uncertainty = self._algorithmic_uncertainty_map(model_id)
        model_uncertainty = self._model_output_uncertainty_map(model_id)
        combined = combine_uncertainty_maps(
            algorithmic_uncertainty,
            model_uncertainty,
            algorithmic_threshold=threshold,
            model_threshold=threshold,
        )
        agreement = np.asarray(combined.agreement, dtype=np.float32)
        algorithmic_only = np.asarray(combined.algorithmic_only, dtype=np.float32)
        model_only = np.asarray(combined.model_only, dtype=np.float32)
        if agreement.ndim != 2 or agreement.size == 0:
            return QPixmap()
        height, width = agreement.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        both_color = np.asarray([255.0, 216.0, 32.0], dtype=np.float32)
        algorithmic_color = np.asarray([0.0, 210.0, 255.0], dtype=np.float32)
        model_color = np.asarray([255.0, 0.0, 160.0], dtype=np.float32)
        both_weight = 1.45 * agreement
        algorithmic_weight = 0.85 * algorithmic_only
        model_weight = 0.95 * model_only
        total = both_weight + algorithmic_weight + model_weight
        active = total > 1e-6
        red = np.zeros_like(total, dtype=np.float32)
        green = np.zeros_like(total, dtype=np.float32)
        blue = np.zeros_like(total, dtype=np.float32)
        red[active] = (
            both_weight[active] * both_color[0]
            + algorithmic_weight[active] * algorithmic_color[0]
            + model_weight[active] * model_color[0]
        ) / total[active]
        green[active] = (
            both_weight[active] * both_color[1]
            + algorithmic_weight[active] * algorithmic_color[1]
            + model_weight[active] * model_color[1]
        ) / total[active]
        blue[active] = (
            both_weight[active] * both_color[2]
            + algorithmic_weight[active] * algorithmic_color[2]
            + model_weight[active] * model_color[2]
        ) / total[active]
        alpha = np.clip(0.95 * agreement + 0.62 * algorithmic_only + 0.68 * model_only, 0.0, 1.0)
        rgba[..., 0] = np.clip(np.round(red), 0.0, 255.0).astype(np.uint8)
        rgba[..., 1] = np.clip(np.round(green), 0.0, 255.0).astype(np.uint8)
        rgba[..., 2] = np.clip(np.round(blue), 0.0, 255.0).astype(np.uint8)
        rgba[..., 3] = np.clip(np.round(alpha * 255.0), 0.0, 255.0).astype(np.uint8)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(image)
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _confidence_color(self, value: float) -> tuple[int, int, int]:
        score = float(max(0.0, min(1.0, value)))
        if score <= 0.5:
            factor = score / 0.5
            return 255, int(round(200.0 * factor + 40.0)), 48
        factor = (score - 0.5) / 0.5
        return int(round(255.0 * (1.0 - factor))), 220, 70

    def _confidence_color_arrays(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        score = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
        red = np.empty(score.shape, dtype=np.uint8)
        green = np.empty(score.shape, dtype=np.uint8)
        blue = np.empty(score.shape, dtype=np.uint8)
        low_mask = score <= 0.5
        if np.any(low_mask):
            factor = np.clip(score[low_mask] / 0.5, 0.0, 1.0)
            red[low_mask] = 255
            green[low_mask] = np.clip(np.round(200.0 * factor + 40.0), 0.0, 255.0).astype(np.uint8)
            blue[low_mask] = 48
        high_mask = ~low_mask
        if np.any(high_mask):
            factor = np.clip((score[high_mask] - 0.5) / 0.5, 0.0, 1.0)
            red[high_mask] = np.clip(np.round(255.0 * (1.0 - factor)), 0.0, 255.0).astype(np.uint8)
            green[high_mask] = 220
            blue[high_mask] = 70
        return red, green, blue

    def _point_object_confidence_overlay_pixmap(self, model_id: str | None) -> QPixmap:
        cache_key = ('point_confidence', model_id, self._selection_cache_key())
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        if model_id is None:
            return QPixmap()
        confidence_row = self._confidence_metrics_for_model(model_id)
        if confidence_row is None:
            return QPixmap()
        objects = tuple(getattr(confidence_row, 'objects', ()))
        tile_rect = self._selection_crop_rect()
        x_offset = 0.0
        y_offset = 0.0
        if tile_rect is not None:
            x0 = float(tile_rect.left())
            y0 = float(tile_rect.top())
            x1 = float(tile_rect.right())
            y1 = float(tile_rect.bottom())
            x_offset = x0
            y_offset = y0
            objects = tuple(
                point_row for point_row in objects
                if x0 <= float(getattr(point_row, 'x', 0.0)) <= x1 and y0 <= float(getattr(point_row, 'y', 0.0)) <= y1
            )
        if not objects:
            return QPixmap()
        base = self._base_array()
        height, width = base.shape
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for point_row in objects:
            score = float(getattr(point_row, 'local_confidence', getattr(point_row, 'center_confidence', 0.0)))
            red, green, blue = self._confidence_color(score)
            point_probability = float(getattr(point_row, 'point_probability', 0.0))
            support_weight = float(max(0.0, min(1.0, (point_probability - float(POINT_SUPPORT_THRESHOLD)) / max(1e-8, 1.0 - float(POINT_SUPPORT_THRESHOLD))))) if point_probability >= float(POINT_SUPPORT_THRESHOLD) else 0.0
            color = QColor(red, green, blue, int(round(80.0 + 175.0 * support_weight)))
            radius = max(2, int(round(float(getattr(point_row, 'radius', 1.0)) + 1.0 + 1.5 * score)))
            painter.setPen(QPen(color, 1.4))
            painter.setBrush(color)
            x = int(round(float(getattr(point_row, 'x', 0.0)) - x_offset))
            y = int(round(float(getattr(point_row, 'y', 0.0)) - y_offset))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
        painter.end()
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _polygon_object_confidence_overlay_pixmap(self, model_id: str | None) -> QPixmap:
        tile_rect = self._selection_crop_rect()
        cache_key = (
            'polygon_confidence_frame',
            model_id,
            int(self._payload.get('boundary_radius') or 1),
            self._selection_cache_key(),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        render_data = self._polygon_confidence_render_data(model_id)
        if render_data is None:
            return QPixmap()
        confidence_map, support_weights, support_mask = render_data
        if tile_rect is not None:
            confidence_map = np.asarray(self._crop_ndarray(np.asarray(confidence_map, dtype=np.float32), tile_rect), dtype=np.float32)
            support_weights = np.asarray(self._crop_ndarray(np.asarray(support_weights, dtype=np.float32), tile_rect), dtype=np.float32)
            support_mask = np.asarray(self._crop_ndarray(np.asarray(support_mask, dtype=bool), tile_rect), dtype=bool)
        if not np.any(support_mask):
            return QPixmap()
        height, width = confidence_map.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        red, green, blue = self._confidence_color_arrays(confidence_map)
        final_alpha = np.clip(np.round(235.0 * support_weights), 0.0, 255.0).astype(np.uint8)
        rgba[..., 0][support_mask] = red[support_mask]
        rgba[..., 1][support_mask] = green[support_mask]
        rgba[..., 2][support_mask] = blue[support_mask]
        rgba[..., 3][support_mask] = final_alpha[support_mask]
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(image)
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _confidence_debug_mask_pixmap(self, model_id: str | None, key: str, color: QColor) -> QPixmap:
        cache_key = (
            'confidence_debug_mask',
            model_id,
            key,
            color.red(),
            color.green(),
            color.blue(),
            color.alpha(),
            self._selection_cache_key(),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        mask = self._confidence_debug_mask(model_id, key)
        tile_rect = self._selection_crop_rect()
        if mask is not None and tile_rect is not None:
            mask = np.asarray(self._crop_ndarray(np.asarray(mask, dtype=bool), tile_rect), dtype=bool)
        pixmap = self._mask_to_pixmap(mask, color) if mask is not None else QPixmap()
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _confidence_object_contours_pixmap(self, model_id: str | None) -> QPixmap:
        cache_key = ('confidence_contours', model_id, self._selection_cache_key())
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        object_labels = self._confidence_object_labels(model_id)
        if object_labels is not None:
            contour_mask = self._label_boundary_mask(object_labels)
        else:
            final_mask = self._confidence_debug_mask(model_id, 'final_mask')
            if final_mask is None:
                return QPixmap()
            contour_mask = _boundary_mask(final_mask)
        if contour_mask is None:
            return QPixmap()
        tile_rect = self._selection_crop_rect()
        if tile_rect is not None:
            contour_mask = np.asarray(self._crop_ndarray(np.asarray(contour_mask, dtype=bool), tile_rect), dtype=bool)
        pixmap = self._mask_to_pixmap(contour_mask, self._named_color("boundary_mask"))
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _confidence_branch_debug_pixmap(self, model_id: str | None) -> QPixmap:
        cache_key = ('confidence_branch_debug', model_id, self._selection_cache_key())
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        branch_masks = {
            'large_polygon': self._confidence_debug_mask(model_id, 'large_polygon'),
            'global_hysteresis': self._confidence_debug_mask(model_id, 'global_hysteresis'),
            'elongated': self._confidence_debug_mask(model_id, 'elongated'),
            'small_weak': self._confidence_debug_mask(model_id, 'small_weak'),
            'adaptive_local': self._confidence_debug_mask(model_id, 'adaptive_local'),
        }
        base = self._base_array()
        height, width = base.shape
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        tile_rect = self._selection_crop_rect()
        color_map = {
            'large_polygon': (255, 225, 70),
            'global_hysteresis': (80, 210, 255),
            'elongated': (255, 170, 0),
            'small_weak': (255, 0, 140),
            'adaptive_local': (100, 255, 120),
        }
        for branch_key, rgb in color_map.items():
            branch_mask = branch_masks.get(branch_key)
            if branch_mask is None:
                continue
            branch_bool = np.asarray(branch_mask, dtype=bool)
            if tile_rect is not None:
                branch_bool = np.asarray(self._crop_ndarray(branch_bool, tile_rect), dtype=bool)
            rgba[..., 0][branch_bool] = rgb[0]
            rgba[..., 1][branch_bool] = rgb[1]
            rgba[..., 2][branch_bool] = rgb[2]
            rgba[..., 3][branch_bool] = np.maximum(rgba[..., 3][branch_bool], 155)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(image)
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _confidence_overlay_pixmap(self, model_id: str | None) -> QPixmap:
        tile_rect = self._selection_crop_rect()
        cache_key = ('confidence_frame_overlay', model_id, self._selection_cache_key())
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        if model_id is None:
            return QPixmap()
        if self._is_point_geometry():
            confidence_pixmap = self._point_object_confidence_overlay_pixmap(model_id)
            if not confidence_pixmap.isNull():
                self._overlay_cache[cache_key] = confidence_pixmap
                return confidence_pixmap
        render_data = self._polygon_confidence_render_data(model_id)
        if render_data is None:
            return QPixmap()
        confidence_map, support_weights, support_mask = render_data
        if tile_rect is not None:
            confidence_map = np.asarray(self._crop_ndarray(np.asarray(confidence_map, dtype=np.float32), tile_rect), dtype=np.float32)
            support_weights = np.asarray(self._crop_ndarray(np.asarray(support_weights, dtype=np.float32), tile_rect), dtype=np.float32)
            support_mask = np.asarray(self._crop_ndarray(np.asarray(support_mask, dtype=bool), tile_rect), dtype=bool)
        height, width = confidence_map.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        red, green, blue = self._confidence_color_arrays(confidence_map)
        final_alpha = np.clip(np.round(235.0 * support_weights), 0.0, 255.0).astype(np.uint8)
        rgba[..., 0][support_mask] = red[support_mask]
        rgba[..., 1][support_mask] = green[support_mask]
        rgba[..., 2][support_mask] = blue[support_mask]
        rgba[..., 3][support_mask] = final_alpha[support_mask]
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(image)
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _comparison_model_ids(self) -> list[str]:
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        result: list[str] = []
        for source_key in (first_key, second_key):
            if isinstance(source_key, str) and source_key.startswith("model:"):
                model_id = source_key.split(":", 1)[1]
                if model_id not in result:
                    result.append(model_id)
        return result

    def _pairwise_comparison_entry(self, model_a_id: str | None, model_b_id: str | None) -> dict[str, float | str] | None:
        if not model_a_id or not model_b_id:
            return None
        pairwise = self._payload.get("pairwise_model_comparisons") or ()
        for row in pairwise:
            left = str(row.get("model_a", ""))
            right = str(row.get("model_b", ""))
            if {left, right} == {str(model_a_id), str(model_b_id)}:
                return row
        return None

    def _selected_comparison_score_info(self) -> tuple[str, float | None, list[str]]:
        model_ids = self._comparison_model_ids()
        group_key = self._current_comparison_group()
        preferred_confidence_model = self._preferred_confidence_model_id()
        if preferred_confidence_model is None and self._selected_result_kind() == "confidence":
            preferred_confidence_model = self._confidence_model_id()
        if preferred_confidence_model is not None:
            confidence_row = self._confidence_metrics_for_model(preferred_confidence_model)
            if confidence_row is not None:
                if hasattr(confidence_row, 'mean_object_confidence'):
                    lines = [
                        f"frame_uncertainty_score: {float(getattr(confidence_row, 'frame_uncertainty_score', 0.0)):.4f}",
                        f"mean_uncertainty: {float(getattr(confidence_row, 'mean_uncertainty', 0.0)):.4f}",
                        f"low_conf_fraction: {float(getattr(confidence_row, 'low_conf_fraction', 0.0)):.4f}",
                        f"worst_tail_uncertainty: {float(getattr(confidence_row, 'worst_tail_uncertainty', 0.0)):.4f}",
                        f"largest_low_conf_component: {float(getattr(confidence_row, 'largest_low_conf_component', 0.0)):.4f}",
                        f"mean_frame_confidence: {float(getattr(confidence_row, 'mean_object_confidence', 0.0)):.4f}",
                        f"mean_frame_probability: {float(getattr(confidence_row, 'mean_object_probability', 0.0)):.4f}",
                        f"uncertain_fraction: {float(getattr(confidence_row, 'uncertain_fraction', 0.0)):.4f}",
                        f"uncertain_support_fraction: {float(getattr(confidence_row, 'uncertain_support_fraction', 0.0)):.4f}",
                        f"top_uncertainty_mean: {float(getattr(confidence_row, 'top_uncertainty_mean', 0.0)):.4f}",
                        f"largest_uncertain_region_fraction: {float(getattr(confidence_row, 'largest_uncertain_region_fraction', 0.0)):.4f}",
                        f"focus_area_fraction: {float(getattr(confidence_row, 'object_area_fraction', 0.0)):.4f}",
                    ]
                    return (
                        self._t("details.frame_score"),
                        float(getattr(confidence_row, 'frame_uncertainty_score', 0.0)),
                        lines,
                    )
                lines = [
                    f"frame_uncertainty_score: {float(getattr(confidence_row, 'frame_uncertainty_score', 0.0)):.4f}",
                    f"mean_uncertainty: {float(getattr(confidence_row, 'mean_uncertainty', 0.0)):.4f}",
                    f"low_conf_fraction: {float(getattr(confidence_row, 'low_conf_fraction', 0.0)):.4f}",
                    f"worst_tail_uncertainty: {float(getattr(confidence_row, 'worst_tail_uncertainty', 0.0)):.4f}",
                    f"largest_low_conf_component: {float(getattr(confidence_row, 'largest_low_conf_component', 0.0)):.4f}",
                    f"mean_point_confidence: {float(getattr(confidence_row, 'mean_point_confidence', 0.0)):.4f}",
                    f"mean_center_confidence: {float(getattr(confidence_row, 'mean_center_confidence', 0.0)):.4f}",
                    f"mean_local_confidence: {float(getattr(confidence_row, 'mean_local_confidence', 0.0)):.4f}",
                    f"mean_point_probability: {float(getattr(confidence_row, 'mean_point_probability', 0.0)):.4f}",
                    f"uncertain_support_fraction: {float(getattr(confidence_row, 'uncertain_support_fraction', 0.0)):.4f}",
                    f"top_uncertainty_mean: {float(getattr(confidence_row, 'top_uncertainty_mean', 0.0)):.4f}",
                    f"largest_uncertain_region_fraction: {float(getattr(confidence_row, 'largest_uncertain_region_fraction', 0.0)):.4f}",
                    f"mean_point_contrast: {float(getattr(confidence_row, 'mean_point_contrast', 0.0)):.4f}",
                    f"point_count: {int(getattr(confidence_row, 'point_count', 0))}",
                ]
                return (
                    self._t("details.frame_score"),
                    float(getattr(confidence_row, 'frame_uncertainty_score', 0.0)),
                    lines,
                )
            quick = self._quick_confidence_score(preferred_confidence_model)
            if quick is not None:
                if self._is_point_geometry():
                    lines = [
                        f"frame_uncertainty_score: {quick['frame_uncertainty_score']:.4f}",
                        f"mean_uncertainty: {quick['mean_uncertainty']:.4f}",
                        f"low_conf_fraction: {quick['low_conf_fraction']:.4f}",
                        f"worst_tail_uncertainty: {quick['worst_tail_uncertainty']:.4f}",
                        f"largest_low_conf_component: {quick['largest_low_conf_component']:.4f}",
                        f"mean_point_confidence: {quick['mean_confidence']:.4f}",
                        f"mean_point_probability: {quick['mean_probability']:.4f}",
                        f"uncertain_fraction: {quick['uncertain_fraction']:.4f}",
                        f"uncertain_support_fraction: {quick['uncertain_support_fraction']:.4f}",
                        f"top_uncertainty_mean: {quick['top_uncertainty_mean']:.4f}",
                        f"largest_uncertain_region_fraction: {quick['largest_uncertain_region_fraction']:.4f}",
                        f"object_area_fraction: {quick['focus_fraction']:.4f}",
                    ]
                    return self._t("details.frame_score"), float(quick['frame_uncertainty_score']), lines
                lines = [
                    f"frame_uncertainty_score: {quick['frame_uncertainty_score']:.4f}",
                    f"mean_uncertainty: {quick['mean_uncertainty']:.4f}",
                    f"low_conf_fraction: {quick['low_conf_fraction']:.4f}",
                    f"worst_tail_uncertainty: {quick['worst_tail_uncertainty']:.4f}",
                    f"largest_low_conf_component: {quick['largest_low_conf_component']:.4f}",
                    f"mean_frame_confidence: {quick['mean_confidence']:.4f}",
                    f"mean_frame_probability: {quick['mean_probability']:.4f}",
                    f"uncertain_fraction: {quick['uncertain_fraction']:.4f}",
                    f"uncertain_support_fraction: {quick['uncertain_support_fraction']:.4f}",
                    f"top_uncertainty_mean: {quick['top_uncertainty_mean']:.4f}",
                    f"largest_uncertain_region_fraction: {quick['largest_uncertain_region_fraction']:.4f}",
                    f"focus_area_fraction: {quick['focus_fraction']:.4f}",
                ]
                return self._t("details.frame_score"), float(quick['frame_uncertainty_score']), lines
        if group_key == "model_model":
            if len(model_ids) >= 2:
                pairwise = self._pairwise_comparison_entry(model_ids[0], model_ids[1])
                if pairwise is not None:
                    score = float(pairwise.get("agreement_score", 0.0))
                    if self._is_point_geometry():
                        return (
                            self._t("details.frame_score"),
                            score,
                            [
                                f"active precision: {float(pairwise.get('precision', 0.0)):.4f}",
                                f"active recall: {float(pairwise.get('recall', 0.0)):.4f}",
                                f"active f1_at_r: {float(pairwise.get('f1', 0.0)):.4f}",
                                f"active mean_localization_error: {float(pairwise.get('mean_localization_error', 0.0)):.4f}",
                                f"active localization_agreement: {float(pairwise.get('localization_agreement', 0.0)):.4f}",
                                f"active count_agreement: {float(pairwise.get('count_agreement', 0.0)):.4f}",
                            ],
                        )
                    return (
                        self._t("details.frame_score"),
                        score,
                        [
                            f"active soft_dice: {float(pairwise.get('soft_dice', 0.0)):.4f}",
                            f"active soft_iou: {float(pairwise.get('soft_iou', 0.0)):.4f}",
                            f"active ssim: {float(pairwise.get('ssim', 0.0)):.4f}",
                            f"active dice: {float(pairwise.get('dice', 0.0)):.4f}",
                            f"active iou: {float(pairwise.get('iou', 0.0)):.4f}",
                            f"active hausdorff_distance: {float(pairwise.get('hausdorff_distance', 0.0)):.4f}",
                            f"active centroid_distance: {float(pairwise.get('centroid_distance', 0.0)):.4f}",
                            f"auxiliary mae: {float(pairwise.get('mae', 0.0)):.4f}",
                            f"auxiliary rmse: {float(pairwise.get('rmse', 0.0)):.4f}",
                            f"auxiliary count_agreement: {float(pairwise.get('count_agreement', 0.0)):.4f}",
                        ],
                    )
            return self._t("details.frame_score"), None, [self._t("details.score_unavailable")]
        if group_key == "model_labeled":
            model_id = model_ids[0] if model_ids else None
            model_metrics = self._payload.get("model_metrics") or {}
            selected = model_metrics.get(model_id) if model_id is not None else None
            if selected is not None:
                score = float(selected.quality_score)
                if hasattr(selected, "f1_at_radius"):
                    return (
                        self._t("details.frame_score"),
                        score,
                        [
                            f"active precision: {float(selected.precision_at_radius):.4f}",
                            f"active recall: {float(selected.recall_at_radius):.4f}",
                            f"active f1_at_r: {float(selected.f1_at_radius):.4f}",
                            f"active mean_localization_error: {float(selected.mean_localization_error):.4f}",
                            f"active localization_score: {float(selected.localization_score):.4f}",
                            f"active count_error: {float(selected.count_error):.4f}",
                            f"auxiliary chamfer_score: {float(selected.chamfer_score):.4f}",
                            f"auxiliary hausdorff_score: {float(selected.hausdorff_score):.4f}",
                        ],
                    )
                return (
                    self._t("details.frame_score"),
                    score,
                    [
                        f"active soft_dice: {float(selected.soft_dice):.4f}",
                        f"active soft_iou: {float(selected.soft_iou):.4f}",
                        f"active ssim: {float(selected.ssim):.4f}",
                        f"active dice: {float(selected.dice):.4f}",
                        f"active iou: {float(selected.iou):.4f}",
                        f"active hausdorff_distance: {float(selected.hausdorff_distance):.4f}",
                        f"active centroid_distance: {float(selected.centroid_distance):.4f}",
                        f"auxiliary mae: {float(selected.mae):.4f}",
                        f"auxiliary rmse: {float(selected.rmse):.4f}",
                        f"auxiliary precision: {float(selected.precision):.4f}",
                        f"auxiliary recall: {float(selected.recall):.4f}",
                        f"auxiliary count_error: {float(selected.count_error):.4f}",
                        f"auxiliary connected_component_error: {float(selected.connected_component_error):.4f}",
                    ],
                )
            return self._t("details.frame_score"), None, [self._t("details.score_unavailable")]
        return self._t("details.frame_score"), None, []

    def _selected_subpixel_score_info(self) -> tuple[str, float | None, list[str]]:
        if self._subpixel_selection is None:
            return self._t("details.subpixel_score"), None, []
        selection = self._subpixel_selection
        spec = selection.spec or self._default_subpixel_spec()
        lines = [
            f"{self._t('details.subpixel_selection')}: r{int(selection.sub_row) + 1}, c{int(selection.sub_column) + 1}",
            f"{self._t('details.parent_frame')}: r{int(selection.parent_row) + 1}, c{int(selection.parent_column) + 1}",
            f"{int(spec.rows)} x {int(spec.columns)}",
        ]
        return self._t("details.subpixel_score"), float(selection.subpixel_value), lines

    def _on_model_a_changed(self, *_args) -> None:
        self._refresh_scene(reset_view=False)
        self._refresh_info()
        self._ensure_confidence_ready()
        self._store_view_settings()

    def _on_model_pair_changed(self, *_args) -> None:
        self._apply_selected_model(self._selected_model_for_current_comparison())
        self._refresh_scene(reset_view=False)
        self._refresh_info()
        self._ensure_confidence_ready()
        self._store_view_settings()

    def _refresh_scene_from_controls(self, *_args) -> None:
        self._refresh_scene(reset_view=False)
        self._refresh_info()
        if self._result_kind_requires_confidence():
            self._ensure_confidence_ready()
        else:
            self._set_confidence_loading_state(False)
        source_mode = self._selected_layer_view() == "source"
        self.first_color.setEnabled(not source_mode)
        self.second_color.setEnabled(not source_mode)
        self._store_view_settings()

    def _set_confidence_loading_state(self, active: bool, model_id: str | None = None) -> None:
        if not active:
            self.confidence_loading_label.setVisible(False)
            self.confidence_loading_bar.setVisible(False)
            return
        name = self._model_display_name(model_id) if model_id is not None else "model"
        self.confidence_loading_label.setText(f"Loading confidence: {name}")
        self.confidence_loading_label.setVisible(True)
        self.confidence_loading_bar.setVisible(True)

    def _selected_layer_view(self) -> str:
        return str(self.layer_view_combo.currentData() or "binary")

    def _tile_selection_rect(self, plan: TileGridPlan | None = None) -> QRectF | None:
        if self._tile_selection is None:
            return None
        resolved_plan = plan or self._tile_plan_for_view()
        if resolved_plan is None:
            return None
        row, column = self._tile_selection
        if row < 0 or column < 0 or row >= int(resolved_plan.rows) or column >= int(resolved_plan.columns):
            return None
        return self._tile_rect_for_index(int(row), int(column), resolved_plan)

    def _selection_crop_rect(self) -> QRectF | None:
        if self._subpixel_selection is not None:
            return self._subpixel_selection_rect()
        return self._tile_selection_rect()

    def _selection_cache_key(self) -> tuple[int, int, int, int] | None:
        rect = self._selection_crop_rect()
        if rect is None:
            return None
        return (
            max(0, int(np.floor(float(rect.left())))),
            max(0, int(np.floor(float(rect.top())))),
            max(0, int(np.ceil(float(rect.right())))),
            max(0, int(np.ceil(float(rect.bottom())))),
        )

    def _subpixel_selection_rect(self) -> QRectF | None:
        if self._subpixel_selection is None:
            return None
        spec = self._subpixel_selection.spec
        if spec is None:
            spec = self._default_subpixel_spec()
        full = self._full_base_array()
        left, top, width, height = subpixel_bounds_for_index(
            float(full.shape[1]),
            float(full.shape[0]),
            int(self._subpixel_selection.sub_row),
            int(self._subpixel_selection.sub_column),
            spec,
        )
        if width <= 0 or height <= 0:
            return None
        return QRectF(float(left), float(top), float(width), float(height))

    @staticmethod
    def _crop_ndarray(array: np.ndarray, rect: QRectF | None) -> np.ndarray:
        if rect is None:
            return array
        if not isinstance(array, np.ndarray) or array.ndim < 2:
            return array
        left = max(0, int(np.floor(float(rect.left()))))
        top = max(0, int(np.floor(float(rect.top()))))
        right = min(int(array.shape[1]), int(np.ceil(float(rect.right()))))
        bottom = min(int(array.shape[0]), int(np.ceil(float(rect.bottom()))))
        if right <= left or bottom <= top:
            return array[:0, :0]
        if array.ndim == 2:
            return array[top:bottom, left:right]
        return array[top:bottom, left:right, ...]

    def _full_base_array(self) -> np.ndarray:
        cache_key = ("base_array", str(self._payload.get("selected_model_id") or ""))
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, np.ndarray):
            return cached
        original = self._payload.get("original_gray")
        if original is not None:
            base_array = np.asarray(original, dtype=np.uint8)
            if base_array.size > 0:
                self._derived_cache[cache_key] = base_array
                return base_array
        selected_prob = np.asarray(self._payload.get("selected_prob"), dtype=np.float32)
        if selected_prob.ndim == 2 and selected_prob.size > 0:
            base_array = np.clip(np.rint(selected_prob * 255.0), 0, 255).astype(np.uint8)
            self._derived_cache[cache_key] = base_array
            return base_array
        fallback = np.zeros((1, 1), dtype=np.uint8)
        self._derived_cache[cache_key] = fallback
        return fallback

    def _base_array(self) -> np.ndarray:
        base = self._full_base_array()
        rect = self._subpixel_selection_rect()
        if rect is None:
            return base
        return self._crop_ndarray(base, rect)

    def _source_mask(self, source_key: str | None) -> np.ndarray | None:
        cache_key = ("source_mask", source_key, bool(self._is_point_geometry()))
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, np.ndarray):
            return self._crop_ndarray(cached, self._selection_crop_rect())
        if self._is_point_geometry():
            point_map = self._source_point_map(source_key)
            mask = np.asarray(point_map > 0.1, dtype=bool) if point_map is not None else None
            if mask is not None:
                self._derived_cache[cache_key] = mask
            return self._crop_ndarray(mask, self._selection_crop_rect()) if mask is not None else None
        if source_key == "gt":
            gt_mask = self._payload.get("gt_mask")
            mask = np.asarray(gt_mask, dtype=bool) if gt_mask is not None else None
            if mask is not None:
                self._derived_cache[cache_key] = mask
            return self._crop_ndarray(mask, self._selection_crop_rect()) if mask is not None else None
        if isinstance(source_key, str) and source_key.startswith("model:"):
            model_id = source_key.split(":", 1)[1]
            masks = self._payload.get("model_masks") or {}
            model_mask = masks.get(model_id)
            mask = np.asarray(model_mask, dtype=bool) if model_mask is not None else None
            if mask is not None:
                self._derived_cache[cache_key] = mask
            return self._crop_ndarray(mask, self._selection_crop_rect()) if mask is not None else None
        return None

    def _source_float_map(self, source_key: str | None) -> np.ndarray | None:
        if self._is_point_geometry():
            point_map = self._source_point_map(source_key)
            value = np.asarray(point_map, dtype=np.float32) if point_map is not None else None
            return self._crop_ndarray(value, self._selection_crop_rect()) if value is not None else None
        if source_key == "original":
            original = self._payload.get("original_gray")
            value = np.asarray(original, dtype=np.float32) / 255.0 if original is not None else None
            return self._crop_ndarray(value, self._selection_crop_rect()) if value is not None else None
        if source_key == "gt":
            gt_mask = self._payload.get("gt_mask")
            value = np.asarray(gt_mask, dtype=np.float32) if gt_mask is not None else None
            return self._crop_ndarray(value, self._selection_crop_rect()) if value is not None else None
        if isinstance(source_key, str) and source_key.startswith("model:"):
            model_id = source_key.split(":", 1)[1]
            probabilities = self._payload.get("model_probabilities") or {}
            model_prob = probabilities.get(model_id)
            value = np.asarray(model_prob, dtype=np.float32) if model_prob is not None else None
            return self._crop_ndarray(value, self._selection_crop_rect()) if value is not None else None
        return None

    def _source_render_float_map(self, source_key: str | None) -> np.ndarray | None:
        if source_key == "original":
            original = self._payload.get("original_gray")
            value = np.asarray(original, dtype=np.float32) / 255.0 if original is not None else None
            return self._crop_ndarray(value, self._selection_crop_rect()) if value is not None else None
        if source_key == "gt":
            gt_mask = self._payload.get("gt_mask")
            value = np.asarray(gt_mask, dtype=np.float32) if gt_mask is not None else None
            return self._crop_ndarray(value, self._selection_crop_rect()) if value is not None else None
        if isinstance(source_key, str) and source_key.startswith("model:"):
            model_id = source_key.split(":", 1)[1]
            probabilities = self._payload.get("model_probabilities") or {}
            model_prob = probabilities.get(model_id)
            value = np.asarray(model_prob, dtype=np.float32) if model_prob is not None else None
            return self._crop_ndarray(value, self._selection_crop_rect()) if value is not None else None
        return self._source_float_map(source_key)

    def _refresh_result_kind_options(self, preferred_kind: str | None = None) -> None:
        current_kind = str(
            preferred_kind
            or self._sticky_result_kind
            or self.result_kind_combo.currentData()
            or ""
        )
        if not current_kind:
            if self._preferred_confidence_model_id() is not None:
                current_kind = "confidence"
            elif self._is_point_geometry():
                current_kind = "point_matches"
            else:
                current_kind = "diff"
        confidence_model_id = self._confidence_model_id()
        confidence_output_available = self._model_confidence_output_available(confidence_model_id)
        if not confidence_output_available and current_kind in {"confidence_bad_areas", "confidence_mix"}:
            current_kind = "point_matches" if self._is_point_geometry() else "diff"
        items: list[tuple[str, str]] = [(self._t("details.comparison_difference"), "diff")]
        if self._is_point_geometry():
            items.append((self._t("details.point_matches"), "point_matches"))
        else:
            items.append((self._t("details.boundary_difference"), "boundary"))
        items.append((self._t("details.model_confidence_map"), "confidence"))
        if confidence_output_available:
            items.append((self._t("details.confidence_bad_areas"), "confidence_bad_areas"))
            items.append((self._t("details.confidence_mix"), "confidence_mix"))
        self.result_kind_combo.blockSignals(True)
        self.result_kind_combo.clear()
        for label, key in items:
            self.result_kind_combo.addItem(label, key)
            combo_index = self.result_kind_combo.count() - 1
            self.result_kind_combo.setItemData(combo_index, self._result_kind_hint(str(key)) or "", Qt.ItemDataRole.ToolTipRole)
        index = self.result_kind_combo.findData(current_kind)
        if index < 0 and self._sticky_result_kind is not None:
            index = self.result_kind_combo.findData(self._sticky_result_kind)
        self.result_kind_combo.setCurrentIndex(index if index >= 0 else 0)
        self.result_kind_combo.blockSignals(False)
        self._sticky_result_kind = self._selected_result_kind()
        self.result_kind_combo.setToolTip(self._result_kind_hint(self._selected_result_kind()) or '')

    def _selected_result_kind(self) -> str:
        return str(self.result_kind_combo.currentData() or "diff")

    def _is_point_geometry(self) -> bool:
        return str(self._payload.get("geometry_mode") or "mask") == "point"

    def _source_point_map(self, source_key: str | None) -> np.ndarray | None:
        cache_key = ("source_point_map", source_key)
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, np.ndarray):
            return cached
        if source_key == "gt":
            gt_view = self._payload.get("gt_point_view")
            point_map = _point_map_from_view(gt_view) if gt_view is not None else None
            if point_map is not None:
                self._derived_cache[cache_key] = point_map
            return point_map
        if isinstance(source_key, str) and source_key.startswith("model:"):
            model_id = source_key.split(":", 1)[1]
            model_views = self._payload.get("model_views") or {}
            view = model_views.get(model_id)
            point_map = _point_map_from_view(view) if view is not None else None
            if point_map is not None:
                self._derived_cache[cache_key] = point_map
            return point_map
        return None

    def _source_point_view(self, source_key: str | None):
        if source_key == 'gt':
            return self._payload.get('gt_point_view')
        if isinstance(source_key, str) and source_key.startswith('model:'):
            model_id = source_key.split(':', 1)[1]
            return (self._payload.get('model_views') or {}).get(model_id)
        return None

    def _boundary_input(self, source_key: str | None) -> np.ndarray:
        mask = self._source_mask(source_key)
        if mask is None:
            return np.zeros_like(self._base_array(), dtype=bool)
        return np.asarray(_boundary_mask(mask), dtype=bool)

    def _point_match_pairs(self, first_points: tuple[object, ...], second_points: tuple[object, ...], radius: float) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
        threshold = max(0.0, float(radius))
        candidate_pairs: list[tuple[float, int, int]] = []
        for index_a, point_a in enumerate(first_points):
            for index_b, point_b in enumerate(second_points):
                distance = float(np.hypot(float(getattr(point_a, 'x', 0.0)) - float(getattr(point_b, 'x', 0.0)), float(getattr(point_a, 'y', 0.0)) - float(getattr(point_b, 'y', 0.0))))
                if distance <= threshold:
                    candidate_pairs.append((distance, index_a, index_b))
        candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        pairs: list[tuple[int, int, float]] = []
        matched_a: set[int] = set()
        matched_b: set[int] = set()
        for distance, index_a, index_b in candidate_pairs:
            if index_a in matched_a or index_b in matched_b:
                continue
            matched_a.add(index_a)
            matched_b.add(index_b)
            pairs.append((index_a, index_b, distance))
        return pairs, matched_a, matched_b

    def _point_matches_pixmap(self, first_key: str | None, second_key: str | None) -> QPixmap:
        cache_key = (
            "point_matches",
            first_key,
            second_key,
            float(self._payload.get("point_match_radius") or 3.0),
            self._selection_cache_key(),
            self._named_color("difference_mask").name(QColor.NameFormat.HexArgb),
            self._named_color("first_mask").name(QColor.NameFormat.HexArgb),
            self._named_color("second_mask").name(QColor.NameFormat.HexArgb),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        base = self._base_array()
        height, width = base.shape
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        first_view = self._source_point_view(first_key)
        second_view = self._source_point_view(second_key)
        first_points = tuple(getattr(first_view, 'points', ())) if first_view is not None else tuple()
        second_points = tuple(getattr(second_view, 'points', ())) if second_view is not None else tuple()
        tile_rect = self._selection_crop_rect()
        x_offset = 0.0
        y_offset = 0.0
        if tile_rect is not None:
            x0 = float(tile_rect.left())
            y0 = float(tile_rect.top())
            x1 = float(tile_rect.right())
            y1 = float(tile_rect.bottom())
            x_offset = x0
            y_offset = y0
            first_points = tuple(
                point for point in first_points
                if x0 <= float(getattr(point, 'x', 0.0)) <= x1 and y0 <= float(getattr(point, 'y', 0.0)) <= y1
            )
            second_points = tuple(
                point for point in second_points
                if x0 <= float(getattr(point, 'x', 0.0)) <= x1 and y0 <= float(getattr(point, 'y', 0.0)) <= y1
            )
        pairs, matched_first, matched_second = self._point_match_pairs(first_points, second_points, float(self._payload.get('point_match_radius') or 3.0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        line_pen = QPen(self._named_color("difference_mask"), 1.5)
        painter.setPen(line_pen)
        for index_a, index_b, _distance in pairs:
            point_a = first_points[index_a]
            point_b = second_points[index_b]
            painter.drawLine(
                int(round(float(getattr(point_a, 'x', 0.0)) - x_offset)),
                int(round(float(getattr(point_a, 'y', 0.0)) - y_offset)),
                int(round(float(getattr(point_b, 'x', 0.0)) - x_offset)),
                int(round(float(getattr(point_b, 'y', 0.0)) - y_offset)),
            )
        for index, point in enumerate(first_points):
            radius = max(2, int(round(float(getattr(point, 'radius', 0.0)) + 1.0)))
            color = QColor(70, 220, 120, 240) if index in matched_first else self._named_color("first_mask")
            painter.setPen(QPen(color, 1.2))
            painter.setBrush(color)
            x = int(round(float(getattr(point, 'x', 0.0)) - x_offset))
            y = int(round(float(getattr(point, 'y', 0.0)) - y_offset))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
        for index, point in enumerate(second_points):
            if index in matched_second:
                continue
            radius = max(2, int(round(float(getattr(point, 'radius', 0.0)) + 1.0)))
            color = self._named_color("second_mask")
            painter.setPen(QPen(color, 1.2))
            painter.setBrush(color)
            x = int(round(float(getattr(point, 'x', 0.0)) - x_offset))
            y = int(round(float(getattr(point, 'y', 0.0)) - y_offset))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
        painter.end()
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _current_difference_mode_label(self) -> str:
        return self._t("details.grayscale_difference") if self.grayscale_diff_checkbox.isChecked() else self._t("details.difference")

    def _current_operation_mode(self) -> ComparisonMode:
        return ComparisonMode.GRAYSCALE_DIFF if self.grayscale_diff_checkbox.isChecked() else ComparisonMode.DISAGREEMENT

    def _update_result_controls(self) -> None:
        show_difference_controls = self._selected_result_kind() == "diff"
        self.operation_label.setVisible(show_difference_controls)
        self.grayscale_diff_checkbox.setVisible(show_difference_controls)
        show_mask_colors = self._selected_layer_view() == "binary"
        self.first_mask_color_button.setVisible(show_mask_colors)
        self.second_mask_color_button.setVisible(show_mask_colors)
        show_result_color = show_mask_colors and self._selected_result_kind() in {"diff", "boundary"}
        self.result_mask_color_button.setVisible(show_result_color)
        self.result_mask_color_button.setEnabled(show_result_color)
        self._update_color_button_styles()

    def _result_overlay_title(self) -> str:
        kind = self._selected_result_kind()
        if kind == "boundary":
            return self._t("details.boundary_difference")
        if kind == "point_matches":
            return self._t("details.point_matches")
        if kind == "confidence":
            return self._t("details.model_confidence_map")
        if kind == "confidence_bad_areas":
            return self._t("details.confidence_bad_areas")
        if kind == "confidence_mix":
            return self._t("details.confidence_mix")
        return self._t("details.difference_heatmap")

    def _refresh_dynamic_labels(self) -> None:
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        self.first_source_layer_title.setText(self._source_display_name(first_key))
        self.second_source_layer_title.setText(self._source_display_name(second_key))
        self.result_layer_title.setText(self._result_overlay_title())
        self._update_color_button_styles()

    def _source_display_name(self, source_key: str | None) -> str:
        if source_key == "original":
            return self._t("details.original")
        if source_key == "gt":
            return self._t("details.ground_truth")
        if isinstance(source_key, str) and source_key.startswith("model:"):
            return self._model_display_name(source_key.split(":", 1)[1])
        return str(source_key or "-")

    def _source_pixmap(self, source_key: str | None, color: QColor, *, prefer_grayscale: bool) -> QPixmap:
        cache_key = (
            "source_pixmap",
            source_key,
            self._selected_layer_view(),
            bool(prefer_grayscale),
            bool(self._is_point_geometry()),
            self._selection_cache_key(),
            color.name(QColor.NameFormat.HexArgb),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        view_mode = self._selected_layer_view()
        render_float_map = self._source_render_float_map(source_key)
        float_map = self._source_float_map(source_key)
        if source_key == "original" and render_float_map is not None:
            pixmap = self._grayscale_to_pixmap(np.clip(np.rint(render_float_map * 255.0), 0, 255).astype(np.uint8))
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        if view_mode == "source" and render_float_map is not None:
            pixmap = self._grayscale_to_pixmap(np.clip(np.rint(render_float_map * 255.0), 0, 255).astype(np.uint8))
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        if self._is_point_geometry() and float_map is not None and source_key != "original":
            pixmap = self._intensity_to_pixmap(np.asarray(float_map, dtype=np.float32), color)
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        if self._is_point_geometry() and source_key == "original" and float_map is not None:
            pixmap = self._intensity_to_pixmap(np.asarray(float_map, dtype=np.float32), color)
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        if prefer_grayscale and render_float_map is not None:
            pixmap = self._grayscale_to_pixmap(np.clip(np.rint(render_float_map * 255.0), 0, 255).astype(np.uint8))
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        mask = self._source_mask(source_key)
        if mask is not None:
            pixmap = self._mask_to_pixmap(mask, color)
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        if render_float_map is not None:
            pixmap = self._grayscale_to_pixmap(np.clip(np.rint(render_float_map * 255.0), 0, 255).astype(np.uint8))
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        empty = QPixmap()
        self._overlay_cache[cache_key] = empty
        return empty

    def _comparison_input(self, source_key: str | None, mode: ComparisonMode) -> np.ndarray:
        if mode == ComparisonMode.GRAYSCALE_DIFF:
            value = self._source_float_map(source_key)
            if value is not None:
                return np.asarray(value, dtype=np.float32)
            reference = self._base_array()
            return np.zeros_like(reference, dtype=np.float32)
        value = self._source_mask(source_key)
        if value is not None:
            return np.asarray(value, dtype=bool)
        reference = self._base_array()
        return np.zeros_like(reference, dtype=bool)

    def _pairwise_comparison_inputs(self, first_key: str | None, second_key: str | None, mode: ComparisonMode) -> tuple[np.ndarray, np.ndarray]:
        return self._comparison_input(first_key, mode), self._comparison_input(second_key, mode)

    def _comparison_overlay_result(
        self,
        first_key: str | None,
        second_key: str | None,
        mode: ComparisonMode,
        color: QColor,
    ) -> tuple[QPixmap, float]:
        cache_key = (
            "comparison_overlay",
            first_key,
            second_key,
            str(mode.value),
            self._selection_cache_key(),
            color.name(QColor.NameFormat.HexArgb),
        )
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            pixmap, score = cached
            if isinstance(pixmap, QPixmap):
                return pixmap, float(score)
        first, second = self._pairwise_comparison_inputs(first_key, second_key, mode)
        heatmap, score = compute_comparison(first, second, mode)
        pixmap = self._intensity_to_pixmap(np.asarray(heatmap, dtype=np.float32), color)
        result = (pixmap, float(score))
        self._derived_cache[cache_key] = result
        return result

    def _boundary_overlay_result(self, first_key: str | None, second_key: str | None, color: QColor) -> tuple[QPixmap, float]:
        cache_key = (
            "boundary_overlay",
            first_key,
            second_key,
            self._selection_cache_key(),
            color.name(QColor.NameFormat.HexArgb),
        )
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            pixmap, score = cached
            if isinstance(pixmap, QPixmap):
                return pixmap, float(score)
        first = self._boundary_input(first_key)
        second = self._boundary_input(second_key)
        heatmap, score = compute_comparison(first, second, ComparisonMode.DISAGREEMENT)
        pixmap = self._intensity_to_pixmap(np.asarray(heatmap, dtype=np.float32), color)
        result = (pixmap, float(score))
        self._derived_cache[cache_key] = result
        return result

    def _refresh_scene(self, *, reset_view: bool) -> None:
        if self._payload_loading and not self._payload:
            self._tile_plan = None
            self._tile_selection = None
            self.original_item.setPixmap(QPixmap())
            self.first_source_item.setPixmap(QPixmap())
            self.second_source_item.setPixmap(QPixmap())
            self.result_item.setPixmap(QPixmap())
            self.tile_preview_item.setPixmap(QPixmap())
            self.tile_grid_item.setPixmap(QPixmap())
            self.tile_selection_item.setVisible(False)
            self._update_layer_states()
            return
        base_array = self._base_array()
        height, width = base_array.shape
        scene = self.overlay_view.scene()
        assert scene is not None
        scene.setSceneRect(0.0, 0.0, float(width), float(height))
        base_pixmap_key = (
            "base_pixmap",
            str(self._payload.get("selected_model_id") or ""),
            self._tile_selection,
            None
            if self._subpixel_selection is None
            else (
                int(self._subpixel_selection.parent_row),
                int(self._subpixel_selection.parent_column),
                int(self._subpixel_selection.sub_row),
                int(self._subpixel_selection.sub_column),
                int(getattr(self._subpixel_selection.spec, "rows", 0) or 0),
                int(getattr(self._subpixel_selection.spec, "columns", 0) or 0),
                str(getattr(self._subpixel_selection.spec, "mode", "grid") or "grid"),
                int(getattr(self._subpixel_selection.spec, "tile_width", 0) or 0),
                int(getattr(self._subpixel_selection.spec, "tile_height", 0) or 0),
                int(getattr(self._subpixel_selection.spec, "overlap", 0) or 0),
            ),
        )
        base_pixmap = self._overlay_cache.get(base_pixmap_key)
        if base_pixmap is None:
            base_pixmap = self._grayscale_to_pixmap(base_array)
            self._overlay_cache[base_pixmap_key] = base_pixmap
        self.original_item.setPixmap(base_pixmap)
        self._refresh_tile_overlays(base_array)
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        mode = self._current_operation_mode()
        prefer_grayscale = self._selected_result_kind() == "diff" and mode == ComparisonMode.GRAYSCALE_DIFF
        self.first_source_item.setPixmap(self._source_pixmap(first_key, self._named_color("first_mask"), prefer_grayscale=prefer_grayscale))
        self.second_source_item.setPixmap(self._source_pixmap(second_key, self._named_color("second_mask"), prefer_grayscale=prefer_grayscale))
        self._refresh_result_layer()
        self._refresh_dynamic_labels()
        self._update_result_controls()
        self._update_layer_states()
        if reset_view:
            self._schedule_reset_view()

    def _refresh_result_layer(self) -> None:
        if self._payload_loading and not self._payload:
            self._comparison_score = None
            self.result_item.setPixmap(QPixmap())
            self._update_layer_states()
            return
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        mode = self._current_operation_mode()
        kind = self._selected_result_kind()
        confidence_model_id = self._confidence_model_id()
        if kind == "confidence":
            self._comparison_score = None
            self.result_item.setPixmap(self._confidence_overlay_pixmap(confidence_model_id))
        elif kind == "confidence_bad_areas":
            self._comparison_score = None
            self.result_item.setPixmap(self._confidence_bad_areas_pixmap(confidence_model_id))
        elif kind == "confidence_mix":
            self._comparison_score = None
            self.result_item.setPixmap(self._confidence_mix_pixmap(confidence_model_id))
        elif kind == "boundary":
            pixmap, score = self._boundary_overlay_result(first_key, second_key, self._named_color("boundary_mask"))
            self._comparison_score = float(score)
            self.result_item.setPixmap(pixmap)
        elif kind == "point_matches":
            self._comparison_score = None
            self.result_item.setPixmap(self._point_matches_pixmap(first_key, second_key))
        else:
            pixmap, score = self._comparison_overlay_result(first_key, second_key, mode, self._named_color("difference_mask"))
            self._comparison_score = float(score)
            self.result_item.setPixmap(pixmap)
        self._update_layer_states()

    def _tile_plan_for_view(self) -> TileGridPlan | None:
        if str(getattr(self._build_result.options, "tile_mode", "pixel") or "pixel") != "tile":
            return None
        base = self._full_base_array()
        if base.ndim != 2 or base.size == 0:
            return None
        return plan_tile_grid(
            base.shape,
            int(getattr(self._build_result.options, "tile_width", 256)),
            int(getattr(self._build_result.options, "tile_height", 256)),
            int(getattr(self._build_result.options, "tile_overlap", 0)),
        )

    def _tile_rect_for_index(self, row: int, column: int, plan: TileGridPlan) -> QRectF:
        left, top, width, height = tile_bounds_for_index(plan, row, column)
        return QRectF(float(left), float(top), float(width), float(height))

    def _tile_mosaic_pixmap(self, base_array: np.ndarray, plan: TileGridPlan) -> QPixmap:
        cache_key = (
            "tile_mosaic",
            id(base_array),
            int(plan.tile_width),
            int(plan.tile_height),
            int(plan.overlap),
            int(plan.rows),
            int(plan.columns),
        )
        cached = self._overlay_cache.get(cache_key)
        if isinstance(cached, QPixmap):
            return cached
        mosaic = np.ascontiguousarray(np.asarray(base_array, dtype=np.uint8)).copy()
        for row in range(int(plan.rows)):
            for column in range(int(plan.columns)):
                left, top, width, height = tile_bounds_for_index(plan, row, column)
                if width <= 0 or height <= 0:
                    continue
                tile = mosaic[top:top + height, left:left + width]
                if tile.size == 0:
                    continue
                mean_value = int(np.clip(np.rint(float(np.mean(tile, dtype=np.float64))), 0, 255))
                mosaic[top:top + height, left:left + width] = mean_value
        pixmap = self._grayscale_to_pixmap(mosaic)
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _tile_grid_pixmap(self, plan: TileGridPlan) -> QPixmap:
        cache_key = (
            "tile_grid",
            int(plan.source_shape[0]),
            int(plan.source_shape[1]),
            int(plan.tile_width),
            int(plan.tile_height),
            int(plan.overlap),
            int(plan.rows),
            int(plan.columns),
        )
        cached = self._overlay_cache.get(cache_key)
        if isinstance(cached, QPixmap):
            return cached
        source_height, source_width = (int(plan.source_shape[0]), int(plan.source_shape[1]))
        image = QImage(max(1, source_width), max(1, source_height), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setPen(QPen(QColor(92, 196, 255, 180), 1.0))
        for row in range(int(plan.rows)):
            for column in range(int(plan.columns)):
                rect = self._tile_rect_for_index(row, column, plan)
                if rect.isNull():
                    continue
                painter.drawRect(rect)
        painter.end()
        pixmap = QPixmap.fromImage(image)
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _update_tile_hint(self) -> None:
        plan = self._tile_plan
        if plan is None and self._subpixel_selection is None:
            self.zoom_hint_label.hide()
            return
        if self._subpixel_selection is not None:
            selection = self._subpixel_selection
            spec = selection.spec or self._default_subpixel_spec()
            hint = self._t(
                "details.subpixel_view.hint",
                parent_row=int(selection.parent_row) + 1,
                parent_column=int(selection.parent_column) + 1,
                sub_row=int(selection.sub_row) + 1,
                sub_column=int(selection.sub_column) + 1,
                rows=int(spec.rows),
                columns=int(spec.columns),
                value=float(selection.subpixel_value),
                parent_value=float(selection.parent_value),
            )
            self.zoom_hint_label.setText(hint)
            self.zoom_hint_label.show()
            return
        scale = max(0.01, float(self.overlay_view.transform().m11()))
        tile_screen_size = min(int(plan.tile_width), int(plan.tile_height)) * scale if plan is not None else 0.0
        mode_text = self._t("details.tile_view.zoomed_in") if tile_screen_size >= DETAIL_PIXEL_VIEW_THRESHOLD else self._t("details.tile_view.zoomed_out")
        hint = self._t(
            "details.tile_view.hint",
            width=int(plan.tile_width) if plan is not None else 0,
            height=int(plan.tile_height) if plan is not None else 0,
            overlap=int(plan.overlap) if plan is not None else 0,
            mode=mode_text,
        )
        if self._tile_selection is not None:
            row, column = self._tile_selection
            hint = f"{hint} | {self._t('details.tile_view.selected')}: r{row + 1}, c{column + 1}"
        self.zoom_hint_label.setText(hint)
        self.zoom_hint_label.show()

    def _update_tile_lod(self) -> None:
        if self._subpixel_selection is not None:
            self.original_item.setVisible(True)
            self.tile_preview_item.setVisible(False)
            self.tile_grid_item.setVisible(False)
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        plan = self._tile_plan
        if plan is None:
            self.tile_preview_item.setVisible(False)
            self.tile_grid_item.setVisible(False)
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        if self._tile_selection is not None:
            self.tile_preview_item.setVisible(False)
            self.tile_grid_item.setVisible(False)
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        base_visible = self.original_item.isVisible()
        if not base_visible:
            self.original_item.setVisible(False)
            self.tile_preview_item.setVisible(False)
            self.tile_grid_item.setVisible(False)
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        scale = max(0.01, float(self.overlay_view.transform().m11()))
        tile_screen_size = min(int(plan.tile_width), int(plan.tile_height)) * scale
        show_pixel_view = tile_screen_size >= DETAIL_PIXEL_VIEW_THRESHOLD
        self.original_item.setVisible(show_pixel_view)
        self.tile_preview_item.setVisible(not show_pixel_view)
        self.tile_preview_item.setOpacity(self.original_item.opacity())
        self.tile_grid_item.setVisible(show_pixel_view)
        self.tile_grid_item.setOpacity(0.85)
        if self._tile_selection is not None and show_pixel_view:
            row, column = self._tile_selection
            rect = self._tile_rect_for_index(row, column, plan)
            self.tile_selection_item.setRect(rect)
            self.tile_selection_item.setVisible(True)
            self.tile_selection_item.setOpacity(1.0)
        else:
            self.tile_selection_item.setVisible(False)
        self._update_tile_hint()

    def _on_view_clicked(self, scene_pos: QPointF) -> None:
        plan = self._tile_plan
        if plan is None or not self.original_item.isVisible() or self._tile_selection is not None:
            return
        scale = max(0.01, float(self.overlay_view.transform().m11()))
        if min(int(plan.tile_width), int(plan.tile_height)) * scale < DETAIL_PIXEL_VIEW_THRESHOLD:
            return
        position = QPointF(float(scene_pos.x()), float(scene_pos.y()))
        if position.x() < 0.0 or position.y() < 0.0:
            return
        row_guess = min(max(int(position.y() // max(1, int(plan.stride_y))), 0), max(0, int(plan.rows) - 1))
        column_guess = min(max(int(position.x() // max(1, int(plan.stride_x))), 0), max(0, int(plan.columns) - 1))
        candidates: list[tuple[float, int, int]] = []
        for row in range(max(0, row_guess - 1), min(int(plan.rows), row_guess + 2)):
            for column in range(max(0, column_guess - 1), min(int(plan.columns), column_guess + 2)):
                rect = self._tile_rect_for_index(row, column, plan)
                if rect.contains(position):
                    center = rect.center()
                    distance = abs(center.x() - position.x()) + abs(center.y() - position.y())
                    candidates.append((distance, row, column))
        if not candidates:
            return
        _distance, row, column = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        self._tile_selection = (int(row), int(column))
        self._update_tile_lod()

    def _refresh_tile_overlays(self, base_array: np.ndarray) -> None:
        if self._subpixel_selection is not None:
            self._tile_plan = None
            self.tile_preview_item.setPixmap(QPixmap())
            self.tile_grid_item.setPixmap(QPixmap())
            self._update_tile_lod()
            return
        self._tile_plan = self._tile_plan_for_view()
        plan = self._tile_plan
        if plan is None:
            self._tile_selection = None
            self.tile_preview_item.setPixmap(QPixmap())
            self.tile_grid_item.setPixmap(QPixmap())
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        if self._tile_selection is not None:
            row, column = self._tile_selection
            if row < 0 or column < 0 or row >= int(plan.rows) or column >= int(plan.columns):
                self._tile_selection = None
        if self._tile_selection is not None:
            self.tile_preview_item.setPixmap(QPixmap())
            self.tile_grid_item.setPixmap(QPixmap())
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        self.tile_preview_item.setPixmap(self._tile_mosaic_pixmap(base_array, plan))
        self.tile_grid_item.setPixmap(self._tile_grid_pixmap(plan))
        self._update_tile_lod()

    def _quick_confidence_score(self, model_id: str | None) -> dict[str, float] | None:
        if model_id is None:
            return None
        model_probabilities = self._payload.get("model_probabilities") or {}
        model_probability = model_probabilities.get(model_id)
        if model_probability is None:
            return None
        probability = np.clip(np.asarray(model_probability, dtype=np.float32), 0.0, 1.0)
        if probability.ndim != 2 or probability.size == 0:
            return None
        model_mask = self._source_mask(f"model:{model_id}")
        if model_mask is None:
            model_mask = np.asarray(probability >= self._model_threshold(model_id), dtype=bool)
        confidence = _confidence_map_from_probability(probability)
        support_weights = _support_weights_from_probability(probability, POLYGON_SUPPORT_THRESHOLD)
        support_mask = support_weights > 0.0
        weight_total = float(np.sum(np.asarray(support_weights, dtype=np.float64), dtype=np.float64))
        delta = float(self._payload.get("confidence_uncertainty_delta") or MODEL_CONFIDENCE_UNCERTAIN_DELTA)
        uncertain = np.abs(probability - 0.5) <= max(0.0, delta)
        frame_uncertainty_score, mean_uncertainty, low_conf_fraction, worst_tail_uncertainty, largest_low_conf_component = _frame_uncertainty_components_from_probability(
            probability,
            support_threshold=POLYGON_SUPPORT_THRESHOLD,
        )
        return {
            "mean_confidence": float(np.sum(np.asarray(support_weights * confidence, dtype=np.float64), dtype=np.float64) / max(1e-8, weight_total)) if weight_total > 0.0 else 0.0,
            "mean_probability": float(np.sum(np.asarray(support_weights * probability, dtype=np.float64), dtype=np.float64) / max(1e-8, weight_total)) if weight_total > 0.0 else 0.0,
            "uncertain_fraction": float(np.mean(uncertain[support_mask], dtype=np.float64)) if np.any(support_mask) else 0.0,
            "frame_uncertainty_score": float(frame_uncertainty_score),
            "mean_uncertainty": float(mean_uncertainty),
            "low_conf_fraction": float(low_conf_fraction),
            "worst_tail_uncertainty": float(worst_tail_uncertainty),
            "largest_low_conf_component": float(largest_low_conf_component),
            "uncertain_support_fraction": float(low_conf_fraction),
            "top_uncertainty_mean": float(worst_tail_uncertainty),
            "largest_uncertain_region_fraction": float(largest_low_conf_component),
            "focus_fraction": float(np.count_nonzero(support_mask) / max(1, support_mask.size)),
        }

    def _comparison_score_metric_key(self) -> str:
        return str(self._preferred_metric_key or self._build_result.selected_metric_key or "overall_frame_score")

    def _comparison_score_style(self, value: float | None, metric_key: str | None = None) -> str:
        active_metric = str(metric_key or self._comparison_score_metric_key())
        ratio = metric_visual_ratio(
            active_metric,
            value,
            point_match_radius=float(getattr(self._build_result.options, "point_match_radius", 3.0) or 3.0),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        level_key = metric_level_key(
            active_metric,
            value,
            point_match_radius=float(getattr(self._build_result.options, "point_match_radius", 3.0) or 3.0),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        higher_is_better = metric_higher_is_better(active_metric)
        family = str(active_metric or "").split("::", 1)[0]
        if ratio is None or level_key is None:
            background = "#2f3844"
            foreground = "#edf3fb"
        elif family == "model_confidence":
            if level_key == "score.level.low":
                background = "#1f5f3b"
                foreground = "#e9fff1"
            elif level_key == "score.level.moderate":
                background = "#6f7a18"
                foreground = "#f7ffd8"
            elif level_key == "score.level.elevated":
                background = "#a75d12"
                foreground = "#fff0dc"
            else:
                background = "#8c2f39"
                foreground = "#ffe9ec"
        elif higher_is_better:
            if ratio < 0.33:
                background = "#8c2f39"
                foreground = "#ffe9ec"
            elif ratio < 0.66:
                background = "#8a6a12"
                foreground = "#fff7da"
            else:
                background = "#1f5f3b"
                foreground = "#e9fff1"
        else:
            if ratio < 0.33:
                background = "#1f5f3b"
                foreground = "#e9fff1"
            elif ratio < 0.66:
                background = "#8a6a12"
                foreground = "#fff7da"
            else:
                background = "#8c2f39"
                foreground = "#ffe9ec"
        return f"padding: 8px 12px; border-radius: 10px; background-color: {background}; color: {foreground}; font-weight: 700;"

    def _comparison_score_text(self, value: float | None, metric_key: str | None = None) -> str:
        if value is None:
            return "-"
        active_metric = str(metric_key or self._comparison_score_metric_key())
        level_key = metric_level_key(
            active_metric,
            value,
            point_match_radius=float(getattr(self._build_result.options, "point_match_radius", 3.0) or 3.0),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        if level_key is None:
            return "-"
        level = self._t(level_key)
        if "::" in active_metric:
            return f"{level} {float(value) * 100.0:.1f}%"
        return f"{level} {float(value):.4f}"


    def _refresh_info(self) -> None:
        if self._payload_loading and not self._payload:
            self.comparison_score_card.set_payload(
                self._t("details.selected_comparison_score"),
                "Loading...",
                self._comparison_score_style(None),
                "Loading frame details...",
            )
            return
        if self._subpixel_selection is not None:
            selection = self._subpixel_selection
            self.comparison_score_card.hide()
            self.subpixel_score_card.show()
            spec = selection.spec or self._default_subpixel_spec()
            self.subpixel_selection_value.setText(
                self._t(
                    "details.subpixel_selection_value",
                    parent_row=int(selection.parent_row) + 1,
                    parent_column=int(selection.parent_column) + 1,
                    sub_row=int(selection.sub_row) + 1,
                    sub_column=int(selection.sub_column) + 1,
                    rows=int(spec.rows),
                    columns=int(spec.columns),
                    parent_value=float(selection.parent_value),
                )
            )
            self.subpixel_selection_value.show()
        else:
            self.comparison_score_card.show()
            self.subpixel_selection_value.hide()
            self.subpixel_score_card.hide()
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        result_kind = self._selected_result_kind()

        score_title, score_value, score_lines = self._selected_comparison_score_info()
        score_metric_key = str(self._comparison_score_metric_key() or "overall_frame_score")

        comparison_lines: list[str] = []
        comparison_lines.append(f"{self._t('details.frame_type')}: {self._frame_type_label()}")
        if score_value is not None:
            comparison_lines.append(f"{score_title}: {float(score_value):.4f}")
        else:
            comparison_lines.append(f"{score_title}: {self._t('details.score_unavailable')}")
        comparison_lines.extend(self._localize_metric_lines(score_lines))
        if self._loading_confidence_model_id is not None and self._result_kind_requires_confidence(result_kind):
            comparison_lines.append("Loading confidence visualization...")
        self.comparison_score_card.set_payload(
            score_title,
            self._comparison_score_text(score_value, score_metric_key),
            self._comparison_score_style(score_value, score_metric_key),
            "\n".join(comparison_lines),
            self._selected_score_hint() or "",
        )
        if self._subpixel_selection is not None:
            selection = self._subpixel_selection
            subpixel_metric_key = str(selection.metric_key or score_metric_key)
            subpixel_title, subpixel_value, subpixel_lines = self._selected_subpixel_score_info()
            self.subpixel_score_card.set_payload(
                subpixel_title,
                self._comparison_score_text(subpixel_value, subpixel_metric_key),
                self._comparison_score_style(subpixel_value, subpixel_metric_key),
                "\n".join(self._localize_metric_lines(subpixel_lines)),
                self._selected_score_hint() or "",
            )
        else:
            self.subpixel_score_card.hide()


    def _update_layer_states(self) -> None:
        if self._legacy_base_hold_active:
            show_base = self._hold_preview_mode != "confidence_source"
            self.original_item.setVisible(show_base)
            self.original_item.setOpacity(self.original_opacity.value() / 100.0)
            self.first_source_item.setVisible(False)
            self.second_source_item.setVisible(False)
            self.result_item.setVisible(False)
            if self._hold_preview_mode == "confidence_source" and not self._hold_preview_pixmap.isNull():
                self.hold_preview_item.setPixmap(self._hold_preview_pixmap)
                self.hold_preview_item.setVisible(True)
                self.hold_preview_item.setOpacity(1.0)
            else:
                self.hold_preview_item.setVisible(False)
            self._update_tile_lod()
            return
        self.hold_preview_item.setVisible(False)
        self.original_item.setVisible(self.original_visible.isChecked())
        self.original_item.setOpacity(self.original_opacity.value() / 100.0)
        self.first_source_item.setVisible(self.first_source_visible.isChecked())
        self.first_source_item.setOpacity(self.first_source_opacity.value() / 100.0)
        self.second_source_item.setVisible(self.second_source_visible.isChecked())
        self.second_source_item.setOpacity(self.second_source_opacity.value() / 100.0)
        self.result_item.setVisible(self.result_visible.isChecked())
        self.result_item.setOpacity(self.result_opacity.value() / 100.0)
        self._update_tile_lod()

    def _grayscale_to_pixmap(self, array: np.ndarray) -> QPixmap:
        contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.uint8))
        height, width = contiguous.shape
        image = QImage(contiguous.data, width, height, int(contiguous.strides[0]), QImage.Format.Format_Grayscale8).copy()
        return QPixmap.fromImage(image.convertToFormat(QImage.Format.Format_RGB888))

    def _mask_to_pixmap(self, mask: np.ndarray, color: QColor) -> QPixmap:
        mask_uint8 = np.ascontiguousarray(np.asarray(mask, dtype=bool).astype(np.uint8))
        height, width = mask_uint8.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = mask_uint8 * 255
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(image)

    def _intensity_to_pixmap(self, intensity: np.ndarray, color: QColor) -> QPixmap:
        alpha = np.clip(np.asarray(intensity, dtype=np.float32), 0.0, 1.0)
        if alpha.ndim != 2:
            alpha = np.zeros((1, 1), dtype=np.float32)
        height, width = alpha.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(image)

    def _dual_intensity_to_pixmap(self, confident: np.ndarray, uncertain: np.ndarray, confident_color: QColor, uncertain_color: QColor) -> QPixmap:
        confident_alpha = np.clip(np.asarray(confident, dtype=np.float32), 0.0, 1.0)
        uncertain_alpha = np.clip(np.asarray(uncertain, dtype=np.float32), 0.0, 1.0)
        if confident_alpha.ndim != 2:
            confident_alpha = np.zeros((1, 1), dtype=np.float32)
        if uncertain_alpha.ndim != 2:
            uncertain_alpha = np.zeros_like(confident_alpha)
        if confident_alpha.shape != uncertain_alpha.shape:
            uncertain_alpha = np.zeros_like(confident_alpha)
        height, width = confident_alpha.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        red = uncertain_alpha * float(uncertain_color.red()) + confident_alpha * float(confident_color.red())
        green = uncertain_alpha * float(uncertain_color.green()) + confident_alpha * float(confident_color.green())
        blue = uncertain_alpha * float(uncertain_color.blue()) + confident_alpha * float(confident_color.blue())
        alpha = np.maximum(confident_alpha, uncertain_alpha)
        rgba[..., 0] = np.clip(red, 0.0, 255.0).astype(np.uint8)
        rgba[..., 1] = np.clip(green, 0.0, 255.0).astype(np.uint8)
        rgba[..., 2] = np.clip(blue, 0.0, 255.0).astype(np.uint8)
        rgba[..., 3] = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(image)

    def _named_color(self, key: str) -> QColor:
        defaults = {
            "first_mask": QColor(80, 210, 255, 255),
            "second_mask": QColor(255, 0, 140, 255),
            "difference_mask": QColor(255, 196, 0, 255),
            "boundary_mask": QColor(255, 210, 0, 235),
        }
        value = (self._session_view_state.get("colors") or {}).get(key)
        if isinstance(value, str):
            color = QColor(value)
            if color.isValid():
                return color
        return QColor(defaults[key])

    def _set_named_color(self, key: str, color: QColor) -> None:
        if not color.isValid():
            return
        colors = dict(self._session_view_state.get("colors") or {})
        colors[key] = color.name(QColor.NameFormat.HexArgb)
        self._session_view_state["colors"] = colors
        self._overlay_cache.clear()
        self._derived_cache.clear()
        self._update_color_button_styles()
        self._refresh_scene(reset_view=False)
        self._refresh_info()
        self._store_view_settings()

    def _choose_named_color(self, key: str) -> None:
        initial = self._named_color(key)
        color = QColorDialog.getColor(initial, self, self._t("details.color_dialog_title"))
        if color.isValid():
            self._set_named_color(key, color)

    def _active_result_color_key(self) -> str | None:
        kind = self._selected_result_kind()
        if kind == "boundary":
            return "boundary_mask"
        if kind == "diff":
            return "difference_mask"
        return None

    def _choose_active_result_color(self) -> None:
        key = self._active_result_color_key()
        if key is not None:
            self._choose_named_color(key)

    def _update_color_button_styles(self) -> None:
        mapping = {
            self.first_mask_color_button: "first_mask",
            self.second_mask_color_button: "second_mask",
            self.result_mask_color_button: self._active_result_color_key() or "difference_mask",
        }
        for button, key in mapping.items():
            color = self._named_color(key)
            button.setStyleSheet(
                f"padding: 0px; border-radius: 4px; border: 1px solid #30445a; background-color: {color.name(QColor.NameFormat.HexRgb)};"
            )

    def _build_view_settings_payload(self) -> dict[str, object]:
        payload = {
            "layer_view": self._selected_layer_view(),
            "result_kind": self._selected_result_kind(),
            "grayscale_diff": bool(self.grayscale_diff_checkbox.isChecked()),
            "original_visible": bool(self.original_visible.isChecked()),
            "first_visible": bool(self.first_source_visible.isChecked()),
            "second_visible": bool(self.second_source_visible.isChecked()),
            "result_visible": bool(self.result_visible.isChecked()),
            "original_opacity": int(self.original_opacity.value()),
            "first_opacity": int(self.first_source_opacity.value()),
            "second_opacity": int(self.second_source_opacity.value()),
            "result_opacity": int(self.result_opacity.value()),
            "colors": dict(self._session_view_state.get("colors") or {}),
        }
        if self._tile_selection is not None:
            payload["tile_selection"] = {
                "row": int(self._tile_selection[0]),
                "column": int(self._tile_selection[1]),
            }
        if self._subpixel_selection is not None:
            payload["subpixel_selection"] = {
                "parent_row": int(self._subpixel_selection.parent_row),
                "parent_column": int(self._subpixel_selection.parent_column),
                "sub_row": int(self._subpixel_selection.sub_row),
                "sub_column": int(self._subpixel_selection.sub_column),
                "parent_value": float(self._subpixel_selection.parent_value),
                "subpixel_value": float(self._subpixel_selection.subpixel_value),
                "subpixel_confidence": None if self._subpixel_selection.subpixel_confidence is None else float(self._subpixel_selection.subpixel_confidence),
                "aggregation": str(self._subpixel_selection.aggregation),
                "metric_key": str(self._subpixel_selection.metric_key),
                "spec_rows": int(self._subpixel_selection.spec.rows if self._subpixel_selection.spec is not None else 0),
                "spec_columns": int(self._subpixel_selection.spec.columns if self._subpixel_selection.spec is not None else 0),
                "spec_mode": str(self._subpixel_selection.spec.mode if self._subpixel_selection.spec is not None else "grid"),
                "spec_tile_width": int(self._subpixel_selection.spec.tile_width if self._subpixel_selection.spec is not None else 0),
                "spec_tile_height": int(self._subpixel_selection.spec.tile_height if self._subpixel_selection.spec is not None else 0),
                "spec_overlap": int(self._subpixel_selection.spec.overlap if self._subpixel_selection.spec is not None else 0),
            }
        return payload

    def _restore_view_settings(self) -> None:
        payload = dict(self._session_view_state or {})
        layer_view = payload.get("layer_view")
        if layer_view in {"binary", "source"}:
            index = self.layer_view_combo.findData(layer_view)
            if index >= 0:
                self.layer_view_combo.setCurrentIndex(index)
        self._restored_result_kind = str(payload.get("result_kind") or "") or None
        self._sticky_result_kind = self._restored_result_kind
        self.grayscale_diff_checkbox.setChecked(bool(payload.get("grayscale_diff", False)))
        self.original_visible.setChecked(bool(payload.get("original_visible", self.original_visible.isChecked())))
        self.first_source_visible.setChecked(bool(payload.get("first_visible", self.first_source_visible.isChecked())))
        self.second_source_visible.setChecked(bool(payload.get("second_visible", self.second_source_visible.isChecked())))
        self.result_visible.setChecked(bool(payload.get("result_visible", self.result_visible.isChecked())))
        self.original_opacity.setValue(int(payload.get("original_opacity", self.original_opacity.value())))
        self.first_source_opacity.setValue(int(payload.get("first_opacity", self.first_source_opacity.value())))
        self.second_source_opacity.setValue(int(payload.get("second_opacity", self.second_source_opacity.value())))
        self.result_opacity.setValue(int(payload.get("result_opacity", self.result_opacity.value())))
        self._update_color_button_styles()

    def _store_view_settings(self, *_args) -> None:
        payload = self._build_view_settings_payload()
        self._session_view_state.clear()
        self._session_view_state.update(payload)
        self._sticky_result_kind = str(payload.get("result_kind") or "") or self._sticky_result_kind
        callback = self._on_view_state_changed
        if callable(callback):
            callback(dict(self._session_view_state))

    def _activate_context_hold(self) -> None:
        if self._result_kind_requires_confidence():
            model_id = self._confidence_model_id()
            source_key = f"model:{model_id}" if model_id else None
            pixmap = self._source_pixmap(source_key, self._named_color("first_mask"), prefer_grayscale=True) if source_key else QPixmap()
            if not pixmap.isNull():
                self._legacy_base_hold_active = True
                self._hold_preview_mode = "confidence_source"
                self._hold_preview_pixmap = pixmap
                self._update_layer_states()
                return
        self._hold_preview_mode = "base"
        self._activate_base_hold()


# Backward-compatible alias for legacy lite imports.
LiteFrameDetailsDialog = ExtendFrameDetailsDialog
