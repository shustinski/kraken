"""Render a lightweight mismatch-only frame details dialog."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.domain import BuildResult, ComparisonMode, FrameRecord
from ..core.repository import compute_comparison, load_frame_layers
from .i18n import Translator


class _ColorButton(QPushButton):
    """Pick one overlay color used in the lightweight details dialog."""

    colorChanged = pyqtSignal(object)

    def __init__(self, color: QColor, parent=None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(28, 28)
        self.setText("")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(Translator().tr("details.color_button_tooltip"))
        self.clicked.connect(self._choose_color)
        self._refresh_style()

    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: QColor) -> None:
        self._color = QColor(color)
        self._refresh_style()
        self.colorChanged.emit(self.color())

    def _choose_color(self) -> None:
        dialog = QColorDialog(self.window() if isinstance(self.window(), QWidget) else None)
        dialog.setWindowTitle(Translator().tr("details.color_dialog_title"))
        dialog.setCurrentColor(self._color)
        dialog.setStyleSheet("")
        if dialog.exec() == QColorDialog.DialogCode.Accepted:
            color = dialog.currentColor()
            if color.isValid():
                self.set_color(color)

    def _refresh_style(self) -> None:
        color = self._color
        self.setStyleSheet(
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
            "border: 1px solid #c4c8ce; border-radius: 2px; padding: 0px;"
        )


class _OverlayGraphicsView(QGraphicsView):
    """Provide zoom, pan and middle-button base preview for the overlay preview."""

    middleHoldStarted = pyqtSignal()
    middleHoldEnded = pyqtSignal()

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
        self._pan_active = False
        self._pan_start: QPointF | None = None

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        current_scale = self.transform().m11()
        next_scale = current_scale * factor
        if 0.05 <= next_scale <= 60.0:
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = True
            self._pan_start = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            self.middleHoldStarted.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_active and self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_active:
            self._pan_active = False
            self._pan_start = None
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            self.middleHoldEnded.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def reset_view(self) -> None:
        self.resetTransform()
        scene = self.scene()
        if scene is not None and not scene.sceneRect().isNull():
            self.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class LiteFrameDetailsDialog(QDialog):
    """Show mismatch-only overlays for a single frame."""

    def __init__(self, record: FrameRecord, build_result: BuildResult, parent=None) -> None:
        super().__init__(parent)
        self._i18n = Translator()
        self._t = self._i18n.tr
        self._record = record
        self._build_result = build_result
        self._records_by_key = {item.key: item for item in build_result.records}
        self._ordered_keys = [item.key for item in build_result.records]
        self._position_to_key = {
            (item.identity.tile_y, item.identity.tile_x): item.key
            for item in build_result.records
            if item.identity is not None and item.identity.tile_x is not None and item.identity.tile_y is not None
        }
        self._layers: dict[str, object] = {}
        self._first_gray = np.zeros((1, 1), dtype=np.uint8)
        self._second_gray = np.zeros((1, 1), dtype=np.uint8)
        self._first_binary = np.zeros((1, 1), dtype=bool)
        self._second_binary = np.zeros((1, 1), dtype=bool)
        self._base_gray: np.ndarray | None = None
        self._base_hold_active = False
        self._initial_fit_pending = True
        self._base_pixmap = QPixmap()
        self._first_pixmap = QPixmap()
        self._second_pixmap = QPixmap()
        self._result_pixmap = QPixmap()

        self.resize(1360, 900)
        self.setWindowTitle(self._t("details.window_title", name=record.display_name or record.key))

        self._build_ui()
        scene = self.overlay_view.scene()
        assert scene is not None
        self.base_item = scene.addPixmap(QPixmap())
        self.first_item = scene.addPixmap(QPixmap())
        self.second_item = scene.addPixmap(QPixmap())
        self.result_item = scene.addPixmap(QPixmap())
        self.base_item.setZValue(0.0)
        self.first_item.setZValue(1.0)
        self.second_item.setZValue(2.0)
        self.result_item.setZValue(3.0)

        self._connect_signals()
        self._setup_navigation_shortcuts()
        self._load_record(record, reset_view=False)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._initial_fit_pending:
            self._initial_fit_pending = False
            QTimer.singleShot(0, self.overlay_view.reset_view)

    def _setup_navigation_shortcuts(self) -> None:
        for key, direction in (
            (Qt.Key.Key_Left, "left"),
            (Qt.Key.Key_Right, "right"),
            (Qt.Key.Key_Up, "up"),
            (Qt.Key.Key_Down, "down"),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda d=direction: self._navigate(d))

    def _load_record(self, record: FrameRecord, *, reset_view: bool) -> None:
        self._record = record
        self._layers = load_frame_layers(record)
        self._first_gray = np.asarray(self._layers["first_gray"], dtype=np.uint8)
        self._second_gray = np.asarray(self._layers["second_gray"], dtype=np.uint8)
        self._first_binary = np.asarray(self._layers["first_binary"], dtype=bool)
        self._second_binary = np.asarray(self._layers["second_binary"], dtype=bool)
        base_gray = self._layers.get("base_gray")
        self._base_gray = np.asarray(base_gray, dtype=np.uint8) if base_gray is not None else None
        self.setWindowTitle(self._t("details.window_title", name=record.display_name or record.key))
        self._refresh_info()
        self._sync_base_controls()
        self._refresh_layer_pixmaps()
        self._refresh_result_layer()
        if reset_view:
            QTimer.singleShot(0, self.overlay_view.reset_view)

    def _sync_base_controls(self) -> None:
        has_base = self._base_gray is not None
        self.base_visible.setEnabled(has_base)
        self.base_visible.setChecked(has_base)
        self.base_opacity.setEnabled(has_base)
        self.base_color.setEnabled(False)
        if not has_base:
            self._base_hold_active = False

    def _navigate(self, direction: str) -> None:
        target = self._neighbor_record(direction)
        if target is None or target.key == self._record.key:
            return
        self._load_record(target, reset_view=True)

    def _neighbor_record(self, direction: str) -> FrameRecord | None:
        identity = self._record.identity
        if identity is not None and identity.tile_x is not None and identity.tile_y is not None:
            row = int(identity.tile_y)
            column = int(identity.tile_x)
            offsets = {
                "left": (0, -1),
                "right": (0, 1),
                "up": (-1, 0),
                "down": (1, 0),
            }
            delta = offsets.get(str(direction).lower())
            if delta is not None:
                candidate = self._position_to_key.get((row + delta[0], column + delta[1]))
                if candidate is not None:
                    return self._records_by_key.get(candidate)

        try:
            current_index = self._ordered_keys.index(self._record.key)
        except ValueError:
            return None
        if str(direction).lower() in {"left", "up"}:
            next_index = current_index - 1
        else:
            next_index = current_index + 1
        if next_index < 0 or next_index >= len(self._ordered_keys):
            return None
        return self._records_by_key.get(self._ordered_keys[next_index])

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(splitter, stretch=1)

        viewer_widget = QWidget(splitter)
        viewer_host = QVBoxLayout(viewer_widget)
        viewer_host.setContentsMargins(0, 0, 0, 0)
        self.overlay_view = _OverlayGraphicsView(viewer_widget)
        viewer_host.addWidget(self.overlay_view, stretch=1)
        viewer_buttons = QHBoxLayout()
        self.btn_reset_view = QPushButton(self._t("details.reset_view"), viewer_widget)
        viewer_buttons.addWidget(self.btn_reset_view)
        viewer_buttons.addStretch(1)
        viewer_host.addLayout(viewer_buttons)

        controls_scroll = QScrollArea(splitter)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(430)
        controls_host = QWidget(controls_scroll)
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(6, 6, 6, 6)
        controls_scroll.setWidget(controls_host)
        splitter.addWidget(viewer_widget)
        splitter.addWidget(controls_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([920, 440])

        operations_group = QGroupBox(self._t("details.operations"), controls_host)
        operations_form = QFormLayout(operations_group)
        self.layer_view_combo = QComboBox(operations_group)
        self.layer_view_combo.addItem(self._t("details.layer_view.binary"), "binary")
        self.layer_view_combo.addItem(self._t("details.layer_view.source"), "source")
        self.operation_combo = QComboBox(operations_group)
        for mode in (
            ComparisonMode.OVERLAY_ONLY,
            ComparisonMode.FIRST_MINUS_SECOND,
            ComparisonMode.SECOND_MINUS_FIRST,
            ComparisonMode.DISAGREEMENT,
            ComparisonMode.GRAYSCALE_DIFF,
        ):
            self.operation_combo.addItem(mode.label, mode)
        current_mode = self._build_result.options.comparison_mode
        if current_mode == ComparisonMode.XOR:
            current_mode = ComparisonMode.DISAGREEMENT
        self.operation_combo.setCurrentIndex(max(0, self.operation_combo.findData(current_mode)))
        operations_form.addRow(self._t("details.layer_view"), self.layer_view_combo)
        operations_form.addRow(self._t("details.operation"), self.operation_combo)
        controls_layout.addWidget(operations_group)

        frame_group = QGroupBox(self._t("details.frame_info"), controls_host)
        frame_form = QFormLayout(frame_group)
        self.frame_id_value = QLabel("-", frame_group)
        self.base_id_value = QLabel("-", frame_group)
        self.tile_x_value = QLabel("-", frame_group)
        self.tile_y_value = QLabel("-", frame_group)
        self.original_layer_value = QLabel("-", frame_group)
        frame_form.addRow(self._t("details.frame_id"), self.frame_id_value)
        frame_form.addRow(self._t("details.base_id"), self.base_id_value)
        frame_form.addRow(self._t("details.tile_x"), self.tile_x_value)
        frame_form.addRow(self._t("details.tile_y"), self.tile_y_value)
        frame_form.addRow(self._t("details.original_layer"), self.original_layer_value)
        controls_layout.addWidget(frame_group)

        mismatch_group = QGroupBox(self._t("details.mismatch_info"), controls_host)
        mismatch_form = QFormLayout(mismatch_group)
        self.absolute_mismatch_value = QLabel("-", mismatch_group)
        self.relative_mismatch_value = QLabel("-", mismatch_group)
        mismatch_form.addRow(self._t("details.absolute_mismatch"), self.absolute_mismatch_value)
        mismatch_form.addRow(self._t("details.relative_mismatch"), self.relative_mismatch_value)
        controls_layout.addWidget(mismatch_group)

        layers_group = QGroupBox(self._t("details.layers"), controls_host)
        layers_grid = QGridLayout(layers_group)
        layers_grid.addWidget(QLabel(""), 0, 0)
        layers_grid.addWidget(QLabel(self._t("details.visible")), 0, 1)
        layers_grid.addWidget(QLabel(self._t("details.opacity")), 0, 2)
        layers_grid.addWidget(QLabel(self._t("details.color")), 0, 3)

        self.base_visible = QCheckBox(layers_group)
        self.base_visible.setChecked(self._base_gray is not None)
        self.base_visible.setEnabled(self._base_gray is not None)
        self.base_opacity = self._build_opacity_slider(100)
        self.base_opacity.setEnabled(self._base_gray is not None)
        self.base_color = _ColorButton(QColor(255, 255, 255), layers_group)
        self.base_color.setEnabled(False)
        self._add_layer_row(layers_grid, 1, self._t("details.base"), self.base_visible, self.base_opacity, self.base_color)

        self.first_visible = QCheckBox(layers_group)
        self.first_visible.setChecked(True)
        self.first_opacity = self._build_opacity_slider(45)
        self.first_color = _ColorButton(QColor(0, 220, 120), layers_group)
        self._add_layer_row(layers_grid, 2, self._t("details.first"), self.first_visible, self.first_opacity, self.first_color)

        self.second_visible = QCheckBox(layers_group)
        self.second_visible.setChecked(True)
        self.second_opacity = self._build_opacity_slider(45)
        self.second_color = _ColorButton(QColor(255, 0, 140), layers_group)
        self._add_layer_row(layers_grid, 3, self._t("details.second"), self.second_visible, self.second_opacity, self.second_color)

        self.result_visible = QCheckBox(layers_group)
        self.result_visible.setChecked(True)
        self.result_opacity = self._build_opacity_slider(65)
        self.result_color = _ColorButton(QColor(255, 196, 0), layers_group)
        self._add_layer_row(layers_grid, 4, self._t("details.operation_result"), self.result_visible, self.result_opacity, self.result_color)
        controls_layout.addWidget(layers_group)

        controls_layout.addStretch(1)

    def _connect_signals(self) -> None:
        self.btn_reset_view.clicked.connect(self.overlay_view.reset_view)
        self.overlay_view.middleHoldStarted.connect(self._activate_base_hold)
        self.overlay_view.middleHoldEnded.connect(self._deactivate_base_hold)

        self.layer_view_combo.currentIndexChanged.connect(self._refresh_layer_pixmaps)
        self.operation_combo.currentIndexChanged.connect(self._refresh_result_layer)

        self.base_visible.toggled.connect(self._update_layer_states)
        self.base_opacity.valueChanged.connect(self._update_layer_states)

        self.first_visible.toggled.connect(self._update_layer_states)
        self.first_opacity.valueChanged.connect(self._update_layer_states)
        self.first_color.colorChanged.connect(self._refresh_first_layer)

        self.second_visible.toggled.connect(self._update_layer_states)
        self.second_opacity.valueChanged.connect(self._update_layer_states)
        self.second_color.colorChanged.connect(self._refresh_second_layer)

        self.result_visible.toggled.connect(self._update_layer_states)
        self.result_opacity.valueChanged.connect(self._update_layer_states)
        self.result_color.colorChanged.connect(self._refresh_result_layer)

    def _build_opacity_slider(self, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setRange(0, 100)
        slider.setValue(int(value))
        return slider

    def _add_layer_row(self, grid: QGridLayout, row: int, title: str, visible: QWidget, opacity: QWidget, color: QWidget) -> None:
        grid.addWidget(QLabel(title, grid.parentWidget()), row, 0)
        grid.addWidget(visible, row, 1)
        grid.addWidget(opacity, row, 2)
        grid.addWidget(color, row, 3)

    def _refresh_info(self) -> None:
        identity = self._record.identity
        self.frame_id_value.setText("-" if identity is None or identity.frame_id is None else str(identity.frame_id))
        self.base_id_value.setText("-" if identity is None or identity.base_id is None else str(identity.base_id))
        self.tile_x_value.setText("-" if identity is None or identity.tile_x is None else str(identity.tile_x))
        self.tile_y_value.setText("-" if identity is None or identity.tile_y is None else str(identity.tile_y))
        self.original_layer_value.setText(self._record.base_path or self._t("details.original_layer.not_set"))
        if self._record.absolute_score is None:
            self.absolute_mismatch_value.setText(self._t("matrix.not_computed"))
        else:
            self.absolute_mismatch_value.setText(f"{self._record.absolute_score:.4f}")
        if self._record.relative_score is None:
            self.relative_mismatch_value.setText(self._t("matrix.not_computed"))
        else:
            self.relative_mismatch_value.setText(f"{self._record.relative_score:.4f}")

    def _refresh_layer_pixmaps(self) -> None:
        height, width = self._first_gray.shape
        scene = self.overlay_view.scene()
        assert scene is not None
        scene.setSceneRect(0.0, 0.0, float(width), float(height))

        layer_view = str(self.layer_view_combo.currentData() or "binary")
        use_source = layer_view == "source"
        self.first_color.setEnabled(not use_source)
        self.second_color.setEnabled(not use_source)
        self._base_pixmap = self._grayscale_to_pixmap(self._base_gray) if self._base_gray is not None else QPixmap()
        self.base_item.setPixmap(self._base_pixmap)
        if use_source:
            self._first_pixmap = self._grayscale_to_pixmap(self._first_gray)
            self._second_pixmap = self._grayscale_to_pixmap(self._second_gray)
            self.first_item.setPixmap(self._first_pixmap)
            self.second_item.setPixmap(self._second_pixmap)
        else:
            self._first_pixmap = self._binary_mask_to_pixmap(self._first_binary, self.first_color.color())
            self._second_pixmap = self._binary_mask_to_pixmap(self._second_binary, self.second_color.color())
            self.first_item.setPixmap(self._first_pixmap)
            self.second_item.setPixmap(self._second_pixmap)
        self._refresh_result_layer()
        self._update_layer_states()
    def _refresh_first_layer(self) -> None:
        if str(self.layer_view_combo.currentData() or "binary") != "binary":
            self._update_layer_states()
            return
        self._first_pixmap = self._binary_mask_to_pixmap(self._first_binary, self.first_color.color())
        self.first_item.setPixmap(self._first_pixmap)
        self._update_layer_states()
    def _refresh_second_layer(self) -> None:
        if str(self.layer_view_combo.currentData() or "binary") != "binary":
            self._update_layer_states()
            return
        self._second_pixmap = self._binary_mask_to_pixmap(self._second_binary, self.second_color.color())
        self.second_item.setPixmap(self._second_pixmap)
        self._update_layer_states()
    def _refresh_result_layer(self) -> None:
        mode = self.operation_combo.currentData() or ComparisonMode.DISAGREEMENT
        if mode == ComparisonMode.GRAYSCALE_DIFF:
            heatmap = self._result_heatmap()
            grayscale_heatmap = np.clip(np.rint(heatmap * 255.0), 0, 255).astype(np.uint8)
            self._result_pixmap = self._grayscale_to_pixmap(grayscale_heatmap)
            self.result_color.setEnabled(False)
        else:
            self._result_pixmap = self._binary_mask_to_pixmap(self._result_heatmap() > 0.0, self.result_color.color())
            self.result_color.setEnabled(True)
        self.result_item.setPixmap(self._result_pixmap)
        self._update_layer_states()
    def _activate_base_hold(self) -> None:
        if self._base_pixmap.isNull():
            return
        self._base_hold_active = True
        self._update_layer_states()

    def _deactivate_base_hold(self) -> None:
        if not self._base_hold_active:
            return
        self._base_hold_active = False
        self._update_layer_states()

    def _update_layer_states(self) -> None:
        has_base = not self._base_pixmap.isNull()
        if self._base_hold_active and has_base:
            self.base_item.setVisible(True)
            self.base_item.setOpacity(1.0)
            self.first_item.setVisible(False)
            self.second_item.setVisible(False)
            self.result_item.setVisible(False)
            return

        self.base_item.setVisible(has_base and self.base_visible.isChecked())
        self.base_item.setOpacity(self.base_opacity.value() / 100.0)

        self.first_item.setVisible(self.first_visible.isChecked())
        self.first_item.setOpacity(self.first_opacity.value() / 100.0)

        self.second_item.setVisible(self.second_visible.isChecked())
        self.second_item.setOpacity(self.second_opacity.value() / 100.0)

        self.result_item.setVisible(self.result_visible.isChecked())
        self.result_item.setOpacity(self.result_opacity.value() / 100.0)

    def _result_heatmap(self) -> np.ndarray:
        mode = self.operation_combo.currentData() or ComparisonMode.DISAGREEMENT
        if mode == ComparisonMode.GRAYSCALE_DIFF:
            first_source = self._first_gray
            second_source = self._second_gray
        else:
            first_source = self._first_binary
            second_source = self._second_binary
        heatmap, _score = compute_comparison(first_source, second_source, mode)
        return np.asarray(heatmap, dtype=np.float32)

    def _grayscale_to_pixmap(self, image_gray: np.ndarray | None) -> QPixmap:
        if image_gray is None:
            return QPixmap()
        contiguous = np.ascontiguousarray(np.asarray(image_gray, dtype=np.uint8))
        height, width = contiguous.shape
        image = QImage(contiguous.data, width, height, int(contiguous.strides[0]), QImage.Format.Format_Grayscale8).copy()
        return QPixmap.fromImage(image.convertToFormat(QImage.Format.Format_RGB888))

    def _binary_mask_to_pixmap(self, mask: np.ndarray, color: QColor) -> QPixmap:
        mask_uint8 = np.ascontiguousarray(mask.astype(np.uint8))
        height, width = mask_uint8.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = mask_uint8 * 255
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(image)
    def _intensity_map_to_pixmap(self, intensity: np.ndarray, color: QColor) -> QPixmap:
        alpha = np.clip(np.asarray(intensity, dtype=np.float32), 0.0, 1.0)
        height, width = alpha.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(image)



