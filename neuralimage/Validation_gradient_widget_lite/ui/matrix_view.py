"""Implement mismatch-only matrix layout, rendering and overview widgets for the lite tool."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ..core.backend_constants import FRAME_NUMBER_PATTERN
from ..core.domain import FrameRecord
from .i18n import Translator
from ..core.repository import natural_sort_key
from .ui_constants import (
    CARD_CONTENT_SPACING,
    DEFAULT_BORDER,
    DEFAULT_CELL_SIZE,
    DEFAULT_ERROR_WINDOW,
    DEFAULT_GRADIENT_NAME,
    GRADIENT_LABELS,
    GRADIENT_PRESETS,
    GRADIENT_PREVIEW_MIN_HEIGHT,
    GRADIENT_RANGE_SELECTOR_MIN_HEIGHT,
    HOVER_BORDER,
    MATRIX_BACKGROUND,
    MATRIX_BACKGROUND_ALT,
    MATRIX_CELL_GAP,
    MATRIX_DEFAULT_PEN_WIDTH,
    MATRIX_HOVER_PEN_WIDTH,
    MATRIX_MAX_SCALE,
    MATRIX_MIN_CELL_SIZE,
    MATRIX_MIN_SCALE,
    MATRIX_PROCESSING_PEN_WIDTH,
    MATRIX_REFERENCE_PEN_WIDTH,
    MATRIX_SCENE_PADDING,
    MATRIX_SELECTED_BLEND_RATIO,
    MINIMAP_FRAME_MARGIN,
    MINIMAP_MIN_SIZE,
    MINIMAP_PROCESSING_TRIANGLE_HALF_WIDTH,
    MINIMAP_PROCESSING_TRIANGLE_HEIGHT,
    MINIMAP_REFERENCE_MARKER_SIDE,
    MINIMAP_REFERENCE_PEN_WIDTH,
    MINIMAP_SELECTED_COLOR,
    MINIMAP_SELECTED_OUTLINE_WIDTH,
    MINIMAP_SELECTED_RADIUS_OFF,
    MINIMAP_SELECTED_RADIUS_ON,
    NORMALIZATION_EPSILON,
    PANEL_BACKGROUND,
    PANEL_TEXT,
    PROCESSING_BORDER,
    PROCESSING_FILL,
    REFERENCE_BORDER,
    SELECTION_BLINK_INTERVAL_MS,
    SELECTED_BLINK_COLOR,
    SUBDUED_TEXT_COLOR,
    VISIBLE_RECT_MIN_SIZE,
)


@dataclass(frozen=True, slots=True)
class MatrixLayoutConfig:
    """Describe the supported matrix layouts used by the lite mismatch matrix."""

    mode: str = "indexed_grid"
    total_frames: int = 0
    frames_per_row: int = 0
    rows: int = 1
    columns: int = 1


class _MatrixCellItem(QGraphicsRectItem):
    """Represent one visible matrix cell bound to one frame record."""

    def __init__(self, rect: QRectF, record: FrameRecord, row: int, column: int, index: int) -> None:
        super().__init__(rect)
        self.record = record
        self.row = int(row)
        self.column = int(column)
        self.index = int(index)


def extract_frame_number(value: str) -> int:
    """Extract the zero-based frame number from the last underscore-separated filename segment."""
    stem = Path(str(value)).stem
    last_segment = stem.rsplit("_", 1)[-1]
    if not FRAME_NUMBER_PATTERN.fullmatch(last_segment):
        raise ValueError(f"Unable to extract frame number from '{value}'")
    return int(last_segment)


def blend_colors(base_color: QColor, overlay_color: QColor, alpha: float) -> QColor:
    """Blend two colors with the provided overlay alpha in the range 0..1."""
    ratio = max(0.0, min(float(alpha), 1.0))
    inv = 1.0 - ratio
    return QColor(
        int(base_color.red() * inv + overlay_color.red() * ratio),
        int(base_color.green() * inv + overlay_color.green() * ratio),
        int(base_color.blue() * inv + overlay_color.blue() * ratio),
    )


def interpolate_gradient_color(gradient_name: str, score: float) -> QColor:
    """Interpolate a QColor inside one named gradient preset."""
    preset = GRADIENT_PRESETS.get(gradient_name) or GRADIENT_PRESETS[DEFAULT_GRADIENT_NAME]
    value = max(0.0, min(float(score), 1.0))
    for index in range(1, len(preset)):
        left_pos, left_rgb = preset[index - 1]
        right_pos, right_rgb = preset[index]
        if value <= right_pos:
            span = max(NORMALIZATION_EPSILON, right_pos - left_pos)
            ratio = (value - left_pos) / span
            return QColor(
                int(left_rgb[0] + (right_rgb[0] - left_rgb[0]) * ratio),
                int(left_rgb[1] + (right_rgb[1] - left_rgb[1]) * ratio),
                int(left_rgb[2] + (right_rgb[2] - left_rgb[2]) * ratio),
            )
    return QColor(*preset[-1][1])


def error_palette_color(position: float, gradient_name: str = DEFAULT_GRADIENT_NAME) -> QColor:
    """Return one color from the named error gradient preset."""
    return interpolate_gradient_color(gradient_name, position)


def map_score_to_palette_position(score: float, low_bound: float, high_bound: float) -> float:
    """Map a score to the displayed gradient range, including inverted windows."""
    value = max(0.0, min(float(score), 1.0))
    low = max(0.0, min(float(low_bound), 1.0))
    high = max(0.0, min(float(high_bound), 1.0))
    if abs(high - low) < NORMALIZATION_EPSILON:
        return 1.0 if value >= high else 0.0
    if low < high:
        return max(0.0, min((value - low) / (high - low), 1.0))
    return max(0.0, min((low - value) / (low - high), 1.0))


def build_matrix_layout(records: list[FrameRecord], layout_config: MatrixLayoutConfig) -> tuple[list[tuple[FrameRecord, int, int]], int, int]:
    """Place frame records into one indexed or custom matrix layout."""
    if not records:
        return [], 0, 0
    mode = str(getattr(layout_config, "mode", "indexed_grid") or "indexed_grid")
    fixed_positions = all(
        record.identity is not None and record.identity.tile_x is not None and record.identity.tile_y is not None
        for record in records
    )
    if mode == "manual_grid":
        rows = int(layout_config.rows)
        columns = int(layout_config.columns)
        if rows <= 0 or columns <= 0:
            raise ValueError("Invalid custom matrix layout")
        capacity = rows * columns
        if len(records) > capacity:
            raise ValueError(f"Custom matrix layout capacity {capacity} is smaller than the frame count {len(records)}")
        if fixed_positions:
            placements: list[tuple[FrameRecord, int, int]] = []
            for record in records:
                row = int(record.identity.tile_y)
                column = int(record.identity.tile_x)
                if row < 0 or row >= rows or column < 0 or column >= columns:
                    raise ValueError("Stored matrix coordinates are outside the configured custom layout")
                placements.append((record, row, column))
            return sorted(placements, key=lambda item: (item[1], item[2], natural_sort_key(item[0].key))), columns, rows
        placements: list[tuple[FrameRecord, int, int]] = []
        for index, record in enumerate(records):
            row = index // columns
            column = index % columns
            placements.append((record, row, column))
        return placements, columns, rows

    columns = int(layout_config.frames_per_row)
    total_frames = int(layout_config.total_frames)
    if columns <= 0 or total_frames <= 0:
        raise ValueError("Invalid indexed matrix layout")
    rows = max(1, math.ceil(total_frames / columns))
    if fixed_positions:
        placements = []
        for record in records:
            row = int(record.identity.tile_y)
            column = int(record.identity.tile_x)
            if row < 0 or row >= rows or column < 0 or column >= columns:
                raise ValueError("Stored matrix coordinates are outside the configured indexed layout")
            placements.append((record, row, column))
        return sorted(placements, key=lambda item: (item[1], item[2], natural_sort_key(item[0].key))), columns, rows
    placements: list[tuple[FrameRecord, int, int]] = []
    for record in sorted(records, key=lambda item: natural_sort_key(item.key)):
        frame_index = extract_frame_number(record.display_name or record.key)
        if frame_index < 0 or frame_index >= total_frames:
            raise ValueError(f"Frame index {frame_index} is outside the configured matrix range 0..{total_frames - 1}")
        row = frame_index // columns
        column = frame_index % columns
        placements.append((record, row, column))
    return placements, columns, rows

class _GradientPreviewBar(QWidget):
    """Render a compact horizontal preview for one gradient preset."""

    def __init__(self, parent=None) -> None:
        """Initialize the preview bar."""
        super().__init__(parent)
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self.setMinimumHeight(GRADIENT_PREVIEW_MIN_HEIGHT)

    def set_gradient_name(self, gradient_name: str) -> None:
        """Set the gradient shown in the preview bar."""
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the current gradient preview."""
        super().paintEvent(event)
        painter = QPainter(self)
        target = self.rect().adjusted(0, 2, 0, -2)
        if target.width() <= 0 or target.height() <= 0:
            painter.end()
            return
        for x in range(target.width()):
            position = x / max(1, target.width() - 1)
            painter.setPen(interpolate_gradient_color(self._gradient_name, position))
            painter.drawLine(target.left() + x, target.top(), target.left() + x, target.bottom())
        painter.setPen(QPen(SUBDUED_TEXT_COLOR, 1.0))
        painter.drawRect(target.adjusted(0, 0, -1, -1))
        painter.end()


class _GradientWindowBar(QWidget):
    """Render and edit the active error window on top of the current gradient."""

    rangeEdited = pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        """Initialize the interactive error-window bar."""
        super().__init__(parent)
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self._low_bound, self._high_bound = DEFAULT_ERROR_WINDOW
        self._active_handle: str | None = None
        self.setMinimumHeight(max(26, GRADIENT_RANGE_SELECTOR_MIN_HEIGHT // 2))
        self.setMouseTracking(True)

    def set_gradient_name(self, gradient_name: str) -> None:
        """Set the gradient used by the range bar."""
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME
        self.update()

    def set_error_window(self, low_bound: float, high_bound: float) -> None:
        """Update the current low and high bounds without emitting signals."""
        self._low_bound = max(0.0, min(float(low_bound), 1.0))
        self._high_bound = max(0.0, min(float(high_bound), 1.0))
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the gradient and the current selected error window."""
        super().paintEvent(event)
        painter = QPainter(self)
        bar_rect = self.rect().adjusted(0, 6, 0, -8)
        if bar_rect.width() <= 0 or bar_rect.height() <= 0:
            painter.end()
            return
        for x in range(bar_rect.width()):
            position = x / max(1, bar_rect.width() - 1)
            painter.setPen(interpolate_gradient_color(self._gradient_name, position))
            painter.drawLine(bar_rect.left() + x, bar_rect.top(), bar_rect.left() + x, bar_rect.bottom())

        low_x = self._position_to_x(self._low_bound, bar_rect)
        high_x = self._position_to_x(self._high_bound, bar_rect)
        excluded_color = QColor(0, 0, 0, 110)
        if low_x < high_x:
            painter.fillRect(QRectF(bar_rect.left(), bar_rect.top(), max(0.0, low_x - bar_rect.left()), bar_rect.height()), excluded_color)
            painter.fillRect(QRectF(high_x, bar_rect.top(), max(0.0, bar_rect.right() - high_x), bar_rect.height()), excluded_color)
        elif low_x > high_x:
            painter.fillRect(QRectF(high_x, bar_rect.top(), max(0.0, low_x - high_x), bar_rect.height()), excluded_color)

        painter.setPen(QPen(PANEL_TEXT, 1.0))
        painter.drawRect(bar_rect.adjusted(0, 0, -1, -1))
        self._draw_handle(painter, low_x, bar_rect)
        self._draw_handle(painter, high_x, bar_rect)
        painter.end()

    def mousePressEvent(self, event) -> None:
        """Start dragging the nearest range handle."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        bar_rect = self.rect().adjusted(0, 6, 0, -8)
        low_x = self._position_to_x(self._low_bound, bar_rect)
        high_x = self._position_to_x(self._high_bound, bar_rect)
        self._active_handle = 'low' if abs(event.position().x() - low_x) <= abs(event.position().x() - high_x) else 'high'
        self._update_active_handle(event.position().x(), bar_rect)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        """Update the active range handle while dragging."""
        if self._active_handle is None:
            super().mouseMoveEvent(event)
            return
        bar_rect = self.rect().adjusted(0, 6, 0, -8)
        self._update_active_handle(event.position().x(), bar_rect)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        """Finish dragging the active range handle."""
        if event.button() == Qt.MouseButton.LeftButton and self._active_handle is not None:
            self._active_handle = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _draw_handle(self, painter: QPainter, x_pos: float, bar_rect: QRectF) -> None:
        """Draw one vertical handle on the gradient bar."""
        painter.setPen(QPen(PANEL_TEXT, 2.0))
        painter.drawLine(QPointF(x_pos, bar_rect.top() - 3), QPointF(x_pos, bar_rect.bottom() + 3))

    def _position_to_x(self, value: float, bar_rect: QRectF) -> float:
        """Convert a normalized value into a horizontal bar coordinate."""
        return bar_rect.left() + max(0.0, min(float(value), 1.0)) * max(1.0, bar_rect.width() - 1.0)

    def _x_to_position(self, x_pos: float, bar_rect: QRectF) -> float:
        """Convert a horizontal bar coordinate into a normalized value."""
        if bar_rect.width() <= 1.0:
            return 0.0
        return max(0.0, min((x_pos - bar_rect.left()) / (bar_rect.width() - 1.0), 1.0))

    def _update_active_handle(self, x_pos: float, bar_rect: QRectF) -> None:
        """Update the active handle and emit the new selected range."""
        position = self._x_to_position(x_pos, bar_rect)
        if self._active_handle == 'low':
            self._low_bound = position
        elif self._active_handle == 'high':
            self._high_bound = position
        self.update()
        self.rangeEdited.emit(self._low_bound, self._high_bound)


class _GradientPresetCard(QFrame):
    """Render one clickable gradient preset card."""

    clicked = pyqtSignal(str)

    def __init__(self, gradient_name: str, parent=None) -> None:
        """Initialize one gradient preset card."""
        super().__init__(parent)
        self._gradient_name = gradient_name
        self._selected = False
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(CARD_CONTENT_SPACING)
        self._label = QLabel(GRADIENT_LABELS.get(gradient_name, gradient_name.title()), self)
        self._preview = _GradientPreviewBar(self)
        self._preview.set_gradient_name(gradient_name)
        layout.addWidget(self._label)
        layout.addWidget(self._preview)
        self._refresh_style()

    def gradient_name(self) -> str:
        """Return the gradient preset name represented by this card."""
        return self._gradient_name

    def set_selected(self, selected: bool) -> None:
        """Update the selected state of the card."""
        self._selected = bool(selected)
        self._refresh_style()

    def mousePressEvent(self, event) -> None:
        """Emit the card click when the user presses the left mouse button."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._gradient_name)
            event.accept()
            return
        super().mousePressEvent(event)

    def _refresh_style(self) -> None:
        """Refresh the card border according to the selected state."""
        border_color = PANEL_TEXT if self._selected else SUBDUED_TEXT_COLOR
        background_color = '#2f2f31' if self._selected else '#262628'
        self.setStyleSheet(
            f'QFrame {{ border: 1px solid {border_color.name()}; border-radius: 4px; background: {background_color}; }}'
            'QLabel { border: none; background: transparent; color: #f0f0f0; }'
        )


class GradientPresetSelectorWidget(QWidget):
    """Select the active gradient preset used to render mismatch scores."""

    gradientChanged = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        """Initialize the gradient preset selector."""
        super().__init__(parent)
        self._selected_gradient_name = DEFAULT_GRADIENT_NAME
        self._cards: dict[str, _GradientPresetCard] = {}
        self._i18n = Translator()
        self._t = self._i18n.tr
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(CARD_CONTENT_SPACING)
        self._title_label = QLabel(self._t("matrix.gradient"), self)
        layout.addWidget(self._title_label)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(CARD_CONTENT_SPACING)
        grid.setVerticalSpacing(CARD_CONTENT_SPACING)
        for index, name in enumerate(GRADIENT_PRESETS):
            card = _GradientPresetCard(name, self)
            card.clicked.connect(self._on_card_clicked)
            self._cards[name] = card
            grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)
        self.set_selected_gradient(DEFAULT_GRADIENT_NAME, emit_signal=False)

    def selected_gradient(self) -> str:
        """Return the currently selected gradient preset name."""
        return self._selected_gradient_name

    def set_selected_gradient(self, name: str, *, emit_signal: bool = True) -> None:
        """Select one gradient preset and refresh the visible card states."""
        normalized = name if name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME
        previous = self._selected_gradient_name
        self._selected_gradient_name = normalized
        for gradient_name, card in self._cards.items():
            card.set_selected(gradient_name == normalized)
        if emit_signal and normalized != previous:
            self.gradientChanged.emit(normalized)

    def _on_card_clicked(self, gradient_name: str) -> None:
        """Handle one direct click on a gradient preset card."""
        self.set_selected_gradient(gradient_name, emit_signal=True)

    def retranslate_ui(self) -> None:
        """Update translated captions inside the gradient selector."""
        self._title_label.setText(self._t("matrix.gradient"))


class GradientRangeSelectorWidget(QWidget):
    """Edit the low and high bounds of the active error gradient window."""

    rangeChanged = pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        """Initialize the error-window selector."""
        super().__init__(parent)
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self._i18n = Translator()
        self._t = self._i18n.tr
        self._low_spin = QDoubleSpinBox(self)
        self._high_spin = QDoubleSpinBox(self)
        self._bar = _GradientWindowBar(self)
        for spin, value in ((self._low_spin, DEFAULT_ERROR_WINDOW[0]), (self._high_spin, DEFAULT_ERROR_WINDOW[1])):
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(2)
            spin.setValue(value)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(CARD_CONTENT_SPACING)
        self._title_label = QLabel(self._t("matrix.error_window"), self)
        layout.addWidget(self._title_label)
        layout.addWidget(self._bar)
        row = QHBoxLayout()
        self._low_label = QLabel(self._t("matrix.low"), self)
        row.addWidget(self._low_label)
        row.addWidget(self._low_spin)
        self._high_label = QLabel(self._t("matrix.high"), self)
        row.addWidget(self._high_label)
        row.addWidget(self._high_spin)
        layout.addLayout(row)
        self.setMinimumHeight(GRADIENT_RANGE_SELECTOR_MIN_HEIGHT)
        self._low_spin.valueChanged.connect(self._emit_range_changed)
        self._high_spin.valueChanged.connect(self._emit_range_changed)
        self._bar.rangeEdited.connect(self._on_bar_range_edited)
        self._bar.set_gradient_name(self._gradient_name)
        self._bar.set_error_window(*DEFAULT_ERROR_WINDOW)

    def error_window(self) -> tuple[float, float]:
        """Return the currently selected error window."""
        return float(self._low_spin.value()), float(self._high_spin.value())

    def set_error_window(self, low_bound: float, high_bound: float) -> None:
        """Set the error window without emitting change signals."""
        low_value = max(0.0, min(float(low_bound), 1.0))
        high_value = max(0.0, min(float(high_bound), 1.0))
        self._low_spin.blockSignals(True)
        self._high_spin.blockSignals(True)
        self._low_spin.setValue(low_value)
        self._high_spin.setValue(high_value)
        self._low_spin.blockSignals(False)
        self._high_spin.blockSignals(False)
        self._bar.set_error_window(low_value, high_value)

    def set_gradient_name(self, gradient_name: str) -> None:
        """Set the gradient used by the error-window preview bar."""
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME
        self._bar.set_gradient_name(self._gradient_name)

    def _emit_range_changed(self, _value: float) -> None:
        """Propagate range changes coming from the numeric editors."""
        low_value, high_value = self.error_window()
        self._bar.set_error_window(low_value, high_value)
        self.rangeChanged.emit(low_value, high_value)

    def _on_bar_range_edited(self, low_bound: float, high_bound: float) -> None:
        """Apply range changes coming from the draggable gradient bar."""
        self.set_error_window(low_bound, high_bound)
        self.rangeChanged.emit(*self.error_window())

    def retranslate_ui(self) -> None:
        """Update translated captions inside the range selector."""
        self._title_label.setText(self._t("matrix.error_window"))
        self._low_label.setText(self._t("matrix.low"))
        self._high_label.setText(self._t("matrix.high"))


class MatrixMiniMapWidget(QWidget):
    """Render a compact overview image for the active matrix tab."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._visible_rect = QRectF()
        self._selected_position: tuple[int, int] | None = None
        self._selected_blink_on = False
        self._processing_positions: tuple[tuple[int, int], ...] = ()
        self._reference_position: tuple[int, int] | None = None
        self.setMinimumSize(*MINIMAP_MIN_SIZE)

    def set_overview(self, image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position) -> None:
        self._image = image
        self._visible_rect = QRectF(visible_rect)
        self._selected_position = selected_position
        self._selected_blink_on = bool(selected_blink_on)
        self._processing_positions = tuple(processing_positions)
        self._reference_position = reference_position
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), PANEL_BACKGROUND)
        if self._image is None or self._image.isNull():
            painter.setPen(PANEL_TEXT)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._t("matrix.no_matrix"))
            painter.end()
            return
        target = self.rect().adjusted(MINIMAP_FRAME_MARGIN, MINIMAP_FRAME_MARGIN, -MINIMAP_FRAME_MARGIN, -MINIMAP_FRAME_MARGIN)
        painter.drawImage(target, self._image)
        width = max(1, self._image.width())
        height = max(1, self._image.height())
        cell_w = target.width() / width
        cell_h = target.height() / height
        if not self._visible_rect.isNull():
            rect = QRectF(
                target.left() + self._visible_rect.left() * target.width(),
                target.top() + self._visible_rect.top() * target.height(),
                max(VISIBLE_RECT_MIN_SIZE, self._visible_rect.width() * target.width()),
                max(VISIBLE_RECT_MIN_SIZE, self._visible_rect.height() * target.height()),
            )
            painter.setPen(QPen(PANEL_TEXT, 1.5))
            painter.drawRect(rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PROCESSING_FILL)
        for row, column in self._processing_positions:
            painter.drawEllipse(
                QPointF(target.left() + (column + 0.5) * cell_w, target.top() + (row + 0.5) * cell_h),
                MINIMAP_PROCESSING_TRIANGLE_HALF_WIDTH,
                MINIMAP_PROCESSING_TRIANGLE_HEIGHT,
            )
        if self._reference_position is not None:
            row, column = self._reference_position
            painter.setPen(QPen(REFERENCE_BORDER, MINIMAP_REFERENCE_PEN_WIDTH))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                QRectF(
                    target.left() + column * cell_w,
                    target.top() + row * cell_h,
                    max(MINIMAP_REFERENCE_MARKER_SIDE, cell_w),
                    max(MINIMAP_REFERENCE_MARKER_SIDE, cell_h),
                )
            )
        if self._selected_position is not None:
            row, column = self._selected_position
            radius = MINIMAP_SELECTED_RADIUS_ON if self._selected_blink_on else MINIMAP_SELECTED_RADIUS_OFF
            painter.setPen(QPen(MINIMAP_SELECTED_COLOR, MINIMAP_SELECTED_OUTLINE_WIDTH))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QPointF(target.left() + (column + 0.5) * cell_w, target.top() + (row + 0.5) * cell_h),
                radius,
                radius,
            )
        painter.end()


class MatrixListWidget(QGraphicsView):
    """Render the frame matrix and provide navigation, selection and overview data."""
    recordActivated = pyqtSignal(object)
    recordSelected = pyqtSignal(object)
    contextMenuRequested = pyqtSignal(object, object)
    overviewChanged = pyqtSignal(object, object, object, object, object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._i18n = Translator()
        self._t = self._i18n.tr
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._records: list[FrameRecord] = []
        self._cell_size = DEFAULT_CELL_SIZE
        self._gap = MATRIX_CELL_GAP
        self._scene_padding = MATRIX_SCENE_PADDING
        self._selected_item: _MatrixCellItem | None = None
        self._hovered_item: _MatrixCellItem | None = None
        self._processing_keys: set[str] = set()
        self._reference_key: str | None = None
        self._columns = 0
        self._rows = 0
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self._error_window_low, self._error_window_high = DEFAULT_ERROR_WINDOW
        self._overview_image: QImage | None = None
        self._selection_blink_on = False
        self._selection_blink_timer = QTimer(self)
        self._selection_blink_timer.setInterval(SELECTION_BLINK_INTERVAL_MS)
        self._selection_blink_timer.timeout.connect(self._toggle_selection_blink)
        self._selection_blink_timer.start()
        self._layout_config = MatrixLayoutConfig()
        self._item_by_key: dict[str, _MatrixCellItem] = {}
        self._record_by_position: dict[tuple[int, int], FrameRecord] = {}
        self._record_positions: dict[str, tuple[int, int]] = {}
        self._pan_active = False
        self._pan_start = None
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(MATRIX_BACKGROUND)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.horizontalScrollBar().valueChanged.connect(self._emit_overview_state)
        self.verticalScrollBar().valueChanged.connect(self._emit_overview_state)

    def set_cell_size(self, cell_size: int) -> None:
        self._cell_size = max(MATRIX_MIN_CELL_SIZE, int(cell_size))

    def set_gradient_preset(self, gradient_name: str) -> None:
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME

    def set_error_window(self, low_bound: float, high_bound: float) -> None:
        self._error_window_low = max(0.0, min(float(low_bound), 1.0))
        self._error_window_high = max(0.0, min(float(high_bound), 1.0))

    def set_layout_config(self, layout_config: MatrixLayoutConfig) -> None:
        self._layout_config = layout_config


    def set_processing_keys(self, processing_keys) -> None:
        previous = set(self._processing_keys)
        self._processing_keys = {str(key) for key in processing_keys}
        for key in previous | self._processing_keys:
            item = self._item_by_key.get(key)
            if item is not None:
                self._apply_item_style(item)
        self._emit_overview_state()

    def set_reference_key(self, reference_key: str | None) -> None:
        previous = self._reference_key
        self._reference_key = str(reference_key) if reference_key else None
        for key in {previous, self._reference_key}:
            if key and key in self._item_by_key:
                self._apply_item_style(self._item_by_key[key])
        self._emit_overview_state()

    def set_records(self, records: list[FrameRecord], *, sort_mode: str = "name", reset_view: bool = False) -> None:
        ordered = self._sort_records(list(records), sort_mode)
        if reset_view:
            self.resetTransform()
        self._rebuild_scene(ordered)

    def refresh_scene(self) -> None:
        if not self._records:
            return
        ordered_items = sorted(self._item_by_key.values(), key=lambda item: item.index)
        for item in ordered_items:
            item.setToolTip(self._tooltip_for_record(item.record))
            self._apply_item_style(item)
        placements = [(item.record, item.row, item.column) for item in ordered_items]
        self._overview_image = self._build_overview_image(placements)
        self._scene.update()
        self.viewport().update()
        self._emit_overview_state()

    def current_record(self) -> FrameRecord | None:
        return self._selected_item.record if self._selected_item is not None else None

    def select_record_by_key(self, key: str, *, ensure_visible: bool = True) -> FrameRecord | None:
        item = self._item_by_key.get(str(key))
        if item is None:
            return None
        self._select_item(item)
        self.recordSelected.emit(item.record)
        if ensure_visible:
            self.centerOn(item)
            self._emit_overview_state()
        return item.record

    def neighbor_record(self, record: FrameRecord | str, direction: str) -> FrameRecord | None:
        key = record.key if isinstance(record, FrameRecord) else str(record)
        position = self._record_positions.get(key)
        if position is None:
            return None
        row, column = position
        direction_name = str(direction).lower()
        if direction_name == "right":
            return self._next_horizontal_record(row, column, step=1)
        if direction_name == "left":
            return self._next_horizontal_record(row, column, step=-1)
        if direction_name == "down":
            return self._next_vertical_record(row, column, step=1)
        if direction_name == "up":
            return self._next_vertical_record(row, column, step=-1)
        return None

    def _next_horizontal_record(self, row: int, column: int, *, step: int) -> FrameRecord | None:
        row_columns = sorted(col for (item_row, col) in self._record_by_position if item_row == row)
        if step > 0:
            for candidate_col in row_columns:
                if candidate_col > column:
                    return self._record_by_position.get((row, candidate_col))
            for candidate_row in sorted({item_row for (item_row, _col) in self._record_by_position if item_row > row}):
                candidate_cols = sorted(col for (item_row, col) in self._record_by_position if item_row == candidate_row)
                if candidate_cols:
                    return self._record_by_position.get((candidate_row, candidate_cols[0]))
            return None
        for candidate_col in reversed(row_columns):
            if candidate_col < column:
                return self._record_by_position.get((row, candidate_col))
        for candidate_row in sorted({item_row for (item_row, _col) in self._record_by_position if item_row < row}, reverse=True):
            candidate_cols = sorted(col for (item_row, col) in self._record_by_position if item_row == candidate_row)
            if candidate_cols:
                return self._record_by_position.get((candidate_row, candidate_cols[-1]))
        return None

    def _next_vertical_record(self, row: int, column: int, *, step: int) -> FrameRecord | None:
        rows = sorted({item_row for (item_row, _col) in self._record_by_position if item_row != row})
        if step < 0:
            rows = list(reversed(rows))
        for candidate_row in rows:
            if step > 0 and candidate_row <= row:
                continue
            if step < 0 and candidate_row >= row:
                continue
            if (candidate_row, column) in self._record_by_position:
                return self._record_by_position[(candidate_row, column)]
            candidate_cols = sorted(col for (item_row, col) in self._record_by_position if item_row == candidate_row)
            if candidate_cols:
                nearest_col = min(candidate_cols, key=lambda value: abs(value - column))
                return self._record_by_position.get((candidate_row, nearest_col))
        return None
    def _sort_records(self, records: list[FrameRecord], sort_mode: str) -> list[FrameRecord]:
        mode = str(sort_mode or "name")
        if mode == "score_desc":
            return sorted(records, key=lambda item: float(item.score), reverse=True)
        if mode == "score_asc":
            return sorted(records, key=lambda item: float(item.score))
        if mode == "input_order":
            return list(records)
        return sorted(records, key=lambda item: natural_sort_key(item.display_name or item.key))

    def _handle_zoom_wheel(self, event) -> bool:
        modifiers = event.modifiers() | QApplication.keyboardModifiers()
        if not bool(modifiers & Qt.KeyboardModifier.ControlModifier):
            return False
        delta_y = event.angleDelta().y() or event.pixelDelta().y()
        if delta_y == 0:
            event.accept()
            return True
        factor = 1.15 if delta_y > 0 else (1.0 / 1.15)
        next_scale = self.transform().m11() * factor
        if MATRIX_MIN_SCALE <= next_scale <= MATRIX_MAX_SCALE:
            self.scale(factor, factor)
            self._emit_overview_state()
        event.accept()
        return True

    def wheelEvent(self, event) -> None:
        if self._handle_zoom_wheel(event):
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = True
            self._pan_start = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, _MatrixCellItem):
                self._select_item(item)
                self.recordSelected.emit(item.record)
            else:
                self._clear_selection()
                self.recordSelected.emit(None)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, _MatrixCellItem):
                self._select_item(item)
                self.recordSelected.emit(item.record)
                self.recordActivated.emit(item.record)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:
        item = self.itemAt(event.pos())
        record = None
        if isinstance(item, _MatrixCellItem):
            self._select_item(item)
            self.recordSelected.emit(item.record)
            record = item.record
        self.contextMenuRequested.emit(record, event.globalPos())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._pan_active and self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self._emit_overview_state()
            event.accept()
            return
        item = self.itemAt(event.pos())
        if isinstance(item, _MatrixCellItem):
            self._set_hover_item(item)
            QToolTip.showText(event.globalPosition().toPoint(), self._hover_text(item.record), self.viewport())
        else:
            self._set_hover_item(None)
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_active:
            self._pan_active = False
            self._pan_start = None
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_hover_item(None)
        QToolTip.hideText()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._emit_overview_state()

    def _rebuild_scene(self, records: list[FrameRecord]) -> None:
        self._records = list(records)
        selected_key = self._selected_item.record.key if self._selected_item is not None else None
        hovered_key = self._hovered_item.record.key if self._hovered_item is not None else None
        self._selected_item = None
        self._hovered_item = None
        self._item_by_key.clear()
        self._record_by_position.clear()
        self._record_positions.clear()
        self._scene.clear()
        if not self._records:
            self._columns = 0
            self._rows = 0
            self._overview_image = None
            self.overviewChanged.emit(None, QRectF(), None, False, tuple(), None)
            return
        placements, self._columns, self._rows = build_matrix_layout(self._records, self._layout_config)
        matrix_width = self._columns * (self._cell_size + self._gap)
        matrix_height = self._rows * (self._cell_size + self._gap)
        self._scene.setSceneRect(0, 0, matrix_width + self._scene_padding * 2, matrix_height + self._scene_padding * 2)
        self._scene.addRect(QRectF(self._scene_padding, self._scene_padding, matrix_width, matrix_height), QPen(SUBDUED_TEXT_COLOR, 1.0))
        for index, (record, row, column) in enumerate(placements):
            x = self._scene_padding + column * (self._cell_size + self._gap)
            y = self._scene_padding + row * (self._cell_size + self._gap)
            item = _MatrixCellItem(QRectF(x, y, self._cell_size, self._cell_size), record, row, column, index)
            item.setToolTip(self._tooltip_for_record(record))
            self._scene.addItem(item)
            self._item_by_key[record.key] = item
            self._record_positions[record.key] = (row, column)
            self._record_by_position[(row, column)] = record
            self._apply_item_style(item)
            if selected_key == record.key:
                self._selected_item = item
            if hovered_key == record.key:
                self._hovered_item = item
        if self._selected_item is not None:
            self._apply_item_style(self._selected_item)
        if self._hovered_item is not None:
            self._apply_item_style(self._hovered_item)
        self._overview_image = self._build_overview_image(placements)
        self._emit_overview_state()

    def _build_overview_image(self, placements: list[tuple[FrameRecord, int, int]]) -> QImage:
        image = QImage(max(1, self._columns), max(1, self._rows), QImage.Format.Format_RGB888)
        image.fill(MATRIX_BACKGROUND_ALT)
        for record, row, column in placements:
            image.setPixelColor(column, row, self._background_color_for_record(record))
        return image

    def _emit_overview_state(self) -> None:
        if self._overview_image is None or self._columns <= 0 or self._rows <= 0:
            self.overviewChanged.emit(None, QRectF(), None, False, tuple(), None)
            return
        scene_rect = self._scene.sceneRect()
        visible_scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        selected_position = None if self._selected_item is None else (self._selected_item.row, self._selected_item.column)
        processing_positions = tuple(self._record_positions[key] for key in self._processing_keys if key in self._record_positions)
        reference_position = self._record_positions.get(self._reference_key) if self._reference_key in self._record_positions else None
        if scene_rect.width() <= 0 or scene_rect.height() <= 0:
            self.overviewChanged.emit(self._overview_image, QRectF(), selected_position, self._selection_blink_on, processing_positions, reference_position)
            return
        left_padding = self._scene_padding / max(1.0, scene_rect.width())
        top_padding = self._scene_padding / max(1.0, scene_rect.height())
        width_padding = (self._scene_padding * 2) / max(1.0, scene_rect.width())
        height_padding = (self._scene_padding * 2) / max(1.0, scene_rect.height())
        normalized = QRectF(
            max(0.0, (visible_scene_rect.left() / scene_rect.width() - left_padding) / max(NORMALIZATION_EPSILON, 1.0 - width_padding)),
            max(0.0, (visible_scene_rect.top() / scene_rect.height() - top_padding) / max(NORMALIZATION_EPSILON, 1.0 - height_padding)),
            min(1.0, (visible_scene_rect.width() / scene_rect.width()) / max(NORMALIZATION_EPSILON, 1.0 - width_padding)),
            min(1.0, (visible_scene_rect.height() / scene_rect.height()) / max(NORMALIZATION_EPSILON, 1.0 - height_padding)),
        )
        self.overviewChanged.emit(self._overview_image, normalized, selected_position, self._selection_blink_on, processing_positions, reference_position)

    def _select_item(self, item: _MatrixCellItem) -> None:
        if self._selected_item is item:
            return
        previous = self._selected_item
        self._selected_item = item
        self._selection_blink_on = True
        if previous is not None:
            self._apply_item_style(previous)
        self._apply_item_style(item)
        self._emit_overview_state()

    def _clear_selection(self) -> None:
        if self._selected_item is not None:
            previous = self._selected_item
            self._selected_item = None
            self._selection_blink_on = False
            self._apply_item_style(previous)
        self._emit_overview_state()

    def _set_hover_item(self, item: _MatrixCellItem | None) -> None:
        if self._hovered_item is item:
            return
        previous = self._hovered_item
        self._hovered_item = item
        if previous is not None:
            self._apply_item_style(previous)
        if item is not None:
            self._apply_item_style(item)

    def _apply_item_style(self, item: _MatrixCellItem) -> None:
        if item is self._hovered_item:
            pen = QPen(HOVER_BORDER, MATRIX_HOVER_PEN_WIDTH)
        elif item.record.key in self._processing_keys:
            pen = QPen(PROCESSING_BORDER, MATRIX_PROCESSING_PEN_WIDTH)
        elif self._reference_key is not None and item.record.key == self._reference_key:
            pen = QPen(REFERENCE_BORDER, MATRIX_REFERENCE_PEN_WIDTH)
        else:
            pen = QPen(DEFAULT_BORDER, MATRIX_DEFAULT_PEN_WIDTH)
        brush_color = self._background_color_for_record(item.record)
        if item is self._selected_item and self._selection_blink_on:
            brush_color = blend_colors(brush_color, SELECTED_BLINK_COLOR, MATRIX_SELECTED_BLEND_RATIO)
        item.setPen(pen)
        item.setBrush(brush_color)

    def _toggle_selection_blink(self) -> None:
        if self._selected_item is None:
            if self._selection_blink_on:
                self._selection_blink_on = False
                self._emit_overview_state()
            return
        self._selection_blink_on = not self._selection_blink_on
        self._apply_item_style(self._selected_item)
        self._emit_overview_state()

    def _display_score(self, record: FrameRecord) -> float | None:
        if not bool(getattr(record, "score_ready", False)):
            return None
        return float(record.score)

    @staticmethod
    def _format_metric_value(value: float) -> str:
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            return f"{numeric * 100.0:.3f}%"
        return f"{numeric:.3f}"

    def _background_color(self, score: float) -> QColor:
        position = map_score_to_palette_position(score, self._error_window_low, self._error_window_high)
        return interpolate_gradient_color(self._gradient_name, position)

    def _background_color_for_record(self, record: FrameRecord) -> QColor:
        score = self._display_score(record)
        if score is None:
            return QColor(MATRIX_BACKGROUND_ALT)
        return self._background_color(score)

    def _tooltip_for_record(self, record: FrameRecord) -> str:
        if not bool(getattr(record, "score_ready", False)):
            suffix = f"\n{self._t('matrix.reference_frame')}" if self._reference_key == record.key else ""
            return f"{record.display_name}\n{self._t('matrix.mismatch_not_computed')}{suffix}"
        lines = [record.display_name]
        if record.absolute_score is not None:
            lines.append(f"{self._t('matrix.absolute_mismatch')}: {self._format_metric_value(record.absolute_score)}")
        if record.relative_score is not None:
            lines.append(f"{self._t('matrix.relative_mismatch')}: {record.relative_score * 100.0:.3f}%")
        if record.score_percentile is not None:
            lines.append(f"Score percentile: P{float(record.score_percentile):.1f}")
        if self._reference_key == record.key:
            lines.append(self._t('matrix.reference_frame'))
        return "\n".join(lines)

    def _hover_text(self, record: FrameRecord) -> str:
        if not bool(getattr(record, "score_ready", False)):
            return f"{record.display_name} | {self._t('matrix.mismatch_not_computed').lower()}"
        parts = [record.display_name]
        if self._reference_key == record.key:
            parts.append(self._t('matrix.reference_short'))
        if record.absolute_score is not None:
            parts.append(f"{self._t('matrix.absolute_short')} {self._format_metric_value(record.absolute_score)}")
        if record.relative_score is not None:
            parts.append(f"{self._t('matrix.relative_short')} {record.relative_score * 100.0:.3f}%")
        if record.score_percentile is not None:
            parts.append(f"P{float(record.score_percentile):.1f}")
        return " | ".join(parts)


__all__ = [
    "GradientPresetSelectorWidget",
    "GradientRangeSelectorWidget",
    "MatrixLayoutConfig",
    "MatrixListWidget",
    "MatrixMiniMapWidget",
    "blend_colors",
    "build_matrix_layout",
    "error_palette_color",
    "extract_frame_number",
    "interpolate_gradient_color",
    "map_score_to_palette_position",
]





