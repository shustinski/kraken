"""Spatial frame-grid dimension editor.

This widget borrows the spatial layout of NeuralImage's resize control, but is
implemented independently and deals exclusively in frame counts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from PyQt6.QtCore import QEvent, QRectF, QSignalBlocker, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class GridOrientation(StrEnum):
    """Direction in which project Y coordinates increase."""

    Y_DOWN = "y_down"
    Y_UP = "y_up"


@dataclass(frozen=True, slots=True)
class GridDimensions:
    """A UI snapshot of immutable project-grid dimensions."""

    width: int
    height: int
    orientation: GridOrientation = GridOrientation.Y_DOWN

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("grid dimensions must be positive")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        try:
            normalized_orientation = GridOrientation(str(self.orientation))
        except ValueError as exc:
            raise ValueError(f"unsupported grid orientation: {self.orientation!r}") from exc
        object.__setattr__(self, "orientation", normalized_orientation)

    @property
    def frame_count(self) -> int:
        return self.width * self.height


class _NoWheelSpinBox(QSpinBox):
    """Ignore wheel changes while preserving normal scrolling in parent views."""

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()


class _ChainLinkButton(QToolButton):
    """Small font-independent link icon used for square grids."""

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        color = QColor("#f8fafc" if self.isChecked() else "#94a3b8")
        if not self.isEnabled():
            color = QColor("#64748b")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.translate(self.rect().center())
        painter.rotate(-35.0)
        painter.drawRoundedRect(QRectF(-12.0, -4.5, 15.0, 9.0), 4.5, 4.5)
        painter.drawRoundedRect(QRectF(-3.0, -4.5, 15.0, 9.0), 4.5, 4.5)


class GridDimensionsWidget(QWidget):
    """Edit width, height and Y orientation with a frame-count cap.

    Values may temporarily exceed ``maximum_frames`` so a wizard can explain
    the problem instead of silently changing user input. Call ``is_valid`` or
    ``validated_dimensions`` before committing a project.
    """

    dimensionsChanged = pyqtSignal(int, int, str)
    orientationChanged = pyqtSignal(str)
    validityChanged = pyqtSignal(bool, str)

    DEFAULT_MAXIMUM_FRAMES = 100_000
    MAXIMUM_AXIS = 10_000_000
    PREVIEW_MAX_WIDTH = 210
    PREVIEW_MAX_HEIGHT = 135
    PREVIEW_MIN_WIDTH = 112
    PREVIEW_MIN_HEIGHT = 82
    DEFAULT_TEXTS = {
        "preview": "Предпросмотр матрицы кадров",
        "width_input": "Количество кадров по горизонтали",
        "height_input": "Количество кадров по вертикали",
        "width_axis": "Горизонталь X",
        "height_axis": "Вертикаль Y",
        "frames": "Кадры",
        "unit": "кадр.",
        "linked": "X и Y связаны. Нажмите, чтобы менять их независимо.",
        "unlinked": "X и Y независимы. Нажмите, чтобы сделать матрицу квадратной.",
        "height_linked": "Высота равна ширине.",
        "orientation": "Направление координаты Y",
        "y_down": "Y вниз · начало сверху",
        "y_up": "Y вверх · начало снизу",
        "frame_count": "Всего кадров: {count:n}",
        "valid": "Размер допустим: {count:n} из {maximum:n} кадров",
        "cap_exceeded": "Превышен лимит: {count:n} из {maximum:n} кадров",
    }

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        orientation: GridOrientation | str = GridOrientation.Y_DOWN,
        parent: QWidget | None = None,
        *,
        maximum_frames: int = DEFAULT_MAXIMUM_FRAMES,
        maximum_axis: int = MAXIMUM_AXIS,
        link_dimensions: bool | None = None,
    ) -> None:
        super().__init__(parent)
        self._validate_positive(width, height)
        if int(maximum_frames) <= 0:
            raise ValueError("maximum_frames must be positive")
        if int(maximum_axis) <= 0:
            raise ValueError("maximum_axis must be positive")
        if width > maximum_axis or height > maximum_axis:
            raise ValueError("grid dimensions must not exceed maximum_axis")
        self._maximum_frames = int(maximum_frames)
        self._maximum_axis = int(maximum_axis)
        self._texts = dict(self.DEFAULT_TEXTS)
        self._last_validity: tuple[bool, str] | None = None

        initially_linked = width == height if link_dimensions is None else bool(link_dimensions and width == height)
        self._init_controls(width, height, self._coerce_orientation(orientation), initially_linked)
        self._init_layout()
        self._apply_stylesheet()
        self._connect_signals()
        self.apply_texts()
        self._publish(emit_dimensions=False)

    @staticmethod
    def _validate_positive(width: int, height: int) -> None:
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("grid dimensions must be positive")

    @staticmethod
    def _coerce_orientation(value: GridOrientation | str) -> GridOrientation:
        try:
            return GridOrientation(str(value))
        except ValueError as exc:
            raise ValueError(f"unsupported grid orientation: {value!r}") from exc

    def _init_controls(
        self,
        width: int,
        height: int,
        orientation: GridOrientation,
        link_dimensions: bool,
    ) -> None:
        self.setObjectName("gridDimensionsWidget")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.preview = QLabel()
        self.preview.setObjectName("gridPreview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.width_spinbox = self._make_spinbox(width)
        self.height_spinbox = self._make_spinbox(height)

        self.width_container = QFrame()
        self.width_container.setObjectName("axisContainer")
        width_layout = QHBoxLayout(self.width_container)
        width_layout.setContentsMargins(8, 5, 7, 5)
        width_layout.setSpacing(5)
        self.width_axis_badge = self._make_badge("↔", "axisBadge")
        self.width_unit_badge = self._make_badge("", "unitBadge")
        width_layout.addWidget(self.width_axis_badge)
        width_layout.addWidget(self.width_spinbox, 1)
        width_layout.addWidget(self.width_unit_badge)

        self.height_container = QFrame()
        self.height_container.setObjectName("axisContainer")
        height_layout = QVBoxLayout(self.height_container)
        height_layout.setContentsMargins(6, 7, 6, 7)
        height_layout.setSpacing(5)
        height_layout.addStretch(1)
        self.height_axis_badge = self._make_badge("↕", "axisBadge")
        self.height_unit_badge = self._make_badge("", "unitBadge")
        height_layout.addWidget(self.height_axis_badge, alignment=Qt.AlignmentFlag.AlignHCenter)
        height_layout.addWidget(self.height_spinbox)
        height_layout.addWidget(self.height_unit_badge, alignment=Qt.AlignmentFlag.AlignHCenter)
        height_layout.addStretch(1)

        self.size_lock_button = _ChainLinkButton()
        self.size_lock_button.setObjectName("sizeLockButton")
        self.size_lock_button.setCheckable(True)
        self.size_lock_button.setChecked(bool(link_dimensions))
        self.size_lock_button.setFixedSize(44, 44)

        self.orientation_label = QLabel()
        self.orientation_label.setObjectName("fieldLabel")
        self.orientation_combo = QComboBox()
        self.orientation_combo.setObjectName("orientationCombo")
        self.orientation_combo.addItem("", GridOrientation.Y_DOWN.value)
        self.orientation_combo.addItem("", GridOrientation.Y_UP.value)
        self.orientation_combo.setCurrentIndex(
            self.orientation_combo.findData(orientation.value)
        )

        self.frame_count_label = QLabel()
        self.frame_count_label.setObjectName("frameCountLabel")
        self.validation_label = QLabel()
        self.validation_label.setObjectName("validationLabel")
        self.validation_label.setWordWrap(True)

    def _make_spinbox(self, value: int) -> _NoWheelSpinBox:
        spinbox = _NoWheelSpinBox()
        spinbox.setObjectName("dimensionSpinBox")
        spinbox.setRange(1, self._maximum_axis)
        spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spinbox.setKeyboardTracking(True)
        spinbox.setValue(int(value))
        spinbox.setMinimumWidth(72)
        spinbox.setFixedHeight(32)
        return spinbox

    @staticmethod
    def _make_badge(text: str, object_name: str) -> QLabel:
        badge = QLabel(text)
        badge.setObjectName(object_name)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return badge

    def _init_layout(self) -> None:
        layout = QGridLayout(self)
        self.grid_layout = layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(7)
        layout.setVerticalSpacing(7)
        layout.addWidget(
            self.preview,
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
        )
        layout.addWidget(
            self.height_container,
            0,
            1,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
        )
        layout.addWidget(
            self.width_container,
            1,
            0,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )
        layout.addWidget(
            self.size_lock_button,
            1,
            1,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        orientation_row = QHBoxLayout()
        orientation_row.setContentsMargins(0, 0, 0, 0)
        orientation_row.addWidget(self.orientation_label)
        orientation_row.addWidget(self.orientation_combo, 1)
        layout.addLayout(orientation_row, 2, 0, 1, 2)
        layout.addWidget(self.frame_count_label, 3, 0, 1, 2)
        layout.addWidget(self.validation_label, 4, 0, 1, 2)

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QWidget#gridDimensionsWidget { background: transparent; color: #e2e8f0; }
            QLabel#gridPreview {
                background-color: #1e293b; color: #cbd5e1;
                border: 2px dashed #475569; border-radius: 10px;
                font-weight: 600;
            }
            QFrame#axisContainer {
                background-color: #0f172a; border: 1px solid #475569;
                border-radius: 8px;
            }
            QSpinBox#dimensionSpinBox, QComboBox#orientationCombo {
                background-color: #1e293b; color: #f8fafc;
                border: 1px solid #64748b; border-radius: 7px; padding: 3px 6px;
            }
            QSpinBox#dimensionSpinBox:focus, QComboBox#orientationCombo:focus {
                border: 2px solid #818cf8;
            }
            QLabel#axisBadge { color: #94a3b8; border: none; font-size: 15px; font-weight: 600; }
            QLabel#unitBadge {
                color: #cbd5e1; background-color: #1e293b; border: 1px solid #475569;
                border-radius: 6px; padding: 2px 5px; font-size: 10px; font-weight: 600;
            }
            QToolButton#sizeLockButton {
                background-color: #1e293b; border: 1px solid #475569; border-radius: 9px;
            }
            QToolButton#sizeLockButton:checked { background-color: #4f46e5; border-color: #818cf8; }
            QLabel#frameCountLabel { color: #e2e8f0; font-weight: 600; }
            QLabel#validationLabel[valid="true"] { color: #86efac; }
            QLabel#validationLabel[valid="false"] { color: #fca5a5; }
            QLabel#fieldLabel { color: #cbd5e1; }
            """
        )

    def _connect_signals(self) -> None:
        self.width_spinbox.valueChanged.connect(self._on_width_changed)
        self.height_spinbox.valueChanged.connect(self._on_height_changed)
        self.size_lock_button.toggled.connect(self._on_lock_toggled)
        self.orientation_combo.currentIndexChanged.connect(self._on_orientation_changed)

    def _on_width_changed(self, width: int) -> None:
        if self.size_lock_button.isChecked():
            with QSignalBlocker(self.height_spinbox):
                self.height_spinbox.setValue(width)
        self._publish()

    def _on_height_changed(self, height: int) -> None:
        if self.size_lock_button.isChecked():
            with QSignalBlocker(self.width_spinbox):
                self.width_spinbox.setValue(height)
        self._publish()

    def _on_lock_toggled(self, checked: bool) -> None:
        self._sync_lock_state()
        if checked:
            with QSignalBlocker(self.height_spinbox):
                self.height_spinbox.setValue(self.width_spinbox.value())
        self._publish()

    def _on_orientation_changed(self) -> None:
        value = self.orientation().value
        self.orientationChanged.emit(value)
        self._publish()

    def _sync_lock_state(self) -> None:
        linked = self.size_lock_button.isChecked()
        self.height_spinbox.setEnabled(not linked)
        self.size_lock_button.setToolTip(self._texts["linked" if linked else "unlinked"])
        self.size_lock_button.setAccessibleName(self.size_lock_button.toolTip())
        self.height_spinbox.setToolTip(
            self._texts["height_linked" if linked else "height_input"]
        )

    def _publish(self, *, emit_dimensions: bool = True) -> None:
        self._update_preview()
        valid = self.is_valid()
        message = self.validation_message()
        self.validation_label.setText(message)
        self.validation_label.setProperty("valid", valid)
        self.validation_label.style().unpolish(self.validation_label)
        self.validation_label.style().polish(self.validation_label)
        state = (valid, message)
        if state != self._last_validity:
            self._last_validity = state
            self.validityChanged.emit(valid, message)
        if emit_dimensions:
            dimensions = self.dimensions()
            self.dimensionsChanged.emit(
                dimensions.width,
                dimensions.height,
                dimensions.orientation.value,
            )

    def _update_preview(self) -> None:
        dimensions = self.dimensions()
        display_size = self._preview_size(dimensions.width, dimensions.height)
        self.preview.setFixedSize(display_size)
        self.preview.setText(f"{dimensions.width} × {dimensions.height}")
        self.width_container.setFixedSize(display_size.width(), 44)
        self.height_container.setFixedSize(96, display_size.height())
        self.frame_count_label.setText(
            self._texts["frame_count"].format(count=dimensions.frame_count)
        )
        self.updateGeometry()

    def _preview_size(self, width: int, height: int) -> QSize:
        ratio = float(width) / float(height)
        if ratio >= self.PREVIEW_MAX_WIDTH / self.PREVIEW_MAX_HEIGHT:
            display_width = self.PREVIEW_MAX_WIDTH
            display_height = round(display_width / ratio)
        else:
            display_height = self.PREVIEW_MAX_HEIGHT
            display_width = round(display_height * ratio)
        return QSize(
            max(self.PREVIEW_MIN_WIDTH, min(self.PREVIEW_MAX_WIDTH, display_width)),
            max(self.PREVIEW_MIN_HEIGHT, min(self.PREVIEW_MAX_HEIGHT, display_height)),
        )

    def dimensions(self) -> GridDimensions:
        """Return the current values, including a temporarily invalid size."""
        return GridDimensions(
            self.width_spinbox.value(),
            self.height_spinbox.value(),
            self.orientation(),
        )

    def validated_dimensions(self) -> GridDimensions:
        """Return dimensions suitable for commit or raise a validation error."""
        dimensions = self.dimensions()
        if not self.is_valid():
            raise ValueError(self.validation_message())
        return dimensions

    def target_size(self) -> QSize:
        """Compatibility-friendly QSize representation of frame dimensions."""
        return QSize(self.width_spinbox.value(), self.height_spinbox.value())

    def orientation(self) -> GridOrientation:
        value = self.orientation_combo.currentData()
        return self._coerce_orientation(str(value))

    def frame_count(self) -> int:
        return self.width_spinbox.value() * self.height_spinbox.value()

    def maximum_frames(self) -> int:
        return self._maximum_frames

    def is_valid(self) -> bool:
        return self.frame_count() <= self._maximum_frames

    def validation_message(self) -> str:
        key = "valid" if self.is_valid() else "cap_exceeded"
        return self._texts[key].format(
            count=self.frame_count(),
            maximum=self._maximum_frames,
        )

    def set_dimensions(
        self,
        width: int,
        height: int,
        orientation: GridOrientation | str | None = None,
    ) -> None:
        self._validate_positive(width, height)
        if width > self._maximum_axis or height > self._maximum_axis:
            raise ValueError("grid dimensions must not exceed maximum_axis")
        with (
            QSignalBlocker(self.width_spinbox),
            QSignalBlocker(self.height_spinbox),
            QSignalBlocker(self.size_lock_button),
            QSignalBlocker(self.orientation_combo),
        ):
            self.width_spinbox.setValue(int(width))
            self.height_spinbox.setValue(int(height))
            self.size_lock_button.setChecked(int(width) == int(height))
            if orientation is not None:
                normalized = self._coerce_orientation(orientation)
                self.orientation_combo.setCurrentIndex(
                    self.orientation_combo.findData(normalized.value)
                )
        self._sync_lock_state()
        self._publish()

    set_target_size = set_dimensions

    def set_maximum_frames(self, maximum_frames: int) -> None:
        if int(maximum_frames) <= 0:
            raise ValueError("maximum_frames must be positive")
        self._maximum_frames = int(maximum_frames)
        self._publish()

    def set_orientation(self, orientation: GridOrientation | str) -> None:
        normalized = self._coerce_orientation(orientation)
        index = self.orientation_combo.findData(normalized.value)
        if index < 0:
            raise ValueError(f"unsupported grid orientation: {orientation!r}")
        self.orientation_combo.setCurrentIndex(index)

    def apply_texts(self, texts: Mapping[str, str] | None = None) -> None:
        """Apply localized strings without coupling the widget to an i18n service."""
        localized = dict(self.DEFAULT_TEXTS)
        if texts is not None:
            localized.update({str(key): str(value) for key, value in texts.items()})
        self._texts = localized
        self.preview.setAccessibleName(localized["preview"])
        self.width_spinbox.setAccessibleName(localized["width_input"])
        self.width_spinbox.setToolTip(localized["width_input"])
        self.height_spinbox.setAccessibleName(localized["height_input"])
        self.width_axis_badge.setToolTip(localized["width_axis"])
        self.height_axis_badge.setToolTip(localized["height_axis"])
        for badge in (self.width_unit_badge, self.height_unit_badge):
            badge.setText(localized["unit"])
            badge.setToolTip(localized["frames"])
            badge.setAccessibleName(localized["frames"])
        self.orientation_label.setText(localized["orientation"])
        self.orientation_combo.setItemText(0, localized["y_down"])
        self.orientation_combo.setItemText(1, localized["y_up"])
        self._sync_lock_state()
        self._publish(emit_dimensions=False)

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            self._sync_lock_state()
