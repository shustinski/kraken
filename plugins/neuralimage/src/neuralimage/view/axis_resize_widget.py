"""Modern axis-based image resize control for the NeuralImage plugin."""

from __future__ import annotations

import sys
from collections.abc import Mapping

from PyQt6.QtCore import QRectF, QSignalBlocker, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from neuralimage.view.settings_panel_widgets import NoWheelSpinBox


class _ChainLinkButton(QToolButton):
    """Tool button with a font-independent pair of connected chain links."""

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        color = QColor("#ffffff" if self.isChecked() else "#a1a1aa")
        if not self.isEnabled():
            color = QColor("#52525b")

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.translate(self.rect().center())
        painter.rotate(-35.0)
        painter.drawRoundedRect(QRectF(-13.0, -5.0, 16.0, 10.0), 5.0, 5.0)
        painter.drawRoundedRect(QRectF(-3.0, -5.0, 16.0, 10.0), 5.0, 5.0)


class AxisResizeWidget(QWidget):
    """A spatial width/height editor with an aspect-ratio preview."""

    sizeChanged = pyqtSignal(int, int)

    MINIMUM_PIXELS = 1
    MAXIMUM_PIXELS = 16_384
    PREVIEW_MAX_WIDTH = 200
    PREVIEW_MAX_HEIGHT = 130
    PREVIEW_MIN_WIDTH = 110
    PREVIEW_MIN_HEIGHT = 80
    DEFAULT_TEXTS = {
        "preview": "Image size preview",
        "width_input": "Width in pixels",
        "height_input": "Height in pixels",
        "width_axis": "Horizontal size",
        "height_axis": "Vertical size",
        "pixels": "Pixels",
        "unit": "px",
        "linked": "X and Y are linked. Click to edit them independently.",
        "unlinked": "X and Y are independent. Click to make them equal.",
        "height_linked": "Height is kept equal to width.",
    }

    def __init__(
        self,
        target_width: int = 256,
        target_height: int = 256,
        parent: QWidget | None = None,
        *,
        minimum_pixels: int = MINIMUM_PIXELS,
        maximum_pixels: int = MAXIMUM_PIXELS,
        single_step: int = 1,
    ) -> None:
        super().__init__(parent)
        self._validate_dimensions(target_width, target_height, "target")
        if int(minimum_pixels) <= 0 or int(maximum_pixels) < int(minimum_pixels):
            raise ValueError("pixel range must be positive and ordered")
        if int(single_step) <= 0:
            raise ValueError("single_step must be positive")
        if not int(minimum_pixels) <= int(target_width) <= int(maximum_pixels):
            raise ValueError("target width must be inside the pixel range")
        if not int(minimum_pixels) <= int(target_height) <= int(maximum_pixels):
            raise ValueError("target height must be inside the pixel range")
        self._minimum_pixels = int(minimum_pixels)
        self._maximum_pixels = int(maximum_pixels)
        self._single_step = int(single_step)
        self._localized_texts = dict(self.DEFAULT_TEXTS)

        self._init_controls(target_width, target_height)
        self._init_layout()
        self._apply_stylesheet()
        self._connect_signals()
        self.apply_texts()
        self._update_preview()

    @staticmethod
    def _validate_dimensions(width: int, height: int, label: str) -> None:
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError(f"{label} dimensions must be positive")

    def _init_controls(self, target_width: int, target_height: int) -> None:
        self.setObjectName("axisResizeWidget")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.preview = QLabel()
        self.preview.setObjectName("imagePreview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setAccessibleName(self._localized_texts["preview"])

        self.width_spinbox = self._create_spinbox(target_width, "Width in pixels")
        self.height_spinbox = self._create_spinbox(target_height, "Height in pixels")

        self.width_container = QFrame()
        self.width_container.setObjectName("axisContainer")
        width_layout = QHBoxLayout(self.width_container)
        width_layout.setContentsMargins(8, 5, 7, 5)
        width_layout.setSpacing(5)
        self.width_axis_badge = self._create_badge("↔", "axisBadge", self._localized_texts["width_axis"])
        width_layout.addWidget(self.width_axis_badge)
        width_layout.addWidget(self.width_spinbox, 1)
        self.width_unit_badge = self._create_badge(
            self._localized_texts["unit"],
            "unitBadge",
            self._localized_texts["pixels"],
        )
        width_layout.addWidget(self.width_unit_badge)

        self.height_container = QFrame()
        self.height_container.setObjectName("axisContainer")
        height_layout = QVBoxLayout(self.height_container)
        height_layout.setContentsMargins(6, 7, 6, 7)
        height_layout.setSpacing(5)
        height_layout.addStretch(1)
        self.height_axis_badge = self._create_badge("↕", "axisBadge", self._localized_texts["height_axis"])
        height_layout.addWidget(self.height_axis_badge, alignment=Qt.AlignmentFlag.AlignHCenter)
        height_layout.addWidget(self.height_spinbox)
        self.height_unit_badge = self._create_badge(
            self._localized_texts["unit"],
            "unitBadge",
            self._localized_texts["pixels"],
        )
        height_layout.addWidget(self.height_unit_badge, alignment=Qt.AlignmentFlag.AlignHCenter)
        height_layout.addStretch(1)

        self.size_lock_button = _ChainLinkButton()
        self.size_lock_button.setObjectName("sizeLockButton")
        self.size_lock_button.setText("")
        self.size_lock_button.setCheckable(True)
        self.size_lock_button.setChecked(int(target_width) == int(target_height))
        self.size_lock_button.setAccessibleName(self._localized_texts["linked"])
        self.size_lock_button.setFixedSize(44, 44)

    def _create_spinbox(self, value: int, accessible_name: str) -> NoWheelSpinBox:
        spinbox = NoWheelSpinBox()
        spinbox.setObjectName("dimensionSpinBox")
        spinbox.setRange(self._minimum_pixels, self._maximum_pixels)
        spinbox.setSingleStep(self._single_step)
        spinbox.setButtonSymbols(NoWheelSpinBox.ButtonSymbols.NoButtons)
        spinbox.setKeyboardTracking(True)
        spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spinbox.setAccessibleName(accessible_name)
        spinbox.setValue(int(value))
        spinbox.setMinimumWidth(64)
        spinbox.setFixedHeight(32)
        return spinbox

    @staticmethod
    def _create_badge(text: str, object_name: str, accessible_name: str) -> QLabel:
        badge = QLabel(text)
        badge.setObjectName(object_name)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setAccessibleName(accessible_name)
        return badge

    def _init_layout(self) -> None:
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(7)
        self.grid_layout.setVerticalSpacing(7)
        self.grid_layout.addWidget(
            self.preview,
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
        )
        self.grid_layout.addWidget(
            self.height_container,
            0,
            1,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
        )
        self.grid_layout.addWidget(
            self.width_container,
            1,
            0,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )
        self.grid_layout.addWidget(
            self.size_lock_button,
            1,
            1,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QWidget#axisResizeWidget {
                background: transparent;
                color: #f4f4f5;
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 13px;
            }
            QLabel#imagePreview {
                background-color: #27272a;
                color: #a1a1aa;
                border: 2px dashed #3f3f46;
                border-radius: 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QFrame#axisContainer {
                background-color: #18181b;
                border: 1px solid #3f3f46;
                border-radius: 8px;
            }
            QSpinBox#dimensionSpinBox {
                background-color: #27272a;
                color: #fafafa;
                border: 1px solid #52525b;
                border-radius: 7px;
                padding: 3px 6px;
                selection-background-color: #4f46e5;
            }
            QSpinBox#dimensionSpinBox:focus {
                border: 2px solid #818cf8;
                padding: 2px 5px;
            }
            QSpinBox#dimensionSpinBox:disabled {
                background-color: #18181b;
                color: #71717a;
                border-color: #3f3f46;
            }
            QLabel#axisBadge {
                color: #a1a1aa;
                font-size: 15px;
                font-weight: 600;
                border: none;
                background: transparent;
            }
            QLabel#unitBadge {
                color: #a1a1aa;
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 2px 5px;
                font-size: 10px;
                font-weight: 600;
            }
            QToolButton#sizeLockButton {
                background-color: #27272a;
                color: #71717a;
                border: 1px solid #3f3f46;
                border-radius: 9px;
                font-size: 16px;
            }
            QToolButton#sizeLockButton:hover {
                background-color: #3f3f46;
                color: #d4d4d8;
            }
            QToolButton#sizeLockButton:focus {
                border: 2px solid #818cf8;
            }
            QToolButton#sizeLockButton:checked {
                background-color: #4f46e5;
                color: #ffffff;
                border-color: #6366f1;
            }
            """
        )

    def _connect_signals(self) -> None:
        self.width_spinbox.valueChanged.connect(self._on_width_changed)
        self.height_spinbox.valueChanged.connect(self._on_height_changed)
        self.size_lock_button.toggled.connect(self._on_lock_toggled)

    def _on_width_changed(self, width: int) -> None:
        if self.size_lock_button.isChecked():
            height = self._bounded(width)
            with QSignalBlocker(self.height_spinbox):
                self.height_spinbox.setValue(height)
        self._publish_size()

    def _on_height_changed(self, height: int) -> None:
        if self.size_lock_button.isChecked():
            width = self._bounded(height)
            with QSignalBlocker(self.width_spinbox):
                self.width_spinbox.setValue(width)
        self._publish_size()

    def _on_lock_toggled(self, checked: bool) -> None:
        self._sync_lock_state()
        if checked:
            height = self._bounded(self.width_spinbox.value())
            with QSignalBlocker(self.height_spinbox):
                self.height_spinbox.setValue(height)
        self._publish_size()

    def _sync_lock_state(self) -> None:
        locked = self.size_lock_button.isChecked()
        self.height_spinbox.setEnabled(not locked)
        self.size_lock_button.setToolTip(
            self._localized_texts["linked"] if locked else self._localized_texts["unlinked"]
        )
        self.size_lock_button.setAccessibleName(self.size_lock_button.toolTip())
        self.height_spinbox.setToolTip(
            self._localized_texts["height_linked"] if locked else self._localized_texts["height_input"]
        )

    def _publish_size(self) -> None:
        self._update_preview()
        self.sizeChanged.emit(self.width_spinbox.value(), self.height_spinbox.value())

    def _bounded(self, value: int) -> int:
        return max(self._minimum_pixels, min(self._maximum_pixels, int(value)))

    def _update_preview(self) -> None:
        width = self.width_spinbox.value()
        height = self.height_spinbox.value()
        display_size = self._preview_size(width, height)

        self.preview.setFixedSize(display_size)
        self.width_container.setFixedSize(display_size.width(), 44)
        self.height_container.setFixedSize(96, display_size.height())
        self.preview.setText(f"{width} × {height}")
        self.updateGeometry()

    def _preview_size(self, width: int, height: int) -> QSize:
        ratio = float(width) / float(max(1, height))
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

    def target_size(self) -> QSize:
        """Return the currently selected target dimensions."""
        return QSize(self.width_spinbox.value(), self.height_spinbox.value())

    def apply_texts(self, texts: Mapping[str, str] | None = None) -> None:
        """Apply localized UI strings without changing dimensions or lock state."""
        localized = dict(self.DEFAULT_TEXTS)
        if texts is not None:
            localized.update({str(key): str(value) for key, value in texts.items() if value is not None})
        self._localized_texts = localized

        self.preview.setAccessibleName(localized["preview"])
        self.width_spinbox.setAccessibleName(localized["width_input"])
        self.width_spinbox.setToolTip(localized["width_input"])
        self.height_spinbox.setAccessibleName(localized["height_input"])
        self.width_axis_badge.setAccessibleName(localized["width_axis"])
        self.width_axis_badge.setToolTip(localized["width_axis"])
        self.height_axis_badge.setAccessibleName(localized["height_axis"])
        self.height_axis_badge.setToolTip(localized["height_axis"])
        for badge in (self.width_unit_badge, self.height_unit_badge):
            badge.setText(localized["unit"])
            badge.setAccessibleName(localized["pixels"])
            badge.setToolTip(localized["pixels"])
        self._sync_lock_state()

    def set_target_size(self, width: int, height: int) -> None:
        """Set both dimensions and link them only when their values are equal."""
        self._validate_dimensions(width, height, "target")
        bounded_width = self._bounded(width)
        bounded_height = self._bounded(height)
        with (
            QSignalBlocker(self.width_spinbox),
            QSignalBlocker(self.height_spinbox),
            QSignalBlocker(self.size_lock_button),
        ):
            self.width_spinbox.setValue(bounded_width)
            self.height_spinbox.setValue(bounded_height)
            self.size_lock_button.setChecked(bounded_width == bounded_height)
        self._sync_lock_state()
        self._publish_size()


class ResizeDemoWindow(QWidget):
    """Small standalone host used when this module is run directly."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NeuralImage · Resize")
        self.setStyleSheet("background-color: #09090b;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 32)
        layout.setSpacing(18)

        title = QLabel("Image dimensions")
        title.setStyleSheet(
            'color: #fafafa; font-family: "Segoe UI", "Inter", sans-serif; '
            "font-size: 20px; font-weight: 700;"
        )
        layout.addWidget(title)
        layout.addWidget(AxisResizeWidget(), alignment=Qt.AlignmentFlag.AlignCenter)


def main() -> int:
    app = QApplication(sys.argv)
    window = ResizeDemoWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
