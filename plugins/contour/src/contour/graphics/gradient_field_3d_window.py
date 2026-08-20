"""Separate window for the interactive 3D gradient-field preview."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer
from PyQt6.QtGui import QMouseEvent, QPixmap, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..ui.no_wheel_controls import NoWheelComboBox as QComboBox

from ..adapters.qt.image_conversion import cv_to_qimage
from .gradient_field_3d import (
    DEFAULT_AZIMUTH_DEG,
    DEFAULT_ELEVATION_DEG,
    HEIGHT_MODE_INTENSITY,
    HEIGHT_MODE_MAGNITUDE,
    PREVIEW_MAX_SIDE,
    GradientField3DModel,
    prepare_gradient_field_3d,
    render_gradient_field_3d_bgr,
)


class GradientField3DWindow(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setModal(False)
        self.resize(980, 760)
        self._gradient_x = None
        self._gradient_y = None
        self._intensity = None
        self._model: GradientField3DModel | None = None
        self._preview_model: GradientField3DModel | None = None
        self._azimuth = DEFAULT_AZIMUTH_DEG
        self._elevation = DEFAULT_ELEVATION_DEG
        self._zoom = 1.0
        self._drag_origin: QPoint | None = None
        self._drag_azimuth = self._azimuth
        self._drag_elevation = self._elevation
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(12)
        self._render_timer.timeout.connect(self._redraw)
        self._interacting = False
        self._full_redraw_timer = QTimer(self)
        self._full_redraw_timer.setSingleShot(True)
        self._full_redraw_timer.setInterval(70)
        self._full_redraw_timer.timeout.connect(self._redraw_full)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(8, 6, 8, 6)
        self._height_combo = QComboBox()
        self._height_combo.addItem("", HEIGHT_MODE_MAGNITUDE)
        self._height_combo.addItem("", HEIGHT_MODE_INTENSITY)
        self._height_combo.currentIndexChanged.connect(self._rebuild_model)
        self._height_combo.installEventFilter(self)
        self._hint = QLabel()
        self._hint.setWordWrap(True)
        top.addWidget(self._height_combo, 0)
        top.addWidget(self._hint, 1)
        layout.addLayout(top)
        self._view = QLabel()
        self._view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view.setMinimumSize(320, 240)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view.setStyleSheet("background: #161820; margin: 0; padding: 0;")
        self._view.setMouseTracking(True)
        layout.addWidget(self._view, 1)
        self._view.installEventFilter(self)
        self.set_ui_language("ru")

    def set_ui_language(self, language: str) -> None:
        ru = language == "ru"
        self.setWindowTitle("Градиентное поле — 3D" if ru else "Gradient field — 3D")
        self._height_combo.setItemText(0, "Высота: модуль градиента" if ru else "Height: gradient magnitude")
        self._height_combo.setItemText(1, "Высота: яркость кадра" if ru else "Height: image intensity")
        self._hint.setText(
            "ЛКМ — вращение, колесо — масштаб."
            if ru
            else "Left drag rotates, mouse wheel zooms."
        )

    def set_field(
        self,
        gradient_x,
        gradient_y,
        *,
        intensity=None,
        language: str = "ru",
    ) -> None:
        self.set_ui_language(language)
        self._gradient_x = gradient_x
        self._gradient_y = gradient_y
        self._intensity = intensity
        self._rebuild_model()
        self.show()
        self.raise_()
        self.activateWindow()

    def eventFilter(self, watched, event) -> bool:  # noqa: ANN001
        if watched is self._height_combo and event.type() == QEvent.Type.Wheel:
            view = self._height_combo.view()
            if view is not None and view.isVisible():
                return False
            return True
        if watched is not self._view:
            return super().eventFilter(watched, event)
        if isinstance(event, QMouseEvent):
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_origin = event.position().toPoint()
                self._drag_azimuth = self._azimuth
                self._drag_elevation = self._elevation
                self._interacting = True
                return True
            if event.type() == QEvent.Type.MouseMove and self._drag_origin is not None:
                delta = event.position().toPoint() - self._drag_origin
                self._azimuth = self._drag_azimuth + delta.x() * 0.45
                self._elevation = max(8.0, min(82.0, self._drag_elevation - delta.y() * 0.28))
                self._schedule_redraw()
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                self._drag_origin = None
                self._interacting = False
                self._full_redraw_timer.start()
                return True
        if isinstance(event, QWheelEvent):
            steps = event.angleDelta().y() / 120.0
            self._zoom = max(0.45, min(2.6, self._zoom * (1.12 ** steps)))
            self._interacting = True
            self._schedule_redraw()
            self._full_redraw_timer.start()
            return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._interacting = False
        self._schedule_redraw()

    def _rebuild_model(self) -> None:
        if self._gradient_x is None or self._gradient_y is None:
            self._model = None
            self._preview_model = None
            self._redraw()
            return
        height_mode = str(self._height_combo.currentData() or HEIGHT_MODE_MAGNITUDE)
        self._preview_model = prepare_gradient_field_3d(
            self._gradient_x,
            self._gradient_y,
            intensity=self._intensity,
            height_mode=height_mode,
            max_side=PREVIEW_MAX_SIDE,
            streamline_count=0,
        )
        self._model = prepare_gradient_field_3d(
            self._gradient_x,
            self._gradient_y,
            intensity=self._intensity,
            height_mode=height_mode,
        )
        self._interacting = False
        self._redraw()

    def _schedule_redraw(self) -> None:
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _redraw_full(self) -> None:
        self._interacting = False
        self._redraw()

    def _redraw(self) -> None:
        preview = self._interacting
        model = self._preview_model if preview and self._preview_model is not None else self._model
        if model is None:
            self._view.setPixmap(QPixmap())
            return
        size = self._view.size()
        width = max(2, size.width())
        height = max(2, size.height())
        if preview:
            width = max(2, min(width, 640))
            height = max(2, min(height, 480))
        image = render_gradient_field_3d_bgr(
            model,
            width=width,
            height=height,
            azimuth_deg=self._azimuth,
            elevation_deg=self._elevation,
            zoom=self._zoom,
            preview=preview,
        )
        qimage = cv_to_qimage(image)
        if qimage.isNull():
            self._view.setPixmap(QPixmap())
            return
        pixmap = QPixmap.fromImage(qimage)
        if pixmap.size() != size:
            pixmap = pixmap.scaled(
                size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation if preview else Qt.TransformationMode.SmoothTransformation,
            )
        self._view.setPixmap(pixmap)
