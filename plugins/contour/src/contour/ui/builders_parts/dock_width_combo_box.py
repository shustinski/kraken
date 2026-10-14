"""builders: dock width combo box."""
from __future__ import annotations
from PyQt6.QtCore import QEvent, QPoint, QSize, Qt, QTimer
from PyQt6.QtWidgets import (
    QSizePolicy,
    QWidget,
)
from contour.ui.no_wheel_controls import NoWheelComboBox as QComboBox

class _DockWidthComboBox(QComboBox):
    """Keep the popup inside the same horizontal bounds as the combo box."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._width_locked_popup: QWidget | None = None
        self._locking_popup_width = False
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.view().setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def showPopup(self) -> None:
        # Do not set a fixed width on the internal view: QComboBox includes that
        # width in its own minimum-size hint, which can make the field extend
        # beyond a narrow dock and only appear clipped.
        super().showPopup()
        popup = self.view().window()
        if popup is not self._width_locked_popup:
            if self._width_locked_popup is not None:
                self._width_locked_popup.removeEventFilter(self)
            self._width_locked_popup = popup
            popup.installEventFilter(self)
        self._fit_popup_to_combo()
        QTimer.singleShot(0, self._fit_popup_to_combo)
        QTimer.singleShot(50, self._fit_popup_to_combo)

    def _fit_popup_to_combo(self) -> None:
        popup = self._width_locked_popup or self.view().window()
        if popup is None or not popup.isVisible():
            return
        visible_rect = self.visibleRegion().boundingRect()
        target_width = max(1, min(self.width(), visible_rect.width()))
        self._locking_popup_width = True
        try:
            popup.setFixedWidth(target_width)
            popup_top_left = self.mapToGlobal(QPoint(visible_rect.left(), self.height()))
            popup.setGeometry(
                popup_top_left.x(),
                popup_top_left.y(),
                target_width,
                popup.height(),
            )
            window_handle = popup.windowHandle()
            if window_handle is not None:
                window_handle.setMinimumSize(QSize(target_width, 1))
                window_handle.setMaximumSize(QSize(target_width, 16_777_215))
        finally:
            self._locking_popup_width = False

    def eventFilter(self, watched: object, event: QEvent | None) -> bool:
        if event is None:
            return False
        if (
            watched is self._width_locked_popup
            and not self._locking_popup_width
            and event.type()
            in (
                QEvent.Type.Show,
                QEvent.Type.Resize,
                QEvent.Type.LayoutRequest,
            )
        ):
            QTimer.singleShot(0, self._fit_popup_to_combo)
        return super().eventFilter(watched, event)
