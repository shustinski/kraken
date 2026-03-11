from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRect
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

SIZE_WIDGET_SPACING = 5


class SlidingPanel(QScrollArea):
    """Animated sliding panel wrapper."""

    def __init__(self, widget: QWidget, width: int = 450, duration: int = 350, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._content = widget
        self._width = width
        self._duration = duration
        self._animating_in = False

        self.setWidget(self._content)
        self.setWidgetResizable(True)

        self.hide()
        self._animation = QPropertyAnimation(self, b'geometry')
        self._animation.setDuration(self._duration)
        self._animation.setEasingCurve(QEasingCurve.Type.BezierSpline)
        self._animation.finished.connect(self._on_animation_finished)

    def toggle(self) -> None:
        if self.isVisible():
            self._slide_out()
        else:
            self._slide_in()

    def set_width(self, width: int) -> None:
        self._width = width
        self.setMinimumWidth(self._width)

    def _slide_in(self) -> None:
        self.show()
        parent = self.parent()
        if not isinstance(parent, QWidget):
            return

        parent_rect = parent.rect()
        start = QRect(parent_rect.right(), 0, self._width, parent_rect.height())
        end = QRect(parent_rect.right() - self._width, 0, self._width, parent_rect.height())
        self._animating_in = True
        self._animation.stop()
        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.start()

    def _slide_out(self) -> None:
        parent = self.parent()
        if not isinstance(parent, QWidget):
            return

        parent_rect = parent.rect()
        start = QRect(parent_rect.right() - self._width, 0, self._width, parent_rect.height())
        end = QRect(parent_rect.right(), 0, self._width, parent_rect.height())
        self._animating_in = False
        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.start()

    def _on_animation_finished(self) -> None:
        if not self._animating_in:
            self.hide()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


def get_text_index_in_qcombobox(combobox: QComboBox, text: str) -> int:
    """Return the index of an exact text match in a combobox or `-1`."""
    if not isinstance(combobox, QComboBox):
        raise TypeError('combobox must be a QComboBox instance')
    item_texts = [combobox.itemText(i) for i in range(combobox.count())]
    try:
        text_location = item_texts.index(text)
    except ValueError:
        text_location = -1
    return text_location


def create_spinbox(
    spin_range: tuple[int, int],
    step: int,
    default_value: int,
    policy: QSizePolicy | None = None,
) -> QSpinBox:
    """Create a `QSpinBox` with validation and disabled wheel scrolling."""
    if len(spin_range) != 2:
        raise ValueError('spin_range must contain exactly two values')
    min_value, max_value = spin_range
    if min_value > max_value:
        raise ValueError('spin_range min value must be <= max value')
    if step <= 0:
        raise ValueError('step must be > 0')
    if not (min_value <= default_value <= max_value):
        raise ValueError('default_value must be inside spin_range')
    spinbox = NoWheelSpinBox()
    spinbox.setRange(min_value, max_value)
    spinbox.setValue(default_value)
    spinbox.setSingleStep(step)
    if isinstance(policy, QSizePolicy):
        spinbox.setSizePolicy(policy)
    return spinbox


def create_double_spinbox(
    spin_range: tuple[float, float],
    step: float,
    default_value: float,
    decimals: int = 6,
    policy: QSizePolicy | None = None,
) -> QDoubleSpinBox:
    """Create a `QDoubleSpinBox` with validation and disabled wheel scrolling."""
    if len(spin_range) != 2:
        raise ValueError('spin_range must contain exactly two values')
    min_value, max_value = spin_range
    if min_value > max_value:
        raise ValueError('spin_range min value must be <= max value')
    if step <= 0:
        raise ValueError('step must be > 0')
    if decimals < 0:
        raise ValueError('decimals must be >= 0')
    if not (min_value <= default_value <= max_value):
        raise ValueError('default_value must be inside spin_range')
    spinbox = NoWheelDoubleSpinBox()
    spinbox.setRange(min_value, max_value)
    spinbox.setValue(default_value)
    spinbox.setSingleStep(step)
    spinbox.setDecimals(decimals)
    spinbox.setKeyboardTracking(False)
    if isinstance(policy, QSizePolicy):
        spinbox.setSizePolicy(policy)
    return spinbox


def create_size_widget(x_size: QWidget, y_size: QWidget) -> QWidget:
    """Compose two controls into a single `X x Y` size widget row."""
    if not isinstance(x_size, QWidget) or not isinstance(y_size, QWidget):
        raise TypeError('x_size and y_size must be QWidget instances')
    size_widget = QWidget()
    row_layout = QHBoxLayout(size_widget)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(SIZE_WIDGET_SPACING)
    row_layout.addWidget(x_size)
    row_layout.addWidget(QLabel('X'))
    row_layout.addWidget(y_size)
    return size_widget
