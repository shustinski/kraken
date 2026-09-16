from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRect, Qt
from PyQt6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

SIZE_WIDGET_SPACING = 5


class SettingsCard(QFrame):
    """Small collapsible settings container with a concise status header."""

    def __init__(
        self,
        title: str,
        content: QWidget,
        *,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName('settingsCard')
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._content = content
        self._defaults: list[tuple[QWidget, object]] = []

        self.toggle_button = QToolButton(self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.toggle_button.setText(title)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.summary_label = QLabel('')
        self.summary_label.setObjectName('settingsCardSummary')
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.error_label = QLabel('')
        self.error_label.setObjectName('settingsCardError')
        self.error_label.setStyleSheet('color: #ef5350; font-weight: 600;')
        self.reset_button = QPushButton('Reset')
        self.reset_button.setObjectName('settingsCardReset')
        self.reset_button.setFlat(True)

        header = QHBoxLayout()
        header.setContentsMargins(6, 2, 6, 2)
        header.setSpacing(6)
        header.addWidget(self.toggle_button, 1)
        header.addWidget(self.summary_label)
        header.addWidget(self.error_label)
        header.addWidget(self.reset_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 6)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(content)
        content.setVisible(expanded)

        self.toggle_button.toggled.connect(self.set_expanded)
        self.reset_button.clicked.connect(self.reset_to_defaults)

    def title(self) -> str:
        return self.toggle_button.text()

    def set_title(self, title: str) -> None:
        self.toggle_button.setText(str(title))

    def set_summary(self, summary: str) -> None:
        self.summary_label.setText(str(summary))

    def set_error_count(self, count: int) -> None:
        self.error_label.setText(f'! {int(count)}' if count else '')

    def is_expanded(self) -> bool:
        return self.toggle_button.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        visible = bool(expanded)
        self.toggle_button.setChecked(visible)
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(visible)

    def capture_defaults(self) -> None:
        self._defaults = []
        controls = self._content.findChildren(QWidget)
        for control in controls:
            if control is self.reset_button:
                continue
            if isinstance(control, (QAbstractButton, QGroupBox)):
                value: object = control.isChecked()
            elif isinstance(control, QComboBox):
                value = control.currentData() if control.currentData() is not None else control.currentText()
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                value = control.value()
            elif isinstance(control, QLineEdit):
                value = control.text()
            else:
                continue
            self._defaults.append((control, value))

    def reset_to_defaults(self) -> None:
        for control, value in self._defaults:
            if isinstance(control, (QAbstractButton, QGroupBox)):
                control.setChecked(bool(value))
            elif isinstance(control, QComboBox):
                index = control.findData(value)
                if index < 0:
                    index = control.findText(str(value))
                if index >= 0:
                    control.setCurrentIndex(index)
            elif isinstance(control, QSpinBox):
                control.setValue(int(value))
            elif isinstance(control, QDoubleSpinBox):
                control.setValue(float(value))
            elif isinstance(control, QLineEdit):
                control.setText(str(value))


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


class NoWheelSlider(QSlider):
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


def create_min_max_widget(
    min_widget: QWidget,
    max_widget: QWidget,
    *,
    min_text: str = 'Min',
    max_text: str = 'Max',
) -> QWidget:
    """Compose two controls into a single `Min / Max` range widget row."""
    if not isinstance(min_widget, QWidget) or not isinstance(max_widget, QWidget):
        raise TypeError('min_widget and max_widget must be QWidget instances')
    range_widget = QWidget()
    row_layout = QHBoxLayout(range_widget)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(SIZE_WIDGET_SPACING)
    row_layout.addWidget(QLabel(str(min_text)))
    row_layout.addWidget(min_widget)
    row_layout.addWidget(QLabel(str(max_text)))
    row_layout.addWidget(max_widget)
    return range_widget


def create_slider(
    slider_range: tuple[int, int],
    *,
    default_value: int,
) -> QSlider:
    if len(slider_range) != 2:
        raise ValueError('slider_range must contain exactly two values')
    min_value, max_value = slider_range
    if min_value > max_value:
        raise ValueError('slider_range min value must be <= max value')
    if not (min_value <= default_value <= max_value):
        raise ValueError('default_value must be inside slider_range')
    slider = NoWheelSlider()
    slider.setOrientation(Qt.Orientation.Horizontal)
    slider.setRange(int(min_value), int(max_value))
    slider.setValue(int(default_value))
    slider.setSingleStep(1)
    slider.setPageStep(max(1, (int(max_value) - int(min_value)) // 10))
    return slider
