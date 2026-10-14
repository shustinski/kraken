"""matrix_view: gradient preset selector widget."""
from __future__ import annotations
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from neuralimage.Validation_gradient_widget_lite.ui.i18n import Translator
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    CARD_CONTENT_SPACING,
    DEFAULT_GRADIENT_NAME,
    GRADIENT_LABELS,
    GRADIENT_PRESETS,
    PANEL_TEXT,
    SUBDUED_TEXT_COLOR,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.gradient_preview_bar import _GradientPreviewBar

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
    """Select the active gradient preset used to render matrix scores."""

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
