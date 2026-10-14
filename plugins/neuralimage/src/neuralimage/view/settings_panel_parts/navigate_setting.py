"""settings_panel: navigate setting."""
from __future__ import annotations
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QGroupBox,
)
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )


class NavigateSettingMixin:
    def _activate_first_search_result(self) -> None:
        query = self.settings_search.text().strip().casefold()
        if not query:
            return
        for key, item in self._settings_registry.items():
            if query in item['search']:
                self._navigate_to_setting(key)
                return

    def _activate_search_result(self, display: str) -> None:
        key = self._search_display_to_key.get(str(display))
        if key is not None:
            self._navigate_to_setting(key)

    def _navigate_to_setting(self, key: str) -> None:
        item = self._settings_registry.get(key)
        if item is None:
            return
        card = self._settings_cards[item['card']]
        widget = item['widget']
        expert_only = bool(card.property('expertOnly'))
        ancestor: QWidget | None = widget
        while ancestor is not None and ancestor is not card:
            expert_only = expert_only or bool(ancestor.property('expertOnly'))
            ancestor = ancestor.parentWidget()
        if expert_only and not self.expert_mode_check_box.isChecked():
            self.expert_mode_check_box.setChecked(True)
        self.show_settings_page(item['page'])
        card.set_expanded(True)
        ancestor = widget.parentWidget()
        while ancestor is not None and ancestor is not card:
            if isinstance(ancestor, QGroupBox) and ancestor.isCheckable():
                ancestor.setChecked(True)
            ancestor = ancestor.parentWidget()
        self._scroll_area.ensureWidgetVisible(widget, 24, 48)
        previous_style = widget.styleSheet()
        widget.setStyleSheet(f'{previous_style}; border: 1px solid #42a5f5;')
        QTimer.singleShot(1400, lambda w=widget, style=previous_style: w.setStyleSheet(style))
        widget.setFocus(Qt.FocusReason.ShortcutFocusReason)

    @staticmethod
    def _configuration_error_field(message: str) -> str | None:
        lowered = message.casefold()
        mappings = (
            ('attention_dim', 'context_dim'),
            ('attention dimensions', 'context_dim'),
            ('hard-mining', 'hard_geometry_weight'),
            ('sdf target', 'target_sdf'),
            ('heads and targets', 'target_boundary'),
            ('confidence loss', 'confidence_training_loss_weight'),
            ('mc dropout', 'uncertainty_samples'),
            ('boundary kernel', 'target_boundary_kernel'),
            ('experiment seed', 'experiment_seed_1'),
        )
        for fragment, key in mappings:
            if fragment in lowered:
                return key
        return None
