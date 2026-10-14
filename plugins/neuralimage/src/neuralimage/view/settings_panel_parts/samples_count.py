"""settings_panel: samples count."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget,
)
from neuralimage.view.settings_panel_bindings import connect_settings_panel_signals
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )

from neuralimage.view.settings_panel_parts.settings_panel_qsettings import _settings_panel_qsettings


class SamplesCountMixin:
    def set_samples_count(self, total_samples: int) -> None:
        try:
            self._sample_count_value = int(total_samples)
        except (TypeError, ValueError):
            self._sample_count_value = 0
        self._sample_count_pending = False
        template = str(self._texts.get('samples_count_template', self._texts.get('samples_count', 'Кадров в выборке: {count}')))
        if '{count}' in template:
            text = template.format(count=self._sample_count_value)
        else:
            text = template
        self.samples_number.setText(text)

    def set_samples_count_loading(self) -> None:
        self._sample_count_pending = True
        text = str(self._texts.get('samples_count_loading', 'Идет расчет...'))
        self.samples_number.setText(text)

    def connect_internal_signals(self) -> None:
        connect_settings_panel_signals(self)

    def show_settings_page(self, page_key: str) -> None:
        normalized_key = str(page_key or '').strip().lower()
        if normalized_key == 'base':
            normalized_key = 'data'
        index = self._page_indexes.get(normalized_key)
        if index is None:
            return
        self.settings_tabs.setCurrentIndex(index)

    def _set_settings_page_visible(self, page_key: str, visible: bool) -> None:
        index = self._page_indexes.get(page_key)
        if index is None:
            return
        if hasattr(self.settings_tabs, 'setTabVisible'):
            self.settings_tabs.setTabVisible(index, visible)

    def _ensure_visible_settings_page_selected(self) -> None:
        current_index = self.settings_tabs.currentIndex()
        if hasattr(self.settings_tabs, 'isTabVisible') and self.settings_tabs.isTabVisible(current_index):
            return
        for page_key in ('data', 'training', 'recognition'):
            index = self._page_indexes.get(page_key)
            if index is None:
                continue
            if hasattr(self.settings_tabs, 'isTabVisible') and self.settings_tabs.isTabVisible(index):
                self.settings_tabs.setCurrentIndex(index)
                return

    def _sync_expert_mode(self, enabled: bool) -> None:
        visible = bool(enabled)
        for widget in self._expert_widgets:
            widget.setVisible(visible)
        settings = _settings_panel_qsettings()
        settings.setValue('expert_mode', visible)
        settings.sync()

    def _card_page(self, card_key: str) -> str:
        if card_key in {'data_source', 'preprocessing', 'augmentation', 'sampling', 'validation_data'}:
            return 'data'
        if card_key in {'recognition', 'inference_uncertainty', 'active_learning'}:
            return 'recognition'
        return 'training'

    def _find_card_for_widget(self, widget: QWidget) -> str | None:
        for key, card in self._settings_cards.items():
            if card is widget or card.isAncestorOf(widget):
                return key
        return None
