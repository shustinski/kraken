"""settings_panel: rebuild settings registry."""
from __future__ import annotations
from typing import Any
from PyQt6.QtWidgets import (
    QWidget,
)
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )
from neuralimage.configuration import (
    SEM_UI_FIELDS,
    sem_ui_field_label,
)


class RebuildSettingsRegistryMixin:
    def _rebuild_settings_registry(self) -> None:
        registry: dict[str, dict[str, Any]] = {}

        def _register(key: str, label: str, widget: QWidget, aliases: str = '') -> None:
            card_key = self._find_card_for_widget(widget)
            if card_key is None:
                return
            tooltip = widget.toolTip()
            registry[key] = {
                'label': str(label).strip() or key,
                'widget': widget,
                'card': card_key,
                'page': self._card_page(card_key),
                'search': f'{label} {tooltip} {aliases} {key} {card_key}'.casefold(),
            }

        for key, widget in self._desc_fields.items():
            label_widget = self._desc_labels.get(key)
            _register(key, label_widget.text() if label_widget is not None else key, widget)
        for field in SEM_UI_FIELDS:
            widget = self.sem_segmentation_controls.get(field.key)
            label = self.sem_segmentation_field_labels.get(field.key)
            if widget is not None:
                _register(
                    f'sem:{field.key}',
                    label.text() if label is not None else field.label_en,
                    widget,
                    aliases=f'{field.label_en} {sem_ui_field_label(field, "ru")}',
                )
        card_aliases = {
            'data_source': 'данные датасет патчи выборка dataset patches',
            'preprocessing': 'нормализация препроцессинг SEM normalization preprocessing',
            'augmentation': 'аугментация augmentation',
            'sampling': 'выбор патчей сложные примеры sampling hard mining',
            'validation_data': 'валидационная выборка validation data',
            'architecture': 'архитектура модель context architecture model',
            'training': 'обучение эпохи оптимизатор training epochs optimizer',
            'supervision': 'цели геометрия топология supervision targets geometry topology',
            'losses': 'функции потерь losses',
            'schedule': 'расписание warmup scheduler',
            'confidence_training': 'обучение уверенности confidence head training',
            'validation_metrics': 'метрики эксперимент metrics experiment',
            'runtime': 'ресурсы видеокарта GPU runtime',
            'recognition': 'распознавание маска inference mask',
            'inference_uncertainty': 'неопределенность уверенность uncertainty confidence',
            'active_learning': 'активное обучение разметка NeedsAnnotation active learning',
        }
        for key, card in self._settings_cards.items():
            _register(
                f'card:{key}',
                card.title(),
                card,
                aliases=f'{key.replace("_", " ")} {card_aliases.get(key, "")}',
            )
        self._settings_registry = registry
        self._search_display_to_key = {
            f"{item['label']} — {self._settings_cards[item['card']].title()}": key
            for key, item in registry.items()
        }
        self.settings_search_model.setStringList(sorted(self._search_display_to_key))
