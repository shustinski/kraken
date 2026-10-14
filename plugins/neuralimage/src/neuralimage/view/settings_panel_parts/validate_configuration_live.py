"""settings_panel: validate configuration live."""
from __future__ import annotations
from neuralimage.lib.ui_texts import get_ui_language
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )
from neuralimage.configuration import (
    build_sem_segmentation_config,
)


class ValidateConfigurationLiveMixin:
    def _validate_configuration_live(self) -> None:
        for key, label in self.sem_segmentation_error_labels.items():
            label.clear()
            label.setVisible(False)
            self.sem_segmentation_controls[key].setProperty('configurationError', False)
            base_tooltip = self.sem_segmentation_controls[key].property('baseToolTip')
            if isinstance(base_tooltip, str):
                self.sem_segmentation_controls[key].setToolTip(base_tooltip)
        for card in self._settings_cards.values():
            card.set_error_count(0)

        errors: list[tuple[str | None, str]] = []
        try:
            build_sem_segmentation_config(self.get_sem_segmentation_config())
        except (TypeError, ValueError) as error:
            message = str(error)
            errors.append((self._configuration_error_field(message), message))

        card_errors: dict[str, int] = {}
        for field_key, message in errors:
            if field_key is None:
                continue
            error_label = self.sem_segmentation_error_labels.get(field_key)
            control = self.sem_segmentation_controls.get(field_key)
            if error_label is not None:
                error_label.setText('!')
                error_label.setToolTip(message)
                error_label.setVisible(True)
            if control is not None:
                control.setProperty('configurationError', True)
                control.setToolTip(f'{control.toolTip()}\n\n{message}'.strip())
                card_key = self._find_card_for_widget(control)
                if card_key is not None:
                    card_errors[card_key] = card_errors.get(card_key, 0) + 1
        for card_key, count in card_errors.items():
            self._settings_cards[card_key].set_error_count(count)

        self._configuration_errors = errors
        self._update_card_summaries()
        russian = get_ui_language() == 'ru'
        if errors:
            self.configuration_status_button.setText(
                f'Ошибок: {len(errors)}' if russian else f'Errors: {len(errors)}'
            )
            self.configuration_status_button.setToolTip(errors[0][1])
        else:
            self.configuration_status_button.setText('OK')
            self.configuration_status_button.setToolTip(
                'Конфигурация корректна' if russian else 'Configuration is valid'
            )
        self.configuration_validity_changed.emit(not errors, tuple(message for _key, message in errors))
