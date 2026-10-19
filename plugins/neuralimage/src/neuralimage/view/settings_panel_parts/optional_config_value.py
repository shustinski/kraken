"""settings_panel: optional config value."""
from __future__ import annotations
from typing import Any
from collections.abc import Mapping
from copy import deepcopy
from PyQt6.QtWidgets import (
    QMessageBox,
)
from neuralimage.lib.ui_texts import get_ui_language, get_ui_section, set_ui_language as set_global_ui_language
from neuralimage.view.settings_panel_i18n import apply_settings_panel_texts
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )


class OptionalConfigValueMixin:
    def _apply_localized_texts(self) -> None:
        apply_settings_panel_texts(self)
        if hasattr(self, 'training_augmentation_editor'):
            self.training_augmentation_editor.set_language(get_ui_language())
            russian = get_ui_language().startswith('ru')
            self.training_augmentation_editor.setTitle(
                'Аугментации обучения' if russian else 'Training augmentations'
            )
            self.sem_normalization_editor.setTitle(
                'Нормализация SEM' if russian else 'SEM normalization'
            )
        if hasattr(self, 'patch_sampling_mode_groupbox'):
            russian = get_ui_language().startswith('ru')
            self.patch_sampling_mode_groupbox.setTitle('Режим нарезки' if russian else 'Patch layout')
            self.grid_sampling_radio.setText('По сетке' if russian else 'Grid')
            self.random_sampling_radio.setText('Случайная' if russian else 'Random')

    def _request_reset_defaults(self) -> None:
        language = get_ui_language()
        title = 'Сбросить настройки' if language.startswith('ru') else 'Reset settings'
        message = (
            'Вернуть все настройки к значениям по умолчанию?'
            if language.startswith('ru')
            else 'Restore all settings to their default values?'
        )
        answer = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.reset_defaults_requested.emit()

    def set_ui_language(self, language: str) -> None:
        active_language = set_global_ui_language(language)
        self._texts = get_ui_section('settings_panel')
        self._apply_localized_texts()
        self._sync_validation_path_labels()
        if self._sample_count_pending:
            self.set_samples_count_loading()
        else:
            self.set_samples_count(self._sample_count_value)
        self._validate_configuration_live()
        self.ui_language_changed.emit(active_language)

    def get_validation_source_value(self) -> str:
        current_data = self.validation_mode_combo.currentData()
        if current_data is None:
            return 'split'
        return str(current_data)

    def set_validation_source_value(self, value: str) -> None:
        normalized = str(value or 'split').strip().lower() or 'split'
        index = self.validation_mode_combo.findData(normalized)
        if index < 0:
            index = 0
        self.validation_mode_combo.setCurrentIndex(index)

    def validation_image_path(self) -> str:
        return str(self._validation_image_path_value)

    def set_validation_image_path(self, path: str) -> None:
        self._validation_image_path_value = str(path or '').strip()
        self._sync_validation_path_labels()

    def validation_label_path(self) -> str:
        return str(self._validation_label_path_value)

    def set_validation_label_path(self, path: str) -> None:
        self._validation_label_path_value = str(path or '').strip()
        self._sync_validation_path_labels()

    @staticmethod
    def _clone_plain_payload(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): OptionalConfigValueMixin._clone_plain_payload(item) for key, item in value.items()}
        if hasattr(value, '__dict__'):
            return {
                str(key): OptionalConfigValueMixin._clone_plain_payload(item)
                for key, item in vars(value).items()
                if not str(key).startswith('_')
            }
        if isinstance(value, list):
            return [OptionalConfigValueMixin._clone_plain_payload(item) for item in value]
        if isinstance(value, tuple):
            return [OptionalConfigValueMixin._clone_plain_payload(item) for item in value]
        return deepcopy(value)

    @staticmethod
    def _config_value_matches_default(value: Any, default: Any) -> bool:
        if isinstance(value, bool) or isinstance(default, bool):
            return bool(value) is bool(default)
        if isinstance(value, (int, float)) or isinstance(default, (int, float)):
            try:
                return abs(float(value) - float(default)) <= 1e-9
            except (TypeError, ValueError):
                return False
        return value == default

    @staticmethod
    def _set_optional_config_value(
        payload: dict[str, Any],
        path: tuple[str, ...],
        value: Any,
        default: Any,
    ) -> None:
        parents: list[tuple[dict[str, Any], str]] = []
        node = payload
        for key in path[:-1]:
            nested = node.get(key)
            if not isinstance(nested, dict):
                nested = {}
                node[key] = nested
            parents.append((node, key))
            node = nested
        leaf_key = path[-1]
        if OptionalConfigValueMixin._config_value_matches_default(value, default):
            node.pop(leaf_key, None)
        else:
            node[leaf_key] = value
        for parent, key in reversed(parents):
            nested = parent.get(key)
            if isinstance(nested, dict) and not nested:
                parent.pop(key, None)
            else:
                break
