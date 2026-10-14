"""settings_panel: sync patch size controls."""
from __future__ import annotations
from typing import Iterable
from collections.abc import Mapping
from PyQt6.QtCore import QSize
from neuralimage.view.settings_panel_widgets import (
    NoWheelComboBox,
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )


class SyncPatchSizeControlsMixin:
    def _sync_patch_size_controls(self, _index: int | None = None) -> None:
        if getattr(self, '_patch_size_sync_guard', False):
            return
        self._patch_size_sync_guard = True
        try:
            random_enabled = (
                self._training_controls_applicable
                and bool(self.random_patch_size_check_box.isChecked())
            )
            if random_enabled and self.sync_patch_sizes_check_box.isChecked():
                self.sync_patch_sizes_check_box.setChecked(False)
            self.sync_patch_sizes_check_box.setEnabled(
                self._recognition_controls_applicable and not random_enabled
            )
            patch_sync = bool(self.sync_patch_sizes_check_box.isChecked()) and not random_enabled
            if patch_sync:
                train_x = int(self.train_patch_x_size.value())
                train_y = int(self.train_patch_y_size.value())
                if self.recognition_patch_size_widget.target_size() != QSize(train_x, train_y):
                    self.recognition_patch_size_widget.set_target_size(
                        train_x,
                        train_y,
                    )
            self._set_field_enabled(
                self.recognition_patch_size_widget,
                self._recognition_controls_applicable and not patch_sync,
            )
            self._set_field_enabled(
                self.recognition_tab_patch_size_widget,
                self._recognition_controls_applicable,
            )
            self._sync_patch_size_layout(random_enabled=random_enabled, patch_sync=patch_sync)
        finally:
            self._patch_size_sync_guard = False

    def _sync_recognition_patch_copy_from_primary(self, width: int, height: int) -> None:
        if self._recognition_patch_copy_sync_guard:
            return
        if self.recognition_tab_patch_size_widget.target_size() == QSize(width, height):
            return
        self._recognition_patch_copy_sync_guard = True
        try:
            self.recognition_tab_patch_size_widget.set_target_size(width, height)
        finally:
            self._recognition_patch_copy_sync_guard = False

    def _sync_recognition_patch_primary_from_copy(self, width: int, height: int) -> None:
        if self._recognition_patch_copy_sync_guard:
            return
        self._recognition_patch_copy_sync_guard = True
        try:
            random_enabled = (
                self._training_controls_applicable
                and bool(self.random_patch_size_check_box.isChecked())
            )
            patch_sync = bool(self.sync_patch_sizes_check_box.isChecked()) and not random_enabled
            if patch_sync:
                self.train_patch_size_widget.set_target_size(width, height)
            self.recognition_patch_size_widget.set_target_size(width, height)
        finally:
            self._recognition_patch_copy_sync_guard = False
        if patch_sync:
            self.sample_size_changed.emit()
        self.optimizer_settings_changed.emit()

    @staticmethod
    def _normalize_model_group(values: Iterable[str] | str | None) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        try:
            return [str(name) for name in values]
        except TypeError:
            return []

    def _populate_model_combo(
        self,
        combo: NoWheelComboBox,
        models: list[str],
        *,
        select_first_model: bool = False,
    ) -> None:
        combo.clear()
        combo.addItem('-', '')
        for model_name in models:
            combo.addItem(model_name, model_name)
        if select_first_model and len(models) > 0:
            combo.setCurrentIndex(1)
        else:
            combo.setCurrentIndex(0)

    def _clear_model_combos(self, *, keep: NoWheelComboBox | None = None) -> None:
        for combo in (self.nn_model_type, self.deprecated_model_type, self.experimental_model_type):
            if combo is keep:
                continue
            if combo.count() > 0:
                combo.setCurrentIndex(0)

    @staticmethod
    def _combo_selected_model(combo: NoWheelComboBox, *, allow_placeholder: bool = False) -> str:
        current_data = combo.currentData()
        if current_data is not None:
            resolved = str(current_data).strip()
            if resolved or allow_placeholder:
                return resolved
        if not allow_placeholder and combo.currentIndex() <= 0:
            return ''
        return str(combo.currentText() or '').strip()

    def _sync_model_selection(self, source: NoWheelComboBox) -> None:
        if self._model_combo_guard:
            return
        self._model_combo_guard = True
        try:
            if source is self.nn_model_type:
                if self._combo_selected_model(source):
                    self._clear_model_combos(keep=source)
            elif source in {self.deprecated_model_type, self.experimental_model_type}:
                selected_model = self._combo_selected_model(source)
                if selected_model:
                    self._clear_model_combos(keep=source)
        finally:
            self._model_combo_guard = False

    def get_selected_model(self) -> str:
        deprecated_model = self._combo_selected_model(self.deprecated_model_type)
        if deprecated_model:
            return deprecated_model
        experimental_model = self._combo_selected_model(self.experimental_model_type)
        if experimental_model:
            return experimental_model
        return self._combo_selected_model(self.nn_model_type, allow_placeholder=True)

    def model_type_init(self, models: Iterable[str] | Mapping[str, Iterable[str]] | None) -> None:
        """Populate stable, deprecated, and experimental model selectors."""
        stable_models: list[str]
        deprecated_models: list[str]
        experimental_models: list[str]
        if isinstance(models, Mapping):
            stable_models = self._normalize_model_group(
                models.get('stable') or models.get('ModelType.stable') or models.get(None)
            )
            deprecated_models = self._normalize_model_group(models.get('deprecated'))
            experimental_models = self._normalize_model_group(models.get('experimental'))
            for key, values in models.items():
                key_name = str(getattr(key, 'value', key))
                if key_name == 'stable':
                    stable_models = self._normalize_model_group(values)
                elif key_name == 'deprecated':
                    deprecated_models = self._normalize_model_group(values)
                elif key_name == 'experimental':
                    experimental_models = self._normalize_model_group(values)
        else:
            stable_models = self._normalize_model_group(models)
            deprecated_models = []
            experimental_models = []

        self.models = list(stable_models)
        self.deprecated_models = list(deprecated_models)
        self.experimental_models = list(experimental_models)
        self._populate_model_combo(self.nn_model_type, self.models, select_first_model=bool(self.models))
        self._populate_model_combo(self.deprecated_model_type, self.deprecated_models)
        self._populate_model_combo(self.experimental_model_type, self.experimental_models)
