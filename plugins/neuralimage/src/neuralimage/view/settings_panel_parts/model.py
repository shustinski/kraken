"""settings_panel: model."""
from __future__ import annotations
from typing import Any
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )


class ModelMixin:
    def set_model(self, model: str) -> None:
        """Select the current model across stable, deprecated, and experimental selectors."""
        if not isinstance(model, str):
            return
        if model in self.models:
            self._model_combo_guard = True
            try:
                self.nn_model_type.setCurrentText(model)
                self._clear_model_combos(keep=self.nn_model_type)
            finally:
                self._model_combo_guard = False
            return
        if model in self.deprecated_models:
            self._model_combo_guard = True
            try:
                self.deprecated_model_type.setCurrentText(model)
                self._clear_model_combos(keep=self.deprecated_model_type)
            finally:
                self._model_combo_guard = False
            return
        if model in self.experimental_models:
            self._model_combo_guard = True
            try:
                self.experimental_model_type.setCurrentText(model)
                self._clear_model_combos(keep=self.experimental_model_type)
            finally:
                self._model_combo_guard = False
            return
        if model and model not in self.models:
            self.models.append(model)
            self.nn_model_type.addItem(model, model)
        self._model_combo_guard = True
        try:
            self.nn_model_type.setCurrentText(model)
            self._clear_model_combos(keep=self.nn_model_type)
        finally:
            self._model_combo_guard = False

    def restore_cut_mode(self, mode: Any) -> None:
        """Compatibility hook: persisted disk mode is migrated to online generation."""
        del mode

    def apply_style(self, style: str) -> None:
        """Apply stylesheet to the settings panel."""
        if not isinstance(style, str):
            return
        self.setStyleSheet(style)
