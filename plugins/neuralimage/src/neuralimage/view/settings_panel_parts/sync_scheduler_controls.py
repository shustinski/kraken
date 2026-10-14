"""settings_panel: sync scheduler controls."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget,
    QCheckBox,
    QLineEdit,
)
from neuralimage.view.settings_panel_widgets import (
    NoWheelComboBox,
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    create_double_spinbox,
    create_spinbox,
)
from neuralimage.configuration import (
    SemUiField,
)

from neuralimage.view.settings_panel_parts.module_state import (
    LOSS_PRESET_FLOAT_TOLERANCE,
    LOSS_PRESET_WEIGHTS,
    OPTIMIZERS,
    OPTIMIZER_PRESET_FLOAT_TOLERANCE,
)


class SyncSchedulerControlsMixin:
    def _update_card_summaries(self) -> None:
        pre_mode = self.sem_segmentation_controls['pre_mode'].currentData() or 'none'
        hard_mode = self.sem_segmentation_controls['hard_mode'].currentData() or 'off'
        uncertainty_method = self.sem_segmentation_controls['uncertainty_method'].currentData() or 'off'
        uncertainty_enabled = self.sem_segmentation_controls['uncertainty_enabled'].isChecked()
        summaries = {
            'preprocessing': str(pre_mode).replace('_', ' '),
            'sampling': str(hard_mode).replace('_', ' '),
            'architecture': self.nn_model_type.currentText(),
            'training': f'{self.epochs_spinbox.value()} epochs',
            'losses': self.optimizer_type.currentText(),
            'recognition': f'batch {self.recognition_batch_spinbox.value()}',
            'inference_uncertainty': (
                str(uncertainty_method).replace('_', ' ') if uncertainty_enabled else 'off'
            ),
        }
        for key, summary in summaries.items():
            card = self._settings_cards.get(key)
            if card is not None:
                card.set_summary(summary)

    def _focus_first_configuration_error(self) -> None:
        if not self._configuration_errors:
            return
        field_key, _message = self._configuration_errors[0]
        if field_key is not None:
            self._navigate_to_setting(f'sem:{field_key}')

    def _apply_optimizer_preset(self, optimizer_name: str, learning_rate: float, weight_decay: float) -> None:
        if optimizer_name not in OPTIMIZERS:
            return
        try:
            learning_rate_value = float(learning_rate)
            weight_decay_value = float(weight_decay)
        except (TypeError, ValueError):
            return
        self.optimizer_type.setCurrentText(optimizer_name)
        self.learning_rate_spinbox.setValue(learning_rate_value)
        self.weight_decay_spinbox.setValue(weight_decay_value)
        self._sync_active_optimizer_preset()
        self.optimizer_settings_changed.emit()

    def _sync_active_optimizer_preset(self) -> None:
        current_optimizer = self.optimizer_type.currentText()
        current_learning_rate = float(self.learning_rate_spinbox.value())
        current_weight_decay = float(self.weight_decay_spinbox.value())
        for btn, (_title, optimizer_name, learning_rate, weight_decay) in zip(
            self.optimizer_preset_buttons,
            self._visible_optimizer_presets,
        ):
            is_active = (
                current_optimizer == optimizer_name
                and abs(current_learning_rate - learning_rate) < OPTIMIZER_PRESET_FLOAT_TOLERANCE
                and abs(current_weight_decay - weight_decay) < OPTIMIZER_PRESET_FLOAT_TOLERANCE
            )
            btn.setChecked(is_active)

    @staticmethod
    def _loss_weights_match_preset(
        current_weights: dict[str, float],
        preset_weights: dict[str, float],
    ) -> bool:
        if set(current_weights) != set(preset_weights):
            return False
        for loss_name, preset_value in preset_weights.items():
            if abs(float(current_weights.get(loss_name, 0.0)) - float(preset_value)) > LOSS_PRESET_FLOAT_TOLERANCE:
                return False
        return True

    def _sync_active_loss_preset(self) -> None:
        current_weights = self.get_loss_term_weights()
        for preset_name, button in self.loss_preset_buttons.items():
            preset_weights = LOSS_PRESET_WEIGHTS.get(preset_name, {})
            button.setChecked(self._loss_weights_match_preset(current_weights, preset_weights))

    def _sync_validation_controls(self, enabled: bool) -> None:
        validation_enabled = self._training_controls_applicable and bool(enabled)
        external_mode = self.get_validation_source_value() == 'external'
        self._set_field_visible(self.validation_mode_combo, validation_enabled)
        self._set_field_visible(self.validation_spinbox, validation_enabled and not external_mode)
        self._set_field_visible(self.validation_image_path_label, validation_enabled and external_mode)
        self._set_field_visible(self.validation_label_path_label, validation_enabled and external_mode)
        self.save_validation_binary_images_check_box.setVisible(validation_enabled)
        self._set_field_enabled(self.validation_mode_combo, validation_enabled)
        self._set_field_enabled(
            self.validation_spinbox,
            validation_enabled and not external_mode,
        )
        self._set_field_enabled(
            self.validation_image_path_label,
            validation_enabled and external_mode,
        )
        self._set_field_enabled(
            self.validation_label_path_label,
            validation_enabled and external_mode,
        )
        self.save_validation_binary_images_check_box.setEnabled(validation_enabled)
        self._sync_validation_path_labels()
        if hasattr(self, 'early_stopping_check_box'):
            self._sync_early_stopping_controls(self.early_stopping_check_box.isChecked())

    def _sync_validation_path_labels(self) -> None:
        texts = self._texts if isinstance(self._texts, dict) else {}
        image_placeholder = str(texts.get('validation_image_path_placeholder', 'Click to choose validation image folder'))
        label_placeholder = str(texts.get('validation_label_path_placeholder', 'Click to choose validation label folder'))
        self.validation_image_path_label.setText(self._validation_image_path_value or image_placeholder)
        self.validation_label_path_label.setText(self._validation_label_path_value or label_placeholder)

    def _sync_warmup_controls(self, enabled: bool) -> None:
        control_enabled = self._training_controls_applicable and bool(enabled)
        self._set_field_enabled(self.warmup_epochs_spinbox, control_enabled)
        self._set_field_enabled(self.warmup_start_factor_spinbox, control_enabled)

    def _sync_scheduler_controls(self, _index: int | None = None) -> None:
        scheduler_name = self.get_scheduler_value()
        scheduler_enabled = self._training_controls_applicable
        plateau_fields = (
            self.scheduler_plateau_factor_spinbox,
            self.scheduler_plateau_patience_spinbox,
            self.scheduler_plateau_threshold_spinbox,
            self.scheduler_plateau_min_lr_spinbox,
            self.scheduler_plateau_cooldown_spinbox,
        )
        cosine_fields = (
            self.scheduler_cosine_t_max_spinbox,
            self.scheduler_cosine_eta_min_spinbox,
        )
        one_cycle_fields = (
            self.scheduler_one_cycle_max_lr_spinbox,
            self.scheduler_one_cycle_pct_start_spinbox,
            self.scheduler_one_cycle_anneal_strategy_combo,
            self.scheduler_one_cycle_div_factor_spinbox,
            self.scheduler_one_cycle_final_div_factor_spinbox,
            self.scheduler_one_cycle_three_phase_check_box,
        )
        step_lr_fields = (
            self.scheduler_step_lr_step_size_spinbox,
            self.scheduler_step_lr_gamma_spinbox,
        )

        self._set_field_enabled(self.scheduler_type_combo, scheduler_enabled)

        visibility_map = {
            'reduce_on_plateau': plateau_fields,
            'cosine_annealing': cosine_fields,
            'one_cycle': one_cycle_fields,
            'step_lr': step_lr_fields,
        }
        all_fields = plateau_fields + cosine_fields + one_cycle_fields + step_lr_fields
        visible_fields = set(visibility_map.get(scheduler_name, ()))

        for field in all_fields:
            is_visible = field in visible_fields
            self._set_field_visible(field, is_visible)
            self._set_field_enabled(field, scheduler_enabled and is_visible)

    def _create_sem_segmentation_control(self, field: SemUiField) -> QWidget:
        if field.kind == 'bool':
            control: QWidget = QCheckBox('')
            control.toggled.connect(self._on_sem_segmentation_control_changed)
        elif field.kind == 'int':
            control = create_spinbox(
                (int(field.minimum), int(field.maximum)),
                step=int(field.step or 1),
                default_value=int(field.default),
            )
            control.valueChanged.connect(self._on_sem_segmentation_control_changed)
        elif field.kind == 'float':
            control = create_double_spinbox(
                (float(field.minimum), float(field.maximum)),
                step=float(field.step or 0.01),
                default_value=float(field.default),
                decimals=int(field.decimals),
            )
            control.valueChanged.connect(self._on_sem_segmentation_control_changed)
        elif field.kind == 'choice':
            combo = NoWheelComboBox()
            for value, label in field.choices:
                combo.addItem(label, value)
            combo.currentIndexChanged.connect(self._on_sem_segmentation_control_changed)
            control = combo
        else:
            line_edit = QLineEdit()
            line_edit.textChanged.connect(self._on_sem_segmentation_control_changed)
            control = line_edit
        control.setProperty('semFieldKey', field.key)
        return control
