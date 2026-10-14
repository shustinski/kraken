"""settings_panel: sync optimizer mode controls."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
)
from neuralimage.lib.data_interfaces import (
    normalize_scheduler_name,
)
from neuralimage.view.settings_panel_widgets import (
    NoWheelComboBox,
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    create_spinbox,
)

from neuralimage.view.settings_panel_parts.module_state import (
    COMPRESSION_FACTOR_DEFAULT,
    COMPRESSION_FACTOR_RANGE,
    COMPRESSION_FACTOR_STEP,
    EDGE_CUT_RANGE,
    EDGE_CUT_STEP,
    FORM_DEFAULT_MARGINS,
    FORM_HORIZONTAL_SPACING,
    FORM_VERTICAL_SPACING,
    ONE_CYCLE_ANNEAL_STRATEGIES,
)


class SyncOptimizerModeControlsMixin:
    def _sync_optimizer_mode_controls(
        self,
        training_applicable: bool,
        recognition_applicable: bool,
        batch_related: bool,
    ) -> None:
        self._set_fields_enabled(
            (
                self.optimizer_presets_widget,
                self.learning_rate_spinbox,
                self.weight_decay_spinbox,
                self.dataloader_num_workers_spinbox,
                self.log_update_frequency_spinbox,
                self.mixed_precision_type,
                self.loss_terms_groupbox,
                self.optimizer_advanced_groupbox,
                self.loss_advanced_groupbox,
            ),
            training_applicable,
        )
        self._set_fields_enabled((self.train_batch_spinbox,), training_applicable)
        self._set_fields_enabled(
            (
                self.recognition_batch_spinbox,
                self.overlap_spinbox,
                self.recognition_jpeg_quality_spinbox,
            ),
            recognition_applicable,
        )
        self._set_fields_enabled((self.multi_gpu_mode_combo,), training_applicable)
        self._set_widgets_enabled((self.skip_uniform_labels_check_box,), training_applicable)
        self._set_widgets_enabled(
            (self.rare_patch_oversampling_check_box, self.edit_rare_regions_button),
            training_applicable,
        )
        self._set_widgets_enabled((self.torch_compile_check_box,), batch_related)
        self._sync_loss_controls()
        self._sync_rare_patch_oversampling_controls(self.rare_patch_oversampling_check_box.isChecked())
        self._sync_recognition_output_controls()

    def _sync_optional_training_mode_controls(self, training_applicable: bool) -> None:
        self._set_widgets_enabled(
            (
                self.warmup_groupbox,
                self.warmup_check_box,
                self.scheduler_groupbox,
                self.hard_mining_groupbox,
                self.hard_mining_check_box,
                self.hard_pixel_mining_check_box,
                self.early_stopping_groupbox,
                self.early_stopping_check_box,
            ),
            training_applicable,
        )
        self._sync_warmup_controls(self.warmup_check_box.isChecked())
        self._sync_scheduler_controls()
        self._sync_hard_mining_controls(self.hard_mining_check_box.isChecked())
        self._sync_early_stopping_controls(self.early_stopping_check_box.isChecked())

    def sync_business_logic_controls(self, work_mode: str | None) -> None:
        """Apply enable/disable rules for controls based on the selected work mode."""
        training_applicable, recognition_applicable, model_selector_applicable = (
            self._resolve_work_mode_applicability(work_mode)
        )
        self._set_work_mode_flags(
            training_applicable,
            recognition_applicable,
            model_selector_applicable,
        )

        batch_related = training_applicable or recognition_applicable
        self._sync_general_mode_controls(
            training_applicable,
            recognition_applicable,
            model_selector_applicable,
            batch_related,
        )
        self._sync_training_data_mode_controls(training_applicable)
        self._sync_optimizer_mode_controls(
            training_applicable,
            recognition_applicable,
            batch_related,
        )
        self._sync_optional_training_mode_controls(training_applicable)
        self._sync_patch_size_controls()
        self._sync_random_patch_size_controls()
        self._set_settings_page_visible('data', training_applicable)
        self._set_settings_page_visible('training', training_applicable)
        self._set_settings_page_visible('recognition', recognition_applicable)
        self._ensure_visible_settings_page_selected()

    @staticmethod
    def _create_form_layout(
        margins: tuple[int, int, int, int] = FORM_DEFAULT_MARGINS,
    ) -> QFormLayout:
        layout = QFormLayout()
        layout.setContentsMargins(*margins)
        layout.setHorizontalSpacing(FORM_HORIZONTAL_SPACING)
        layout.setVerticalSpacing(FORM_VERTICAL_SPACING)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        return layout

    def _init_color_type_combobox(self) -> None:
        self.color_type = NoWheelComboBox()
        self.color_type.addItem('', '')
        self.color_type.addItem('RGB', 'RGB')
        self.color_type.addItem('ЧБ', 'ЧБ')
        self.color_type.setSizePolicy(self.size_policy)

    def set_color_mode_items(self, options: list[tuple[str, str]]) -> None:
        current_value = self.get_color_mode_value()
        self.color_type.clear()
        for value, label in options:
            self.color_type.addItem(label, value)
        self.set_color_mode_value(current_value)

    def get_color_mode_value(self) -> str:
        value = self.color_type.currentData()
        if isinstance(value, str) and value:
            return value
        return self.color_type.currentText()

    def set_color_mode_value(self, value: str) -> None:
        index = self.color_type.findData(value)
        if index >= 0:
            self.color_type.setCurrentIndex(index)
            return
        self.color_type.setCurrentText(value)

    def get_scheduler_value(self) -> str:
        value = self.scheduler_type_combo.currentData()
        if isinstance(value, str) and value:
            return normalize_scheduler_name(value)
        return normalize_scheduler_name(self.scheduler_type_combo.currentText())

    def set_scheduler_value(self, value: str) -> None:
        normalized = normalize_scheduler_name(value)
        index = self.scheduler_type_combo.findData(normalized)
        if index >= 0:
            self.scheduler_type_combo.setCurrentIndex(index)
            return
        self.scheduler_type_combo.setCurrentText(normalized)

    def get_scheduler_one_cycle_anneal_strategy_value(self) -> str:
        value = self.scheduler_one_cycle_anneal_strategy_combo.currentData()
        if isinstance(value, str) and value in ONE_CYCLE_ANNEAL_STRATEGIES:
            return value
        normalized = str(self.scheduler_one_cycle_anneal_strategy_combo.currentText() or 'cos').strip().lower()
        return normalized if normalized in ONE_CYCLE_ANNEAL_STRATEGIES else 'cos'

    def set_scheduler_one_cycle_anneal_strategy_value(self, value: str) -> None:
        normalized = str(value or 'cos').strip().lower()
        if normalized not in ONE_CYCLE_ANNEAL_STRATEGIES:
            normalized = 'cos'
        index = self.scheduler_one_cycle_anneal_strategy_combo.findData(normalized)
        if index >= 0:
            self.scheduler_one_cycle_anneal_strategy_combo.setCurrentIndex(index)
            return
        self.scheduler_one_cycle_anneal_strategy_combo.setCurrentText(normalized)

    def _init_preprocess_groupbox(self) -> None:
        self.prepare_samples_groupbox = QGroupBox('')
        self.enable_crop_processing = QCheckBox('')
        self.enable_resize_processing = QCheckBox('')
        self.cut_corner_spinbox = create_spinbox(EDGE_CUT_RANGE, step=EDGE_CUT_STEP, default_value=0)
        self.compression_factor_spinbox = create_spinbox(
            COMPRESSION_FACTOR_RANGE,
            step=COMPRESSION_FACTOR_STEP,
            default_value=COMPRESSION_FACTOR_DEFAULT,
        )
        # Legacy aliases kept for tests/plugins that inspected the old target-size controls.
        self.target_size_widget = self.compression_factor_spinbox
        self.target_x_size = self.compression_factor_spinbox
        self.target_y_size = self.compression_factor_spinbox

        form = self._create_form_layout()
        self.prepare_samples_groupbox.setLayout(form)
        self.prepare_samples_form = form
        form.addRow(self.enable_crop_processing)
        form.addRow(self.enable_resize_processing)
        self._add_labeled_row(form, self.cut_corner_spinbox, 'edge_cut')
        self._add_labeled_row(form, self.compression_factor_spinbox, 'compression_factor')
        self.enable_crop_processing.toggled.connect(self._sync_preprocess_controls)
        self.enable_resize_processing.toggled.connect(self._sync_preprocess_controls)
        self._sync_preprocess_controls()
