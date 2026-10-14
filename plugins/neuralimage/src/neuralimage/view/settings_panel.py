import os
from pathlib import Path
from typing import Any, Iterable
from collections.abc import Mapping
from copy import deepcopy
from PyQt6.QtCore import QSettings, QStringListModel, QTimer, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget,
    QDockWidget,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QButtonGroup,
    QLabel,
    QPushButton,
    QApplication,
    QScrollArea,
    QSizePolicy,
    QGroupBox,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QCompleter,
    QMessageBox,
    QToolButton,
    QFrame,
    QRadioButton,
)
from neuralimage.UI import ClickableLabel
from neuralimage.lib.data_interfaces import (
    IC_TOPOLOGY_FAMILIES,
    PCB_TOPOLOGY_FAMILIES,
    RANDOM_ARTIFACT_TYPES,
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
    build_tech_augmentation_config,
    normalize_scheduler_name,
)
from neuralimage.lib.loss_config import (
    DEFAULT_LOSS_TERM_WEIGHTS,
    LOSS_SELECTION_NAMES,
    LOSS_TERM_DISPLAY_NAMES,
    MAX_LOSS_TERM_WEIGHT_SUM,
    format_loss_formula_html,
    loss_term_weight_sum,
    sanitize_loss_term_weights,
)
from neuralimage.lib.ui_texts import get_ui_language, get_ui_section, set_ui_language as set_global_ui_language
from neuralimage.view.axis_resize_widget import AxisResizeWidget
from neuralimage.view.settings_panel_bindings import connect_settings_panel_signals
from neuralimage.view.settings_panel_i18n import apply_settings_panel_texts
from neuralimage.view.settings_panel_policy import (
    resolve_work_mode_applicability as resolve_work_mode_applicability_policy,
)
from neuralimage.view.settings_panel_widgets import (
    NoWheelComboBox,
    SettingsCard,
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    create_double_spinbox,
    create_min_max_widget,
    create_slider,
    create_spinbox,
)
from neuralimage.view.compact_settings_row import add_compact_row
from neuralimage.view.sem_compact_section_editor import CompactSemSectionEditor
from neuralimage.view.training_transform_editors import (
    AugmentationBlock,
    AugmentationRow,
    SemNormalizationEditor,
    TrainingAugmentationEditor,
)
from neuralimage.configuration import (
    SEM_UI_FIELDS,
    SEM_UI_SECTIONS,
    SemUiField,
    build_sem_segmentation_config,
    fields_for_section,
    sem_config_from_form_values,
    sem_config_to_form_values,
    sem_ui_field_help,
    sem_ui_field_label,
    sem_ui_section_help,
)

from .settings_panel_parts.module_state import (
    FIELD_DESCRIPTION_ROW_SPACING,
    FORM_DEFAULT_MARGINS,
    FORM_HORIZONTAL_SPACING,
    FORM_VERTICAL_SPACING,
    CONTENT_LAYOUT_MARGINS,
    CONTENT_LAYOUT_SPACING,
    OPTIMIZER_PRESET_FLOAT_TOLERANCE,
    LOSS_PRESET_FLOAT_TOLERANCE,
    EDGE_CUT_RANGE,
    EDGE_CUT_STEP,
    COMPRESSION_FACTOR_RANGE,
    COMPRESSION_FACTOR_STEP,
    COMPRESSION_FACTOR_DEFAULT,
    EPOCHS_RANGE,
    EPOCHS_DEFAULT,
    SHIFT_RANGE_MIN,
    SHIFT_RANGE_MAX,
    SAMPLE_SIZE_MIN,
    SAMPLE_SIZE_MAX,
    VALIDATION_MIN,
    VALIDATION_MAX,
    MIN_BATCH,
    MAX_BATCH,
    MIN_DATALOADER_WORKERS,
    MAX_DATALOADER_WORKERS,
    MIN_OVERLAP,
    MAX_OVERLAP,
    MIN_JPEG_QUALITY,
    MAX_JPEG_QUALITY,
    MIN_LOG_UPDATE_FREQUENCY,
    MAX_LOG_UPDATE_FREQUENCY,
    MIN_LEARNING_RATE,
    MAX_LEARNING_RATE,
    MIN_WEIGHT_DECAY,
    MAX_WEIGHT_DECAY,
    MIN_EARLY_STOPPING_PATIENCE,
    MAX_EARLY_STOPPING_PATIENCE,
    MIN_EARLY_STOPPING_MIN_DELTA,
    MAX_EARLY_STOPPING_MIN_DELTA,
    MIN_WARMUP_EPOCHS,
    MAX_WARMUP_EPOCHS,
    MIN_WARMUP_START_FACTOR,
    MAX_WARMUP_START_FACTOR,
    MIN_SCHEDULER_FACTOR,
    MAX_SCHEDULER_FACTOR,
    MIN_SCHEDULER_PATIENCE,
    MAX_SCHEDULER_PATIENCE,
    MIN_SCHEDULER_THRESHOLD,
    MAX_SCHEDULER_THRESHOLD,
    MIN_SCHEDULER_MIN_LR,
    MAX_SCHEDULER_MIN_LR,
    MIN_SCHEDULER_COOLDOWN,
    MAX_SCHEDULER_COOLDOWN,
    MIN_SCHEDULER_T_MAX,
    MAX_SCHEDULER_T_MAX,
    MIN_SCHEDULER_STEP_SIZE,
    MAX_SCHEDULER_STEP_SIZE,
    MIN_SCHEDULER_DIV_FACTOR,
    MAX_SCHEDULER_DIV_FACTOR,
    MIN_SCHEDULER_FINAL_DIV_FACTOR,
    MAX_SCHEDULER_FINAL_DIV_FACTOR,
    MIN_HARD_MINING_STRENGTH,
    MAX_HARD_MINING_STRENGTH,
    MIN_HARD_MINING_EMA_ALPHA,
    MAX_HARD_MINING_EMA_ALPHA,
    MIN_HARD_PIXEL_KEEP_RATIO,
    MAX_HARD_PIXEL_KEEP_RATIO,
    MIN_RARE_PATCH_OVERSAMPLING_FACTOR,
    MAX_RARE_PATCH_OVERSAMPLING_FACTOR,
    MIN_AUG_STRENGTH,
    MAX_AUG_STRENGTH,
    MIN_AUG_NOISE_SIGMA,
    MAX_AUG_NOISE_SIGMA,
    MIN_AUG_BLUR_RADIUS,
    MAX_AUG_BLUR_RADIUS,
    MIN_CROPS_PER_IMAGE,
    MAX_CROPS_PER_IMAGE,
    MIN_CUTOUT_HOLES,
    MAX_CUTOUT_HOLES,
    MIN_RANDOM_ARTIFACTS_COUNT,
    MAX_RANDOM_ARTIFACTS_COUNT,
    MIN_AUGMENTATION_PROBABILITY,
    MAX_AUGMENTATION_PROBABILITY,
    MIN_AUGMENTATION_MULTIPLIER,
    MAX_AUGMENTATION_MULTIPLIER,
    MIN_SYNTHETIC_DATASET_FACTOR,
    MAX_SYNTHETIC_DATASET_FACTOR,
    MIN_SYNTHETIC_IMAGE_SIZE,
    MAX_SYNTHETIC_IMAGE_SIZE,
    MIN_SYNTHETIC_TRACE_COUNT,
    MAX_SYNTHETIC_TRACE_COUNT,
    MIN_SYNTHETIC_SEGMENT_COUNT,
    MAX_SYNTHETIC_SEGMENT_COUNT,
    MIN_SYNTHETIC_TRACE_HALF_WIDTH,
    MAX_SYNTHETIC_TRACE_HALF_WIDTH,
    MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
    MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
    MIN_SYNTHETIC_TRACE_NOISE_SIGMA,
    MAX_SYNTHETIC_TRACE_NOISE_SIGMA,
    MIN_TECH_AUG_OPERATIONS,
    MAX_TECH_AUG_OPERATIONS,
    MIN_MIXUP_ALPHA,
    MAX_MIXUP_ALPHA,
    MIN_PCB_DEFECT_COUNT,
    MAX_PCB_DEFECT_COUNT,
    MIN_PCB_DEFECT_WEIGHT,
    MAX_PCB_DEFECT_WEIGHT,
    MIN_RECOGNITION_THRESHOLD,
    MAX_RECOGNITION_THRESHOLD,
    MIN_POSTPROCESS_KERNEL_SIZE,
    MAX_POSTPROCESS_KERNEL_SIZE,
    OPTIMIZERS,
    MIXED_PRECISION_MODES,
    LOSS_FUNCTIONS,
    MAX_TOPOGRAPH_LOSS_WEIGHT,
    TOPOGRAPH_LOSS_DISPLAY_NAME,
    MULTI_GPU_MODES,
    SCHEDULER_NAMES,
    ONE_CYCLE_ANNEAL_STRATEGIES,
    CONFIDENCE_SAVE_MODES,
    OPTIMIZER_PRESETS,
    VISIBLE_OPTIMIZER_PRESET_NAMES,
    LOSS_PRESET_WEIGHTS,
    PCB_DEFECT_WEIGHT_FIELDS,
    IC_DEFECT_WEIGHT_FIELDS,
)
from .settings_panel_parts.construction import ConstructionMixin
from .settings_panel_parts.create_collapsible_groupbox import CreateCollapsibleGroupboxMixin
from .settings_panel_parts.build_loss_terms_widget import BuildLossTermsWidgetMixin
from .settings_panel_parts.build_topograph_loss_rows import BuildTopographLossRowsMixin
from .settings_panel_parts.sync_training_data_mode import SyncTrainingDataModeMixin
from .settings_panel_parts.sync_optimizer_mode_controls import SyncOptimizerModeControlsMixin
from .settings_panel_parts.construction_2 import Construction2Mixin
from .settings_panel_parts.construction_3 import Construction3Mixin
from .settings_panel_parts.optional_config_value import OptionalConfigValueMixin
from .settings_panel_parts.synthetic_defect_generator_config import SyntheticDefectGeneratorConfigMixin
from .settings_panel_parts.synthetic_defect_generator_config_2 import SyntheticDefectGeneratorConfig2Mixin
from .settings_panel_parts.tech_aug_config import TechAugConfigMixin
from .settings_panel_parts.pcb_defects_config import PcbDefectsConfigMixin
from .settings_panel_parts.samples_count import SamplesCountMixin
from .settings_panel_parts.rebuild_settings_registry import RebuildSettingsRegistryMixin
from .settings_panel_parts.navigate_setting import NavigateSettingMixin
from .settings_panel_parts.validate_configuration_live import ValidateConfigurationLiveMixin
from .settings_panel_parts.sync_scheduler_controls import SyncSchedulerControlsMixin
from .settings_panel_parts.preprocessing_group_toggled import PreprocessingGroupToggledMixin
from .settings_panel_parts.sync_sem_segmentation_controls import SyncSemSegmentationControlsMixin
from .settings_panel_parts.sync_loss_controls import SyncLossControlsMixin
from .settings_panel_parts.sync_patch_size_layout import SyncPatchSizeLayoutMixin
from .settings_panel_parts.reset_training_augmentation_defaults import ResetTrainingAugmentationDefaultsMixin
from .settings_panel_parts.training_augmentation_config import TrainingAugmentationConfigMixin
from .settings_panel_parts.sync_basic_photometric_master import SyncBasicPhotometricMasterMixin
from .settings_panel_parts.sync_augmentation_controls import SyncAugmentationControlsMixin
from .settings_panel_parts.sync_tech_augmentation_controls import SyncTechAugmentationControlsMixin
from .settings_panel_parts.sync_synthetic_defect_generator import SyncSyntheticDefectGeneratorMixin
from .settings_panel_parts.confidence_export_mode_value import ConfidenceExportModeValueMixin
from .settings_panel_parts.sync_training_augmentation_controls import SyncTrainingAugmentationControlsMixin
from .settings_panel_parts.sync_patch_size_controls import SyncPatchSizeControlsMixin
from .settings_panel_parts.model import ModelMixin
from .settings_panel_parts.settings_panel_qsettings import _settings_panel_qsettings

class SettingsPanel(
    ConstructionMixin,
    CreateCollapsibleGroupboxMixin,
    BuildLossTermsWidgetMixin,
    BuildTopographLossRowsMixin,
    SyncTrainingDataModeMixin,
    SyncOptimizerModeControlsMixin,
    Construction2Mixin,
    Construction3Mixin,
    OptionalConfigValueMixin,
    SyntheticDefectGeneratorConfigMixin,
    SyntheticDefectGeneratorConfig2Mixin,
    TechAugConfigMixin,
    PcbDefectsConfigMixin,
    SamplesCountMixin,
    RebuildSettingsRegistryMixin,
    NavigateSettingMixin,
    ValidateConfigurationLiveMixin,
    SyncSchedulerControlsMixin,
    PreprocessingGroupToggledMixin,
    SyncSemSegmentationControlsMixin,
    SyncLossControlsMixin,
    SyncPatchSizeLayoutMixin,
    ResetTrainingAugmentationDefaultsMixin,
    TrainingAugmentationConfigMixin,
    SyncBasicPhotometricMasterMixin,
    SyncAugmentationControlsMixin,
    SyncTechAugmentationControlsMixin,
    SyncSyntheticDefectGeneratorMixin,
    ConfidenceExportModeValueMixin,
    SyncTrainingAugmentationControlsMixin,
    SyncPatchSizeControlsMixin,
    ModelMixin,
    QDockWidget,
):
    cut_slider_shifted: pyqtSignal = pyqtSignal()
    horisontal_rotate_clicked: pyqtSignal = pyqtSignal()
    vertical_rotate_clicked: pyqtSignal = pyqtSignal()
    model_changed: pyqtSignal = pyqtSignal()
    segment_size_changed: pyqtSignal = pyqtSignal()
    sample_size_changed: pyqtSignal = pyqtSignal()
    optimizer_settings_changed: pyqtSignal = pyqtSignal()
    validation_settings_changed: pyqtSignal = pyqtSignal()
    validation_image_path_requested: pyqtSignal = pyqtSignal()
    validation_label_path_requested: pyqtSignal = pyqtSignal()
    reset_defaults_requested: pyqtSignal = pyqtSignal()
    augmentation_preview_requested: pyqtSignal = pyqtSignal()
    ui_language_changed: pyqtSignal = pyqtSignal(str)
    rare_patch_editor_requested: pyqtSignal = pyqtSignal()
    configuration_validity_changed: pyqtSignal = pyqtSignal(bool, object)

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = SettingsPanel()
    window.show()
    sys.exit(app.exec())

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    FIELD_DESCRIPTION_ROW_SPACING,
    FORM_DEFAULT_MARGINS,
    FORM_HORIZONTAL_SPACING,
    FORM_VERTICAL_SPACING,
    CONTENT_LAYOUT_MARGINS,
    CONTENT_LAYOUT_SPACING,
    OPTIMIZER_PRESET_FLOAT_TOLERANCE,
    LOSS_PRESET_FLOAT_TOLERANCE,
    EDGE_CUT_RANGE,
    EDGE_CUT_STEP,
    COMPRESSION_FACTOR_RANGE,
    COMPRESSION_FACTOR_STEP,
    COMPRESSION_FACTOR_DEFAULT,
    EPOCHS_RANGE,
    EPOCHS_DEFAULT,
    SHIFT_RANGE_MIN,
    SHIFT_RANGE_MAX,
    SAMPLE_SIZE_MIN,
    SAMPLE_SIZE_MAX,
    VALIDATION_MIN,
    VALIDATION_MAX,
    MIN_BATCH,
    MAX_BATCH,
    MIN_DATALOADER_WORKERS,
    MAX_DATALOADER_WORKERS,
    MIN_OVERLAP,
    MAX_OVERLAP,
    MIN_JPEG_QUALITY,
    MAX_JPEG_QUALITY,
    MIN_LOG_UPDATE_FREQUENCY,
    MAX_LOG_UPDATE_FREQUENCY,
    MIN_LEARNING_RATE,
    MAX_LEARNING_RATE,
    MIN_WEIGHT_DECAY,
    MAX_WEIGHT_DECAY,
    MIN_EARLY_STOPPING_PATIENCE,
    MAX_EARLY_STOPPING_PATIENCE,
    MIN_EARLY_STOPPING_MIN_DELTA,
    MAX_EARLY_STOPPING_MIN_DELTA,
    MIN_WARMUP_EPOCHS,
    MAX_WARMUP_EPOCHS,
    MIN_WARMUP_START_FACTOR,
    MAX_WARMUP_START_FACTOR,
    MIN_SCHEDULER_FACTOR,
    MAX_SCHEDULER_FACTOR,
    MIN_SCHEDULER_PATIENCE,
    MAX_SCHEDULER_PATIENCE,
    MIN_SCHEDULER_THRESHOLD,
    MAX_SCHEDULER_THRESHOLD,
    MIN_SCHEDULER_MIN_LR,
    MAX_SCHEDULER_MIN_LR,
    MIN_SCHEDULER_COOLDOWN,
    MAX_SCHEDULER_COOLDOWN,
    MIN_SCHEDULER_T_MAX,
    MAX_SCHEDULER_T_MAX,
    MIN_SCHEDULER_STEP_SIZE,
    MAX_SCHEDULER_STEP_SIZE,
    MIN_SCHEDULER_DIV_FACTOR,
    MAX_SCHEDULER_DIV_FACTOR,
    MIN_SCHEDULER_FINAL_DIV_FACTOR,
    MAX_SCHEDULER_FINAL_DIV_FACTOR,
    MIN_HARD_MINING_STRENGTH,
    MAX_HARD_MINING_STRENGTH,
    MIN_HARD_MINING_EMA_ALPHA,
    MAX_HARD_MINING_EMA_ALPHA,
    MIN_HARD_PIXEL_KEEP_RATIO,
    MAX_HARD_PIXEL_KEEP_RATIO,
    MIN_RARE_PATCH_OVERSAMPLING_FACTOR,
    MAX_RARE_PATCH_OVERSAMPLING_FACTOR,
    MIN_AUG_STRENGTH,
    MAX_AUG_STRENGTH,
    MIN_AUG_NOISE_SIGMA,
    MAX_AUG_NOISE_SIGMA,
    MIN_AUG_BLUR_RADIUS,
    MAX_AUG_BLUR_RADIUS,
    MIN_CROPS_PER_IMAGE,
    MAX_CROPS_PER_IMAGE,
    MIN_CUTOUT_HOLES,
    MAX_CUTOUT_HOLES,
    MIN_RANDOM_ARTIFACTS_COUNT,
    MAX_RANDOM_ARTIFACTS_COUNT,
    MIN_AUGMENTATION_PROBABILITY,
    MAX_AUGMENTATION_PROBABILITY,
    MIN_AUGMENTATION_MULTIPLIER,
    MAX_AUGMENTATION_MULTIPLIER,
    MIN_SYNTHETIC_DATASET_FACTOR,
    MAX_SYNTHETIC_DATASET_FACTOR,
    MIN_SYNTHETIC_IMAGE_SIZE,
    MAX_SYNTHETIC_IMAGE_SIZE,
    MIN_SYNTHETIC_TRACE_COUNT,
    MAX_SYNTHETIC_TRACE_COUNT,
    MIN_SYNTHETIC_SEGMENT_COUNT,
    MAX_SYNTHETIC_SEGMENT_COUNT,
    MIN_SYNTHETIC_TRACE_HALF_WIDTH,
    MAX_SYNTHETIC_TRACE_HALF_WIDTH,
    MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
    MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
    MIN_SYNTHETIC_TRACE_NOISE_SIGMA,
    MAX_SYNTHETIC_TRACE_NOISE_SIGMA,
    MIN_TECH_AUG_OPERATIONS,
    MAX_TECH_AUG_OPERATIONS,
    MIN_MIXUP_ALPHA,
    MAX_MIXUP_ALPHA,
    MIN_PCB_DEFECT_COUNT,
    MAX_PCB_DEFECT_COUNT,
    MIN_PCB_DEFECT_WEIGHT,
    MAX_PCB_DEFECT_WEIGHT,
    MIN_RECOGNITION_THRESHOLD,
    MAX_RECOGNITION_THRESHOLD,
    MIN_POSTPROCESS_KERNEL_SIZE,
    MAX_POSTPROCESS_KERNEL_SIZE,
    OPTIMIZERS,
    MIXED_PRECISION_MODES,
    LOSS_FUNCTIONS,
    MAX_TOPOGRAPH_LOSS_WEIGHT,
    TOPOGRAPH_LOSS_DISPLAY_NAME,
    MULTI_GPU_MODES,
    SCHEDULER_NAMES,
    ONE_CYCLE_ANNEAL_STRATEGIES,
    CONFIDENCE_SAVE_MODES,
    OPTIMIZER_PRESETS,
    VISIBLE_OPTIMIZER_PRESET_NAMES,
    LOSS_PRESET_WEIGHTS,
    PCB_DEFECT_WEIGHT_FIELDS,
    IC_DEFECT_WEIGHT_FIELDS,
    _settings_panel_qsettings,
)
