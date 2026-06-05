from typing import Any, Iterable
from collections.abc import Mapping
from copy import deepcopy

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QDockWidget,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QApplication,
    QScrollArea,
    QSizePolicy,
    QGroupBox,
    QRadioButton,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
)

from neuralimage.UI import ClickableLabel
from neuralimage.lib.data_interfaces import (
    IC_TOPOLOGY_FAMILIES,
    PCB_TOPOLOGY_FAMILIES,
    RANDOM_ARTIFACT_TYPES,
    SampleCutMode,
    build_ic_defect_parameters,
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
from neuralimage.lib.ui_texts import get_ui_section, set_ui_language as set_global_ui_language
from neuralimage.view.settings_panel_bindings import connect_settings_panel_signals
from neuralimage.view.settings_panel_i18n import apply_settings_panel_texts
from neuralimage.view.settings_panel_policy import (
    resolve_work_mode_applicability as resolve_work_mode_applicability_policy,
)
from neuralimage.view.settings_panel_widgets import (
    NoWheelComboBox,
    SlidingPanel,
    create_double_spinbox,
    create_min_max_widget,
    create_slider,
    create_size_widget,
    create_spinbox,
)

FIELD_DESCRIPTION_ROW_SPACING = 8
FORM_DEFAULT_MARGINS = (8, 6, 8, 8)
FORM_HORIZONTAL_SPACING = 10
FORM_VERTICAL_SPACING = 6
CONTENT_LAYOUT_MARGINS = (8, 8, 8, 8)
CONTENT_LAYOUT_SPACING = 10
OPTIMIZER_PRESET_FLOAT_TOLERANCE = 1e-12
LOSS_PRESET_FLOAT_TOLERANCE = 1e-12
EDGE_CUT_RANGE = (0, 500)
EDGE_CUT_STEP = 10
COMPRESSION_FACTOR_RANGE = (1, 64)
COMPRESSION_FACTOR_STEP = 1
COMPRESSION_FACTOR_DEFAULT = 1
EPOCHS_RANGE = (0, 1000)
EPOCHS_DEFAULT = 20

SHIFT_RANGE_MIN = 4
SHIFT_RANGE_MAX = 2000

SAMPLE_SIZE_MIN = 8
SAMPLE_SIZE_MAX = 2000

VALIDATION_MIN = 0
VALIDATION_MAX = 50

MIN_BATCH = 1
MAX_BATCH = 64
MIN_DATALOADER_WORKERS = -1
MAX_DATALOADER_WORKERS = 64

MIN_OVERLAP = 0
MAX_OVERLAP = 32
MIN_JPEG_QUALITY = 1
MAX_JPEG_QUALITY = 100
MIN_LOG_UPDATE_FREQUENCY = 0
MAX_LOG_UPDATE_FREQUENCY = 5000

MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1.0
MIN_WEIGHT_DECAY = 0.0
MAX_WEIGHT_DECAY = 1.0
MIN_EARLY_STOPPING_PATIENCE = 0
MAX_EARLY_STOPPING_PATIENCE = 1000
MIN_EARLY_STOPPING_MIN_DELTA = 0.0
MAX_EARLY_STOPPING_MIN_DELTA = 1.0
MIN_WARMUP_EPOCHS = 1
MAX_WARMUP_EPOCHS = 2000
MIN_WARMUP_START_FACTOR = 0.0
MAX_WARMUP_START_FACTOR = 1.0
MIN_SCHEDULER_FACTOR = 0.01
MAX_SCHEDULER_FACTOR = 0.99
MIN_SCHEDULER_PATIENCE = 0
MAX_SCHEDULER_PATIENCE = 2000
MIN_SCHEDULER_THRESHOLD = 0.0
MAX_SCHEDULER_THRESHOLD = 10.0
MIN_SCHEDULER_MIN_LR = 0.0
MAX_SCHEDULER_MIN_LR = 1.0
MIN_SCHEDULER_COOLDOWN = 0
MAX_SCHEDULER_COOLDOWN = 2000
MIN_SCHEDULER_T_MAX = 1
MAX_SCHEDULER_T_MAX = 10000
MIN_SCHEDULER_STEP_SIZE = 1
MAX_SCHEDULER_STEP_SIZE = 10000
MIN_SCHEDULER_DIV_FACTOR = 1.0
MAX_SCHEDULER_DIV_FACTOR = 1000000.0
MIN_SCHEDULER_FINAL_DIV_FACTOR = 1.0
MAX_SCHEDULER_FINAL_DIV_FACTOR = 1000000.0
MIN_HARD_MINING_STRENGTH = 0.0
MAX_HARD_MINING_STRENGTH = 10.0
MIN_HARD_MINING_EMA_ALPHA = 0.0
MAX_HARD_MINING_EMA_ALPHA = 1.0
MIN_HARD_PIXEL_KEEP_RATIO = 0.01
MAX_HARD_PIXEL_KEEP_RATIO = 1.0
MIN_RARE_PATCH_OVERSAMPLING_FACTOR = 2
MAX_RARE_PATCH_OVERSAMPLING_FACTOR = 64
MIN_AUG_STRENGTH = 0.0
MAX_AUG_STRENGTH = 1.0
MIN_AUG_NOISE_SIGMA = 0.0
MAX_AUG_NOISE_SIGMA = 0.2
MIN_AUG_BLUR_RADIUS = 0.0
MAX_AUG_BLUR_RADIUS = 5.0
MIN_CROPS_PER_IMAGE = 1
MAX_CROPS_PER_IMAGE = 5000
MIN_CUTOUT_HOLES = 1
MAX_CUTOUT_HOLES = 32
MIN_RANDOM_ARTIFACTS_COUNT = 1
MAX_RANDOM_ARTIFACTS_COUNT = 16
MIN_AUGMENTATION_PROBABILITY = 0.0
MAX_AUGMENTATION_PROBABILITY = 1.0
MIN_SYNTHETIC_DATASET_FACTOR = 0.0
MAX_SYNTHETIC_DATASET_FACTOR = 10.0
MIN_SYNTHETIC_IMAGE_SIZE = 64
MAX_SYNTHETIC_IMAGE_SIZE = 8192
MIN_SYNTHETIC_TRACE_COUNT = 1
MAX_SYNTHETIC_TRACE_COUNT = 200
MIN_SYNTHETIC_SEGMENT_COUNT = 1
MAX_SYNTHETIC_SEGMENT_COUNT = 24
MIN_SYNTHETIC_TRACE_HALF_WIDTH = 1
MAX_SYNTHETIC_TRACE_HALF_WIDTH = 50
MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA = 0.0
MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA = 0.2
MIN_SYNTHETIC_TRACE_NOISE_SIGMA = 0.0
MAX_SYNTHETIC_TRACE_NOISE_SIGMA = 0.2
MIN_TECH_AUG_OPERATIONS = 1
MAX_TECH_AUG_OPERATIONS = 6
MIN_MIXUP_ALPHA = 0.0
MAX_MIXUP_ALPHA = 10.0
MIN_PCB_DEFECT_COUNT = 1
MAX_PCB_DEFECT_COUNT = 8
MIN_PCB_DEFECT_WEIGHT = 0.0
MAX_PCB_DEFECT_WEIGHT = 5.0
MIN_RECOGNITION_THRESHOLD = 0.0
MAX_RECOGNITION_THRESHOLD = 1.0
MIN_POSTPROCESS_KERNEL_SIZE = 1
MAX_POSTPROCESS_KERNEL_SIZE = 31
OPTIMIZERS = ('adam', 'adamw', 'adamw_muon')
MIXED_PRECISION_MODES = ('off', 'fp16', 'bf16')
LOSS_FUNCTIONS = LOSS_SELECTION_NAMES
MULTI_GPU_MODES = ('off', 'dataparallel', 'distributeddataparallel')
SCHEDULER_NAMES = ('off', 'reduce_on_plateau', 'cosine_annealing', 'one_cycle', 'step_lr')
ONE_CYCLE_ANNEAL_STRATEGIES = ('cos', 'linear')
CONFIDENCE_SAVE_MODES = ('off', 'model_output', 'tta')
OPTIMIZER_PRESETS = (
    ('Adam', 'adam', 1e-3, 0.0),
    ('AdamW', 'adamw', 5e-4, 1e-2),
    ('AdamW + Muon', 'adamw_muon', 3e-4, 2e-2),
)
VISIBLE_OPTIMIZER_PRESET_NAMES = frozenset({'adam', 'adamw', 'adamw_muon'})
LOSS_PRESET_WEIGHTS = {
    'conductors': {'bce': 0.5, 'dice': 0.5},
    'contacts': {'bce': 0.5, 'focal_tversky': 0.5},
}
PCB_DEFECT_WEIGHT_FIELDS = (
    ('break', 'pcb_break_severity'),
    ('short', 'pcb_short_severity'),
    ('missing_copper', 'pcb_missing_copper_severity'),
    ('excess_copper', 'pcb_excess_copper_severity'),
    ('pinhole', 'pcb_pinhole_severity'),
    ('spurious_copper', 'pcb_spurious_copper_severity'),
    ('via', 'pcb_via_severity'),
    ('misalignment', 'pcb_misalignment_severity'),
)
IC_DEFECT_WEIGHT_FIELDS = (
    ('line_break', 'ic_line_break_severity'),
    ('bridge', 'ic_bridge_severity'),
    ('necking', 'ic_necking_severity'),
    ('missing_metal', 'ic_missing_metal_severity'),
    ('spur', 'ic_spur_severity'),
    ('pinhole', 'ic_pinhole_severity'),
    ('via_open', 'ic_via_open_severity'),
    ('line_shift', 'ic_line_shift_severity'),
)


class SettingsPanel(QDockWidget):
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        texts = get_ui_section('settings_panel')
        self._texts = texts if isinstance(texts, dict) else {}
        self._content_widget = QWidget()
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setWidget(self._content_widget)
        self.setWidget(self._scroll_area)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.models: list[str] = []
        self.deprecated_models: list[str] = []
        self.experimental_models: list[str] = []
        self._page_indexes: dict[str, int] = {}
        self.size_policy = QSizePolicy()
        self._desc_labels: dict[str, QLabel] = {}
        self._field_rows: dict[QWidget, QWidget] = {}
        self._desc_fields: dict[str, QWidget] = {}
        self._training_controls_applicable = True
        self._recognition_controls_applicable = True
        self._model_selector_applicable = True
        self._patch_batch_sync_guard = False
        self._loss_terms_guard = False
        self._model_combo_guard = False

        self._sample_count_value = 0
        self._sample_count_pending = False
        self._synthetic_defect_generator_payload: dict[str, Any] = {}
        self._tech_aug_config_payload: dict[str, Any] = {}
        self._pcb_defects_config_payload: dict[str, Any] = {}
        self.loss_term_checkboxes: dict[str, QCheckBox] = {}
        self.loss_term_spinboxes: dict[str, Any] = {}
        self.loss_term_labels: dict[str, QLabel] = {}
        self.loss_preset_buttons: dict[str, QPushButton] = {}

        self.horizontal_rotation = QCheckBox('')
        self.vertical_rotation = QCheckBox('')
        self.flip_x = QCheckBox('')
        self.flip_y = QCheckBox('')
        self.additional_augmentation_check_box = QCheckBox('')
        self.random_crop_check_box = QCheckBox('')
        self.scale_augmentation_check_box = QCheckBox('')
        self.synthetic_defect_generator_check_box = QCheckBox('')
        self.tech_augmentation_check_box = QCheckBox('')
        self.tech_augmentation_debug_pair_check_box = QCheckBox('')
        self.cutout_check_box = QCheckBox('')
        self.random_artifacts_check_box = QCheckBox('')
        self.random_artifact_type_checkboxes: dict[str, QCheckBox] = {
            artifact_name: QCheckBox('') for artifact_name in RANDOM_ARTIFACT_TYPES
        }
        for checkbox in self.random_artifact_type_checkboxes.values():
            checkbox.setChecked(True)
        self.mixup_check_box = QCheckBox('')
        self.pcb_defects_check_box = QCheckBox('')
        self.pcb_defects_use_input_mask_check_box = QCheckBox('')
        self.pcb_defects_use_defect_mask_as_label_check_box = QCheckBox('')
        self.validation_check_box = QCheckBox('')
        self.save_validation_binary_images_check_box = QCheckBox('')
        self.samples_number = QLabel('')
        self.augmentation_preview_button = QPushButton('')
        self.shuffle_frames_check_box = QCheckBox('')
        self.shuffle_patches_in_frame_check_box = QCheckBox('')
        self.shuffle_frames_check_box.setChecked(True)
        self.shuffle_patches_in_frame_check_box.setChecked(True)
        # Backward-compatible alias for old code expecting a generic shuffle checkbox.
        self.shuffle_check_box = self.shuffle_frames_check_box
        self.recursive_file_search_check_box = QCheckBox('')

        self.nn_model_type = NoWheelComboBox()
        self.nn_model_type.setSizePolicy(self.size_policy)
        self.deprecated_model_type = NoWheelComboBox()
        self.deprecated_model_type.setSizePolicy(self.size_policy)
        self.experimental_model_type = NoWheelComboBox()
        self.experimental_model_type.setSizePolicy(self.size_policy)
        self.nn_model_type.currentIndexChanged.connect(lambda *_args: self._sync_model_selection(self.nn_model_type))
        self.deprecated_model_type.currentIndexChanged.connect(
            lambda *_args: self._sync_model_selection(self.deprecated_model_type)
        )
        self.experimental_model_type.currentIndexChanged.connect(
            lambda *_args: self._sync_model_selection(self.experimental_model_type)
        )

        self.shift_spinbox = create_spinbox((SHIFT_RANGE_MIN, SHIFT_RANGE_MAX), default_value=100, step=10)
        self.crops_per_image_spinbox = create_spinbox(
            (MIN_CROPS_PER_IMAGE, MAX_CROPS_PER_IMAGE),
            default_value=64,
            step=1,
        )
        self.augmentation_brightness_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=0.1,
            decimals=2,
        )
        self.augmentation_contrast_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=0.1,
            decimals=2,
        )
        self.augmentation_gamma_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=0.15,
            decimals=2,
        )
        self.augmentation_noise_probability_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=0.5,
            decimals=2,
        )
        self.augmentation_noise_sigma_spinbox = create_double_spinbox(
            (MIN_AUG_NOISE_SIGMA, MAX_AUG_NOISE_SIGMA),
            step=0.005,
            default_value=0.01,
            decimals=3,
        )
        self.augmentation_blur_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=0.25,
            decimals=2,
        )
        self.augmentation_blur_radius_spinbox = create_double_spinbox(
            (MIN_AUG_BLUR_RADIUS, MAX_AUG_BLUR_RADIUS),
            step=0.1,
            default_value=1.0,
            decimals=2,
        )
        self.scale_augmentation_strength_spinbox = create_double_spinbox(
            (0.0, 1.0),
            step=0.05,
            default_value=0.2,
            decimals=2,
        )
        self.synthetic_dataset_factor_spinbox = create_double_spinbox(
            (MIN_SYNTHETIC_DATASET_FACTOR, MAX_SYNTHETIC_DATASET_FACTOR),
            step=0.25,
            default_value=1.0,
            decimals=2,
        )
        self.synthetic_image_width_spinbox = create_spinbox(
            (MIN_SYNTHETIC_IMAGE_SIZE, MAX_SYNTHETIC_IMAGE_SIZE),
            default_value=1024,
            step=32,
        )
        self.synthetic_image_height_spinbox = create_spinbox(
            (MIN_SYNTHETIC_IMAGE_SIZE, MAX_SYNTHETIC_IMAGE_SIZE),
            default_value=1024,
            step=32,
        )
        self.synthetic_image_size_widget = create_size_widget(
            self.synthetic_image_width_spinbox,
            self.synthetic_image_height_spinbox,
        )
        self.synthetic_trace_count_min_spinbox = create_spinbox(
            (MIN_SYNTHETIC_TRACE_COUNT, MAX_SYNTHETIC_TRACE_COUNT),
            default_value=5,
            step=1,
        )
        self.synthetic_trace_count_max_spinbox = create_spinbox(
            (MIN_SYNTHETIC_TRACE_COUNT, MAX_SYNTHETIC_TRACE_COUNT),
            default_value=5,
            step=1,
        )
        self.synthetic_trace_count_range_widget = create_min_max_widget(
            self.synthetic_trace_count_min_spinbox,
            self.synthetic_trace_count_max_spinbox,
        )
        self.synthetic_segment_count_min_spinbox = create_spinbox(
            (MIN_SYNTHETIC_SEGMENT_COUNT, MAX_SYNTHETIC_SEGMENT_COUNT),
            default_value=4,
            step=1,
        )
        self.synthetic_segment_count_max_spinbox = create_spinbox(
            (MIN_SYNTHETIC_SEGMENT_COUNT, MAX_SYNTHETIC_SEGMENT_COUNT),
            default_value=4,
            step=1,
        )
        self.synthetic_segment_count_range_widget = create_min_max_widget(
            self.synthetic_segment_count_min_spinbox,
            self.synthetic_segment_count_max_spinbox,
        )
        self.synthetic_trace_half_width_min_spinbox = create_spinbox(
            (MIN_SYNTHETIC_TRACE_HALF_WIDTH, MAX_SYNTHETIC_TRACE_HALF_WIDTH),
            default_value=2,
            step=1,
        )
        self.synthetic_trace_half_width_max_spinbox = create_spinbox(
            (MIN_SYNTHETIC_TRACE_HALF_WIDTH, MAX_SYNTHETIC_TRACE_HALF_WIDTH),
            default_value=2,
            step=1,
        )
        self.synthetic_trace_half_width_range_widget = create_min_max_widget(
            self.synthetic_trace_half_width_min_spinbox,
            self.synthetic_trace_half_width_max_spinbox,
        )
        self.synthetic_background_noise_sigma_min_spinbox = create_double_spinbox(
            (MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA, MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA),
            step=0.005,
            default_value=0.02,
            decimals=3,
        )
        self.synthetic_background_noise_sigma_max_spinbox = create_double_spinbox(
            (MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA, MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA),
            step=0.005,
            default_value=0.02,
            decimals=3,
        )
        self.synthetic_background_noise_sigma_range_widget = create_min_max_widget(
            self.synthetic_background_noise_sigma_min_spinbox,
            self.synthetic_background_noise_sigma_max_spinbox,
        )
        self.synthetic_trace_noise_sigma_min_spinbox = create_double_spinbox(
            (MIN_SYNTHETIC_TRACE_NOISE_SIGMA, MAX_SYNTHETIC_TRACE_NOISE_SIGMA),
            step=0.005,
            default_value=0.01,
            decimals=3,
        )
        self.synthetic_trace_noise_sigma_max_spinbox = create_double_spinbox(
            (MIN_SYNTHETIC_TRACE_NOISE_SIGMA, MAX_SYNTHETIC_TRACE_NOISE_SIGMA),
            step=0.005,
            default_value=0.01,
            decimals=3,
        )
        self.synthetic_trace_noise_sigma_range_widget = create_min_max_widget(
            self.synthetic_trace_noise_sigma_min_spinbox,
            self.synthetic_trace_noise_sigma_max_spinbox,
        )
        self.synthetic_topology_domain_combo = NoWheelComboBox()
        self.synthetic_topology_domain_combo.addItem('PCB', 'pcb')
        self.synthetic_topology_domain_combo.addItem('IC', 'ic')
        self.pcb_topology_family_combo = NoWheelComboBox()
        for family in PCB_TOPOLOGY_FAMILIES:
            self.pcb_topology_family_combo.addItem(family, family)
        self.ic_topology_family_combo = NoWheelComboBox()
        for family in IC_TOPOLOGY_FAMILIES:
            self.ic_topology_family_combo.addItem(family, family)
        self.tech_aug_min_operations_spinbox = create_spinbox(
            (MIN_TECH_AUG_OPERATIONS, MAX_TECH_AUG_OPERATIONS),
            default_value=1,
            step=1,
        )
        self.tech_aug_max_operations_spinbox = create_spinbox(
            (MIN_TECH_AUG_OPERATIONS, MAX_TECH_AUG_OPERATIONS),
            default_value=3,
            step=1,
        )
        self.tech_aug_max_changed_pixels_ratio_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.01,
            default_value=0.2,
            decimals=2,
        )
        self.tech_aug_max_foreground_ratio_delta_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.01,
            default_value=0.12,
            decimals=2,
        )
        self.tech_aug_global_width_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=0.45,
            decimals=2,
        )
        self.tech_aug_scale_rethreshold_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=0.35,
            decimals=2,
        )
        self.tech_aug_blur_threshold_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=0.3,
            decimals=2,
        )
        self.tech_aug_boundary_aware_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=0.7,
            decimals=2,
        )
        self.tech_aug_local_morphology_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=0.35,
            decimals=2,
        )
        self.tech_aug_gap_variation_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=0.3,
            decimals=2,
        )
        self.cutout_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=1.0,
            decimals=2,
        )
        self.cutout_holes_spinbox = create_spinbox(
            (MIN_CUTOUT_HOLES, MAX_CUTOUT_HOLES),
            default_value=1,
            step=1,
        )
        self.cutout_size_ratio_spinbox = create_double_spinbox(
            (0.0, 1.0),
            step=0.05,
            default_value=0.25,
            decimals=2,
        )
        self.random_artifacts_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=1.0,
            decimals=2,
        )
        self.random_artifacts_count_spinbox = create_spinbox(
            (MIN_RANDOM_ARTIFACTS_COUNT, MAX_RANDOM_ARTIFACTS_COUNT),
            default_value=1,
            step=1,
        )
        self.random_artifacts_size_ratio_spinbox = create_double_spinbox(
            (0.0, 1.0),
            step=0.05,
            default_value=0.25,
            decimals=2,
        )
        self.mixup_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=1.0,
            decimals=2,
        )
        self.mixup_alpha_spinbox = create_double_spinbox(
            (MIN_MIXUP_ALPHA, MAX_MIXUP_ALPHA),
            step=0.05,
            default_value=0.2,
            decimals=2,
        )
        self.pcb_defects_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=0.5,
            decimals=2,
        )
        self.pcb_defects_min_count_spinbox = create_spinbox(
            (MIN_PCB_DEFECT_COUNT, MAX_PCB_DEFECT_COUNT),
            default_value=1,
            step=1,
        )
        self.pcb_defects_max_count_spinbox = create_spinbox(
            (MIN_PCB_DEFECT_COUNT, MAX_PCB_DEFECT_COUNT),
            default_value=3,
            step=1,
        )
        self.pcb_defect_type_spinboxes: dict[str, Any] = {
            defect_name: create_slider((0, 100), default_value=50)
            for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS
        }
        self.pcb_defect_type_checkboxes: dict[str, QCheckBox] = {
            defect_name: QCheckBox('')
            for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS
        }
        for checkbox in self.pcb_defect_type_checkboxes.values():
            checkbox.setChecked(True)
        self.ic_defect_type_spinboxes: dict[str, Any] = {
            defect_name: create_slider((0, 100), default_value=50)
            for defect_name, _label_key in IC_DEFECT_WEIGHT_FIELDS
        }
        self.ic_defect_type_checkboxes: dict[str, QCheckBox] = {
            defect_name: QCheckBox('')
            for defect_name, _label_key in IC_DEFECT_WEIGHT_FIELDS
        }
        for checkbox in self.ic_defect_type_checkboxes.values():
            checkbox.setChecked(True)

        self.train_patch_x_size = create_spinbox((SAMPLE_SIZE_MIN, SAMPLE_SIZE_MAX), default_value=256, step=10)
        self.train_patch_y_size = create_spinbox((SAMPLE_SIZE_MIN, SAMPLE_SIZE_MAX), default_value=256, step=10)
        self.recognition_patch_x_size = create_spinbox((SAMPLE_SIZE_MIN, SAMPLE_SIZE_MAX), default_value=256, step=10)
        self.recognition_patch_y_size = create_spinbox((SAMPLE_SIZE_MIN, SAMPLE_SIZE_MAX), default_value=256, step=10)
        self.epochs_spinbox = create_spinbox(EPOCHS_RANGE, default_value=EPOCHS_DEFAULT, step=1)
        # Backward-compatible aliases (legacy code expects training patch controls here).
        self.sample_x_size = self.train_patch_x_size
        self.sample_y_size = self.train_patch_y_size

        self.validation_spinbox = create_spinbox((VALIDATION_MIN, VALIDATION_MAX), default_value=20, step=5)
        self.validation_mode_combo = NoWheelComboBox()
        self.validation_mode_combo.addItem('split', 'split')
        self.validation_mode_combo.addItem('external', 'external')
        self.validation_image_path_label = ClickableLabel()
        self.validation_label_path_label = ClickableLabel()
        self._validation_image_path_value = ''
        self._validation_label_path_value = ''
        self.validation_check_box.toggled.connect(self._sync_validation_controls)
        self.validation_mode_combo.currentIndexChanged.connect(
            lambda *_args, **_kwargs: self._sync_validation_controls(self.validation_check_box.isChecked())
        )
        self._sync_validation_controls(self.validation_check_box.isChecked())

        self._init_color_type_combobox()
        self._init_sample_type()
        self._init_nn_auxilary_settings()
        self._init_preprocess_groupbox()
        self._init_layout()
        self._apply_localized_texts()
        self.sync_business_logic_controls(None)

    def _get_desc_label(self, key: str) -> QLabel:
        label = self._desc_labels.get(key)
        if label is not None:
            return label
        label = QLabel('')
        label.setWordWrap(False)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setStyleSheet('')
        self._desc_labels[key] = label
        return label

    def _field_with_description(self, field: QWidget, key: str) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(FIELD_DESCRIPTION_ROW_SPACING)
        row_layout.addWidget(self._get_desc_label(key), 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        row_layout.addWidget(field, 1, Qt.AlignmentFlag.AlignVCenter)
        self._field_rows[field] = row_widget
        self._desc_fields[key] = field
        return row_widget

    def _add_labeled_row(self, form: QFormLayout, field: QWidget, key: str) -> None:
        """Add a single visible label inside the row to avoid duplicate captions."""
        form.addRow(self._field_with_description(field, key))

    def _create_collapsible_groupbox(self) -> tuple[QGroupBox, QVBoxLayout, QWidget]:
        groupbox = QGroupBox('')
        groupbox.setCheckable(True)
        groupbox.setChecked(False)
        groupbox_layout = QVBoxLayout(groupbox)
        groupbox_layout.setContentsMargins(*CONTENT_LAYOUT_MARGINS)
        groupbox_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        content_widget.setVisible(False)
        groupbox_layout.addWidget(content_widget)
        groupbox.toggled.connect(content_widget.setVisible)
        return groupbox, content_layout, content_widget

    @staticmethod
    def _apply_tooltip_to_widget_and_children(widget: QWidget, text: str) -> None:
        widget.setToolTip(text)

    def _build_loss_terms_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.loss_presets_widget = QWidget()
        self.loss_presets_layout = QHBoxLayout(self.loss_presets_widget)
        self.loss_presets_layout.setContentsMargins(0, 0, 0, 0)
        self.loss_presets_layout.setSpacing(8)
        for preset_name in ('conductors', 'contacts'):
            button = QPushButton('')
            button.setCheckable(True)
            button.setProperty("selectionRole", "mode")
            button.setProperty("presetRole", "quickPreset")
            button.clicked.connect(lambda _checked=False, name=preset_name: self._apply_loss_preset(name))
            self.loss_presets_layout.addWidget(button)
            self.loss_preset_buttons[preset_name] = button
        self.loss_presets_layout.addStretch(1)

        for loss_name in LOSS_FUNCTIONS:
            checkbox = QCheckBox('')
            label = QLabel(LOSS_TERM_DISPLAY_NAMES.get(loss_name, loss_name))
            spinbox = create_double_spinbox(
                (0.0, MAX_LOSS_TERM_WEIGHT_SUM),
                step=0.05,
                default_value=0.0,
                decimals=2,
            )
            spinbox.setEnabled(False)

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            row_layout.addWidget(checkbox, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            row_layout.addWidget(label, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            row_layout.addWidget(spinbox, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(row_widget)

            self.loss_term_checkboxes[loss_name] = checkbox
            self.loss_term_spinboxes[loss_name] = spinbox
            self.loss_term_labels[loss_name] = label

            checkbox.toggled.connect(lambda checked, name=loss_name: self._on_loss_term_toggled(name, checked))
            spinbox.valueChanged.connect(lambda _value, name=loss_name: self._on_loss_term_value_changed(name))

        self.loss_formula_label = QLabel('')
        self.loss_formula_label.setWordWrap(True)
        self.loss_formula_label.setTextFormat(Qt.TextFormat.RichText)
        self.loss_formula_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.loss_formula_label)

        self.set_loss_term_weights(DEFAULT_LOSS_TERM_WEIGHTS)
        return widget

    def _loss_term_total(self, *, exclude: str | None = None) -> float:
        total = 0.0
        for loss_name in LOSS_FUNCTIONS:
            if exclude is not None and loss_name == exclude:
                continue
            if not self.loss_term_checkboxes[loss_name].isChecked():
                continue
            total += float(self.loss_term_spinboxes[loss_name].value())
        return total

    def _max_loss_weight_for(self, loss_name: str) -> float:
        current_value = float(self.loss_term_spinboxes[loss_name].value())
        return max(0.0, MAX_LOSS_TERM_WEIGHT_SUM - (self._loss_term_total() - current_value))

    def _on_loss_term_toggled(self, loss_name: str, checked: bool) -> None:
        if self._loss_terms_guard:
            return
        spinbox = self.loss_term_spinboxes[loss_name]
        self._loss_terms_guard = True
        try:
            if checked:
                remaining = max(0.0, MAX_LOSS_TERM_WEIGHT_SUM - self._loss_term_total(exclude=loss_name))
                initial_value = float(spinbox.value())
                if initial_value <= 0.0 and remaining > 0.0:
                    initial_value = remaining if self._loss_term_total(exclude=loss_name) <= 0.0 else min(0.1, remaining)
                spinbox.setValue(min(initial_value, remaining))
            else:
                spinbox.setValue(0.0)
        finally:
            self._loss_terms_guard = False
        self._sync_loss_controls()

    def _on_loss_term_value_changed(self, loss_name: str) -> None:
        if self._loss_terms_guard:
            return
        if not self.loss_term_checkboxes[loss_name].isChecked():
            return
        spinbox = self.loss_term_spinboxes[loss_name]
        max_value = max(0.0, MAX_LOSS_TERM_WEIGHT_SUM - self._loss_term_total(exclude=loss_name))
        self._loss_terms_guard = True
        try:
            if float(spinbox.value()) > max_value:
                spinbox.setValue(max_value)
        finally:
            self._loss_terms_guard = False
        self._sync_loss_controls()

    def get_loss_term_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for loss_name in LOSS_FUNCTIONS:
            if not self.loss_term_checkboxes[loss_name].isChecked():
                continue
            value = float(self.loss_term_spinboxes[loss_name].value())
            if value > 0.0:
                weights[loss_name] = value
        return sanitize_loss_term_weights(weights)

    def set_loss_term_weights(self, weights: dict[str, float] | None) -> None:
        sanitized = sanitize_loss_term_weights(weights)
        self._loss_terms_guard = True
        try:
            for loss_name in LOSS_FUNCTIONS:
                checkbox = self.loss_term_checkboxes[loss_name]
                spinbox = self.loss_term_spinboxes[loss_name]
                spinbox.setMaximum(MAX_LOSS_TERM_WEIGHT_SUM)
                checkbox.setChecked(loss_name in sanitized)
                spinbox.setValue(float(sanitized.get(loss_name, 0.0)))
        finally:
            self._loss_terms_guard = False
        self._sync_loss_controls()

    def _apply_loss_preset(self, preset_name: str) -> None:
        weights = LOSS_PRESET_WEIGHTS.get(str(preset_name))
        if weights is None:
            return
        self.set_loss_term_weights(dict(weights))
        self.optimizer_settings_changed.emit()

    def _set_field_enabled(self, field: QWidget, enabled: bool) -> None:
        row_widget = self._field_rows.get(field)
        if row_widget is not None:
            row_widget.setEnabled(enabled)
        field.setEnabled(enabled)

    def _set_field_visible(self, field: QWidget, visible: bool) -> None:
        row_widget = self._field_rows.get(field)
        if row_widget is not None:
            row_widget.setVisible(visible)
        else:
            field.setVisible(visible)

    def _set_fields_enabled(self, fields: Iterable[QWidget], enabled: bool) -> None:
        for field in fields:
            if isinstance(field, QWidget):
                self._set_field_enabled(field, enabled)

    @staticmethod
    def _set_widgets_enabled(widgets: Iterable[QWidget], enabled: bool) -> None:
        for widget in widgets:
            if isinstance(widget, QWidget):
                widget.setEnabled(enabled)

    @staticmethod
    def _resolve_work_mode_applicability(work_mode: str | None) -> tuple[bool, bool, bool]:
        applicability = resolve_work_mode_applicability_policy(work_mode)
        return applicability.training, applicability.recognition, applicability.model_selector

    def _set_work_mode_flags(
        self,
        training_applicable: bool,
        recognition_applicable: bool,
        model_selector_applicable: bool,
    ) -> None:
        self._training_controls_applicable = training_applicable
        self._recognition_controls_applicable = recognition_applicable
        self._model_selector_applicable = model_selector_applicable

    def _sync_general_mode_controls(
        self,
        training_applicable: bool,
        recognition_applicable: bool,
        model_selector_applicable: bool,
        batch_related: bool,
    ) -> None:
        self._set_widgets_enabled((self.samples_number,), training_applicable)
        self._set_widgets_enabled(
            (self.shuffle_frames_check_box, self.shuffle_patches_in_frame_check_box),
            training_applicable,
        )
        self._set_widgets_enabled((self.model_variants_groupbox,), model_selector_applicable)
        self._set_fields_enabled((self.epochs_spinbox,), training_applicable)
        self._set_fields_enabled(
            (
                self.nn_model_type,
                self.deprecated_model_type,
                self.experimental_model_type,
            ),
            model_selector_applicable,
        )
        self._set_fields_enabled((self.color_type,), training_applicable)
        self._set_fields_enabled((self.train_patch_size_widget,), batch_related)
        self._set_fields_enabled((self.recursive_file_search_check_box,), training_applicable or recognition_applicable)
        self._set_fields_enabled((self.sync_patch_sizes_check_box,), recognition_applicable)
        self._set_fields_enabled((self.recognition_patch_size_widget,), recognition_applicable)

    def _sync_training_data_mode_controls(self, training_applicable: bool) -> None:
        self._set_widgets_enabled(
            (
                self.shuffle_groupbox,
                self.horizontal_rotation,
                self.vertical_rotation,
                self.flip_x,
                self.flip_y,
                self.additional_augmentation_check_box,
                self.random_crop_check_box,
                self.scale_augmentation_check_box,
                self.synthetic_defect_generator_groupbox,
                self.synthetic_defect_generator_check_box,
                self.tech_augmentation_check_box,
                self.tech_augmentation_debug_pair_check_box,
                self.cutout_check_box,
                self.random_artifacts_check_box,
                *self.random_artifact_type_checkboxes.values(),
                self.mixup_check_box,
                self.pcb_defects_groupbox,
                self.pcb_defects_check_box,
                self.pcb_defects_use_input_mask_check_box,
                self.pcb_defects_use_defect_mask_as_label_check_box,
                self.validation_groupbox,
                self.validation_check_box,
                self.augmentation_preview_button,
                self.sample_type_groupbox,
                self.cut_dataset_type,
                self.no_cut_dataset_type,
                self.prepare_samples_groupbox,
                self.enable_crop_processing,
                self.enable_resize_processing,
                self.compression_factor_spinbox,
            ),
            training_applicable,
        )
        self._set_fields_enabled((self.shift_spinbox,), training_applicable)
        self._sync_augmentation_controls(self.additional_augmentation_check_box.isChecked())
        self._sync_tech_augmentation_controls(self.tech_augmentation_check_box.isChecked())
        self._sync_synthetic_defect_generator_controls(self.synthetic_defect_generator_check_box.isChecked())
        self._sync_validation_controls(self.validation_check_box.isChecked())
        self._sync_preprocess_controls()

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

    def _init_sample_type(self) -> None:
        self.sample_type_groupbox = QGroupBox('')
        vbox_layout = QVBoxLayout()
        vbox_layout.setContentsMargins(*FORM_DEFAULT_MARGINS)
        vbox_layout.setSpacing(FORM_VERTICAL_SPACING)
        self.sample_type_groupbox.setLayout(vbox_layout)
        self.cut_dataset_type = QRadioButton('')
        self.no_cut_dataset_type = QRadioButton('')
        self.no_cut_dataset_type.setChecked(True)
        vbox_layout.addWidget(self.cut_dataset_type)
        vbox_layout.addWidget(self.no_cut_dataset_type)

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

    def _init_nn_auxilary_settings(self) -> None:
        self.nn_auxilary_settings_groupbox = QGroupBox('')
        nn_sections_layout = QVBoxLayout()
        nn_sections_layout.setContentsMargins(*FORM_DEFAULT_MARGINS)
        nn_sections_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        self.nn_auxilary_settings_groupbox.setLayout(nn_sections_layout)

        def _build_subgroup() -> tuple[QGroupBox, QFormLayout]:
            group = QGroupBox('')
            form = self._create_form_layout()
            group.setLayout(form)
            return group, form

        self.optimizer_preset_buttons: list[QPushButton] = []
        self._visible_optimizer_presets: list[tuple[str, str, float, float]] = []
        self.optimizer_presets_widget = QWidget()
        preset_layout = QHBoxLayout(self.optimizer_presets_widget)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(6)
        for title, optimizer_name, learning_rate, weight_decay in OPTIMIZER_PRESETS:
            if optimizer_name not in VISIBLE_OPTIMIZER_PRESET_NAMES:
                continue
            btn = QPushButton(title)
            btn.setCheckable(True)
            btn.setProperty("selectionRole", "mode")
            btn.setProperty("presetRole", "quickPreset")
            btn.clicked.connect(
                lambda _checked=False, n=optimizer_name, lr=learning_rate, wd=weight_decay: self._apply_optimizer_preset(
                    n, lr, wd
                )
            )
            self.optimizer_preset_buttons.append(btn)
            self._visible_optimizer_presets.append((title, optimizer_name, learning_rate, weight_decay))
            preset_layout.addWidget(btn)
        preset_layout.addStretch(1)

        self.optimizer_type = NoWheelComboBox()
        self.optimizer_type.addItems(list(OPTIMIZERS))
        self.mixed_precision_type = NoWheelComboBox()
        self.mixed_precision_type.addItems(list(MIXED_PRECISION_MODES))
        self.mixed_precision_type.setCurrentText('fp16')
        self.deep_supervision_check_box = QCheckBox('')
        self.deep_supervision_check_box.setChecked(False)
        self.loss_function_type = NoWheelComboBox()
        self.loss_function_type.addItems(list(LOSS_FUNCTIONS))
        self.dice_loss_weight_spinbox = create_double_spinbox((0.0, 1.0), step=0.05, default_value=0.5, decimals=2)
        self.iou_loss_weight_spinbox = create_double_spinbox((0.0, 1.0), step=0.05, default_value=0.5, decimals=2)
        self.loss_function_type.setVisible(False)
        self.dice_loss_weight_spinbox.setVisible(False)
        self.iou_loss_weight_spinbox.setVisible(False)
        self.loss_terms_widget = self._build_loss_terms_widget()
        self.loss_terms_groupbox = QGroupBox('')
        loss_terms_group_layout = QVBoxLayout(self.loss_terms_groupbox)
        loss_terms_group_layout.setContentsMargins(8, 8, 8, 8)
        loss_terms_group_layout.setSpacing(6)
        loss_terms_group_layout.addWidget(self.loss_terms_widget)

        self.learning_rate_spinbox = create_double_spinbox(
            (MIN_LEARNING_RATE, MAX_LEARNING_RATE), step=1e-4, default_value=1e-3, decimals=6
        )
        self.weight_decay_spinbox = create_double_spinbox(
            (MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY), step=1e-4, default_value=0.0, decimals=6
        )

        self.train_batch_spinbox = create_spinbox((MIN_BATCH, MAX_BATCH), step=1, default_value=16)
        self.dataloader_num_workers_spinbox = create_spinbox(
            (MIN_DATALOADER_WORKERS, MAX_DATALOADER_WORKERS),
            step=1,
            default_value=-1,
        )
        self.dataloader_num_workers_spinbox.setSpecialValueText('auto')
        self.recognition_batch_spinbox = create_spinbox((MIN_BATCH, MAX_BATCH), step=1, default_value=16)
        # Backward-compatible alias (legacy code expects training batch control).
        self.batch_spinbox = self.train_batch_spinbox
        self.overlap_spinbox = create_spinbox((MIN_OVERLAP, MAX_OVERLAP), step=4, default_value=8)
        self.recognition_jpeg_quality_spinbox = create_spinbox(
            (MIN_JPEG_QUALITY, MAX_JPEG_QUALITY),
            step=1,
            default_value=95,
        )
        self.recognition_multiprocessing_check_box = QCheckBox('')
        self.recognition_multiprocessing_check_box.setChecked(True)
        self.recognition_binarize_output_check_box = QCheckBox('')
        self.recognition_binarize_output_check_box.setChecked(True)
        self.recognition_use_auto_threshold_check_box = QCheckBox('')
        self.recognition_use_auto_threshold_check_box.setChecked(True)
        self.recognition_threshold_spinbox = create_double_spinbox(
            (MIN_RECOGNITION_THRESHOLD, MAX_RECOGNITION_THRESHOLD),
            step=0.05,
            default_value=0.5,
            decimals=2,
        )
        self.recognition_tta_check_box = QCheckBox('')
        self.recognition_tta_check_box.setChecked(False)
        self.confidence_save_mode_combo = NoWheelComboBox()
        for value in CONFIDENCE_SAVE_MODES:
            self.confidence_save_mode_combo.addItem(value, value)
        self.recognition_postprocess_check_box = QCheckBox('')
        self.recognition_postprocess_kernel_size_spinbox = create_spinbox(
            (MIN_POSTPROCESS_KERNEL_SIZE, MAX_POSTPROCESS_KERNEL_SIZE),
            step=2,
            default_value=3,
        )
        self.log_update_frequency_spinbox = create_spinbox(
            (MIN_LOG_UPDATE_FREQUENCY, MAX_LOG_UPDATE_FREQUENCY),
            step=1,
            default_value=0,
        )
        self.multi_gpu_mode_combo = NoWheelComboBox()
        self.multi_gpu_mode_combo.addItems(list(MULTI_GPU_MODES))
        # Backward-compatible alias used by tests/older code paths.
        self.multi_gpu_check_box = self.multi_gpu_mode_combo
        self.torch_compile_check_box = QCheckBox('')
        self.warmup_check_box = QCheckBox('')
        self.warmup_epochs_spinbox = create_spinbox((MIN_WARMUP_EPOCHS, MAX_WARMUP_EPOCHS), step=1, default_value=3)
        self.warmup_start_factor_spinbox = create_double_spinbox(
            (MIN_WARMUP_START_FACTOR, MAX_WARMUP_START_FACTOR),
            step=0.01,
            default_value=0.1,
            decimals=3,
        )
        self.scheduler_type_combo = NoWheelComboBox()
        for value in SCHEDULER_NAMES:
            self.scheduler_type_combo.addItem(value, value)
        self.scheduler_plateau_factor_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_FACTOR, MAX_SCHEDULER_FACTOR),
            step=0.05,
            default_value=0.5,
            decimals=2,
        )
        self.scheduler_plateau_patience_spinbox = create_spinbox(
            (MIN_SCHEDULER_PATIENCE, MAX_SCHEDULER_PATIENCE),
            step=1,
            default_value=3,
        )
        self.scheduler_plateau_threshold_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_THRESHOLD, MAX_SCHEDULER_THRESHOLD),
            step=1e-4,
            default_value=1e-4,
            decimals=6,
        )
        self.scheduler_plateau_min_lr_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_MIN_LR, MAX_SCHEDULER_MIN_LR),
            step=1e-5,
            default_value=1e-6,
            decimals=6,
        )
        self.scheduler_plateau_cooldown_spinbox = create_spinbox(
            (MIN_SCHEDULER_COOLDOWN, MAX_SCHEDULER_COOLDOWN),
            step=1,
            default_value=0,
        )
        self.scheduler_cosine_t_max_spinbox = create_spinbox(
            (MIN_SCHEDULER_T_MAX, MAX_SCHEDULER_T_MAX),
            step=1,
            default_value=10,
        )
        self.scheduler_cosine_eta_min_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_MIN_LR, MAX_SCHEDULER_MIN_LR),
            step=1e-5,
            default_value=1e-6,
            decimals=6,
        )
        self.scheduler_one_cycle_max_lr_spinbox = create_double_spinbox(
            (MIN_LEARNING_RATE, MAX_LEARNING_RATE),
            step=1e-4,
            default_value=1e-3,
            decimals=6,
        )
        self.scheduler_one_cycle_pct_start_spinbox = create_double_spinbox(
            (0.0, 1.0),
            step=0.05,
            default_value=0.3,
            decimals=2,
        )
        self.scheduler_one_cycle_anneal_strategy_combo = NoWheelComboBox()
        for value in ONE_CYCLE_ANNEAL_STRATEGIES:
            self.scheduler_one_cycle_anneal_strategy_combo.addItem(value, value)
        self.scheduler_one_cycle_div_factor_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_DIV_FACTOR, MAX_SCHEDULER_DIV_FACTOR),
            step=1.0,
            default_value=25.0,
            decimals=2,
        )
        self.scheduler_one_cycle_final_div_factor_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_FINAL_DIV_FACTOR, MAX_SCHEDULER_FINAL_DIV_FACTOR),
            step=100.0,
            default_value=10000.0,
            decimals=2,
        )
        self.scheduler_one_cycle_three_phase_check_box = QCheckBox('')
        self.scheduler_step_lr_step_size_spinbox = create_spinbox(
            (MIN_SCHEDULER_STEP_SIZE, MAX_SCHEDULER_STEP_SIZE),
            step=1,
            default_value=10,
        )
        self.scheduler_step_lr_gamma_spinbox = create_double_spinbox(
            (MIN_SCHEDULER_FACTOR, MAX_WARMUP_START_FACTOR),
            step=0.05,
            default_value=0.1,
            decimals=2,
        )
        self.hard_mining_check_box = QCheckBox('')
        self.hard_mining_strength_spinbox = create_double_spinbox(
            (MIN_HARD_MINING_STRENGTH, MAX_HARD_MINING_STRENGTH),
            step=0.1,
            default_value=2.0,
            decimals=2,
        )
        self.hard_mining_ema_alpha_spinbox = create_double_spinbox(
            (MIN_HARD_MINING_EMA_ALPHA, MAX_HARD_MINING_EMA_ALPHA),
            step=0.05,
            default_value=0.2,
            decimals=2,
        )
        self.hard_pixel_mining_check_box = QCheckBox('')
        self.hard_pixel_mining_ratio_spinbox = create_double_spinbox(
            (MIN_HARD_PIXEL_KEEP_RATIO, MAX_HARD_PIXEL_KEEP_RATIO),
            step=0.05,
            default_value=0.25,
            decimals=2,
        )
        self.skip_uniform_labels_check_box = QCheckBox('')
        self.rare_patch_oversampling_check_box = QCheckBox('')
        self.rare_patch_oversampling_factor_spinbox = create_spinbox(
            (MIN_RARE_PATCH_OVERSAMPLING_FACTOR, MAX_RARE_PATCH_OVERSAMPLING_FACTOR),
            step=1,
            default_value=2,
        )
        self.edit_rare_regions_button = QPushButton('')
        self.early_stopping_check_box = QCheckBox('')
        self.early_stopping_patience_spinbox = create_spinbox(
            (MIN_EARLY_STOPPING_PATIENCE, MAX_EARLY_STOPPING_PATIENCE),
            step=1,
            default_value=10,
        )
        self.early_stopping_min_delta_spinbox = create_double_spinbox(
            (MIN_EARLY_STOPPING_MIN_DELTA, MAX_EARLY_STOPPING_MIN_DELTA),
            step=1e-4,
            default_value=0.0,
            decimals=6,
        )
        self.restore_best_weights_check_box = QCheckBox('')
        self.restore_best_weights_check_box.setChecked(True)

        self.optimizer_groupbox, self.optimizer_form = _build_subgroup()
        self.precision_loss_groupbox, self.precision_loss_form = _build_subgroup()
        self.rare_patch_groupbox, self.rare_patch_form = _build_subgroup()
        self.recognition_groupbox, self.recognition_form = _build_subgroup()
        self.runtime_groupbox, self.runtime_form = _build_subgroup()
        self.warmup_groupbox, self.warmup_form = _build_subgroup()
        self.scheduler_groupbox, self.scheduler_form = _build_subgroup()
        self.hard_mining_groupbox, self.hard_mining_form = _build_subgroup()
        self.early_stopping_groupbox, self.early_stopping_form = _build_subgroup()

        # Keep backward compatibility for code that may inspect this attr.
        self.nn_aux_form = self.optimizer_form

        self.optimizer_form.addRow(self.optimizer_presets_widget)
        (
            self.optimizer_advanced_groupbox,
            self.optimizer_advanced_layout,
            self.optimizer_advanced_content_widget,
        ) = self._create_collapsible_groupbox()
        self.optimizer_advanced_form = self._create_form_layout(margins=(0, 0, 0, 0))
        self.optimizer_advanced_layout.addLayout(self.optimizer_advanced_form)
        self._add_labeled_row(self.optimizer_advanced_form, self.learning_rate_spinbox, 'learning_rate')
        self._add_labeled_row(self.optimizer_advanced_form, self.weight_decay_spinbox, 'weight_decay')
        self.optimizer_form.addRow(self.optimizer_advanced_groupbox)

        self.precision_loss_form.addRow(self.loss_presets_widget)
        (
            self.loss_advanced_groupbox,
            self.loss_advanced_layout,
            self.loss_advanced_content_widget,
        ) = self._create_collapsible_groupbox()
        self.loss_advanced_form = self._create_form_layout(margins=(0, 0, 0, 0))
        self.loss_advanced_layout.addLayout(self.loss_advanced_form)
        self.loss_advanced_form.addRow(self.deep_supervision_check_box)
        self.loss_advanced_form.addRow(self.loss_terms_groupbox)
        self.precision_loss_form.addRow(self.loss_advanced_groupbox)

        self.rare_patch_form.addRow(self.skip_uniform_labels_check_box)
        self.rare_patch_form.addRow(self.rare_patch_oversampling_check_box)
        self._add_labeled_row(
            self.rare_patch_form,
            self.rare_patch_oversampling_factor_spinbox,
            'rare_patch_oversampling_factor',
        )
        self.rare_patch_form.addRow(self.edit_rare_regions_button)

        self._add_labeled_row(self.recognition_form, self.recognition_batch_spinbox, 'recognition_batch_size')
        self._add_labeled_row(self.recognition_form, self.overlap_spinbox, 'overlap')
        self._add_labeled_row(self.recognition_form, self.recognition_jpeg_quality_spinbox, 'recognition_jpeg_quality')
        self.recognition_form.addRow(self.recognition_multiprocessing_check_box)
        self.recognition_form.addRow(self.recognition_binarize_output_check_box)
        self.recognition_form.addRow(self.recognition_use_auto_threshold_check_box)
        self._add_labeled_row(self.recognition_form, self.recognition_threshold_spinbox, 'recognition_threshold')
        self.recognition_form.addRow(self.recognition_tta_check_box)
        self._add_labeled_row(self.recognition_form, self.confidence_save_mode_combo, 'confidence_save_mode')
        self.recognition_form.addRow(self.recognition_postprocess_check_box)
        self._add_labeled_row(
            self.recognition_form,
            self.recognition_postprocess_kernel_size_spinbox,
            'recognition_postprocess_kernel_size',
        )
        self._add_labeled_row(self.runtime_form, self.train_batch_spinbox, 'train_batch_size')
        self._add_labeled_row(self.runtime_form, self.dataloader_num_workers_spinbox, 'dataloader_num_workers')
        self._add_labeled_row(self.runtime_form, self.multi_gpu_mode_combo, 'multi_gpu')
        self._add_labeled_row(self.runtime_form, self.mixed_precision_type, 'mixed_precision')
        self._add_labeled_row(self.runtime_form, self.log_update_frequency_spinbox, 'log_update_frequency')
        self.runtime_form.addRow(self.torch_compile_check_box)

        self.warmup_form.addRow(self.warmup_check_box)
        self._add_labeled_row(self.warmup_form, self.warmup_epochs_spinbox, 'warmup_epochs')
        self._add_labeled_row(self.warmup_form, self.warmup_start_factor_spinbox, 'warmup_start_factor')

        self._add_labeled_row(self.scheduler_form, self.scheduler_type_combo, 'scheduler_name')
        self._add_labeled_row(self.scheduler_form, self.scheduler_plateau_factor_spinbox, 'scheduler_plateau_factor')
        self._add_labeled_row(self.scheduler_form, self.scheduler_plateau_patience_spinbox, 'scheduler_plateau_patience')
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_plateau_threshold_spinbox,
            'scheduler_plateau_threshold',
        )
        self._add_labeled_row(self.scheduler_form, self.scheduler_plateau_min_lr_spinbox, 'scheduler_plateau_min_lr')
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_plateau_cooldown_spinbox,
            'scheduler_plateau_cooldown',
        )
        self._add_labeled_row(self.scheduler_form, self.scheduler_cosine_t_max_spinbox, 'scheduler_cosine_t_max')
        self._add_labeled_row(self.scheduler_form, self.scheduler_cosine_eta_min_spinbox, 'scheduler_cosine_eta_min')
        self._add_labeled_row(self.scheduler_form, self.scheduler_one_cycle_max_lr_spinbox, 'scheduler_one_cycle_max_lr')
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_one_cycle_pct_start_spinbox,
            'scheduler_one_cycle_pct_start',
        )
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_one_cycle_anneal_strategy_combo,
            'scheduler_one_cycle_anneal_strategy',
        )
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_one_cycle_div_factor_spinbox,
            'scheduler_one_cycle_div_factor',
        )
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_one_cycle_final_div_factor_spinbox,
            'scheduler_one_cycle_final_div_factor',
        )
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_one_cycle_three_phase_check_box,
            'scheduler_one_cycle_three_phase',
        )
        self._add_labeled_row(
            self.scheduler_form,
            self.scheduler_step_lr_step_size_spinbox,
            'scheduler_step_lr_step_size',
        )
        self._add_labeled_row(self.scheduler_form, self.scheduler_step_lr_gamma_spinbox, 'scheduler_step_lr_gamma')

        self.hard_mining_form.addRow(self.hard_mining_check_box)
        self._add_labeled_row(self.hard_mining_form, self.hard_mining_strength_spinbox, 'hard_mining_strength')
        self._add_labeled_row(self.hard_mining_form, self.hard_mining_ema_alpha_spinbox, 'hard_mining_ema_alpha')
        self.hard_mining_form.addRow(self.hard_pixel_mining_check_box)
        self._add_labeled_row(self.hard_mining_form, self.hard_pixel_mining_ratio_spinbox, 'hard_pixel_mining_ratio')

        self.early_stopping_form.addRow(self.early_stopping_check_box)
        self._add_labeled_row(self.early_stopping_form, self.early_stopping_patience_spinbox, 'early_stop_patience')
        self._add_labeled_row(self.early_stopping_form, self.early_stopping_min_delta_spinbox, 'early_stop_min_delta')
        self.early_stopping_form.addRow(self.restore_best_weights_check_box)

        nn_sections_layout.addWidget(self.optimizer_groupbox)
        nn_sections_layout.addWidget(self.precision_loss_groupbox)
        nn_sections_layout.addWidget(self.rare_patch_groupbox)
        nn_sections_layout.addWidget(self.warmup_groupbox)
        nn_sections_layout.addWidget(self.scheduler_groupbox)
        nn_sections_layout.addWidget(self.hard_mining_groupbox)
        nn_sections_layout.addWidget(self.early_stopping_groupbox)
        nn_sections_layout.addWidget(self.runtime_groupbox)
        nn_sections_layout.addStretch(1)

        self._sync_active_optimizer_preset()
        self.warmup_check_box.toggled.connect(self._sync_warmup_controls)
        self.scheduler_type_combo.currentIndexChanged.connect(self._sync_scheduler_controls)
        self.hard_mining_check_box.toggled.connect(self._sync_hard_mining_controls)
        self.hard_pixel_mining_check_box.toggled.connect(self._sync_hard_mining_controls)
        self.early_stopping_check_box.toggled.connect(self._sync_early_stopping_controls)
        self.rare_patch_oversampling_check_box.toggled.connect(self._sync_rare_patch_oversampling_controls)
        self.recognition_binarize_output_check_box.toggled.connect(self._sync_recognition_output_controls)
        self.recognition_use_auto_threshold_check_box.toggled.connect(self._sync_recognition_output_controls)
        self.recognition_postprocess_check_box.toggled.connect(self._sync_recognition_output_controls)
        self._sync_warmup_controls(self.warmup_check_box.isChecked())
        self._sync_scheduler_controls()
        self._sync_loss_controls()
        self._sync_hard_mining_controls(self.hard_mining_check_box.isChecked())
        self._sync_early_stopping_controls(self.early_stopping_check_box.isChecked())
        self._sync_rare_patch_oversampling_controls(self.rare_patch_oversampling_check_box.isChecked())
        self._sync_recognition_output_controls()

    def _init_layout(self) -> None:
        self.train_patch_size_widget = create_size_widget(self.train_patch_x_size, self.train_patch_y_size)
        self.recognition_patch_size_widget = create_size_widget(
            self.recognition_patch_x_size,
            self.recognition_patch_y_size,
        )
        # Backward-compatible alias (legacy code expects training patch widget).
        self.sample_size_widget = self.train_patch_size_widget

        self.general_groupbox = QGroupBox('')
        self.general_form = self._create_form_layout()
        self.general_groupbox.setLayout(self.general_form)
        self._add_labeled_row(self.general_form, self.nn_model_type, 'model')
        self._add_labeled_row(self.general_form, self.epochs_spinbox, 'epochs')
        self._add_labeled_row(self.general_form, self.color_type, 'image_format')
        self.sync_patch_sizes_check_box = QCheckBox('')
        self.sync_patch_sizes_check_box.setChecked(True)
        self._add_labeled_row(self.general_form, self.sync_patch_sizes_check_box, 'sync_patch_sizes')
        self._add_labeled_row(self.general_form, self.train_patch_size_widget, 'train_patch_size')
        self._add_labeled_row(self.general_form, self.recognition_patch_size_widget, 'recognition_patch_size')
        self.model_variants_groupbox = QGroupBox('')
        self.model_variants_form = self._create_form_layout()
        self.model_variants_groupbox.setLayout(self.model_variants_form)
        self._add_labeled_row(self.model_variants_form, self.deprecated_model_type, 'deprecated_models')
        self._add_labeled_row(self.model_variants_form, self.experimental_model_type, 'experimental_models')

        self.augmentation_groupbox = QGroupBox('')
        self.augmentation_form = self._create_form_layout()
        self.augmentation_groupbox.setLayout(self.augmentation_form)
        self.shuffle_groupbox = QGroupBox('')
        self.shuffle_form = self._create_form_layout()
        self.shuffle_groupbox.setLayout(self.shuffle_form)
        self.shuffle_form.addRow(self.shuffle_frames_check_box)
        self.shuffle_form.addRow(self.shuffle_patches_in_frame_check_box)
        self.spatial_groupbox = QGroupBox('')
        self.spatial_form = self._create_form_layout()
        self.spatial_groupbox.setLayout(self.spatial_form)
        self.spatial_orientation_widget = QWidget()
        self.spatial_orientation_layout = QGridLayout(self.spatial_orientation_widget)
        self.spatial_orientation_layout.setContentsMargins(0, 0, 0, 0)
        self.spatial_orientation_layout.setHorizontalSpacing(FIELD_DESCRIPTION_ROW_SPACING)
        self.spatial_orientation_layout.setVerticalSpacing(FORM_VERTICAL_SPACING)
        self.spatial_orientation_layout.addWidget(self.horizontal_rotation, 0, 0)
        self.spatial_orientation_layout.addWidget(self.flip_x, 0, 1)
        self.spatial_orientation_layout.addWidget(self.vertical_rotation, 1, 0)
        self.spatial_orientation_layout.addWidget(self.flip_y, 1, 1)
        self.spatial_form.addRow(self.spatial_orientation_widget)
        self.spatial_form.addRow(self.random_crop_check_box)
        self.spatial_sampling_row_widget = QWidget()
        self.spatial_sampling_row_layout = QHBoxLayout(self.spatial_sampling_row_widget)
        self.spatial_sampling_row_layout.setContentsMargins(0, 0, 0, 0)
        self.spatial_sampling_row_layout.setSpacing(FIELD_DESCRIPTION_ROW_SPACING)
        self.spatial_sampling_row_layout.addWidget(self._field_with_description(self.shift_spinbox, 'shift'), 1)
        self.spatial_sampling_row_layout.addWidget(
            self._field_with_description(self.crops_per_image_spinbox, 'crops_per_image'),
            1,
        )
        self.spatial_form.addRow(self.spatial_sampling_row_widget)
        self.spatial_form.addRow(self.scale_augmentation_check_box)
        self._add_labeled_row(self.spatial_form, self.scale_augmentation_strength_spinbox, 'scale_augmentation_strength')
        self.photometric_groupbox = QGroupBox('')
        self.photometric_form = self._create_form_layout()
        self.photometric_groupbox.setLayout(self.photometric_form)
        self.photometric_form.addRow(self.additional_augmentation_check_box)
        self._add_labeled_row(self.photometric_form, self.augmentation_brightness_spinbox, 'augmentation_brightness_strength')
        self._add_labeled_row(self.photometric_form, self.augmentation_contrast_spinbox, 'augmentation_contrast_strength')
        self._add_labeled_row(self.photometric_form, self.augmentation_gamma_spinbox, 'augmentation_gamma_strength')
        self._add_labeled_row(self.photometric_form, self.augmentation_noise_probability_spinbox, 'augmentation_noise_probability')
        self._add_labeled_row(self.photometric_form, self.augmentation_noise_sigma_spinbox, 'augmentation_noise_sigma')
        self._add_labeled_row(self.photometric_form, self.augmentation_blur_probability_spinbox, 'augmentation_blur_probability')
        self._add_labeled_row(self.photometric_form, self.augmentation_blur_radius_spinbox, 'augmentation_blur_radius')
        self.synthetic_defect_generator_groupbox = QGroupBox('')
        self.synthetic_defect_generator_form = self._create_form_layout()
        self.synthetic_defect_generator_groupbox.setLayout(self.synthetic_defect_generator_form)
        self.synthetic_defect_generator_form.addRow(self.synthetic_defect_generator_check_box)
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.synthetic_topology_domain_combo,
            'synthetic_topology_domain',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.pcb_topology_family_combo,
            'pcb_topology_family',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.ic_topology_family_combo,
            'ic_topology_family',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.synthetic_dataset_factor_spinbox,
            'synthetic_dataset_factor',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.synthetic_image_size_widget,
            'synthetic_image_size',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.synthetic_trace_count_range_widget,
            'synthetic_trace_count',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.synthetic_segment_count_range_widget,
            'synthetic_segment_count',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.synthetic_trace_half_width_range_widget,
            'synthetic_trace_half_width',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.synthetic_background_noise_sigma_range_widget,
            'synthetic_background_noise_sigma',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.synthetic_trace_noise_sigma_range_widget,
            'synthetic_trace_noise_sigma',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.pcb_defects_probability_spinbox,
            'pcb_defects_probability',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.pcb_defects_min_count_spinbox,
            'pcb_defects_min_count',
        )
        self._add_labeled_row(
            self.synthetic_defect_generator_form,
            self.pcb_defects_max_count_spinbox,
            'pcb_defects_max_count',
        )
        for defect_name, label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self._add_labeled_row(
                self.synthetic_defect_generator_form,
                self.pcb_defect_type_checkboxes[defect_name],
                f'pcb_{defect_name}',
            )
            self._add_labeled_row(
                self.synthetic_defect_generator_form,
                self.pcb_defect_type_spinboxes[defect_name],
                label_key,
            )
        for defect_name, label_key in IC_DEFECT_WEIGHT_FIELDS:
            self._add_labeled_row(
                self.synthetic_defect_generator_form,
                self.ic_defect_type_checkboxes[defect_name],
                f'ic_{defect_name}',
            )
            self._add_labeled_row(
                self.synthetic_defect_generator_form,
                self.ic_defect_type_spinboxes[defect_name],
                label_key,
            )
        self.cutout_groupbox = QGroupBox('')
        self.cutout_form = self._create_form_layout()
        self.cutout_groupbox.setLayout(self.cutout_form)
        self.cutout_form.addRow(self.cutout_check_box)
        self._add_labeled_row(self.cutout_form, self.cutout_probability_spinbox, 'cutout_probability')
        self._add_labeled_row(self.cutout_form, self.cutout_holes_spinbox, 'cutout_holes')
        self._add_labeled_row(self.cutout_form, self.cutout_size_ratio_spinbox, 'cutout_size_ratio')
        self.random_artifacts_groupbox = QGroupBox('')
        self.random_artifacts_form = self._create_form_layout()
        self.random_artifacts_groupbox.setLayout(self.random_artifacts_form)
        self.random_artifacts_form.addRow(self.random_artifacts_check_box)
        for artifact_name in RANDOM_ARTIFACT_TYPES:
            self.random_artifacts_form.addRow(self.random_artifact_type_checkboxes[artifact_name])
        self._add_labeled_row(
            self.random_artifacts_form,
            self.random_artifacts_probability_spinbox,
            'random_artifacts_probability',
        )
        self._add_labeled_row(
            self.random_artifacts_form,
            self.random_artifacts_count_spinbox,
            'random_artifacts_count',
        )
        self._add_labeled_row(
            self.random_artifacts_form,
            self.random_artifacts_size_ratio_spinbox,
            'random_artifacts_size_ratio',
        )
        self.mixup_groupbox = QGroupBox('')
        self.mixup_form = self._create_form_layout()
        self.mixup_groupbox.setLayout(self.mixup_form)
        self.mixup_form.addRow(self.mixup_check_box)
        self._add_labeled_row(self.mixup_form, self.mixup_probability_spinbox, 'mixup_probability')
        self._add_labeled_row(self.mixup_form, self.mixup_alpha_spinbox, 'mixup_alpha')
        self.pcb_defects_groupbox = QGroupBox('')
        self.pcb_defects_form = self._create_form_layout()
        self.pcb_defects_groupbox.setLayout(self.pcb_defects_form)
        self.additional_augmentation_check_box.toggled.connect(self._sync_augmentation_controls)
        self.cutout_check_box.toggled.connect(self._sync_training_augmentation_controls)
        self.random_artifacts_check_box.toggled.connect(self._sync_training_augmentation_controls)
        for checkbox in self.random_artifact_type_checkboxes.values():
            checkbox.toggled.connect(self._sync_training_augmentation_controls)
        self.mixup_check_box.toggled.connect(self._sync_training_augmentation_controls)
        self.synthetic_defect_generator_check_box.toggled.connect(self._sync_synthetic_defect_generator_controls)
        self.synthetic_topology_domain_combo.currentIndexChanged.connect(self._sync_synthetic_domain_controls)
        self.pcb_defects_min_count_spinbox.valueChanged.connect(self._sync_pcb_defect_count_bounds)
        self.pcb_defects_max_count_spinbox.valueChanged.connect(self._sync_pcb_defect_count_bounds)
        for checkbox in self.pcb_defect_type_checkboxes.values():
            checkbox.toggled.connect(self._sync_synthetic_defect_generator_controls)
        for checkbox in self.ic_defect_type_checkboxes.values():
            checkbox.toggled.connect(self._sync_synthetic_defect_generator_controls)
        self.augmentation_blur_probability_spinbox.valueChanged.connect(
            lambda *_args, **_kwargs: self._sync_augmentation_controls(self.additional_augmentation_check_box.isChecked())
        )
        self._sync_synthetic_domain_controls()
        self._sync_augmentation_controls(self.additional_augmentation_check_box.isChecked())
        self._sync_synthetic_defect_generator_controls(self.synthetic_defect_generator_check_box.isChecked())
        self._sync_training_augmentation_controls()

        self.validation_groupbox = QGroupBox('')
        self.validation_form = self._create_form_layout()
        self.validation_groupbox.setLayout(self.validation_form)
        self.validation_form.addRow(self.validation_check_box)
        self._add_labeled_row(self.validation_form, self.validation_mode_combo, 'validation_source')
        self._add_labeled_row(self.validation_form, self.validation_spinbox, 'validation_percent')
        self._add_labeled_row(self.validation_form, self.validation_image_path_label, 'validation_image_path')
        self._add_labeled_row(self.validation_form, self.validation_label_path_label, 'validation_label_path')
        self.validation_form.addRow(self.save_validation_binary_images_check_box)

        self.main_form = self.general_form
        self.reset_defaults_button = QPushButton('')

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setDocumentMode(True)

        self.training_page = QWidget()
        self.training_page_layout = QVBoxLayout(self.training_page)
        self.training_page_layout.setContentsMargins(0, 0, 0, 0)
        self.training_page_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        self.training_page_layout.addWidget(self.samples_number)
        self.training_page_layout.addWidget(self.general_groupbox)
        self.training_page_layout.addWidget(self.spatial_groupbox)
        self.training_page_layout.addWidget(self.prepare_samples_groupbox)
        self.training_page_layout.addWidget(self.rare_patch_groupbox)
        self.training_page_layout.addWidget(self.optimizer_groupbox)
        self.training_page_layout.addWidget(self.precision_loss_groupbox)
        self.expert_groupbox = QGroupBox('')
        self.expert_groupbox.setCheckable(True)
        self.expert_groupbox.setChecked(False)
        self.expert_groupbox_layout = QVBoxLayout(self.expert_groupbox)
        self.expert_groupbox_layout.setContentsMargins(*CONTENT_LAYOUT_MARGINS)
        self.expert_groupbox_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        self.expert_content_widget = QWidget()
        self.expert_content_layout = QVBoxLayout(self.expert_content_widget)
        self.expert_content_layout.setContentsMargins(0, 0, 0, 0)
        self.expert_content_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        self.expert_content_layout.addWidget(self.sample_type_groupbox)
        self.expert_content_layout.addWidget(self.validation_groupbox)
        self.expert_content_layout.addWidget(self.shuffle_groupbox)
        self.expert_content_layout.addWidget(self.photometric_groupbox)
        self.expert_content_layout.addWidget(self.cutout_groupbox)
        self.expert_content_layout.addWidget(self.random_artifacts_groupbox)
        self.expert_content_layout.addWidget(self.synthetic_defect_generator_groupbox)
        self.expert_content_layout.addWidget(self.mixup_groupbox)
        self.expert_content_layout.addWidget(self.augmentation_preview_button)
        self.expert_content_layout.addWidget(self.model_variants_groupbox)
        self.expert_content_layout.addWidget(self.warmup_groupbox)
        self.expert_content_layout.addWidget(self.scheduler_groupbox)
        self.expert_content_layout.addWidget(self.hard_mining_groupbox)
        self.expert_content_layout.addWidget(self.early_stopping_groupbox)
        self.expert_content_layout.addWidget(self.runtime_groupbox)
        self.expert_groupbox_layout.addWidget(self.expert_content_widget)
        self.training_page_layout.addWidget(self.expert_groupbox)
        self.training_page_layout.addStretch(1)

        self.recognition_page = QWidget()
        self.recognition_page_layout = QVBoxLayout(self.recognition_page)
        self.recognition_page_layout.setContentsMargins(0, 0, 0, 0)
        self.recognition_page_layout.setSpacing(CONTENT_LAYOUT_SPACING)
        self.recognition_page_layout.addWidget(self.recognition_groupbox)
        self.recognition_page_layout.addStretch(1)

        self._page_indexes = {
            'training': self.settings_tabs.addTab(self.training_page, ''),
            'recognition': self.settings_tabs.addTab(self.recognition_page, ''),
        }

        layout = QVBoxLayout()
        layout.setContentsMargins(*CONTENT_LAYOUT_MARGINS)
        layout.setSpacing(CONTENT_LAYOUT_SPACING)
        layout.addWidget(self.settings_tabs)

        reset_row = QHBoxLayout()
        reset_row.setContentsMargins(0, 0, 0, 0)
        reset_row.addStretch(1)
        reset_row.addWidget(self.reset_defaults_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(reset_row)
        layout.addStretch(1)

        self.general_form.setAlignment(self.nn_model_type, Qt.AlignmentFlag.AlignRight)
        self.general_form.setAlignment(self.epochs_spinbox, Qt.AlignmentFlag.AlignRight)
        self.general_form.setAlignment(self.color_type, Qt.AlignmentFlag.AlignRight)
        self.model_variants_form.setAlignment(self.deprecated_model_type, Qt.AlignmentFlag.AlignRight)
        self.model_variants_form.setAlignment(self.experimental_model_type, Qt.AlignmentFlag.AlignRight)
        self.validation_form.setAlignment(self.validation_mode_combo, Qt.AlignmentFlag.AlignRight)
        self.validation_form.setAlignment(self.validation_spinbox, Qt.AlignmentFlag.AlignRight)
        self.recognition_form.setAlignment(self.recognition_batch_spinbox, Qt.AlignmentFlag.AlignRight)
        self.recognition_form.setAlignment(self.overlap_spinbox, Qt.AlignmentFlag.AlignRight)
        self.recognition_form.setAlignment(self.recognition_jpeg_quality_spinbox, Qt.AlignmentFlag.AlignRight)
        self.expert_groupbox.toggled.connect(self._sync_expert_groupbox_controls)
        self._sync_expert_groupbox_controls(self.expert_groupbox.isChecked())

        self._content_widget.setLayout(layout)

    def _apply_localized_texts(self) -> None:
        apply_settings_panel_texts(self)

    def set_ui_language(self, language: str) -> None:
        active_language = set_global_ui_language(language)
        self._texts = get_ui_section('settings_panel')
        self._apply_localized_texts()
        self._sync_validation_path_labels()
        if self._sample_count_pending:
            self.set_samples_count_loading()
        else:
            self.set_samples_count(self._sample_count_value)
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
            return {str(key): SettingsPanel._clone_plain_payload(item) for key, item in value.items()}
        if hasattr(value, '__dict__'):
            return {
                str(key): SettingsPanel._clone_plain_payload(item)
                for key, item in vars(value).items()
                if not str(key).startswith('_')
            }
        if isinstance(value, list):
            return [SettingsPanel._clone_plain_payload(item) for item in value]
        if isinstance(value, tuple):
            return [SettingsPanel._clone_plain_payload(item) for item in value]
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
        if SettingsPanel._config_value_matches_default(value, default):
            node.pop(leaf_key, None)
        else:
            node[leaf_key] = value
        for parent, key in reversed(parents):
            nested = parent.get(key)
            if isinstance(nested, dict) and not nested:
                parent.pop(key, None)
            else:
                break

    def set_synthetic_defect_generator_config(self, config: Any) -> None:
        self._synthetic_defect_generator_payload = (
            self._clone_plain_payload(config) if config is not None else {}
        )
        resolved = build_synthetic_defect_generator_parameters(config)
        self.synthetic_defect_generator_check_box.setChecked(bool(resolved.enabled))
        self.synthetic_dataset_factor_spinbox.setValue(float(resolved.epoch_size_factor))
        self.synthetic_image_width_spinbox.setValue(int(resolved.image_size_xy[0]))
        self.synthetic_image_height_spinbox.setValue(int(resolved.image_size_xy[1]))
        self.synthetic_trace_count_min_spinbox.setValue(int(resolved.trace_count_range[0]))
        self.synthetic_trace_count_max_spinbox.setValue(int(resolved.trace_count_range[1]))
        self.synthetic_segment_count_min_spinbox.setValue(int(resolved.segment_count_range[0]))
        self.synthetic_segment_count_max_spinbox.setValue(int(resolved.segment_count_range[1]))
        self.synthetic_trace_half_width_min_spinbox.setValue(int(resolved.trace_half_width_range[0]))
        self.synthetic_trace_half_width_max_spinbox.setValue(int(resolved.trace_half_width_range[1]))
        self.synthetic_background_noise_sigma_min_spinbox.setValue(float(resolved.background_noise_sigma_range[0]))
        self.synthetic_background_noise_sigma_max_spinbox.setValue(float(resolved.background_noise_sigma_range[1]))
        self.synthetic_trace_noise_sigma_min_spinbox.setValue(float(resolved.trace_noise_sigma_range[0]))
        self.synthetic_trace_noise_sigma_max_spinbox.setValue(float(resolved.trace_noise_sigma_range[1]))
        self.set_synthetic_topology_domain_value(str(resolved.topology_domain))
        payload = self._synthetic_defect_generator_payload if isinstance(self._synthetic_defect_generator_payload, dict) else {}
        self.set_combo_value(
            self.pcb_topology_family_combo,
            str(
                payload.get(
                    'pcb_topology_family',
                    resolved.topology_family if resolved.topology_domain == 'pcb' else PCB_TOPOLOGY_FAMILIES[0],
                )
            ),
        )
        self.set_combo_value(
            self.ic_topology_family_combo,
            str(
                payload.get(
                    'ic_topology_family',
                    resolved.topology_family if resolved.topology_domain == 'ic' else IC_TOPOLOGY_FAMILIES[0],
                )
            ),
        )
        active_defects = resolved.ic_defects if resolved.topology_domain == 'ic' else resolved.pcb_defects
        self.pcb_defects_probability_spinbox.setValue(float(active_defects.defect_probability))
        self.pcb_defects_min_count_spinbox.setValue(int(active_defects.min_defects))
        self.pcb_defects_max_count_spinbox.setValue(int(active_defects.max_defects))
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self.pcb_defect_type_checkboxes[defect_name].setChecked(
                float(resolved.pcb_defects.defect_probabilities.get(defect_name, 1.0)) > 0.0
            )
            self.pcb_defect_type_spinboxes[defect_name].setValue(
                int(round(float(resolved.pcb_defects.defect_severities.get(defect_name, 0.5)) * 100.0))
            )
        for defect_name, _label_key in IC_DEFECT_WEIGHT_FIELDS:
            self.ic_defect_type_checkboxes[defect_name].setChecked(
                float(resolved.ic_defects.defect_probabilities.get(defect_name, 1.0)) > 0.0
            )
            self.ic_defect_type_spinboxes[defect_name].setValue(
                int(round(float(resolved.ic_defects.defect_severities.get(defect_name, 0.5)) * 100.0))
            )
        self._sync_synthetic_domain_controls()
        self._sync_synthetic_defect_generator_controls(self.synthetic_defect_generator_check_box.isChecked())

    def get_synthetic_defect_generator_config(self) -> dict[str, Any]:
        defaults = build_synthetic_defect_generator_parameters(None)
        payload = self._clone_plain_payload(self._synthetic_defect_generator_payload)
        if not isinstance(payload, dict):
            payload = {}

        self._set_optional_config_value(
            payload,
            ('enabled',),
            bool(self.synthetic_defect_generator_check_box.isChecked()),
            bool(defaults.enabled),
        )
        self._set_optional_config_value(
            payload,
            ('epoch_size_factor',),
            float(self.synthetic_dataset_factor_spinbox.value()),
            float(defaults.epoch_size_factor),
        )
        self._set_optional_config_value(
            payload,
            ('image_size_xy',),
            [
                int(self.synthetic_image_width_spinbox.value()),
                int(self.synthetic_image_height_spinbox.value()),
            ],
            list(defaults.image_size_xy),
        )
        topology_domain = self.get_synthetic_topology_domain_value()
        pcb_topology_family = self.get_combo_value(self.pcb_topology_family_combo)
        ic_topology_family = self.get_combo_value(self.ic_topology_family_combo)
        active_defaults = defaults.ic_defects if topology_domain == 'ic' else defaults.pcb_defects
        active_fields = IC_DEFECT_WEIGHT_FIELDS if topology_domain == 'ic' else PCB_DEFECT_WEIGHT_FIELDS
        active_sliders = self.ic_defect_type_spinboxes if topology_domain == 'ic' else self.pcb_defect_type_spinboxes
        active_checkboxes = self.ic_defect_type_checkboxes if topology_domain == 'ic' else self.pcb_defect_type_checkboxes

        self._set_optional_config_value(
            payload,
            ('topology_domain',),
            topology_domain,
            str(defaults.topology_domain),
        )
        self._set_optional_config_value(
            payload,
            ('pcb_topology_family',),
            pcb_topology_family,
            PCB_TOPOLOGY_FAMILIES[0],
        )
        self._set_optional_config_value(
            payload,
            ('ic_topology_family',),
            ic_topology_family,
            IC_TOPOLOGY_FAMILIES[0],
        )
        self._set_optional_config_value(
            payload,
            ('topology_family',),
            ic_topology_family if topology_domain == 'ic' else pcb_topology_family,
            str(defaults.topology_family),
        )
        trace_count_min = int(self.synthetic_trace_count_min_spinbox.value())
        trace_count_max = int(self.synthetic_trace_count_max_spinbox.value())
        if trace_count_min > trace_count_max:
            trace_count_min, trace_count_max = trace_count_max, trace_count_min
        segment_count_min = int(self.synthetic_segment_count_min_spinbox.value())
        segment_count_max = int(self.synthetic_segment_count_max_spinbox.value())
        if segment_count_min > segment_count_max:
            segment_count_min, segment_count_max = segment_count_max, segment_count_min
        trace_half_width_min = int(self.synthetic_trace_half_width_min_spinbox.value())
        trace_half_width_max = int(self.synthetic_trace_half_width_max_spinbox.value())
        if trace_half_width_min > trace_half_width_max:
            trace_half_width_min, trace_half_width_max = trace_half_width_max, trace_half_width_min
        background_noise_sigma_min = float(self.synthetic_background_noise_sigma_min_spinbox.value())
        background_noise_sigma_max = float(self.synthetic_background_noise_sigma_max_spinbox.value())
        if background_noise_sigma_min > background_noise_sigma_max:
            background_noise_sigma_min, background_noise_sigma_max = background_noise_sigma_max, background_noise_sigma_min
        trace_noise_sigma_min = float(self.synthetic_trace_noise_sigma_min_spinbox.value())
        trace_noise_sigma_max = float(self.synthetic_trace_noise_sigma_max_spinbox.value())
        if trace_noise_sigma_min > trace_noise_sigma_max:
            trace_noise_sigma_min, trace_noise_sigma_max = trace_noise_sigma_max, trace_noise_sigma_min
        self._set_optional_config_value(
            payload,
            ('trace_count_range',),
            [trace_count_min, trace_count_max],
            list(defaults.trace_count_range),
        )
        self._set_optional_config_value(
            payload,
            ('trace_count',),
            trace_count_max,
            int(defaults.trace_count),
        )
        self._set_optional_config_value(
            payload,
            ('segment_count_range',),
            [segment_count_min, segment_count_max],
            list(defaults.segment_count_range),
        )
        self._set_optional_config_value(
            payload,
            ('segment_count',),
            segment_count_max,
            int(defaults.segment_count),
        )
        self._set_optional_config_value(
            payload,
            ('trace_half_width_range',),
            [trace_half_width_min, trace_half_width_max],
            list(defaults.trace_half_width_range),
        )
        self._set_optional_config_value(
            payload,
            ('trace_half_width',),
            trace_half_width_max,
            int(defaults.trace_half_width),
        )
        self._set_optional_config_value(
            payload,
            ('background_noise_sigma_range',),
            [background_noise_sigma_min, background_noise_sigma_max],
            list(defaults.background_noise_sigma_range),
        )
        self._set_optional_config_value(
            payload,
            ('background_noise_sigma',),
            background_noise_sigma_max,
            float(defaults.background_noise_sigma),
        )
        self._set_optional_config_value(
            payload,
            ('trace_noise_sigma_range',),
            [trace_noise_sigma_min, trace_noise_sigma_max],
            list(defaults.trace_noise_sigma_range),
        )
        self._set_optional_config_value(
            payload,
            ('trace_noise_sigma',),
            trace_noise_sigma_max,
            float(defaults.trace_noise_sigma),
        )
        min_defects = int(self.pcb_defects_min_count_spinbox.value())
        max_defects = int(self.pcb_defects_max_count_spinbox.value())
        if min_defects > max_defects:
            min_defects, max_defects = max_defects, min_defects
        for group_name, group_defaults, field_defs, slider_map, checkbox_map in (
            ('pcb_defects', defaults.pcb_defects, PCB_DEFECT_WEIGHT_FIELDS, self.pcb_defect_type_spinboxes, self.pcb_defect_type_checkboxes),
            ('ic_defects', defaults.ic_defects, IC_DEFECT_WEIGHT_FIELDS, self.ic_defect_type_spinboxes, self.ic_defect_type_checkboxes),
        ):
            existing_group_payload = payload.get(group_name, {}) if isinstance(payload.get(group_name), dict) else {}
            existing_active_payload = payload.get('defects', {}) if isinstance(payload.get('defects'), dict) else {}
            self._set_optional_config_value(
                payload,
                (group_name, 'enabled'),
                bool(self.synthetic_defect_generator_check_box.isChecked()),
                bool(group_defaults.enabled),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'defect_probability'),
                float(self.pcb_defects_probability_spinbox.value()),
                float(group_defaults.defect_probability),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'min_defects'),
                min_defects,
                int(group_defaults.min_defects),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'max_defects'),
                max_defects,
                int(group_defaults.max_defects),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'use_input_mask'),
                True,
                bool(group_defaults.use_input_mask),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'use_defect_mask_as_label'),
                False,
                bool(group_defaults.use_defect_mask_as_label),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'max_attempts_per_defect'),
                int(existing_group_payload.get('max_attempts_per_defect', existing_active_payload.get('max_attempts_per_defect', group_defaults.max_attempts_per_defect))),
                int(group_defaults.max_attempts_per_defect),
            )
            self._set_optional_config_value(
                payload,
                (group_name, 'min_component_area'),
                int(existing_group_payload.get('min_component_area', existing_active_payload.get('min_component_area', group_defaults.min_component_area))),
                int(group_defaults.min_component_area),
            )
            for defect_name, _label_key in field_defs:
                self._set_optional_config_value(
                    payload,
                    (group_name, 'defect_probabilities', defect_name),
                    1.0 if bool(checkbox_map[defect_name].isChecked()) else 0.0,
                    float(group_defaults.defect_probabilities.get(defect_name, 1.0)),
                )
                self._set_optional_config_value(
                    payload,
                    (group_name, 'defect_severities', defect_name),
                    float(slider_map[defect_name].value()) / 100.0,
                    float(group_defaults.defect_severities.get(defect_name, 0.5)),
                )
        self._set_optional_config_value(
            payload,
            ('defects', 'enabled'),
            bool(self.synthetic_defect_generator_check_box.isChecked()),
            bool(active_defaults.enabled),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'defect_probability'),
            float(self.pcb_defects_probability_spinbox.value()),
            float(active_defaults.defect_probability),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'min_defects'),
            min_defects,
            int(active_defaults.min_defects),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'max_defects'),
            max_defects,
            int(active_defaults.max_defects),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'use_input_mask'),
            True,
            bool(active_defaults.use_input_mask),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'use_defect_mask_as_label'),
            False,
            bool(active_defaults.use_defect_mask_as_label),
        )
        existing_active_payload = payload.get('defects', {}) if isinstance(payload.get('defects'), dict) else {}
        self._set_optional_config_value(
            payload,
            ('defects', 'max_attempts_per_defect'),
            int(existing_active_payload.get('max_attempts_per_defect', active_defaults.max_attempts_per_defect)),
            int(active_defaults.max_attempts_per_defect),
        )
        self._set_optional_config_value(
            payload,
            ('defects', 'min_component_area'),
            int(existing_active_payload.get('min_component_area', active_defaults.min_component_area)),
            int(active_defaults.min_component_area),
        )
        for defect_name, _label_key in active_fields:
            self._set_optional_config_value(
                payload,
                ('defects', 'defect_probabilities', defect_name),
                1.0 if bool(active_checkboxes[defect_name].isChecked()) else 0.0,
                float(active_defaults.defect_probabilities.get(defect_name, 1.0)),
            )
            self._set_optional_config_value(
                payload,
                ('defects', 'defect_severities', defect_name),
                float(active_sliders[defect_name].value()) / 100.0,
                float(active_defaults.defect_severities.get(defect_name, 0.5)),
            )
        self._synthetic_defect_generator_payload = self._clone_plain_payload(payload)
        return payload

    def set_tech_aug_config(self, config: Any) -> None:
        self._tech_aug_config_payload = self._clone_plain_payload(config) if config is not None else {}
        resolved = build_tech_augmentation_config(config)
        self.tech_augmentation_check_box.setChecked(bool(resolved.enabled))
        self.tech_aug_min_operations_spinbox.setValue(int(resolved.min_operations))
        self.tech_aug_max_operations_spinbox.setValue(int(resolved.max_operations))
        self.tech_augmentation_debug_pair_check_box.setChecked(bool(resolved.debug_return_pair))
        self.tech_aug_max_changed_pixels_ratio_spinbox.setValue(float(resolved.max_changed_pixels_ratio))
        self.tech_aug_max_foreground_ratio_delta_spinbox.setValue(float(resolved.max_foreground_ratio_delta))
        self.tech_aug_global_width_probability_spinbox.setValue(float(resolved.global_width.probability))
        self.tech_aug_scale_rethreshold_probability_spinbox.setValue(
            float(resolved.scale_rethreshold.probability)
        )
        self.tech_aug_blur_threshold_probability_spinbox.setValue(float(resolved.blur_threshold.probability))
        self.tech_aug_boundary_aware_probability_spinbox.setValue(float(resolved.boundary_aware.probability))
        self.tech_aug_local_morphology_probability_spinbox.setValue(float(resolved.local_morphology.probability))
        self.tech_aug_gap_variation_probability_spinbox.setValue(float(resolved.gap_variation.probability))
        self._sync_tech_augmentation_controls(self.tech_augmentation_check_box.isChecked())

    def get_tech_aug_config(self) -> dict[str, Any]:
        defaults = build_tech_augmentation_config(None)
        payload = self._clone_plain_payload(self._tech_aug_config_payload)
        if not isinstance(payload, dict):
            payload = {}

        self._set_optional_config_value(
            payload,
            ('enabled',),
            bool(self.tech_augmentation_check_box.isChecked()),
            bool(defaults.enabled),
        )
        min_operations = int(self.tech_aug_min_operations_spinbox.value())
        max_operations = int(self.tech_aug_max_operations_spinbox.value())
        if min_operations > max_operations:
            min_operations, max_operations = max_operations, min_operations
        self._set_optional_config_value(payload, ('min_operations',), min_operations, int(defaults.min_operations))
        self._set_optional_config_value(payload, ('max_operations',), max_operations, int(defaults.max_operations))
        self._set_optional_config_value(
            payload,
            ('debug_return_pair',),
            bool(self.tech_augmentation_debug_pair_check_box.isChecked()),
            bool(defaults.debug_return_pair),
        )
        self._set_optional_config_value(
            payload,
            ('max_changed_pixels_ratio',),
            float(self.tech_aug_max_changed_pixels_ratio_spinbox.value()),
            float(defaults.max_changed_pixels_ratio),
        )
        self._set_optional_config_value(
            payload,
            ('max_foreground_ratio_delta',),
            float(self.tech_aug_max_foreground_ratio_delta_spinbox.value()),
            float(defaults.max_foreground_ratio_delta),
        )
        self._set_optional_config_value(
            payload,
            ('global_width', 'probability'),
            float(self.tech_aug_global_width_probability_spinbox.value()),
            float(defaults.global_width.probability),
        )
        self._set_optional_config_value(
            payload,
            ('scale_rethreshold', 'probability'),
            float(self.tech_aug_scale_rethreshold_probability_spinbox.value()),
            float(defaults.scale_rethreshold.probability),
        )
        self._set_optional_config_value(
            payload,
            ('blur_threshold', 'probability'),
            float(self.tech_aug_blur_threshold_probability_spinbox.value()),
            float(defaults.blur_threshold.probability),
        )
        self._set_optional_config_value(
            payload,
            ('boundary_aware', 'probability'),
            float(self.tech_aug_boundary_aware_probability_spinbox.value()),
            float(defaults.boundary_aware.probability),
        )
        self._set_optional_config_value(
            payload,
            ('local_morphology', 'probability'),
            float(self.tech_aug_local_morphology_probability_spinbox.value()),
            float(defaults.local_morphology.probability),
        )
        self._set_optional_config_value(
            payload,
            ('gap_variation', 'probability'),
            float(self.tech_aug_gap_variation_probability_spinbox.value()),
            float(defaults.gap_variation.probability),
        )
        self._tech_aug_config_payload = self._clone_plain_payload(payload)
        return payload

    def set_pcb_defects_config(self, config: Any) -> None:
        self._pcb_defects_config_payload = self._clone_plain_payload(config) if config is not None else {}
        resolved = build_pcb_defect_parameters(config)
        self.pcb_defects_check_box.setChecked(bool(resolved.enabled))
        self.pcb_defects_probability_spinbox.setValue(float(resolved.defect_probability))
        self.pcb_defects_min_count_spinbox.setValue(int(resolved.min_defects))
        self.pcb_defects_max_count_spinbox.setValue(int(resolved.max_defects))
        self.pcb_defects_use_input_mask_check_box.setChecked(bool(resolved.use_input_mask))
        self.pcb_defects_use_defect_mask_as_label_check_box.setChecked(bool(resolved.use_defect_mask_as_label))
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self.pcb_defect_type_spinboxes[defect_name].setValue(
                int(round(float(resolved.defect_severities.get(defect_name, 0.5)) * 100.0))
            )
        self._sync_training_augmentation_controls()

    def get_pcb_defects_config(self) -> dict[str, Any]:
        defaults = build_pcb_defect_parameters(None)
        payload = self._clone_plain_payload(self._pcb_defects_config_payload)
        if not isinstance(payload, dict):
            payload = {}

        self._set_optional_config_value(
            payload,
            ('enabled',),
            bool(self.pcb_defects_check_box.isChecked()),
            bool(defaults.enabled),
        )
        min_defects = int(self.pcb_defects_min_count_spinbox.value())
        max_defects = int(self.pcb_defects_max_count_spinbox.value())
        if min_defects > max_defects:
            min_defects, max_defects = max_defects, min_defects
        self._set_optional_config_value(payload, ('defect_probability',), float(self.pcb_defects_probability_spinbox.value()), float(defaults.defect_probability))
        self._set_optional_config_value(payload, ('min_defects',), min_defects, int(defaults.min_defects))
        self._set_optional_config_value(payload, ('max_defects',), max_defects, int(defaults.max_defects))
        self._set_optional_config_value(
            payload,
            ('use_input_mask',),
            bool(self.pcb_defects_use_input_mask_check_box.isChecked()),
            bool(defaults.use_input_mask),
        )
        self._set_optional_config_value(
            payload,
            ('use_defect_mask_as_label',),
            bool(self.pcb_defects_use_defect_mask_as_label_check_box.isChecked()),
            bool(defaults.use_defect_mask_as_label),
        )
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self._set_optional_config_value(
                payload,
                ('defect_probabilities', defect_name),
                1.0 if bool(self.pcb_defects_check_box.isChecked()) else 0.0,
                float(defaults.defect_probabilities.get(defect_name, 1.0)),
            )
            self._set_optional_config_value(
                payload,
                ('defect_severities', defect_name),
                float(self.pcb_defect_type_spinboxes[defect_name].value()) / 100.0,
                float(defaults.defect_severities.get(defect_name, 0.5)),
            )
        self._pcb_defects_config_payload = self._clone_plain_payload(payload)
        return payload

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
            normalized_key = 'training'
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
        for page_key in ('training', 'recognition'):
            index = self._page_indexes.get(page_key)
            if index is None:
                continue
            if hasattr(self.settings_tabs, 'isTabVisible') and self.settings_tabs.isTabVisible(index):
                self.settings_tabs.setCurrentIndex(index)
                return

    def _sync_expert_groupbox_controls(self, expanded: bool) -> None:
        visible = bool(expanded)
        self.expert_content_widget.setVisible(visible)

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

    def _sync_hard_mining_controls(self, enabled: bool) -> None:
        sample_control_enabled = self._training_controls_applicable and bool(self.hard_mining_check_box.isChecked())
        pixel_control_enabled = self._training_controls_applicable and bool(self.hard_pixel_mining_check_box.isChecked())
        self._set_field_enabled(self.hard_mining_strength_spinbox, sample_control_enabled)
        self._set_field_enabled(self.hard_mining_ema_alpha_spinbox, sample_control_enabled)
        self._set_field_enabled(self.hard_pixel_mining_ratio_spinbox, pixel_control_enabled)

    def _sync_loss_controls(self, _index: int | None = None) -> None:
        total = loss_term_weight_sum(self.get_loss_term_weights())
        for loss_name in LOSS_FUNCTIONS:
            checkbox = self.loss_term_checkboxes[loss_name]
            spinbox = self.loss_term_spinboxes[loss_name]
            label = self.loss_term_labels[loss_name]
            spinbox.setMaximum(self._max_loss_weight_for(loss_name))
            checkbox.setEnabled(self._training_controls_applicable)
            label.setEnabled(self._training_controls_applicable)
            spinbox.setEnabled(self._training_controls_applicable and checkbox.isChecked())
        self.loss_formula_label.setEnabled(self._training_controls_applicable)
        formula_html = format_loss_formula_html(self.get_loss_term_weights())
        total_html = (
            "<span style=\"color:#666; font-size:11px;\">"
            f"&Sigma;&nbsp;n<sub>i</sub> = {total:.2f}/{MAX_LOSS_TERM_WEIGHT_SUM:.2f}"
            "</span>"
        )
        self.loss_formula_label.setText(f'{formula_html}<br>{total_html}')
        self._sync_active_loss_preset()

    def _sync_early_stopping_controls(self, enabled: bool) -> None:
        control_enabled = self._training_controls_applicable and bool(enabled)
        self._set_field_enabled(self.early_stopping_patience_spinbox, control_enabled)
        self._set_field_enabled(self.early_stopping_min_delta_spinbox, control_enabled)
        self.restore_best_weights_check_box.setEnabled(control_enabled)

    def _sync_rare_patch_oversampling_controls(self, _enabled: bool | None = None) -> None:
        online_mode_enabled = self._training_controls_applicable and bool(self.no_cut_dataset_type.isChecked())
        self.rare_patch_oversampling_check_box.setEnabled(online_mode_enabled)
        self.edit_rare_regions_button.setEnabled(online_mode_enabled)
        self._set_field_enabled(
            self.rare_patch_oversampling_factor_spinbox,
            online_mode_enabled and bool(self.rare_patch_oversampling_check_box.isChecked()),
        )

    def _sync_preprocess_controls(self, _enabled: bool | None = None) -> None:
        self._set_field_enabled(
            self.cut_corner_spinbox,
            self._training_controls_applicable and self.enable_crop_processing.isChecked(),
        )
        self._set_field_enabled(
            self.compression_factor_spinbox,
            self._training_controls_applicable and self.enable_resize_processing.isChecked(),
        )

    def _sync_augmentation_controls(self, enabled: bool) -> None:
        control_enabled = self._training_controls_applicable and bool(enabled)
        online_mode_enabled = self._training_controls_applicable and bool(self.no_cut_dataset_type.isChecked())
        random_crop_enabled = online_mode_enabled and bool(self.random_crop_check_box.isChecked())
        self.spatial_groupbox.setEnabled(self._training_controls_applicable)
        self.photometric_groupbox.setEnabled(self._training_controls_applicable)
        self._set_field_enabled(self.augmentation_brightness_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_contrast_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_gamma_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_noise_probability_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_noise_sigma_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_blur_probability_spinbox, control_enabled)
        self._set_field_enabled(
            self.augmentation_blur_radius_spinbox,
            control_enabled and float(self.augmentation_blur_probability_spinbox.value()) > 0.0,
        )
        self.random_crop_check_box.setEnabled(online_mode_enabled)
        self.scale_augmentation_check_box.setEnabled(online_mode_enabled)
        self._set_field_enabled(self.crops_per_image_spinbox, random_crop_enabled)
        self._set_field_enabled(
            self.shift_spinbox,
            self._training_controls_applicable and not random_crop_enabled,
        )
        self._set_field_enabled(
            self.scale_augmentation_strength_spinbox,
            online_mode_enabled and bool(self.scale_augmentation_check_box.isChecked()),
        )
        self._sync_tech_augmentation_controls(self.tech_augmentation_check_box.isChecked())
        self._sync_synthetic_defect_generator_controls(self.synthetic_defect_generator_check_box.isChecked())
        self._sync_training_augmentation_controls()

    def _sync_tech_augmentation_controls(self, _enabled: bool | None = None) -> None:
        training_enabled = self._training_controls_applicable
        tech_enabled = training_enabled and bool(self.tech_augmentation_check_box.isChecked())
        self.tech_augmentation_check_box.setEnabled(training_enabled)
        self.tech_augmentation_debug_pair_check_box.setEnabled(tech_enabled)
        self._set_field_enabled(self.tech_aug_min_operations_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_max_operations_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_max_changed_pixels_ratio_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_max_foreground_ratio_delta_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_global_width_probability_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_scale_rethreshold_probability_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_blur_threshold_probability_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_boundary_aware_probability_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_local_morphology_probability_spinbox, tech_enabled)
        self._set_field_enabled(self.tech_aug_gap_variation_probability_spinbox, tech_enabled)

    def _sync_synthetic_defect_generator_controls(self, _enabled: bool | None = None) -> None:
        training_enabled = self._training_controls_applicable
        synthetic_enabled = training_enabled and bool(self.synthetic_defect_generator_check_box.isChecked())
        current_domain = self.get_synthetic_topology_domain_value()
        self.synthetic_defect_generator_groupbox.setEnabled(training_enabled)
        self.synthetic_defect_generator_check_box.setEnabled(training_enabled)
        self._set_field_enabled(self.synthetic_topology_domain_combo, synthetic_enabled)
        self._set_field_enabled(
            self.pcb_topology_family_combo,
            synthetic_enabled and current_domain == 'pcb',
        )
        self._set_field_enabled(
            self.ic_topology_family_combo,
            synthetic_enabled and current_domain == 'ic',
        )
        self._set_field_enabled(self.synthetic_dataset_factor_spinbox, synthetic_enabled)
        self._set_field_enabled(self.synthetic_image_size_widget, synthetic_enabled)
        self._set_field_enabled(self.synthetic_trace_count_range_widget, synthetic_enabled)
        self._set_field_enabled(self.synthetic_segment_count_range_widget, synthetic_enabled)
        self._set_field_enabled(self.synthetic_trace_half_width_range_widget, synthetic_enabled)
        self._set_field_enabled(self.synthetic_background_noise_sigma_range_widget, synthetic_enabled)
        self._set_field_enabled(self.synthetic_trace_noise_sigma_range_widget, synthetic_enabled)
        self._set_field_enabled(self.pcb_defects_probability_spinbox, synthetic_enabled)
        self._set_field_enabled(self.pcb_defects_min_count_spinbox, synthetic_enabled)
        self._set_field_enabled(self.pcb_defects_max_count_spinbox, synthetic_enabled)
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self._set_field_enabled(
                self.pcb_defect_type_checkboxes[defect_name],
                synthetic_enabled and current_domain == 'pcb',
            )
            self._set_field_enabled(
                self.pcb_defect_type_spinboxes[defect_name],
                synthetic_enabled and current_domain == 'pcb' and bool(self.pcb_defect_type_checkboxes[defect_name].isChecked()),
            )
        for defect_name, _label_key in IC_DEFECT_WEIGHT_FIELDS:
            self._set_field_enabled(
                self.ic_defect_type_checkboxes[defect_name],
                synthetic_enabled and current_domain == 'ic',
            )
            self._set_field_enabled(
                self.ic_defect_type_spinboxes[defect_name],
                synthetic_enabled and current_domain == 'ic' and bool(self.ic_defect_type_checkboxes[defect_name].isChecked()),
            )
        self._sync_synthetic_domain_controls()

    def _sync_synthetic_domain_controls(self, _enabled: bool | None = None) -> None:
        is_ic = self.get_synthetic_topology_domain_value() == 'ic'
        self._set_field_visible(self.pcb_topology_family_combo, not is_ic)
        self._set_field_visible(self.ic_topology_family_combo, is_ic)
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self._set_field_visible(self.pcb_defect_type_checkboxes[defect_name], not is_ic)
            self._set_field_visible(self.pcb_defect_type_spinboxes[defect_name], not is_ic)
        for defect_name, _label_key in IC_DEFECT_WEIGHT_FIELDS:
            self._set_field_visible(self.ic_defect_type_checkboxes[defect_name], is_ic)
            self._set_field_visible(self.ic_defect_type_spinboxes[defect_name], is_ic)

    def _sync_recognition_output_controls(self, _enabled: bool | None = None) -> None:
        recognition_enabled = self._recognition_controls_applicable
        binarize_output = recognition_enabled and bool(self.recognition_binarize_output_check_box.isChecked())
        auto_threshold = binarize_output and bool(self.recognition_use_auto_threshold_check_box.isChecked())
        postprocess_enabled = binarize_output and bool(self.recognition_postprocess_check_box.isChecked())

        self.recognition_multiprocessing_check_box.setEnabled(recognition_enabled)
        self.recognition_binarize_output_check_box.setEnabled(recognition_enabled)
        self.recognition_use_auto_threshold_check_box.setEnabled(binarize_output)
        self._set_field_enabled(self.recognition_threshold_spinbox, binarize_output and not auto_threshold)
        self.recognition_tta_check_box.setEnabled(recognition_enabled)
        self._set_field_enabled(self.confidence_save_mode_combo, recognition_enabled)
        self.recognition_postprocess_check_box.setEnabled(binarize_output)
        self._set_field_enabled(self.recognition_postprocess_kernel_size_spinbox, postprocess_enabled)

    def get_confidence_export_mode_value(self) -> str:
        normalized = str(self.confidence_save_mode_combo.currentData() or self.confidence_save_mode_combo.currentText() or 'off')
        normalized = normalized.strip().lower()
        if normalized in CONFIDENCE_SAVE_MODES:
            return normalized
        if normalized == 'separate_grayscale':
            return 'model_output'
        return 'off'

    def get_confidence_save_mode_value(self) -> str:
        export_mode = self.get_confidence_export_mode_value()
        if export_mode == 'off':
            return 'off'
        return 'separate_grayscale'

    def is_confidence_tta_enabled(self) -> bool:
        return self.get_confidence_export_mode_value() == 'tta'

    def set_confidence_output_mode(self, confidence_save_mode: str | None, confidence_tta_enabled: bool = False) -> None:
        normalized_save_mode = str(confidence_save_mode or 'off').strip().lower()
        if normalized_save_mode in CONFIDENCE_SAVE_MODES:
            self.set_confidence_export_mode_value(normalized_save_mode)
            return
        if normalized_save_mode == 'off':
            self.set_confidence_export_mode_value('off')
            return
        if confidence_tta_enabled:
            self.set_confidence_export_mode_value('tta')
            return
        self.set_confidence_export_mode_value('model_output')

    def set_confidence_save_mode_value(self, value: str) -> None:
        normalized = str(value or 'off').strip().lower()
        if normalized in CONFIDENCE_SAVE_MODES:
            self.set_confidence_export_mode_value(normalized)
            return
        if normalized == 'off':
            self.set_confidence_export_mode_value('off')
            return
        self.set_confidence_export_mode_value('model_output')

    def set_confidence_export_mode_value(self, value: str) -> None:
        normalized = str(value or 'off').strip().lower()
        index = self.confidence_save_mode_combo.findData(normalized)
        if index < 0:
            index = self.confidence_save_mode_combo.findText(normalized)
        if index < 0:
            index = 0
        self.confidence_save_mode_combo.setCurrentIndex(index)

    @staticmethod
    def get_combo_value(combo: NoWheelComboBox) -> str:
        return str(combo.currentData() or combo.currentText() or '')

    @staticmethod
    def set_combo_value(combo: NoWheelComboBox, value: str) -> None:
        normalized = str(value or '').strip().lower()
        index = combo.findData(normalized)
        if index < 0:
            index = combo.findText(normalized)
        if index < 0:
            index = 0
        combo.setCurrentIndex(index)

    def get_synthetic_topology_domain_value(self) -> str:
        return self.get_combo_value(self.synthetic_topology_domain_combo) or 'pcb'

    def set_synthetic_topology_domain_value(self, value: str) -> None:
        self.set_combo_value(self.synthetic_topology_domain_combo, value)

    def _sync_pcb_defect_count_bounds(self, _value: int | None = None) -> None:
        min_defects = int(self.pcb_defects_min_count_spinbox.value())
        max_defects = int(self.pcb_defects_max_count_spinbox.value())
        if min_defects <= max_defects:
            return
        if self.sender() is self.pcb_defects_min_count_spinbox:
            self.pcb_defects_max_count_spinbox.setValue(min_defects)
        else:
            self.pcb_defects_min_count_spinbox.setValue(max_defects)

    def _sync_training_augmentation_controls(self, _enabled: bool | None = None) -> None:
        training_enabled = self._training_controls_applicable
        self.cutout_check_box.setEnabled(training_enabled)
        self.random_artifacts_check_box.setEnabled(training_enabled)
        self.cutout_groupbox.setEnabled(training_enabled)
        random_artifacts_enabled = training_enabled and bool(self.random_artifacts_check_box.isChecked())
        self.random_artifacts_groupbox.setEnabled(training_enabled)
        for checkbox in self.random_artifact_type_checkboxes.values():
            checkbox.setEnabled(random_artifacts_enabled)
        self.mixup_check_box.setEnabled(training_enabled)
        self.mixup_groupbox.setEnabled(training_enabled)
        pcb_defects_enabled = training_enabled and bool(self.pcb_defects_check_box.isChecked())
        self.pcb_defects_groupbox.setEnabled(training_enabled)
        self.pcb_defects_check_box.setEnabled(training_enabled)
        self.pcb_defects_use_input_mask_check_box.setEnabled(pcb_defects_enabled)
        self.pcb_defects_use_defect_mask_as_label_check_box.setEnabled(pcb_defects_enabled)
        self._set_field_enabled(
            self.cutout_probability_spinbox,
            training_enabled and bool(self.cutout_check_box.isChecked()),
        )
        self._set_field_enabled(
            self.cutout_holes_spinbox,
            training_enabled and bool(self.cutout_check_box.isChecked()),
        )
        self._set_field_enabled(
            self.cutout_size_ratio_spinbox,
            training_enabled and bool(self.cutout_check_box.isChecked()),
        )
        self._set_field_enabled(
            self.random_artifacts_probability_spinbox,
            random_artifacts_enabled,
        )
        self._set_field_enabled(
            self.random_artifacts_count_spinbox,
            random_artifacts_enabled,
        )
        self._set_field_enabled(
            self.random_artifacts_size_ratio_spinbox,
            random_artifacts_enabled,
        )
        self._set_field_enabled(
            self.mixup_probability_spinbox,
            training_enabled and bool(self.mixup_check_box.isChecked()),
        )
        self._set_field_enabled(
            self.mixup_alpha_spinbox,
            training_enabled and bool(self.mixup_check_box.isChecked()),
        )
        self._set_field_enabled(self.pcb_defects_probability_spinbox, pcb_defects_enabled)
        self._set_field_enabled(self.pcb_defects_min_count_spinbox, pcb_defects_enabled)
        self._set_field_enabled(self.pcb_defects_max_count_spinbox, pcb_defects_enabled)
        for defect_name, _label_key in PCB_DEFECT_WEIGHT_FIELDS:
            self._set_field_enabled(self.pcb_defect_type_spinboxes[defect_name], pcb_defects_enabled)
        self._sync_synthetic_defect_generator_controls(self.synthetic_defect_generator_check_box.isChecked())

    def _sync_patch_size_controls(self, _index: int | None = None) -> None:
        if getattr(self, '_patch_size_sync_guard', False):
            return
        self._patch_size_sync_guard = True
        try:
            patch_sync = bool(self.sync_patch_sizes_check_box.isChecked())
            if patch_sync:
                train_x = int(self.train_patch_x_size.value())
                train_y = int(self.train_patch_y_size.value())
                if int(self.recognition_patch_x_size.value()) != train_x:
                    self.recognition_patch_x_size.setValue(train_x)
                if int(self.recognition_patch_y_size.value()) != train_y:
                    self.recognition_patch_y_size.setValue(train_y)
            self._set_field_enabled(
                self.recognition_patch_size_widget,
                self._recognition_controls_applicable and not patch_sync,
            )
        finally:
            self._patch_size_sync_guard = False

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
        """Restore sample cut mode radio button selection from persisted value."""
        if mode == SampleCutMode.disk.value:
            self.cut_dataset_type.setChecked(True)
        else:
            self.no_cut_dataset_type.setChecked(True)

    def apply_style(self, style: str) -> None:
        """Apply stylesheet to the settings panel."""
        if not isinstance(style, str):
            return
        self.setStyleSheet(style)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = SettingsPanel()
    window.show()
    sys.exit(app.exec())





