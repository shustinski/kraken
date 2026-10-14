"""settings_panel: construction."""
from __future__ import annotations
from typing import Any
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QDockWidget,
    QCheckBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
)
from neuralimage.UI import ClickableLabel
from neuralimage.lib.data_interfaces import (
    IC_TOPOLOGY_FAMILIES,
    PCB_TOPOLOGY_FAMILIES,
    RANDOM_ARTIFACT_TYPES,
)
from neuralimage.lib.ui_texts import get_ui_section
from neuralimage.view.axis_resize_widget import AxisResizeWidget
from neuralimage.view.settings_panel_widgets import (
    NoWheelComboBox,
    SettingsCard,
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    create_double_spinbox,
    create_min_max_widget,
    create_slider,
    create_spinbox,
)

from neuralimage.view.settings_panel_parts.module_state import (
    EPOCHS_DEFAULT,
    EPOCHS_RANGE,
    IC_DEFECT_WEIGHT_FIELDS,
    MAX_AUGMENTATION_MULTIPLIER,
    MAX_AUGMENTATION_PROBABILITY,
    MAX_AUG_BLUR_RADIUS,
    MAX_AUG_NOISE_SIGMA,
    MAX_AUG_STRENGTH,
    MAX_CROPS_PER_IMAGE,
    MAX_CUTOUT_HOLES,
    MAX_MIXUP_ALPHA,
    MAX_PCB_DEFECT_COUNT,
    MAX_RANDOM_ARTIFACTS_COUNT,
    MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
    MAX_SYNTHETIC_DATASET_FACTOR,
    MAX_SYNTHETIC_IMAGE_SIZE,
    MAX_SYNTHETIC_SEGMENT_COUNT,
    MAX_SYNTHETIC_TRACE_COUNT,
    MAX_SYNTHETIC_TRACE_HALF_WIDTH,
    MAX_SYNTHETIC_TRACE_NOISE_SIGMA,
    MAX_TECH_AUG_OPERATIONS,
    MIN_AUGMENTATION_MULTIPLIER,
    MIN_AUGMENTATION_PROBABILITY,
    MIN_AUG_BLUR_RADIUS,
    MIN_AUG_NOISE_SIGMA,
    MIN_AUG_STRENGTH,
    MIN_CROPS_PER_IMAGE,
    MIN_CUTOUT_HOLES,
    MIN_MIXUP_ALPHA,
    MIN_PCB_DEFECT_COUNT,
    MIN_RANDOM_ARTIFACTS_COUNT,
    MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
    MIN_SYNTHETIC_DATASET_FACTOR,
    MIN_SYNTHETIC_IMAGE_SIZE,
    MIN_SYNTHETIC_SEGMENT_COUNT,
    MIN_SYNTHETIC_TRACE_COUNT,
    MIN_SYNTHETIC_TRACE_HALF_WIDTH,
    MIN_SYNTHETIC_TRACE_NOISE_SIGMA,
    MIN_TECH_AUG_OPERATIONS,
    PCB_DEFECT_WEIGHT_FIELDS,
    SAMPLE_SIZE_MAX,
    SAMPLE_SIZE_MIN,
    SHIFT_RANGE_MAX,
    SHIFT_RANGE_MIN,
    VALIDATION_MAX,
    VALIDATION_MIN,
)


class ConstructionMixin:
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('settingsDock')
        texts = get_ui_section('settings_panel')
        self._texts = texts if isinstance(texts, dict) else {}
        self._content_widget = QWidget()
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setWidget(self._content_widget)
        self.setWidget(self._scroll_area)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
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
        self._legacy_uncertainty_sync_guard = False
        self._legacy_uncertainty_pending = False
        self._settings_cards: dict[str, SettingsCard] = {}
        self._settings_registry: dict[str, dict[str, Any]] = {}
        self._expert_widgets: list[QWidget] = []
        self._configuration_errors: list[tuple[str | None, str]] = []

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
        self.brightness_augmentation_check_box = QCheckBox('')
        self.contrast_augmentation_check_box = QCheckBox('')
        self.gamma_augmentation_check_box = QCheckBox('')
        self.noise_augmentation_check_box = QCheckBox('')
        self.blur_augmentation_check_box = QCheckBox('')
        for checkbox in (
            self.brightness_augmentation_check_box,
            self.contrast_augmentation_check_box,
            self.gamma_augmentation_check_box,
            self.noise_augmentation_check_box,
            self.blur_augmentation_check_box,
        ):
            checkbox.setChecked(False)
        self.random_crop_check_box = QCheckBox('')
        self.scale_augmentation_check_box = QCheckBox('')
        self.tech_augmentation_debug_pair_check_box = QCheckBox('')
        self.tech_augmentation_operation_checkboxes = {
            key: QCheckBox('')
            for key in (
                'global_width', 'scale_rethreshold', 'blur_threshold',
                'boundary_aware', 'local_morphology', 'gap_variation',
            )
        }
        self.random_artifact_type_checkboxes: dict[str, QCheckBox] = {
            artifact_name: QCheckBox('') for artifact_name in RANDOM_ARTIFACT_TYPES
        }
        for checkbox in self.random_artifact_type_checkboxes.values():
            checkbox.setChecked(True)
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
        self.augmentation_multiplier_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_MULTIPLIER, MAX_AUGMENTATION_MULTIPLIER),
            step=0.5,
            default_value=0.0,
            decimals=1,
        )
        self.horizontal_rotation_probability_spinbox = create_double_spinbox(
            (0.0, 1.0), step=0.05, default_value=1.0, decimals=2
        )
        self.vertical_rotation_probability_spinbox = create_double_spinbox(
            (0.0, 1.0), step=0.05, default_value=1.0, decimals=2
        )
        self.flip_x_probability_spinbox = create_double_spinbox(
            (0.0, 1.0), step=0.05, default_value=1.0, decimals=2
        )
        self.flip_y_probability_spinbox = create_double_spinbox(
            (0.0, 1.0), step=0.05, default_value=1.0, decimals=2
        )
        self.scale_augmentation_probability_spinbox = create_double_spinbox(
            (0.0, 1.0), step=0.05, default_value=1.0, decimals=2
        )
        self.augmentation_brightness_probability_spinbox = create_double_spinbox(
            (0.0, 1.0), step=0.05, default_value=1.0, decimals=2
        )
        self.augmentation_contrast_probability_spinbox = create_double_spinbox(
            (0.0, 1.0), step=0.05, default_value=1.0, decimals=2
        )
        self.augmentation_gamma_probability_spinbox = create_double_spinbox(
            (0.0, 1.0), step=0.05, default_value=1.0, decimals=2
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
        self.synthetic_image_size_widget = AxisResizeWidget(
            target_width=256,
            target_height=256,
            minimum_pixels=MIN_SYNTHETIC_IMAGE_SIZE,
            maximum_pixels=MAX_SYNTHETIC_IMAGE_SIZE,
            single_step=32,
        )
        # Preserve the established settings-panel control attributes for bindings,
        # state synchronization, tests, and third-party integrations.
        self.synthetic_image_width_spinbox = self.synthetic_image_size_widget.width_spinbox
        self.synthetic_image_height_spinbox = self.synthetic_image_size_widget.height_spinbox
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

        self.train_patch_size_widget = AxisResizeWidget(
            minimum_pixels=SAMPLE_SIZE_MIN,
            maximum_pixels=SAMPLE_SIZE_MAX,
            single_step=10,
        )
        self.train_patch_x_size = self.train_patch_size_widget.width_spinbox
        self.train_patch_y_size = self.train_patch_size_widget.height_spinbox
        self.recognition_patch_size_widget = AxisResizeWidget(
            minimum_pixels=SAMPLE_SIZE_MIN,
            maximum_pixels=SAMPLE_SIZE_MAX,
            single_step=10,
        )
        self.recognition_patch_x_size = self.recognition_patch_size_widget.width_spinbox
        self.recognition_patch_y_size = self.recognition_patch_size_widget.height_spinbox
        self.recognition_tab_patch_size_widget = AxisResizeWidget(
            minimum_pixels=SAMPLE_SIZE_MIN,
            maximum_pixels=SAMPLE_SIZE_MAX,
            single_step=10,
        )
        self._recognition_patch_copy_sync_guard = False
        self.recognition_patch_size_widget.sizeChanged.connect(
            self._sync_recognition_patch_copy_from_primary
        )
        self.recognition_tab_patch_size_widget.sizeChanged.connect(
            self._sync_recognition_patch_primary_from_copy
        )
        self.random_patch_size_check_box = QCheckBox('')
        self.random_patch_min_size_widget = AxisResizeWidget(
            target_width=128,
            target_height=128,
            minimum_pixels=32,
            maximum_pixels=SAMPLE_SIZE_MAX,
            single_step=32,
        )
        self.random_patch_min_x_size = self.random_patch_min_size_widget.width_spinbox
        self.random_patch_min_y_size = self.random_patch_min_size_widget.height_spinbox
        self.random_patch_max_size_widget = AxisResizeWidget(
            target_width=512,
            target_height=512,
            minimum_pixels=32,
            maximum_pixels=SAMPLE_SIZE_MAX,
            single_step=32,
        )
        self.random_patch_max_x_size = self.random_patch_max_size_widget.width_spinbox
        self.random_patch_max_y_size = self.random_patch_max_size_widget.height_spinbox
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
        self._init_nn_auxilary_settings()
        self._init_preprocess_groupbox()
        self._init_layout()
        self._apply_localized_texts()
        self.sync_business_logic_controls(None)
