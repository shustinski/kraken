from __future__ import annotations

import copy
import random
import zlib
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from augmentations import (
    ICDefectAugmentor,
    PCBDefectAugmentor,
    SyntheticTopologyGenerator,
    SyntheticTopologyParameters,
    TechVariationAugmentor,
)
from lib.data_interfaces import (
    IC_TOPOLOGY_FAMILIES,
    PCB_TOPOLOGY_FAMILIES,
    TrainingParameters,
    build_ic_defect_parameters,
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
    build_tech_augmentation_config,
)
from lib.images import ImagePreparator, SampleFastCutter
from lib.random_artifacts import generate_random_artifact_patch
from lib.rare_patch_masks import collect_matching_sample_label_pairs
from lib.ui_texts import get_ui_section
from model.NeuralNetwork.dataset import _apply_binary_tech_augmentation_to_pair
from view.settings_panel_widgets import NoWheelComboBox, create_double_spinbox, create_size_widget, create_slider, create_spinbox

MIN_AUG_STRENGTH = 0.0
MAX_AUG_STRENGTH = 1.0
MIN_AUGMENTATION_PROBABILITY = 0.0
MAX_AUGMENTATION_PROBABILITY = 1.0
MIN_AUG_NOISE_SIGMA = 0.0
MAX_AUG_NOISE_SIGMA = 0.2
MIN_AUG_BLUR_RADIUS = 0.0
MAX_AUG_BLUR_RADIUS = 5.0
MIN_CROPS_PER_IMAGE = 1
MAX_CROPS_PER_IMAGE = 5000
MIN_TECH_AUG_OPERATIONS = 1
MAX_TECH_AUG_OPERATIONS = 6
MIN_CUTOUT_HOLES = 1
MAX_CUTOUT_HOLES = 32
MIN_RANDOM_ARTIFACTS_COUNT = 1
MAX_RANDOM_ARTIFACTS_COUNT = 16
MIN_MIXUP_ALPHA = 0.0
MAX_MIXUP_ALPHA = 10.0
MIN_PCB_DEFECT_COUNT = 1
MAX_PCB_DEFECT_COUNT = 8
MIN_PCB_DEFECT_WEIGHT = 0.0
MAX_PCB_DEFECT_WEIGHT = 5.0
MIN_SYNTHETIC_TRACE_COUNT = 1
MAX_SYNTHETIC_TRACE_COUNT = 200
MIN_SYNTHETIC_IMAGE_SIZE = 64
MAX_SYNTHETIC_IMAGE_SIZE = 8192
MIN_SYNTHETIC_SEGMENT_COUNT = 1
MAX_SYNTHETIC_SEGMENT_COUNT = 24
MIN_SYNTHETIC_TRACE_HALF_WIDTH = 1
MAX_SYNTHETIC_TRACE_HALF_WIDTH = 50
MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA = 0.0
MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA = 0.2
MIN_SYNTHETIC_TRACE_NOISE_SIGMA = 0.0
MAX_SYNTHETIC_TRACE_NOISE_SIGMA = 0.2
IC_DEFECT_WEIGHT_FIELDS: tuple[tuple[str, str], ...] = (
    ('line_break', 'ic_line_break_severity'),
    ('bridge', 'ic_bridge_severity'),
    ('necking', 'ic_necking_severity'),
    ('missing_metal', 'ic_missing_metal_severity'),
    ('spur', 'ic_spur_severity'),
    ('pinhole', 'ic_pinhole_severity'),
    ('via_open', 'ic_via_open_severity'),
    ('line_shift', 'ic_line_shift_severity'),
)

PREVIEW_VALUE_LABELS_EN = {
    'crops_per_image': 'Crops per image',
    'scale_augmentation_strength': 'Scale strength',
    'augmentation_brightness_strength': 'Brightness change',
    'augmentation_contrast_strength': 'Contrast change',
    'augmentation_gamma_strength': 'Gamma change',
    'augmentation_noise_probability': 'Noise probability',
    'augmentation_noise_sigma': 'Noise strength',
    'augmentation_blur_probability': 'Blur probability',
    'augmentation_blur_radius': 'Blur radius',
    'tech_aug_min_operations': 'Min operations',
    'tech_aug_max_operations': 'Max operations',
    'tech_aug_global_width_probability': 'Width variation probability',
    'tech_aug_scale_rethreshold_probability': 'Scale + rethreshold probability',
    'tech_aug_blur_threshold_probability': 'Blur + threshold probability',
    'tech_aug_boundary_aware_probability': 'Boundary-aware probability',
    'tech_aug_local_morphology_probability': 'Local morphology probability',
    'tech_aug_gap_variation_probability': 'Gap variation probability',
    'cutout_probability': 'Cutout probability',
    'cutout_holes': 'Cutout holes',
    'cutout_size_ratio': 'Cutout size ratio',
    'random_artifacts_probability': 'Artifacts probability',
    'random_artifacts_count': 'Artifacts count',
    'random_artifacts_size_ratio': 'Artifacts size ratio',
    'mixup_probability': 'Mixup probability',
    'mixup_alpha': 'Mixup alpha',
    'pcb_defects_probability': 'Defect probability',
    'pcb_defects_min_count': 'Min defects',
    'pcb_defects_max_count': 'Max defects',
    'synthetic_image_size': 'Synthetic image size',
    'synthetic_trace_count': 'Trace count',
    'synthetic_segment_count': 'Segments per trace',
    'synthetic_trace_half_width': 'Trace half-width',
    'synthetic_background_noise_sigma': 'Background noise sigma',
    'synthetic_trace_noise_sigma': 'Trace noise sigma',
    'pcb_break_severity': 'Break severity',
    'pcb_short_severity': 'Short severity',
    'pcb_missing_copper_severity': 'Missing copper severity',
    'pcb_excess_copper_severity': 'Excess copper severity',
    'pcb_pinhole_severity': 'Pinhole severity',
    'pcb_spurious_copper_severity': 'Spurious copper severity',
    'pcb_via_severity': 'Via defect severity',
    'pcb_misalignment_severity': 'Misalignment severity',
}
PREVIEW_VALUE_LABELS_RU = {
    'crops_per_image': 'Фрагментов на изображение',
    'scale_augmentation_strength': 'Сила масштабирования',
    'augmentation_brightness_strength': 'Изменение яркости',
    'augmentation_contrast_strength': 'Изменение контраста',
    'augmentation_gamma_strength': 'Сила гаммы',
    'augmentation_noise_probability': 'Вероятность шума',
    'augmentation_noise_sigma': 'Сила шума',
    'augmentation_blur_probability': 'Вероятность размытия',
    'augmentation_blur_radius': 'Радиус размытия',
    'tech_aug_min_operations': 'Минимум операций',
    'tech_aug_max_operations': 'Максимум операций',
    'tech_aug_global_width_probability': 'Вероятность вариации ширины',
    'tech_aug_scale_rethreshold_probability': 'Вероятность scale + threshold',
    'tech_aug_blur_threshold_probability': 'Вероятность blur + threshold',
    'tech_aug_boundary_aware_probability': 'Вероятность пограничной вариации',
    'tech_aug_local_morphology_probability': 'Вероятность локальной морфологии',
    'tech_aug_gap_variation_probability': 'Вероятность вариации зазоров',
    'cutout_probability': 'Вероятность cutout',
    'cutout_holes': 'Количество областей cutout',
    'cutout_size_ratio': 'Размер области cutout',
    'random_artifacts_probability': 'Вероятность артефактов',
    'random_artifacts_count': 'Количество артефактов',
    'random_artifacts_size_ratio': 'Размер артефактов',
    'mixup_probability': 'Вероятность mixup',
    'mixup_alpha': 'Параметр alpha',
    'pcb_defects_probability': 'Вероятность дефектов',
    'pcb_defects_min_count': 'Минимум дефектов',
    'pcb_defects_max_count': 'Максимум дефектов',
}


@contextmanager
def _seeded_random(seed: int):
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    torch_random_state = torch.random.get_rng_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        torch.random.set_rng_state(torch_random_state)


class _PreviewLabel(QLabel):
    middle_pressed = pyqtSignal()
    middle_released = pyqtSignal()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.middle_pressed.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.middle_released.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class AugmentationPreviewDialog(QDialog):
    apply_to_main_requested = pyqtSignal(object)

    def __init__(self, training_parameters: TrainingParameters, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._training_parameters = training_parameters
        self._texts = get_ui_section('augmentation_preview_dialog')
        self._settings_texts = get_ui_section('settings_panel')
        self._is_russian_ui = any('\u0400' <= char <= '\u04FF' for char in str(self._texts.get('window_title', '')))
        settings_form = dict(self._settings_texts.get('settings_form', {}))
        self._settings_form_labels = dict(settings_form.get('labels', {}))
        self._settings_form_tooltips = dict(settings_form.get('tooltips', {}))
        self._sample_pairs, self._load_error = collect_matching_sample_label_pairs(
            training_parameters.image_path,
            training_parameters.label_path,
            strict=False,
        )
        self._current_sample_index = 0
        self._variant_serial = 0
        self._show_augmented = True
        self._sample_list_updating = False
        self._toggle_boxes: dict[str, QCheckBox] = {}
        self._value_widgets: dict[str, list[QWidget]] = {}
        self._value_rows: dict[str, list[QWidget]] = {}
        self._original_image_array: np.ndarray | None = None
        self._augmented_image_array: np.ndarray | None = None
        self._original_label_array: np.ndarray | None = None
        self._augmented_label_array: np.ndarray | None = None

        self._create_value_controls()
        self._build_ui()
        self._initialize_toggle_states()
        self._connect_signals()
        self._sync_group_boxes()
        self._refresh_preview()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_visible_preview()

    def _build_ui(self) -> None:
        self.setWindowTitle(str(self._texts.get('window_title', 'Augmentation preview')))
        self.resize(1420, 860)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(6)
        self.prev_button = QPushButton(str(self._texts.get('prev_button', 'Previous')))
        self.next_button = QPushButton(str(self._texts.get('next_button', 'Next')))
        self.resample_button = QPushButton(str(self._texts.get('resample_button', 'Resample')))
        self.apply_to_main_button = QPushButton(
            str(
                self._texts.get(
                    'apply_to_main_button',
                    'Перенести в основное окно' if self._is_russian_ui else 'Apply to main window',
                )
            )
        )
        self.full_image_check_box = QCheckBox(
            str(
                self._texts.get(
                    'full_image_toggle',
                    'Показывать целиком' if self._is_russian_ui else 'Show full image',
                )
            )
        )
        self.sample_label = QLabel('')
        self.sample_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        nav_row.addWidget(self.prev_button)
        nav_row.addWidget(self.next_button)
        nav_row.addWidget(self.resample_button)
        nav_row.addWidget(self.apply_to_main_button)
        nav_row.addWidget(self.full_image_check_box)
        nav_row.addWidget(self.sample_label, 1)
        left_layout.addLayout(nav_row)

        self.mode_label = QLabel('')
        self.mode_label.setWordWrap(True)
        left_layout.addWidget(self.mode_label)

        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(10)

        self.sample_list_group = QGroupBox(
            str(
                self._texts.get(
                    'sample_list_group',
                    'Изображения' if self._is_russian_ui else 'Images',
                )
            )
        )
        sample_list_layout = QVBoxLayout(self.sample_list_group)
        sample_list_layout.setContentsMargins(6, 6, 6, 6)
        self.sample_list_widget = QListWidget(self.sample_list_group)
        self.sample_list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.sample_list_widget.setMinimumWidth(220)
        sample_list_layout.addWidget(self.sample_list_widget)

        preview_widget = QWidget(self)
        preview_row = QHBoxLayout(preview_widget)
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(10)

        self.image_group = QGroupBox(str(self._texts.get('image_group', 'Image')))
        image_layout = QVBoxLayout(self.image_group)
        self.image_preview = _PreviewLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumSize(420, 420)
        self.image_preview.setStyleSheet('border: 1px solid #666; background: #111;')
        image_layout.addWidget(self.image_preview)

        self.label_group = QGroupBox(str(self._texts.get('label_group', 'Label')))
        label_layout = QVBoxLayout(self.label_group)
        self.label_preview = _PreviewLabel()
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview.setMinimumSize(420, 420)
        self.label_preview.setStyleSheet('border: 1px solid #666; background: #111;')
        label_layout.addWidget(self.label_preview)

        preview_row.addWidget(self.image_group, 1)
        preview_row.addWidget(self.label_group, 1)
        content_row.addWidget(self.sample_list_group, 0)
        content_row.addWidget(preview_widget, 1)
        left_layout.addLayout(content_row, 1)

        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        right_scroll = QScrollArea(self)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(420)
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self._build_synthetic_group())
        right_layout.addWidget(self._build_spatial_group())
        right_layout.addWidget(self._build_photometric_group())
        right_layout.addWidget(self._build_batch_group())
        right_layout.addStretch(1)
        right_scroll.setWidget(right_content)

        self._populate_sample_list()
        root_layout.addWidget(left_widget, 1)
        root_layout.addWidget(right_scroll, 0)

    def _create_value_controls(self) -> None:
        generation = self._training_parameters.generation
        tech_aug = build_tech_augmentation_config(getattr(generation, 'tech_aug', None))
        cutout = getattr(self._training_parameters, 'cutout', None)
        random_artifacts = getattr(self._training_parameters, 'random_artifacts', None)
        mixup = getattr(self._training_parameters, 'mixup', None)
        synthetic_generator = build_synthetic_defect_generator_parameters(
            getattr(self._training_parameters, 'synthetic_defect_generator', None)
        )
        pcb_defects = synthetic_generator.pcb_defects
        ic_defects = synthetic_generator.ic_defects

        self.crops_per_image_spinbox = create_spinbox(
            (MIN_CROPS_PER_IMAGE, MAX_CROPS_PER_IMAGE),
            default_value=self._bounded_int(getattr(generation, 'crops_per_image', 64), MIN_CROPS_PER_IMAGE, MAX_CROPS_PER_IMAGE),
            step=1,
        )
        self.scale_augmentation_strength_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=self._bounded_float(
                getattr(generation, 'scale_augmentation_strength', 0.2),
                MIN_AUG_STRENGTH,
                MAX_AUG_STRENGTH,
            ),
            decimals=2,
        )
        self.augmentation_brightness_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=self._bounded_float(
                getattr(generation, 'augmentation_brightness_strength', 0.1),
                MIN_AUG_STRENGTH,
                MAX_AUG_STRENGTH,
            ),
            decimals=2,
        )
        self.augmentation_contrast_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=self._bounded_float(
                getattr(generation, 'augmentation_contrast_strength', 0.1),
                MIN_AUG_STRENGTH,
                MAX_AUG_STRENGTH,
            ),
            decimals=2,
        )
        self.augmentation_gamma_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=self._bounded_float(
                getattr(generation, 'augmentation_gamma_strength', 0.15),
                MIN_AUG_STRENGTH,
                MAX_AUG_STRENGTH,
            ),
            decimals=2,
        )
        self.augmentation_noise_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                getattr(generation, 'augmentation_noise_probability', 0.5),
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.augmentation_noise_sigma_spinbox = create_double_spinbox(
            (MIN_AUG_NOISE_SIGMA, MAX_AUG_NOISE_SIGMA),
            step=0.005,
            default_value=self._bounded_float(
                getattr(generation, 'augmentation_noise_sigma', 0.01),
                MIN_AUG_NOISE_SIGMA,
                MAX_AUG_NOISE_SIGMA,
            ),
            decimals=3,
        )
        self.augmentation_blur_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                getattr(generation, 'augmentation_blur_probability', 0.25),
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.augmentation_blur_radius_spinbox = create_double_spinbox(
            (MIN_AUG_BLUR_RADIUS, MAX_AUG_BLUR_RADIUS),
            step=0.1,
            default_value=self._bounded_float(
                getattr(generation, 'augmentation_blur_radius', 1.0),
                MIN_AUG_BLUR_RADIUS,
                MAX_AUG_BLUR_RADIUS,
            ),
            decimals=2,
        )
        self.tech_aug_min_operations_spinbox = create_spinbox(
            (MIN_TECH_AUG_OPERATIONS, MAX_TECH_AUG_OPERATIONS),
            default_value=self._bounded_int(tech_aug.min_operations, MIN_TECH_AUG_OPERATIONS, MAX_TECH_AUG_OPERATIONS),
            step=1,
        )
        self.tech_aug_max_operations_spinbox = create_spinbox(
            (MIN_TECH_AUG_OPERATIONS, MAX_TECH_AUG_OPERATIONS),
            default_value=self._bounded_int(tech_aug.max_operations, MIN_TECH_AUG_OPERATIONS, MAX_TECH_AUG_OPERATIONS),
            step=1,
        )
        self.tech_aug_global_width_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                tech_aug.global_width.probability,
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.tech_aug_scale_rethreshold_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                tech_aug.scale_rethreshold.probability,
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.tech_aug_blur_threshold_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                tech_aug.blur_threshold.probability,
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.tech_aug_boundary_aware_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                tech_aug.boundary_aware.probability,
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.tech_aug_local_morphology_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                tech_aug.local_morphology.probability,
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.tech_aug_gap_variation_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                tech_aug.gap_variation.probability,
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.cutout_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                getattr(cutout, 'probability', 1.0),
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.cutout_holes_spinbox = create_spinbox(
            (MIN_CUTOUT_HOLES, MAX_CUTOUT_HOLES),
            default_value=self._bounded_int(getattr(cutout, 'holes', 1), MIN_CUTOUT_HOLES, MAX_CUTOUT_HOLES),
            step=1,
        )
        self.cutout_size_ratio_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=self._bounded_float(
                getattr(cutout, 'size_ratio', 0.25),
                MIN_AUG_STRENGTH,
                MAX_AUG_STRENGTH,
            ),
            decimals=2,
        )
        self.random_artifacts_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                getattr(random_artifacts, 'probability', 1.0),
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.random_artifacts_count_spinbox = create_spinbox(
            (MIN_RANDOM_ARTIFACTS_COUNT, MAX_RANDOM_ARTIFACTS_COUNT),
            default_value=self._bounded_int(
                getattr(random_artifacts, 'count', 1),
                MIN_RANDOM_ARTIFACTS_COUNT,
                MAX_RANDOM_ARTIFACTS_COUNT,
            ),
            step=1,
        )
        self.random_artifacts_size_ratio_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=self._bounded_float(
                getattr(random_artifacts, 'size_ratio', 0.25),
                MIN_AUG_STRENGTH,
                MAX_AUG_STRENGTH,
            ),
            decimals=2,
        )
        self.mixup_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                getattr(mixup, 'probability', 1.0),
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.mixup_alpha_spinbox = create_double_spinbox(
            (MIN_MIXUP_ALPHA, MAX_MIXUP_ALPHA),
            step=0.05,
            default_value=self._bounded_float(getattr(mixup, 'alpha', 0.2), MIN_MIXUP_ALPHA, MAX_MIXUP_ALPHA),
            decimals=2,
        )
        self.pcb_defects_probability_spinbox = create_double_spinbox(
            (MIN_AUGMENTATION_PROBABILITY, MAX_AUGMENTATION_PROBABILITY),
            step=0.05,
            default_value=self._bounded_float(
                pcb_defects.defect_probability,
                MIN_AUGMENTATION_PROBABILITY,
                MAX_AUGMENTATION_PROBABILITY,
            ),
            decimals=2,
        )
        self.pcb_defects_min_count_spinbox = create_spinbox(
            (MIN_PCB_DEFECT_COUNT, MAX_PCB_DEFECT_COUNT),
            default_value=self._bounded_int(pcb_defects.min_defects, MIN_PCB_DEFECT_COUNT, MAX_PCB_DEFECT_COUNT),
            step=1,
        )
        self.pcb_defects_max_count_spinbox = create_spinbox(
            (MIN_PCB_DEFECT_COUNT, MAX_PCB_DEFECT_COUNT),
            default_value=self._bounded_int(pcb_defects.max_defects, MIN_PCB_DEFECT_COUNT, MAX_PCB_DEFECT_COUNT),
            step=1,
        )
        self.pcb_defect_type_spinboxes = {
            defect_name: create_slider(
                (0, 100),
                default_value=self._bounded_int(
                    int(round(float(pcb_defects.defect_severities.get(defect_name, 0.5)) * 100.0)),
                    0,
                    100,
                ),
            )
            for defect_name in (
                'break',
                'short',
                'missing_copper',
                'excess_copper',
                'pinhole',
                'spurious_copper',
                'via',
                'misalignment',
            )
        }
        self.ic_defect_type_spinboxes = {
            defect_name: create_slider(
                (0, 100),
                default_value=self._bounded_int(
                    int(round(float(ic_defects.defect_severities.get(defect_name, 0.5)) * 100.0)),
                    0,
                    100,
                ),
            )
            for defect_name, _label_key in IC_DEFECT_WEIGHT_FIELDS
        }
        self.synthetic_topology_domain_combo = NoWheelComboBox()
        self.synthetic_topology_domain_combo.addItem('PCB', 'pcb')
        self.synthetic_topology_domain_combo.addItem('IC', 'ic')
        self.pcb_topology_family_combo = NoWheelComboBox()
        for family in PCB_TOPOLOGY_FAMILIES:
            self.pcb_topology_family_combo.addItem(family, family)
        self.ic_topology_family_combo = NoWheelComboBox()
        for family in IC_TOPOLOGY_FAMILIES:
            self.ic_topology_family_combo.addItem(family, family)
        self._set_combo_value(self.synthetic_topology_domain_combo, str(synthetic_generator.topology_domain))
        self._set_combo_value(
            self.pcb_topology_family_combo,
            str(getattr(synthetic_generator, 'topology_family', PCB_TOPOLOGY_FAMILIES[0])),
        )
        self._set_combo_value(
            self.ic_topology_family_combo,
            str(getattr(synthetic_generator, 'topology_family', IC_TOPOLOGY_FAMILIES[0])),
        )
        self.synthetic_trace_count_min_spinbox = create_spinbox(
            (MIN_SYNTHETIC_TRACE_COUNT, MAX_SYNTHETIC_TRACE_COUNT),
            default_value=self._bounded_int(
                getattr(synthetic_generator, 'trace_count_range', (5, 5))[0],
                MIN_SYNTHETIC_TRACE_COUNT,
                MAX_SYNTHETIC_TRACE_COUNT,
            ),
            step=1,
        )
        self.synthetic_image_width_spinbox = create_spinbox(
            (MIN_SYNTHETIC_IMAGE_SIZE, MAX_SYNTHETIC_IMAGE_SIZE),
            default_value=self._bounded_int(
                getattr(synthetic_generator, 'image_size_xy', (1024, 1024))[0],
                MIN_SYNTHETIC_IMAGE_SIZE,
                MAX_SYNTHETIC_IMAGE_SIZE,
            ),
            step=32,
        )
        self.synthetic_image_height_spinbox = create_spinbox(
            (MIN_SYNTHETIC_IMAGE_SIZE, MAX_SYNTHETIC_IMAGE_SIZE),
            default_value=self._bounded_int(
                getattr(synthetic_generator, 'image_size_xy', (1024, 1024))[1],
                MIN_SYNTHETIC_IMAGE_SIZE,
                MAX_SYNTHETIC_IMAGE_SIZE,
            ),
            step=32,
        )
        self.synthetic_image_size_widget = create_size_widget(
            self.synthetic_image_width_spinbox,
            self.synthetic_image_height_spinbox,
        )
        self.synthetic_trace_count_max_spinbox = create_spinbox(
            (MIN_SYNTHETIC_TRACE_COUNT, MAX_SYNTHETIC_TRACE_COUNT),
            default_value=self._bounded_int(
                getattr(synthetic_generator, 'trace_count_range', (5, 5))[1],
                MIN_SYNTHETIC_TRACE_COUNT,
                MAX_SYNTHETIC_TRACE_COUNT,
            ),
            step=1,
        )
        self.synthetic_segment_count_min_spinbox = create_spinbox(
            (MIN_SYNTHETIC_SEGMENT_COUNT, MAX_SYNTHETIC_SEGMENT_COUNT),
            default_value=self._bounded_int(
                getattr(synthetic_generator, 'segment_count_range', (4, 4))[0],
                MIN_SYNTHETIC_SEGMENT_COUNT,
                MAX_SYNTHETIC_SEGMENT_COUNT,
            ),
            step=1,
        )
        self.synthetic_segment_count_max_spinbox = create_spinbox(
            (MIN_SYNTHETIC_SEGMENT_COUNT, MAX_SYNTHETIC_SEGMENT_COUNT),
            default_value=self._bounded_int(
                getattr(synthetic_generator, 'segment_count_range', (4, 4))[1],
                MIN_SYNTHETIC_SEGMENT_COUNT,
                MAX_SYNTHETIC_SEGMENT_COUNT,
            ),
            step=1,
        )
        self.synthetic_trace_half_width_min_spinbox = create_spinbox(
            (MIN_SYNTHETIC_TRACE_HALF_WIDTH, MAX_SYNTHETIC_TRACE_HALF_WIDTH),
            default_value=self._bounded_int(
                getattr(synthetic_generator, 'trace_half_width_range', (2, 2))[0],
                MIN_SYNTHETIC_TRACE_HALF_WIDTH,
                MAX_SYNTHETIC_TRACE_HALF_WIDTH,
            ),
            step=1,
        )
        self.synthetic_trace_half_width_max_spinbox = create_spinbox(
            (MIN_SYNTHETIC_TRACE_HALF_WIDTH, MAX_SYNTHETIC_TRACE_HALF_WIDTH),
            default_value=self._bounded_int(
                getattr(synthetic_generator, 'trace_half_width_range', (2, 2))[1],
                MIN_SYNTHETIC_TRACE_HALF_WIDTH,
                MAX_SYNTHETIC_TRACE_HALF_WIDTH,
            ),
            step=1,
        )
        self.synthetic_background_noise_sigma_min_spinbox = create_double_spinbox(
            (MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA, MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA),
            step=0.005,
            default_value=self._bounded_float(
                getattr(synthetic_generator, 'background_noise_sigma_range', (0.02, 0.02))[0],
                MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
                MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
            ),
            decimals=3,
        )
        self.synthetic_background_noise_sigma_max_spinbox = create_double_spinbox(
            (MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA, MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA),
            step=0.005,
            default_value=self._bounded_float(
                getattr(synthetic_generator, 'background_noise_sigma_range', (0.02, 0.02))[1],
                MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
                MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
            ),
            decimals=3,
        )
        self.synthetic_trace_noise_sigma_min_spinbox = create_double_spinbox(
            (MIN_SYNTHETIC_TRACE_NOISE_SIGMA, MAX_SYNTHETIC_TRACE_NOISE_SIGMA),
            step=0.005,
            default_value=self._bounded_float(
                getattr(synthetic_generator, 'trace_noise_sigma_range', (0.01, 0.01))[0],
                MIN_SYNTHETIC_TRACE_NOISE_SIGMA,
                MAX_SYNTHETIC_TRACE_NOISE_SIGMA,
            ),
            decimals=3,
        )
        self.synthetic_trace_noise_sigma_max_spinbox = create_double_spinbox(
            (MIN_SYNTHETIC_TRACE_NOISE_SIGMA, MAX_SYNTHETIC_TRACE_NOISE_SIGMA),
            step=0.005,
            default_value=self._bounded_float(
                getattr(synthetic_generator, 'trace_noise_sigma_range', (0.01, 0.01))[1],
                MIN_SYNTHETIC_TRACE_NOISE_SIGMA,
                MAX_SYNTHETIC_TRACE_NOISE_SIGMA,
            ),
            decimals=3,
        )

    def _build_synthetic_group(self) -> QGroupBox:
        group = QGroupBox(
            str(
                self._texts.get(
                    'synthetic_group',
                    'Синтетическая топология' if self._is_russian_ui else 'Synthetic topology',
                )
            )
        )
        layout = QVBoxLayout(group)
        layout.addWidget(
            self._create_toggle(
                'synthetic_topology',
                'synthetic_topology',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'synthetic_topology_domain',
                self.synthetic_topology_domain_combo,
                parent_key='synthetic_topology_domain',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'pcb_topology_family',
                self.pcb_topology_family_combo,
                parent_key='pcb_topology_family',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'ic_topology_family',
                self.ic_topology_family_combo,
                parent_key='ic_topology_family',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'synthetic_image_size',
                self.synthetic_image_size_widget,
                parent_key='synthetic_topology',
            )
        )
        layout.addWidget(
            self._create_range_row(
                'synthetic_trace_count',
                self.synthetic_trace_count_min_spinbox,
                self.synthetic_trace_count_max_spinbox,
                parent_key='synthetic_topology',
            )
        )
        layout.addWidget(
            self._create_range_row(
                'synthetic_segment_count',
                self.synthetic_segment_count_min_spinbox,
                self.synthetic_segment_count_max_spinbox,
                parent_key='synthetic_topology',
            )
        )
        layout.addWidget(
            self._create_range_row(
                'synthetic_trace_half_width',
                self.synthetic_trace_half_width_min_spinbox,
                self.synthetic_trace_half_width_max_spinbox,
                parent_key='synthetic_topology',
            )
        )
        layout.addWidget(
            self._create_range_row(
                'synthetic_background_noise_sigma',
                self.synthetic_background_noise_sigma_min_spinbox,
                self.synthetic_background_noise_sigma_max_spinbox,
                parent_key='synthetic_topology',
            )
        )
        layout.addWidget(
            self._create_range_row(
                'synthetic_trace_noise_sigma',
                self.synthetic_trace_noise_sigma_min_spinbox,
                self.synthetic_trace_noise_sigma_max_spinbox,
                parent_key='synthetic_topology',
            )
        )
        layout.addWidget(self._create_toggle('pcb_defects', 'pcb_group'))
        layout.addWidget(
            self._create_value_row(
                'pcb_defects_probability',
                self.pcb_defects_probability_spinbox,
                parent_key='pcb_defects',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'pcb_defects_min_count',
                self.pcb_defects_min_count_spinbox,
                parent_key='pcb_defects',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'pcb_defects_max_count',
                self.pcb_defects_max_count_spinbox,
                parent_key='pcb_defects',
            )
        )
        layout.addWidget(self._create_toggle('pcb_break', 'pcb_break', indent=16))
        layout.addWidget(
            self._create_weight_row('pcb_break', self.pcb_defect_type_spinboxes['break'], parent_key='pcb_break')
        )
        layout.addWidget(self._create_toggle('pcb_short', 'pcb_short', indent=16))
        layout.addWidget(
            self._create_weight_row('pcb_short', self.pcb_defect_type_spinboxes['short'], parent_key='pcb_short')
        )
        layout.addWidget(self._create_toggle('pcb_missing_copper', 'pcb_missing_copper', indent=16))
        layout.addWidget(
            self._create_weight_row(
                'pcb_missing_copper',
                self.pcb_defect_type_spinboxes['missing_copper'],
                parent_key='pcb_missing_copper',
            )
        )
        layout.addWidget(self._create_toggle('pcb_excess_copper', 'pcb_excess_copper', indent=16))
        layout.addWidget(
            self._create_weight_row(
                'pcb_excess_copper',
                self.pcb_defect_type_spinboxes['excess_copper'],
                parent_key='pcb_excess_copper',
            )
        )
        layout.addWidget(self._create_toggle('pcb_pinhole', 'pcb_pinhole', indent=16))
        layout.addWidget(
            self._create_weight_row(
                'pcb_pinhole',
                self.pcb_defect_type_spinboxes['pinhole'],
                parent_key='pcb_pinhole',
            )
        )
        layout.addWidget(self._create_toggle('pcb_spurious_copper', 'pcb_spurious_copper', indent=16))
        layout.addWidget(
            self._create_weight_row(
                'pcb_spurious_copper',
                self.pcb_defect_type_spinboxes['spurious_copper'],
                parent_key='pcb_spurious_copper',
            )
        )
        layout.addWidget(self._create_toggle('pcb_via', 'pcb_via', indent=16))
        layout.addWidget(
            self._create_weight_row('pcb_via', self.pcb_defect_type_spinboxes['via'], parent_key='pcb_via')
        )
        layout.addWidget(self._create_toggle('pcb_misalignment', 'pcb_misalignment', indent=16))
        layout.addWidget(
            self._create_weight_row(
                'pcb_misalignment',
                self.pcb_defect_type_spinboxes['misalignment'],
                parent_key='pcb_misalignment',
            )
        )
        layout.addWidget(self._create_toggle('ic_line_break', 'ic_line_break', indent=16))
        layout.addWidget(
            self._create_weight_row('ic_line_break', self.ic_defect_type_spinboxes['line_break'], parent_key='ic_line_break')
        )
        layout.addWidget(self._create_toggle('ic_bridge', 'ic_bridge', indent=16))
        layout.addWidget(
            self._create_weight_row('ic_bridge', self.ic_defect_type_spinboxes['bridge'], parent_key='ic_bridge')
        )
        layout.addWidget(self._create_toggle('ic_necking', 'ic_necking', indent=16))
        layout.addWidget(
            self._create_weight_row('ic_necking', self.ic_defect_type_spinboxes['necking'], parent_key='ic_necking')
        )
        layout.addWidget(self._create_toggle('ic_missing_metal', 'ic_missing_metal', indent=16))
        layout.addWidget(
            self._create_weight_row(
                'ic_missing_metal',
                self.ic_defect_type_spinboxes['missing_metal'],
                parent_key='ic_missing_metal',
            )
        )
        layout.addWidget(self._create_toggle('ic_spur', 'ic_spur', indent=16))
        layout.addWidget(
            self._create_weight_row('ic_spur', self.ic_defect_type_spinboxes['spur'], parent_key='ic_spur')
        )
        layout.addWidget(self._create_toggle('ic_pinhole', 'ic_pinhole', indent=16))
        layout.addWidget(
            self._create_weight_row('ic_pinhole', self.ic_defect_type_spinboxes['pinhole'], parent_key='ic_pinhole')
        )
        layout.addWidget(self._create_toggle('ic_via_open', 'ic_via_open', indent=16))
        layout.addWidget(
            self._create_weight_row('ic_via_open', self.ic_defect_type_spinboxes['via_open'], parent_key='ic_via_open')
        )
        layout.addWidget(self._create_toggle('ic_line_shift', 'ic_line_shift', indent=16))
        layout.addWidget(
            self._create_weight_row('ic_line_shift', self.ic_defect_type_spinboxes['line_shift'], parent_key='ic_line_shift')
        )
        return group

    def _build_spatial_group(self) -> QGroupBox:
        group = QGroupBox(str(self._texts.get('spatial_group', 'Spatial augmentations')))
        layout = QVBoxLayout(group)
        layout.addWidget(self._create_toggle('rotate_90', 'rotate_90'))
        layout.addWidget(self._create_toggle('rotate_180', 'rotate_180'))
        layout.addWidget(self._create_toggle('flip_x', 'flip_x'))
        layout.addWidget(self._create_toggle('flip_y', 'flip_y'))
        layout.addWidget(self._create_toggle('random_crop', 'random_crop'))
        layout.addWidget(
            self._create_value_row('crops_per_image', self.crops_per_image_spinbox, parent_key='random_crop')
        )
        layout.addWidget(self._create_toggle('scale', 'scale'))
        layout.addWidget(
            self._create_value_row(
                'scale_augmentation_strength',
                self.scale_augmentation_strength_spinbox,
                parent_key='scale',
            )
        )
        return group

    def _build_photometric_group(self) -> QGroupBox:
        group = QGroupBox(str(self._texts.get('photometric_group', 'Photometric augmentations')))
        layout = QVBoxLayout(group)
        layout.addWidget(self._create_toggle('brightness', 'brightness'))
        layout.addWidget(
            self._create_value_row(
                'augmentation_brightness_strength',
                self.augmentation_brightness_spinbox,
                parent_key='brightness',
            )
        )
        layout.addWidget(self._create_toggle('contrast', 'contrast'))
        layout.addWidget(
            self._create_value_row(
                'augmentation_contrast_strength',
                self.augmentation_contrast_spinbox,
                parent_key='contrast',
            )
        )
        layout.addWidget(self._create_toggle('gamma', 'gamma'))
        layout.addWidget(
            self._create_value_row(
                'augmentation_gamma_strength',
                self.augmentation_gamma_spinbox,
                parent_key='gamma',
            )
        )
        layout.addWidget(self._create_toggle('noise', 'noise'))
        layout.addWidget(
            self._create_value_row(
                'augmentation_noise_probability',
                self.augmentation_noise_probability_spinbox,
                parent_key='noise',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'augmentation_noise_sigma',
                self.augmentation_noise_sigma_spinbox,
                parent_key='noise',
            )
        )
        layout.addWidget(self._create_toggle('blur', 'blur'))
        layout.addWidget(
            self._create_value_row(
                'augmentation_blur_probability',
                self.augmentation_blur_probability_spinbox,
                parent_key='blur',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'augmentation_blur_radius',
                self.augmentation_blur_radius_spinbox,
                parent_key='blur',
            )
        )
        return group

    def _build_mask_variation_group(self) -> QGroupBox:
        group = QGroupBox(str(self._texts.get('mask_variation_group', 'Mask variations')))
        layout = QVBoxLayout(group)
        layout.addWidget(
            self._create_value_row(
                'tech_aug_min_operations',
                self.tech_aug_min_operations_spinbox,
                parent_key='tech_config',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'tech_aug_max_operations',
                self.tech_aug_max_operations_spinbox,
                parent_key='tech_config',
            )
        )
        layout.addWidget(self._create_toggle('tech_global_width', 'tech_global_width'))
        layout.addWidget(
            self._create_value_row(
                'tech_aug_global_width_probability',
                self.tech_aug_global_width_probability_spinbox,
                parent_key='tech_global_width',
            )
        )
        layout.addWidget(self._create_toggle('tech_scale_rethreshold', 'tech_scale_rethreshold'))
        layout.addWidget(
            self._create_value_row(
                'tech_aug_scale_rethreshold_probability',
                self.tech_aug_scale_rethreshold_probability_spinbox,
                parent_key='tech_scale_rethreshold',
            )
        )
        layout.addWidget(self._create_toggle('tech_blur_threshold', 'tech_blur_threshold'))
        layout.addWidget(
            self._create_value_row(
                'tech_aug_blur_threshold_probability',
                self.tech_aug_blur_threshold_probability_spinbox,
                parent_key='tech_blur_threshold',
            )
        )
        layout.addWidget(self._create_toggle('tech_boundary_aware', 'tech_boundary_aware'))
        layout.addWidget(
            self._create_value_row(
                'tech_aug_boundary_aware_probability',
                self.tech_aug_boundary_aware_probability_spinbox,
                parent_key='tech_boundary_aware',
            )
        )
        layout.addWidget(self._create_toggle('tech_local_morphology', 'tech_local_morphology'))
        layout.addWidget(
            self._create_value_row(
                'tech_aug_local_morphology_probability',
                self.tech_aug_local_morphology_probability_spinbox,
                parent_key='tech_local_morphology',
            )
        )
        layout.addWidget(self._create_toggle('tech_gap_variation', 'tech_gap_variation'))
        layout.addWidget(
            self._create_value_row(
                'tech_aug_gap_variation_probability',
                self.tech_aug_gap_variation_probability_spinbox,
                parent_key='tech_gap_variation',
            )
        )
        return group

    def _build_batch_group(self) -> QGroupBox:
        group = QGroupBox(str(self._texts.get('batch_group', 'Batch augmentations')))
        layout = QVBoxLayout(group)
        layout.addWidget(self._create_toggle('cutout', 'cutout'))
        layout.addWidget(
            self._create_value_row('cutout_probability', self.cutout_probability_spinbox, parent_key='cutout')
        )
        layout.addWidget(
            self._create_value_row('cutout_holes', self.cutout_holes_spinbox, parent_key='cutout')
        )
        layout.addWidget(
            self._create_value_row('cutout_size_ratio', self.cutout_size_ratio_spinbox, parent_key='cutout')
        )
        layout.addWidget(self._create_toggle('random_artifacts', 'random_artifacts'))
        layout.addWidget(
            self._create_value_row(
                'random_artifacts_probability',
                self.random_artifacts_probability_spinbox,
                parent_key='random_artifacts',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'random_artifacts_count',
                self.random_artifacts_count_spinbox,
                parent_key='random_artifacts',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'random_artifacts_size_ratio',
                self.random_artifacts_size_ratio_spinbox,
                parent_key='random_artifacts',
            )
        )
        layout.addWidget(self._create_toggle('artifact_dust', 'artifact_dust', indent=16))
        layout.addWidget(self._create_toggle('artifact_resist_residue', 'artifact_resist_residue', indent=16))
        layout.addWidget(self._create_toggle('artifact_etch_residue', 'artifact_etch_residue', indent=16))
        layout.addWidget(self._create_toggle('artifact_particle_cluster', 'artifact_particle_cluster', indent=16))
        layout.addWidget(self._create_toggle('artifact_flake', 'artifact_flake', indent=16))
        layout.addWidget(self._create_toggle('mixup', 'mixup'))
        layout.addWidget(
            self._create_value_row('mixup_probability', self.mixup_probability_spinbox, parent_key='mixup')
        )
        layout.addWidget(
            self._create_value_row('mixup_alpha', self.mixup_alpha_spinbox, parent_key='mixup')
        )
        return group

    def _build_pcb_defects_group(self) -> QGroupBox:
        group = QGroupBox(str(self._texts.get('pcb_group', 'Synthetic PCB defects')))
        layout = QVBoxLayout(group)
        layout.addWidget(self._create_toggle('pcb_defects', 'pcb_group'))
        layout.addWidget(
            self._create_value_row(
                'pcb_defects_probability',
                self.pcb_defects_probability_spinbox,
                parent_key='pcb_defects',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'pcb_defects_min_count',
                self.pcb_defects_min_count_spinbox,
                parent_key='pcb_defects',
            )
        )
        layout.addWidget(
            self._create_value_row(
                'pcb_defects_max_count',
                self.pcb_defects_max_count_spinbox,
                parent_key='pcb_defects',
            )
        )
        layout.addWidget(self._create_toggle('pcb_break', 'pcb_break', indent=16))
        layout.addWidget(
            self._create_weight_row('pcb_break', self.pcb_defect_type_spinboxes['break'], parent_key='pcb_break')
        )
        layout.addWidget(self._create_toggle('pcb_short', 'pcb_short', indent=16))
        layout.addWidget(
            self._create_weight_row('pcb_short', self.pcb_defect_type_spinboxes['short'], parent_key='pcb_short')
        )
        layout.addWidget(self._create_toggle('pcb_missing_copper', 'pcb_missing_copper', indent=16))
        layout.addWidget(
            self._create_weight_row(
                'pcb_missing_copper',
                self.pcb_defect_type_spinboxes['missing_copper'],
                parent_key='pcb_missing_copper',
            )
        )
        layout.addWidget(self._create_toggle('pcb_excess_copper', 'pcb_excess_copper', indent=16))
        layout.addWidget(
            self._create_weight_row(
                'pcb_excess_copper',
                self.pcb_defect_type_spinboxes['excess_copper'],
                parent_key='pcb_excess_copper',
            )
        )
        layout.addWidget(self._create_toggle('pcb_pinhole', 'pcb_pinhole', indent=16))
        layout.addWidget(
            self._create_weight_row(
                'pcb_pinhole',
                self.pcb_defect_type_spinboxes['pinhole'],
                parent_key='pcb_pinhole',
            )
        )
        layout.addWidget(self._create_toggle('pcb_spurious_copper', 'pcb_spurious_copper', indent=16))
        layout.addWidget(
            self._create_weight_row(
                'pcb_spurious_copper',
                self.pcb_defect_type_spinboxes['spurious_copper'],
                parent_key='pcb_spurious_copper',
            )
        )
        layout.addWidget(self._create_toggle('pcb_via', 'pcb_via', indent=16))
        layout.addWidget(
            self._create_weight_row('pcb_via', self.pcb_defect_type_spinboxes['via'], parent_key='pcb_via')
        )
        layout.addWidget(self._create_toggle('pcb_misalignment', 'pcb_misalignment', indent=16))
        layout.addWidget(
            self._create_weight_row(
                'pcb_misalignment',
                self.pcb_defect_type_spinboxes['misalignment'],
                parent_key='pcb_misalignment',
            )
        )
        return group

    def _create_toggle(
        self,
        key: str,
        text_key: str,
        *,
        indent: int = 0,
    ) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(indent, 0, 0, 0)
        layout.setSpacing(6)
        checkbox = QCheckBox('')
        tooltip = self._resolve_tip(text_key)
        checkbox.setToolTip(tooltip)
        self._toggle_boxes[key] = checkbox
        layout.addWidget(checkbox)
        label = QLabel(self._resolve_text(text_key))
        label.setToolTip(tooltip)
        label.setWordWrap(True)
        layout.addWidget(label, 1)
        layout.addStretch(1)
        return container

    def _create_value_row(
        self,
        text_key: str,
        *widgets: QWidget,
        parent_key: str | None = None,
        indent: int = 16,
    ) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(indent, 0, 0, 0)
        layout.setSpacing(6)
        tooltip = self._resolve_tip(text_key)
        label = QLabel(self._resolve_setting_text(text_key))
        label.setToolTip(tooltip)
        label.setWordWrap(True)
        layout.addWidget(label, 1)
        for widget in widgets:
            widget.setToolTip(widget.toolTip() or tooltip)
            layout.addWidget(widget)
        if parent_key is not None:
            self._register_value_row(parent_key, container, *widgets)
        return container

    def _create_range_row(
        self,
        text_key: str,
        min_widget: QWidget,
        max_widget: QWidget,
        *,
        parent_key: str | None = None,
        indent: int = 16,
    ) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(indent, 0, 0, 0)
        layout.setSpacing(6)
        tooltip = self._resolve_tip(text_key)
        label = QLabel(self._resolve_setting_text(text_key))
        label.setToolTip(tooltip)
        label.setWordWrap(True)
        min_label = QLabel('Min' if not self._is_russian_ui else 'Мин')
        max_label = QLabel('Max' if not self._is_russian_ui else 'Макс')
        min_label.setToolTip(tooltip)
        max_label.setToolTip(tooltip)
        layout.addWidget(label, 1)
        min_widget.setToolTip(min_widget.toolTip() or tooltip)
        max_widget.setToolTip(max_widget.toolTip() or tooltip)
        layout.addWidget(min_label)
        layout.addWidget(min_widget)
        layout.addWidget(max_label)
        layout.addWidget(max_widget)
        if parent_key is not None:
            self._register_value_row(parent_key, container, min_widget, max_widget)
        return container

    def _create_weight_row(
        self,
        base_text_key: str,
        widget: QWidget,
        *,
        parent_key: str,
        indent: int = 32,
    ) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(indent, 0, 0, 0)
        layout.setSpacing(6)
        severity_key = f'{base_text_key}_severity'
        tooltip = self._resolve_tip(severity_key)
        label = QLabel(self._resolve_setting_text(severity_key))
        label.setToolTip(tooltip)
        label.setWordWrap(True)
        layout.addWidget(label, 1)
        widget.setToolTip(widget.toolTip() or tooltip)
        layout.addWidget(widget)
        self._register_value_row(parent_key, container, widget)
        return container

    def _register_value_row(self, parent_key: str, container: QWidget, *widgets: QWidget) -> None:
        self._value_rows.setdefault(parent_key, []).append(container)
        self._value_widgets.setdefault(parent_key, []).extend(widgets)

    def _initialize_toggle_states(self) -> None:
        generation = self._training_parameters.generation
        tech_aug = build_tech_augmentation_config(getattr(generation, 'tech_aug', None))
        synthetic_generator = build_synthetic_defect_generator_parameters(
            getattr(self._training_parameters, 'synthetic_defect_generator', None)
        )
        pcb_defects = synthetic_generator.pcb_defects
        ic_defects = synthetic_generator.ic_defects
        self._toggle_boxes['rotate_90'].setChecked(bool(getattr(generation, 'horizontal_rotation', False)))
        self._toggle_boxes['rotate_180'].setChecked(bool(getattr(generation, 'vertical_rotation', False)))
        self._toggle_boxes['flip_x'].setChecked(bool(getattr(generation, 'flip_x', False)))
        self._toggle_boxes['flip_y'].setChecked(bool(getattr(generation, 'flip_y', False)))
        self._toggle_boxes['random_crop'].setChecked(bool(getattr(generation, 'random_crop', False)))
        self._toggle_boxes['scale'].setChecked(bool(getattr(generation, 'scale_augmentation', False)))
        self._toggle_boxes['synthetic_topology'].setChecked(bool(synthetic_generator.enabled))
        photo_enabled = bool(getattr(generation, 'additional_augmentation', False))
        self._toggle_boxes['brightness'].setChecked(
            photo_enabled and float(getattr(generation, 'augmentation_brightness_strength', 0.0)) > 0.0
        )
        self._toggle_boxes['contrast'].setChecked(
            photo_enabled and float(getattr(generation, 'augmentation_contrast_strength', 0.0)) > 0.0
        )
        self._toggle_boxes['gamma'].setChecked(
            photo_enabled and float(getattr(generation, 'augmentation_gamma_strength', 0.0)) > 0.0
        )
        self._toggle_boxes['noise'].setChecked(
            photo_enabled
            and float(getattr(generation, 'augmentation_noise_probability', 0.0)) > 0.0
            and float(getattr(generation, 'augmentation_noise_sigma', 0.0)) > 0.0
        )
        self._toggle_boxes['blur'].setChecked(
            photo_enabled
            and float(getattr(generation, 'augmentation_blur_probability', 0.0)) > 0.0
            and float(getattr(generation, 'augmentation_blur_radius', 0.0)) > 0.0
        )
        if 'tech_global_width' in self._toggle_boxes:
            self._toggle_boxes['tech_global_width'].setChecked(
                bool(tech_aug.enabled) and float(tech_aug.global_width.probability) > 0.0
            )
            self._toggle_boxes['tech_scale_rethreshold'].setChecked(
                bool(tech_aug.enabled) and float(tech_aug.scale_rethreshold.probability) > 0.0
            )
            self._toggle_boxes['tech_blur_threshold'].setChecked(
                bool(tech_aug.enabled) and float(tech_aug.blur_threshold.probability) > 0.0
            )
            self._toggle_boxes['tech_boundary_aware'].setChecked(
                bool(tech_aug.enabled) and float(tech_aug.boundary_aware.probability) > 0.0
            )
            self._toggle_boxes['tech_local_morphology'].setChecked(
                bool(tech_aug.enabled) and float(tech_aug.local_morphology.probability) > 0.0
            )
            self._toggle_boxes['tech_gap_variation'].setChecked(
                bool(tech_aug.enabled) and float(tech_aug.gap_variation.probability) > 0.0
            )
        cutout = getattr(self._training_parameters, 'cutout', None)
        self._toggle_boxes['cutout'].setChecked(
            bool(getattr(cutout, 'enabled', False))
            and float(getattr(cutout, 'probability', 0.0)) > 0.0
            and float(getattr(cutout, 'size_ratio', 0.0)) > 0.0
        )
        random_artifacts = getattr(self._training_parameters, 'random_artifacts', None)
        random_artifacts_enabled = (
            bool(getattr(random_artifacts, 'enabled', False))
            and float(getattr(random_artifacts, 'probability', 0.0)) > 0.0
            and float(getattr(random_artifacts, 'size_ratio', 0.0)) > 0.0
        )
        self._toggle_boxes['random_artifacts'].setChecked(random_artifacts_enabled)
        self._toggle_boxes['artifact_dust'].setChecked(
            random_artifacts_enabled and bool(getattr(random_artifacts, 'dust_enabled', True))
        )
        self._toggle_boxes['artifact_resist_residue'].setChecked(
            random_artifacts_enabled and bool(getattr(random_artifacts, 'resist_residue_enabled', True))
        )
        self._toggle_boxes['artifact_etch_residue'].setChecked(
            random_artifacts_enabled and bool(getattr(random_artifacts, 'etch_residue_enabled', True))
        )
        self._toggle_boxes['artifact_particle_cluster'].setChecked(
            random_artifacts_enabled and bool(getattr(random_artifacts, 'particle_cluster_enabled', True))
        )
        self._toggle_boxes['artifact_flake'].setChecked(
            random_artifacts_enabled and bool(getattr(random_artifacts, 'flake_enabled', True))
        )
        mixup = getattr(self._training_parameters, 'mixup', None)
        self._toggle_boxes['mixup'].setChecked(
            bool(getattr(mixup, 'enabled', False))
            and float(getattr(mixup, 'probability', 0.0)) > 0.0
            and float(getattr(mixup, 'alpha', 0.0)) > 0.0
        )
        active_defects = ic_defects if synthetic_generator.topology_domain == 'ic' else pcb_defects
        defect_probabilities = dict(getattr(pcb_defects, 'defect_probabilities', {}))
        ic_defect_probabilities = dict(getattr(ic_defects, 'defect_probabilities', {}))
        self._toggle_boxes['pcb_defects'].setChecked(
            bool(active_defects.enabled) and float(getattr(active_defects, 'defect_probability', 0.0)) > 0.0
        )
        self._toggle_boxes['pcb_break'].setChecked(
            bool(pcb_defects.enabled) and float(defect_probabilities.get('break', 0.0)) > 0.0
        )
        self._toggle_boxes['pcb_short'].setChecked(
            bool(pcb_defects.enabled) and float(defect_probabilities.get('short', 0.0)) > 0.0
        )
        self._toggle_boxes['pcb_missing_copper'].setChecked(
            bool(pcb_defects.enabled) and float(defect_probabilities.get('missing_copper', 0.0)) > 0.0
        )
        self._toggle_boxes['pcb_excess_copper'].setChecked(
            bool(pcb_defects.enabled) and float(defect_probabilities.get('excess_copper', 0.0)) > 0.0
        )
        self._toggle_boxes['pcb_pinhole'].setChecked(
            bool(pcb_defects.enabled) and float(defect_probabilities.get('pinhole', 0.0)) > 0.0
        )
        self._toggle_boxes['pcb_spurious_copper'].setChecked(
            bool(pcb_defects.enabled) and float(defect_probabilities.get('spurious_copper', 0.0)) > 0.0
        )
        self._toggle_boxes['pcb_via'].setChecked(
            bool(pcb_defects.enabled) and float(defect_probabilities.get('via', 0.0)) > 0.0
        )
        self._toggle_boxes['pcb_misalignment'].setChecked(
            bool(pcb_defects.enabled) and float(defect_probabilities.get('misalignment', 0.0)) > 0.0
        )
        for defect_name, toggle_key in (
            ('line_break', 'ic_line_break'),
            ('bridge', 'ic_bridge'),
            ('necking', 'ic_necking'),
            ('missing_metal', 'ic_missing_metal'),
            ('spur', 'ic_spur'),
            ('pinhole', 'ic_pinhole'),
            ('via_open', 'ic_via_open'),
            ('line_shift', 'ic_line_shift'),
        ):
            self._toggle_boxes[toggle_key].setChecked(
                bool(ic_defects.enabled) and float(ic_defect_probabilities.get(defect_name, 0.0)) > 0.0
            )

    def _connect_signals(self) -> None:
        self.prev_button.clicked.connect(self._show_previous_sample)
        self.next_button.clicked.connect(self._show_next_sample)
        self.resample_button.clicked.connect(self._resample_current_sample)
        self.apply_to_main_button.clicked.connect(self._emit_apply_to_main)
        self.sample_list_widget.currentRowChanged.connect(self._on_sample_list_row_changed)
        self.image_preview.middle_pressed.connect(self._show_original_preview)
        self.image_preview.middle_released.connect(self._restore_augmented_preview)
        self.label_preview.middle_pressed.connect(self._show_original_preview)
        self.label_preview.middle_released.connect(self._restore_augmented_preview)
        self.full_image_check_box.toggled.connect(self._on_full_image_toggled)
        self.synthetic_topology_domain_combo.currentIndexChanged.connect(self._on_value_changed)
        self.pcb_topology_family_combo.currentIndexChanged.connect(self._on_value_changed)
        self.ic_topology_family_combo.currentIndexChanged.connect(self._on_value_changed)
        for checkbox in self._toggle_boxes.values():
            checkbox.toggled.connect(self._on_toggle_changed)
        connected_widgets: set[int] = set()
        for widgets in self._value_widgets.values():
            for widget in widgets:
                widget_id = id(widget)
                if widget_id in connected_widgets:
                    continue
                connected_widgets.add(widget_id)
                if hasattr(widget, 'valueChanged'):
                    widget.valueChanged.connect(self._on_value_changed)
                elif hasattr(widget, 'currentIndexChanged'):
                    widget.currentIndexChanged.connect(self._on_value_changed)
        self.tech_aug_min_operations_spinbox.valueChanged.connect(self._on_value_changed)
        self.tech_aug_max_operations_spinbox.valueChanged.connect(self._on_value_changed)
        self.synthetic_image_width_spinbox.valueChanged.connect(self._on_value_changed)
        self.synthetic_image_height_spinbox.valueChanged.connect(self._on_value_changed)
        self.pcb_defects_min_count_spinbox.valueChanged.connect(self._on_pcb_defect_count_changed)
        self.pcb_defects_max_count_spinbox.valueChanged.connect(self._on_pcb_defect_count_changed)

    def _sync_group_boxes(self) -> None:
        self._set_value_widgets_enabled('synthetic_topology', self._toggle_boxes['synthetic_topology'].isChecked())
        self._set_value_widgets_enabled('random_crop', self._toggle_boxes['random_crop'].isChecked())
        self._set_value_widgets_enabled('scale', self._toggle_boxes['scale'].isChecked())
        self._set_value_widgets_enabled('brightness', self._toggle_boxes['brightness'].isChecked())
        self._set_value_widgets_enabled('contrast', self._toggle_boxes['contrast'].isChecked())
        self._set_value_widgets_enabled('gamma', self._toggle_boxes['gamma'].isChecked())
        self._set_value_widgets_enabled('noise', self._toggle_boxes['noise'].isChecked())
        self._set_value_widgets_enabled('blur', self._toggle_boxes['blur'].isChecked())

        if 'tech_global_width' in self._toggle_boxes:
            tech_enabled = self._has_selected_tech_variations()
            self._set_value_widgets_enabled('tech_config', tech_enabled)
            for key in (
                'tech_global_width',
                'tech_scale_rethreshold',
                'tech_blur_threshold',
                'tech_boundary_aware',
                'tech_local_morphology',
                'tech_gap_variation',
            ):
                self._set_value_widgets_enabled(key, self._toggle_boxes[key].isChecked())

        self._set_value_widgets_enabled('cutout', self._toggle_boxes['cutout'].isChecked())
        self._set_value_widgets_enabled('mixup', self._toggle_boxes['mixup'].isChecked())
        artifacts_enabled = self._toggle_boxes['random_artifacts'].isChecked()
        self._set_value_widgets_enabled('random_artifacts', artifacts_enabled)
        for key in (
            'artifact_dust',
            'artifact_resist_residue',
            'artifact_etch_residue',
            'artifact_particle_cluster',
            'artifact_flake',
        ):
            self._toggle_boxes[key].setEnabled(artifacts_enabled)

        pcb_enabled = self._toggle_boxes['pcb_defects'].isChecked()
        self._set_value_widgets_enabled('pcb_defects', pcb_enabled)
        current_domain = self._get_synthetic_topology_domain()
        self._set_value_widgets_enabled('synthetic_topology', self._toggle_boxes['synthetic_topology'].isChecked())
        self._set_value_widgets_enabled('pcb_topology_family', current_domain == 'pcb')
        self._set_value_widgets_enabled('ic_topology_family', current_domain == 'ic')
        self._set_value_widgets_visible('pcb_topology_family', current_domain == 'pcb')
        self._set_value_widgets_visible('ic_topology_family', current_domain == 'ic')
        for key in (
            'pcb_break',
            'pcb_short',
            'pcb_missing_copper',
            'pcb_excess_copper',
            'pcb_pinhole',
            'pcb_spurious_copper',
            'pcb_via',
            'pcb_misalignment',
        ):
            visible = current_domain == 'pcb'
            self._toggle_boxes[key].setVisible(visible)
            self._toggle_boxes[key].setEnabled(pcb_enabled and visible)
            self._set_value_widgets_visible(key, visible)
            self._set_value_widgets_enabled(key, pcb_enabled and visible and self._toggle_boxes[key].isChecked())
        for key in (
            'ic_line_break',
            'ic_bridge',
            'ic_necking',
            'ic_missing_metal',
            'ic_spur',
            'ic_pinhole',
            'ic_via_open',
            'ic_line_shift',
        ):
            visible = current_domain == 'ic'
            self._toggle_boxes[key].setVisible(visible)
            self._toggle_boxes[key].setEnabled(pcb_enabled and visible)
            self._set_value_widgets_visible(key, visible)
            self._set_value_widgets_enabled(key, pcb_enabled and visible and self._toggle_boxes[key].isChecked())

    def _on_toggle_changed(self, _checked: bool) -> None:
        self._sync_group_boxes()
        self._refresh_preview()

    def _on_value_changed(self, _value: object) -> None:
        self._sync_group_boxes()
        self._refresh_preview()

    def _on_pcb_defect_count_changed(self, _value: int) -> None:
        min_count = int(self.pcb_defects_min_count_spinbox.value())
        max_count = int(self.pcb_defects_max_count_spinbox.value())
        if min_count > max_count:
            if self.sender() is self.pcb_defects_min_count_spinbox:
                self.pcb_defects_max_count_spinbox.setValue(min_count)
            else:
                self.pcb_defects_min_count_spinbox.setValue(max_count)
            return
        self._on_value_changed(_value)

    def _on_full_image_toggled(self, _checked: bool) -> None:
        self._refresh_preview()

    def _emit_apply_to_main(self) -> None:
        self.apply_to_main_requested.emit(self._build_apply_payload())

    def _populate_sample_list(self) -> None:
        self._sample_list_updating = True
        self.sample_list_widget.clear()
        for sample_path, _label_path in self._sample_pairs:
            item = QListWidgetItem(sample_path.name)
            item.setToolTip(str(sample_path))
            self.sample_list_widget.addItem(item)
        synthetic_enabled = bool(self._toggle_boxes.get('synthetic_topology') and self._toggle_boxes['synthetic_topology'].isChecked())
        self.sample_list_widget.setEnabled(bool(self._sample_pairs) and not synthetic_enabled)
        if self._sample_pairs:
            self.sample_list_widget.setCurrentRow(self._current_sample_index)
        self._sample_list_updating = False

    def _sync_sample_list_selection(self) -> None:
        if self.sample_list_widget.count() != len(self._sample_pairs):
            self._populate_sample_list()
            return
        self._sample_list_updating = True
        self.sample_list_widget.setEnabled(bool(self._sample_pairs) and not self._toggle_boxes['synthetic_topology'].isChecked())
        if self._sample_pairs and self.sample_list_widget.currentRow() != self._current_sample_index:
            self.sample_list_widget.setCurrentRow(self._current_sample_index)
        elif not self._sample_pairs and self.sample_list_widget.currentRow() != -1:
            self.sample_list_widget.setCurrentRow(-1)
        self._sample_list_updating = False

    def _on_sample_list_row_changed(self, row: int) -> None:
        if self._sample_list_updating or not self._sample_pairs:
            return
        if row < 0 or row >= len(self._sample_pairs) or row == self._current_sample_index:
            return
        self._current_sample_index = row
        self._variant_serial = 0
        self._refresh_preview()

    def _build_apply_payload(self) -> dict[str, object]:
        brightness_enabled = self._toggle_boxes['brightness'].isChecked()
        contrast_enabled = self._toggle_boxes['contrast'].isChecked()
        gamma_enabled = self._toggle_boxes['gamma'].isChecked()
        noise_enabled = self._toggle_boxes['noise'].isChecked()
        blur_enabled = self._toggle_boxes['blur'].isChecked()

        return {
            'horizontal_rotation': self._toggle_boxes['rotate_90'].isChecked(),
            'vertical_rotation': self._toggle_boxes['rotate_180'].isChecked(),
            'flip_x': self._toggle_boxes['flip_x'].isChecked(),
            'flip_y': self._toggle_boxes['flip_y'].isChecked(),
            'random_crop': self._toggle_boxes['random_crop'].isChecked(),
            'crops_per_image': int(self.crops_per_image_spinbox.value()),
            'scale_augmentation': self._toggle_boxes['scale'].isChecked(),
            'scale_augmentation_strength': float(self.scale_augmentation_strength_spinbox.value()),
            'additional_augmentation': any(
                (
                    brightness_enabled,
                    contrast_enabled,
                    gamma_enabled,
                    noise_enabled,
                    blur_enabled,
                )
            ),
            'augmentation_brightness_strength': (
                float(self.augmentation_brightness_spinbox.value()) if brightness_enabled else 0.0
            ),
            'augmentation_contrast_strength': (
                float(self.augmentation_contrast_spinbox.value()) if contrast_enabled else 0.0
            ),
            'augmentation_gamma_strength': (
                float(self.augmentation_gamma_spinbox.value()) if gamma_enabled else 0.0
            ),
            'augmentation_noise_probability': (
                float(self.augmentation_noise_probability_spinbox.value()) if noise_enabled else 0.0
            ),
            'augmentation_noise_sigma': (
                float(self.augmentation_noise_sigma_spinbox.value()) if noise_enabled else 0.0
            ),
            'augmentation_blur_probability': (
                float(self.augmentation_blur_probability_spinbox.value()) if blur_enabled else 0.0
            ),
            'augmentation_blur_radius': (
                float(self.augmentation_blur_radius_spinbox.value()) if blur_enabled else 0.0
            ),
            'synthetic_defect_generator': self._build_apply_synthetic_defect_generator_config(),
            'tech_aug': build_tech_augmentation_config(None),
            'cutout_enabled': self._toggle_boxes['cutout'].isChecked(),
            'cutout_probability': float(self.cutout_probability_spinbox.value()),
            'cutout_holes': int(self.cutout_holes_spinbox.value()),
            'cutout_size_ratio': float(self.cutout_size_ratio_spinbox.value()),
            'random_artifacts_enabled': self._toggle_boxes['random_artifacts'].isChecked(),
            'random_artifacts_probability': float(self.random_artifacts_probability_spinbox.value()),
            'random_artifacts_count': int(self.random_artifacts_count_spinbox.value()),
            'random_artifacts_size_ratio': float(self.random_artifacts_size_ratio_spinbox.value()),
            'random_artifacts_dust_enabled': self._toggle_boxes['artifact_dust'].isChecked(),
            'random_artifacts_resist_residue_enabled': self._toggle_boxes['artifact_resist_residue'].isChecked(),
            'random_artifacts_etch_residue_enabled': self._toggle_boxes['artifact_etch_residue'].isChecked(),
            'random_artifacts_particle_cluster_enabled': self._toggle_boxes['artifact_particle_cluster'].isChecked(),
            'random_artifacts_flake_enabled': self._toggle_boxes['artifact_flake'].isChecked(),
            'mixup_enabled': self._toggle_boxes['mixup'].isChecked(),
            'mixup_probability': float(self.mixup_probability_spinbox.value()),
            'mixup_alpha': float(self.mixup_alpha_spinbox.value()),
            'pcb_defects': build_pcb_defect_parameters(None),
        }

    def _build_apply_synthetic_defect_generator_config(self):
        config = copy.deepcopy(
            build_synthetic_defect_generator_parameters(
                getattr(self._training_parameters, 'synthetic_defect_generator', None)
            )
        )
        config.enabled = self._toggle_boxes['synthetic_topology'].isChecked()
        config.topology_domain = self._get_synthetic_topology_domain()
        config.topology_family = self._get_synthetic_topology_family()
        config.image_size_xy = (
            int(self.synthetic_image_width_spinbox.value()),
            int(self.synthetic_image_height_spinbox.value()),
        )
        config.trace_count_range = tuple(
            sorted(
                (
                    int(self.synthetic_trace_count_min_spinbox.value()),
                    int(self.synthetic_trace_count_max_spinbox.value()),
                )
            )
        )
        config.segment_count_range = tuple(
            sorted(
                (
                    int(self.synthetic_segment_count_min_spinbox.value()),
                    int(self.synthetic_segment_count_max_spinbox.value()),
                )
            )
        )
        config.trace_half_width_range = tuple(
            sorted(
                (
                    int(self.synthetic_trace_half_width_min_spinbox.value()),
                    int(self.synthetic_trace_half_width_max_spinbox.value()),
                )
            )
        )
        config.background_noise_sigma_range = tuple(
            sorted(
                (
                    float(self.synthetic_background_noise_sigma_min_spinbox.value()),
                    float(self.synthetic_background_noise_sigma_max_spinbox.value()),
                )
            )
        )
        config.trace_noise_sigma_range = tuple(
            sorted(
                (
                    float(self.synthetic_trace_noise_sigma_min_spinbox.value()),
                    float(self.synthetic_trace_noise_sigma_max_spinbox.value()),
                )
            )
        )
        config.pcb_defects = self._build_apply_pcb_defects_config()
        config.ic_defects = self._build_apply_ic_defects_config()
        config.defects = config.ic_defects if config.topology_domain == 'ic' else config.pcb_defects
        return config

    def _build_apply_tech_aug_config(self):
        config = copy.deepcopy(build_tech_augmentation_config(getattr(self._training_parameters.generation, 'tech_aug', None)))
        min_operations = int(self.tech_aug_min_operations_spinbox.value())
        max_operations = int(self.tech_aug_max_operations_spinbox.value())
        if min_operations > max_operations:
            min_operations, max_operations = max_operations, min_operations
        config.enabled = self._has_selected_tech_variations()
        config.min_operations = min_operations
        config.max_operations = max_operations
        config.global_width.probability = (
            float(self.tech_aug_global_width_probability_spinbox.value())
            if self._toggle_boxes['tech_global_width'].isChecked()
            else 0.0
        )
        config.scale_rethreshold.probability = (
            float(self.tech_aug_scale_rethreshold_probability_spinbox.value())
            if self._toggle_boxes['tech_scale_rethreshold'].isChecked()
            else 0.0
        )
        config.blur_threshold.probability = (
            float(self.tech_aug_blur_threshold_probability_spinbox.value())
            if self._toggle_boxes['tech_blur_threshold'].isChecked()
            else 0.0
        )
        config.boundary_aware.probability = (
            float(self.tech_aug_boundary_aware_probability_spinbox.value())
            if self._toggle_boxes['tech_boundary_aware'].isChecked()
            else 0.0
        )
        config.local_morphology.probability = (
            float(self.tech_aug_local_morphology_probability_spinbox.value())
            if self._toggle_boxes['tech_local_morphology'].isChecked()
            else 0.0
        )
        config.gap_variation.probability = (
            float(self.tech_aug_gap_variation_probability_spinbox.value())
            if self._toggle_boxes['tech_gap_variation'].isChecked()
            else 0.0
        )
        return config

    def _build_apply_pcb_defects_config(self):
        if not self._toggle_boxes['synthetic_topology'].isChecked():
            config = build_pcb_defect_parameters(None)
            config.enabled = False
            return config
        synthetic_generator = build_synthetic_defect_generator_parameters(
            getattr(self._training_parameters, 'synthetic_defect_generator', None)
        )
        config = copy.deepcopy(synthetic_generator.pcb_defects)
        min_defects = int(self.pcb_defects_min_count_spinbox.value())
        max_defects = int(self.pcb_defects_max_count_spinbox.value())
        if min_defects > max_defects:
            min_defects, max_defects = max_defects, min_defects
        config.enabled = self._toggle_boxes['pcb_defects'].isChecked()
        config.defect_probability = float(self.pcb_defects_probability_spinbox.value())
        config.min_defects = min_defects
        config.max_defects = max_defects
        for defect_name, checkbox_key in (
            ('break', 'pcb_break'),
            ('short', 'pcb_short'),
            ('missing_copper', 'pcb_missing_copper'),
            ('excess_copper', 'pcb_excess_copper'),
            ('pinhole', 'pcb_pinhole'),
            ('spurious_copper', 'pcb_spurious_copper'),
            ('via', 'pcb_via'),
            ('misalignment', 'pcb_misalignment'),
        ):
            config.defect_probabilities[defect_name] = (
                1.0 if self._toggle_boxes[checkbox_key].isChecked() else 0.0
            )
            config.defect_severities[defect_name] = float(self.pcb_defect_type_spinboxes[defect_name].value()) / 100.0
        return config

    def _build_apply_ic_defects_config(self):
        if not self._toggle_boxes['synthetic_topology'].isChecked():
            config = build_ic_defect_parameters(None)
            config.enabled = False
            return config
        synthetic_generator = build_synthetic_defect_generator_parameters(
            getattr(self._training_parameters, 'synthetic_defect_generator', None)
        )
        config = copy.deepcopy(synthetic_generator.ic_defects)
        min_defects = int(self.pcb_defects_min_count_spinbox.value())
        max_defects = int(self.pcb_defects_max_count_spinbox.value())
        if min_defects > max_defects:
            min_defects, max_defects = max_defects, min_defects
        config.enabled = self._toggle_boxes['pcb_defects'].isChecked()
        config.defect_probability = float(self.pcb_defects_probability_spinbox.value())
        config.min_defects = min_defects
        config.max_defects = max_defects
        for defect_name, checkbox_key in (
            ('line_break', 'ic_line_break'),
            ('bridge', 'ic_bridge'),
            ('necking', 'ic_necking'),
            ('missing_metal', 'ic_missing_metal'),
            ('spur', 'ic_spur'),
            ('pinhole', 'ic_pinhole'),
            ('via_open', 'ic_via_open'),
            ('line_shift', 'ic_line_shift'),
        ):
            config.defect_probabilities[defect_name] = (
                1.0 if self._toggle_boxes[checkbox_key].isChecked() else 0.0
            )
            config.defect_severities[defect_name] = float(self.ic_defect_type_spinboxes[defect_name].value()) / 100.0
        return config

    def _show_previous_sample(self) -> None:
        if not self._sample_pairs:
            return
        self._current_sample_index = (self._current_sample_index - 1) % len(self._sample_pairs)
        self._variant_serial = 0
        self._refresh_preview()

    def _show_next_sample(self) -> None:
        if not self._sample_pairs:
            return
        self._current_sample_index = (self._current_sample_index + 1) % len(self._sample_pairs)
        self._variant_serial = 0
        self._refresh_preview()

    def _resample_current_sample(self) -> None:
        self._variant_serial += 1
        self._refresh_preview()

    def _show_original_preview(self) -> None:
        self._show_augmented = False
        self._update_preview_mode_label()
        self._update_visible_preview()

    def _restore_augmented_preview(self) -> None:
        self._show_augmented = True
        self._update_preview_mode_label()
        self._update_visible_preview()

    def _refresh_preview(self) -> None:
        synthetic_enabled = self._toggle_boxes['synthetic_topology'].isChecked()
        if not self._sample_pairs and not synthetic_enabled:
            self._sync_sample_list_selection()
            error_text = str(
                self._load_error
                or self._texts.get('empty_error', 'No matched sample/label pairs were found.')
            )
            self.sample_label.setText(error_text)
            self.status_label.setText(error_text)
            self.image_preview.setText(error_text)
            self.label_preview.setText(error_text)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.resample_button.setEnabled(False)
            self._update_preview_mode_label()
            return

        self._sync_sample_list_selection()
        if synthetic_enabled:
            self.sample_label.setText(
                str(
                    self._texts.get(
                        'synthetic_label',
                        'Синтетическая топология' if self._is_russian_ui else 'Synthetic topology',
                    )
                )
            )
        else:
            sample_path, _label_path = self._sample_pairs[self._current_sample_index]
            self.sample_label.setText(
                str(
                    self._texts.get('sample_label_template', '{index}/{total}: {name}')
                ).format(
                    index=self._current_sample_index + 1,
                    total=len(self._sample_pairs),
                    name=sample_path.name,
                )
            )
        self.status_label.setText(
            str(
                self._texts.get(
                    'status_hold_template',
                    'Variant #{variant}. Preview: {preview_mode}. Hold the middle mouse button to inspect the original image and label.',
                )
            ).format(
                variant=self._variant_serial + 1,
                preview_mode=self._preview_mode_text(),
            )
        )
        original_image, original_label, augmented_image, augmented_label = self._build_preview_arrays(
            self._current_sample_index
        )
        self._original_image_array = original_image
        self._original_label_array = original_label
        self._augmented_image_array = augmented_image
        self._augmented_label_array = augmented_label
        self.prev_button.setEnabled((not synthetic_enabled) and len(self._sample_pairs) > 1)
        self.next_button.setEnabled((not synthetic_enabled) and len(self._sample_pairs) > 1)
        self.resample_button.setEnabled(True)
        self._update_preview_mode_label()
        self._update_visible_preview()

    def _update_preview_mode_label(self) -> None:
        mode_key = 'mode_augmented' if self._show_augmented else 'mode_original'
        self.mode_label.setText(
            str(
                self._texts.get(
                    'mode_hold_template',
                    'Mode: {mode}. Preview: {preview_mode}. Hold the middle mouse button to inspect the original image and label.',
                )
            ).format(
                mode=str(self._texts.get(mode_key, 'Augmented')),
                preview_mode=self._preview_mode_text(),
            )
        )

    def _update_visible_preview(self) -> None:
        if self._show_augmented:
            image_array = self._augmented_image_array
            label_array = self._augmented_label_array
        else:
            image_array = self._original_image_array
            label_array = self._original_label_array
        target_width = max(1, min(self.image_preview.width(), self.label_preview.width()))
        target_height = max(1, min(self.image_preview.height(), self.label_preview.height()))
        self._set_preview_image(self.image_preview, image_array, target_width=target_width, target_height=target_height)
        self._set_preview_image(self.label_preview, label_array, target_width=target_width, target_height=target_height)

    def _build_preview_arrays(
        self,
        sample_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._toggle_boxes['synthetic_topology'].isChecked():
            base_image, base_label = self._build_synthetic_base_arrays()
            if self.full_image_check_box.isChecked():
                original_image, original_label = self._build_original_full_image(base_image, base_label)
                augmented_image, augmented_label = self._build_augmented_synthetic_pair(
                    sample_index,
                    base_image,
                    base_label,
                    full_image=True,
                )
            else:
                original_image, original_label = self._build_original_patch(
                    sample_index,
                    base_image,
                    base_label,
                )
                augmented_image, augmented_label = self._build_augmented_synthetic_pair(
                    sample_index,
                    base_image,
                    base_label,
                    full_image=False,
                )
        else:
            base_image, base_label = self._load_prepared_arrays(sample_index)
            if self.full_image_check_box.isChecked():
                original_image, original_label = self._build_original_full_image(base_image, base_label)
                augmented_image, augmented_label = self._build_augmented_full_image(
                    sample_index,
                    base_image,
                    base_label,
                    include_mixup=True,
                )
            else:
                original_image, original_label = self._build_original_patch(
                    sample_index,
                    base_image,
                    base_label,
                )
                augmented_image, augmented_label = self._build_augmented_patch(
                    sample_index,
                    base_image,
                    base_label,
                    include_mixup=True,
                )
        return (
            self._to_display_array(original_image),
            self._to_display_array(original_label),
            self._to_display_array(augmented_image),
            self._to_display_array(augmented_label),
        )

    def _build_synthetic_base_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        patch_height, patch_width = tuple(getattr(self._training_parameters.generation, 'segment_size', (256, 256)))
        size_hw = (
            max(int(patch_height), int(self.synthetic_image_height_spinbox.value())),
            max(int(patch_width), int(self.synthetic_image_width_spinbox.value())),
        )
        trace_count = self._sample_preview_int_range(
            self.synthetic_trace_count_min_spinbox,
            self.synthetic_trace_count_max_spinbox,
            salt='synthetic_trace_count',
        )
        background_noise_sigma = self._sample_preview_float_range(
            self.synthetic_background_noise_sigma_min_spinbox,
            self.synthetic_background_noise_sigma_max_spinbox,
            salt='synthetic_background_noise_sigma',
        )
        trace_noise_sigma = self._sample_preview_float_range(
            self.synthetic_trace_noise_sigma_min_spinbox,
            self.synthetic_trace_noise_sigma_max_spinbox,
            salt='synthetic_trace_noise_sigma',
        )
        params = SyntheticTopologyParameters(
            trace_count=trace_count,
            segment_count_range=tuple(
                sorted(
                    (
                        int(self.synthetic_segment_count_min_spinbox.value()),
                        int(self.synthetic_segment_count_max_spinbox.value()),
                    )
                )
            ),
            trace_half_width_range=tuple(
                sorted(
                    (
                        int(self.synthetic_trace_half_width_min_spinbox.value()),
                        int(self.synthetic_trace_half_width_max_spinbox.value()),
                    )
                )
            ),
            topology_domain=self._get_synthetic_topology_domain(),
            topology_family=self._get_synthetic_topology_family(),
            via_count_range=(1, max(1, min(6, int(round(trace_count / 3.0))))),
            background_noise_sigma=background_noise_sigma,
            trace_noise_sigma=trace_noise_sigma,
        )
        generator = SyntheticTopologyGenerator(params)
        synthetic_channels = 3 if self._get_synthetic_topology_domain() == 'pcb' else int(self._training_parameters.colors)
        image_array, label_array = generator.generate(
            size_hw=size_hw,
            channels=synthetic_channels,
            seed=self._seed_for(self._current_sample_index, f'synthetic:{self._variant_serial}'),
        )
        return image_array.astype(np.float32, copy=False), label_array.astype(np.float32, copy=False)

    def _build_augmented_synthetic_pair(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
        *,
        full_image: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if full_image:
            return self._build_augmented_full_image(
                sample_index,
                image_matrix,
                label_matrix,
                include_mixup=False,
            )
        return self._build_augmented_patch(
            sample_index,
            image_matrix,
            label_matrix,
            include_mixup=False,
        )

    def _load_prepared_arrays(self, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
        sample_path, label_path = self._sample_pairs[sample_index]
        prepared_image = ImagePreparator(sample_path, self._training_parameters.prepare).image
        prepared_label = ImagePreparator(label_path, self._training_parameters.prepare).image
        if self._training_parameters.colors == 1:
            prepared_image = prepared_image.convert('L')
        else:
            prepared_image = prepared_image.convert('RGB')
        prepared_label = prepared_label.convert('L')
        if prepared_label.size != prepared_image.size:
            prepared_label = prepared_label.resize(prepared_image.size, resample=Image.Resampling.NEAREST)
        return (
            SampleFastCutter.get_matrix_from_image(prepared_image, self._training_parameters.colors),
            SampleFastCutter.get_matrix_from_image(prepared_label, 1),
        )

    def _build_original_patch(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._extract_patch(
            sample_index,
            image_matrix,
            label_matrix,
            random_crop=False,
            scale=False,
        )

    def _build_original_full_image(
        self,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            image_matrix.astype(np.float32, copy=True),
            label_matrix.astype(np.float32, copy=True),
        )

    def _build_augmented_patch(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
        *,
        include_mixup: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_patch, label_patch = self._build_pre_batch_patch(
            sample_index,
            image_matrix,
            label_matrix,
        )
        if include_mixup and self._toggle_boxes['mixup'].isChecked():
            image_patch, label_patch = self._apply_mixup(sample_index, image_patch, label_patch)
        image_patch = self._apply_cutout(sample_index, image_patch)
        image_patch = self._apply_random_artifacts(sample_index, image_patch)
        return image_patch, label_patch

    def _build_augmented_full_image(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
        *,
        include_mixup: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_full, label_full = self._build_pre_batch_full_image(
            sample_index,
            image_matrix,
            label_matrix,
        )
        if include_mixup and self._toggle_boxes['mixup'].isChecked():
            image_full, label_full = self._apply_mixup(
                sample_index,
                image_full,
                label_full,
                full_image=True,
            )
        image_full = self._apply_cutout(sample_index, image_full)
        image_full = self._apply_random_artifacts(sample_index, image_full)
        return image_full, label_full

    def _build_pre_batch_patch(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        augmented_image = image_matrix.astype(np.float32, copy=True)
        augmented_label = label_matrix.astype(np.float32, copy=True)
        image_patch, label_patch = self._extract_patch(
            sample_index,
            augmented_image,
            augmented_label,
            random_crop=self._toggle_boxes['random_crop'].isChecked(),
            scale=self._toggle_boxes['scale'].isChecked(),
        )
        image_patch, label_patch = self._apply_rotations(image_patch, label_patch)
        image_patch = self._apply_photometric_augmentations(sample_index, image_patch)
        image_patch, label_patch = self._apply_pcb_defects(sample_index, image_patch, label_patch)
        return image_patch, label_patch

    def _build_pre_batch_full_image(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_full = image_matrix.astype(np.float32, copy=True)
        label_full = label_matrix.astype(np.float32, copy=True)
        image_full, label_full = self._apply_rotations(image_full, label_full)
        image_full = self._apply_photometric_augmentations(sample_index, image_full)
        image_full, label_full = self._apply_pcb_defects(sample_index, image_full, label_full)
        return image_full, label_full

    def _extract_patch(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
        *,
        random_crop: bool,
        scale: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        generation = replace(
            self._training_parameters.generation,
            horizontal_rotation=False,
            vertical_rotation=False,
            flip_x=False,
            flip_y=False,
            additional_augmentation=False,
            random_crop=bool(random_crop),
            crops_per_image=int(self.crops_per_image_spinbox.value()),
            scale_augmentation=bool(scale),
            scale_augmentation_strength=float(self.scale_augmentation_strength_spinbox.value()),
        )
        with _seeded_random(self._seed_for(sample_index, f'extract:{int(random_crop)}:{int(scale)}')):
            cutter = SampleFastCutter((image_matrix, label_matrix), generation, shuffle=False)
        if len(cutter) <= 0:
            return image_matrix.copy(), label_matrix.copy()
        base_locations = max(1, int(getattr(cutter, '_base_locations', len(cutter))))
        scale_variants = max(1, int(getattr(cutter, '_scale_variants', 1)))
        location_index = min(base_locations - 1, max(0, base_locations // 2))
        item_index = location_index * scale_variants
        if scale and scale_variants > 1:
            item_index += 1
        item_index = min(len(cutter) - 1, item_index)
        image_patch, label_patch = cutter[item_index]
        return (
            np.asarray(image_patch, dtype=np.float32).copy(),
            np.asarray(label_patch, dtype=np.float32).copy(),
        )

    def _apply_rotations(
        self,
        image_patch: np.ndarray,
        label_patch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = image_patch.copy()
        label = label_patch.copy()
        if self._toggle_boxes['rotate_90'].isChecked() and image.shape[1] == image.shape[2]:
            image = np.rot90(image, k=-1, axes=(1, 2)).copy()
            label = np.rot90(label, k=-1, axes=(1, 2)).copy()
        if self._toggle_boxes['rotate_180'].isChecked():
            image = image[:, ::-1, ::-1].copy()
            label = label[:, ::-1, ::-1].copy()
        if self._toggle_boxes['flip_x'].isChecked():
            image = image[:, ::-1, :].copy()
            label = label[:, ::-1, :].copy()
        if self._toggle_boxes['flip_y'].isChecked():
            image = image[:, :, ::-1].copy()
            label = label[:, :, ::-1].copy()
        return image, label

    def _apply_photometric_augmentations(self, sample_index: int, image_patch: np.ndarray) -> np.ndarray:
        image = image_patch.astype(np.float32, copy=True)
        with _seeded_random(self._seed_for(sample_index, 'photometric')):
            if self._toggle_boxes['blur'].isChecked() and self._passes_probability(
                sample_index,
                'blur_probability',
                float(self.augmentation_blur_probability_spinbox.value()),
            ):
                blur_radius = max(0.0, float(self.augmentation_blur_radius_spinbox.value()))
                if blur_radius > 0.0:
                    image = SampleFastCutter._apply_gaussian_blur(image, blur_radius)
            if self._toggle_boxes['brightness'].isChecked():
                strength = max(0.0, float(self.augmentation_brightness_spinbox.value()))
                brightness = 1.0 + strength
                image *= float(brightness)
            if self._toggle_boxes['contrast'].isChecked():
                strength = max(0.0, float(self.augmentation_contrast_spinbox.value()))
                contrast = 1.0 + strength
                mean = image.mean(axis=(1, 2), keepdims=True)
                image = (image - mean) * float(contrast) + mean
            if self._toggle_boxes['gamma'].isChecked():
                gamma_strength = max(0.0, float(self.augmentation_gamma_spinbox.value()))
                gamma = max(0.1, 1.0 - min(gamma_strength, 0.9))
                image = np.power(np.clip(image, 0.0, 1.0), float(gamma)).astype(np.float32, copy=False)
            if self._toggle_boxes['noise'].isChecked() and self._passes_probability(
                sample_index,
                'noise_probability',
                float(self.augmentation_noise_probability_spinbox.value()),
            ):
                sigma = max(0.0, float(self.augmentation_noise_sigma_spinbox.value()))
                if sigma > 0.0:
                    image += np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
        np.clip(image, 0.0, 1.0, out=image)
        return image.astype(np.float32, copy=False)

    def _has_selected_tech_variations(self) -> bool:
        return any(
            self._toggle_boxes[key].isChecked()
            for key in (
                'tech_global_width',
                'tech_scale_rethreshold',
                'tech_blur_threshold',
                'tech_boundary_aware',
                'tech_local_morphology',
                'tech_gap_variation',
            )
        )

    def _apply_tech_variations(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        config = copy.deepcopy(build_tech_augmentation_config(getattr(self._training_parameters.generation, 'tech_aug', None)))
        selected_count = sum(
            1
            for key in (
                'tech_global_width',
                'tech_scale_rethreshold',
                'tech_blur_threshold',
                'tech_boundary_aware',
                'tech_local_morphology',
                'tech_gap_variation',
            )
            if self._toggle_boxes[key].isChecked()
        )
        if selected_count <= 0:
            return image_matrix, label_matrix
        config.enabled = True
        config.min_operations = min(selected_count, max(1, int(self.tech_aug_min_operations_spinbox.value())))
        config.max_operations = min(selected_count, max(config.min_operations, int(self.tech_aug_max_operations_spinbox.value())))
        config.global_width.probability = (
            float(self.tech_aug_global_width_probability_spinbox.value())
            if self._toggle_boxes['tech_global_width'].isChecked()
            else 0.0
        )
        config.scale_rethreshold.probability = (
            float(self.tech_aug_scale_rethreshold_probability_spinbox.value())
            if self._toggle_boxes['tech_scale_rethreshold'].isChecked()
            else 0.0
        )
        config.blur_threshold.probability = (
            float(self.tech_aug_blur_threshold_probability_spinbox.value())
            if self._toggle_boxes['tech_blur_threshold'].isChecked()
            else 0.0
        )
        config.boundary_aware.probability = (
            float(self.tech_aug_boundary_aware_probability_spinbox.value())
            if self._toggle_boxes['tech_boundary_aware'].isChecked()
            else 0.0
        )
        config.local_morphology.probability = (
            float(self.tech_aug_local_morphology_probability_spinbox.value())
            if self._toggle_boxes['tech_local_morphology'].isChecked()
            else 0.0
        )
        config.gap_variation.probability = (
            float(self.tech_aug_gap_variation_probability_spinbox.value())
            if self._toggle_boxes['tech_gap_variation'].isChecked()
            else 0.0
        )
        augmentor = TechVariationAugmentor(config)
        source_image = image_matrix.astype(np.float32, copy=False)
        source_label = label_matrix.astype(np.float32, copy=False)
        preview_attempts = max(1, min(12, max(selected_count * 2, int(getattr(config, 'max_operations', 1)))))
        for attempt_index in range(preview_attempts):
            with _seeded_random(self._seed_for(sample_index, f'tech:{attempt_index}')):
                augmented_image, augmented_label = _apply_binary_tech_augmentation_to_pair(
                    source_image,
                    source_label,
                    augmentor,
                    binary_tolerance=float(getattr(config, 'binary_tolerance', 0.15)),
                )
            if np.array_equal(augmented_image, source_image) and np.array_equal(augmented_label, source_label):
                continue
            return augmented_image, augmented_label
        return source_image, source_label

    def _apply_pcb_defects(
        self,
        sample_index: int,
        image_patch: np.ndarray,
        label_patch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._toggle_boxes['synthetic_topology'].isChecked():
            return image_patch, label_patch
        if not self._toggle_boxes['pcb_defects'].isChecked():
            return image_patch, label_patch
        current_domain = self._get_synthetic_topology_domain()
        if current_domain == 'ic':
            selected_probabilities = {
                'line_break': 1.0 if self._toggle_boxes['ic_line_break'].isChecked() else 0.0,
                'bridge': 1.0 if self._toggle_boxes['ic_bridge'].isChecked() else 0.0,
                'necking': 1.0 if self._toggle_boxes['ic_necking'].isChecked() else 0.0,
                'missing_metal': 1.0 if self._toggle_boxes['ic_missing_metal'].isChecked() else 0.0,
                'spur': 1.0 if self._toggle_boxes['ic_spur'].isChecked() else 0.0,
                'pinhole': 1.0 if self._toggle_boxes['ic_pinhole'].isChecked() else 0.0,
                'via_open': 1.0 if self._toggle_boxes['ic_via_open'].isChecked() else 0.0,
                'line_shift': 1.0 if self._toggle_boxes['ic_line_shift'].isChecked() else 0.0,
            }
            selected_severities = {
                defect_name: float(self.ic_defect_type_spinboxes[defect_name].value()) / 100.0
                for defect_name in selected_probabilities
            }
            config = self._build_apply_ic_defects_config()
            augmentor_cls = ICDefectAugmentor
        else:
            selected_probabilities = {
                'break': 1.0 if self._toggle_boxes['pcb_break'].isChecked() else 0.0,
                'short': 1.0 if self._toggle_boxes['pcb_short'].isChecked() else 0.0,
                'missing_copper': 1.0 if self._toggle_boxes['pcb_missing_copper'].isChecked() else 0.0,
                'excess_copper': 1.0 if self._toggle_boxes['pcb_excess_copper'].isChecked() else 0.0,
                'pinhole': 1.0 if self._toggle_boxes['pcb_pinhole'].isChecked() else 0.0,
                'spurious_copper': 1.0 if self._toggle_boxes['pcb_spurious_copper'].isChecked() else 0.0,
                'via': 1.0 if self._toggle_boxes['pcb_via'].isChecked() else 0.0,
                'misalignment': 1.0 if self._toggle_boxes['pcb_misalignment'].isChecked() else 0.0,
            }
            selected_severities = {
                defect_name: float(self.pcb_defect_type_spinboxes[defect_name].value()) / 100.0
                for defect_name in selected_probabilities
            }
            config = self._build_apply_pcb_defects_config()
            augmentor_cls = PCBDefectAugmentor
        active_count = sum(1 for probability in selected_probabilities.values() if probability > 0.0)
        if active_count <= 0:
            return image_patch, label_patch
        if not self._passes_probability(
            sample_index,
            'pcb_defects_probability',
            float(self.pcb_defects_probability_spinbox.value()),
        ):
            return image_patch, label_patch
        config.defect_probability = 1.0
        config.min_defects = min(active_count, max(1, int(self.pcb_defects_min_count_spinbox.value())))
        config.max_defects = min(active_count, max(config.min_defects, int(self.pcb_defects_max_count_spinbox.value())))
        for defect_name in tuple(config.defect_probabilities.keys()):
            config.defect_probabilities[defect_name] = float(selected_probabilities.get(defect_name, 0.0))
            config.defect_severities[defect_name] = float(selected_severities.get(defect_name, 0.5))
        augmentor = augmentor_cls(config)
        source_image = image_patch.astype(np.float32, copy=False)
        source_label = label_patch.astype(np.float32, copy=False)
        preview_attempts = max(1, min(16, int(getattr(config, 'max_attempts_per_defect', 8))))
        for attempt_index in range(preview_attempts):
            augmented_image, defect_mask, _augmented_mask = augmentor(
                source_image,
                source_label,
                seed=self._seed_for(sample_index, f'pcb_defects_{attempt_index}'),
                return_augmented_mask=True,
            )
            defect_mask_array = np.asarray(defect_mask)
            if np.count_nonzero(defect_mask_array) <= 0 and np.array_equal(augmented_image, source_image):
                continue
            return (
                augmented_image,
                source_label,
            )
        return source_image, source_label

    def _apply_mixup(
        self,
        sample_index: int,
        image_patch: np.ndarray,
        label_patch: np.ndarray,
        *,
        full_image: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(self._sample_pairs) <= 1:
            return image_patch, label_patch
        if not self._passes_probability(
            sample_index,
            'mixup_probability',
            float(self.mixup_probability_spinbox.value()),
        ):
            return image_patch, label_patch
        alpha = max(0.0, float(self.mixup_alpha_spinbox.value()))
        if alpha <= 0.0:
            return image_patch, label_patch
        partner_index = (sample_index + 1 + self._variant_serial) % len(self._sample_pairs)
        if partner_index == sample_index:
            partner_index = (partner_index + 1) % len(self._sample_pairs)
        partner_base_image, partner_base_label = self._load_prepared_arrays(partner_index)
        if full_image:
            partner_image, partner_label = self._build_pre_batch_full_image(
                partner_index,
                partner_base_image,
                partner_base_label,
            )
        else:
            partner_image, partner_label = self._build_pre_batch_patch(
                partner_index,
                partner_base_image,
                partner_base_label,
            )
        if partner_image.shape != image_patch.shape or partner_label.shape != label_patch.shape:
            return image_patch, label_patch
        with _seeded_random(self._seed_for(sample_index, 'mixup')):
            lambda_value = float(np.random.beta(alpha, alpha))
        lambda_value = float(min(max(lambda_value, 0.0), 1.0))
        mixed_image = (lambda_value * image_patch) + ((1.0 - lambda_value) * partner_image)
        mixed_label = (lambda_value * label_patch) + ((1.0 - lambda_value) * partner_label)
        return (
            mixed_image.astype(np.float32, copy=False),
            mixed_label.astype(np.float32, copy=False),
        )

    def _apply_cutout(self, sample_index: int, image_patch: np.ndarray) -> np.ndarray:
        if not self._toggle_boxes['cutout'].isChecked():
            return image_patch
        if not self._passes_probability(
            sample_index,
            'cutout_probability',
            float(self.cutout_probability_spinbox.value()),
        ):
            return image_patch
        holes = max(1, int(self.cutout_holes_spinbox.value()))
        size_ratio = float(self.cutout_size_ratio_spinbox.value())
        if size_ratio <= 0.0:
            return image_patch
        image = torch.from_numpy(np.ascontiguousarray(image_patch[None, ...])).float()
        with _seeded_random(self._seed_for(sample_index, 'cutout')):
            _batch, channels, height, width = image.shape
            max_cutout_height = max(1, min(int(height), int(round(int(height) * size_ratio))))
            max_cutout_width = max(1, min(int(width), int(round(int(width) * size_ratio))))
            if max_cutout_height <= 0 or max_cutout_width <= 0:
                return image_patch
            for _ in range(holes):
                cutout_height = (
                    1 if max_cutout_height == 1 else int(torch.randint(1, max_cutout_height + 1, (1,)).item())
                )
                cutout_width = (
                    1 if max_cutout_width == 1 else int(torch.randint(1, max_cutout_width + 1, (1,)).item())
                )
                max_top = max(0, height - cutout_height)
                max_left = max(0, width - cutout_width)
                top = 0 if max_top == 0 else int(torch.randint(0, max_top + 1, (1,)).item())
                left = 0 if max_left == 0 else int(torch.randint(0, max_left + 1, (1,)).item())
                fill_color = torch.rand((channels, 1, 1), dtype=image.dtype)
                image[0, :, top:top + cutout_height, left:left + cutout_width] = fill_color
        return image[0].numpy().astype(np.float32, copy=False)

    def _apply_random_artifacts(self, sample_index: int, image_patch: np.ndarray) -> np.ndarray:
        if not self._toggle_boxes['random_artifacts'].isChecked():
            return image_patch
        artifact_types = self._selected_artifact_types()
        if not artifact_types:
            return image_patch
        if not self._passes_probability(
            sample_index,
            'random_artifacts_probability',
            float(self.random_artifacts_probability_spinbox.value()),
        ):
            return image_patch
        count = max(1, int(self.random_artifacts_count_spinbox.value()))
        size_ratio = float(self.random_artifacts_size_ratio_spinbox.value())
        if size_ratio <= 0.0:
            return image_patch
        image = torch.from_numpy(np.ascontiguousarray(image_patch[None, ...])).float()
        _, channels, height, width = image.shape
        min_h, max_h, min_w, max_w = self._artifact_size_bounds(height, width, size_ratio)
        with _seeded_random(self._seed_for(sample_index, 'random_artifacts')):
            for _ in range(count):
                artifact_height = int(min_h if max_h == min_h else np.random.randint(min_h, max_h + 1))
                artifact_width = int(min_w if max_w == min_w else np.random.randint(min_w, max_w + 1))
                max_top = max(0, height - artifact_height)
                max_left = max(0, width - artifact_width)
                top = 0 if max_top == 0 else int(np.random.randint(0, max_top + 1))
                left = 0 if max_left == 0 else int(np.random.randint(0, max_left + 1))
                overlay, alpha = generate_random_artifact_patch(
                    int(channels),
                    int(artifact_height),
                    int(artifact_width),
                    device=torch.device('cpu'),
                    dtype=torch.float32,
                    artifact_types=artifact_types,
                )
                patch = image[0, :, top:top + artifact_height, left:left + artifact_width]
                image[0, :, top:top + artifact_height, left:left + artifact_width] = torch.clamp(
                    (patch * (1.0 - alpha)) + (overlay * alpha),
                    min=0.0,
                    max=1.0,
                )
        return image[0].numpy().astype(np.float32, copy=False)

    def _selected_artifact_types(self) -> tuple[str, ...]:
        mapping = {
            'artifact_dust': 'dust',
            'artifact_resist_residue': 'resist_residue',
            'artifact_etch_residue': 'etch_residue',
            'artifact_particle_cluster': 'particle_cluster',
            'artifact_flake': 'flake',
        }
        selected: list[str] = []
        for key, artifact_name in mapping.items():
            if self._toggle_boxes[key].isChecked():
                selected.append(artifact_name)
        return tuple(selected)

    def _set_value_widgets_enabled(self, key: str, enabled: bool) -> None:
        for row in self._value_rows.get(key, ()):
            row.setEnabled(bool(enabled))
        for widget in self._value_widgets.get(key, ()):
            widget.setEnabled(bool(enabled))

    def _set_value_widgets_visible(self, key: str, visible: bool) -> None:
        for row in self._value_rows.get(key, ()):
            row.setVisible(bool(visible))

    def _preview_mode_text(self) -> str:
        if self.full_image_check_box.isChecked():
            return str(
                self._texts.get(
                    'preview_mode_full',
                    'Полный кадр' if self._is_russian_ui else 'Full image',
                )
            )
        return str(
            self._texts.get(
                'preview_mode_patch',
                'Патч' if self._is_russian_ui else 'Patch',
            )
        )

    def _sample_preview_int_range(self, min_widget: QWidget, max_widget: QWidget, *, salt: str) -> int:
        lower = int(min(getattr(min_widget, 'value')(), getattr(max_widget, 'value')()))
        upper = int(max(getattr(min_widget, 'value')(), getattr(max_widget, 'value')()))
        with _seeded_random(self._seed_for(self._current_sample_index, salt)):
            return int(np.random.randint(lower, upper + 1))

    def _sample_preview_float_range(self, min_widget: QWidget, max_widget: QWidget, *, salt: str) -> float:
        lower = float(min(getattr(min_widget, 'value')(), getattr(max_widget, 'value')()))
        upper = float(max(getattr(min_widget, 'value')(), getattr(max_widget, 'value')()))
        if upper <= lower:
            return lower
        with _seeded_random(self._seed_for(self._current_sample_index, salt)):
            return float(np.random.uniform(lower, upper))

    def _passes_probability(self, sample_index: int, salt: str, probability: float) -> bool:
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        with _seeded_random(self._seed_for(sample_index, salt)):
            return bool(random.random() <= float(probability))

    @staticmethod
    def _set_combo_value(combo: NoWheelComboBox, value: str) -> None:
        normalized = str(value or '').strip().lower()
        index = combo.findData(normalized)
        if index < 0:
            index = combo.findText(normalized)
        if index < 0:
            index = 0
        combo.setCurrentIndex(index)

    def _get_synthetic_topology_domain(self) -> str:
        return str(
            self.synthetic_topology_domain_combo.currentData()
            or self.synthetic_topology_domain_combo.currentText()
            or 'pcb'
        ).strip().lower()

    def _get_synthetic_topology_family(self) -> str:
        combo = self.ic_topology_family_combo if self._get_synthetic_topology_domain() == 'ic' else self.pcb_topology_family_combo
        return str(combo.currentData() or combo.currentText() or '').strip().lower()

    def _resolve_text(self, key: str) -> str:
        if key == 'synthetic_topology':
            return str(
                self._texts.get(
                    key,
                    'Генерировать синтетическую топологию' if self._is_russian_ui else 'Generate synthetic topology',
                )
            )
        ic_fallbacks = {
            'ic_line_break': 'Обрыв линии' if self._is_russian_ui else 'Line break',
            'ic_bridge': 'Мостик' if self._is_russian_ui else 'Bridge',
            'ic_necking': 'Пережатие' if self._is_russian_ui else 'Necking',
            'ic_missing_metal': 'Потеря металла' if self._is_russian_ui else 'Missing metal',
            'ic_spur': 'Шпора' if self._is_russian_ui else 'Spur',
            'ic_pinhole': 'Pinhole' if self._is_russian_ui else 'Pinhole',
            'ic_via_open': 'Via open' if self._is_russian_ui else 'Via open',
            'ic_line_shift': 'Сдвиг линии' if self._is_russian_ui else 'Line shift',
        }
        if key in ic_fallbacks:
            return ic_fallbacks[key]
        return str(self._texts.get(key, self._settings_texts.get(key, key)))

    def _resolve_setting_text(self, key: str) -> str:
        fallback_labels = (
            PREVIEW_VALUE_LABELS_EN | PREVIEW_VALUE_LABELS_RU
            if self._is_russian_ui
            else PREVIEW_VALUE_LABELS_EN
        )
        if key == 'synthetic_background_noise_sigma':
            return str(fallback_labels.get(key, 'Background noise sigma'))
        if key == 'synthetic_topology_domain':
            return 'Домен синтетики' if self._is_russian_ui else 'Synthetic domain'
        if key == 'pcb_topology_family':
            return 'Семейство PCB-топологии' if self._is_russian_ui else 'PCB topology family'
        if key == 'ic_topology_family':
            return 'Семейство IC-топологии' if self._is_russian_ui else 'IC topology family'
        if key == 'synthetic_trace_count':
            return 'Количество трасс' if self._is_russian_ui else 'Trace count'
        if key == 'synthetic_segment_count':
            return 'Сегментов на трассу' if self._is_russian_ui else 'Segments per trace'
        if key == 'synthetic_trace_half_width':
            return 'Полуширина трассы' if self._is_russian_ui else 'Trace half-width'
        if key == 'ic_line_break_severity':
            return 'Сила обрыва линии' if self._is_russian_ui else 'Line break severity'
        if key == 'ic_bridge_severity':
            return 'Сила мостика' if self._is_russian_ui else 'Bridge severity'
        if key == 'ic_necking_severity':
            return 'Сила пережатия' if self._is_russian_ui else 'Necking severity'
        if key == 'ic_missing_metal_severity':
            return 'Сила потери металла' if self._is_russian_ui else 'Missing metal severity'
        if key == 'ic_spur_severity':
            return 'Сила шпоры' if self._is_russian_ui else 'Spur severity'
        if key == 'ic_pinhole_severity':
            return 'Сила pinhole' if self._is_russian_ui else 'Pinhole severity'
        if key == 'ic_via_open_severity':
            return 'Сила via open' if self._is_russian_ui else 'Via open severity'
        if key == 'ic_line_shift_severity':
            return 'Сила сдвига линии' if self._is_russian_ui else 'Line shift severity'
        return str(
            self._settings_texts.get(
                key,
                self._settings_form_labels.get(
                    key,
                    fallback_labels.get(
                        key,
                        self._texts.get(key, key),
                    ),
                ),
            )
        )

    def _resolve_tip(self, key: str) -> str:
        return str(
            self._texts.get(
                f'{key}_tip',
                self._settings_texts.get(
                    f'{key}_tip',
                    self._settings_form_tooltips.get(key, ''),
                ),
            )
        )

    @staticmethod
    def _bounded_int(value: object, lower: int, upper: int) -> int:
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            resolved = int(lower)
        return max(int(lower), min(int(upper), resolved))

    @staticmethod
    def _bounded_float(value: object, lower: float, upper: float) -> float:
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            resolved = float(lower)
        return max(float(lower), min(float(upper), resolved))

    @staticmethod
    def _artifact_size_bounds(
        image_height: int,
        image_width: int,
        size_ratio: float,
    ) -> tuple[int, int, int, int]:
        max_artifact_height = max(1, min(int(image_height), int(round(int(image_height) * float(size_ratio)))))
        max_artifact_width = max(1, min(int(image_width), int(round(int(image_width) * float(size_ratio)))))
        min_artifact_height = 1 if max_artifact_height <= 2 else max(2, int(round(max_artifact_height * 0.35)))
        min_artifact_width = 1 if max_artifact_width <= 2 else max(2, int(round(max_artifact_width * 0.35)))
        min_artifact_height = min(max_artifact_height, min_artifact_height)
        min_artifact_width = min(max_artifact_width, min_artifact_width)
        return (
            int(min_artifact_height),
            int(max_artifact_height),
            int(min_artifact_width),
            int(max_artifact_width),
        )

    def _seed_for(self, sample_index: int, salt: str) -> int:
        if 0 <= int(sample_index) < len(self._sample_pairs):
            sample_key = self._sample_pairs[sample_index][0].as_posix()
        else:
            sample_key = 'synthetic'
        payload = f'{sample_key}|{sample_index}|{self._variant_serial}|{salt}'
        return int(zlib.crc32(payload.encode('utf-8')) & 0xFFFFFFFF)

    @staticmethod
    def _to_display_array(image_array: np.ndarray) -> np.ndarray:
        array = np.asarray(image_array, dtype=np.float32)
        if array.ndim == 2:
            return np.clip(np.round(array * 255.0), 0.0, 255.0).astype(np.uint8)
        if array.ndim == 3 and array.shape[0] == 1:
            return np.clip(np.round(array[0] * 255.0), 0.0, 255.0).astype(np.uint8)
        if array.ndim == 3 and array.shape[0] >= 3:
            rgb = np.transpose(array[:3], (1, 2, 0))
            return np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)
        return np.zeros((16, 16), dtype=np.uint8)

    @staticmethod
    def _set_preview_image(
        widget: QLabel,
        image_data: np.ndarray | None,
        *,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> None:
        if not isinstance(image_data, np.ndarray) or image_data.size == 0:
            widget.clear()
            return
        if image_data.ndim == 2:
            contiguous = np.ascontiguousarray(image_data)
            qimg = QImage(
                contiguous.tobytes(),
                contiguous.shape[1],
                contiguous.shape[0],
                contiguous.strides[0],
                QImage.Format.Format_Grayscale8,
            ).copy()
        elif image_data.ndim == 3 and image_data.shape[2] == 3:
            contiguous = np.ascontiguousarray(image_data)
            qimg = QImage(
                contiguous.tobytes(),
                contiguous.shape[1],
                contiguous.shape[0],
                contiguous.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
        else:
            widget.clear()
            return
        scaled_width = max(1, int(target_width or widget.width()))
        scaled_height = max(1, int(target_height or widget.height()))
        pixmap = QPixmap.fromImage(qimg).scaled(
            scaled_width,
            scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        widget.setPixmap(pixmap)
