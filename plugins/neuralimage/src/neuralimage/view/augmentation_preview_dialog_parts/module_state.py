"""augmentation_preview_dialog: module state."""
from __future__ import annotations
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)

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

SHIFT_RANGE_MIN = 4

SHIFT_RANGE_MAX = 2000

MIN_SYNTHETIC_DATASET_FACTOR = 0.0

MAX_SYNTHETIC_DATASET_FACTOR = 10.0

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

MIN_AUGMENTATION_MULTIPLIER = 0.0

MAX_AUGMENTATION_MULTIPLIER = 20.0

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
    'shift': 'Patch shift',
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
    'tech_aug_max_changed_pixels_ratio': 'Changed pixels limit',
    'tech_aug_max_foreground_ratio_delta': 'Foreground ratio limit',
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
    'synthetic_dataset_factor': 'Synthetic epoch factor',
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
    'shift': 'Шаг нарезки',
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
    'tech_aug_max_changed_pixels_ratio': 'Лимит изменённых пикселей',
    'tech_aug_max_foreground_ratio_delta': 'Лимит изменения foreground',
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
    'synthetic_dataset_factor': 'Коэффициент synthetic-эпохи',
}
