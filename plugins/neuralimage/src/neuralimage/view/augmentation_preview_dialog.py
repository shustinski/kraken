from __future__ import annotations
import copy
from collections import OrderedDict
from dataclasses import asdict
from dataclasses import replace
import numpy as np
from PIL import Image
from PyQt6.QtCore import QEvent, QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)
from neuralimage.lib.data_interfaces import (
    TrainingParameters,
    build_ic_defect_parameters,
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
    build_tech_augmentation_config,
)
from neuralimage.lib.images import ImagePreparator, SampleFastCutter
from neuralimage.lib.rare_patch_masks import (
    collect_matching_sample_label_pairs,
    prepare_cif_label_raster,
)
from neuralimage.lib.ui_texts import get_ui_section
from neuralimage.preprocessing.pipeline import image_to_channel_first_float01
from neuralimage.configuration import (
    build_sem_segmentation_config,
)
from neuralimage.view.settings_panel import SettingsPanel
from neuralimage.view.settings_panel_widgets import NoWheelComboBox

from .augmentation_preview_dialog_parts.module_state import (
    MIN_AUG_STRENGTH,
    MAX_AUG_STRENGTH,
    MIN_AUGMENTATION_PROBABILITY,
    MAX_AUGMENTATION_PROBABILITY,
    MIN_AUG_NOISE_SIGMA,
    MAX_AUG_NOISE_SIGMA,
    MIN_AUG_BLUR_RADIUS,
    MAX_AUG_BLUR_RADIUS,
    MIN_CROPS_PER_IMAGE,
    MAX_CROPS_PER_IMAGE,
    MIN_TECH_AUG_OPERATIONS,
    MAX_TECH_AUG_OPERATIONS,
    SHIFT_RANGE_MIN,
    SHIFT_RANGE_MAX,
    MIN_SYNTHETIC_DATASET_FACTOR,
    MAX_SYNTHETIC_DATASET_FACTOR,
    MIN_CUTOUT_HOLES,
    MAX_CUTOUT_HOLES,
    MIN_RANDOM_ARTIFACTS_COUNT,
    MAX_RANDOM_ARTIFACTS_COUNT,
    MIN_MIXUP_ALPHA,
    MAX_MIXUP_ALPHA,
    MIN_PCB_DEFECT_COUNT,
    MAX_PCB_DEFECT_COUNT,
    MIN_PCB_DEFECT_WEIGHT,
    MAX_PCB_DEFECT_WEIGHT,
    MIN_SYNTHETIC_TRACE_COUNT,
    MAX_SYNTHETIC_TRACE_COUNT,
    MIN_SYNTHETIC_IMAGE_SIZE,
    MAX_SYNTHETIC_IMAGE_SIZE,
    MIN_SYNTHETIC_SEGMENT_COUNT,
    MAX_SYNTHETIC_SEGMENT_COUNT,
    MIN_SYNTHETIC_TRACE_HALF_WIDTH,
    MAX_SYNTHETIC_TRACE_HALF_WIDTH,
    MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
    MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
    MIN_SYNTHETIC_TRACE_NOISE_SIGMA,
    MAX_SYNTHETIC_TRACE_NOISE_SIGMA,
    MIN_AUGMENTATION_MULTIPLIER,
    MAX_AUGMENTATION_MULTIPLIER,
    IC_DEFECT_WEIGHT_FIELDS,
    PREVIEW_VALUE_LABELS_EN,
    PREVIEW_VALUE_LABELS_RU,
)
from .augmentation_preview_dialog_parts.construction import ConstructionMixin
from .augmentation_preview_dialog_parts.key_press_event import KeyPressEventMixin
from .augmentation_preview_dialog_parts.build_ui import BuildUiMixin
from .augmentation_preview_dialog_parts.close_event import CloseEventMixin
from .augmentation_preview_dialog_parts.connect_signals import ConnectSignalsMixin
from .augmentation_preview_dialog_parts.populate_sample_list import PopulateSampleListMixin
from .augmentation_preview_dialog_parts.build_apply_payload import BuildApplyPayloadMixin
from .augmentation_preview_dialog_parts.generation_settings_cutter import GenerationSettingsCutterMixin
from .augmentation_preview_dialog_parts.refresh_preview import RefreshPreviewMixin
from .augmentation_preview_dialog_parts.resolve_setting_text import ResolveSettingTextMixin
from .augmentation_preview_dialog_parts.preview_image import PreviewImageMixin
from .augmentation_preview_dialog_parts.preview_label import _PreviewLabel

class AugmentationPreviewDialog(
    ConstructionMixin,
    KeyPressEventMixin,
    BuildUiMixin,
    CloseEventMixin,
    ConnectSignalsMixin,
    PopulateSampleListMixin,
    BuildApplyPayloadMixin,
    GenerationSettingsCutterMixin,
    RefreshPreviewMixin,
    ResolveSettingTextMixin,
    PreviewImageMixin,
    QDialog,
):
    apply_to_main_requested = pyqtSignal(object)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    MIN_AUG_STRENGTH,
    MAX_AUG_STRENGTH,
    MIN_AUGMENTATION_PROBABILITY,
    MAX_AUGMENTATION_PROBABILITY,
    MIN_AUG_NOISE_SIGMA,
    MAX_AUG_NOISE_SIGMA,
    MIN_AUG_BLUR_RADIUS,
    MAX_AUG_BLUR_RADIUS,
    MIN_CROPS_PER_IMAGE,
    MAX_CROPS_PER_IMAGE,
    MIN_TECH_AUG_OPERATIONS,
    MAX_TECH_AUG_OPERATIONS,
    SHIFT_RANGE_MIN,
    SHIFT_RANGE_MAX,
    MIN_SYNTHETIC_DATASET_FACTOR,
    MAX_SYNTHETIC_DATASET_FACTOR,
    MIN_CUTOUT_HOLES,
    MAX_CUTOUT_HOLES,
    MIN_RANDOM_ARTIFACTS_COUNT,
    MAX_RANDOM_ARTIFACTS_COUNT,
    MIN_MIXUP_ALPHA,
    MAX_MIXUP_ALPHA,
    MIN_PCB_DEFECT_COUNT,
    MAX_PCB_DEFECT_COUNT,
    MIN_PCB_DEFECT_WEIGHT,
    MAX_PCB_DEFECT_WEIGHT,
    MIN_SYNTHETIC_TRACE_COUNT,
    MAX_SYNTHETIC_TRACE_COUNT,
    MIN_SYNTHETIC_IMAGE_SIZE,
    MAX_SYNTHETIC_IMAGE_SIZE,
    MIN_SYNTHETIC_SEGMENT_COUNT,
    MAX_SYNTHETIC_SEGMENT_COUNT,
    MIN_SYNTHETIC_TRACE_HALF_WIDTH,
    MAX_SYNTHETIC_TRACE_HALF_WIDTH,
    MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
    MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA,
    MIN_SYNTHETIC_TRACE_NOISE_SIGMA,
    MAX_SYNTHETIC_TRACE_NOISE_SIGMA,
    MIN_AUGMENTATION_MULTIPLIER,
    MAX_AUGMENTATION_MULTIPLIER,
    IC_DEFECT_WEIGHT_FIELDS,
    PREVIEW_VALUE_LABELS_EN,
    PREVIEW_VALUE_LABELS_RU,
    _PreviewLabel,
)
