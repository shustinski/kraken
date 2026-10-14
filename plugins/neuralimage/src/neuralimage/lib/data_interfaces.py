import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from neuralimage.active_learning.config import ActiveLearningConfig, build_active_learning_config
from neuralimage.augmentations.sem_config import SemAugmentationConfig, build_sem_augmentation_config
from neuralimage.preprocessing.config import PreprocessingConfig, build_preprocessing_config
from neuralimage.targets.config import SupervisionTargetsParameters, build_supervision_targets_parameters
from neuralimage.uncertainty.config import UncertaintyConfig, build_uncertainty_config

from .data_interfaces_parts.module_state import (
    RANDOM_ARTIFACT_TYPES,
    _PCB_DEFECT_TYPE_ALIASES,
    SYNTHETIC_TOPOLOGY_DOMAINS,
    PCB_TOPOLOGY_FAMILIES,
    IC_TOPOLOGY_FAMILIES,
    _IC_DEFECT_TYPE_ALIASES,
)
from .data_interfaces_parts.random_patch_size_parameters import (
    _default_supervision_targets,
    _default_preprocessing_config,
    _default_sem_augmentation_config,
    _default_uncertainty_config,
    _default_active_learning_config,
    WorkMode,
    normalize_work_mode,
    parse_work_mode,
    ValidationSource,
    normalize_validation_source,
    SampleCutMode,
    ConfidenceSaveMode,
    normalize_confidence_save_mode,
    OptimizerName,
    MixedPrecisionMode,
    SchedulerName,
    normalize_scheduler_name,
    MultiGpuMode,
    normalize_multi_gpu_mode,
    PatchBatchSyncMode,
    normalize_patch_batch_sync_mode,
    OptimizerParameters,
    EarlyStoppingParameters,
    RandomPatchSizeParameters,
    WarmupParameters,
    SchedulerParameters,
)
from .data_interfaces_parts.pcb_defect_parameters import (
    HardMiningParameters,
    CutoutParameters,
    RandomArtifactsParameters,
    MixupParameters,
    TechGlobalWidthVariationParameters,
    TechScaleRethresholdParameters,
    TechBlurThresholdParameters,
    TechBoundaryAwareVariationParameters,
    TechLocalMorphologyParameters,
    TechGapVariationParameters,
    TechAugmentationParameters,
    PCBDefectParameters,
)
from .data_interfaces_parts.ic_defect_parameters import (
    ICDefectParameters,
    SyntheticDefectGeneratorParameters,
    _coerce_probability,
    _coerce_positive_int,
    _coerce_positive_float,
    _coerce_size_pair,
    _coerce_range,
)
from .data_interfaces_parts.build_pcb_defect_parameters import build_pcb_defect_parameters
from .data_interfaces_parts.build_ic_defect_parameters import build_ic_defect_parameters
from .data_interfaces_parts.build_synthetic_defect_generator import build_synthetic_defect_generator_parameters
from .data_interfaces_parts.coerce_float_range import (
    _coerce_float_range,
    _coerce_int_range,
)
from .data_interfaces_parts.build_tech_augmentation_config import build_tech_augmentation_config
from .data_interfaces_parts.sample_generation_settings import (
    SampleGenerationSettings,
    SamplePrepareSettings,
)
from .data_interfaces_parts.training_parameters import TrainingParameters
from .data_interfaces_parts.recognition_parameters import (
    RecognitionParameters,
    CutSettings,
    NeuralThreadConfig,
)

_WORK_MODE_ALIASES: dict[str, str] = {
    'recognintion_only': WorkMode.recognition_only.value,
    'futher_training': WorkMode.further_training.value,
}
_CONFIDENCE_SAVE_MODE_ALIASES: dict[str, str] = {
    'none': ConfidenceSaveMode.off.value,
    'disabled': ConfidenceSaveMode.off.value,
    'false': ConfidenceSaveMode.off.value,
    '0': ConfidenceSaveMode.off.value,
    'grayscale': ConfidenceSaveMode.separate_grayscale.value,
    'separate': ConfidenceSaveMode.separate_grayscale.value,
    'separate_gray': ConfidenceSaveMode.separate_grayscale.value,
}
_SCHEDULER_NAME_ALIASES: dict[str, str] = {
    'none': SchedulerName.off.value,
    'reducelronplateau': SchedulerName.reduce_on_plateau.value,
    'reduce_lr_on_plateau': SchedulerName.reduce_on_plateau.value,
    'cosine': SchedulerName.cosine_annealing.value,
    'cosineannealing': SchedulerName.cosine_annealing.value,
    'cosineannealinglr': SchedulerName.cosine_annealing.value,
    'onecycle': SchedulerName.one_cycle.value,
    'onecyclelr': SchedulerName.one_cycle.value,
    'steplr': SchedulerName.step_lr.value,
    'step': SchedulerName.step_lr.value,
}
_MULTI_GPU_MODE_ALIASES: dict[str, str] = {
    'dp': MultiGpuMode.dataparallel.value,
    'ddp': MultiGpuMode.distributeddataparallel.value,
    'none': MultiGpuMode.off.value,
    'false': MultiGpuMode.off.value,
    '0': MultiGpuMode.off.value,
    'true': MultiGpuMode.distributeddataparallel.value,
    '1': MultiGpuMode.distributeddataparallel.value,
    # Common misspelling.
    'distibuteddataparallel': MultiGpuMode.distributeddataparallel.value,
}
_PATCH_BATCH_SYNC_MODE_ALIASES: dict[str, str] = {
    'none': PatchBatchSyncMode.off.value,
    'all': PatchBatchSyncMode.patch_and_batch.value,
    'both': PatchBatchSyncMode.patch_and_batch.value,
    'full': PatchBatchSyncMode.patch_and_batch.value,
}

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    RANDOM_ARTIFACT_TYPES,
    _PCB_DEFECT_TYPE_ALIASES,
    SYNTHETIC_TOPOLOGY_DOMAINS,
    PCB_TOPOLOGY_FAMILIES,
    IC_TOPOLOGY_FAMILIES,
    _IC_DEFECT_TYPE_ALIASES,
    _default_supervision_targets,
    _default_preprocessing_config,
    _default_sem_augmentation_config,
    _default_uncertainty_config,
    _default_active_learning_config,
    WorkMode,
    normalize_work_mode,
    parse_work_mode,
    ValidationSource,
    normalize_validation_source,
    SampleCutMode,
    ConfidenceSaveMode,
    normalize_confidence_save_mode,
    OptimizerName,
    MixedPrecisionMode,
    SchedulerName,
    normalize_scheduler_name,
    MultiGpuMode,
    normalize_multi_gpu_mode,
    PatchBatchSyncMode,
    normalize_patch_batch_sync_mode,
    OptimizerParameters,
    EarlyStoppingParameters,
    RandomPatchSizeParameters,
    WarmupParameters,
    SchedulerParameters,
    HardMiningParameters,
    CutoutParameters,
    RandomArtifactsParameters,
    MixupParameters,
    TechGlobalWidthVariationParameters,
    TechScaleRethresholdParameters,
    TechBlurThresholdParameters,
    TechBoundaryAwareVariationParameters,
    TechLocalMorphologyParameters,
    TechGapVariationParameters,
    TechAugmentationParameters,
    PCBDefectParameters,
    ICDefectParameters,
    SyntheticDefectGeneratorParameters,
    _coerce_probability,
    _coerce_positive_int,
    _coerce_positive_float,
    _coerce_size_pair,
    _coerce_range,
    build_pcb_defect_parameters,
    build_ic_defect_parameters,
    build_synthetic_defect_generator_parameters,
    _coerce_float_range,
    _coerce_int_range,
    build_tech_augmentation_config,
    SampleGenerationSettings,
    SamplePrepareSettings,
    TrainingParameters,
    RecognitionParameters,
    CutSettings,
    NeuralThreadConfig,
)
