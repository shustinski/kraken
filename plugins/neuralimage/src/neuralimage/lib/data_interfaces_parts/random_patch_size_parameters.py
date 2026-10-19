"""data_interfaces: random patch size parameters."""
from __future__ import annotations
import enum
import math
from dataclasses import dataclass
from neuralimage.active_learning.config import ActiveLearningConfig
from neuralimage.augmentations.sem_config import SemAugmentationConfig
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.targets.config import SupervisionTargetsParameters
from neuralimage.uncertainty.config import UncertaintyConfig

def _default_supervision_targets() -> SupervisionTargetsParameters:
    return SupervisionTargetsParameters()

def _default_preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig()

def _default_sem_augmentation_config() -> SemAugmentationConfig:
    return SemAugmentationConfig()

def _default_uncertainty_config() -> UncertaintyConfig:
    return UncertaintyConfig()

def _default_active_learning_config() -> ActiveLearningConfig:
    return ActiveLearningConfig()

class WorkMode(enum.Enum):
    train_only = 'train_only'
    continue_training = 'continue_training'
    train_and_recognition = 'train_and_recognition'
    recognition_only = 'recognition_only'
    further_training = 'further_training'
    # Backward-compatible enum aliases used in old configs/tests.
    recognintion_only = 'recognition_only'
    futher_training = 'further_training'

def normalize_work_mode(value: str | WorkMode | None) -> str:
    from neuralimage.lib.data_interfaces import _WORK_MODE_ALIASES

    if isinstance(value, WorkMode):
        return value.value
    normalized = str(value or '').strip()
    return _WORK_MODE_ALIASES.get(normalized, normalized)

def parse_work_mode(value: str | WorkMode | None) -> WorkMode | None:
    normalized = normalize_work_mode(value)
    if not normalized:
        return None
    try:
        return WorkMode(normalized)
    except ValueError:
        return None

class ValidationSource(enum.Enum):
    split = 'split'
    external = 'external'

def normalize_validation_source(value: str | ValidationSource | None) -> str:
    if isinstance(value, ValidationSource):
        return value.value
    raw = str(value or '').strip().lower()
    if raw in {mode.value for mode in ValidationSource}:
        return raw
    return ValidationSource.split.value

class SampleCutMode(enum.Enum):
    disk = 'disk'
    online = 'online'

class ConfidenceSaveMode(enum.Enum):
    off = 'off'
    separate_grayscale = 'separate_grayscale'

def normalize_confidence_save_mode(value: str | ConfidenceSaveMode | None) -> str:
    from neuralimage.lib.data_interfaces import _CONFIDENCE_SAVE_MODE_ALIASES

    if isinstance(value, ConfidenceSaveMode):
        return value.value
    raw = str(value or '').strip().lower()
    if raw in _CONFIDENCE_SAVE_MODE_ALIASES:
        raw = _CONFIDENCE_SAVE_MODE_ALIASES[raw]
    if raw in {mode.value for mode in ConfidenceSaveMode}:
        return raw
    return ConfidenceSaveMode.off.value

class OptimizerName(enum.Enum):
    adam = 'adam'
    adamw = 'adamw'
    adamw_muon = 'adamw_muon'

class MixedPrecisionMode(enum.Enum):
    off = 'off'
    fp16 = 'fp16'
    bf16 = 'bf16'

class SchedulerName(enum.Enum):
    off = 'off'
    reduce_on_plateau = 'reduce_on_plateau'
    cosine_annealing = 'cosine_annealing'
    one_cycle = 'one_cycle'
    step_lr = 'step_lr'

def normalize_scheduler_name(value: str | SchedulerName | None) -> str:
    from neuralimage.lib.data_interfaces import _SCHEDULER_NAME_ALIASES

    if isinstance(value, SchedulerName):
        return value.value
    raw = str(value or '').strip().lower()
    if raw in _SCHEDULER_NAME_ALIASES:
        raw = _SCHEDULER_NAME_ALIASES[raw]
    if raw in {mode.value for mode in SchedulerName}:
        return raw
    return SchedulerName.off.value

class MultiGpuMode(enum.Enum):
    off = 'off'
    dataparallel = 'dataparallel'
    distributeddataparallel = 'distributeddataparallel'

def normalize_multi_gpu_mode(
    value: str | MultiGpuMode | None,
    *,
    use_multi_gpu_fallback: bool | None = None,
) -> str:
    from neuralimage.lib.data_interfaces import _MULTI_GPU_MODE_ALIASES

    if isinstance(value, MultiGpuMode):
        return value.value

    raw = str(value or '').strip().lower()
    if raw in _MULTI_GPU_MODE_ALIASES:
        raw = _MULTI_GPU_MODE_ALIASES[raw]
    if raw in {mode.value for mode in MultiGpuMode}:
        return raw

    if use_multi_gpu_fallback is None:
        return MultiGpuMode.off.value
    return MultiGpuMode.distributeddataparallel.value if bool(use_multi_gpu_fallback) else MultiGpuMode.off.value

class PatchBatchSyncMode(enum.Enum):
    off = 'off'
    patch = 'patch'
    batch = 'batch'
    patch_and_batch = 'patch_and_batch'

def normalize_patch_batch_sync_mode(value: str | PatchBatchSyncMode | None) -> str:
    from neuralimage.lib.data_interfaces import _PATCH_BATCH_SYNC_MODE_ALIASES

    if isinstance(value, PatchBatchSyncMode):
        return value.value
    raw = str(value or '').strip().lower()
    if raw in _PATCH_BATCH_SYNC_MODE_ALIASES:
        raw = _PATCH_BATCH_SYNC_MODE_ALIASES[raw]
    if raw in {mode.value for mode in PatchBatchSyncMode}:
        return raw
    return PatchBatchSyncMode.patch_and_batch.value

@dataclass
class OptimizerParameters:
    name: OptimizerName = OptimizerName.adam
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class EarlyStoppingParameters:
    enabled: bool = False
    patience: int = 10
    min_delta: float = 0.0
    restore_best_weights: bool = True

@dataclass(frozen=True)
class RandomPatchSizeParameters:
    enabled: bool = False
    min_size: tuple[int, int] = (128, 128)
    max_size: tuple[int, int] = (512, 512)

    def __post_init__(self) -> None:
        min_width, min_height = (int(value) for value in self.min_size)
        max_width, max_height = (int(value) for value in self.max_size)
        if min_width <= 0 or min_height <= 0:
            raise ValueError('random patch minimum size must be positive')
        if max_width < min_width or max_height < min_height:
            raise ValueError('random patch maximum size must be greater than or equal to minimum size')
        if self.enabled:
            for lower, upper, axis in (
                (min_width, max_width, 'width'),
                (min_height, max_height, 'height'),
            ):
                first_aligned = int(math.ceil(lower / 32.0) * 32)
                if first_aligned > upper:
                    raise ValueError(f'random patch {axis} range must contain a multiple of 32')

@dataclass
class WarmupParameters:
    enabled: bool = False
    epochs: int = 3
    start_factor: float = 0.1

@dataclass
class SchedulerParameters:
    name: SchedulerName = SchedulerName.off
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    plateau_threshold: float = 1e-4
    plateau_min_lr: float = 1e-6
    plateau_cooldown: int = 0
    cosine_t_max: int = 10
    cosine_eta_min: float = 1e-6
    one_cycle_max_lr: float = 1e-3
    one_cycle_pct_start: float = 0.3
    one_cycle_anneal_strategy: str = 'cos'
    one_cycle_div_factor: float = 25.0
    one_cycle_final_div_factor: float = 10000.0
    one_cycle_three_phase: bool = False
    step_lr_step_size: int = 10
    step_lr_gamma: float = 0.1
