import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class WorkMode(enum.Enum):
    train_only = 'train_only'
    train_and_recognition = 'train_and_recognition'
    recognition_only = 'recognition_only'
    further_training = 'further_training'
    # Backward-compatible enum aliases used in old configs/tests.
    recognintion_only = 'recognition_only'
    futher_training = 'further_training'


_WORK_MODE_ALIASES: dict[str, str] = {
    'recognintion_only': WorkMode.recognition_only.value,
    'futher_training': WorkMode.further_training.value,
}


def normalize_work_mode(value: str | WorkMode | None) -> str:
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


_CONFIDENCE_SAVE_MODE_ALIASES: dict[str, str] = {
    'none': ConfidenceSaveMode.off.value,
    'disabled': ConfidenceSaveMode.off.value,
    'false': ConfidenceSaveMode.off.value,
    '0': ConfidenceSaveMode.off.value,
    'grayscale': ConfidenceSaveMode.separate_grayscale.value,
    'separate': ConfidenceSaveMode.separate_grayscale.value,
    'separate_gray': ConfidenceSaveMode.separate_grayscale.value,
}


def normalize_confidence_save_mode(value: str | ConfidenceSaveMode | None) -> str:
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


def normalize_scheduler_name(value: str | SchedulerName | None) -> str:
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


def normalize_multi_gpu_mode(
    value: str | MultiGpuMode | None,
    *,
    use_multi_gpu_fallback: bool | None = None,
) -> str:
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


_PATCH_BATCH_SYNC_MODE_ALIASES: dict[str, str] = {
    'none': PatchBatchSyncMode.off.value,
    'all': PatchBatchSyncMode.patch_and_batch.value,
    'both': PatchBatchSyncMode.patch_and_batch.value,
    'full': PatchBatchSyncMode.patch_and_batch.value,
}


def normalize_patch_batch_sync_mode(value: str | PatchBatchSyncMode | None) -> str:
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


@dataclass
class HardMiningParameters:
    enabled: bool = False
    strength: float = 2.0
    ema_alpha: float = 0.2
    pixel_enabled: bool = False
    pixel_keep_ratio: float = 0.25


@dataclass
class CutoutParameters:
    enabled: bool = False
    probability: float = 1.0
    holes: int = 1
    size_ratio: float = 0.25


RANDOM_ARTIFACT_TYPES: tuple[str, ...] = (
    'dust',
    'resist_residue',
    'etch_residue',
    'particle_cluster',
    'flake',
)


@dataclass
class RandomArtifactsParameters:
    enabled: bool = False
    probability: float = 1.0
    count: int = 1
    size_ratio: float = 0.25
    dust_enabled: bool = True
    resist_residue_enabled: bool = True
    etch_residue_enabled: bool = True
    particle_cluster_enabled: bool = True
    flake_enabled: bool = True

    def enabled_types(self) -> tuple[str, ...]:
        enabled_names: list[str] = []
        for artifact_name in RANDOM_ARTIFACT_TYPES:
            if bool(getattr(self, f'{artifact_name}_enabled', True)):
                enabled_names.append(artifact_name)
        return tuple(enabled_names)


@dataclass
class MixupParameters:
    enabled: bool = False
    probability: float = 1.0
    alpha: float = 0.2


@dataclass
class TechGlobalWidthVariationParameters:
    probability: float = 0.45
    kernel_size_range: tuple[int, int] = (2, 3)
    erosion_probability: float = 0.5


@dataclass
class TechScaleRethresholdParameters:
    probability: float = 0.35
    scale_range: tuple[float, float] = (0.82, 1.18)
    threshold: float = 0.5


@dataclass
class TechBlurThresholdParameters:
    probability: float = 0.3
    blur_radius_range: tuple[float, float] = (0.6, 1.8)
    threshold: float = 0.5


@dataclass
class TechBoundaryAwareVariationParameters:
    probability: float = 0.7
    band_width_range: tuple[int, int] = (2, 4)
    noise_cell_size_range: tuple[int, int] = (3, 10)
    add_probability: float = 0.34
    remove_probability: float = 0.2
    min_addition_support: int = 1
    min_removal_support: int = 4
    smoothing_kernel_size: int = 2


@dataclass
class TechLocalMorphologyParameters:
    probability: float = 0.35
    roi_count_range: tuple[int, int] = (2, 4)
    roi_size_ratio_range: tuple[float, float] = (0.18, 0.4)
    kernel_size_range: tuple[int, int] = (2, 3)
    erosion_probability: float = 0.5


@dataclass
class TechGapVariationParameters:
    probability: float = 0.3
    kernel_size_range: tuple[int, int] = (2, 3)
    opening_probability: float = 0.4
    max_bridge_neighbor_count: int = 5
    min_gap_neighbor_count: int = 1


@dataclass
class TechAugmentationParameters:
    enabled: bool = False
    min_operations: int = 1
    max_operations: int = 3
    debug_return_pair: bool = False
    binarization_threshold: float = 0.5
    binary_tolerance: float = 0.15
    max_changed_pixels_ratio: float = 0.32
    max_foreground_ratio_delta: float = 0.2
    global_width: TechGlobalWidthVariationParameters = field(
        default_factory=TechGlobalWidthVariationParameters
    )
    scale_rethreshold: TechScaleRethresholdParameters = field(
        default_factory=TechScaleRethresholdParameters
    )
    blur_threshold: TechBlurThresholdParameters = field(
        default_factory=TechBlurThresholdParameters
    )
    boundary_aware: TechBoundaryAwareVariationParameters = field(
        default_factory=TechBoundaryAwareVariationParameters
    )
    local_morphology: TechLocalMorphologyParameters = field(
        default_factory=TechLocalMorphologyParameters
    )
    gap_variation: TechGapVariationParameters = field(
        default_factory=TechGapVariationParameters
    )


_PCB_DEFECT_TYPE_ALIASES: dict[str, str] = {
    'break': 'break',
    'break_defect': 'break',
    'short': 'short',
    'short_circuit': 'short',
    'missing': 'missing_copper',
    'missing_copper': 'missing_copper',
    'excess': 'excess_copper',
    'excess_copper': 'excess_copper',
    'pinhole': 'pinhole',
    'small_hole': 'pinhole',
    'spurious': 'spurious_copper',
    'spurious_copper': 'spurious_copper',
    'via': 'via',
    'via_defect': 'via',
    'via_defects': 'via',
    'misalignment': 'misalignment',
}

SYNTHETIC_TOPOLOGY_DOMAINS: tuple[str, str] = ('pcb', 'ic')
PCB_TOPOLOGY_FAMILIES: tuple[str, ...] = (
    'pcb_mixed',
    'pcb_parallel',
    'pcb_independent',
    'pcb_fanout',
)
IC_TOPOLOGY_FAMILIES: tuple[str, ...] = (
    'ic_mixed',
    'ic_cell_array',
    'ic_channels',
    'ic_tree',
)

_IC_DEFECT_TYPE_ALIASES: dict[str, str] = {
    'line_break': 'line_break',
    'bridge': 'bridge',
    'necking': 'necking',
    'missing_metal': 'missing_metal',
    'spur': 'spur',
    'pinhole': 'pinhole',
    'via_open': 'via_open',
    'line_shift': 'line_shift',
}


@dataclass
class PCBDefectParameters:
    enabled: bool = False
    defect_probability: float = 0.5
    min_defects: int = 1
    max_defects: int = 3
    max_attempts_per_defect: int = 8
    min_component_area: int = 12
    break_width_range: tuple[int, int] = (1, 5)
    short_bridge_distance_range: tuple[int, int] = (2, 24)
    missing_copper_radius_range: tuple[int, int] = (2, 8)
    excess_copper_radius_range: tuple[int, int] = (2, 7)
    pinhole_radius_range: tuple[int, int] = (1, 3)
    spurious_copper_radius_range: tuple[int, int] = (1, 4)
    via_hole_area_range: tuple[int, int] = (12, 400)
    via_shift_range: tuple[int, int] = (1, 4)
    via_size_delta_range: tuple[int, int] = (1, 3)
    misalignment_shift_range: tuple[int, int] = (1, 5)
    misalignment_roi_scale_range: tuple[float, float] = (0.2, 0.45)
    use_input_mask: bool = True
    use_defect_mask_as_label: bool = False
    defect_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            'break': 1.0,
            'short': 1.0,
            'missing_copper': 1.0,
            'excess_copper': 1.0,
            'pinhole': 1.0,
            'spurious_copper': 1.0,
            'via': 1.0,
            'misalignment': 1.0,
        }
    )
    defect_severities: dict[str, float] = field(
        default_factory=lambda: {
            'break': 0.5,
            'short': 0.5,
            'missing_copper': 0.5,
            'excess_copper': 0.5,
            'pinhole': 0.5,
            'spurious_copper': 0.5,
            'via': 0.5,
            'misalignment': 0.5,
        }
    )


@dataclass
class ICDefectParameters:
    enabled: bool = False
    defect_probability: float = 0.5
    min_defects: int = 1
    max_defects: int = 3
    max_attempts_per_defect: int = 8
    min_component_area: int = 12
    line_break_width_range: tuple[int, int] = (1, 4)
    bridge_distance_range: tuple[int, int] = (1, 12)
    necking_radius_range: tuple[int, int] = (1, 3)
    missing_metal_radius_range: tuple[int, int] = (2, 5)
    spur_radius_range: tuple[int, int] = (1, 4)
    pinhole_radius_range: tuple[int, int] = (1, 2)
    via_hole_area_range: tuple[int, int] = (8, 220)
    via_shift_range: tuple[int, int] = (1, 3)
    via_size_delta_range: tuple[int, int] = (1, 2)
    line_shift_range: tuple[int, int] = (1, 3)
    line_shift_roi_scale_range: tuple[float, float] = (0.12, 0.3)
    use_input_mask: bool = True
    use_defect_mask_as_label: bool = False
    defect_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            'line_break': 1.0,
            'bridge': 1.0,
            'necking': 1.0,
            'missing_metal': 1.0,
            'spur': 1.0,
            'pinhole': 1.0,
            'via_open': 1.0,
            'line_shift': 1.0,
        }
    )
    defect_severities: dict[str, float] = field(
        default_factory=lambda: {
            'line_break': 0.5,
            'bridge': 0.5,
            'necking': 0.5,
            'missing_metal': 0.5,
            'spur': 0.5,
            'pinhole': 0.5,
            'via_open': 0.5,
            'line_shift': 0.5,
        }
    )


@dataclass
class SyntheticDefectGeneratorParameters:
    enabled: bool = False
    epoch_size_factor: float = 1.0
    topology_domain: str = 'pcb'
    topology_family: str = 'pcb_mixed'
    image_size_xy: tuple[int, int] = (1024, 1024)
    trace_count_range: tuple[int, int] = (5, 5)
    segment_count_range: tuple[int, int] = (4, 4)
    trace_half_width_range: tuple[int, int] = (2, 2)
    margin: int = 12
    background_noise_sigma_range: tuple[float, float] = (0.02, 0.02)
    trace_noise_sigma_range: tuple[float, float] = (0.01, 0.01)
    pcb_defects: PCBDefectParameters = field(
        default_factory=lambda: PCBDefectParameters(enabled=True)
    )
    ic_defects: ICDefectParameters = field(
        default_factory=lambda: ICDefectParameters(enabled=True)
    )
    defects: Any = field(default_factory=lambda: PCBDefectParameters(enabled=True))

    @property
    def trace_count(self) -> int:
        return int(self.trace_count_range[1])

    @property
    def segment_count(self) -> int:
        return int(self.segment_count_range[1])

    @property
    def trace_half_width(self) -> int:
        return int(self.trace_half_width_range[1])

    @property
    def background_noise_sigma(self) -> float:
        return float(self.background_noise_sigma_range[1])

    @property
    def trace_noise_sigma(self) -> float:
        return float(self.trace_noise_sigma_range[1])


def _coerce_probability(value: Any, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = float(default)
    return float(min(max(resolved, 0.0), 1.0))


def _coerce_positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    return max(int(minimum), resolved)


def _coerce_positive_float(value: Any, default: float, *, minimum: float = 0.0) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = float(default)
    return max(float(minimum), resolved)


def _coerce_size_pair(
    value: Any,
    default: tuple[int, int],
    *,
    minimum: int = 1,
) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    try:
        width = int(value[0])
        height = int(value[1])
    except (TypeError, ValueError):
        return default
    return max(int(minimum), width), max(int(minimum), height)


def _coerce_range(
    value: Any,
    default: tuple[int, int] | tuple[float, float],
    *,
    cast_type,
) -> tuple[int, int] | tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    try:
        low = cast_type(value[0])
        high = cast_type(value[1])
    except (TypeError, ValueError):
        return default
    if low > high:
        low, high = high, low
    return low, high


def build_pcb_defect_parameters(raw: Any | None) -> PCBDefectParameters:
    defaults = PCBDefectParameters()
    if raw is None:
        return defaults
    if isinstance(raw, PCBDefectParameters):
        return raw
    if isinstance(raw, Mapping):
        payload = dict(raw)
    elif hasattr(raw, '__dict__'):
        payload = dict(vars(raw))
    else:
        return defaults

    probability = _coerce_probability(
        payload.get('defect_probability', payload.get('probability', defaults.defect_probability)),
        defaults.defect_probability,
    )
    min_defects = _coerce_positive_int(payload.get('min_defects', defaults.min_defects), defaults.min_defects)
    max_defects = _coerce_positive_int(payload.get('max_defects', defaults.max_defects), defaults.max_defects)
    if min_defects > max_defects:
        min_defects, max_defects = max_defects, min_defects

    raw_probabilities = payload.get('defect_probabilities', {})
    normalized_probabilities = dict(defaults.defect_probabilities)
    if isinstance(raw_probabilities, Mapping):
        for raw_key, raw_value in raw_probabilities.items():
            canonical_key = _PCB_DEFECT_TYPE_ALIASES.get(str(raw_key).strip().lower())
            if canonical_key is None:
                continue
            try:
                normalized_probabilities[canonical_key] = max(0.0, float(raw_value))
            except (TypeError, ValueError):
                continue

    raw_severities = payload.get('defect_severities', {})
    normalized_severities = dict(defaults.defect_severities)
    if isinstance(raw_severities, Mapping):
        for raw_key, raw_value in raw_severities.items():
            canonical_key = _PCB_DEFECT_TYPE_ALIASES.get(str(raw_key).strip().lower())
            if canonical_key is None:
                continue
            normalized_severities[canonical_key] = _coerce_probability(
                raw_value,
                defaults.defect_severities.get(canonical_key, 0.5),
            )
    else:
        for canonical_key in tuple(normalized_severities.keys()):
            if float(normalized_probabilities.get(canonical_key, 0.0)) > 0.0:
                normalized_severities[canonical_key] = 0.5

    return PCBDefectParameters(
        enabled=bool(payload.get('enabled', defaults.enabled)),
        defect_probability=probability,
        min_defects=min_defects,
        max_defects=max_defects,
        max_attempts_per_defect=_coerce_positive_int(
            payload.get('max_attempts_per_defect', defaults.max_attempts_per_defect),
            defaults.max_attempts_per_defect,
        ),
        min_component_area=_coerce_positive_int(
            payload.get('min_component_area', defaults.min_component_area),
            defaults.min_component_area,
        ),
        break_width_range=_coerce_range(
            payload.get('break_width_range', defaults.break_width_range),
            defaults.break_width_range,
            cast_type=int,
        ),
        short_bridge_distance_range=_coerce_range(
            payload.get('short_bridge_distance_range', defaults.short_bridge_distance_range),
            defaults.short_bridge_distance_range,
            cast_type=int,
        ),
        missing_copper_radius_range=_coerce_range(
            payload.get('missing_copper_radius_range', defaults.missing_copper_radius_range),
            defaults.missing_copper_radius_range,
            cast_type=int,
        ),
        excess_copper_radius_range=_coerce_range(
            payload.get('excess_copper_radius_range', defaults.excess_copper_radius_range),
            defaults.excess_copper_radius_range,
            cast_type=int,
        ),
        pinhole_radius_range=_coerce_range(
            payload.get('pinhole_radius_range', defaults.pinhole_radius_range),
            defaults.pinhole_radius_range,
            cast_type=int,
        ),
        spurious_copper_radius_range=_coerce_range(
            payload.get('spurious_copper_radius_range', defaults.spurious_copper_radius_range),
            defaults.spurious_copper_radius_range,
            cast_type=int,
        ),
        via_hole_area_range=_coerce_range(
            payload.get('via_hole_area_range', defaults.via_hole_area_range),
            defaults.via_hole_area_range,
            cast_type=int,
        ),
        via_shift_range=_coerce_range(
            payload.get('via_shift_range', defaults.via_shift_range),
            defaults.via_shift_range,
            cast_type=int,
        ),
        via_size_delta_range=_coerce_range(
            payload.get('via_size_delta_range', defaults.via_size_delta_range),
            defaults.via_size_delta_range,
            cast_type=int,
        ),
        misalignment_shift_range=_coerce_range(
            payload.get('misalignment_shift_range', defaults.misalignment_shift_range),
            defaults.misalignment_shift_range,
            cast_type=int,
        ),
        misalignment_roi_scale_range=_coerce_range(
            payload.get('misalignment_roi_scale_range', defaults.misalignment_roi_scale_range),
            defaults.misalignment_roi_scale_range,
            cast_type=float,
        ),
        use_input_mask=bool(payload.get('use_input_mask', defaults.use_input_mask)),
        use_defect_mask_as_label=bool(
            payload.get('use_defect_mask_as_label', defaults.use_defect_mask_as_label)
        ),
        defect_probabilities=normalized_probabilities,
        defect_severities=normalized_severities,
    )


def build_ic_defect_parameters(raw: Any | None) -> ICDefectParameters:
    defaults = ICDefectParameters()
    if raw is None:
        return defaults
    if isinstance(raw, ICDefectParameters):
        return raw
    if isinstance(raw, Mapping):
        payload = dict(raw)
    elif hasattr(raw, '__dict__'):
        payload = dict(vars(raw))
    else:
        return defaults

    probability = _coerce_probability(
        payload.get('defect_probability', payload.get('probability', defaults.defect_probability)),
        defaults.defect_probability,
    )
    min_defects = _coerce_positive_int(payload.get('min_defects', defaults.min_defects), defaults.min_defects)
    max_defects = _coerce_positive_int(payload.get('max_defects', defaults.max_defects), defaults.max_defects)
    if min_defects > max_defects:
        min_defects, max_defects = max_defects, min_defects

    raw_probabilities = payload.get('defect_probabilities', {})
    normalized_probabilities = dict(defaults.defect_probabilities)
    if isinstance(raw_probabilities, Mapping):
        for raw_key, raw_value in raw_probabilities.items():
            canonical_key = _IC_DEFECT_TYPE_ALIASES.get(str(raw_key).strip().lower())
            if canonical_key is None:
                continue
            try:
                normalized_probabilities[canonical_key] = max(0.0, float(raw_value))
            except (TypeError, ValueError):
                continue

    raw_severities = payload.get('defect_severities', {})
    normalized_severities = dict(defaults.defect_severities)
    if isinstance(raw_severities, Mapping):
        for raw_key, raw_value in raw_severities.items():
            canonical_key = _IC_DEFECT_TYPE_ALIASES.get(str(raw_key).strip().lower())
            if canonical_key is None:
                continue
            normalized_severities[canonical_key] = _coerce_probability(
                raw_value,
                defaults.defect_severities.get(canonical_key, 0.5),
            )

    return ICDefectParameters(
        enabled=bool(payload.get('enabled', defaults.enabled)),
        defect_probability=probability,
        min_defects=min_defects,
        max_defects=max_defects,
        max_attempts_per_defect=_coerce_positive_int(
            payload.get('max_attempts_per_defect', defaults.max_attempts_per_defect),
            defaults.max_attempts_per_defect,
        ),
        min_component_area=_coerce_positive_int(
            payload.get('min_component_area', defaults.min_component_area),
            defaults.min_component_area,
        ),
        line_break_width_range=_coerce_range(
            payload.get('line_break_width_range', defaults.line_break_width_range),
            defaults.line_break_width_range,
            cast_type=int,
        ),
        bridge_distance_range=_coerce_range(
            payload.get('bridge_distance_range', defaults.bridge_distance_range),
            defaults.bridge_distance_range,
            cast_type=int,
        ),
        necking_radius_range=_coerce_range(
            payload.get('necking_radius_range', defaults.necking_radius_range),
            defaults.necking_radius_range,
            cast_type=int,
        ),
        missing_metal_radius_range=_coerce_range(
            payload.get('missing_metal_radius_range', defaults.missing_metal_radius_range),
            defaults.missing_metal_radius_range,
            cast_type=int,
        ),
        spur_radius_range=_coerce_range(
            payload.get('spur_radius_range', defaults.spur_radius_range),
            defaults.spur_radius_range,
            cast_type=int,
        ),
        pinhole_radius_range=_coerce_range(
            payload.get('pinhole_radius_range', defaults.pinhole_radius_range),
            defaults.pinhole_radius_range,
            cast_type=int,
        ),
        via_hole_area_range=_coerce_range(
            payload.get('via_hole_area_range', defaults.via_hole_area_range),
            defaults.via_hole_area_range,
            cast_type=int,
        ),
        via_shift_range=_coerce_range(
            payload.get('via_shift_range', defaults.via_shift_range),
            defaults.via_shift_range,
            cast_type=int,
        ),
        via_size_delta_range=_coerce_range(
            payload.get('via_size_delta_range', defaults.via_size_delta_range),
            defaults.via_size_delta_range,
            cast_type=int,
        ),
        line_shift_range=_coerce_range(
            payload.get('line_shift_range', defaults.line_shift_range),
            defaults.line_shift_range,
            cast_type=int,
        ),
        line_shift_roi_scale_range=_coerce_float_range(
            payload.get('line_shift_roi_scale_range', defaults.line_shift_roi_scale_range),
            defaults.line_shift_roi_scale_range,
            minimum=0.05,
            maximum=1.0,
        ),
        use_input_mask=True,
        use_defect_mask_as_label=False,
        defect_probabilities=normalized_probabilities,
        defect_severities=normalized_severities,
    )


def build_synthetic_defect_generator_parameters(raw: Any | None) -> SyntheticDefectGeneratorParameters:
    defaults = SyntheticDefectGeneratorParameters()
    if raw is None:
        defaults.defects = defaults.pcb_defects
        return defaults
    if isinstance(raw, SyntheticDefectGeneratorParameters):
        return raw
    if isinstance(raw, Mapping):
        payload = dict(raw)
    elif hasattr(raw, '__dict__'):
        payload = dict(vars(raw))
    else:
        return defaults

    topology_domain = str(payload.get('topology_domain', payload.get('domain', defaults.topology_domain)) or defaults.topology_domain).strip().lower()
    if topology_domain not in SYNTHETIC_TOPOLOGY_DOMAINS:
        topology_domain = defaults.topology_domain
    topology_family = str(payload.get('topology_family', '') or '').strip().lower()
    if not topology_family:
        family_key = f'{topology_domain}_topology_family'
        topology_family = str(
            payload.get(
                family_key,
                IC_TOPOLOGY_FAMILIES[0] if topology_domain == 'ic' else PCB_TOPOLOGY_FAMILIES[0],
            )
            or (IC_TOPOLOGY_FAMILIES[0] if topology_domain == 'ic' else PCB_TOPOLOGY_FAMILIES[0])
        ).strip().lower()

    defects_payload = payload.get('defects')
    if defects_payload is None:
        defects_payload = {
            'enabled': bool(payload.get('enabled', defaults.enabled)),
            'defect_probability': payload.get(
                'defect_probability',
                defaults.defects.defect_probability,
            ),
            'min_defects': payload.get('min_defects', defaults.defects.min_defects),
            'max_defects': payload.get('max_defects', defaults.defects.max_defects),
            'defect_probabilities': payload.get(
                'defect_probabilities',
                defaults.defects.defect_probabilities,
            ),
            'max_attempts_per_defect': payload.get(
                'max_attempts_per_defect',
                defaults.defects.max_attempts_per_defect,
            ),
            'min_component_area': payload.get(
                'min_component_area',
                defaults.defects.min_component_area,
            ),
        }
    pcb_payload = payload.get('pcb_defects', defects_payload if topology_domain == 'pcb' else {})
    ic_payload = payload.get('ic_defects', defects_payload if topology_domain == 'ic' else {})
    pcb_defects = build_pcb_defect_parameters(pcb_payload)
    pcb_enabled_flag = (
        bool(pcb_payload.get('enabled'))
        if isinstance(pcb_payload, Mapping) and 'enabled' in pcb_payload
        else bool(payload.get('enabled', defaults.enabled))
    )
    pcb_defects.enabled = bool(pcb_enabled_flag)
    pcb_defects.use_input_mask = True
    pcb_defects.use_defect_mask_as_label = False
    ic_defects = build_ic_defect_parameters(ic_payload)
    ic_enabled_flag = (
        bool(ic_payload.get('enabled'))
        if isinstance(ic_payload, Mapping) and 'enabled' in ic_payload
        else bool(payload.get('enabled', defaults.enabled))
    )
    ic_defects.enabled = bool(ic_enabled_flag)
    ic_defects.use_input_mask = True
    ic_defects.use_defect_mask_as_label = False
    active_defects: Any = ic_defects if topology_domain == 'ic' else pcb_defects

    trace_count_value = payload.get('trace_count')
    segment_count_value = payload.get('segment_count')
    trace_half_width_value = payload.get('trace_half_width')
    background_noise_sigma_value = payload.get('background_noise_sigma')
    trace_noise_sigma_value = payload.get('trace_noise_sigma')

    trace_count_range = _coerce_int_range(
        payload.get(
            'trace_count_range',
            (
                payload.get('trace_count_min', trace_count_value if trace_count_value is not None else defaults.trace_count_range[0]),
                payload.get('trace_count_max', trace_count_value if trace_count_value is not None else defaults.trace_count_range[1]),
            ),
        ),
        defaults.trace_count_range,
        minimum=1,
    )
    segment_count_range = _coerce_int_range(
        payload.get(
            'segment_count_range',
            (
                payload.get('segment_count_min', segment_count_value if segment_count_value is not None else defaults.segment_count_range[0]),
                payload.get('segment_count_max', segment_count_value if segment_count_value is not None else defaults.segment_count_range[1]),
            ),
        ),
        defaults.segment_count_range,
        minimum=1,
    )
    trace_half_width_range = _coerce_int_range(
        payload.get(
            'trace_half_width_range',
            (
                payload.get('trace_half_width_min', trace_half_width_value if trace_half_width_value is not None else defaults.trace_half_width_range[0]),
                payload.get('trace_half_width_max', trace_half_width_value if trace_half_width_value is not None else defaults.trace_half_width_range[1]),
            ),
        ),
        defaults.trace_half_width_range,
        minimum=1,
    )
    background_noise_sigma_range = _coerce_float_range(
        payload.get(
            'background_noise_sigma_range',
            (
                payload.get('background_noise_sigma_min', background_noise_sigma_value if background_noise_sigma_value is not None else defaults.background_noise_sigma_range[0]),
                payload.get('background_noise_sigma_max', background_noise_sigma_value if background_noise_sigma_value is not None else defaults.background_noise_sigma_range[1]),
            ),
        ),
        defaults.background_noise_sigma_range,
        minimum=0.0,
    )
    trace_noise_sigma_range = _coerce_float_range(
        payload.get(
            'trace_noise_sigma_range',
            (
                payload.get('trace_noise_sigma_min', trace_noise_sigma_value if trace_noise_sigma_value is not None else defaults.trace_noise_sigma_range[0]),
                payload.get('trace_noise_sigma_max', trace_noise_sigma_value if trace_noise_sigma_value is not None else defaults.trace_noise_sigma_range[1]),
            ),
        ),
        defaults.trace_noise_sigma_range,
        minimum=0.0,
    )
    image_size_xy = _coerce_size_pair(
        payload.get(
            'image_size_xy',
            payload.get(
                'frame_size_xy',
                (
                    payload.get(
                        'image_width',
                        payload.get(
                            'frame_width',
                            defaults.image_size_xy[0],
                        ),
                    ),
                    payload.get(
                        'image_height',
                        payload.get(
                            'frame_height',
                            defaults.image_size_xy[1],
                        ),
                    ),
                ),
            ),
        ),
        defaults.image_size_xy,
        minimum=1,
    )

    return SyntheticDefectGeneratorParameters(
        enabled=bool(payload.get('enabled', defaults.enabled)),
        epoch_size_factor=_coerce_positive_float(
            payload.get('epoch_size_factor', defaults.epoch_size_factor),
            defaults.epoch_size_factor,
            minimum=0.0,
        ),
        topology_domain=topology_domain,
        topology_family=topology_family,
        image_size_xy=image_size_xy,
        trace_count_range=trace_count_range,
        segment_count_range=segment_count_range,
        trace_half_width_range=trace_half_width_range,
        margin=_coerce_positive_int(
            payload.get('margin', defaults.margin),
            defaults.margin,
            minimum=1,
        ),
        background_noise_sigma_range=background_noise_sigma_range,
        trace_noise_sigma_range=trace_noise_sigma_range,
        pcb_defects=pcb_defects,
        ic_defects=ic_defects,
        defects=active_defects,
    )


def _coerce_float_range(
    value: Any,
    default: tuple[float, float],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float]:
    resolved = _coerce_range(value, default, cast_type=float)
    low = float(resolved[0])
    high = float(resolved[1])
    if minimum is not None:
        low = max(float(minimum), low)
        high = max(float(minimum), high)
    if maximum is not None:
        low = min(float(maximum), low)
        high = min(float(maximum), high)
    if low > high:
        low, high = high, low
    return low, high


def _coerce_int_range(
    value: Any,
    default: tuple[int, int],
    *,
    minimum: int = 0,
) -> tuple[int, int]:
    resolved = _coerce_range(value, default, cast_type=int)
    low = max(int(minimum), int(resolved[0]))
    high = max(int(minimum), int(resolved[1]))
    if low > high:
        low, high = high, low
    return low, high


def build_tech_augmentation_config(raw: Any | None) -> TechAugmentationParameters:
    defaults = TechAugmentationParameters()
    if raw is None:
        return defaults
    if isinstance(raw, TechAugmentationParameters):
        return raw
    if isinstance(raw, Mapping):
        payload = dict(raw)
    elif hasattr(raw, '__dict__'):
        payload = dict(vars(raw))
    else:
        return defaults

    def _nested_mapping(key: str) -> dict[str, Any]:
        nested = payload.get(key, {})
        if isinstance(nested, Mapping):
            return dict(nested)
        if hasattr(nested, '__dict__'):
            return dict(vars(nested))
        return {}

    global_width_payload = _nested_mapping('global_width')
    scale_payload = _nested_mapping('scale_rethreshold')
    blur_payload = _nested_mapping('blur_threshold')
    boundary_payload = _nested_mapping('boundary_aware')
    local_payload = _nested_mapping('local_morphology')
    gap_payload = _nested_mapping('gap_variation')

    min_operations = _coerce_positive_int(
        payload.get('min_operations', defaults.min_operations),
        defaults.min_operations,
    )
    max_operations = _coerce_positive_int(
        payload.get('max_operations', defaults.max_operations),
        defaults.max_operations,
    )
    if min_operations > max_operations:
        min_operations, max_operations = max_operations, min_operations

    return TechAugmentationParameters(
        enabled=bool(payload.get('enabled', defaults.enabled)),
        min_operations=min_operations,
        max_operations=max_operations,
        debug_return_pair=bool(payload.get('debug_return_pair', defaults.debug_return_pair)),
        binarization_threshold=_coerce_probability(
            payload.get('binarization_threshold', defaults.binarization_threshold),
            defaults.binarization_threshold,
        ),
        binary_tolerance=_coerce_probability(
            payload.get('binary_tolerance', defaults.binary_tolerance),
            defaults.binary_tolerance,
        ),
        max_changed_pixels_ratio=_coerce_probability(
            payload.get('max_changed_pixels_ratio', defaults.max_changed_pixels_ratio),
            defaults.max_changed_pixels_ratio,
        ),
        max_foreground_ratio_delta=_coerce_probability(
            payload.get('max_foreground_ratio_delta', defaults.max_foreground_ratio_delta),
            defaults.max_foreground_ratio_delta,
        ),
        global_width=TechGlobalWidthVariationParameters(
            probability=_coerce_probability(
                global_width_payload.get('probability', defaults.global_width.probability),
                defaults.global_width.probability,
            ),
            kernel_size_range=_coerce_int_range(
                global_width_payload.get('kernel_size_range', defaults.global_width.kernel_size_range),
                defaults.global_width.kernel_size_range,
                minimum=1,
            ),
            erosion_probability=_coerce_probability(
                global_width_payload.get('erosion_probability', defaults.global_width.erosion_probability),
                defaults.global_width.erosion_probability,
            ),
        ),
        scale_rethreshold=TechScaleRethresholdParameters(
            probability=_coerce_probability(
                scale_payload.get('probability', defaults.scale_rethreshold.probability),
                defaults.scale_rethreshold.probability,
            ),
            scale_range=_coerce_float_range(
                scale_payload.get('scale_range', defaults.scale_rethreshold.scale_range),
                defaults.scale_rethreshold.scale_range,
                minimum=0.1,
            ),
            threshold=_coerce_probability(
                scale_payload.get('threshold', defaults.scale_rethreshold.threshold),
                defaults.scale_rethreshold.threshold,
            ),
        ),
        blur_threshold=TechBlurThresholdParameters(
            probability=_coerce_probability(
                blur_payload.get('probability', defaults.blur_threshold.probability),
                defaults.blur_threshold.probability,
            ),
            blur_radius_range=_coerce_float_range(
                blur_payload.get('blur_radius_range', defaults.blur_threshold.blur_radius_range),
                defaults.blur_threshold.blur_radius_range,
                minimum=0.0,
            ),
            threshold=_coerce_probability(
                blur_payload.get('threshold', defaults.blur_threshold.threshold),
                defaults.blur_threshold.threshold,
            ),
        ),
        boundary_aware=TechBoundaryAwareVariationParameters(
            probability=_coerce_probability(
                boundary_payload.get('probability', defaults.boundary_aware.probability),
                defaults.boundary_aware.probability,
            ),
            band_width_range=_coerce_int_range(
                boundary_payload.get('band_width_range', defaults.boundary_aware.band_width_range),
                defaults.boundary_aware.band_width_range,
                minimum=1,
            ),
            noise_cell_size_range=_coerce_int_range(
                boundary_payload.get('noise_cell_size_range', defaults.boundary_aware.noise_cell_size_range),
                defaults.boundary_aware.noise_cell_size_range,
                minimum=1,
            ),
            add_probability=_coerce_probability(
                boundary_payload.get('add_probability', defaults.boundary_aware.add_probability),
                defaults.boundary_aware.add_probability,
            ),
            remove_probability=_coerce_probability(
                boundary_payload.get('remove_probability', defaults.boundary_aware.remove_probability),
                defaults.boundary_aware.remove_probability,
            ),
            min_addition_support=_coerce_positive_int(
                boundary_payload.get('min_addition_support', defaults.boundary_aware.min_addition_support),
                defaults.boundary_aware.min_addition_support,
            ),
            min_removal_support=_coerce_positive_int(
                boundary_payload.get('min_removal_support', defaults.boundary_aware.min_removal_support),
                defaults.boundary_aware.min_removal_support,
            ),
            smoothing_kernel_size=_coerce_positive_int(
                boundary_payload.get('smoothing_kernel_size', defaults.boundary_aware.smoothing_kernel_size),
                defaults.boundary_aware.smoothing_kernel_size,
                minimum=0,
            ),
        ),
        local_morphology=TechLocalMorphologyParameters(
            probability=_coerce_probability(
                local_payload.get('probability', defaults.local_morphology.probability),
                defaults.local_morphology.probability,
            ),
            roi_count_range=_coerce_int_range(
                local_payload.get('roi_count_range', defaults.local_morphology.roi_count_range),
                defaults.local_morphology.roi_count_range,
                minimum=1,
            ),
            roi_size_ratio_range=_coerce_float_range(
                local_payload.get('roi_size_ratio_range', defaults.local_morphology.roi_size_ratio_range),
                defaults.local_morphology.roi_size_ratio_range,
                minimum=0.01,
                maximum=1.0,
            ),
            kernel_size_range=_coerce_int_range(
                local_payload.get('kernel_size_range', defaults.local_morphology.kernel_size_range),
                defaults.local_morphology.kernel_size_range,
                minimum=1,
            ),
            erosion_probability=_coerce_probability(
                local_payload.get('erosion_probability', defaults.local_morphology.erosion_probability),
                defaults.local_morphology.erosion_probability,
            ),
        ),
        gap_variation=TechGapVariationParameters(
            probability=_coerce_probability(
                gap_payload.get('probability', defaults.gap_variation.probability),
                defaults.gap_variation.probability,
            ),
            kernel_size_range=_coerce_int_range(
                gap_payload.get('kernel_size_range', defaults.gap_variation.kernel_size_range),
                defaults.gap_variation.kernel_size_range,
                minimum=1,
            ),
            opening_probability=_coerce_probability(
                gap_payload.get('opening_probability', defaults.gap_variation.opening_probability),
                defaults.gap_variation.opening_probability,
            ),
            max_bridge_neighbor_count=_coerce_positive_int(
                gap_payload.get('max_bridge_neighbor_count', defaults.gap_variation.max_bridge_neighbor_count),
                defaults.gap_variation.max_bridge_neighbor_count,
                minimum=0,
            ),
            min_gap_neighbor_count=_coerce_positive_int(
                gap_payload.get('min_gap_neighbor_count', defaults.gap_variation.min_gap_neighbor_count),
                defaults.gap_variation.min_gap_neighbor_count,
                minimum=0,
            ),
        ),
    )


@dataclass
class SampleGenerationSettings:
    step: int
    segment_size: tuple[int,int]
    vertical_rotation: bool
    horizontal_rotation: bool
    channels: int
    flip_x: bool = False
    flip_y: bool = False
    additional_augmentation: bool = False
    augmentation_brightness_strength: float = 0.1
    augmentation_contrast_strength: float = 0.1
    augmentation_gamma_strength: float = 0.15
    augmentation_noise_probability: float = 0.5
    augmentation_noise_sigma: float = 0.01
    augmentation_blur_probability: float = 0.25
    augmentation_blur_radius: float = 1.0
    shuffle_patches_in_frame: bool = True
    random_crop: bool = False
    crops_per_image: int = 64
    scale_augmentation: bool = False
    scale_augmentation_strength: float = 0.2
    tech_aug: TechAugmentationParameters = field(default_factory=TechAugmentationParameters)

@dataclass
class SamplePrepareSettings:
    enable_crop: bool = False
    enable_resize: bool = False
    edge_cut:tuple[int,int]|None = None
    target_size:tuple[int,int]|None = None

@dataclass
class TrainingParameters:
    image_path:Path
    label_path:Path
    shuffle:bool
    validation:bool
    validation_percent:int
    batch_size: int
    cut_mode:SampleCutMode
    colors: int
    epochs: int
    generation:SampleGenerationSettings
    prepare:SamplePrepareSettings
    validation_source: str = ValidationSource.split.value
    validation_image_path: Path | None = None
    validation_label_path: Path | None = None
    save_validation_binary_images: bool = False
    optimizer: OptimizerParameters = field(default_factory=OptimizerParameters)
    mixed_precision: MixedPrecisionMode = MixedPrecisionMode.bf16
    loss_function: str = 'bce'
    loss_term_weights: dict[str, float] = field(default_factory=dict)
    dice_loss_weight: float = 0.5
    iou_loss_weight: float = 0.5
    early_stopping: EarlyStoppingParameters = field(default_factory=EarlyStoppingParameters)
    warmup: WarmupParameters = field(default_factory=WarmupParameters)
    scheduler: SchedulerParameters = field(default_factory=SchedulerParameters)
    hard_mining: HardMiningParameters = field(default_factory=HardMiningParameters)
    cutout: CutoutParameters = field(default_factory=CutoutParameters)
    random_artifacts: RandomArtifactsParameters = field(default_factory=RandomArtifactsParameters)
    mixup: MixupParameters = field(default_factory=MixupParameters)
    skip_uniform_labels: bool = False
    rare_patch_oversampling_enabled: bool = False
    rare_patch_oversampling_factor: int = 2
    use_multi_gpu: bool = True
    multi_gpu_mode: str = ''
    show_batch_preview: bool = True
    log_update_frequency: int = 0
    local_crop_size: tuple[int, int] | None = None
    context_crop_size: tuple[int, int] | None = None
    context_input_size: tuple[int, int] | None = None
    context_branch_channels: tuple[int, ...] = (16, 32, 64, 128)
    fusion_type: str = 'concat'
    use_context_branch: bool | None = None
    deep_supervision: bool = True
    artifact_dir: Path | None = None
    dataloader_num_workers: int = -1
    pcb_defects: PCBDefectParameters = field(default_factory=PCBDefectParameters)
    synthetic_defect_generator: SyntheticDefectGeneratorParameters = field(
        default_factory=SyntheticDefectGeneratorParameters
    )

@dataclass
class RecognitionParameters:
    source_files: list[Path]
    result_folder: Path
    model: str | Path | Any
    part_size: tuple[int,int]
    batch_size: int
    overlap: int
    source_folder: Path | None = None
    jpeg_quality: int = 95
    recognition_multiprocessing_enabled: bool = True
    binarize_output: bool = True
    use_auto_threshold: bool = True
    threshold: float = 0.5
    postprocess_enabled: bool = False
    postprocess_kernel_size: int = 3
    recognition_tta_enabled: bool = False
    confidence_tta_enabled: bool = False
    confidence_save_mode: str = ConfidenceSaveMode.off.value
    use_context_branch: bool | None = None
    context_crop_size: tuple[int, int] | None = None
    context_input_size: tuple[int, int] | None = None


@dataclass
class CutSettings:
    vertical_rotation: bool
    horizontal_rotation: bool
    step: int
    color_mode:str
    x_size: int
    y_size: int
    model: str
    flip_x: bool = False
    flip_y: bool = False
    additional_augmentation: bool = False
    augmentation_gamma_strength: float = 0.15
    augmentation_blur_probability: float = 0.25
    augmentation_blur_radius: float = 1.0
    random_crop: bool = False
    crops_per_image: int = 64
    scale_augmentation: bool = False
    scale_augmentation_strength: float = 0.2


@dataclass
class NeuralThreadConfig:
    source_folder: str
    result_folder: str
    ready_model: bool
    model_path: str
    model: Any #temporal while not
    sample_image: str
    sample_label: str
    epochs: int
    train_params: CutSettings
