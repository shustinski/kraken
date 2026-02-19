import enum
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

class SampleCutMode(enum.Enum):
    disk = 'disk'
    online = 'online'


class OptimizerName(enum.Enum):
    adam = 'adam'
    adamw = 'adamw'
    adamw_muon = 'adamw_muon'


class MixedPrecisionMode(enum.Enum):
    off = 'off'
    fp16 = 'fp16'
    bf16 = 'bf16'


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
class HardMiningParameters:
    enabled: bool = False
    strength: float = 2.0
    ema_alpha: float = 0.2


@dataclass
class SampleGenerationSettings:
    step: int
    segment_size: tuple[int,int]
    vertical_rotation: bool
    horizontal_rotation: bool
    channels: int
    additional_augmentation: bool = False
    augmentation_brightness_strength: float = 0.1
    augmentation_contrast_strength: float = 0.1
    augmentation_noise_probability: float = 0.5
    augmentation_noise_sigma: float = 0.01

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
    optimizer: OptimizerParameters = field(default_factory=OptimizerParameters)
    mixed_precision: MixedPrecisionMode = MixedPrecisionMode.bf16
    loss_function: str = 'bce'
    dice_loss_weight: float = 0.5
    iou_loss_weight: float = 0.5
    early_stopping: EarlyStoppingParameters = field(default_factory=EarlyStoppingParameters)
    warmup: WarmupParameters = field(default_factory=WarmupParameters)
    hard_mining: HardMiningParameters = field(default_factory=HardMiningParameters)
    skip_uniform_labels: bool = False
    use_multi_gpu: bool = True
    show_batch_preview: bool = True
    log_update_frequency: int = 0

@dataclass
class RecognitionParameters:
    source_files: list[Path]
    result_folder: Path
    model: str | Path | Any
    part_size: tuple[int,int]
    batch_size: int
    overlap: int


@dataclass
class CutSettings:
    vertical_rotation: bool
    horizontal_rotation: bool
    step: int
    color_mode:str
    x_size: int
    y_size: int
    model: str
    additional_augmentation: bool = False


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
