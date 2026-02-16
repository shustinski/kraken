import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class  WorkMode(enum.Enum):
    train_only = 'train_only'
    train_and_recognition = 'train_and_recognition'
    recognition_only = 'recognintion_only'
    futher_training = 'futher_training'

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
class SampleGenerationSettings:
    step: int
    segment_size: tuple[int,int]
    vertical_rotation: bool
    horizontal_rotation: bool
    channels: int

@dataclass
class SamplePrepareSettings:
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
    early_stopping: EarlyStoppingParameters = field(default_factory=EarlyStoppingParameters)
    warmup: WarmupParameters = field(default_factory=WarmupParameters)
    use_multi_gpu: bool = True
    show_batch_preview: bool = True

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
