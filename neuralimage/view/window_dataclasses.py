from dataclasses import dataclass
from pathlib import Path

from lib.data_interfaces import WorkMode


@dataclass
class MainWindowState:
    work_mode: str = ''
    source_folder: str = ''
    result_folder: str = ''
    model_path: str = ''
    label_folder: str = ''
    sample_folder: str = ''
    epochs: int = 20

@dataclass
class SettingsState:
    step:int = 100
    vertical_rotation:bool = True
    horizontal_rotation:bool = True
    sample_size:tuple[int,int] = (256,256)
    model:str = 'M 720k'
    color_mode:str = 'RGB'
    shuffle:bool = True
    use_validation:bool = False
    validation_percent:int = 20
    sample_cut_mode:str = 'online' #disk/online
    batch_size:int = 16
    overlap:int = 8
    additional_processing: bool = False
    edge_cut_size:int = 0
    target_size:tuple[int,int] = (2000,2000)
    optimizer_name: str = 'adam'
    mixed_precision: str = 'bf16'
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    early_stopping_restore_best_weights: bool = True
    warmup_enabled: bool = False
    warmup_epochs: int = 3
    warmup_start_factor: float = 0.1
    use_multi_gpu: bool = True
    show_batch_preview: bool = True
