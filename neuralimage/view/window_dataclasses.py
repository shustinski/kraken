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
    additional_augmentation: bool = False
    augmentation_brightness_strength: float = 0.1
    augmentation_contrast_strength: float = 0.1
    augmentation_noise_probability: float = 0.5
    augmentation_noise_sigma: float = 0.01
    sample_size:tuple[int,int] = (256,256)
    model:str = 'M 720k'
    color_mode:str = 'RGB'
    shuffle:bool = True
    use_validation:bool = False
    validation_percent:int = 20
    sample_cut_mode:str = 'online' #disk/online
    batch_size:int = 16
    overlap:int = 8
    crop_enabled: bool = False
    resize_enabled: bool = False
    edge_cut_size:int = 0
    target_size:tuple[int,int] = (2000,2000)
    optimizer_name: str = 'adam'
    mixed_precision: str = 'bf16'
    loss_function: str = 'bce'
    dice_loss_weight: float = 0.5
    iou_loss_weight: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    early_stopping_restore_best_weights: bool = True
    warmup_enabled: bool = False
    warmup_epochs: int = 3
    warmup_start_factor: float = 0.1
    hard_mining_enabled: bool = False
    hard_mining_strength: float = 2.0
    hard_mining_ema_alpha: float = 0.2
    skip_uniform_labels: bool = False
    use_multi_gpu: bool = True
    torch_compile_enabled: bool = True
    show_batch_preview: bool = True
    log_update_frequency: int = 0
