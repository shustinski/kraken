from dataclasses import dataclass, field


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
    step: int = 100
    vertical_rotation: bool = True
    horizontal_rotation: bool = True
    additional_augmentation: bool = False
    augmentation_brightness_strength: float = 0.1
    augmentation_contrast_strength: float = 0.1
    augmentation_gamma_strength: float = 0.15
    augmentation_noise_probability: float = 0.5
    augmentation_noise_sigma: float = 0.01
    augmentation_blur_probability: float = 0.25
    augmentation_blur_radius: float = 1.0
    sample_size: tuple[int, int] = (256, 256)
    train_patch_size: tuple[int, int] | None = None
    recognition_patch_size: tuple[int, int] | None = None
    model: str = 'M 720k'
    color_mode: str = 'RGB'
    shuffle: bool = True
    shuffle_patches_in_frame: bool = True
    random_crop: bool = False
    crops_per_image: int = 64
    scale_augmentation: bool = False
    scale_augmentation_strength: float = 0.2
    use_validation: bool = False
    validation_percent: int = 20
    sample_cut_mode: str = 'online'
    batch_size: int = 16
    train_batch_size: int | None = None
    recognition_batch_size: int | None = None
    sync_patch_sizes: bool = True
    patch_batch_sync_mode: str = 'patch_and_batch'
    overlap: int = 8
    recognition_jpeg_quality: int = 95
    recognition_binarize_output: bool = True
    recognition_use_auto_threshold: bool = True
    recognition_threshold: float = 0.5
    recognition_postprocess: bool = False
    recognition_postprocess_kernel_size: int = 3
    crop_enabled: bool = False
    resize_enabled: bool = False
    edge_cut_size: int = 0
    target_size: tuple[int, int] = (2000, 2000)
    optimizer_name: str = 'adam'
    mixed_precision: str = 'bf16'
    loss_function: str = 'bce'
    loss_term_weights: dict[str, float] = field(default_factory=lambda: {'bce': 1.0})
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
    hard_pixel_mining_enabled: bool = False
    hard_pixel_mining_ratio: float = 0.25
    cutout_enabled: bool = False
    cutout_probability: float = 1.0
    cutout_holes: int = 1
    cutout_size_ratio: float = 0.25
    mixup_enabled: bool = False
    mixup_probability: float = 1.0
    mixup_alpha: float = 0.2
    skip_uniform_labels: bool = False
    rare_patch_oversampling_enabled: bool = False
    rare_patch_oversampling_factor: int = 2
    use_multi_gpu: bool = True
    multi_gpu_mode: str = ''
    torch_compile_enabled: bool = True
    show_batch_preview: bool = True
    log_update_frequency: int = 0
