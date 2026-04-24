import copy
from dataclasses import dataclass, field, replace
from typing import Any

from neuralimage.lib.data_interfaces import WorkMode, normalize_work_mode


@dataclass
class MainWindowState:
    work_mode: str = ''
    source_folder: str = ''
    result_folder: str = ''
    model_path: str = ''
    label_folder: str = ''
    sample_folder: str = ''
    epochs: int = 20
    ui_mode: str = 'simple'
    mode_state: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class SettingsState:
    step: int = 100
    vertical_rotation: bool = True
    horizontal_rotation: bool = True
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
    validation_source: str = 'split'
    validation_image_folder: str = ''
    validation_label_folder: str = ''
    save_validation_binary_images: bool = False
    sample_cut_mode: str = 'online'
    batch_size: int = 16
    dataloader_num_workers: int = -1
    train_batch_size: int | None = None
    recognition_batch_size: int | None = None
    sync_patch_sizes: bool = True
    patch_batch_sync_mode: str = 'patch_and_batch'
    overlap: int = 8
    recognition_jpeg_quality: int = 95
    recognition_multiprocessing_enabled: bool = True
    recognition_binarize_output: bool = True
    recognition_use_auto_threshold: bool = True
    recognition_threshold: float = 0.5
    recognition_postprocess: bool = False
    recognition_postprocess_kernel_size: int = 3
    recognition_tta_enabled: bool = False
    confidence_tta_enabled: bool = False
    confidence_save_mode: str = 'off'
    crop_enabled: bool = False
    resize_enabled: bool = False
    edge_cut_size: int = 0
    target_size: tuple[int, int] = (2000, 2000)
    compression_factor: int = 1
    recursive_file_search: bool = False
    optimizer_name: str = 'adam'
    mixed_precision: str = 'fp16'
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
    scheduler_name: str = 'off'
    scheduler_plateau_factor: float = 0.5
    scheduler_plateau_patience: int = 3
    scheduler_plateau_threshold: float = 1e-4
    scheduler_plateau_min_lr: float = 1e-6
    scheduler_plateau_cooldown: int = 0
    scheduler_cosine_t_max: int = 10
    scheduler_cosine_eta_min: float = 1e-6
    scheduler_one_cycle_max_lr: float = 1e-3
    scheduler_one_cycle_pct_start: float = 0.3
    scheduler_one_cycle_anneal_strategy: str = 'cos'
    scheduler_one_cycle_div_factor: float = 25.0
    scheduler_one_cycle_final_div_factor: float = 10000.0
    scheduler_one_cycle_three_phase: bool = False
    scheduler_step_lr_step_size: int = 10
    scheduler_step_lr_gamma: float = 0.1
    hard_mining_enabled: bool = False
    hard_mining_strength: float = 2.0
    hard_mining_ema_alpha: float = 0.2
    hard_pixel_mining_enabled: bool = False
    hard_pixel_mining_ratio: float = 0.25
    cutout_enabled: bool = False
    cutout_probability: float = 1.0
    cutout_holes: int = 1
    cutout_size_ratio: float = 0.25
    random_artifacts_enabled: bool = False
    random_artifacts_probability: float = 1.0
    random_artifacts_count: int = 1
    random_artifacts_size_ratio: float = 0.25
    random_artifacts_dust_enabled: bool = True
    random_artifacts_resist_residue_enabled: bool = True
    random_artifacts_etch_residue_enabled: bool = True
    random_artifacts_particle_cluster_enabled: bool = True
    random_artifacts_flake_enabled: bool = True
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
    local_crop_size: tuple[int, int] | None = None
    context_crop_size: tuple[int, int] | None = None
    context_input_size: tuple[int, int] | None = None
    context_branch_channels: tuple[int, ...] = (16, 32, 64, 128)
    fusion_type: str = 'concat'
    use_context_branch: bool | None = None
    deep_supervision: bool = False
    synthetic_defect_generator: dict[str, Any] = field(default_factory=dict)
    tech_aug: dict[str, Any] = field(default_factory=dict)
    pcb_defects: dict[str, Any] = field(default_factory=dict)


_MODE_STATE_SUPPORTED_MODES = {
    WorkMode.train_and_recognition.value,
    WorkMode.further_training.value,
    WorkMode.recognition_only.value,
    WorkMode.train_only.value,
}
_MODE_STATE_PATH_KEYS = (
    'source_folder',
    'result_folder',
    'model_path',
    'label_folder',
    'sample_folder',
)


def build_main_window_mode_state_entry(
    *,
    source_folder: str = '',
    result_folder: str = '',
    model_path: str = '',
    label_folder: str = '',
    sample_folder: str = '',
    epochs: int | str = 20,
) -> dict[str, Any]:
    try:
        normalized_epochs = int(epochs)
    except (TypeError, ValueError):
        normalized_epochs = MainWindowState().epochs
    return {
        'source_folder': str(source_folder or ''),
        'result_folder': str(result_folder or ''),
        'model_path': str(model_path or ''),
        'label_folder': str(label_folder or ''),
        'sample_folder': str(sample_folder or ''),
        'epochs': normalized_epochs,
    }


def build_main_window_mode_state_entry_from_state(state: MainWindowState) -> dict[str, Any]:
    return build_main_window_mode_state_entry(
        source_folder=getattr(state, 'source_folder', ''),
        result_folder=getattr(state, 'result_folder', ''),
        model_path=getattr(state, 'model_path', ''),
        label_folder=getattr(state, 'label_folder', ''),
        sample_folder=getattr(state, 'sample_folder', ''),
        epochs=getattr(state, 'epochs', MainWindowState().epochs),
    )


def normalize_main_window_mode_state(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for raw_mode, raw_entry in value.items():
        mode = normalize_work_mode(raw_mode)
        if mode not in _MODE_STATE_SUPPORTED_MODES or not isinstance(raw_entry, dict):
            continue
        normalized[mode] = build_main_window_mode_state_entry(
            source_folder=raw_entry.get('source_folder', ''),
            result_folder=raw_entry.get('result_folder', ''),
            model_path=raw_entry.get('model_path', ''),
            label_folder=raw_entry.get('label_folder', ''),
            sample_folder=raw_entry.get('sample_folder', ''),
            epochs=raw_entry.get('epochs', MainWindowState().epochs),
        )
    return normalized


def resolve_main_window_mode_state_entry(state: MainWindowState, mode: str) -> dict[str, Any]:
    normalized_mode = normalize_work_mode(mode)
    mode_state = normalize_main_window_mode_state(getattr(state, 'mode_state', {}))
    entry = mode_state.get(normalized_mode)
    if entry is not None:
        return build_main_window_mode_state_entry(**entry)

    current_mode = normalize_work_mode(getattr(state, 'work_mode', ''))
    if current_mode == normalized_mode:
        return build_main_window_mode_state_entry_from_state(state)
    return build_main_window_mode_state_entry()


def clone_main_window_state(state: MainWindowState) -> MainWindowState:
    return replace(
        state,
        mode_state=copy.deepcopy(normalize_main_window_mode_state(getattr(state, 'mode_state', {}))),
    )
