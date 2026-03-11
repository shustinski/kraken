import configparser
import os
from pathlib import Path

from PyQt6.QtCore import QSettings

from application.dto import MainWindowState, SettingsState
from application.ports import StateStore
from lib.data_interfaces import (
    normalize_multi_gpu_mode,
    normalize_patch_batch_sync_mode,
    normalize_work_mode,
)
from lib.loss_config import (
    deserialize_loss_term_weights,
    dominant_loss_function,
    normalize_loss_term_name,
    serialize_loss_term_weights,
)


MAIN_WINDOW_ORG = 'NeuralImage'
MAIN_WINDOW_APP = 'MainWindow'
SETTINGS_ORG = 'NeuralImage'
SETTINGS_APP = 'Settings'

INI_SECTION_MAIN = 'main_window'
INI_SECTION_SETTINGS = 'settings'


def _build_main_window_state(
    *,
    read_str,
    read_int,
) -> MainWindowState:
    defaults = MainWindowState()
    return MainWindowState(
        work_mode=normalize_work_mode(read_str('work_mode', defaults.work_mode)),
        source_folder=read_str('source_path', defaults.source_folder),
        result_folder=read_str('result_path', defaults.result_folder),
        model_path=read_str('model_path', defaults.model_path),
        label_folder=read_str('label_path', defaults.label_folder),
        sample_folder=read_str('sample_path', defaults.sample_folder),
        epochs=read_int('epochs', defaults.epochs),
    )


def _main_window_state_to_storage_dict(state: MainWindowState) -> dict[str, str | int]:
    return {
        'work_mode': normalize_work_mode(state.work_mode),
        'source_path': state.source_folder,
        'result_path': state.result_folder,
        'model_path': state.model_path,
        'label_path': state.label_folder,
        'sample_path': state.sample_folder,
        'epochs': int(state.epochs),
    }


def _build_settings_state(
    *,
    read_bool,
    read_int,
    read_float,
    read_str,
) -> SettingsState:
    defaults = SettingsState()
    legacy_additional_processing = read_bool('additional_processing', False)
    legacy_use_multi_gpu = read_bool('use_multi_gpu', defaults.use_multi_gpu)
    multi_gpu_mode = normalize_multi_gpu_mode(
        read_str('multi_gpu_mode', defaults.multi_gpu_mode),
        use_multi_gpu_fallback=legacy_use_multi_gpu,
    )
    legacy_sample_x = read_int('sample_x_size', defaults.sample_size[0])
    legacy_sample_y = read_int('sample_y_size', defaults.sample_size[1])
    legacy_batch_size = read_int('batch_size', defaults.batch_size)
    legacy_shuffle = read_bool('shuffle', defaults.shuffle)
    patch_batch_sync_mode = normalize_patch_batch_sync_mode(
        read_str('patch_batch_sync_mode', defaults.patch_batch_sync_mode)
    )
    sync_patch_sizes = bool(
        read_bool('sync_patch_sizes', patch_batch_sync_mode in ('patch', 'patch_and_batch'))
    )
    rare_patch_oversampling_factor = max(
        2,
        read_int(
            'rare_patch_oversampling_factor',
            getattr(defaults, 'rare_patch_oversampling_factor', 2),
        ),
    )
    legacy_loss_function = read_str('loss_function', defaults.loss_function)
    loss_term_weights_raw = read_str('loss_term_weights_json', '')
    loss_term_weights = deserialize_loss_term_weights(loss_term_weights_raw)
    if (not loss_term_weights) and (not str(loss_term_weights_raw).strip()):
        legacy_name = normalize_loss_term_name(legacy_loss_function) or defaults.loss_function
        loss_term_weights = {legacy_name: 1.0}
    return SettingsState(
        step=read_int('cut_step', defaults.step),
        vertical_rotation=read_bool('vertical_rotation', defaults.vertical_rotation),
        horizontal_rotation=read_bool('horizontal_rotation', defaults.horizontal_rotation),
        additional_augmentation=read_bool('additional_augmentation', defaults.additional_augmentation),
        augmentation_brightness_strength=read_float(
            'augmentation_brightness_strength', defaults.augmentation_brightness_strength
        ),
        augmentation_contrast_strength=read_float(
            'augmentation_contrast_strength', defaults.augmentation_contrast_strength
        ),
        augmentation_gamma_strength=read_float(
            'augmentation_gamma_strength',
            getattr(defaults, 'augmentation_gamma_strength', 0.15),
        ),
        augmentation_noise_probability=read_float(
            'augmentation_noise_probability', defaults.augmentation_noise_probability
        ),
        augmentation_noise_sigma=read_float('augmentation_noise_sigma', defaults.augmentation_noise_sigma),
        augmentation_blur_probability=read_float(
            'augmentation_blur_probability',
            getattr(defaults, 'augmentation_blur_probability', 0.25),
        ),
        augmentation_blur_radius=read_float(
            'augmentation_blur_radius',
            getattr(defaults, 'augmentation_blur_radius', 1.0),
        ),
        sample_size=(
            legacy_sample_x,
            legacy_sample_y,
        ),
        train_patch_size=(
            read_int('train_patch_x_size', legacy_sample_x),
            read_int('train_patch_y_size', legacy_sample_y),
        ),
        recognition_patch_size=(
            read_int('recognition_patch_x_size', legacy_sample_x),
            read_int('recognition_patch_y_size', legacy_sample_y),
        ),
        model=read_str('model', defaults.model),
        color_mode=read_str('color_mode', defaults.color_mode),
        shuffle=legacy_shuffle,
        shuffle_patches_in_frame=read_bool(
            'shuffle_patches_in_frame',
            legacy_shuffle,
        ),
        random_crop=read_bool('random_crop', defaults.random_crop),
        crops_per_image=read_int('crops_per_image', defaults.crops_per_image),
        scale_augmentation=read_bool('scale_augmentation', defaults.scale_augmentation),
        scale_augmentation_strength=read_float(
            'scale_augmentation_strength',
            defaults.scale_augmentation_strength,
        ),
        use_validation=read_bool('validation', defaults.use_validation),
        validation_percent=read_int('validation_percent', defaults.validation_percent),
        sample_cut_mode=read_str('sample_cut_mode', defaults.sample_cut_mode),
        batch_size=legacy_batch_size,
        train_batch_size=read_int('train_batch_size', legacy_batch_size),
        recognition_batch_size=read_int('recognition_batch_size', legacy_batch_size),
        sync_patch_sizes=sync_patch_sizes,
        patch_batch_sync_mode=patch_batch_sync_mode,
        overlap=read_int('overlap', defaults.overlap),
        recognition_jpeg_quality=read_int('recognition_jpeg_quality', defaults.recognition_jpeg_quality),
        recognition_binarize_output=read_bool(
            'recognition_binarize_output',
            getattr(defaults, 'recognition_binarize_output', True),
        ),
        recognition_use_auto_threshold=read_bool(
            'recognition_use_auto_threshold',
            getattr(defaults, 'recognition_use_auto_threshold', True),
        ),
        recognition_threshold=read_float(
            'recognition_threshold',
            getattr(defaults, 'recognition_threshold', 0.5),
        ),
        recognition_postprocess=read_bool(
            'recognition_postprocess',
            getattr(defaults, 'recognition_postprocess', False),
        ),
        recognition_postprocess_kernel_size=max(
            1,
            read_int(
                'recognition_postprocess_kernel_size',
                getattr(defaults, 'recognition_postprocess_kernel_size', 3),
            ),
        ),
        log_update_frequency=read_int('log_update_frequency', defaults.log_update_frequency),
        crop_enabled=read_bool('crop_enabled', legacy_additional_processing),
        resize_enabled=read_bool('resize_enabled', legacy_additional_processing),
        edge_cut_size=read_int('edge_cut_size', defaults.edge_cut_size),
        target_size=(
            read_int('target_x_size', defaults.target_size[0]),
            read_int('target_y_size', defaults.target_size[1]),
        ),
        optimizer_name=read_str('optimizer_name', defaults.optimizer_name),
        mixed_precision=read_str('mixed_precision', defaults.mixed_precision),
        loss_function=normalize_loss_term_name(legacy_loss_function)
        or dominant_loss_function(loss_term_weights, fallback=defaults.loss_function),
        loss_term_weights=loss_term_weights,
        dice_loss_weight=read_float('dice_loss_weight', defaults.dice_loss_weight),
        iou_loss_weight=read_float('iou_loss_weight', defaults.iou_loss_weight),
        learning_rate=read_float('learning_rate', defaults.learning_rate),
        weight_decay=read_float('weight_decay', defaults.weight_decay),
        early_stopping_enabled=read_bool('early_stopping_enabled', defaults.early_stopping_enabled),
        early_stopping_patience=read_int('early_stopping_patience', defaults.early_stopping_patience),
        early_stopping_min_delta=read_float('early_stopping_min_delta', defaults.early_stopping_min_delta),
        early_stopping_restore_best_weights=read_bool(
            'early_stopping_restore_best_weights', defaults.early_stopping_restore_best_weights
        ),
        warmup_enabled=read_bool('warmup_enabled', defaults.warmup_enabled),
        warmup_epochs=read_int('warmup_epochs', defaults.warmup_epochs),
        warmup_start_factor=read_float('warmup_start_factor', defaults.warmup_start_factor),
        hard_mining_enabled=read_bool('hard_mining_enabled', defaults.hard_mining_enabled),
        hard_mining_strength=read_float('hard_mining_strength', defaults.hard_mining_strength),
        hard_mining_ema_alpha=read_float('hard_mining_ema_alpha', defaults.hard_mining_ema_alpha),
        hard_pixel_mining_enabled=read_bool(
            'hard_pixel_mining_enabled',
            getattr(defaults, 'hard_pixel_mining_enabled', False),
        ),
        hard_pixel_mining_ratio=read_float(
            'hard_pixel_mining_ratio',
            getattr(defaults, 'hard_pixel_mining_ratio', 0.25),
        ),
        cutout_enabled=read_bool(
            'cutout_enabled',
            getattr(defaults, 'cutout_enabled', False),
        ),
        cutout_probability=read_float(
            'cutout_probability',
            getattr(defaults, 'cutout_probability', 1.0),
        ),
        cutout_holes=max(
            1,
            read_int(
                'cutout_holes',
                getattr(defaults, 'cutout_holes', 1),
            ),
        ),
        cutout_size_ratio=read_float(
            'cutout_size_ratio',
            getattr(defaults, 'cutout_size_ratio', 0.25),
        ),
        mixup_enabled=read_bool(
            'mixup_enabled',
            getattr(defaults, 'mixup_enabled', False),
        ),
        mixup_probability=read_float(
            'mixup_probability',
            getattr(defaults, 'mixup_probability', 1.0),
        ),
        mixup_alpha=read_float(
            'mixup_alpha',
            getattr(defaults, 'mixup_alpha', 0.2),
        ),
        skip_uniform_labels=read_bool('skip_uniform_labels', defaults.skip_uniform_labels),
        rare_patch_oversampling_enabled=read_bool(
            'rare_patch_oversampling_enabled',
            getattr(defaults, 'rare_patch_oversampling_enabled', False),
        ),
        rare_patch_oversampling_factor=rare_patch_oversampling_factor,
        use_multi_gpu=multi_gpu_mode != 'off',
        multi_gpu_mode=multi_gpu_mode,
        torch_compile_enabled=read_bool('torch_compile_enabled', defaults.torch_compile_enabled),
        show_batch_preview=read_bool('show_batch_preview', defaults.show_batch_preview),
    )


def _settings_state_to_storage_dict(state: SettingsState) -> dict[str, str | int | float | bool]:
    multi_gpu_mode = normalize_multi_gpu_mode(
        getattr(state, 'multi_gpu_mode', ''),
        use_multi_gpu_fallback=bool(getattr(state, 'use_multi_gpu', False)),
    )
    train_patch_size = tuple(getattr(state, 'train_patch_size', None) or state.sample_size)
    recognition_patch_size = tuple(getattr(state, 'recognition_patch_size', None) or state.sample_size)
    train_batch_size = int(getattr(state, 'train_batch_size', None) or state.batch_size)
    recognition_batch_size = int(getattr(state, 'recognition_batch_size', None) or state.batch_size)
    patch_batch_sync_mode = normalize_patch_batch_sync_mode(getattr(state, 'patch_batch_sync_mode', ''))
    sync_patch_sizes = bool(
        getattr(state, 'sync_patch_sizes', patch_batch_sync_mode in ('patch', 'patch_and_batch'))
    )
    patch_batch_sync_mode = 'patch' if sync_patch_sizes else 'off'
    return {
        'cut_step': int(state.step),
        'horizontal_rotation': bool(state.horizontal_rotation),
        'vertical_rotation': bool(state.vertical_rotation),
        'additional_augmentation': bool(state.additional_augmentation),
        'augmentation_brightness_strength': float(state.augmentation_brightness_strength),
        'augmentation_contrast_strength': float(state.augmentation_contrast_strength),
        'augmentation_gamma_strength': float(getattr(state, 'augmentation_gamma_strength', 0.15)),
        'augmentation_noise_probability': float(state.augmentation_noise_probability),
        'augmentation_noise_sigma': float(state.augmentation_noise_sigma),
        'augmentation_blur_probability': float(getattr(state, 'augmentation_blur_probability', 0.25)),
        'augmentation_blur_radius': float(getattr(state, 'augmentation_blur_radius', 1.0)),
        'model': state.model,
        'color_mode': state.color_mode,
        'shuffle': bool(state.shuffle),
        'shuffle_patches_in_frame': bool(getattr(state, 'shuffle_patches_in_frame', state.shuffle)),
        'random_crop': bool(getattr(state, 'random_crop', False)),
        'crops_per_image': int(getattr(state, 'crops_per_image', 64)),
        'scale_augmentation': bool(getattr(state, 'scale_augmentation', False)),
        'scale_augmentation_strength': float(getattr(state, 'scale_augmentation_strength', 0.2)),
        'validation': bool(state.use_validation),
        'validation_percent': int(state.validation_percent),
        'sample_cut_mode': state.sample_cut_mode,
        'batch_size': int(train_batch_size),
        'train_batch_size': int(train_batch_size),
        'recognition_batch_size': int(recognition_batch_size),
        'sync_patch_sizes': bool(sync_patch_sizes),
        'patch_batch_sync_mode': patch_batch_sync_mode,
        'overlap': int(state.overlap),
        'recognition_jpeg_quality': int(getattr(state, 'recognition_jpeg_quality', 95)),
        'recognition_binarize_output': bool(getattr(state, 'recognition_binarize_output', True)),
        'recognition_use_auto_threshold': bool(getattr(state, 'recognition_use_auto_threshold', True)),
        'recognition_threshold': float(getattr(state, 'recognition_threshold', 0.5)),
        'recognition_postprocess': bool(getattr(state, 'recognition_postprocess', False)),
        'recognition_postprocess_kernel_size': int(
            max(1, int(getattr(state, 'recognition_postprocess_kernel_size', 3)))
        ),
        'log_update_frequency': int(state.log_update_frequency),
        'crop_enabled': bool(state.crop_enabled),
        'resize_enabled': bool(state.resize_enabled),
        'additional_processing': bool(state.crop_enabled or state.resize_enabled),
        'edge_cut_size': int(state.edge_cut_size),
        'sample_x_size': int(train_patch_size[0]),
        'sample_y_size': int(train_patch_size[1]),
        'train_patch_x_size': int(train_patch_size[0]),
        'train_patch_y_size': int(train_patch_size[1]),
        'recognition_patch_x_size': int(recognition_patch_size[0]),
        'recognition_patch_y_size': int(recognition_patch_size[1]),
        'target_x_size': int(state.target_size[0]),
        'target_y_size': int(state.target_size[1]),
        'optimizer_name': state.optimizer_name,
        'mixed_precision': state.mixed_precision,
        'loss_function': normalize_loss_term_name(state.loss_function)
        or dominant_loss_function(
            getattr(state, 'loss_term_weights', None),
            fallback='bce',
        ),
        'loss_term_weights_json': serialize_loss_term_weights(getattr(state, 'loss_term_weights', None)),
        'dice_loss_weight': float(state.dice_loss_weight),
        'iou_loss_weight': float(state.iou_loss_weight),
        'learning_rate': float(state.learning_rate),
        'weight_decay': float(state.weight_decay),
        'early_stopping_enabled': bool(state.early_stopping_enabled),
        'early_stopping_patience': int(state.early_stopping_patience),
        'early_stopping_min_delta': float(state.early_stopping_min_delta),
        'early_stopping_restore_best_weights': bool(state.early_stopping_restore_best_weights),
        'warmup_enabled': bool(state.warmup_enabled),
        'warmup_epochs': int(state.warmup_epochs),
        'warmup_start_factor': float(state.warmup_start_factor),
        'hard_mining_enabled': bool(state.hard_mining_enabled),
        'hard_mining_strength': float(state.hard_mining_strength),
        'hard_mining_ema_alpha': float(state.hard_mining_ema_alpha),
        'hard_pixel_mining_enabled': bool(getattr(state, 'hard_pixel_mining_enabled', False)),
        'hard_pixel_mining_ratio': float(getattr(state, 'hard_pixel_mining_ratio', 0.25)),
        'cutout_enabled': bool(getattr(state, 'cutout_enabled', False)),
        'cutout_probability': float(getattr(state, 'cutout_probability', 1.0)),
        'cutout_holes': int(max(1, int(getattr(state, 'cutout_holes', 1)))),
        'cutout_size_ratio': float(getattr(state, 'cutout_size_ratio', 0.25)),
        'mixup_enabled': bool(getattr(state, 'mixup_enabled', False)),
        'mixup_probability': float(getattr(state, 'mixup_probability', 1.0)),
        'mixup_alpha': float(getattr(state, 'mixup_alpha', 0.2)),
        'skip_uniform_labels': bool(state.skip_uniform_labels),
        'rare_patch_oversampling_enabled': bool(
            getattr(state, 'rare_patch_oversampling_enabled', False)
        ),
        'rare_patch_oversampling_factor': int(
            max(2, int(getattr(state, 'rare_patch_oversampling_factor', 2)))
        ),
        'use_multi_gpu': bool(multi_gpu_mode != 'off'),
        'multi_gpu_mode': multi_gpu_mode,
        'torch_compile_enabled': bool(state.torch_compile_enabled),
        'show_batch_preview': bool(state.show_batch_preview),
    }


class QSettingsStateStore:
    def _settings(self, organization: str, application: str) -> QSettings:
        root = os.getenv('NEURALIMAGE_SETTINGS_DIR')
        if root:
            settings_root = Path(root)
            settings_root.mkdir(parents=True, exist_ok=True)
            return QSettings(
                str(settings_root / f'{organization}_{application}.ini'),
                QSettings.Format.IniFormat,
            )
        return QSettings(organization, application)

    def load_main_window_state(self) -> MainWindowState:
        settings = self._settings(MAIN_WINDOW_ORG, MAIN_WINDOW_APP)
        state = _build_main_window_state(
            read_str=lambda key, default: settings.value(key, default, type=str),
            read_int=lambda key, default: settings.value(key, default, type=int),
        )
        settings.sync()
        return state

    def save_main_window_state(self, state: MainWindowState) -> None:
        settings = self._settings(MAIN_WINDOW_ORG, MAIN_WINDOW_APP)
        for key, value in _main_window_state_to_storage_dict(state).items():
            settings.setValue(key, value)
        settings.sync()

    def load_settings_state(self) -> SettingsState:
        settings = self._settings(SETTINGS_ORG, SETTINGS_APP)
        state = _build_settings_state(
            read_bool=lambda key, default: settings.value(key, default, type=bool),
            read_int=lambda key, default: settings.value(key, default, type=int),
            read_float=lambda key, default: settings.value(key, default, type=float),
            read_str=lambda key, default: settings.value(key, default, type=str),
        )
        settings.sync()
        return state

    def save_settings_state(self, state: SettingsState) -> None:
        settings = self._settings(SETTINGS_ORG, SETTINGS_APP)
        for key, value in _settings_state_to_storage_dict(state).items():
            settings.setValue(key, value)
        settings.sync()


class IniStateStore:
    def __init__(self, ini_path: Path | None = None):
        self._ini_path = ini_path or self._resolve_ini_path()
        self._ini_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_ini_path() -> Path:
        explicit_path = os.getenv('NEURALIMAGE_INI_PATH')
        if explicit_path:
            return Path(explicit_path)

        settings_root = os.getenv('NEURALIMAGE_SETTINGS_DIR')
        if settings_root:
            return Path(settings_root) / 'neuralimage_state.ini'

        return Path.cwd() / 'neuralimage_state.ini'

    def _load_parser(self) -> configparser.ConfigParser:
        parser = configparser.ConfigParser()
        if self._ini_path.exists():
            parser.read(self._ini_path, encoding='utf-8')
        return parser

    def _write_parser(self, parser: configparser.ConfigParser) -> None:
        with self._ini_path.open('w', encoding='utf-8') as file:
            parser.write(file)

    @staticmethod
    def _get_bool(parser: configparser.ConfigParser, section: str, key: str, default: bool) -> bool:
        try:
            return parser.getboolean(section, key)
        except (ValueError, configparser.NoOptionError, configparser.NoSectionError):
            return default

    @staticmethod
    def _get_int(parser: configparser.ConfigParser, section: str, key: str, default: int) -> int:
        try:
            return parser.getint(section, key)
        except (ValueError, configparser.NoOptionError, configparser.NoSectionError):
            return default

    @staticmethod
    def _get_float(parser: configparser.ConfigParser, section: str, key: str, default: float) -> float:
        try:
            return parser.getfloat(section, key)
        except (ValueError, configparser.NoOptionError, configparser.NoSectionError):
            return default

    @staticmethod
    def _get_str(parser: configparser.ConfigParser, section: str, key: str, default: str) -> str:
        try:
            return parser.get(section, key)
        except (configparser.NoOptionError, configparser.NoSectionError):
            return default

    def load_main_window_state(self) -> MainWindowState:
        parser = self._load_parser()
        return _build_main_window_state(
            read_str=lambda key, default: self._get_str(parser, INI_SECTION_MAIN, key, default),
            read_int=lambda key, default: self._get_int(parser, INI_SECTION_MAIN, key, default),
        )

    def save_main_window_state(self, state: MainWindowState) -> None:
        parser = self._load_parser()
        parser[INI_SECTION_MAIN] = {
            key: str(value) for key, value in _main_window_state_to_storage_dict(state).items()
        }
        self._write_parser(parser)

    def load_settings_state(self) -> SettingsState:
        parser = self._load_parser()
        return _build_settings_state(
            read_bool=lambda key, default: self._get_bool(parser, INI_SECTION_SETTINGS, key, default),
            read_int=lambda key, default: self._get_int(parser, INI_SECTION_SETTINGS, key, default),
            read_float=lambda key, default: self._get_float(parser, INI_SECTION_SETTINGS, key, default),
            read_str=lambda key, default: self._get_str(parser, INI_SECTION_SETTINGS, key, default),
        )

    def save_settings_state(self, state: SettingsState) -> None:
        parser = self._load_parser()
        parser[INI_SECTION_SETTINGS] = {
            key: str(value) for key, value in _settings_state_to_storage_dict(state).items()
        }
        self._write_parser(parser)


def create_state_store(*, default_backend: str = 'qsettings') -> StateStore:
    backend = os.getenv('NEURALIMAGE_STATE_BACKEND', default_backend).strip().lower()
    if backend == 'ini':
        return IniStateStore()
    return QSettingsStateStore()


def load_main_window_state() -> MainWindowState:
    return create_state_store().load_main_window_state()


def save_main_window_state(state: MainWindowState) -> None:
    create_state_store().save_main_window_state(state)


def load_settings_state() -> SettingsState:
    return create_state_store().load_settings_state()


def save_settings_state(state: SettingsState) -> None:
    create_state_store().save_settings_state(state)
