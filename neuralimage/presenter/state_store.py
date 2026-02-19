import configparser
import os
from pathlib import Path
from typing import Protocol

from PyQt6.QtCore import QSettings

from lib.data_interfaces import normalize_work_mode
from view.window_dataclasses import MainWindowState, SettingsState


MAIN_WINDOW_ORG = 'NeuralImage'
MAIN_WINDOW_APP = 'MainWindow'
SETTINGS_ORG = 'NeuralImage'
SETTINGS_APP = 'Settings'

INI_SECTION_MAIN = 'main_window'
INI_SECTION_SETTINGS = 'settings'


class StateStore(Protocol):
    def load_main_window_state(self) -> MainWindowState:
        ...

    def save_main_window_state(self, state: MainWindowState) -> None:
        ...

    def load_settings_state(self) -> SettingsState:
        ...

    def save_settings_state(self, state: SettingsState) -> None:
        ...


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
    return SettingsState(
        step=read_int('cut_step', defaults.step),
        vertical_rotation=read_bool('vertical_rotation', defaults.vertical_rotation),
        horizontal_rotation=read_bool('horizontal_rotation', defaults.horizontal_rotation),
        additional_augmentation=read_bool('additional_augmentation', defaults.additional_augmentation),
        augmentation_brightness_strength=read_float(
            'augmentation_brightness_strength', defaults.augmentation_brightness_strength
        ),
        augmentation_contrast_strength=read_float('augmentation_contrast_strength', defaults.augmentation_contrast_strength),
        augmentation_noise_probability=read_float(
            'augmentation_noise_probability', defaults.augmentation_noise_probability
        ),
        augmentation_noise_sigma=read_float('augmentation_noise_sigma', defaults.augmentation_noise_sigma),
        sample_size=(
            read_int('sample_x_size', defaults.sample_size[0]),
            read_int('sample_y_size', defaults.sample_size[1]),
        ),
        model=read_str('model', defaults.model),
        color_mode=read_str('color_mode', defaults.color_mode),
        shuffle=defaults.shuffle,
        use_validation=read_bool('validation', defaults.use_validation),
        validation_percent=read_int('validation_percent', defaults.validation_percent),
        sample_cut_mode=read_str('sample_cut_mode', defaults.sample_cut_mode),
        batch_size=read_int('batch_size', defaults.batch_size),
        overlap=read_int('overlap', defaults.overlap),
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
        loss_function=read_str('loss_function', defaults.loss_function),
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
        skip_uniform_labels=read_bool('skip_uniform_labels', defaults.skip_uniform_labels),
        use_multi_gpu=read_bool('use_multi_gpu', defaults.use_multi_gpu),
        torch_compile_enabled=read_bool('torch_compile_enabled', defaults.torch_compile_enabled),
        show_batch_preview=read_bool('show_batch_preview', defaults.show_batch_preview),
    )


def _settings_state_to_storage_dict(state: SettingsState) -> dict[str, str | int | float | bool]:
    return {
        'cut_step': int(state.step),
        'horizontal_rotation': bool(state.horizontal_rotation),
        'vertical_rotation': bool(state.vertical_rotation),
        'additional_augmentation': bool(state.additional_augmentation),
        'augmentation_brightness_strength': float(state.augmentation_brightness_strength),
        'augmentation_contrast_strength': float(state.augmentation_contrast_strength),
        'augmentation_noise_probability': float(state.augmentation_noise_probability),
        'augmentation_noise_sigma': float(state.augmentation_noise_sigma),
        'model': state.model,
        'color_mode': state.color_mode,
        'validation': bool(state.use_validation),
        'validation_percent': int(state.validation_percent),
        'sample_cut_mode': state.sample_cut_mode,
        'batch_size': int(state.batch_size),
        'overlap': int(state.overlap),
        'log_update_frequency': int(state.log_update_frequency),
        'crop_enabled': bool(state.crop_enabled),
        'resize_enabled': bool(state.resize_enabled),
        'additional_processing': bool(state.crop_enabled or state.resize_enabled),
        'edge_cut_size': int(state.edge_cut_size),
        'sample_x_size': int(state.sample_size[0]),
        'sample_y_size': int(state.sample_size[1]),
        'target_x_size': int(state.target_size[0]),
        'target_y_size': int(state.target_size[1]),
        'optimizer_name': state.optimizer_name,
        'mixed_precision': state.mixed_precision,
        'loss_function': state.loss_function,
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
        'skip_uniform_labels': bool(state.skip_uniform_labels),
        'use_multi_gpu': bool(state.use_multi_gpu),
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


def _create_state_store() -> StateStore:
    backend = os.getenv('NEURALIMAGE_STATE_BACKEND', 'qsettings').strip().lower()
    if backend == 'ini':
        return IniStateStore()
    return QSettingsStateStore()


def load_main_window_state() -> MainWindowState:
    return _create_state_store().load_main_window_state()


def save_main_window_state(state: MainWindowState) -> None:
    _create_state_store().save_main_window_state(state)


def load_settings_state() -> SettingsState:
    return _create_state_store().load_settings_state()


def save_settings_state(state: SettingsState) -> None:
    _create_state_store().save_settings_state(state)
