import configparser
import os
from pathlib import Path
from typing import Protocol

from PyQt6.QtCore import QSettings

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
        state = MainWindowState(
            work_mode=settings.value('work_mode', 'train_and_recognition'),
            source_folder=settings.value('source_path', '', type=str),
            result_folder=settings.value('result_path', '', type=str),
            model_path=settings.value('model_path', '', type=str),
            label_folder=settings.value('label_path', '', type=str),
            sample_folder=settings.value('sample_path', '', type=str),
            epochs=settings.value('epochs', 40, type=int),
        )
        settings.sync()
        return state

    def save_main_window_state(self, state: MainWindowState) -> None:
        settings = self._settings(MAIN_WINDOW_ORG, MAIN_WINDOW_APP)
        settings.setValue('work_mode', state.work_mode)
        settings.setValue('source_path', state.source_folder)
        settings.setValue('result_path', state.result_folder)
        settings.setValue('model_path', state.model_path)
        settings.setValue('label_path', state.label_folder)
        settings.setValue('sample_path', state.sample_folder)
        settings.setValue('epochs', state.epochs)
        settings.sync()

    def load_settings_state(self) -> SettingsState:
        state = SettingsState()
        settings = self._settings(SETTINGS_ORG, SETTINGS_APP)

        state.step = settings.value('cut_step', 100, type=int)
        state.horizontal_rotation = settings.value('horizontal_rotation', False, type=bool)
        state.vertical_rotation = settings.value('vertical_rotation', False, type=bool)
        state.sample_size = (
            settings.value('sample_x_size', 256, type=int),
            settings.value('sample_y_size', 256, type=int),
        )
        state.model = settings.value('model', '', type=str)
        state.color_mode = settings.value('color_mode', 'RGB', type=str)
        state.use_validation = settings.value('validation', False, type=bool)
        state.validation_percent = settings.value('validation_percent', 20, type=int)
        state.sample_cut_mode = settings.value('sample_cut_mode', 'disk', type=str)
        state.batch_size = settings.value('batch_size', 32, type=int)
        state.overlap = settings.value('overlap', 8, type=int)
        state.additional_processing = settings.value('additional_processing', False, type=bool)
        state.edge_cut_size = settings.value('edge_cut_size', 0, type=int)
        state.target_size = (
            settings.value('target_x_size', 2000, type=int),
            settings.value('target_y_size', 2000, type=int),
        )
        state.optimizer_name = settings.value('optimizer_name', 'adam', type=str)
        state.mixed_precision = settings.value('mixed_precision', 'bf16', type=str)
        state.learning_rate = settings.value('learning_rate', 1e-3, type=float)
        state.weight_decay = settings.value('weight_decay', 0.0, type=float)
        state.early_stopping_enabled = settings.value('early_stopping_enabled', False, type=bool)
        state.early_stopping_patience = settings.value('early_stopping_patience', 10, type=int)
        state.early_stopping_min_delta = settings.value('early_stopping_min_delta', 0.0, type=float)
        state.early_stopping_restore_best_weights = settings.value(
            'early_stopping_restore_best_weights', True, type=bool
        )
        state.warmup_enabled = settings.value('warmup_enabled', False, type=bool)
        state.warmup_epochs = settings.value('warmup_epochs', 3, type=int)
        state.warmup_start_factor = settings.value('warmup_start_factor', 0.1, type=float)
        state.use_multi_gpu = settings.value('use_multi_gpu', True, type=bool)
        state.show_batch_preview = settings.value('show_batch_preview', True, type=bool)
        settings.sync()
        return state

    def save_settings_state(self, state: SettingsState) -> None:
        settings = self._settings(SETTINGS_ORG, SETTINGS_APP)
        settings.setValue('cut_step', state.step)
        settings.setValue('horizontal_rotation', state.horizontal_rotation)
        settings.setValue('vertical_rotation', state.vertical_rotation)
        settings.setValue('model', state.model)
        settings.setValue('color_mode', state.color_mode)
        settings.setValue('validation', state.use_validation)
        settings.setValue('validation_percent', state.validation_percent)
        settings.setValue('sample_cut_mode', state.sample_cut_mode)
        settings.setValue('batch_size', state.batch_size)
        settings.setValue('overlap', state.overlap)
        settings.setValue('additional_processing', state.additional_processing)
        settings.setValue('edge_cut_size', state.edge_cut_size)
        settings.setValue('sample_x_size', state.sample_size[0])
        settings.setValue('sample_y_size', state.sample_size[1])
        settings.setValue('target_x_size', state.target_size[0])
        settings.setValue('target_y_size', state.target_size[1])
        settings.setValue('optimizer_name', state.optimizer_name)
        settings.setValue('mixed_precision', state.mixed_precision)
        settings.setValue('learning_rate', state.learning_rate)
        settings.setValue('weight_decay', state.weight_decay)
        settings.setValue('early_stopping_enabled', state.early_stopping_enabled)
        settings.setValue('early_stopping_patience', state.early_stopping_patience)
        settings.setValue('early_stopping_min_delta', state.early_stopping_min_delta)
        settings.setValue('early_stopping_restore_best_weights', state.early_stopping_restore_best_weights)
        settings.setValue('warmup_enabled', state.warmup_enabled)
        settings.setValue('warmup_epochs', state.warmup_epochs)
        settings.setValue('warmup_start_factor', state.warmup_start_factor)
        settings.setValue('use_multi_gpu', state.use_multi_gpu)
        settings.setValue('show_batch_preview', state.show_batch_preview)
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
        defaults = MainWindowState()
        parser = self._load_parser()
        return MainWindowState(
            work_mode=self._get_str(parser, INI_SECTION_MAIN, 'work_mode', defaults.work_mode),
            source_folder=self._get_str(parser, INI_SECTION_MAIN, 'source_path', defaults.source_folder),
            result_folder=self._get_str(parser, INI_SECTION_MAIN, 'result_path', defaults.result_folder),
            model_path=self._get_str(parser, INI_SECTION_MAIN, 'model_path', defaults.model_path),
            label_folder=self._get_str(parser, INI_SECTION_MAIN, 'label_path', defaults.label_folder),
            sample_folder=self._get_str(parser, INI_SECTION_MAIN, 'sample_path', defaults.sample_folder),
            epochs=self._get_int(parser, INI_SECTION_MAIN, 'epochs', defaults.epochs),
        )

    def save_main_window_state(self, state: MainWindowState) -> None:
        parser = self._load_parser()
        parser[INI_SECTION_MAIN] = {
            'work_mode': state.work_mode,
            'source_path': state.source_folder,
            'result_path': state.result_folder,
            'model_path': state.model_path,
            'label_path': state.label_folder,
            'sample_path': state.sample_folder,
            'epochs': str(state.epochs),
        }
        self._write_parser(parser)

    def load_settings_state(self) -> SettingsState:
        defaults = SettingsState()
        parser = self._load_parser()

        return SettingsState(
            step=self._get_int(parser, INI_SECTION_SETTINGS, 'cut_step', defaults.step),
            vertical_rotation=self._get_bool(parser, INI_SECTION_SETTINGS, 'vertical_rotation', defaults.vertical_rotation),
            horizontal_rotation=self._get_bool(parser, INI_SECTION_SETTINGS, 'horizontal_rotation', defaults.horizontal_rotation),
            sample_size=(
                self._get_int(parser, INI_SECTION_SETTINGS, 'sample_x_size', defaults.sample_size[0]),
                self._get_int(parser, INI_SECTION_SETTINGS, 'sample_y_size', defaults.sample_size[1]),
            ),
            model=self._get_str(parser, INI_SECTION_SETTINGS, 'model', defaults.model),
            color_mode=self._get_str(parser, INI_SECTION_SETTINGS, 'color_mode', defaults.color_mode),
            shuffle=defaults.shuffle,
            use_validation=self._get_bool(parser, INI_SECTION_SETTINGS, 'validation', defaults.use_validation),
            validation_percent=self._get_int(parser, INI_SECTION_SETTINGS, 'validation_percent', defaults.validation_percent),
            sample_cut_mode=self._get_str(parser, INI_SECTION_SETTINGS, 'sample_cut_mode', defaults.sample_cut_mode),
            batch_size=self._get_int(parser, INI_SECTION_SETTINGS, 'batch_size', defaults.batch_size),
            overlap=self._get_int(parser, INI_SECTION_SETTINGS, 'overlap', defaults.overlap),
            additional_processing=self._get_bool(parser, INI_SECTION_SETTINGS, 'additional_processing', defaults.additional_processing),
            edge_cut_size=self._get_int(parser, INI_SECTION_SETTINGS, 'edge_cut_size', defaults.edge_cut_size),
            target_size=(
                self._get_int(parser, INI_SECTION_SETTINGS, 'target_x_size', defaults.target_size[0]),
                self._get_int(parser, INI_SECTION_SETTINGS, 'target_y_size', defaults.target_size[1]),
            ),
            optimizer_name=self._get_str(parser, INI_SECTION_SETTINGS, 'optimizer_name', defaults.optimizer_name),
            mixed_precision=self._get_str(parser, INI_SECTION_SETTINGS, 'mixed_precision', defaults.mixed_precision),
            learning_rate=self._get_float(parser, INI_SECTION_SETTINGS, 'learning_rate', defaults.learning_rate),
            weight_decay=self._get_float(parser, INI_SECTION_SETTINGS, 'weight_decay', defaults.weight_decay),
            early_stopping_enabled=self._get_bool(
                parser, INI_SECTION_SETTINGS, 'early_stopping_enabled', defaults.early_stopping_enabled
            ),
            early_stopping_patience=self._get_int(
                parser, INI_SECTION_SETTINGS, 'early_stopping_patience', defaults.early_stopping_patience
            ),
            early_stopping_min_delta=self._get_float(
                parser, INI_SECTION_SETTINGS, 'early_stopping_min_delta', defaults.early_stopping_min_delta
            ),
            early_stopping_restore_best_weights=self._get_bool(
                parser,
                INI_SECTION_SETTINGS,
                'early_stopping_restore_best_weights',
                defaults.early_stopping_restore_best_weights,
            ),
            warmup_enabled=self._get_bool(parser, INI_SECTION_SETTINGS, 'warmup_enabled', defaults.warmup_enabled),
            warmup_epochs=self._get_int(parser, INI_SECTION_SETTINGS, 'warmup_epochs', defaults.warmup_epochs),
            warmup_start_factor=self._get_float(
                parser, INI_SECTION_SETTINGS, 'warmup_start_factor', defaults.warmup_start_factor
            ),
            use_multi_gpu=self._get_bool(parser, INI_SECTION_SETTINGS, 'use_multi_gpu', defaults.use_multi_gpu),
            show_batch_preview=self._get_bool(parser, INI_SECTION_SETTINGS, 'show_batch_preview', defaults.show_batch_preview),
        )

    def save_settings_state(self, state: SettingsState) -> None:
        parser = self._load_parser()
        parser[INI_SECTION_SETTINGS] = {
            'cut_step': str(state.step),
            'horizontal_rotation': str(state.horizontal_rotation),
            'vertical_rotation': str(state.vertical_rotation),
            'model': state.model,
            'color_mode': state.color_mode,
            'validation': str(state.use_validation),
            'validation_percent': str(state.validation_percent),
            'sample_cut_mode': state.sample_cut_mode,
            'batch_size': str(state.batch_size),
            'overlap': str(state.overlap),
            'additional_processing': str(state.additional_processing),
            'edge_cut_size': str(state.edge_cut_size),
            'sample_x_size': str(state.sample_size[0]),
            'sample_y_size': str(state.sample_size[1]),
            'target_x_size': str(state.target_size[0]),
            'target_y_size': str(state.target_size[1]),
            'optimizer_name': state.optimizer_name,
            'mixed_precision': state.mixed_precision,
            'learning_rate': str(state.learning_rate),
            'weight_decay': str(state.weight_decay),
            'early_stopping_enabled': str(state.early_stopping_enabled),
            'early_stopping_patience': str(state.early_stopping_patience),
            'early_stopping_min_delta': str(state.early_stopping_min_delta),
            'early_stopping_restore_best_weights': str(state.early_stopping_restore_best_weights),
            'warmup_enabled': str(state.warmup_enabled),
            'warmup_epochs': str(state.warmup_epochs),
            'warmup_start_factor': str(state.warmup_start_factor),
            'use_multi_gpu': str(state.use_multi_gpu),
            'show_batch_preview': str(state.show_batch_preview),
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
