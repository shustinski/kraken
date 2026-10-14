"""state_store: ini state store."""
from __future__ import annotations
import configparser
import os
from pathlib import Path
from neuralimage.application.dto import MainWindowState, SettingsState

from neuralimage.infrastructure.config.state_store_parts.module_state import (
    INI_SECTION_MAIN,
    INI_SECTION_SETTINGS,
)

from neuralimage.infrastructure.config.state_store_parts.build_main_window_state import (
    _build_main_window_state,
    _main_window_state_to_storage_dict,
)

from neuralimage.infrastructure.config.state_store_parts.build_settings_state import _build_settings_state

from neuralimage.infrastructure.config.state_store_parts.settings_state_storage_dict_2 import _settings_state_to_storage_dict

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
