from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from lib.runtime_paths import resources_root

_UI_TEXTS_DIR = resources_root()
_DEFAULT_LANGUAGE = 'ru'
_current_language = _DEFAULT_LANGUAGE


def _normalize_language(raw_language: str | None) -> str:
    language = str(raw_language or '').strip().lower()
    if language.startswith('ru'):
        return 'ru'
    if language.startswith('en'):
        return 'en'
    return _DEFAULT_LANGUAGE


def normalize_ui_language(raw_language: str | None) -> str:
    return _normalize_language(raw_language)


def _texts_path_for_language(language: str) -> Path:
    return _UI_TEXTS_DIR / f'ui_texts_{language}.json'


@lru_cache(maxsize=4)
def load_ui_texts(language: str | None = None) -> dict[str, Any]:
    normalized_language = _normalize_language(language)
    ui_texts_path = _texts_path_for_language(normalized_language)
    if not ui_texts_path.exists():
        return {}
    with ui_texts_path.open('r', encoding='utf-8-sig') as file:
        data = json.load(file)
    if not isinstance(data, dict):
        return {}
    return data


def get_ui_language() -> str:
    return _current_language


def set_ui_language(language: str | None) -> str:
    global _current_language
    _current_language = _normalize_language(language)
    return _current_language


def _resolve_value_for_path(path: str, language: str) -> Any:
    node: Any = load_ui_texts(language)
    for part in path.split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


def get_ui_section(section: str, language: str | None = None) -> dict[str, Any]:
    active_language = _normalize_language(language) if language is not None else _current_language
    default_data = load_ui_texts(_DEFAULT_LANGUAGE).get(section, {})
    localized_data = load_ui_texts(active_language).get(section, {})
    if isinstance(default_data, dict):
        if not isinstance(localized_data, dict):
            return dict(default_data)
        return _deep_merge_dicts(default_data, localized_data)
    return localized_data if isinstance(localized_data, dict) else {}


def get_ui_text(path: str, default: str = '', language: str | None = None) -> str:
    active_language = _normalize_language(language) if language is not None else _current_language
    localized_node = _resolve_value_for_path(path, active_language)
    if isinstance(localized_node, str):
        return localized_node
    default_node = _resolve_value_for_path(path, _DEFAULT_LANGUAGE)
    if isinstance(default_node, str):
        return default_node
    return default
