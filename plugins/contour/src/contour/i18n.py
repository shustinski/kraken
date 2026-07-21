from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

_DEFAULT_LANGUAGE = "ru"
_RESOURCE_PACKAGE = "contour.resources"


def _normalize_contour_language(language: str | None) -> str:
    text = str(language or "").strip().lower()
    if text.startswith("en"):
        return "en"
    return _DEFAULT_LANGUAGE


def normalize_language(language: str | None) -> str:
    return _normalize_contour_language(language)


def active_language(language: str | None = None) -> str:
    return normalize_language(language)


def _texts_resource_name(language: str) -> str:
    return f"ui_texts_{language}.json"


@lru_cache(maxsize=4)
def load_ui_texts(language: str | None = None) -> dict[str, Any]:
    normalized_language = active_language(language)
    resource = resources.files(_RESOURCE_PACKAGE).joinpath(_texts_resource_name(normalized_language))
    if not resource.is_file():
        return {}
    with resource.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def _section(language: str, name: str) -> dict[str, Any]:
    value = load_ui_texts(language).get(name, {})
    return value if isinstance(value, dict) else {}


def _localized_value(section: str, key: str, default: str, language: str | None = None) -> str:
    lang = active_language(language)
    localized = _section(lang, section).get(key)
    if isinstance(localized, str):
        return localized
    english = _section("en", section).get(key)
    if isinstance(english, str):
        return english
    russian = _section(_DEFAULT_LANGUAGE, section).get(key)
    if isinstance(russian, str):
        return russian
    return default or key


def tr(key: str, default: str = "", language: str | None = None, **kwargs: Any) -> str:
    value = _localized_value("translations", key, default, language)
    if kwargs:
        return value.format(**kwargs)
    return value


def operation_name(operation: str, default: str = "", language: str | None = None) -> str:
    return _localized_value("operation_names", operation, default or operation, language)


def parameter_label(name: str, default: str = "", language: str | None = None) -> str:
    return _localized_value("parameter_labels", name, default or name, language)


def choice_label(parameter_name: str, value: str, default: str = "", language: str | None = None) -> str:
    lang = active_language(language)
    for candidate_language in (lang, "en", _DEFAULT_LANGUAGE):
        choices = _section(candidate_language, "choice_labels").get(parameter_name)
        if not isinstance(choices, dict):
            continue
        label = choices.get(value)
        if isinstance(label, str):
            return label
    return default or value
