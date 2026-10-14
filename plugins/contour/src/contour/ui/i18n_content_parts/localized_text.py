"""i18n_content: localized text."""
from __future__ import annotations

def _localized_text(mapping: LocalizedTextMap, key: str, language: str) -> str:
    entry = mapping.get(key, ("", ""))
    return entry[0] if language == "ru" else entry[1]
