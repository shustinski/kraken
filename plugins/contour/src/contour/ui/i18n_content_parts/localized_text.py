"""i18n_content: localized text."""
from __future__ import annotations
from typing import TYPE_CHECKING

def _localized_text(mapping: LocalizedTextMap, key: str, language: str) -> str:
    entry = mapping.get(key, ("", ""))
    return entry[0] if language == "ru" else entry[1]

if TYPE_CHECKING:
    from contour.ui.i18n_content_parts.module_state_parts.module_state import LocalizedTextMap
