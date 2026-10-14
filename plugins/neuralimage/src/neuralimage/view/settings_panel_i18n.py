from __future__ import annotations
from typing import Any
from neuralimage.configuration import (
    SEM_UI_FIELDS,
    SEM_UI_SECTIONS,
    sem_ui_choice_label,
    sem_ui_field_help,
    sem_ui_field_label,
    sem_ui_section_help,
    sem_ui_section_label,
)
from neuralimage.lib.ui_texts import get_ui_language

from .settings_panel_i18n_parts.module_state import (
    TEXT_FIELD_LOG_UPDATE_FREQUENCY,
    TEXT_FIELD_RECOGNITION_JPEG_QUALITY,
)
from .settings_panel_i18n_parts.apply_description_tooltips import (
    _copy_text_dict,
    _read_first_text,
    _read_text_from_mappings,
    apply_description_tooltips,
)
from .settings_panel_i18n_parts.apply_settings_panel_texts import apply_settings_panel_texts

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    TEXT_FIELD_LOG_UPDATE_FREQUENCY,
    TEXT_FIELD_RECOGNITION_JPEG_QUALITY,
    _copy_text_dict,
    _read_first_text,
    _read_text_from_mappings,
    apply_description_tooltips,
    apply_settings_panel_texts,
)
