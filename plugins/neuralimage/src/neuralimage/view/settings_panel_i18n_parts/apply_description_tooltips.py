"""settings_panel_i18n: apply description tooltips."""
from __future__ import annotations
from typing import Any

def _copy_text_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)

def _read_first_text(mapping: dict[str, Any], keys: tuple[str, ...], default: str = '') -> str:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return str(mapping[key])
    return default

def _read_text_from_mappings(
    mappings: tuple[dict[str, Any], ...],
    keys: tuple[str, ...],
    default: str = '',
) -> str:
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        value = _read_first_text(mapping, keys, '')
        if value:
            return value
    return default

def apply_description_tooltips(
    panel: Any,
    descriptions: dict[str, Any],
    labels_map: dict[str, Any],
) -> None:
    for key, label in panel._desc_labels.items():
        short_text = str(labels_map.get(key, descriptions.get(key, '')))
        detailed_text = str(descriptions.get(key, short_text))
        label.setText(short_text)
        label.setToolTip(detailed_text)
        field = panel._desc_fields.get(key)
        if field is not None:
            panel._apply_tooltip_to_widget_and_children(field, detailed_text)
