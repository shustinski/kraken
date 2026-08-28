from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from contour.i18n import active_language, load_ui_texts, normalize_language, operation_name, tr


MOJIBAKE_MARKERS = (
    "Ð",
    "Ñ",
    "Â",
    "Рџ",
    "РЎ",
    "Р‘",
    "РЃ",
    "Р ",
    "Р”",
    "Рќ",
    "Р",
    "вЂ",
    "ВІ",
    "В°",
    "СЊ",
    "СЃ",
)


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, full_key))
        else:
            result[full_key] = value
    return result


def _contour_source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "contour"


def _tr_keys_from_source() -> set[str]:
    keys: set[str] = set()
    for path in _contour_source_root().rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "_tr":
                continue
            if not node.args:
                continue
            key_node = node.args[0]
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                keys.add(key_node.value)
    return keys


def test_contour_defaults_to_russian_ui_language() -> None:
    assert active_language() == "ru"
    assert normalize_language(None) == "ru"


def test_russian_translation_text_is_not_mojibake() -> None:
    assert tr("tab_paths", language="ru") == "Пути"
    assert tr("browse_input_button", language="ru") == "Выбрать вход"


def test_translations_are_loaded_from_language_resources() -> None:
    assert load_ui_texts("ru")["translations"]["tab_paths"] == "Пути"
    assert load_ui_texts("en")["translations"]["tab_paths"] == "Paths"
    assert operation_name("gaussian_blur", language="ru") == "Размытие Гаусса"


def test_language_resources_have_matching_key_sets() -> None:
    english_keys = set(_flatten(load_ui_texts("en")))
    russian_keys = set(_flatten(load_ui_texts("ru")))

    assert english_keys - russian_keys == set()
    assert russian_keys - english_keys == set()


def test_ui_translation_keys_have_resource_entries() -> None:
    translated_keys = set(load_ui_texts("en")["translations"])

    assert _tr_keys_from_source() - translated_keys == set()


def test_contour_ui_texts_do_not_contain_mojibake_markers() -> None:
    checked_files = [
        *_contour_source_root().rglob("*.py"),
        *_contour_source_root().joinpath("resources").glob("ui_texts_*.json"),
    ]

    offenders: list[str] = []
    for path in checked_files:
        text = path.read_text(encoding="utf-8-sig")
        if any(marker in text for marker in MOJIBAKE_MARKERS):
            offenders.append(str(path.relative_to(_contour_source_root().parents[1])))

    assert offenders == []


def test_editor_toolbar_retranslates_to_russian() -> None:
    from contour.widget import PolygonExtractionWidget

    widget = PolygonExtractionWidget()
    try:
        widget.set_ui_language("ru")

        assert widget.trace_width_label.text() == "Ширина"
        assert widget.antialias_opened_cif_button.text() == "Сгладить все векторы"
    finally:
        widget.close()
        widget.deleteLater()


def _has_cyrillic(text: str) -> bool:
    return any("\u0400" <= character <= "\u04ff" for character in text)


def test_contact_and_metal_panels_retranslate_to_english() -> None:
    from contour.widget import PolygonExtractionWidget

    widget = PolygonExtractionWidget()
    try:
        widget.set_ui_language("en")

        assert widget.bright_via_basics_group.title() == "Basic parameters"
        assert widget.bright_via_quality_group.title() == "Quality filters"
        assert widget.via_heuristic_polarity_combo.itemText(0) == "Bright"
        assert widget.metal_show_border_checkbox.text() == "Highlight objects at frame border"
        assert not _has_cyrillic(widget.pick_input_files_button.toolTip())
        assert not _has_cyrillic(widget.browse_input_button.toolTip())
        overlay_index = widget.metal_debug_visual_combo.findData("overlay")
        assert overlay_index >= 0
        assert widget.metal_debug_visual_combo.itemText(overlay_index) == "Final overlay"
        ridge_index = widget.metal_debug_visual_combo.findData("metal_structural_ridge_response")
        assert ridge_index >= 0
        assert not _has_cyrillic(widget.metal_debug_visual_combo.itemText(ridge_index))
        widget.set_ui_language("ru")
        assert widget.bright_via_basics_group.title() == "Основные параметры"
        assert widget.via_heuristic_polarity_combo.itemText(0) == "Светлые"
    finally:
        widget.close()
        widget.deleteLater()


def test_strategy_registry_has_russian_display_names() -> None:
    from contour.ui.metal_strategy_i18n import strategy_description, strategy_name
    from contour.vision.metal_recovery.strategy_registry import STRATEGY_REGISTRY

    for spec in STRATEGY_REGISTRY.values():
        russian_name = strategy_name(spec, "ru")
        russian_description = strategy_description(spec, "ru")
        assert russian_name
        assert _has_cyrillic(russian_description)


def test_menu_clear_vectors_is_catalogued() -> None:
    assert load_ui_texts("ru")["translations"]["menu_clear_vectors"] == "Убрать вектор"
    assert load_ui_texts("en")["translations"]["menu_clear_vectors"] == "Clear vectors"
    assert tr("min_solidity_label", language="ru") == "Мин. заполненность"
