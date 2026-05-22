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
        assert widget.antialias_opened_cif_button.toolTip() == "Сгладить все открытые CIF"
    finally:
        widget.close()
        widget.deleteLater()
