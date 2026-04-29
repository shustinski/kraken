from __future__ import annotations

from contour.i18n import active_language, normalize_language, tr


def test_contour_defaults_to_russian_ui_language() -> None:
    assert active_language() == "ru"
    assert normalize_language(None) == "ru"


def test_russian_translation_text_is_not_mojibake() -> None:
    assert tr("tab_paths", language="ru") == "Пути"
    assert tr("browse_input_button", language="ru") == "Выбрать вход"
