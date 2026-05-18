from __future__ import annotations

from contour.i18n import active_language, load_ui_texts, normalize_language, operation_name, tr


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
