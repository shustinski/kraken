import pytest

pytest.importorskip('PyQt6')

from neuralimage.presenter.main_presenter_parts.rare_patch_editor_preparation import _format_auto_answer_button_text


def test_format_auto_answer_button_text_includes_countdown():
    assert _format_auto_answer_button_text('Нет', 15) == 'Нет (15)'
    assert _format_auto_answer_button_text('Да', 1) == 'Да (1)'
    assert _format_auto_answer_button_text('Нет', 0) == 'Нет'
