from lib.loss_config import (
    deserialize_loss_term_weights,
    format_loss_formula_html,
    sanitize_loss_term_weights,
)


def test_sanitize_loss_term_weights_ignores_none_values():
    weights = sanitize_loss_term_weights({'bce': None, 'dice': 0.6})

    assert weights == {'dice': 0.6}


def test_deserialize_loss_term_weights_ignores_json_null_values():
    weights = deserialize_loss_term_weights('{"bce": null, "dice": 0.6}')

    assert weights == {'dice': 0.6}


def test_format_loss_formula_html_uses_math_like_markup():
    formula_html = format_loss_formula_html({'bce': 0.4, 'dice': 0.6})

    assert '<i>L</i>' in formula_html
    assert '<sub>BCE</sub>' in formula_html
    assert '<sub>Dice</sub>' in formula_html
    assert '&middot;' in formula_html
