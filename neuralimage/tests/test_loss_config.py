from lib.loss_config import (
    deserialize_loss_term_weights,
    format_loss_formula_html,
    resolve_loss_term_weights,
    sanitize_loss_term_weights,
)


def test_sanitize_loss_term_weights_ignores_none_values():
    weights = sanitize_loss_term_weights({'bce': None, 'dice': 0.6})

    assert weights == {'dice': 0.6}


def test_deserialize_loss_term_weights_ignores_json_null_values():
    weights = deserialize_loss_term_weights('{"bce": null, "dice": 0.6}')

    assert weights == {'dice': 0.6}


def test_sanitize_loss_term_weights_expands_legacy_combined_terms():
    weights = sanitize_loss_term_weights({'bce_dice': 1.0})

    assert weights == {'bce': 0.5, 'dice': 0.5}


def test_resolve_loss_term_weights_expands_legacy_fallback_with_saved_mix_weights():
    weights = resolve_loss_term_weights(
        {},
        fallback_loss_function='bce_dice',
        dice_weight=0.7,
    )

    assert weights == {'bce': 0.3, 'dice': 0.7}


def test_format_loss_formula_html_uses_math_like_markup():
    formula_html = format_loss_formula_html({'bce': 0.4, 'dice': 0.6})

    assert '<i>L</i>' in formula_html
    assert '<sub>BCE</sub>' in formula_html
    assert '<sub>Dice</sub>' in formula_html
    assert '&middot;' in formula_html
