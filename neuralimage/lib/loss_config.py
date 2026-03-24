from __future__ import annotations

import json
import math
from collections.abc import Mapping
from html import escape
from typing import Any

LOSS_TERM_NAMES: tuple[str, ...] = (
    'bce',
    'dice',
    'iou',
    'focal_bce',
    'boundary',
    'focal_tversky',
    'ce',
)
LOSS_SELECTION_NAMES: tuple[str, ...] = LOSS_TERM_NAMES
LEGACY_COMBINED_LOSS_TERM_NAMES: tuple[str, ...] = (
    'bce_dice',
    'bce_iou',
    'focal_dice',
    'focal_iou',
    'ce_dice',
)
ALL_LOSS_TERM_NAMES: tuple[str, ...] = LOSS_TERM_NAMES + LEGACY_COMBINED_LOSS_TERM_NAMES
DEFAULT_LOSS_TERM_WEIGHTS: dict[str, float] = {'bce': 1.0}
MAX_LOSS_TERM_WEIGHT_SUM = 1.0
LOSS_TERM_DISPLAY_NAMES: dict[str, str] = {
    'bce': 'BCE',
    'dice': 'Dice',
    'bce_dice': 'BCE+Dice',
    'iou': 'IoU',
    'bce_iou': 'BCE+IoU',
    'focal_bce': 'Focal BCE',
    'focal_dice': 'Focal Dice',
    'focal_iou': 'Focal IoU',
    'boundary': 'Boundary',
    'focal_tversky': 'Focal Tversky',
    'ce': 'CE',
    'ce_dice': 'CE+Dice',
}


def normalize_loss_term_name(value: Any) -> str | None:
    normalized = str(value or '').strip().lower()
    if normalized in ALL_LOSS_TERM_NAMES:
        return normalized
    return None


def _coerce_loss_term_weight(raw: Any) -> float | None:
    if raw is None:
        return None

    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None
    return value


def _normalize_mix_weight(raw: Any, *, default: float = 0.5) -> float:
    value = _coerce_loss_term_weight(raw)
    if value is None:
        value = float(default)
    return float(min(max(value, 0.0), 1.0))


def _finalize_loss_term_weight(value: float) -> float:
    return round(float(value), 12)


def loss_function_to_weights(
    value: Any,
    *,
    dice_weight: float = 0.5,
    iou_weight: float = 0.5,
) -> dict[str, float]:
    normalized = normalize_loss_term_name(value)
    if normalized is None:
        return dict(DEFAULT_LOSS_TERM_WEIGHTS)
    if normalized in LOSS_TERM_NAMES:
        return {normalized: 1.0}
    if normalized == 'bce_dice':
        mix_weight = _normalize_mix_weight(dice_weight)
        return sanitize_loss_term_weights({'bce': 1.0 - mix_weight, 'dice': mix_weight})
    if normalized == 'bce_iou':
        mix_weight = _normalize_mix_weight(iou_weight)
        return sanitize_loss_term_weights({'bce': 1.0 - mix_weight, 'iou': mix_weight})
    if normalized == 'focal_dice':
        mix_weight = _normalize_mix_weight(dice_weight)
        return sanitize_loss_term_weights({'focal_bce': 1.0 - mix_weight, 'dice': mix_weight})
    if normalized == 'focal_iou':
        mix_weight = _normalize_mix_weight(iou_weight)
        return sanitize_loss_term_weights({'focal_bce': 1.0 - mix_weight, 'iou': mix_weight})
    if normalized == 'ce_dice':
        mix_weight = _normalize_mix_weight(dice_weight)
        return sanitize_loss_term_weights({'ce': 1.0 - mix_weight, 'dice': mix_weight})
    return dict(DEFAULT_LOSS_TERM_WEIGHTS)


def sanitize_loss_term_weights(weights: Mapping[str, Any] | None) -> dict[str, float]:
    sanitized: dict[str, float] = {}
    if not isinstance(weights, Mapping):
        return sanitized

    for loss_name in ALL_LOSS_TERM_NAMES:
        value = _coerce_loss_term_weight(weights.get(loss_name))
        if value is None:
            continue
        value = float(min(max(value, 0.0), MAX_LOSS_TERM_WEIGHT_SUM))
        value = _finalize_loss_term_weight(value)
        if value <= 0.0:
            continue
        if loss_name in LOSS_TERM_NAMES:
            sanitized[loss_name] = _finalize_loss_term_weight(sanitized.get(loss_name, 0.0) + value)
            continue
        legacy_weights = loss_function_to_weights(loss_name)
        for normalized_name, coefficient in legacy_weights.items():
            sanitized[normalized_name] = _finalize_loss_term_weight(
                sanitized.get(normalized_name, 0.0) + (value * coefficient)
            )

    total = sum(sanitized.values())
    if total > MAX_LOSS_TERM_WEIGHT_SUM and total > 0.0:
        scale = MAX_LOSS_TERM_WEIGHT_SUM / total
        sanitized = {
            loss_name: _finalize_loss_term_weight(value * scale)
            for loss_name, value in sanitized.items()
            if _finalize_loss_term_weight(value * scale) > 0.0
        }
    return sanitized


def deserialize_loss_term_weights(raw: Any) -> dict[str, float]:
    if isinstance(raw, Mapping):
        return sanitize_loss_term_weights(raw)
    text = str(raw or '').strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return sanitize_loss_term_weights(payload if isinstance(payload, Mapping) else None)


def serialize_loss_term_weights(weights: Mapping[str, Any] | None) -> str:
    sanitized = sanitize_loss_term_weights(weights)
    ordered = {loss_name: sanitized[loss_name] for loss_name in LOSS_TERM_NAMES if loss_name in sanitized}
    return json.dumps(ordered, ensure_ascii=True, separators=(',', ':'))


def resolve_loss_term_weights(
    weights: Mapping[str, Any] | None,
    *,
    fallback_loss_function: str = 'bce',
    dice_weight: float = 0.5,
    iou_weight: float = 0.5,
) -> dict[str, float]:
    sanitized = sanitize_loss_term_weights(weights)
    if sanitized:
        return sanitized
    return loss_function_to_weights(
        fallback_loss_function,
        dice_weight=dice_weight,
        iou_weight=iou_weight,
    )


def loss_term_weight_sum(weights: Mapping[str, Any] | None) -> float:
    return float(sum(sanitize_loss_term_weights(weights).values()))


def dominant_loss_function(
    weights: Mapping[str, Any] | None,
    *,
    fallback: str = 'bce',
) -> str:
    sanitized = sanitize_loss_term_weights(weights)
    if not sanitized:
        normalized_fallback = normalize_loss_term_name(fallback)
        if normalized_fallback in LOSS_TERM_NAMES:
            return normalized_fallback
        fallback_weights = loss_function_to_weights(normalized_fallback or fallback)
        if not fallback_weights:
            return 'bce'
        return max(
            LOSS_TERM_NAMES,
            key=lambda loss_name: fallback_weights.get(loss_name, -1.0),
        )
    return max(
        LOSS_TERM_NAMES,
        key=lambda loss_name: sanitized.get(loss_name, -1.0),
    )


def format_loss_formula(weights: Mapping[str, Any] | None) -> str:
    sanitized = sanitize_loss_term_weights(weights)
    if not sanitized:
        return 'Loss = 0'
    expression = ' + '.join(f'{value:.2f}*{loss_name}' for loss_name, value in sanitized.items())
    return f'Loss = {expression}'


def format_loss_formula_html(weights: Mapping[str, Any] | None) -> str:
    sanitized = sanitize_loss_term_weights(weights)
    style = "font-family:'Cambria Math','Times New Roman',serif; font-size:16px;"
    if not sanitized:
        return f'<span style="{style}"><i>L</i> = 0</span>'

    terms: list[str] = []
    for loss_name, value in sanitized.items():
        display_name = escape(LOSS_TERM_DISPLAY_NAMES.get(loss_name, loss_name.upper()))
        terms.append(f'{value:.2f}&middot;<i>L</i><sub>{display_name}</sub>')
    expression = ' + '.join(terms)
    return f'<span style="{style}"><i>L</i> = {expression}</span>'
