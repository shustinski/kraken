from __future__ import annotations

import json
import math
from collections.abc import Mapping
from html import escape
from typing import Any

LOSS_TERM_NAMES: tuple[str, ...] = (
    'bce',
    'dice',
    'bce_dice',
    'iou',
    'bce_iou',
    'focal_bce',
    'focal_dice',
    'focal_iou',
    'boundary',
    'focal_tversky',
    'ce',
    'ce_dice',
)
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
    if normalized in LOSS_TERM_NAMES:
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


def sanitize_loss_term_weights(weights: Mapping[str, Any] | None) -> dict[str, float]:
    sanitized: dict[str, float] = {}
    if not isinstance(weights, Mapping):
        return sanitized

    for loss_name in LOSS_TERM_NAMES:
        value = _coerce_loss_term_weight(weights.get(loss_name))
        if value is None:
            continue
        value = min(max(value, 0.0), MAX_LOSS_TERM_WEIGHT_SUM)
        if value > 0.0:
            sanitized[loss_name] = value

    total = sum(sanitized.values())
    if total > MAX_LOSS_TERM_WEIGHT_SUM and total > 0.0:
        scale = MAX_LOSS_TERM_WEIGHT_SUM / total
        sanitized = {
            loss_name: (value * scale)
            for loss_name, value in sanitized.items()
            if (value * scale) > 0.0
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
) -> dict[str, float]:
    sanitized = sanitize_loss_term_weights(weights)
    if sanitized:
        return sanitized
    fallback = normalize_loss_term_name(fallback_loss_function) or 'bce'
    return {fallback: 1.0}


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
        return normalized_fallback or 'bce'
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
