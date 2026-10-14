"""bright_tophat_dog: draw overlay."""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from contour.vision.io_normalize import to_gray_u8

from contour.vision.via.bright_tophat_dog_parts.module_state import (
    _COLOR_ACCEPT,
    _COLOR_HARD,
    _COLOR_SOFT,
)

def _as_gray_f32(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2 and image.dtype == np.float32:
        return image
    return to_gray_u8(image).astype(np.float32)

def _paint_score_disk(
    image: np.ndarray,
    center: tuple[float, float],
    radius: float,
    value: float,
    scale_max: float,
) -> None:
    if image.size == 0:
        return
    ratio = float(value) * 255.0 / max(1.0, float(scale_max))
    if ratio <= 0.0:
        encoded = 0
    elif ratio >= 255.0:
        encoded = 255
    else:
        encoded = round(ratio)
    cv2.circle(image, (round(center[0]), round(center[1])), max(1, round(radius)), encoded, thickness=-1)

def _normalize_u8(gray: np.ndarray) -> np.ndarray:
    data = to_gray_u8(gray)
    if data.size == 0:
        return data
    min_value = float(data.min())
    max_value = float(data.max())
    if max_value - min_value <= 1e-6:
        return np.zeros_like(data, dtype=np.uint8)
    return np.clip((data.astype(np.float32) - min_value) * (255.0 / (max_value - min_value)), 0, 255).astype(np.uint8)

def _normalize_response(response: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(response.astype(np.float32), copy=False)
    max_value = float(data.max()) if data.size else 0.0
    if max_value <= 1e-6:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.clip(data * (255.0 / max_value), 0, 255).astype(np.uint8)

def _normalize_metal_constraint_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"disabled", "off", "none", "false", "0"}:
        return "disabled"
    if text in {"strict", "hard"}:
        return "strict"
    return "soft"

def _draw_overlay(
    gray: np.ndarray,
    candidates: list[BrightViaDetection],
    cfg: BrightViaDetectorConfig,
    *,
    show_rejected: bool | None = None,
) -> np.ndarray:
    base = cv2.cvtColor(to_gray_u8(gray), cv2.COLOR_GRAY2BGR)
    display_rejected = cfg.show_rejected_candidates if show_rejected is None else bool(show_rejected)
    for det in candidates:
        if not display_rejected and det.status != "accepted":
            continue
        color = _COLOR_ACCEPT
        if det.status == "soft_reject":
            color = _COLOR_SOFT
        elif det.status == "hard_reject":
            color = _COLOR_HARD
        x, y, w, h = det.bbox
        cv2.rectangle(base, (x, y), (x + w, y + h), color, 1)
        cx, cy = round(det.center[0]), round(det.center[1])
        r = max(2, round(0.5 * max(w, h)))
        cv2.circle(base, (cx, cy), r, color, 1)
        if det.status != "accepted" and det.hard_reason:
            cv2.putText(
                base,
                str(det.hard_reason)[:28],
                (x, max(10, y - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    return base
