from __future__ import annotations

from typing import Any


MAX_LOG_MESSAGES = 500


def should_forward_log_event(topic: str, payload: Any) -> bool:
    normalized_topic = str(topic or "").strip().lower()
    if normalized_topic == "training":
        return False
    if normalized_topic != "logging" or not isinstance(payload, str):
        return True

    message = " ".join(payload.strip().lower().split())
    if not message:
        return True

    redundant_markers = (
        "средняя потеря на обучающей выборке",
        "validation loss:",
        "frame:",
    )
    if any(marker in message for marker in redundant_markers):
        return False
    if message.startswith("frame ") and "per-frame time:" in message:
        return False
    return True
