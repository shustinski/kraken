"""Decide whether one history event belongs to the block a window asked for."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

_SCOPE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,80}\Z")


def require_scope_id(value: str) -> str:
    token = str(value).strip()
    if _SCOPE_ID.fullmatch(token) is None:
        raise ValueError("Invalid history scope")
    return token


def event_mentions(event: object, identifier: str) -> bool:
    """True when the event stream or payload names this layer, frame, job, or series."""

    token = str(identifier).strip()
    if not token:
        return True
    if isinstance(event, Mapping):
        stream_id = str(event.get("stream_id", ""))
        payload = event.get("payload")
    else:
        stream_id = str(getattr(event, "stream_id", "") or "")
        payload = getattr(event, "payload", None)
    if token in stream_id:
        return True
    return token in _payload_text(payload)


def _payload_text(payload: object) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, default=str, ensure_ascii=False)
    except TypeError:
        return str(payload)
