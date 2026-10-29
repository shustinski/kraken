"""Portable Agent progress and plugin checkpoint contract.

Checkpoint files are immutable snapshots below ``checkpoints/`` in staging.
The plugin writes checkpoint.json atomically after finishing those files.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping

from kraken_core.plugin_protocol import safe_relative_path

MAX_STATE_BYTES = 1024 * 1024


def progress_fraction(payload: object, default: float = 0.0) -> float:
    if isinstance(payload, Mapping):
        value = payload.get("fraction", payload.get("progress"))
        if value is None and payload.get("total"):
            try:
                value = float(payload.get("completed", 0)) / float(payload["total"])
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return default
    else:
        value = payload
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    try:
        fraction = float(value)
    except OverflowError:
        return default
    return fraction if math.isfinite(fraction) and 0 <= fraction <= 1 else default


def execution_state(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Execution state must be a JSON object")
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_STATE_BYTES:
        raise ValueError("Execution state exceeds 1 MiB")
    state = json.loads(encoded)
    if "progress" in state and state["progress"] is not None and not isinstance(state["progress"], dict):
        raise ValueError("Progress must be a JSON object")
    checkpoint = state.get("checkpoint")
    if checkpoint is not None:
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("files", []), list):
            raise ValueError("Checkpoint must contain a files list")
        paths = set()
        for item in checkpoint.get("files", []):
            if not isinstance(item, dict):
                raise ValueError("Invalid checkpoint file")
            if not isinstance(item.get("relative_path"), str):
                raise ValueError("Invalid checkpoint path")
            path = safe_relative_path(item["relative_path"])
            if not path.startswith("checkpoints/") or path in paths:
                raise ValueError("Checkpoint files must have unique paths below checkpoints/")
            paths.add(path)
            digest, size = item.get("sha256"), item.get("size_bytes")
            if not isinstance(digest, str) or re.fullmatch("[a-f0-9]{64}", digest) is None:
                raise ValueError("Invalid checkpoint digest")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("Invalid checkpoint size")
    return state
