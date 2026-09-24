"""Request journal for Kraken Server.

Levels accumulate: medium includes low, high includes every request.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

LOG_LEVELS = ("none", "low", "medium", "high")
_RANK = {name: index for index, name in enumerate(LOG_LEVELS)}
LOGGER = logging.getLogger("kraken_server.access")
_HANDLER_NAME = "kraken-server-operation"

_PROJECT_CREATE = "/api/v1/projects"
_ACCOUNT_CREATE = frozenset({"/api/v1/auth/accounts"})
_LAYER_CREATE = re.compile(r"/api/v1/projects/[^/]+/layers\Z")


@dataclass
class _Threshold:
    rank: int = 0


_threshold = _Threshold()


def parse_log_level(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _RANK:
        allowed = ", ".join(LOG_LEVELS)
        raise ValueError(f"server.log_level must be one of: {allowed}")
    return normalized


def configure_operation_log(level_name: str) -> str:
    """Apply the journal level. ``none`` records nothing."""
    level = parse_log_level(level_name)
    _threshold.rank = _RANK[level]
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.disabled = level == "none"
    if not any(getattr(handler, "name", "") == _HANDLER_NAME for handler in LOGGER.handlers):
        handler = logging.StreamHandler()
        handler.name = _HANDLER_NAME
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        LOGGER.addHandler(handler)
    return level


def request_rank(method: str, path: str) -> int:
    """Minimum level rank that records this call. High-only calls rank as 3."""
    normalized = path.split("?", 1)[0].rstrip("/") or "/"
    verb = method.upper()
    if verb == "POST" and (normalized == _PROJECT_CREATE or normalized in _ACCOUNT_CREATE):
        return _RANK["low"]
    if verb == "POST" and _LAYER_CREATE.fullmatch(normalized):
        return _RANK["medium"]
    if verb in {"POST", "PUT"} and _is_data_upload(normalized):
        return _RANK["medium"]
    return _RANK["high"]


def record_http_request(method: str, path: str, status_code: int) -> None:
    required = request_rank(method, path)
    if _threshold.rank < required:
        return
    if _threshold.rank < _RANK["high"] and status_code >= 400:
        return
    LOGGER.info("%s %s %s %s", _label(method, path), method.upper(), path.split("?", 1)[0], status_code)


def _is_data_upload(path: str) -> bool:
    if "/uploads" in path or "/outputs/" in path:
        return True
    return "/artifacts/" in path and path.endswith("/versions")


def _label(method: str, path: str) -> str:
    rank = request_rank(method, path)
    normalized = path.split("?", 1)[0].rstrip("/") or "/"
    if rank == _RANK["low"]:
        if normalized == _PROJECT_CREATE:
            return "создание проекта"
        return "создание учётной записи"
    if rank == _RANK["medium"]:
        if _LAYER_CREATE.fullmatch(normalized):
            return "создание слоя"
        return "загрузка данных"
    return "обращение"
