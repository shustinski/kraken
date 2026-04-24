"""Pipeline JSON persistence service.

Provides pure helpers for reading and writing pipeline configurations as JSON
files, decoupled from :class:`PolygonExtractionWidget`. The widget keeps thin
methods that delegate to these helpers and wire the file-dialog + logging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_pipeline_config_to_path(path: str | Path, config: dict[str, Any]) -> None:
    """Serialise *config* to *path* as pretty-printed JSON (UTF-8, non-ASCII preserved)."""
    Path(path).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def load_pipeline_config_from_path(path: str | Path) -> dict[str, Any]:
    """Read *path* and parse it as a pipeline JSON configuration."""
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload


__all__ = [
    "load_pipeline_config_from_path",
    "save_pipeline_config_to_path",
]
