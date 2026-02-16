from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_UI_TEXTS_PATH = Path(__file__).resolve().parent.parent / 'resources' / 'ui_texts_ru.json'


@lru_cache(maxsize=1)
def load_ui_texts() -> dict[str, Any]:
    if not _UI_TEXTS_PATH.exists():
        return {}
    with _UI_TEXTS_PATH.open('r', encoding='utf-8-sig') as file:
        data = json.load(file)
    if not isinstance(data, dict):
        return {}
    return data


def get_ui_section(section: str) -> dict[str, Any]:
    data = load_ui_texts().get(section, {})
    return data if isinstance(data, dict) else {}


def get_ui_text(path: str, default: str = '') -> str:
    node: Any = load_ui_texts()
    for part in path.split('.'):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return str(node) if isinstance(node, str) else default

