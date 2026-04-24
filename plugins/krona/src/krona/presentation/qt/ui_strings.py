from __future__ import annotations

import json
from pathlib import Path


LANGUAGE_FILES: dict[str, str] = {
    "English": "ui_strings.json",
    "Русский": "ui_strings_ru.json",
}


def available_languages() -> list[str]:
    return list(LANGUAGE_FILES.keys())


def load_ui_strings(language: str = "English", path: Path | None = None) -> dict[str, str]:
    if path is None:
        filename = LANGUAGE_FILES.get(language, LANGUAGE_FILES["English"])
        path = Path(__file__).with_name(filename)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}
