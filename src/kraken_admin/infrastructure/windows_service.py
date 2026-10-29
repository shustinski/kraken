"""Fail closed before changing a service owned by another installation."""

from __future__ import annotations

import os
from pathlib import Path


def service_image_path() -> str | None:
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Services\KrakenServer",
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as key:
            value, kind = winreg.QueryValueEx(key, "ImagePath")
    except FileNotFoundError:
        return None
    if kind not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) or not isinstance(value, str):
        raise RuntimeError("KrakenServer has an unsupported service ImagePath")
    return os.path.expandvars(value)


def require_owned_service(executable: Path, *, image_path: str | None) -> bool:
    if image_path is None:
        return False
    value = image_path.strip()
    if value.startswith('"'):
        end = value.find('"', 1)
        candidate = value[1:end] if end > 1 else ""
        remainder = value[end + 1 :] if end > 1 else ""
        if remainder and not remainder[0].isspace():
            candidate = ""
    else:
        candidate = value.split(maxsplit=1)[0] if value else ""
    if not candidate or Path(candidate).resolve() != executable.resolve():
        raise RuntimeError("KrakenServer belongs to another executable. Use that installation to manage its service.")
    return True
