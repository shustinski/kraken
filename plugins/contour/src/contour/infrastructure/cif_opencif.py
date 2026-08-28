"""Load CIF primitives through LibOpenCIF when the native extension is built."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CifLoadStatus = Literal["ok", "cant_open", "incomplete", "incorrect", "unknown"]

_NATIVE_MODULE = None
_NATIVE_IMPORT_ERROR: str | None = None


from .cif_primitives import CifBox, CifComment, CifPolygon, CifPrimitive
class CifOpenCifLoadResult:
    status: CifLoadStatus
    messages: tuple[str, ...]
    primitives: tuple[CifPrimitive, ...]


def opencif_loader_available() -> bool:
    """Return True when the LibOpenCIF pybind11 extension is importable."""

    _ensure_native_module()
    return _NATIVE_MODULE is not None


def opencif_loader_import_error() -> str | None:
    """Return the import error for the native loader, if any."""

    _ensure_native_module()
    return _NATIVE_IMPORT_ERROR


def opencif_use_enabled() -> bool:
    """Return whether Contour should prefer LibOpenCIF for CIF loading."""

    return str(os.environ.get("CONTOUR_CIF_USE_OPENCIF", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def load_cif_primitives(path: str | Path, *, continue_on_error: bool = True) -> CifOpenCifLoadResult | None:
    """Parse a CIF file with LibOpenCIF, or return None when the extension is unavailable."""

    native = _ensure_native_module()
    if native is None:
        return None

    source_path = Path(path)
    payload = _load_native_cif_file(native, source_path, continue_on_error)
    status = str(payload.get("status", "unknown"))
    messages = tuple(str(item) for item in payload.get("messages", ()))
    primitives: list[CifPrimitive] = []
    for command in payload.get("commands", ()):
        command_type = str(command.get("type", ""))
        if command_type == "comment":
            primitives.append(CifComment(content=str(command.get("content", ""))))
            continue
        if command_type == "polygon":
            raw_points = command.get("points", ())
            points = tuple((int(x_coord), int(y_coord)) for x_coord, y_coord in raw_points)
            primitives.append(CifPolygon(points=points))
            continue
        if command_type == "box":
            primitives.append(
                CifBox(
                    width=int(command.get("width", 0)),
                    height=int(command.get("height", 0)),
                    center_x=int(command.get("center_x", 0)),
                    center_y=int(command.get("center_y", 0)),
                    rotation_x=int(command.get("rotation_x", 1)),
                    rotation_y=int(command.get("rotation_y", 0)),
                )
            )
    return CifOpenCifLoadResult(
        status=status,  # type: ignore[arg-type]
        messages=messages,
        primitives=tuple(primitives),
    )


def _ensure_native_module():
    global _NATIVE_MODULE, _NATIVE_IMPORT_ERROR
    if _NATIVE_MODULE is not None or _NATIVE_IMPORT_ERROR is not None:
        return _NATIVE_MODULE
    try:
        from contour._native import cif_loader as native_module
    except ImportError as exc:
        _NATIVE_IMPORT_ERROR = str(exc)
        return None
    _NATIVE_MODULE = native_module
    return _NATIVE_MODULE


def _load_native_cif_file(native, source_path: Path, continue_on_error: bool) -> dict[str, object]:
    """Load through LibOpenCIF, copying to an ASCII temp path when needed on Windows."""

    payload = native.load_cif_file(os.fspath(source_path), bool(continue_on_error))
    if payload.get("status") != "cant_open" or not source_path.is_file():
        return payload

    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as handle:
        handle.write(source_path.read_bytes())
        temp_path = handle.name
    try:
        return native.load_cif_file(temp_path, bool(continue_on_error))
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
