"""layer_management: selection summary."""
from __future__ import annotations
from collections.abc import Mapping

def _selection_summary(selection: object) -> str | None:
    if not isinstance(selection, Mapping):
        return None
    mode = str(selection.get("mode") or selection.get("kind") or "").strip()
    frames = selection.get("frames") or selection.get("frame_ids") or selection.get("items")
    count: int | None = None
    if isinstance(frames, (list, tuple, set)):
        count = len(frames)
    elif "count" in selection:
        try:
            count = int(selection["count"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            count = None
    parts: list[str] = []
    if mode:
        parts.append(mode)
    if count is not None:
        parts.append(f"{count} кадр(ов)")
    return ", ".join(parts) if parts else None
