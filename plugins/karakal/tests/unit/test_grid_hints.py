"""Unit checks for optional grid hints."""

from __future__ import annotations


from karakal.core.grid_calibration import select_diverse_calibration_keys
from karakal.core.grid_hints import frame_suspicion
from types import SimpleNamespace


def test_diverse_keys_cover_start_middle_and_end() -> None:
    records = [SimpleNamespace(key=f"f{index}") for index in range(20)]
    keys = select_diverse_calibration_keys(records, limit=8)
    assert keys[0] == "f0"
    assert "f10" in keys or "f9" in keys or "f11" in keys
    assert keys[-1] == "f19" or "f19" in keys
    assert len(keys) <= 8


def test_frame_suspicion_uses_marked_cells_only() -> None:
    cells = (
        SimpleNamespace(score=0.2, reasons=()),
        SimpleNamespace(score=0.8, reasons=("filled_cell",)),
        SimpleNamespace(score=0.4, reasons=("small_artifact",)),
    )
    assert frame_suspicion(cells) == {"max": 0.8, "count": 2.0}
