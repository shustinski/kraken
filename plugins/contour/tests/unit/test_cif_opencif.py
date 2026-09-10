from pathlib import Path
from types import SimpleNamespace

import pytest

from contour.infrastructure import cif_opencif
from contour.infrastructure.cif_primitives import CifBox, CifComment, CifPolygon

pytestmark = pytest.mark.fast


def test_native_result_contains_primitives(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def load_cif_file(path: str, continue_on_error: bool) -> dict:
        assert path == str(tmp_path / "frame.cif")
        assert continue_on_error is False
        return {
            "status": "ok",
            "messages": ["parsed"],
            "commands": [
                {"type": "comment", "content": "layer"},
                {"type": "polygon", "points": [(1, 2), (3, 4), (5, 6)]},
                {"type": "box", "width": 10, "height": 20, "center_x": 5, "center_y": 6},
            ],
        }

    monkeypatch.setattr(cif_opencif, "_NATIVE_MODULE", SimpleNamespace(load_cif_file=load_cif_file))
    result = cif_opencif.load_cif_primitives(tmp_path / "frame.cif", continue_on_error=False)
    assert result is not None
    assert result.status == "ok"
    assert result.messages == ("parsed",)
    assert result.primitives == (
        CifComment("layer"),
        CifPolygon(((1, 2), (3, 4), (5, 6))),
        CifBox(10, 20, 5, 6, 1, 0),
    )
