"""Regression tests for the index-based KLayout CIF reader."""

from __future__ import annotations

import textwrap
from pathlib import Path

from contour.infrastructure.cif_klayout_reader import (
    _CifTextStream,
    load_cif_primitives_klayout,
)
from contour.infrastructure.cif_primitives import CifBox, CifComment, CifPolygon


def _write_cif(tmp_path: Path, body: str) -> Path:
    cif_path = tmp_path / "sample.cif"
    cif_path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return cif_path


def test_stream_read_sinteger_bulk(tmp_path: Path) -> None:
    stream = _CifTextStream("  -12  34 ;")
    assert stream.read_sinteger() == -12
    assert stream.read_sinteger() == 34
    assert stream.test_semi()


def test_load_box_and_polygon_commands(tmp_path: Path) -> None:
    cif_path = _write_cif(
        tmp_path,
        """
        DS 1 1 1;
        L NM;
        B 100 200 50 -30;
        P 0 0 10 0 10 10 0 10;
        ( nested ( comment ) ) ;
        DF;
        E
        """,
    )
    result = load_cif_primitives_klayout(cif_path)
    assert len(result.warnings) == 0
    assert any(isinstance(item, CifBox) for item in result.primitives)
    polygon = next(item for item in result.primitives if isinstance(item, CifPolygon))
    assert polygon.points == ((0, 0), (10, 0), (10, 10), (0, 10))
    assert any(isinstance(item, CifComment) for item in result.primitives)


def test_skip_to_semicolon_consumes_terminator(tmp_path: Path) -> None:
    stream = _CifTextStream("abc;def")
    stream.skip_to_semicolon()
    assert stream.peek_char() == "d"
