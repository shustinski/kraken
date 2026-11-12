"""Edges off the chip's angles are reported, not a crash of the whole run (regression).

``_add_reasons`` was called with ``regions`` twice: any frame with enough edges off 0/45/90 degrees
(a hair across a via layer, a torn hole) raised TypeError and stopped the heuristic check of the
whole run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from karakal.core.heuristic_check import analyze_frame
from karakal.core.synthetic_chip import LayerStyle, make_frame


@pytest.mark.parametrize("kind", ["hair", "hole"])
def test_off_axis_edges_on_a_via_layer_are_a_reason(kind: str) -> None:
    found = False
    for seed in (100, 101, 102):
        findings = analyze_frame(make_frame(kind, seed, LayerStyle(vias=True)).image)
        codes = {reason.code for reason in findings.reasons}
        regions = {region.code for region in findings.regions}
        assert ("off_axis_edges" in codes) == ("off_axis_edges" in regions)
        found = found or "off_axis_edges" in codes
    assert found
