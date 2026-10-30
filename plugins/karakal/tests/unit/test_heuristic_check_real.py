"""The heuristic check on real SEM frames with gross defects (skipped when the set is not mounted).

G:/test_validation holds three 2000 x 2000 frames: test_00 with a field of craters left of the
pattern, test_01 with a hole torn in the top layer, test_02 with a hair lying across the pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from karakal.core.heuristic_check import GROSS_REASONS, analyze_frame, folder_holds_photos
from karakal.core.image_io import load_grayscale_image

REAL_SET = Path("G:/test_validation")
pytestmark = pytest.mark.skipif(not REAL_SET.is_dir(), reason="real SEM set with gross defects is not mounted")

# Frame, gross codes it must have, where (frame-relative x range) they must lie.
EXPECTED = {
    "test_00.jpg": ("crater", (0.0, 0.2)),
    "test_01.jpg": ("crater", (0.55, 1.0)),
    "test_02.jpg": ("hair", (0.3, 0.85)),
}


def test_the_folder_reads_as_photos() -> None:
    assert folder_holds_photos(REAL_SET)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_gross_defect_is_found_and_nothing_else_is_gross(name: str) -> None:
    code, (x_low, x_high) = EXPECTED[name]

    findings = analyze_frame(load_grayscale_image(REAL_SET / name))
    gross = [reason for reason in findings.reasons if reason.code in GROSS_REASONS]

    assert findings.capture_issues == ()
    assert findings.gross_raw == 1.0
    assert any(reason.code == code for reason in gross), [reason.code for reason in gross]
    # Every gross finding lies on the defect, none on the design around it.
    for reason in gross:
        centre = reason.bbox[0] + reason.bbox[2] / 2.0
        assert x_low <= centre <= x_high, (reason.code, reason.bbox)
