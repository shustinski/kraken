"""The heuristic check against synthetic chip photos of many kinds, with known answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from karakal.core.domain import BuildOptions, BuildResult, FrameRecord, ModelSpec
from karakal.core.heuristic_check import (
    CONTAMINATION_REASONS,
    GEOMETRY_REASONS,
    GROSS_REASONS,
    TOPOLOGY_REASONS,
    HeuristicCheckConfig,
    analyze_frame,
    compute_heuristic_check,
)
from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.core.synthetic_chip import (
    DEFECT_EXPECTATIONS,
    LayerStyle,
    PhotoStyle,
    SyntheticChipFrame,
    box_overlaps,
    draw_layer,
    make_frame,
)

STYLES: dict[str, tuple[LayerStyle, PhotoStyle, HeuristicCheckConfig]] = {
    "manhattan": (LayerStyle(), PhotoStyle(), HeuristicCheckConfig()),
    "diagonals": (LayerStyle(diagonals=True), PhotoStyle(), HeuristicCheckConfig(diagonal_routing=True)),
    "mixed_widths": (LayerStyle(mixed_widths=True), PhotoStyle(), HeuristicCheckConfig()),
    "dark_metal": (
        LayerStyle(),
        PhotoStyle(metal_level=60.0, substrate_level=190.0),
        HeuristicCheckConfig(metal_brighter=False),
    ),
    "uneven_light": (LayerStyle(), PhotoStyle(light_gradient=0.5, vignette=0.25), HeuristicCheckConfig()),
    "grain": (LayerStyle(), PhotoStyle(grain=8.0), HeuristicCheckConfig()),
    "rough_edges": (LayerStyle(), PhotoStyle(edge_roughness=1.0), HeuristicCheckConfig()),
    "realistic": (
        LayerStyle(),
        PhotoStyle(light_gradient=0.3, vignette=0.15, grain=5.0, edge_roughness=0.6, noise=5.0),
        HeuristicCheckConfig(),
    ),
}
GROSS_KINDS = ("hair", "craters", "hole")
LOCAL_KINDS = ("open", "neck", "bridge", "particles", "residue", "stain")
FINDING_CODES = set(GROSS_REASONS + TOPOLOGY_REASONS + CONTAMINATION_REASONS + GEOMETRY_REASONS)


def _found(frame: SyntheticChipFrame, findings) -> bool:
    """Whether a finding with an expected code touches a planted defect."""

    return any(
        reason.code in defect.expected_codes
        and (defect.box is None or (reason.bbox is not None and box_overlaps(reason.bbox, defect.box, frame.image.shape)))
        for defect in frame.defects
        for reason in findings.reasons
    )


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("style", list(STYLES))
def test_clean_layer_of_every_style_has_no_findings(style: str, seed: int) -> None:
    layer, photo, config = STYLES[style]
    frame = make_frame("clean", seed, layer, photo)

    findings = analyze_frame(frame.image, config)

    assert findings.capture_issues == ()
    assert {reason.code for reason in findings.reasons} & FINDING_CODES == set()


@pytest.mark.parametrize("kind", LOCAL_KINDS)
@pytest.mark.parametrize("style", list(STYLES))
def test_planted_defect_is_found_where_it_is(style: str, kind: str) -> None:
    layer, photo, config = STYLES[style]
    if style == "uneven_light" and kind == "stain":
        pytest.xfail("a stain in a darkened corner is not rare enough for the dirt gate")
    frame = make_frame(kind, 0, layer, photo)

    findings = analyze_frame(frame.image, config)

    assert findings.capture_issues == ()
    assert _found(frame, findings), [(reason.code, reason.bbox) for reason in findings.reasons]


@pytest.mark.parametrize("kind", GROSS_KINDS)
@pytest.mark.parametrize("style", list(STYLES))
def test_gross_defect_is_found_and_named(style: str, kind: str) -> None:
    layer, photo, config = STYLES[style]
    if style == "dark_metal" and kind == "hole":
        pytest.skip("the synthetic hole draws a bright rim; on dark metal it is not a hole photo")
    if style == "diagonals" and kind == "craters":
        pytest.xfail("with 45-degree routing allowed a round rim breaks into short curved pieces")
    found = 0
    for seed in range(3):
        frame = make_frame(kind, seed, layer, photo)
        findings = analyze_frame(frame.image, config)
        found += int(findings.gross_raw == 1.0 and _found(frame, findings))

    assert found >= 2


@pytest.mark.parametrize("kind", ["particles", "residue", "stain", "neck", "open"])
def test_small_defects_are_not_gross(kind: str) -> None:
    for seed in range(3):
        findings = analyze_frame(make_frame(kind, seed).image)

        assert findings.gross_raw == 0.0, (seed, [reason.code for reason in findings.reasons])


def test_via_layer_vias_are_the_pattern_not_crumbs() -> None:
    for seed in range(3):
        frame = make_frame("clean", seed, LayerStyle(vias=True))
        findings = analyze_frame(frame.image)

        assert findings.extra["via_count"] >= 20
        assert {reason.code for reason in findings.reasons} & FINDING_CODES == set()


def test_particles_beside_vias_are_found() -> None:
    frame = make_frame("particles", 0, LayerStyle(vias=True))

    assert _found(frame, analyze_frame(frame.image))


@pytest.mark.xfail(
    strict=True,
    reason="Sparse metal: a mid-tone film or stain on a via layer moves the local levels; stage 2 (vias).",
)
@pytest.mark.parametrize("kind", ["residue", "stain"])
def test_film_and_stain_on_a_via_layer(kind: str) -> None:
    frame = make_frame(kind, 0, LayerStyle(vias=True))

    assert _found(frame, analyze_frame(frame.image))


def test_dark_metal_with_its_polarity_finds_the_open() -> None:
    layer, photo, config = STYLES["dark_metal"]
    frame = make_frame("open", 0, layer, photo)

    assert _found(frame, analyze_frame(frame.image, config))


def test_layers_have_the_designed_rules() -> None:
    """The generator's own promise: no gap narrower than designed, no track narrower than designed."""

    from karakal.core.mask_primitives import _distance_transform

    for layer in (LayerStyle(), LayerStyle(diagonals=True), LayerStyle(mixed_widths=True)):
        metal = draw_layer(layer, 3).astype(bool)
        gap_distance = _distance_transform(~metal)
        metal_distance = _distance_transform(metal)
        inner = (slice(4, -4), slice(4, -4))
        # Local maxima of the distance are half the local gap (width); none below half the design.
        assert float(np.max(gap_distance[inner])) >= layer.gap / 2.0
        ridge = metal & (metal_distance >= 1.0)
        assert float(np.percentile(2.0 * metal_distance[ridge] - 1.0, 99)) >= layer.width - 1


def _write(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert _grayscale_array_to_qimage(np.ascontiguousarray(array, dtype=np.uint8)).save(str(path))
    return path


def test_realistic_run_gets_the_expected_verdicts(tmp_path: Path) -> None:
    """Uneven light, grain and rough edges at once, a run of clean frames and one of each defect."""

    layer = LayerStyle()
    photo = PhotoStyle(light_gradient=0.3, vignette=0.15, grain=5.0, edge_roughness=0.6, noise=5.0)
    plan = [("clean", seed) for seed in range(12)] + [
        ("hair", 26),
        ("craters", 27),
        ("hole", 28),
        ("open", 20),
        ("bridge", 21),
        ("particles", 22),
        ("residue", 23),
        ("over_etch", 24),
        ("under_etch", 25),
    ]
    folder = tmp_path / "photos"
    records = []
    frames: dict[str, SyntheticChipFrame] = {}
    for index, (kind, seed) in enumerate(plan):
        frame = make_frame(kind, seed, layer, photo)
        key = f"F_{index:04d}"
        frames[key] = frame
        path = _write(folder / f"{key}.png", frame.image)
        records.append(FrameRecord(key, key, first_path=str(path), second_path=str(path), model_mask_paths={"p": str(path)}))
    build_result = BuildResult(
        records=tuple(records), model_specs=(ModelSpec("p", "Photos", folder),), options=BuildOptions()
    )

    result = compute_heuristic_check(build_result)

    verdicts = {record.key: record.summary.heuristic_check.verdict for record in result.records}
    for key, frame in frames.items():
        expected = DEFECT_EXPECTATIONS[frame.kind][1]
        assert verdicts[key] == expected, (key, frame.kind, record_reasons(result, key))


def record_reasons(result: BuildResult, key: str) -> list[str]:
    record = next(record for record in result.records if record.key == key)
    return [reason.code for reason in record.summary.heuristic_check.reasons]
