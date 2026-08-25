from __future__ import annotations

import numpy as np

from contour.vision.metal_recovery.marker_consolidation import (
    ConsolidationEvidence,
    consolidate_markers,
)
from contour.vision.metal_recovery.gradient_watershed import GradientWatershedConfig
from contour.vision.metal_recovery.structural_watershed import (
    clamped_structural_watershed_config,
    run_structural_watershed,
    _finalize_instance_labels,
    _geodesic_label_competition,
)


def _horizontal_evidence(shape: tuple[int, int], line_rows: tuple[int, ...]) -> ConsolidationEvidence:
    height, width = shape
    intensity = np.full(shape, 40.0, np.float32)
    ridge_confidence = np.zeros(shape, np.float32)
    ridge_orientation = np.zeros(shape, np.float32)
    structure_orientation = np.full(shape, 0.5 * np.pi, np.float32)
    coherence = np.ones(shape, np.float32)
    persistent_edge = np.zeros(shape, np.float32)
    magnitude = np.zeros(shape, np.float32)
    rim_response = np.zeros(shape, np.float32)
    for row in line_rows:
        intensity[row, :] = 120.0
        ridge_confidence[row, :] = 0.9
        persistent_edge[max(0, row - 3), :] = 40.0
        persistent_edge[min(height - 1, row + 3), :] = 40.0
        rim_response[max(0, row - 3), :] = 80.0
        rim_response[min(height - 1, row + 3), :] = 80.0
    return ConsolidationEvidence(
        intensity=intensity,
        ridge_confidence=ridge_confidence,
        ridge_orientation=ridge_orientation,
        structure_orientation=structure_orientation,
        coherence=coherence,
        persistent_edge=persistent_edge,
        magnitude=magnitude,
        rim_response=rim_response,
        gradient_x=np.zeros(shape, np.float32),
        gradient_y=np.zeros(shape, np.float32),
    )


def test_collinear_ridge_fragments_share_logical_id() -> None:
    ridge = np.zeros((80, 140), np.uint8)
    ridge[40, 10:55] = 255
    ridge[40, 61:120] = 255
    wide = np.zeros_like(ridge)
    result = consolidate_markers(
        ridge,
        wide,
        _horizontal_evidence(ridge.shape, (40,)),
        link_ridge=True,
        link_wide=False,
    )
    left = int(result.ridge_logical_labels[40, 30])
    right = int(result.ridge_logical_labels[40, 90])
    assert left > 0
    assert left == right
    assert result.stats.raw_ridge_count == 2
    assert result.stats.logical_ridge_count == 1
    assert result.stats.accepted_ridge_links == 1


def test_parallel_ridge_fragments_stay_separate() -> None:
    ridge = np.zeros((80, 140), np.uint8)
    ridge[32, 10:120] = 255
    ridge[48, 10:120] = 255
    wide = np.zeros_like(ridge)
    result = consolidate_markers(
        ridge,
        wide,
        _horizontal_evidence(ridge.shape, (32, 48)),
        link_ridge=True,
        link_wide=False,
    )
    top = int(result.ridge_logical_labels[32, 70])
    bottom = int(result.ridge_logical_labels[48, 70])
    assert top > 0
    assert bottom > 0
    assert top != bottom
    assert result.stats.logical_ridge_count == 2
    assert result.stats.accepted_ridge_links == 0


def test_wide_islands_in_one_rim_basin_share_logical_id() -> None:
    wide = np.zeros((120, 160), np.uint8)
    wide[40:55, 40:55] = 255
    wide[70:88, 90:110] = 255
    ridge = np.zeros_like(wide)
    intensity = np.full(wide.shape, 40.0, np.float32)
    intensity[30:100, 30:130] = 70.0
    rim = np.zeros(wide.shape, np.float32)
    rim[28:32, 28:132] = 200.0
    rim[98:102, 28:132] = 200.0
    rim[28:102, 28:32] = 200.0
    rim[28:102, 128:132] = 200.0
    evidence = ConsolidationEvidence(
        intensity=intensity,
        ridge_confidence=np.zeros(wide.shape, np.float32),
        ridge_orientation=np.zeros(wide.shape, np.float32),
        structure_orientation=np.zeros(wide.shape, np.float32),
        coherence=np.ones(wide.shape, np.float32),
        persistent_edge=np.zeros(wide.shape, np.float32),
        magnitude=np.zeros(wide.shape, np.float32),
        rim_response=rim,
        gradient_x=np.zeros(wide.shape, np.float32),
        gradient_y=np.zeros(wide.shape, np.float32),
    )
    result = consolidate_markers(
        ridge,
        wide,
        evidence,
        link_ridge=False,
        link_wide=True,
    )
    first = int(result.wide_logical_labels[45, 45])
    second = int(result.wide_logical_labels[80, 100])
    assert first > 0
    assert first == second
    assert result.stats.raw_wide_count == 2
    assert result.stats.logical_wide_count == 1


def test_wide_plate_absorbs_rim_ridges_instead_of_dropping() -> None:
    wide = np.zeros((100, 140), np.uint8)
    wide[20:80, 20:120] = 255
    ridge = np.zeros_like(wide)
    ridge[20, 20:120] = 255
    ridge[79, 20:120] = 255
    intensity = np.full(wide.shape, 40.0, np.float32)
    intensity[20:80, 20:120] = 70.0
    rim = np.zeros(wide.shape, np.float32)
    rim[18:22, 18:122] = 200.0
    rim[78:82, 18:122] = 200.0
    rim[18:82, 18:22] = 200.0
    rim[18:82, 118:122] = 200.0
    evidence = ConsolidationEvidence(
        intensity=intensity,
        ridge_confidence=np.zeros(wide.shape, np.float32),
        ridge_orientation=np.zeros(wide.shape, np.float32),
        structure_orientation=np.zeros(wide.shape, np.float32),
        coherence=np.ones(wide.shape, np.float32),
        persistent_edge=np.zeros(wide.shape, np.float32),
        magnitude=np.zeros(wide.shape, np.float32),
        rim_response=rim,
        gradient_x=np.zeros(wide.shape, np.float32),
        gradient_y=np.zeros(wide.shape, np.float32),
    )
    result = consolidate_markers(
        ridge,
        wide,
        evidence,
        link_ridge=False,
        link_wide=True,
        separator_aware_combine=True,
    )
    plate = int(result.combined_labels[50, 70])
    assert plate > 0
    assert result.stats.combined_logical_count == 1
    assert int(result.combined_labels[20, 70]) in {0, plate}


def test_labeled_geodesic_keeps_shared_id_across_a_gap() -> None:
    fg_labels = np.zeros((70, 90), np.int32)
    fg_labels[20:28, 10:22] = 1
    fg_labels[20:28, 40:52] = 1
    fg_labels[50:58, 10:22] = 2
    background = np.zeros((70, 90), np.uint8)
    background[:, 0:3] = 255
    background[:, 87:90] = 255
    cost = np.zeros((70, 90), np.float32)
    markers = _geodesic_label_competition(
        np.where(fg_labels > 0, 255, 0).astype(np.uint8),
        background,
        cost,
        foreground_labels=fg_labels,
    )
    labels = _finalize_instance_labels(markers)
    assert int(labels[24, 16]) == int(labels[24, 46])
    assert int(labels[24, 16]) != int(labels[54, 16])


def test_s8_does_not_merge_parallel_traces() -> None:
    image = np.full((160, 220), 40, np.uint8)
    for x0 in (40, 70, 100):
        image[20:140, x0 : x0 + 12] = 80
        image[20:140, x0 : x0 + 3] = 220
        image[20:140, x0 + 9 : x0 + 12] = 220
    result = run_structural_watershed(
        image,
        GradientWatershedConfig(),
        clamped_structural_watershed_config(variant="s8"),
        check_presence=False,
    )
    left = int(result.instance_labels[80, 46])
    mid = int(result.instance_labels[80, 76])
    right = int(result.instance_labels[80, 106])
    assert left > 0
    assert mid > 0
    assert right > 0
    assert len({left, mid, right}) == 3


def test_s13_groups_two_ridges_inside_one_boundary_pair() -> None:
    from contour.vision.metal_recovery.conductor_bands import (
        BandEvidence,
        detect_boundary_pair,
        sample_transverse_profile,
    )

    height, width = 80, 120
    intensity = np.full((height, width), 40.0, np.float32)
    intensity[30:50, 40:80] = 90.0
    ridge = np.zeros((height, width), np.uint8)
    ridge[36, 45:75] = 255
    ridge[44, 45:75] = 255
    rim = np.zeros((height, width), np.float32)
    rim[30, 40:80] = 180.0
    rim[49, 40:80] = 180.0
    magnitude = np.zeros((height, width), np.float32)
    magnitude[30, 40:80] = 80.0
    magnitude[49, 40:80] = 80.0
    persistent = magnitude.copy()
    gy = np.zeros((height, width), np.float32)
    gy[30, 40:80] = 40.0
    gy[49, 40:80] = -40.0
    evidence = BandEvidence(
        intensity=intensity,
        ridge_confidence=(ridge > 0).astype(np.float32),
        ridge_orientation=np.zeros((height, width), np.float32),
        structure_orientation=np.full((height, width), 0.5 * np.pi, np.float32),
        coherence=np.ones((height, width), np.float32),
        persistent_edge=persistent,
        magnitude=magnitude,
        rim_response=rim,
        gradient_x=np.zeros((height, width), np.float32),
        gradient_y=gy,
    )
    profile = sample_transverse_profile(evidence, 40.0, 60.0, 1.0, 0.0, half_extent=16.0)
    pair = detect_boundary_pair(profile)
    assert pair is not None
    assert pair.left_offset < -4.0
    assert pair.right_offset > 4.0
    result = consolidate_markers(
        ridge,
        np.zeros_like(ridge),
        ConsolidationEvidence(
            intensity=intensity,
            ridge_confidence=evidence.ridge_confidence,
            ridge_orientation=evidence.ridge_orientation,
            structure_orientation=evidence.structure_orientation,
            coherence=evidence.coherence,
            persistent_edge=persistent,
            magnitude=magnitude,
            rim_response=rim,
            gradient_x=evidence.gradient_x,
            gradient_y=gy,
        ),
        link_ridge=False,
        link_wide=False,
        group_bands=True,
    )
    top = int(result.ridge_logical_labels[36, 60])
    bottom = int(result.ridge_logical_labels[44, 60])
    assert top > 0
    assert top == bottom


def test_s13_keeps_two_traces_separated_by_a_gap() -> None:
    height, width = 90, 140
    intensity = np.full((height, width), 35.0, np.float32)
    intensity[28:42, 30:110] = 95.0
    intensity[52:66, 30:110] = 95.0
    ridge = np.zeros((height, width), np.uint8)
    ridge[35, 35:105] = 255
    ridge[59, 35:105] = 255
    rim = np.zeros((height, width), np.float32)
    for row in (28, 41, 52, 65):
        rim[row, 30:110] = 180.0
    magnitude = rim * 0.4
    persistent = magnitude.copy()
    gy = np.zeros((height, width), np.float32)
    gy[28, 30:110] = 50.0
    gy[41, 30:110] = -50.0
    gy[52, 30:110] = 50.0
    gy[65, 30:110] = -50.0
    evidence = ConsolidationEvidence(
        intensity=intensity,
        ridge_confidence=(ridge > 0).astype(np.float32),
        ridge_orientation=np.zeros((height, width), np.float32),
        structure_orientation=np.full((height, width), 0.5 * np.pi, np.float32),
        coherence=np.ones((height, width), np.float32),
        persistent_edge=persistent,
        magnitude=magnitude,
        rim_response=rim,
        gradient_x=np.zeros((height, width), np.float32),
        gradient_y=gy,
    )
    result = consolidate_markers(
        ridge,
        np.zeros_like(ridge),
        evidence,
        link_ridge=False,
        link_wide=False,
        group_bands=True,
    )
    top = int(result.ridge_logical_labels[35, 70])
    bottom = int(result.ridge_logical_labels[59, 70])
    assert top > 0
    assert bottom > 0
    assert top != bottom


def test_orientation_aware_veto_ignores_parallel_rims() -> None:
    ridge = np.zeros((80, 160), np.uint8)
    ridge[40, 10:58] = 255
    ridge[40, 66:120] = 255
    intensity = np.full(ridge.shape, 40.0, np.float32)
    intensity[37:44, 10:120] = 90.0
    persistent = np.zeros(ridge.shape, np.float32)
    persistent[37, 10:120] = 80.0
    persistent[43, 10:120] = 80.0
    persistent[40, 58:66] = 80.0
    gy = np.zeros(ridge.shape, np.float32)
    gy[37, 10:120] = 40.0
    gy[43, 10:120] = -40.0
    magnitude = np.abs(gy)
    evidence = ConsolidationEvidence(
        intensity=intensity,
        ridge_confidence=(ridge > 0).astype(np.float32),
        ridge_orientation=np.zeros(ridge.shape, np.float32),
        structure_orientation=np.full(ridge.shape, 0.5 * np.pi, np.float32),
        coherence=np.ones(ridge.shape, np.float32),
        persistent_edge=persistent,
        magnitude=np.maximum(magnitude, 1.0),
        rim_response=persistent,
        gradient_x=np.zeros(ridge.shape, np.float32),
        gradient_y=gy,
    )
    blocked = consolidate_markers(
        ridge,
        np.zeros_like(ridge),
        evidence,
        link_ridge=True,
        link_wide=False,
        orientation_aware_veto=False,
    )
    allowed = consolidate_markers(
        ridge,
        np.zeros_like(ridge),
        evidence,
        link_ridge=True,
        link_wide=False,
        orientation_aware_veto=True,
    )
    assert int(blocked.ridge_logical_labels[40, 30]) != int(blocked.ridge_logical_labels[40, 90])
    assert int(allowed.ridge_logical_labels[40, 30]) == int(allowed.ridge_logical_labels[40, 90])


