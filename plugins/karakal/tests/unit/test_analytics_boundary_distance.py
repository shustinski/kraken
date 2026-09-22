from __future__ import annotations

from karakal.core.analytics import _metric_requires_boundary_distance


def test_metric_requires_boundary_distance_for_overall_and_hausdorff() -> None:
    assert _metric_requires_boundary_distance("overall_frame_score")
    assert _metric_requires_boundary_distance("disagreement_score")
    assert _metric_requires_boundary_distance("model_model_score")
    assert _metric_requires_boundary_distance("hausdorff_distance")
    assert _metric_requires_boundary_distance("hausdorff_score")


def test_metric_skips_boundary_distance_for_dice_iou_bce() -> None:
    assert not _metric_requires_boundary_distance("dice_score")
    assert not _metric_requires_boundary_distance("iou_score")
    assert not _metric_requires_boundary_distance("dice")
    assert not _metric_requires_boundary_distance("iou")
    assert not _metric_requires_boundary_distance("polygon_bce_score")
    assert not _metric_requires_boundary_distance("bce")
