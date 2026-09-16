import pytest

from neuralimage.training.experiments import ExperimentRun, paired_bootstrap_delta, rank_topology_first


def _run(name: str, *, breaks: float, dice: float) -> ExperimentRun:
    return ExperimentRun(
        name=name,
        seed=17,
        wire_break_count=breaks,
        false_bridge_count=0,
        topology_violation_count=breaks,
        boundary_f1=0.9,
        boundary_iou=0.8,
        hausdorff_distance=1.0,
        dice=dice,
        iou=0.8,
    )


def test_topology_first_ranking_prefers_fewer_breaks_over_higher_dice():
    high_dice = _run('pixel-first', breaks=3, dice=0.99)
    topology_first = _run('topology-first', breaks=1, dice=0.96)

    assert rank_topology_first([high_dice, topology_first])[0].name == 'topology-first'


def test_paired_bootstrap_delta_is_deterministic():
    result = paired_bootstrap_delta([4, 5, 6], [2, 3, 4], samples=500, seed=11)

    assert result[0] == pytest.approx(-2.0)
    assert result[1:] == pytest.approx((-2.0, -2.0))
