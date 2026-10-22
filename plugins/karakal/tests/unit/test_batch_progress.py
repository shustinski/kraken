"""Progress of «Compute and export»: stage weights, the run's share, time left, pauses."""

from __future__ import annotations

import pytest

from karakal.core.batch_progress import DONE, RUNNING, WAITING, BatchProgress, clock_text


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _progress(clock: Clock) -> BatchProgress:
    return BatchProgress.plan(
        [("ones", "binary_cif", 100), ("zeros", "binary_cif_inv", 100)],
        [("pair:ones::zeros", "pair", 100)],
        titles=lambda kind, name: f"{kind} {name}".strip(),
        clock=clock,
    )


def test_plan_lists_every_stage_in_order() -> None:
    progress = _progress(Clock())
    assert [stage.key for stage in progress.stages] == [
        "index", "bank:ones", "cells:ones", "bank:zeros", "cells:zeros", "pair:pair:ones::zeros", "export", "project",
    ]
    assert all(stage.state == WAITING for stage in progress.stages)
    # Checking the cells weighs the most.
    weights = {stage.key: stage.weight for stage in progress.stages}
    assert weights["cells:ones"] == max(weights.values())


def test_share_follows_the_weights_and_the_time_left_the_pace() -> None:
    clock = Clock()
    progress = _progress(clock)
    progress.start("index")
    clock.now += 2
    progress.finish("index")
    progress.start("bank:ones")
    clock.now += 3
    progress.update("cells:ones", 50, 100, "TEST_00050")
    view = progress.view()
    states = {stage.key: stage.state for stage in view.stages}
    assert states["index"] == DONE and states["bank:ones"] == DONE and states["cells:ones"] == RUNNING
    assert states["cells:zeros"] == WAITING
    total = sum(stage.weight for stage in progress.stages)
    expected = (progress.stage("index").weight + progress.stage("bank:ones").weight + 50) / total
    assert view.share == pytest.approx(expected)
    assert view.elapsed == pytest.approx(5.0)
    assert view.eta == pytest.approx(5.0 * (1 - expected) / expected)
    current = next(stage for stage in view.stages if stage.key == "cells:ones")
    assert current.detail == "TEST_00050" and current.done == 50


def test_paused_time_does_not_count() -> None:
    clock = Clock()
    progress = _progress(clock)
    progress.start("cells:ones")
    clock.now += 10
    progress.update("cells:ones", 20)
    progress.pause()
    clock.now += 60
    assert progress.view().paused and progress.elapsed() == pytest.approx(10.0)
    progress.resume()
    clock.now += 10
    assert progress.elapsed() == pytest.approx(20.0)
    stage = next(stage for stage in progress.view().stages if stage.key == "cells:ones")
    assert stage.seconds == pytest.approx(20.0)


def test_finish_all_and_clock_text() -> None:
    progress = _progress(Clock())
    progress.finish_all()
    assert progress.share() == 1.0 and progress.eta() == 0.0
    assert clock_text(42) == "0:42" and clock_text(425) == "7:05" and clock_text(3730) == "1:02:10"
    assert clock_text(None) == ""
