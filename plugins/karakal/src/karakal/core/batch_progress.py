"""Progress of «Compute and export»: its stages, how far each is, the whole run's share and the
time left. Qt-free.

Every stage has a weight in frame-equivalents (checking the cells of a layer weighs one per frame;
indexing, the bank, the conflicts and the export weigh less), so the whole run's share is the
weighted share of its stages and the time left follows the pace so far. Paused time does not count.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

WAITING = "waiting"
RUNNING = "running"
DONE = "done"

# Weights in frame-equivalents per frame of the stage.
WEIGHT_INDEX = 0.03
WEIGHT_BANK = 0.04
WEIGHT_CELLS = 1.0
WEIGHT_PAIR = 0.35
WEIGHT_EXPORT = 0.2
WEIGHT_PROJECT = 0.02
# The time left is told once this share of the run is done (earlier it is a guess).
ETA_FROM_SHARE = 0.02


@dataclass(slots=True)
class Stage:
    key: str
    title: str
    weight: float
    total: int = 0
    done: int = 0
    state: str = WAITING
    detail: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    paused_for: float = 0.0
    name: str = ""  # the layer or pair of the stage, empty for the run's own stages

    def share(self) -> float:
        if self.state == DONE:
            return 1.0
        if self.state == WAITING or self.total <= 0:
            return 0.0
        return min(1.0, self.done / float(self.total))


@dataclass(slots=True)
class StageView:
    """What the page shows of one stage."""

    key: str
    title: str
    state: str
    done: int
    total: int
    share: float
    seconds: float
    rate: float
    detail: str
    name: str = ""


@dataclass(slots=True)
class ProgressView:
    stages: tuple[StageView, ...]
    share: float
    elapsed: float
    eta: float | None
    paused: bool


@dataclass(slots=True)
class BatchProgress:
    stages: list[Stage] = field(default_factory=list)
    clock: Callable[[], float] = time.monotonic
    started_at: float = 0.0
    paused_since: float = 0.0
    paused_total: float = 0.0

    # Plan --------------------------------------------------------------------------------------

    @classmethod
    def plan(
        cls,
        layers: list[tuple[str, str, int]],
        pairs: list[tuple[str, str, int]],
        *,
        titles: Callable[[str, str], str],
        clock: Callable[[], float] = time.monotonic,
    ) -> "BatchProgress":
        """Stages of a run: ``layers`` (model id, name, frames), ``pairs`` (pair key, name, frames);
        ``titles(kind, name)`` names a stage for people."""

        frames = max((count for _id, _name, count in layers), default=0)
        stages = [Stage("index", titles("index", ""), WEIGHT_INDEX * frames)]
        for model_id, name, count in layers:
            stages.append(Stage(f"bank:{model_id}", titles("bank", name), WEIGHT_BANK * count, name=name))
            stages.append(Stage(f"cells:{model_id}", titles("cells", name), WEIGHT_CELLS * count, total=count, name=name))
        for key, name, count in pairs:
            stages.append(Stage(f"pair:{key}", titles("pair", name), WEIGHT_PAIR * count, total=count, name=name))
        exported = sum(count for _id, _name, count in layers)
        stages.append(Stage("export", titles("export", ""), WEIGHT_EXPORT * exported, total=exported))
        stages.append(Stage("project", titles("project", ""), WEIGHT_PROJECT * frames))
        return cls(stages=stages, clock=clock)

    def stage(self, key: str) -> Stage | None:
        return next((stage for stage in self.stages if stage.key == key), None)

    # Events ------------------------------------------------------------------------------------

    def start(self, key: str, total: int | None = None) -> None:
        now = self.clock()
        if not self.started_at:
            self.started_at = now
        for stage in self.stages:
            if stage.state == RUNNING and stage.key != key:
                self._finish(stage, now)
        stage = self.stage(key)
        if stage is None or stage.state == DONE:
            return
        if stage.state == WAITING:
            stage.state = RUNNING
            stage.started_at = now
        if total is not None:
            stage.total = int(total)

    def update(self, key: str, done: int, total: int | None = None, detail: str = "") -> None:
        stage = self.stage(key)
        if stage is None:
            return
        if stage.state != RUNNING:
            self.start(key, total)
        if total is not None and total > 0:
            stage.total = int(total)
        stage.done = max(0, int(done))
        if detail:
            stage.detail = str(detail)

    def finish(self, key: str) -> None:
        stage = self.stage(key)
        if stage is not None and stage.state != DONE:
            if stage.state == WAITING:
                stage.started_at = self.clock()
            self._finish(stage, self.clock())

    def _finish(self, stage: Stage, now: float) -> None:
        stage.state = DONE
        stage.finished_at = now
        if stage.total:
            stage.done = stage.total
        stage.detail = ""

    def finish_all(self) -> None:
        for stage in self.stages:
            if stage.state != DONE:
                self.finish(stage.key)

    def pause(self) -> None:
        if not self.paused_since:
            self.paused_since = self.clock()

    def resume(self) -> None:
        if self.paused_since:
            pause = self.clock() - self.paused_since
            self.paused_total += pause
            for stage in self.stages:
                if stage.state == RUNNING:
                    stage.paused_for += pause
            self.paused_since = 0.0

    # Reading -----------------------------------------------------------------------------------

    def share(self) -> float:
        total = sum(stage.weight for stage in self.stages)
        if total <= 0:
            return 0.0
        return min(1.0, sum(stage.weight * stage.share() for stage in self.stages) / total)

    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        now = self.paused_since or self.clock()
        return max(0.0, now - self.started_at - self.paused_total)

    def eta(self) -> float | None:
        share = self.share()
        if share < ETA_FROM_SHARE or share >= 1.0:
            return None if share < 1.0 else 0.0
        return self.elapsed() * (1.0 - share) / share

    def _stage_seconds(self, stage: Stage) -> float:
        if stage.state == WAITING or not stage.started_at:
            return 0.0
        end = stage.finished_at if stage.state == DONE else (self.paused_since or self.clock())
        paused = stage.paused_for
        return max(0.0, end - stage.started_at - paused)

    def view(self) -> ProgressView:
        stages = []
        for stage in self.stages:
            seconds = self._stage_seconds(stage)
            rate = stage.done / seconds if stage.state == RUNNING and seconds > 0 and stage.done else 0.0
            stages.append(
                StageView(
                    stage.key,
                    stage.title,
                    stage.state,
                    stage.done,
                    stage.total,
                    stage.share(),
                    seconds,
                    rate,
                    stage.detail,
                    stage.name,
                )
            )
        return ProgressView(tuple(stages), self.share(), self.elapsed(), self.eta(), bool(self.paused_since))


def clock_text(seconds: float | None) -> str:
    """0:42, 7:05, 1:02:10."""

    if seconds is None:
        return ""
    total = int(round(max(0.0, seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
