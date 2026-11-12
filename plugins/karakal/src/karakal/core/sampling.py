"""Picking frames for labelling: a sample that is useful now and honest later.

A sample is built by rules, applied in order; a frame taken by one rule is not taken again:

- ``worst``: the frames a metric ranks worst (what the validation thinks is wrong);
- ``per_kind``: up to N frames of every kind of finding (or of the kinds named), so that every
  detector gets examples to be checked against ground truth later; a diagnosis can narrow it
  (only ``model_error``, say);
- ``diverse``: frames spread over the run (farthest from each other on the matrix, or by their
  metrics when the matrix places are unknown), so the sample is not 30 neighbours of one defect;
- ``random``: a random part with a seed. Only a random part tells later how often the model is
  really wrong and how much the checks miss: the frames the checks chose say nothing about that.

Frames already labelled or used for training (``exclude``) are never taken.

Every frame of the sample says why it was taken; ``core/sample_export.py`` writes that into the
manifest of the exported sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .domain import BuildResult, FrameRecord
from .findings import Finding, frame_findings
from .metric_keys import metric_value_for_record, rank_records_by_metric_badness

STRATEGY_WORST = "worst"
STRATEGY_PER_KIND = "per_kind"
STRATEGY_DIVERSE = "diverse"
STRATEGY_RANDOM = "random"
STRATEGIES = (STRATEGY_WORST, STRATEGY_PER_KIND, STRATEGY_DIVERSE, STRATEGY_RANDOM)


@dataclass(frozen=True, slots=True)
class SampleRule:
    strategy: str
    count: int
    metric_key: str | None = None
    kinds: tuple[str, ...] = ()
    diagnosis: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "count": self.count,
            "metric_key": self.metric_key,
            "kinds": list(self.kinds),
            "diagnosis": self.diagnosis,
        }


@dataclass(frozen=True, slots=True)
class SamplePlan:
    rules: tuple[SampleRule, ...]
    seed: int = 0
    exclude: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, object]:
        return {"rules": [rule.to_dict() for rule in self.rules], "seed": self.seed, "excluded": len(self.exclude)}


@dataclass(frozen=True, slots=True)
class SampledFrame:
    key: str
    strategy: str
    reason: str
    metric_key: str | None = None
    metric_value: float | None = None
    findings: tuple[Finding, ...] = ()


def default_plan(metric_key: str, *, total: int = 40, seed: int = 0, exclude: Iterable[str] = ()) -> SamplePlan:
    """A balanced plan: 40% worst, 25% per finding kind, 15% spread, 20% random control."""

    worst = max(1, round(total * 0.40))
    per_kind = max(1, round(total * 0.25))
    diverse = max(0, round(total * 0.15))
    random_part = max(1, total - worst - per_kind - diverse)
    return SamplePlan(
        rules=(
            SampleRule(STRATEGY_WORST, worst, metric_key=metric_key),
            SampleRule(STRATEGY_PER_KIND, per_kind),
            SampleRule(STRATEGY_DIVERSE, diverse, metric_key=metric_key),
            SampleRule(STRATEGY_RANDOM, random_part),
        ),
        seed=seed,
        exclude=frozenset(str(key) for key in exclude),
    )


def _worst(records: list[FrameRecord], rule: SampleRule) -> list[SampledFrame]:
    if not rule.metric_key:
        raise ValueError("the 'worst' rule needs a metric")
    taken = []
    for record in rank_records_by_metric_badness(records, rule.metric_key)[: max(0, rule.count)]:
        value = metric_value_for_record(record, rule.metric_key)
        taken.append(
            SampledFrame(
                str(record.key),
                STRATEGY_WORST,
                f"worst by {rule.metric_key}",
                rule.metric_key,
                None if value is None else float(value),
                frame_findings(record.summary),
            )
        )
    return taken


def _per_kind(records: list[FrameRecord], rule: SampleRule) -> list[SampledFrame]:
    by_kind: dict[str, list[tuple[float, FrameRecord]]] = {}
    for record in records:
        found = frame_findings(record.summary)
        best: dict[str, float] = {}
        for item in found:
            if rule.diagnosis and item.diagnosis != rule.diagnosis:
                continue
            if rule.kinds and item.kind not in rule.kinds:
                continue
            best[item.kind] = max(best.get(item.kind, 0.0), item.severity)
        for kind, severity in best.items():
            by_kind.setdefault(kind, []).append((severity, record))
    kinds = list(rule.kinds) if rule.kinds else sorted(by_kind)
    for kind in kinds:
        by_kind.setdefault(kind, []).sort(key=lambda item: (-item[0], str(item[1].key)))
    taken: list[SampledFrame] = []
    used: set[str] = set()
    # Round robin over the kinds: the rarest kind is not starved by the commonest.
    depth = 0
    while len(taken) < rule.count and any(depth < len(by_kind[kind]) for kind in kinds):
        for kind in kinds:
            if len(taken) >= rule.count:
                break
            if depth >= len(by_kind[kind]):
                continue
            severity, record = by_kind[kind][depth]
            if str(record.key) in used:
                continue
            used.add(str(record.key))
            taken.append(
                SampledFrame(
                    str(record.key),
                    STRATEGY_PER_KIND,
                    f"has {kind}" + (f" ({rule.diagnosis})" if rule.diagnosis else ""),
                    None,
                    float(severity),
                    frame_findings(record.summary),
                )
            )
        depth += 1
    return taken


def _place(record: FrameRecord, index: int) -> tuple[float, float] | None:
    identity = record.identity
    if identity is not None and identity.tile_x is not None and identity.tile_y is not None:
        return float(identity.tile_x), float(identity.tile_y)
    return None


def _diverse(records: list[FrameRecord], rule: SampleRule, chosen: list[str], all_records: dict[str, FrameRecord]) -> list[SampledFrame]:
    """Farthest point sampling: every next frame the farthest from all taken so far."""

    if rule.count <= 0 or not records:
        return []
    places = [_place(record, index) for index, record in enumerate(records)]
    if all(place is not None for place in places):
        points = np.asarray(places, dtype=np.float64)
        anchors = [_place(all_records[key], 0) for key in chosen if key in all_records]
        anchor_points = np.asarray([place for place in anchors if place is not None], dtype=np.float64).reshape(-1, 2)
        basis = "matrix place"
    else:
        # Without matrix places: the order of the run and the metric (when there is one).
        order = np.arange(len(records), dtype=np.float64) / max(1, len(records) - 1)
        values = np.zeros(len(records), dtype=np.float64)
        if rule.metric_key:
            raw = [metric_value_for_record(record, rule.metric_key) for record in records]
            finite = [float(value) for value in raw if value is not None and math.isfinite(float(value))]
            if finite:
                low, high = min(finite), max(finite)
                values = np.asarray(
                    [0.0 if value is None else (float(value) - low) / max(1e-9, high - low) for value in raw]
                )
        points = np.stack([order, values], axis=1)
        anchor_points = np.zeros((0, 2))
        basis = "run order and " + (rule.metric_key or "nothing else")
    distance = np.full(len(records), np.inf)
    for anchor in anchor_points:
        distance = np.minimum(distance, np.hypot(*(points - anchor).T))
    taken: list[SampledFrame] = []
    for _step in range(min(rule.count, len(records))):
        index = int(np.argmax(distance)) if np.isfinite(distance).any() else 0
        record = records[index]
        taken.append(
            SampledFrame(str(record.key), STRATEGY_DIVERSE, f"far from the others ({basis})", rule.metric_key, None, frame_findings(record.summary))
        )
        distance = np.minimum(distance, np.hypot(*(points - points[index]).T))
        distance[index] = -1.0
    return taken


def _random(records: list[FrameRecord], rule: SampleRule, rng: np.random.Generator) -> list[SampledFrame]:
    if not records or rule.count <= 0:
        return []
    picks = rng.choice(len(records), size=min(rule.count, len(records)), replace=False)
    return [
        SampledFrame(str(records[int(index)].key), STRATEGY_RANDOM, "random control", None, None, frame_findings(records[int(index)].summary))
        for index in sorted(picks.tolist())
    ]


def build_sample(build_result: BuildResult, plan: SamplePlan) -> tuple[SampledFrame, ...]:
    """The sample of a computed run by the rules of ``plan`` (deterministic for one seed)."""

    rng = np.random.default_rng(int(plan.seed))
    all_records = {str(record.key): record for record in build_result.records}
    chosen: list[str] = []
    result: list[SampledFrame] = []
    for rule in plan.rules:
        if rule.strategy not in STRATEGIES:
            raise ValueError(f"unknown sampling strategy: {rule.strategy}")
        taken_keys = set(chosen)
        available = [
            record
            for record in build_result.records
            if str(record.key) not in plan.exclude and str(record.key) not in taken_keys
        ]
        if rule.strategy == STRATEGY_WORST:
            picked = _worst([record for record in available if record.summary is not None], rule)
        elif rule.strategy == STRATEGY_PER_KIND:
            picked = _per_kind([record for record in available if record.summary is not None], rule)
        elif rule.strategy == STRATEGY_DIVERSE:
            picked = _diverse(available, rule, chosen, all_records)
        else:
            picked = _random(available, rule, rng)
        for frame in picked:
            if frame.key in taken_keys:
                continue
            taken_keys.add(frame.key)
            chosen.append(frame.key)
            result.append(frame)
    # A rule that found fewer frames than asked (few findings of a kind, a small run) leaves its
    # share to chance: the sample keeps its size and the filling is honest.
    wanted = sum(max(0, rule.count) for rule in plan.rules)
    if len(result) < wanted:
        taken_keys = set(chosen)
        available = [
            record
            for record in build_result.records
            if str(record.key) not in plan.exclude and str(record.key) not in taken_keys
        ]
        for frame in _random(available, SampleRule(STRATEGY_RANDOM, wanted - len(result)), rng):
            result.append(
                SampledFrame(frame.key, STRATEGY_RANDOM, "random (filling a rule that found too few)", None, None, frame.findings)
            )
    return tuple(result)


def read_exclusions(path: Path | str | None) -> frozenset[str]:
    """Frame keys from a file, one per line (``#`` starts a comment); a missing file is empty."""

    if path is None:
        return frozenset()
    file = Path(path)
    if not file.is_file():
        return frozenset()
    keys = []
    for line in file.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            keys.append(text)
    return frozenset(keys)


def append_exclusions(path: Path | str, keys: Iterable[str]) -> None:
    """Add the keys of an exported sample to the registry, so the next sample skips them."""

    file = Path(path)
    known = read_exclusions(file)
    new = [str(key) for key in keys if str(key) not in known]
    if not new:
        return
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as handle:
        for key in new:
            handle.write(f"{key}\n")


__all__ = [
    "STRATEGIES",
    "STRATEGY_DIVERSE",
    "STRATEGY_PER_KIND",
    "STRATEGY_RANDOM",
    "STRATEGY_WORST",
    "SamplePlan",
    "SampleRule",
    "SampledFrame",
    "append_exclusions",
    "build_sample",
    "default_plan",
    "read_exclusions",
]
