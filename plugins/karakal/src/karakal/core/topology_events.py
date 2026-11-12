"""Where two model outputs disagree about the objects themselves, not only about pixels.

Pixel overlap (IoU, Dice) hardly moves when a conductor breaks for three pixels or two conductors
touch, yet that is a different circuit. This check matches the objects (8-connected pieces) of a
mask with those of a reference and names every disagreement as a finding (core/findings.py):

- ``open``: the mask splits an object of the reference in two (a broken conductor);
- ``bridge``: the mask joins objects the reference keeps apart (a short); ``merged_points`` for points;
- ``missing_object`` / ``missing_point``: an object of the reference the mask has not;
- ``extra_object`` / ``extra_point``: an object of the mask the reference has not;
- ``shifted_point``: a point found by both, somewhere else;
- ``neck`` / ``bulge``: the mask is clearly narrower / wider than the reference in one place, at
  least three pixels deep (a one-pixel wobble of the edge is not a finding);
- ``boundary_shift``: the edges of the whole frame, or of one tile of it, lie further out or in
  than the reference's (over- or under-segmentation), with the shift in pixels.

Two models: each is compared with the other, so one disagreement shows as two findings (an open
in one is a bridge in the other); which one is right cannot be told, the diagnosis is "unknown".
Three or more: each is compared with the majority of them (the consensus), and a model that
disagrees with the majority is the likely culprit ("model_error").

Masks are read at full resolution: a three-pixel gap does not survive a scaled-down frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .findings import DIAG_MODEL_ERROR, DIAG_UNKNOWN, Finding, make_finding, sort_findings

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

TOPOLOGY_VERSION = "topology.v1"
DETECTOR_TOPOLOGY = "topology"
CONSENSUS_ID = "consensus"

# Kinds that change what is connected to what: they weigh most in the frame's risk.
TOPOLOGY_KINDS = frozenset(
    {"open", "bridge", "merged_points", "missing_object", "missing_point", "extra_object", "extra_point", "shifted_point"}
)
_SEVERITY = {
    "open": 0.9,
    "bridge": 0.9,
    "merged_points": 0.9,
    "missing_object": 0.9,
    "missing_point": 0.9,
    "extra_object": 0.8,
    "extra_point": 0.8,
    "shifted_point": 0.6,
    "neck": 0.6,
    "bulge": 0.5,
}

TOPOLOGY_RISK_METRIC = "topology_risk"
TOPOLOGY_EVENTS_METRIC = "topology_events"


@dataclass(frozen=True, slots=True)
class TopologySettings:
    # Objects smaller than this (px) are matched but never reported missing or extra.
    min_object_area: int = 9
    # Two objects are the same one when they share this many pixels of the smaller one (and at
    # least ``min_overlap_px``).
    overlap_share: float = 0.05
    min_overlap_px: int = 3
    # A neck or bulge is this thick somewhere (distance to its edge, px) and this large.
    # (OpenCV's distance in a three pixel wide band is 1.9; in a two pixel one, an edge wobble, 0.96.)
    local_min_depth: float = 1.5
    local_min_area: int = 6
    local_max_findings: int = 25
    # Tiles across the frame for a local boundary shift, and the shifts (px) worth a finding.
    tiles: int = 8
    frame_shift_px: float = 0.75
    tile_shift_px: float = 1.5
    tile_min_boundary_px: int = 40
    # A missing and an extra point this many diameters apart are one shifted point.
    point_shift_diameters: float = 3.0


@dataclass(slots=True)
class _Objects:
    labels: np.ndarray
    count: int
    stats: np.ndarray
    centroids: np.ndarray

    def area(self, label: int) -> int:
        return int(self.stats[label, cv2.CC_STAT_AREA])

    def box(self, label: int) -> tuple[int, int, int, int]:
        return (
            int(self.stats[label, cv2.CC_STAT_LEFT]),
            int(self.stats[label, cv2.CC_STAT_TOP]),
            int(self.stats[label, cv2.CC_STAT_WIDTH]),
            int(self.stats[label, cv2.CC_STAT_HEIGHT]),
        )


def _as_uint8(mask: np.ndarray) -> np.ndarray:
    """A 0/1 mask as uint8 without a copy when it is a contiguous bool array."""

    if mask.dtype == np.bool_ and mask.flags.c_contiguous:
        return mask.view(np.uint8)
    return mask.astype(np.uint8)


def _objects(mask: np.ndarray) -> _Objects:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(_as_uint8(mask), connectivity=8)
    return _Objects(labels, int(count), stats, centroids)


def _relative(box: tuple[float, float, float, float], shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = shape
    x, y, w, h = box
    return (x / max(1, width), y / max(1, height), w / max(1, width), h / max(1, height))


def _union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _pad(box: tuple[int, int, int, int], margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    x0, y0 = max(0, box[0] - margin), max(0, box[1] - margin)
    x1, y1 = min(width, box[0] + box[2] + margin), min(height, box[1] + box[3] + margin)
    return (x0, y0, x1 - x0, y1 - y0)


def _diameter(objects: _Objects, label: int) -> float:
    return 2.0 * math.sqrt(objects.area(label) / math.pi)


@dataclass(frozen=True, slots=True)
class _Event:
    kind: str
    box: tuple[int, int, int, int] | None
    evidence: tuple[tuple[str, float | str], ...] = ()


@dataclass(slots=True)
class _Difference:
    """Where the mask and the reference differ, and the objects of each that come near it.

    An object no differing pixel comes within one pixel of is the same object in both outputs (it
    is inside the other output, and nothing of the other output touches it), so it can have no
    event and its partner is itself. Everything below therefore looks only at the touched objects,
    each inside its own box: the work follows the disagreements, not the frame.
    """

    flat: np.ndarray  # flat indices of the differing pixels
    touched_mine: list[int]
    touched_theirs: list[int]


def _difference(mask: np.ndarray, reference: np.ndarray, mine: _Objects, theirs: _Objects) -> _Difference | None:
    differ = mask ^ reference
    flat = np.flatnonzero(differ)
    if flat.size == 0:
        return None
    near = np.flatnonzero(cv2.dilate(differ.view(np.uint8), np.ones((3, 3), np.uint8)))
    mine_labels = mine.labels.ravel()[near]
    theirs_labels = theirs.labels.ravel()[near]
    return _Difference(
        flat=flat,
        touched_mine=np.unique(mine_labels[mine_labels > 0]).tolist(),
        touched_theirs=np.unique(theirs_labels[theirs_labels > 0]).tolist(),
    )


def _crop(objects: _Objects, label: int, shape: tuple[int, int], margin: int = 1) -> tuple[int, int, int, int]:
    return _pad(objects.box(label), margin, shape)


def _partners(
    reference: np.ndarray, mine: _Objects, theirs: _Objects, touched_mine: list[int], settings: TopologySettings
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Partners of the touched objects (they only ever pair with touched objects)."""

    partners_of_mine: dict[int, set[int]] = {}
    partners_of_theirs: dict[int, set[int]] = {}
    for own in touched_mine:
        x, y, w, h = mine.box(own)
        inside = mine.labels[y : y + h, x : x + w] == own
        shared = inside & reference[y : y + h, x : x + w]
        if not np.any(shared):
            continue
        counts = np.bincount(theirs.labels[y : y + h, x : x + w][shared])
        for other in np.flatnonzero(counts).tolist():
            if other == 0:
                continue
            smaller = min(mine.area(own), theirs.area(other))
            if int(counts[other]) >= max(settings.min_overlap_px, settings.overlap_share * smaller):
                partners_of_mine.setdefault(own, set()).add(other)
                partners_of_theirs.setdefault(other, set()).add(own)
    return partners_of_mine, partners_of_theirs


def _joining_place(
    owner: np.ndarray, others: _Objects, partners: set[int], box: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Inside ``box``: the pieces of ``owner`` (a window of that box) that touch two or more of ``partners``."""

    x, y, w, h = box
    window = np.ascontiguousarray(owner.astype(np.uint8))
    if not np.any(window):
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(window, connectivity=8)
    other = others.labels[y : y + h, x : x + w]
    places = []
    for label in range(1, count):
        near = cv2.dilate((labels == label).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        touched = set(np.unique(other[near]).tolist()) & partners
        if len(touched) >= 2:
            places.append(
                (
                    x + int(stats[label, cv2.CC_STAT_LEFT]),
                    y + int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                )
            )
    return _union(places) if places else None


def _object_events(
    mask: np.ndarray,
    reference: np.ndarray,
    mine: _Objects,
    theirs: _Objects,
    difference: _Difference,
    *,
    point: bool,
    settings: TopologySettings,
    point_match_radius: float,
) -> tuple[list[_Event], dict[int, set[int]], dict[int, set[int]]]:
    shape = mask.shape
    partners_of_mine, partners_of_theirs = _partners(reference, mine, theirs, difference.touched_mine, settings)
    events: list[_Event] = []
    missing = [
        other
        for other in difference.touched_theirs
        if other not in partners_of_theirs and theirs.area(other) >= settings.min_object_area
    ]
    extra = [
        own
        for own in difference.touched_mine
        if own not in partners_of_mine and mine.area(own) >= settings.min_object_area
    ]
    if point:
        # A point missing here and an extra one close by is one point found elsewhere.
        used_extra: set[int] = set()
        still_missing = []
        for other in missing:
            reach = settings.point_shift_diameters * _diameter(theirs, other)
            cx, cy = theirs.centroids[other]
            best = None
            for own in extra:
                if own in used_extra:
                    continue
                ox, oy = mine.centroids[own]
                distance = math.hypot(ox - cx, oy - cy)
                if distance <= reach and (best is None or distance < best[0]):
                    best = (distance, own)
            if best is None:
                still_missing.append(other)
                continue
            used_extra.add(best[1])
            events.append(
                _Event("shifted_point", _union([theirs.box(other), mine.box(best[1])]), (("shift_px", round(best[0], 2)),))
            )
        missing = still_missing
        extra = [own for own in extra if own not in used_extra]
    for other in missing:
        events.append(_Event("missing_point" if point else "missing_object", theirs.box(other), (("area_px", float(theirs.area(other))),)))
    for own in extra:
        events.append(_Event("extra_point" if point else "extra_object", mine.box(own), (("area_px", float(mine.area(own))),)))
    for other, owns in partners_of_theirs.items():
        if len(owns) < 2:
            continue
        box = _pad(theirs.box(other), 2, shape)
        x, y, w, h = box
        gap = reference[y : y + h, x : x + w] & ~mask[y : y + h, x : x + w]
        place = _joining_place(gap, mine, owns, box)
        events.append(_Event("open", place or theirs.box(other), (("pieces", float(len(owns))),)))
    for own, others in partners_of_mine.items():
        if len(others) < 2:
            continue
        box = _pad(mine.box(own), 2, shape)
        x, y, w, h = box
        strand = mask[y : y + h, x : x + w] & ~reference[y : y + h, x : x + w]
        place = _joining_place(strand, theirs, others, box)
        events.append(_Event("merged_points" if point else "bridge", place or mine.box(own), (("objects", float(len(others))),)))
    if point:
        for own, others in partners_of_mine.items():
            if len(others) != 1:
                continue
            (other,) = tuple(others)
            if len(partners_of_theirs.get(other, ())) != 1:
                continue
            ox, oy = mine.centroids[own]
            cx, cy = theirs.centroids[other]
            distance = math.hypot(ox - cx, oy - cy)
            if distance > max(float(point_match_radius), 0.5 * _diameter(theirs, other)):
                events.append(
                    _Event("shifted_point", _union([theirs.box(other), mine.box(own)]), (("shift_px", round(distance, 2)),))
                )
    return events, partners_of_mine, partners_of_theirs


def _boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(_as_uint8(mask), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask & (eroded == 0)


def _object_boundary_flat(objects: _Objects, label: int, shape: tuple[int, int]) -> np.ndarray:
    """Flat indices (in the frame) of the boundary pixels of one object."""

    x, y, w, h = _crop(objects, label, shape)
    inside = np.pad(objects.labels[y : y + h, x : x + w] == label, 1)
    ys, xs = np.nonzero(_boundary(inside))
    return (ys - 1 + y) * shape[1] + (xs - 1 + x)


def _shift_events(
    mask: np.ndarray,
    reference: np.ndarray,
    mine: _Objects,
    theirs: _Objects,
    difference: _Difference,
    one_to_one_mine: set[int],
    one_to_one_theirs: set[int],
    settings: TopologySettings,
) -> tuple[list[_Event], list[tuple[int, int, int, int]], float]:
    """A frame-wide shift of the edges and tiles shifted differently: events, those tiles, the frame shift.

    Only objects matched one to one count: a missing, extra, split or joined object is its own
    event. Every untouched object is matched one to one (it is itself): the edges of the frame go
    in whole, those of the touched objects that are not one to one come out again.
    """

    height, width = mask.shape
    tiles = max(1, int(settings.tiles))

    def tile_counts(flat: np.ndarray) -> np.ndarray:
        rows, columns = np.divmod(flat, width)
        tile = (rows * tiles // height) * tiles + (columns * tiles // width)
        return np.bincount(tile, minlength=tiles * tiles)

    edge = tile_counts(np.flatnonzero(_boundary(reference)))
    for other in difference.touched_theirs:
        if other not in one_to_one_theirs:
            edge = edge - tile_counts(_object_boundary_flat(theirs, other, mask.shape))
    total_edge = int(edge.sum())
    if total_edge <= 0:
        return [], [], 0.0
    differ = difference.flat
    in_mine = mask.ravel()[differ]
    mine_labels = mine.labels.ravel()[differ[in_mine]]
    theirs_labels = theirs.labels.ravel()[differ[~in_mine]]
    keep_mine = np.isin(mine_labels, np.fromiter(one_to_one_mine, dtype=np.int64, count=len(one_to_one_mine)))
    keep_theirs = np.isin(theirs_labels, np.fromiter(one_to_one_theirs, dtype=np.int64, count=len(one_to_one_theirs)))
    excess = tile_counts(differ[in_mine][keep_mine])
    lack = tile_counts(differ[~in_mine][keep_theirs])
    frame_shift = float(int(excess.sum()) - int(lack.sum())) / total_edge
    events: list[_Event] = []
    if abs(frame_shift) >= settings.frame_shift_px:
        events.append(_Event("boundary_shift", None, (("shift_px", round(frame_shift, 2)),)))
    shifted_tiles: list[tuple[int, int, int, int]] = []
    for index in np.flatnonzero(edge >= settings.tile_min_boundary_px).tolist():
        shift = float(int(excess[index]) - int(lack[index])) / int(edge[index])
        if abs(shift - frame_shift) < settings.tile_shift_px:
            continue
        row, column = divmod(index, tiles)
        # The tile of a pixel is row * tiles // height: tile ``row`` starts at ceil(row * height / tiles).
        y0, y1 = -(-row * height // tiles), -(-(row + 1) * height // tiles)
        x0, x1 = -(-column * width // tiles), -(-(column + 1) * width // tiles)
        box = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        shifted_tiles.append(box)
        events.append(_Event("boundary_shift", box, (("shift_px", round(shift, 2)),)))
    return events, shifted_tiles, frame_shift


def _local_events(
    mask: np.ndarray,
    reference: np.ndarray,
    mine: _Objects,
    theirs: _Objects,
    matched_mine: list[int],
    matched_theirs: list[int],
    skip_tiles: list[tuple[int, int, int, int]],
    settings: TopologySettings,
) -> list[_Event]:
    """Necks and bulges of objects both outputs have: differences at least three pixels deep.

    A neck lies inside one object of the reference, a bulge inside one of the mask (objects never
    touch), so each object is read in its own box.
    """

    shape = mask.shape
    events: list[tuple[float, _Event]] = []
    for kind, owners, other_mask, matched in (
        ("neck", theirs, mask, matched_theirs),
        ("bulge", mine, reference, matched_mine),
    ):
        for owner in matched:
            x0, y0, w0, h0 = _crop(owners, owner, shape)
            inside = owners.labels[y0 : y0 + h0, x0 : x0 + w0] == owner
            difference = np.pad(inside & ~other_mask[y0 : y0 + h0, x0 : x0 + w0], 1).astype(np.uint8)
            if not np.any(difference):
                continue
            depth = cv2.distanceTransform(difference, cv2.DIST_L2, 3)
            if float(depth.max()) < settings.local_min_depth:
                continue
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(difference, connectivity=8)
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < settings.local_min_area:
                    continue
                deepest = float(depth[labels == label].max())
                if deepest < settings.local_min_depth:
                    continue
                x = x0 - 1 + int(stats[label, cv2.CC_STAT_LEFT])
                y = y0 - 1 + int(stats[label, cv2.CC_STAT_TOP])
                w, h = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
                cx, cy = x + w / 2.0, y + h / 2.0
                if any(tx <= cx < tx + tw and ty <= cy < ty + th for tx, ty, tw, th in skip_tiles):
                    continue
                events.append((float(area), _Event(kind, (x, y, w, h), (("depth_px", round(deepest, 2)), ("area_px", float(area))))))
    events.sort(key=lambda item: -item[0])
    return [event for _area, event in events[: max(0, settings.local_max_findings)]]


def compare_to_reference(
    mask: np.ndarray,
    reference: np.ndarray,
    *,
    model_id: str,
    reference_id: str,
    point: bool = False,
    diagnosis: str = DIAG_UNKNOWN,
    point_match_radius: float = 3.0,
    settings: TopologySettings | None = None,
    mine: _Objects | None = None,
    theirs: _Objects | None = None,
) -> tuple[Finding, ...]:
    """Findings about ``mask`` (the output of ``model_id``) against ``reference``.

    ``mine`` / ``theirs``: the objects of the mask / the reference when already known.
    """

    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV is required for topology events")
    settings = settings or TopologySettings()
    mask_bool = np.asarray(mask, dtype=bool)
    reference_bool = np.asarray(reference, dtype=bool)
    if mask_bool.shape != reference_bool.shape:
        raise ValueError(f"mask and reference differ in shape: {mask_bool.shape} != {reference_bool.shape}")
    shape = mask_bool.shape
    mine = mine if mine is not None else _objects(mask_bool)
    theirs = theirs if theirs is not None else _objects(reference_bool)
    difference = _difference(mask_bool, reference_bool, mine, theirs)
    if difference is None:
        return ()
    events, partners_of_mine, partners_of_theirs = _object_events(
        mask_bool,
        reference_bool,
        mine,
        theirs,
        difference,
        point=point,
        settings=settings,
        point_match_radius=point_match_radius,
    )
    # Edges are compared on objects matched one to one: a joined or split object is its own event.
    one_to_one_mine = {
        own
        for own, others in partners_of_mine.items()
        if len(others) == 1 and all(len(partners_of_theirs.get(o, ())) == 1 for o in others)
    }
    one_to_one_theirs = {
        other
        for other, owns in partners_of_theirs.items()
        if len(owns) == 1 and all(len(partners_of_mine.get(o, ())) == 1 for o in owns)
    }
    shift_events, shifted_tiles, _frame_shift = _shift_events(
        mask_bool, reference_bool, mine, theirs, difference, one_to_one_mine, one_to_one_theirs, settings
    )
    events.extend(shift_events)
    if not point:
        matched_mine = sorted(own for own, others in partners_of_mine.items() if len(others) == 1)
        matched_theirs = sorted(other for other, owns in partners_of_theirs.items() if len(owns) == 1)
        events.extend(
            _local_events(mask_bool, reference_bool, mine, theirs, matched_mine, matched_theirs, shifted_tiles, settings)
        )
    findings = []
    for event in events:
        severity = _SEVERITY.get(event.kind)
        if severity is None:  # boundary_shift: as strong as the shift, three pixels and more is full
            shift = abs(float(dict(event.evidence).get("shift_px", 0.0)))
            severity = min(1.0, max(0.2, shift / 3.0))
        evidence = dict(event.evidence)
        evidence["reference"] = reference_id
        findings.append(
            make_finding(
                event.kind,
                DETECTOR_TOPOLOGY,
                severity,
                None if event.box is None else _relative(event.box, shape),
                diagnosis=diagnosis,
                model_id=model_id,
                detector_version=TOPOLOGY_VERSION,
                evidence=evidence,
            )
        )
    return sort_findings(findings)


def topology_findings(
    masks: Mapping[str, np.ndarray],
    *,
    point: bool = False,
    point_match_radius: float = 3.0,
    settings: TopologySettings | None = None,
) -> tuple[Finding, ...]:
    """Findings of every model: against the other one (two models) or the majority (three and more)."""

    items = [(str(model_id), np.asarray(mask, dtype=bool)) for model_id, mask in masks.items() if mask is not None]
    if len(items) < 2:
        return ()
    shape = items[0][1].shape
    items = [(model_id, mask) for model_id, mask in items if mask.shape == shape]
    if len(items) < 2:
        return ()
    findings: list[Finding] = []
    if len(items) == 2:
        (first_id, first), (second_id, second) = items
        found = compare_to_reference(
            first,
            second,
            model_id=first_id,
            reference_id=second_id,
            point=point,
            diagnosis=DIAG_UNKNOWN,
            point_match_radius=point_match_radius,
            settings=settings,
        )
        # The second model against the first is the same comparison seen from the other side.
        findings.extend(found)
        findings.extend(_mirrored(finding, second_id, first_id) for finding in found)
        return sort_findings(findings)
    votes = np.zeros(shape, dtype=np.uint8 if len(items) < 256 else np.uint16)
    for _model_id, mask in items:
        votes += _as_uint8(mask)
    # More than half of the models: votes * 2 > n.
    consensus = votes > len(items) // 2
    consensus_objects = _objects(consensus)
    for model_id, mask in items:
        findings.extend(
            compare_to_reference(
                mask,
                consensus,
                model_id=model_id,
                reference_id=CONSENSUS_ID,
                point=point,
                diagnosis=DIAG_MODEL_ERROR,
                point_match_radius=point_match_radius,
                settings=settings,
                theirs=consensus_objects,
            )
        )
    return sort_findings(findings)


# What a disagreement is called from the other side.
_MIRROR_KIND = {
    "open": "bridge",
    "bridge": "open",
    "merged_points": "open",
    "missing_object": "extra_object",
    "extra_object": "missing_object",
    "missing_point": "extra_point",
    "extra_point": "missing_point",
    "neck": "bulge",
    "bulge": "neck",
}


def _mirrored(finding: Finding, model_id: str, reference_id: str) -> Finding:
    evidence = dict(finding.evidence)
    evidence["reference"] = reference_id
    if "shift_px" in evidence and finding.kind == "boundary_shift":
        evidence["shift_px"] = -float(evidence["shift_px"])
    kind = _MIRROR_KIND.get(finding.kind, finding.kind)
    return make_finding(
        kind,
        finding.detector,
        _SEVERITY.get(kind, finding.severity),
        finding.box,
        diagnosis=finding.diagnosis,
        model_id=model_id,
        detector_version=finding.detector_version,
        evidence=evidence,
        detail=finding.detail,
    )


def topology_metrics(findings: tuple[Finding, ...] | list[Finding], model_count: int) -> dict[str, float]:
    """The frame's topology risk (0-100, higher is worse) and its number of events.

    Two models report every disagreement twice (once from each side): it counts once.
    """

    events = [finding for finding in findings if finding.detector == DETECTOR_TOPOLOGY]
    weight = sum(finding.severity for finding in events)
    count = float(len(events))
    if model_count == 2:
        weight /= 2.0
        count /= 2.0
    return {
        TOPOLOGY_RISK_METRIC: 100.0 * (1.0 - math.exp(-weight / 2.0)),
        TOPOLOGY_EVENTS_METRIC: count,
    }


__all__ = [
    "CONSENSUS_ID",
    "DETECTOR_TOPOLOGY",
    "TOPOLOGY_EVENTS_METRIC",
    "TOPOLOGY_KINDS",
    "TOPOLOGY_RISK_METRIC",
    "TOPOLOGY_VERSION",
    "TopologySettings",
    "compare_to_reference",
    "topology_findings",
    "topology_metrics",
]
