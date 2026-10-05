"""Frame filtering rules and frame sets for the grid inspection matrix.

The matrix shows every frame of a run. The user narrows the work down without
recomputing anything: threshold rules on the colour scale drop the worst
frames, single frames can be dropped by hand, and the drop can be inverted so
that only the dropped frames take part. Named sets keep any list of frames for
viewing and export.

Scores here are frame qualities in 0..1, higher is better (the same value the
matrix colours by). A rule drops every frame whose quality is below its cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VIEW_ALL = "all"
VIEW_KEPT = "kept"
VIEW_DROPPED = "dropped"
VIEW_SELECTED = "selected"
USER_SET_PREFIX = "set:"


@dataclass(slots=True)
class FrameFilterRule:
    """Drop frames whose quality is below ``cutoff``."""

    rule_id: int
    position: float
    cutoff: float
    layer_label: str = ""
    enabled: bool = True


@dataclass(slots=True)
class FrameSet:
    """A named list of frame keys kept by the user."""

    set_id: int
    name: str
    keys: tuple[str, ...] = ()


def frames_below(qualities: dict[str, float], cutoff: float) -> set[str]:
    return {key for key, value in qualities.items() if float(value) < float(cutoff)}


@dataclass(slots=True)
class FrameSetModel:
    """Drop rules, manual drops, inversion and user sets of one matrix tab."""

    rules: list[FrameFilterRule] = field(default_factory=list)
    manual_dropped: set[str] = field(default_factory=set)
    manual_returned: set[str] = field(default_factory=set)
    inverted: bool = False
    user_sets: list[FrameSet] = field(default_factory=list)
    view_id: str = VIEW_ALL
    # Colour window of the matrix at the moment of the first drop.
    frozen_window: tuple[float, float] | None = None
    next_id: int = 1

    def _take_id(self) -> int:
        value = self.next_id
        self.next_id += 1
        return value

    # Rules -----------------------------------------------------------------

    def add_rule(self, position: float, cutoff: float, layer_label: str = "") -> FrameFilterRule:
        rule = FrameFilterRule(self._take_id(), float(position), float(cutoff), str(layer_label))
        self.rules.append(rule)
        return rule

    def remove_rule(self, rule_id: int) -> None:
        self.rules = [rule for rule in self.rules if rule.rule_id != int(rule_id)]

    def set_rule_enabled(self, rule_id: int, enabled: bool) -> None:
        for rule in self.rules:
            if rule.rule_id == int(rule_id):
                rule.enabled = bool(enabled)

    def rule_dropped_keys(self, rule: FrameFilterRule, qualities: dict[str, float]) -> set[str]:
        return frames_below(qualities, rule.cutoff)

    def drop_frame(self, key: str) -> None:
        self.manual_returned.discard(str(key))
        self.manual_dropped.add(str(key))

    def return_frame(self, key: str, qualities: dict[str, float]) -> None:
        key = str(key)
        self.manual_dropped.discard(key)
        if key in self._rule_dropped(qualities):
            self.manual_returned.add(key)

    def clear_manual(self) -> None:
        self.manual_dropped.clear()
        self.manual_returned.clear()

    def has_filter(self) -> bool:
        return bool(self.manual_dropped) or any(rule.enabled for rule in self.rules)

    def _rule_dropped(self, qualities: dict[str, float]) -> set[str]:
        dropped: set[str] = set()
        for rule in self.rules:
            if rule.enabled:
                dropped |= frames_below(qualities, rule.cutoff)
        return dropped

    def dropped_keys(self, qualities: dict[str, float]) -> set[str]:
        """Frames removed by enabled rules or by hand (before inversion)."""

        known = set(qualities)
        dropped = (self._rule_dropped(qualities) - self.manual_returned) | (self.manual_dropped & known)
        return dropped

    def kept_keys(self, all_keys, qualities: dict[str, float]) -> set[str]:
        """Frames taking part in histograms, the error list and export."""

        keys = {str(key) for key in all_keys}
        dropped = self.dropped_keys(qualities) & keys
        return dropped if self.inverted else keys - dropped

    # Sets ------------------------------------------------------------------

    def add_set(self, name: str, keys) -> FrameSet:
        frame_set = FrameSet(self._take_id(), str(name), tuple(sorted({str(key) for key in keys})))
        self.user_sets.append(frame_set)
        return frame_set

    def extend_set(self, set_id: int, keys) -> FrameSet | None:
        for index, frame_set in enumerate(self.user_sets):
            if frame_set.set_id == int(set_id):
                merged = tuple(sorted(set(frame_set.keys) | {str(key) for key in keys}))
                self.user_sets[index] = FrameSet(frame_set.set_id, frame_set.name, merged)
                return self.user_sets[index]
        return None

    def remove_set(self, set_id: int) -> None:
        self.user_sets = [frame_set for frame_set in self.user_sets if frame_set.set_id != int(set_id)]
        if self.view_id == f"{USER_SET_PREFIX}{int(set_id)}":
            self.view_id = VIEW_ALL

    def user_set(self, set_id: int) -> FrameSet | None:
        for frame_set in self.user_sets:
            if frame_set.set_id == int(set_id):
                return frame_set
        return None

    def next_set_name(self, base: str) -> str:
        names = {frame_set.name for frame_set in self.user_sets}
        index = 1
        while f"{base} {index}" in names:
            index += 1
        return f"{base} {index}"

    def set_keys(self, view_id: str, all_keys, qualities: dict[str, float], selected_keys=()) -> set[str]:
        """Resolve a view or set id into frame keys."""

        keys = {str(key) for key in all_keys}
        view_id = str(view_id or VIEW_ALL)
        if view_id == VIEW_KEPT:
            return self.kept_keys(keys, qualities)
        if view_id == VIEW_DROPPED:
            dropped = self.dropped_keys(qualities) & keys
            return keys - dropped if self.inverted else dropped
        if view_id == VIEW_SELECTED:
            return {str(key) for key in selected_keys} & keys
        if view_id.startswith(USER_SET_PREFIX):
            try:
                frame_set = self.user_set(int(view_id[len(USER_SET_PREFIX):]))
            except ValueError:
                frame_set = None
            return set(frame_set.keys) & keys if frame_set is not None else set()
        return keys

    def participating_keys(self, all_keys, qualities: dict[str, float], selected_keys=()) -> set[str] | None:
        """Frames of the shown set: drawn at full strength and used by the analysis.

        ``None`` means the whole run is shown and every frame takes part.
        """

        if self.view_id == VIEW_ALL:
            return None
        return self.set_keys(self.view_id, all_keys, qualities, selected_keys)
