from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .graphs import GraphBundle
from .model import CircuitModel, EdgeSemantic


@dataclass(frozen=True)
class PhasePathCheck:
    source: str
    target: str
    assignment: dict[str, bool]
    conduction_reachable: bool
    influence_reachable: bool

    @property
    def reachable(self) -> bool:
        return self.conduction_reachable or self.influence_reachable


@dataclass(frozen=True)
class ClockChainAnalysis:
    clock_candidates: list[str]
    inverse_pairs: list[tuple[str, str]]
    switch_controlled_edges: list[dict]
    diagnostics: list[str]

    def as_dict(self) -> dict:
        return {
            "clock_candidates": list(self.clock_candidates),
            "inverse_pairs": [list(pair) for pair in self.inverse_pairs],
            "switch_controlled_edges": list(self.switch_controlled_edges),
            "diagnostics": list(self.diagnostics),
        }


class PhaseAnalyzer:
    """
    Enumerates clock/reset phases and evaluates conditionally-enabled paths.
    """

    def __init__(self, circuit: CircuitModel, graphs: GraphBundle):
        self._circuit = circuit
        self._graphs = graphs

    def analyze_clock_chains(self) -> ClockChainAnalysis:
        control_signals: dict[str, int] = {}
        switch_controlled_edges: list[dict] = []

        for edge in self._graphs.conduction.edges:
            if edge.semantic != EdgeSemantic.CONDUCTION:
                continue
            controls = []
            for literal in edge.controls:
                control_signals[literal.signal] = control_signals.get(literal.signal, 0) + 1
                controls.append({"signal": literal.signal, "required_level": int(literal.required_level)})
            if controls:
                switch_controlled_edges.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "through_devices": list(edge.through_devices),
                        "controls": controls,
                    }
                )

        for edge in self._graphs.influence.edges:
            if edge.semantic == EdgeSemantic.CLOCKING:
                control_signals[edge.source] = control_signals.get(edge.source, 0) + 1

        ranked = sorted(control_signals, key=lambda net: (-self._clock_score(net), -control_signals[net], net))
        inverse_pairs = self._detect_inverse_pairs(set(control_signals))
        diagnostics = []
        if not ranked:
            diagnostics.append("No clock-like control signals detected on switch/control edges.")
        return ClockChainAnalysis(
            clock_candidates=ranked,
            inverse_pairs=inverse_pairs,
            switch_controlled_edges=switch_controlled_edges,
            diagnostics=diagnostics,
        )

    def enumerate_phase_assignments(
        self,
        signals: list[str],
        *,
        fixed: dict[str, bool] | None = None,
        max_combinations: int = 64,
    ) -> list[dict[str, bool]]:
        fixed = dict(fixed or {})
        unique_signals = []
        for signal in signals:
            if signal in fixed:
                continue
            if signal not in unique_signals:
                unique_signals.append(signal)
        if len(unique_signals) > 12:
            # Keep analysis bounded for large clock/reset buses.
            unique_signals = unique_signals[:12]
        assignments: list[dict[str, bool]] = []
        for values in product([False, True], repeat=len(unique_signals)):
            assignment = dict(fixed)
            assignment.update(dict(zip(unique_signals, values)))
            assignments.append(assignment)
            if len(assignments) >= max_combinations:
                break
        return assignments

    def conduction_reachable(
        self,
        source: str,
        target: str,
        assignment: dict[str, bool],
        *,
        allow_unknown_controls: bool = False,
    ) -> bool:
        if source == target:
            return True
        visited = {source}
        stack = [source]
        while stack:
            node = stack.pop()
            for edge in self._graphs.conduction.adjacency.get(node, []):
                state = edge.is_enabled(assignment)
                if state is False:
                    continue
                if state is None and not allow_unknown_controls:
                    continue
                nxt = edge.target
                if nxt == target:
                    return True
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        return False

    def influence_reachable(
        self,
        source: str,
        target: str,
        assignment: dict[str, bool],
        *,
        allow_unknown_controls: bool = False,
        allowed_semantics: set[EdgeSemantic] | None = None,
    ) -> bool:
        if source == target:
            return True
        visited = {source}
        stack = [source]
        while stack:
            node = stack.pop()
            for edge in self._graphs.influence.outgoing.get(node, []):
                if allowed_semantics is not None and edge.semantic not in allowed_semantics:
                    continue
                state = edge.is_enabled(assignment)
                if state is False:
                    continue
                if state is None and not allow_unknown_controls:
                    continue
                nxt = edge.target
                if nxt == target:
                    return True
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        return False

    def transparency_by_phase(
        self,
        data_sources: list[str],
        storage_nodes: list[str],
        control_signals: list[str],
        *,
        fixed: dict[str, bool] | None = None,
        max_combinations: int = 32,
    ) -> dict[str, dict]:
        assignments = self.enumerate_phase_assignments(control_signals, fixed=fixed, max_combinations=max_combinations)
        result: dict[str, dict] = {}
        for assignment in assignments:
            key = self._assignment_key(assignment, control_signals)
            paths: list[dict] = []
            for source in data_sources:
                for target in storage_nodes:
                    cond = self.conduction_reachable(source, target, assignment)
                    infl = self.influence_reachable(source, target, assignment)
                    if cond or infl:
                        paths.append(
                            {
                                "source": source,
                                "target": target,
                                "conduction": cond,
                                "influence": infl,
                            }
                        )
            result[key] = {
                "assignment": {name: int(value) for name, value in assignment.items()},
                "transparent": bool(paths),
                "paths": paths,
            }
        return result

    def hold_break_check(
        self,
        data_sources: list[str],
        storage_nodes: list[str],
        hold_assignments: list[dict[str, bool]],
    ) -> list[dict]:
        checks: list[dict] = []
        for assignment in hold_assignments:
            leak_paths = []
            for source in data_sources:
                for target in storage_nodes:
                    if self.conduction_reachable(source, target, assignment):
                        leak_paths.append({"source": source, "target": target, "path_type": "conduction"})
                    elif self.influence_reachable(source, target, assignment):
                        leak_paths.append({"source": source, "target": target, "path_type": "influence"})
            checks.append(
                {
                    "assignment": {name: int(v) for name, v in assignment.items()},
                    "hold_isolated": len(leak_paths) == 0,
                    "leak_paths": leak_paths,
                }
            )
        return checks

    def _detect_inverse_pairs(self, controls: set[str]) -> list[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()

        # 1) Naming-based complementary clocks (CLK/CLKB, CLK/CLK_N, etc.)
        by_upper = {name.upper(): name for name in controls}
        for name in controls:
            upper = name.upper()
            candidates = self._complement_name_candidates(upper)
            for cand in candidates:
                if cand in by_upper:
                    left = name
                    right = by_upper[cand]
                    if left != right:
                        pairs.add(tuple(sorted((left, right))))

        # 2) Structural inverter relation on control network.
        for edge in self._graphs.influence.edges:
            if edge.inversion != 1:
                continue
            if edge.source in controls and edge.target in controls:
                pairs.add(tuple(sorted((edge.source, edge.target))))

        return sorted(pairs)

    @staticmethod
    def _complement_name_candidates(name_upper: str) -> set[str]:
        candidates = set()
        if name_upper.endswith("B"):
            candidates.add(name_upper[:-1])
        else:
            candidates.add(name_upper + "B")
        if name_upper.endswith("_N"):
            candidates.add(name_upper[:-2])
        else:
            candidates.add(name_upper + "_N")
        if name_upper.endswith("N"):
            candidates.add(name_upper[:-1])
        else:
            candidates.add(name_upper + "N")
        if name_upper.endswith("BAR"):
            candidates.add(name_upper[:-3])
        else:
            candidates.add(name_upper + "BAR")
        return {item for item in candidates if item}

    @staticmethod
    def _clock_score(net: str) -> int:
        upper = net.upper()
        score = 0
        if "CLK" in upper or "CLOCK" in upper:
            score += 20
        if upper in {"CK", "CP"}:
            score += 10
        if upper.endswith("B") or upper.endswith("_N") or upper.endswith("N"):
            score += 2
        return score

    @staticmethod
    def _assignment_key(assignment: dict[str, bool], order: list[str]) -> str:
        ordered = [name for name in order if name in assignment]
        if not ordered:
            ordered = sorted(assignment)
        return ",".join(f"{name}={int(assignment[name])}" for name in ordered)
