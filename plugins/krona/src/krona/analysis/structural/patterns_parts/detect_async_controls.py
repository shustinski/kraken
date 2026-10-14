"""patterns: detect async controls."""
from __future__ import annotations
from itertools import combinations, product
from krona.analysis.structural.model import (
    ConditionalEdge,
    EdgeSemantic,
    LogicLevel,
)
from krona.analysis.structural.phases import ClockChainAnalysis

from krona.analysis.structural.patterns_parts.storage_phase_profile import (
    AsyncControlCandidate,
    InputMuxCandidate,
)


class DetectAsyncControlsMixin:
    def _has_positive_feedback(self, nodes: tuple[str, ...], edges: list[ConditionalEdge]) -> bool:
        parity_edges = [edge for edge in edges if edge.inversion in {0, 1}]
        if not parity_edges:
            return False
        adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
        for node in nodes:
            adjacency[(node, 0)] = set()
            adjacency[(node, 1)] = set()
        for edge in parity_edges:
            inv = int(edge.inversion or 0)
            for parity in (0, 1):
                adjacency[(edge.source, parity)].add((edge.target, parity ^ inv))

        for start in nodes:
            start_state = (start, 0)
            visited = {start_state}
            stack = [start_state]
            while stack:
                state = stack.pop()
                for nxt in adjacency.get(state, set()):
                    if nxt == start_state:
                        return True
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    stack.append(nxt)
        return False

    def _stable_states(self, nodes: tuple[str, ...], edges: list[ConditionalEdge]) -> list[dict[str, int]]:
        if len(nodes) > 6:
            return []
        effective_edges = [edge for edge in edges if not edge.controls and edge.inversion in {0, 1}]
        if not effective_edges:
            return []
        stable: list[dict[str, int]] = []
        for bits in product([0, 1], repeat=len(nodes)):
            assign = dict(zip(nodes, bits))
            if all((assign[edge.source] ^ int(edge.inversion or 0)) == assign[edge.target] for edge in effective_edges):
                stable.append({node: int(assign[node]) for node in nodes})
        return stable

    def _cross_coupled_inverters(self, nodes: tuple[str, ...], edges: list[ConditionalEdge]) -> list[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for a, b in combinations(nodes, 2):
            has_ab = any(edge.source == a and edge.target == b and edge.inversion == 1 for edge in edges)
            has_ba = any(edge.source == b and edge.target == a and edge.inversion == 1 for edge in edges)
            if has_ab and has_ba:
                pairs.add((a, b))
        return sorted(pairs)

    def _has_outgoing_to_noncore(self, node: str, core_nodes: tuple[str, ...]) -> bool:
        core = set(core_nodes)
        return any(edge.source == node and edge.target not in core for edge in self._graphs.influence.outgoing.get(node, []))

    def _detect_input_muxes(
        self,
        cores: list[BistableCoreCandidate],
        clock_analysis: ClockChainAnalysis,
    ) -> list[InputMuxCandidate]:
        muxes: list[InputMuxCandidate] = []
        clock_set = set(clock_analysis.clock_candidates)
        for core in cores:
            core_set = set(core.nodes)
            for storage_node in core.nodes:
                incoming = [
                    edge
                    for edge in core.incoming_edges
                    if edge.target == storage_node and edge.semantic == EdgeSemantic.INFLUENCE
                ]
                incoming += [
                    edge
                    for edge in self._graphs.conduction.edges
                    if edge.target == storage_node and edge.source not in self._power and edge.source not in self._ground
                ]
                if len(incoming) < 2:
                    continue
                by_select: dict[str, dict[bool, list[ConditionalEdge]]] = {}
                for edge in incoming:
                    if len(edge.controls) != 1:
                        continue
                    literal = edge.controls[0]
                    by_select.setdefault(literal.signal, {False: [], True: []})[literal.required_level].append(edge)
                for select_signal, branches in by_select.items():
                    if not branches[False] or not branches[True]:
                        continue
                    edges = [*branches[False], *branches[True]]
                    data_sources = sorted({edge.source for edge in edges if edge.source not in core_set})
                    feedback_sources = sorted({edge.source for edge in edges if edge.source in core_set})
                    if not data_sources or not feedback_sources:
                        continue
                    muxes.append(
                        InputMuxCandidate(
                            storage_node=storage_node,
                            select_net=select_signal,
                            data_sources=tuple(data_sources),
                            feedback_sources=tuple(feedback_sources),
                            edges=tuple(edges),
                            enable_net=None if select_signal in clock_set else select_signal,
                        )
                    )
        return muxes

    def _detect_async_controls(self, cores: list[BistableCoreCandidate]) -> list[AsyncControlCandidate]:
        results: list[AsyncControlCandidate] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        for core in cores:
            core_set = set(core.nodes)
            for edge in self._graphs.conduction.edges:
                if edge.target not in core_set or len(edge.controls) != 1:
                    continue
                if edge.source not in self._power and edge.source not in self._ground:
                    continue
                literal = edge.controls[0]
                if literal.signal in core_set:
                    # Part of the bistable inverter itself, not an external async force path.
                    continue
                role = "set" if edge.source in self._power else "reset"
                active_level = LogicLevel.HIGH if literal.required_level else LogicLevel.LOW
                key = (literal.signal, role, tuple(sorted(core_set)))
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    AsyncControlCandidate(
                        signal=literal.signal,
                        role=role,
                        active_level=active_level,
                        target_nodes=tuple(sorted(core_set)),
                        path_edges=(edge,),
                    )
                )
            for edge in core.incoming_edges:
                if edge.semantic != EdgeSemantic.ASYNC_CONTROL:
                    continue
                role = str(edge.metadata.get("role", "reset"))
                active_low = bool(edge.metadata.get("active_low_name", False))
                active_level = LogicLevel.LOW if active_low else LogicLevel.HIGH
                key = (edge.source, role, tuple(sorted(core_set)))
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    AsyncControlCandidate(
                        signal=edge.source,
                        role=role,
                        active_level=active_level,
                        target_nodes=tuple(sorted(core_set)),
                        path_edges=(edge,),
                    )
                )
        return results

    def _detect_clock_gating(self, clock_analysis: ClockChainAnalysis) -> list[dict]:
        clock_candidates = set(clock_analysis.clock_candidates)
        features: list[dict] = []
        for clock_net in clock_candidates:
            incoming = self._graphs.influence.incoming.get(clock_net, [])
            if len(incoming) < 2:
                continue
            sources = {edge.source for edge in incoming if edge.source != clock_net}
            source_clocks = [net for net in sources if self._looks_like_clock(net)]
            non_clocks = [net for net in sources if net not in source_clocks]
            if source_clocks and non_clocks:
                features.append(
                    {
                        "kind": "clock_gating",
                        "gated_clock": clock_net,
                        "clock_sources": sorted(source_clocks),
                        "enable_sources": sorted(non_clocks),
                        "incoming_edges": [edge.as_dict() for edge in incoming],
                    }
                )
        return features
