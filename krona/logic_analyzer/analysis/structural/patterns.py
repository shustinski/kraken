from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Iterable

from .graphs import GraphBundle
from .model import (
    ActivationMode,
    ActivationSpec,
    BistableCoreEvidence,
    CircuitDevice,
    CircuitModel,
    CompositeFeature,
    ConditionalEdge,
    EdgeSemantic,
    LogicLevel,
    RecognizedCellKind,
    RecognizedStructure,
)
from .phases import ClockChainAnalysis, PhaseAnalyzer
from .scc import TarjanSCCDetector


@dataclass(frozen=True)
class BistableCoreCandidate:
    nodes: tuple[str, ...]
    evidence: BistableCoreEvidence
    incoming_edges: tuple[ConditionalEdge, ...]
    outgoing_edges: tuple[ConditionalEdge, ...]
    storage_outputs: tuple[str, ...]
    cross_coupled_pair: tuple[str, str] | None = None


@dataclass(frozen=True)
class InputMuxCandidate:
    storage_node: str
    select_net: str | None
    data_sources: tuple[str, ...]
    feedback_sources: tuple[str, ...]
    edges: tuple[ConditionalEdge, ...]
    enable_net: str | None = None

    def as_dict(self) -> dict:
        return {
            "storage_node": self.storage_node,
            "select_net": self.select_net,
            "data_sources": list(self.data_sources),
            "feedback_sources": list(self.feedback_sources),
            "edges": [edge.as_dict() for edge in self.edges],
            "enable_net": self.enable_net,
        }


@dataclass(frozen=True)
class AsyncControlCandidate:
    signal: str
    role: str
    active_level: LogicLevel
    target_nodes: tuple[str, ...]
    path_edges: tuple[ConditionalEdge, ...]

    def as_dict(self) -> dict:
        return {
            "signal": self.signal,
            "role": self.role,
            "active_level": self.active_level.value,
            "asynchronous": True,
            "target_nodes": list(self.target_nodes),
            "path_edges": [edge.as_dict() for edge in self.path_edges],
        }


@dataclass(frozen=True)
class StoragePhaseProfile:
    storage_nodes: tuple[str, ...]
    data_sources: tuple[str, ...]
    feedback_sources: tuple[str, ...]
    control_signals: tuple[str, ...]
    transparency: dict[str, dict]
    hold_checks: list[dict]

    def inferred_enable(self) -> tuple[str | None, str | None]:
        if not self.transparency:
            return None, None
        counts: dict[tuple[str, bool], int] = {}
        for info in self.transparency.values():
            if not info.get("transparent"):
                continue
            assignment = info.get("assignment", {})
            for signal in self.control_signals:
                if signal in assignment:
                    counts[(signal, bool(assignment[signal]))] = counts.get((signal, bool(assignment[signal])), 0) + 1
        if not counts:
            return None, None
        (signal, level), _ = max(counts.items(), key=lambda item: (item[1], item[0][0]))
        return signal, ("high" if level else "low")


@dataclass(frozen=True)
class PatternDatabase:
    scc_summaries: list[dict]
    bistable_cores: list[BistableCoreCandidate]
    input_muxes: list[InputMuxCandidate]
    async_controls: list[AsyncControlCandidate]
    clock_analysis: ClockChainAnalysis
    clock_gating_features: list[dict]
    keeper_features: list[dict]
    dynamic_features: list[dict]
    xor_toggle_features: list[dict]
    toggle_chains: list[dict]
    diagnostics: list[str] = field(default_factory=list)


class PatternMatcher:
    """
    Finds structural patterns and low-level evidence needed by semantic classification.
    """

    def __init__(self, circuit: CircuitModel, graphs: GraphBundle, phase_analyzer: PhaseAnalyzer):
        self._circuit = circuit
        self._graphs = graphs
        self._phase = phase_analyzer
        self._power = circuit.power_nets()
        self._ground = circuit.ground_nets()

    def run(self) -> PatternDatabase:
        clock_analysis = self._phase.analyze_clock_chains()
        scc_summaries, bistable_cores = self._detect_bistable_cores()
        input_muxes = self._detect_input_muxes(bistable_cores, clock_analysis)
        async_controls = self._detect_async_controls(bistable_cores)
        clock_gating = self._detect_clock_gating(clock_analysis)
        dynamic_features = self._detect_dynamic_precharge_evaluate(clock_analysis)
        keeper_features = self._detect_keepers(dynamic_features, bistable_cores)
        xor_toggle = self._detect_xor_toggle_feedback()
        toggle_chains = self._detect_toggle_chains(xor_toggle)

        diagnostics = list(self._graphs.diagnostics)
        return PatternDatabase(
            scc_summaries=scc_summaries,
            bistable_cores=bistable_cores,
            input_muxes=input_muxes,
            async_controls=async_controls,
            clock_analysis=clock_analysis,
            clock_gating_features=clock_gating,
            keeper_features=keeper_features,
            dynamic_features=dynamic_features,
            xor_toggle_features=xor_toggle,
            toggle_chains=toggle_chains,
            diagnostics=diagnostics,
        )

    def _detect_bistable_cores(self) -> tuple[list[dict], list[BistableCoreCandidate]]:
        scc_detector = TarjanSCCDetector(self._graphs.influence.adjacency)
        sccs = scc_detector.run()
        edge_set = self._graphs.influence.edges

        scc_summaries: list[dict] = []
        cores: list[BistableCoreCandidate] = []
        for component in sccs:
            nodes = tuple(sorted(component.nodes))
            internal_edges = [
                edge
                for edge in edge_set
                if edge.source in nodes and edge.target in nodes and edge.semantic == EdgeSemantic.INFLUENCE
            ]
            has_self_loop = any(edge.source == edge.target for edge in internal_edges)
            if component.size == 1 and not has_self_loop:
                scc_summaries.append(
                    {
                        "nodes": list(nodes),
                        "size": component.size,
                        "candidate_storage": False,
                        "positive_feedback": False,
                        "stable_state_count": 0,
                    }
                )
                continue

            positive_feedback = self._has_positive_feedback(nodes, internal_edges)
            stable_states = self._stable_states(nodes, internal_edges)
            cross_pairs = self._cross_coupled_inverters(nodes, internal_edges)
            candidate_storage = positive_feedback and len(stable_states) >= 2

            scc_summaries.append(
                {
                    "nodes": list(nodes),
                    "size": component.size,
                    "candidate_storage": candidate_storage,
                    "positive_feedback": positive_feedback,
                    "stable_state_count": len(stable_states),
                    "stable_states": stable_states,
                    "cross_coupled_inverters": [list(pair) for pair in cross_pairs],
                }
            )
            if not candidate_storage:
                continue

            incoming = tuple(edge for edge in edge_set if edge.target in nodes and edge.source not in nodes)
            outgoing = tuple(edge for edge in edge_set if edge.source in nodes and edge.target not in nodes)
            outputs = tuple(sorted({edge.source for edge in outgoing} | {node for node in nodes if self._has_outgoing_to_noncore(node, nodes)}))
            evidence = BistableCoreEvidence(
                scc_nodes=nodes,
                positive_feedback=positive_feedback,
                stable_states=stable_states,
                cross_coupled_inverters=cross_pairs,
            )
            cores.append(
                BistableCoreCandidate(
                    nodes=nodes,
                    evidence=evidence,
                    incoming_edges=incoming,
                    outgoing_edges=outgoing,
                    storage_outputs=outputs or nodes,
                    cross_coupled_pair=cross_pairs[0] if cross_pairs else None,
                )
            )
        return scc_summaries, cores

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

    def _detect_dynamic_precharge_evaluate(self, clock_analysis: ClockChainAnalysis) -> list[dict]:
        clock_set = set(clock_analysis.clock_candidates)
        inverse_pairs = {tuple(sorted(pair)) for pair in clock_analysis.inverse_pairs}
        features: list[dict] = []
        for node in self._circuit.nets:
            incoming_conduction = [edge for edge in self._graphs.conduction.edges if edge.target == node]
            from_power = [edge for edge in incoming_conduction if edge.source in self._power and edge.controls]
            if not from_power:
                continue
            out_edges = [edge for edge in self._graphs.conduction.adjacency.get(node, []) if edge.controls]
            if not out_edges:
                continue
            for p_edge in from_power:
                if len(p_edge.controls) != 1:
                    continue
                pre_lit = p_edge.controls[0]
                if pre_lit.signal not in clock_set:
                    continue
                complementary_eval = [
                    edge
                    for edge in out_edges
                    if len(edge.controls) == 1
                    and edge.controls[0].signal in clock_set
                    and (
                        (
                            edge.controls[0].signal == pre_lit.signal
                            and edge.controls[0].required_level != pre_lit.required_level
                        )
                        or tuple(sorted((edge.controls[0].signal, pre_lit.signal))) in inverse_pairs
                    )
                ]
                if complementary_eval:
                    features.append(
                        {
                            "kind": "dynamic_precharge_evaluate",
                            "dynamic_node": node,
                            "precharge_edge": p_edge.as_dict(),
                            "evaluate_edges": [edge.as_dict() for edge in complementary_eval],
                        }
                    )
                    break
        return self._dedupe_dicts(features)

    def _detect_keepers(self, dynamic_features: list[dict], cores: list[BistableCoreCandidate]) -> list[dict]:
        dynamic_nodes = {item["dynamic_node"] for item in dynamic_features if "dynamic_node" in item}
        core_nodes = {node for core in cores for node in core.nodes}
        keepers: list[dict] = []
        for node in dynamic_nodes:
            incoming = self._graphs.influence.incoming.get(node, [])
            for edge in incoming:
                if edge.source == node or edge.source in core_nodes or edge.inversion not in {0, 1}:
                    continue
                has_return = any(
                    e2.source == node and e2.target == edge.source and e2.inversion in {0, 1}
                    for e2 in self._graphs.influence.outgoing.get(node, [])
                )
                if has_return:
                    keepers.append(
                        {
                            "kind": "keeper",
                            "dynamic_node": node,
                            "feedback_pair": [edge.source, node],
                            "evidence_edges": [edge.as_dict()],
                        }
                    )
        return self._dedupe_dicts(keepers)

    def _detect_xor_toggle_feedback(self) -> list[dict]:
        features: list[dict] = []
        devices = self._circuit.devices
        xor_drivers: dict[str, dict] = {}
        for device in devices.values():
            if device.kind.name not in {"XOR", "XNOR"}:
                continue
            out_net = self._pick_gate_output(device)
            in_nets = self._pick_gate_inputs(device)
            if not out_net or len(in_nets) < 2:
                continue
            xor_drivers[out_net] = {
                "device": device,
                "inputs": in_nets,
                "xnor": device.kind.name == "XNOR",
            }
        for device in devices.values():
            if device.kind.name != "FF_D":
                continue
            d_net = self._pick_pin(device, "D", "DATA", "IN")
            q_net = self._pick_pin(device, "Q", "QN", "QB", "OUT")
            if not d_net or not q_net:
                continue
            xor_info = xor_drivers.get(d_net)
            if not xor_info or q_net not in xor_info["inputs"]:
                continue
            toggle_inputs = [net for net in xor_info["inputs"] if net != q_net]
            if not toggle_inputs:
                continue
            features.append(
                {
                    "kind": "xor_toggle_feedback",
                    "ff_device": device.name,
                    "d_net": d_net,
                    "q_net": q_net,
                    "xor_device": xor_info["device"].name,
                    "toggle_inputs": sorted(toggle_inputs),
                    "xnor": bool(xor_info["xnor"]),
                }
            )
        return features

    def _detect_toggle_chains(self, xor_toggle: list[dict]) -> list[dict]:
        q_to_stage: dict[str, str] = {}
        for item in xor_toggle:
            q_to_stage[str(item.get("q_net"))] = str(item.get("ff_device"))
        links: list[tuple[str, str, str]] = []
        for item in xor_toggle:
            dst_ff = str(item.get("ff_device"))
            for toggle_in in item.get("toggle_inputs", []):
                src_ff = q_to_stage.get(str(toggle_in))
                if src_ff and src_ff != dst_ff:
                    links.append((src_ff, dst_ff, str(toggle_in)))
        if not links:
            return []
        adjacency: dict[str, set[str]] = {}
        edge_payload: dict[tuple[str, str], list[str]] = {}
        for src, dst, net in links:
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set())
            edge_payload.setdefault((src, dst), []).append(net)
        indegree = {node: 0 for node in adjacency}
        for dsts in adjacency.values():
            for dst in dsts:
                indegree[dst] = indegree.get(dst, 0) + 1
        starts = [node for node, deg in indegree.items() if deg == 0] or list(adjacency)
        chains: list[dict] = []
        seen_paths: set[tuple[str, ...]] = set()
        for start in starts:
            path = [start]
            current = start
            local_seen = {start}
            while adjacency.get(current):
                nxt = sorted(adjacency[current])[0]
                if nxt in local_seen:
                    break
                path.append(nxt)
                local_seen.add(nxt)
                current = nxt
            if len(path) < 2:
                continue
            key = tuple(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            chains.append(
                {
                    "kind": "toggle_chain",
                    "stages": path,
                    "links": [
                        {"src_stage": s, "dst_stage": d, "feedback_nets": sorted(edge_payload.get((s, d), []))}
                        for s, d in zip(path, path[1:])
                    ],
                }
            )
        return chains

    @staticmethod
    def _looks_like_clock(net: str) -> bool:
        upper = net.upper()
        return "CLK" in upper or "CLOCK" in upper or upper in {"CK", "CP"}

    @staticmethod
    def _pick_pin(device: CircuitDevice, *aliases: str) -> str | None:
        normalized = {name.upper(): net for name, net in device.pins.items()}
        for alias in aliases:
            value = normalized.get(alias.upper())
            if value:
                return value
        return None

    def _pick_gate_output(self, device: CircuitDevice) -> str | None:
        return self._pick_pin(device, "Y", "Z", "Q", "OUT", "O")

    def _pick_gate_inputs(self, device: CircuitDevice) -> list[str]:
        out = self._pick_gate_output(device)
        result: list[str] = []
        for pin, net in device.pins.items():
            if net == out:
                continue
            if pin.upper() in {"VDD", "VSS", "GND", "VCC", "VPWR", "VGND"}:
                continue
            result.append(net)
        seen: set[str] = set()
        unique: list[str] = []
        for net in result:
            if net in seen:
                continue
            seen.add(net)
            unique.append(net)
        return unique

    @staticmethod
    def _dedupe_dicts(items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for item in items:
            key = repr(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out


class SemanticClassifier:
    """
    Converts structural evidence into semantic storage-cell classes.
    """

    def __init__(self, circuit: CircuitModel, graphs: GraphBundle, phase_analyzer: PhaseAnalyzer):
        self._circuit = circuit
        self._graphs = graphs
        self._phase = phase_analyzer

    def classify(self, patterns: PatternDatabase) -> list[RecognizedStructure]:
        recognized: list[RecognizedStructure] = []
        recognized.extend(self._classify_sequential_macros(patterns))
        recognized.extend(self._classify_bistable_cores(patterns))
        recognized = self._upgrade_jk_t_counter(recognized, patterns)
        return self._dedupe_recognized(recognized)

    def _classify_sequential_macros(self, patterns: PatternDatabase) -> list[RecognizedStructure]:
        out: list[RecognizedStructure] = []
        clock_gated_nets = {item.get("gated_clock") for item in patterns.clock_gating_features}
        toggle_features = {str(item.get("ff_device")): item for item in patterns.xor_toggle_features}

        for device in self._circuit.devices.values():
            if device.domain.name != "SEQUENTIAL_MACRO":
                continue
            q = self._pin(device, "Q", "QN", "QB", "OUT", "Z")
            d = self._pin(device, "D", "DATA", "IN")
            clk = self._pin(device, "CLK", "CLOCK", "CK", "CP")
            rst = self._pin(device, "RST", "RESET", "CLR", "RN", "CLRB", "RESETB")
            set_ = self._pin(device, "SET", "PRE", "SN", "PREB", "SETB")
            sync_rst = self._pin(device, "SRST", "SRESET", "SCLR", "SCLRN", "SRN")
            sync_set = self._pin(device, "SSET", "SPRE", "SPRESET", "SPREN", "SSETN")
            en = self._pin(device, "EN", "E", "G", "CE")
            j = self._pin(device, "J")
            k = self._pin(device, "K")
            t = self._pin(device, "T")

            kind = RecognizedCellKind.UNKNOWN_STORAGE
            base_conf = 0.75
            if device.kind.name == "LATCH_D":
                kind = RecognizedCellKind.GATED_D_LATCH
            elif device.kind.name == "LATCH_RS":
                kind = RecognizedCellKind.RS_LATCH
            elif device.kind.name == "FF_D":
                kind = RecognizedCellKind.D_FLIP_FLOP
            elif device.kind.name == "FF_JK":
                kind = RecognizedCellKind.JK_FLIP_FLOP
            elif device.kind.name == "FF_T":
                kind = RecognizedCellKind.T_FLIP_FLOP
            elif device.kind.name == "COUNTER":
                kind = RecognizedCellKind.COUNTER_CELL

            controls: list[dict] = []
            features: list[CompositeFeature] = []
            inputs = [net for net in [d, j, k, t] if net]
            outputs = [q] if q else []
            clocks = [clk] if clk else []

            if clk:
                controls.append({"role": "clock", "net": clk, "activation": {"mode": "edge_or_level_unknown"}})
                if clk in clock_gated_nets:
                    features.append(CompositeFeature.CLOCK_GATING)
            if en:
                controls.append(
                    {
                        "role": "enable",
                        "net": en,
                        "activation": ActivationSpec(
                            mode=ActivationMode.LEVEL,
                            level=LogicLevel.LOW if self._is_active_low_name(en) else LogicLevel.HIGH,
                        ).as_dict(),
                    }
                )
                features.append(CompositeFeature.CLOCK_ENABLE)
            if rst:
                controls.append(
                    {
                        "role": "reset",
                        "net": rst,
                        "activation": ActivationSpec(
                            mode=ActivationMode.LEVEL,
                            level=LogicLevel.LOW if self._is_active_low_name(rst) else LogicLevel.HIGH,
                        ).as_dict(),
                        "timing": "asynchronous",
                    }
                )
                features.append(CompositeFeature.ASYNC_RESET)
            if set_:
                controls.append(
                    {
                        "role": "set",
                        "net": set_,
                        "activation": ActivationSpec(
                            mode=ActivationMode.LEVEL,
                            level=LogicLevel.LOW if self._is_active_low_name(set_) else LogicLevel.HIGH,
                        ).as_dict(),
                        "timing": "asynchronous",
                    }
                )
                features.append(CompositeFeature.ASYNC_SET)
            if sync_rst:
                controls.append(
                    {
                        "role": "reset",
                        "net": sync_rst,
                        "activation": ActivationSpec(
                            mode=ActivationMode.LEVEL,
                            level=LogicLevel.LOW if self._is_active_low_name(sync_rst) else LogicLevel.HIGH,
                        ).as_dict(),
                        "timing": "synchronous",
                    }
                )
                features.append(CompositeFeature.SYNC_RESET)
            if sync_set:
                controls.append(
                    {
                        "role": "set",
                        "net": sync_set,
                        "activation": ActivationSpec(
                            mode=ActivationMode.LEVEL,
                            level=LogicLevel.LOW if self._is_active_low_name(sync_set) else LogicLevel.HIGH,
                        ).as_dict(),
                        "timing": "synchronous",
                    }
                )
                features.append(CompositeFeature.SYNC_SET)

            evidence = {
                "macro_device": device.name,
                "device_kind": device.kind.value,
                "pins": dict(device.pins),
            }
            if device.name in toggle_features and kind in {RecognizedCellKind.D_FLIP_FLOP, RecognizedCellKind.EDGE_TRIGGERED_DFF}:
                features.append(CompositeFeature.XOR_TOGGLE_FEEDBACK)
                evidence["xor_toggle_feedback"] = toggle_features[device.name]

            out.append(
                RecognizedStructure(
                    kind=kind,
                    confidence=base_conf,
                    storage_nodes=outputs or ([q] if q else []),
                    outputs=outputs,
                    inputs=inputs,
                    clocks=clocks,
                    controls=controls,
                    features=self._unique_features(features),
                    evidence=evidence,
                    components=[device.name],
                )
            )
        return out

    def _classify_bistable_cores(self, patterns: PatternDatabase) -> list[RecognizedStructure]:
        recognized: list[RecognizedStructure] = []
        clock_set = set(patterns.clock_analysis.clock_candidates)

        for core in patterns.bistable_cores:
            core_set = set(core.nodes)
            core_muxes = [mux for mux in patterns.input_muxes if mux.storage_node in core_set]
            core_async = [ctrl for ctrl in patterns.async_controls if core_set.intersection(ctrl.target_nodes)]
            data_sources = sorted({src for mux in core_muxes for src in mux.data_sources})
            controls = self._controls_from_async(core_async)
            features: list[CompositeFeature] = []
            if core.evidence.cross_coupled_inverters:
                features.append(CompositeFeature.CROSS_COUPLED_INVERTERS)

            phase_profiles = self._analyze_core_phases(core, core_muxes, patterns.clock_analysis)
            enable_net = None
            enable_level = None
            if phase_profiles:
                enable_net, enable_level = phase_profiles[0].inferred_enable()

            if core_muxes:
                features.append(CompositeFeature.INPUT_MUX)
                if any(mux.feedback_sources for mux in core_muxes):
                    features.append(CompositeFeature.FEEDBACK_MUX)

            if enable_net and enable_net not in clock_set:
                controls.append(
                    {
                        "role": "enable",
                        "net": enable_net,
                        "activation": {"mode": "level", "level": enable_level or "unknown"},
                    }
                )
                features.append(CompositeFeature.CLOCK_ENABLE)

            if any(ctrl.role == "reset" for ctrl in core_async):
                features.append(CompositeFeature.ASYNC_RESET)
            if any(ctrl.role == "set" for ctrl in core_async):
                features.append(CompositeFeature.ASYNC_SET)

            inferred_clocks = []
            for mux in core_muxes:
                if mux.select_net and mux.select_net in clock_set:
                    inferred_clocks.append(mux.select_net)
            if phase_profiles and enable_net and enable_net in clock_set:
                inferred_clocks.append(enable_net)
            inferred_clocks = self._unique(inferred_clocks)

            kind = RecognizedCellKind.RS_LATCH
            confidence = 0.62
            if core_muxes and data_sources:
                kind = RecognizedCellKind.GATED_D_LATCH
                confidence = 0.8 if inferred_clocks else 0.72

            recognized.append(
                RecognizedStructure(
                    kind=kind,
                    confidence=confidence,
                    storage_nodes=list(core.nodes),
                    outputs=list(core.storage_outputs) or list(core.nodes),
                    inputs=data_sources,
                    clocks=inferred_clocks,
                    controls=controls,
                    features=self._unique_features(features),
                    evidence={
                        "bistable_core": {
                            "nodes": list(core.nodes),
                            "positive_feedback": core.evidence.positive_feedback,
                            "stable_states": core.evidence.stable_states,
                            "cross_coupled_inverters": [list(pair) for pair in core.evidence.cross_coupled_inverters],
                        },
                        "input_muxes": [mux.as_dict() for mux in core_muxes],
                        "async_controls": [ctrl.as_dict() for ctrl in core_async],
                        "phase_analysis": [self._phase_profile_to_dict(item) for item in phase_profiles],
                    },
                    components=self._core_component_devices(core),
                )
            )

        recognized.extend(self._compose_master_slave_ffs(recognized, patterns))
        return recognized

    def _analyze_core_phases(
        self,
        core: BistableCoreCandidate,
        muxes: list[InputMuxCandidate],
        clock_analysis: ClockChainAnalysis,
    ) -> list[StoragePhaseProfile]:
        if not muxes:
            return []
        profiles: list[StoragePhaseProfile] = []
        for mux in muxes:
            controls = []
            if mux.select_net:
                controls.append(mux.select_net)
            for a, b in clock_analysis.inverse_pairs:
                if mux.select_net == a and b not in controls:
                    controls.append(b)
                if mux.select_net == b and a not in controls:
                    controls.append(a)
            if not controls:
                continue
            transparency = self._phase.transparency_by_phase(
                data_sources=list(mux.data_sources),
                storage_nodes=[mux.storage_node],
                control_signals=controls,
                max_combinations=8,
            )
            hold_assignments = [
                {k: bool(v) for k, v in info.get("assignment", {}).items()}
                for info in transparency.values()
                if not info.get("transparent")
            ][:4]
            hold_checks = self._phase.hold_break_check(
                data_sources=list(mux.data_sources),
                storage_nodes=[mux.storage_node],
                hold_assignments=hold_assignments,
            )
            profiles.append(
                StoragePhaseProfile(
                    storage_nodes=(mux.storage_node,),
                    data_sources=tuple(mux.data_sources),
                    feedback_sources=tuple(mux.feedback_sources),
                    control_signals=tuple(controls),
                    transparency=transparency,
                    hold_checks=hold_checks,
                )
            )
        return profiles

    def _compose_master_slave_ffs(
        self,
        recognized: list[RecognizedStructure],
        patterns: PatternDatabase,
    ) -> list[RecognizedStructure]:
        latches = [item for item in recognized if item.kind == RecognizedCellKind.GATED_D_LATCH]
        if len(latches) < 2:
            return []
        inverse_pairs = {tuple(sorted(pair)) for pair in patterns.clock_analysis.inverse_pairs}
        composed: list[RecognizedStructure] = []
        for a, b in combinations(latches, 2):
            a_clk = a.clocks[0] if a.clocks else None
            b_clk = b.clocks[0] if b.clocks else None
            if not a_clk or not b_clk or a_clk == b_clk:
                continue
            if tuple(sorted((a_clk, b_clk))) not in inverse_pairs and not self._is_likely_complement_name_pair(a_clk, b_clk):
                continue
            if not a.outputs or not b.storage_nodes:
                continue
            a_to_b = bool(set(a.outputs).intersection(b.inputs))
            b_to_a = bool(set(b.outputs).intersection(a.inputs))
            if not a_to_b and not b_to_a:
                a_to_b = any(self._influence_path_exists(src, dst) for src in a.outputs for dst in b.storage_nodes)
                b_to_a = any(self._influence_path_exists(src, dst) for src in b.outputs for dst in a.storage_nodes)
            if a_to_b == b_to_a:
                continue
            master, slave = (a, b) if a_to_b else (b, a)
            base_clock = self._base_clock_name(master.clocks[0], slave.clocks[0]) or master.clocks[0]
            edge = self._infer_edge_polarity(master, slave, base_clock)
            features = self._unique_features([*master.features, *slave.features, CompositeFeature.CROSS_COUPLED_INVERTERS])
            controls = [*master.controls, *slave.controls, {"role": "clock", "net": base_clock, "activation": {"mode": "edge", "edge": edge}}]
            common_payload = {
                "storage_nodes": self._unique([*master.storage_nodes, *slave.storage_nodes]),
                "outputs": list(slave.outputs or slave.storage_nodes),
                "inputs": list(master.inputs),
                "clocks": [base_clock],
                "controls": controls,
                "features": features,
                "components": self._unique([*master.components, *slave.components]),
                "evidence": {
                    "master": master.as_dict(),
                    "slave": slave.as_dict(),
                    "phase_relation": "complementary_latches",
                },
            }
            composed.append(
                RecognizedStructure(
                    kind=RecognizedCellKind.MASTER_SLAVE_DFF,
                    confidence=0.88,
                    **common_payload,
                )
            )
            composed.append(
                RecognizedStructure(
                    kind=RecognizedCellKind.EDGE_TRIGGERED_DFF,
                    confidence=0.84,
                    **common_payload,
                )
            )
        return composed

    def _upgrade_jk_t_counter(self, recognized: list[RecognizedStructure], patterns: PatternDatabase) -> list[RecognizedStructure]:
        upgraded = list(recognized)
        xor_toggle_by_ff = {str(item.get("ff_device")): item for item in patterns.xor_toggle_features}
        for item in list(recognized):
            if item.kind not in {RecognizedCellKind.D_FLIP_FLOP, RecognizedCellKind.MASTER_SLAVE_DFF, RecognizedCellKind.EDGE_TRIGGERED_DFF}:
                continue
            macro_device = str(item.evidence.get("macro_device", "")) if isinstance(item.evidence, dict) else ""
            xor_toggle = xor_toggle_by_ff.get(macro_device)
            if xor_toggle:
                upgraded.append(
                    RecognizedStructure(
                        kind=RecognizedCellKind.T_FLIP_FLOP,
                        confidence=min(0.95, item.confidence + 0.1),
                        storage_nodes=list(item.storage_nodes),
                        outputs=list(item.outputs),
                        inputs=list(item.inputs),
                        clocks=list(item.clocks),
                        controls=list(item.controls),
                        features=self._unique_features([*item.features, CompositeFeature.XOR_TOGGLE_FEEDBACK]),
                        evidence={**item.evidence, "decoded_from": item.kind.value, "xor_toggle_feedback": xor_toggle},
                        components=self._unique([*item.components, str(xor_toggle.get("xor_device"))]),
                    )
                )
            jk_evidence = self._detect_jk_feedback_equation(item)
            if jk_evidence:
                upgraded.append(
                    RecognizedStructure(
                        kind=RecognizedCellKind.JK_FLIP_FLOP,
                        confidence=min(0.9, item.confidence + 0.08),
                        storage_nodes=list(item.storage_nodes),
                        outputs=list(item.outputs),
                        inputs=jk_evidence.get("inputs", item.inputs),
                        clocks=list(item.clocks),
                        controls=list(item.controls),
                        features=self._unique_features(item.features),
                        evidence={**item.evidence, "decoded_from": item.kind.value, "jk_decode": jk_evidence},
                        components=list(item.components),
                    )
                )
        for chain in patterns.toggle_chains:
            upgraded.append(
                RecognizedStructure(
                    kind=RecognizedCellKind.COUNTER_CELL,
                    confidence=0.78,
                    storage_nodes=[],
                    outputs=[],
                    inputs=[],
                    clocks=[],
                    controls=[],
                    features=[CompositeFeature.TOGGLE_CHAIN],
                    evidence={"toggle_chain": chain},
                    components=list(chain.get("stages", [])),
                )
            )
        return upgraded

    def _detect_jk_feedback_equation(self, item: RecognizedStructure) -> dict | None:
        d_net = None
        if isinstance(item.evidence, dict):
            pins = item.evidence.get("pins", {})
            if isinstance(pins, dict):
                d_net = pins.get("D") or pins.get("DATA") or pins.get("IN")
        q_nets = set(item.outputs or item.storage_nodes)
        if not d_net:
            return None
        incoming = self._graphs.influence.incoming.get(d_net, [])
        feedback_sources = [edge.source for edge in incoming if edge.source in q_nets]
        external_sources = [edge.source for edge in incoming if edge.source not in q_nets]
        if not feedback_sources or len(set(external_sources)) < 2:
            return None
        has_inv = any(edge.inversion == 1 for edge in incoming)
        has_noninv = any(edge.inversion == 0 for edge in incoming)
        if not (has_inv and has_noninv):
            return None
        return {
            "d_net": d_net,
            "feedback_sources": sorted(set(feedback_sources)),
            "inputs": sorted(set(external_sources))[:2],
            "method": "mixed_polarity_feedback_cone_heuristic",
        }

    def _controls_from_async(self, controls: list[AsyncControlCandidate]) -> list[dict]:
        return [
            {
                "role": control.role,
                "net": control.signal,
                "activation": {"mode": "level", "level": control.active_level.value},
                "timing": "asynchronous",
            }
            for control in controls
        ]

    def _phase_profile_to_dict(self, profile: StoragePhaseProfile) -> dict:
        return {
            "storage_nodes": list(profile.storage_nodes),
            "data_sources": list(profile.data_sources),
            "feedback_sources": list(profile.feedback_sources),
            "control_signals": list(profile.control_signals),
            "transparency": profile.transparency,
            "hold_checks": profile.hold_checks,
        }

    def _core_component_devices(self, core: BistableCoreCandidate) -> list[str]:
        core_set = set(core.nodes)
        components: set[str] = set()
        for edge in self._graphs.influence.edges:
            if edge.source in core_set or edge.target in core_set:
                components.update(edge.through_devices)
        for edge in self._graphs.conduction.edges:
            if edge.source in core_set or edge.target in core_set:
                components.update(edge.through_devices)
        return sorted(components)

    def _influence_path_exists(self, source: str, target: str) -> bool:
        return self._phase.influence_reachable(source, target, assignment={}, allow_unknown_controls=True)

    @staticmethod
    def _pin(device: CircuitDevice, *aliases: str) -> str | None:
        normalized = {name.upper(): net for name, net in device.pins.items()}
        for alias in aliases:
            value = normalized.get(alias.upper())
            if value:
                return value
        return None

    @staticmethod
    def _is_active_low_name(net: str) -> bool:
        upper = net.upper()
        return upper.endswith("B") or upper.endswith("_N") or upper.endswith("N") or upper.endswith("BAR")

    @classmethod
    def _is_likely_complement_name_pair(cls, a: str, b: str) -> bool:
        return cls._base_clock_name(a, b) is not None

    @staticmethod
    def _base_clock_name(a: str, b: str) -> str | None:
        au = a.upper()
        bu = b.upper()
        candidates = []
        for name in (au, bu):
            if name.endswith("B"):
                candidates.append(name[:-1])
            if name.endswith("_N"):
                candidates.append(name[:-2])
            if name.endswith("N"):
                candidates.append(name[:-1])
            if name.endswith("BAR"):
                candidates.append(name[:-3])
        for base in candidates:
            if not base:
                continue
            variants = {base + "B", base + "_N", base + "N", base + "BAR"}
            if au == base and bu in variants:
                return a
            if bu == base and au in variants:
                return b
        return None

    def _infer_edge_polarity(self, master: RecognizedStructure, slave: RecognizedStructure, base_clock: str) -> str:
        master_level = self._dominant_clock_level(master, base_clock)
        slave_level = self._dominant_clock_level(slave, base_clock)
        if master_level == 0 and slave_level == 1:
            return "rising"
        if master_level == 1 and slave_level == 0:
            return "falling"
        return "unknown"

    @staticmethod
    def _dominant_clock_level(item: RecognizedStructure, base_clock: str) -> int | None:
        evidence = item.evidence if isinstance(item.evidence, dict) else {}
        phase_analysis = evidence.get("phase_analysis", [])
        counts = {0: 0, 1: 0}
        if not isinstance(phase_analysis, list):
            return None
        for profile in phase_analysis:
            if not isinstance(profile, dict):
                continue
            transparency = profile.get("transparency", {})
            if not isinstance(transparency, dict):
                continue
            for phase in transparency.values():
                if not isinstance(phase, dict) or not phase.get("transparent"):
                    continue
                assignment = phase.get("assignment", {})
                if isinstance(assignment, dict) and base_clock in assignment:
                    counts[int(assignment[base_clock])] += 1
        if counts[0] == counts[1] == 0:
            return None
        return 1 if counts[1] > counts[0] else 0

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    @staticmethod
    def _unique_features(features: Iterable[CompositeFeature]) -> list[CompositeFeature]:
        out: list[CompositeFeature] = []
        seen: set[CompositeFeature] = set()
        for feature in features:
            if feature in seen:
                continue
            seen.add(feature)
            out.append(feature)
        return out

    @staticmethod
    def _dedupe_recognized(items: list[RecognizedStructure]) -> list[RecognizedStructure]:
        by_key: dict[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], RecognizedStructure] = {}
        for item in items:
            key = (
                item.kind.value,
                tuple(sorted(item.storage_nodes)),
                tuple(sorted(item.outputs)),
                tuple(sorted(item.components)),
            )
            prev = by_key.get(key)
            if prev is None or item.confidence > prev.confidence:
                by_key[key] = item
        return sorted(
            by_key.values(),
            key=lambda item: (-item.confidence, item.kind.value, ",".join(sorted(item.storage_nodes))),
        )
