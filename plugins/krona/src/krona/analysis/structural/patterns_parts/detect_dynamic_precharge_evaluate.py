"""patterns: detect dynamic precharge evaluate."""
from __future__ import annotations
from krona.analysis.structural.phases import ClockChainAnalysis


class DetectDynamicPrechargeEvaluateMixin:
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
