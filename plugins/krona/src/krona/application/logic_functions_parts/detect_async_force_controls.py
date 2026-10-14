"""logic_functions: detect async force controls."""
from __future__ import annotations


class DetectAsyncForceControlsMixin:
    def _detect_async_force_controls_for_element(
        self,
        element: dict,
        transistors: list[_Transistor],
        power_nets: set[str],
        ground_nets: set[str],
    ) -> list[dict]:
        core_nets = set(str(net) for net in (element.get("_core_nets") or []))
        if not core_nets:
            core_nets = {str(item.get("net")) for item in (element.get("outputs") or []) if item.get("net")}
        core_nets.discard("")
        if not core_nets:
            return []

        recognized_tx_indexes = {int(idx) for idx in (element.get("_transistor_indexes") or []) if isinstance(idx, int) or str(idx).isdigit()}
        known_control_nets = {str(item.get("net")) for item in (element.get("control_signals") or []) if item.get("net")}
        known_control_nets.discard("")
        data_nets = {str(item.get("net")) for item in (element.get("data_inputs") or []) if item.get("net")}
        data_nets.discard("")

        output_polarity_by_net: dict[str, str] = {}
        for output_item in element.get("outputs", []) or []:
            net_name = str(output_item.get("net") or "")
            polarity = str(output_item.get("polarity") or "non_inverted")
            if net_name:
                output_polarity_by_net[net_name] = polarity
        if not output_polarity_by_net:
            return []

        candidates: list[dict] = []

        def add_force_candidate(
            role: str,
            gate_net: str,
            active_level: str,
            core_net: str,
            path_depth: int,
        ) -> None:
            if not role or not gate_net:
                return
            candidates.append(
                {
                    "role": role,
                    "port": None,
                    "net": gate_net,
                    "activation": {"mode": "level", "level": active_level},
                    "timing": "asynchronous",
                    "topology": {
                        "kind": "force_path",
                        "path_depth": path_depth,
                        "target_storage_net": core_net,
                    },
                }
            )

        for idx, transistor in enumerate(transistors):
            if idx in recognized_tx_indexes:
                continue
            terminals = {transistor.source, transistor.drain}
            touched_core = list(terminals & core_nets)
            if len(touched_core) != 1:
                continue
            core_net = touched_core[0]
            if transistor.kind == "pmos":
                if not (terminals & power_nets):
                    continue
                forced_value = 1
                active_level = "low"
            else:
                if not (terminals & ground_nets):
                    continue
                forced_value = 0
                active_level = "high"
            gate_net = transistor.gate
            if gate_net in core_nets or gate_net in data_nets:
                continue
            role = self._force_role_from_output_polarity(core_net, forced_value, output_polarity_by_net)
            if role is None:
                continue
            add_force_candidate(role=role, gate_net=gate_net, active_level=active_level, core_net=core_net, path_depth=1)

        for core_net in sorted(core_nets, key=self._natural_key):
            for kind, targets, forced_value, active_level in (
                ("pmos", power_nets, 1, "low"),
                ("nmos", ground_nets, 0, "high"),
            ):
                for path in self._two_transistor_paths_to_supply(
                    transistors=transistors,
                    kind=kind,
                    start_net=core_net,
                    target_nets=targets,
                ):
                    path_indexes = {int(idx) for idx in path.get("transistor_indexes", [])}
                    if path_indexes and path_indexes.issubset(recognized_tx_indexes):
                        continue
                    external_gates = []
                    for gate_net in path.get("gates", []):
                        if gate_net in core_nets or gate_net in data_nets or gate_net in known_control_nets:
                            continue
                        if gate_net not in external_gates:
                            external_gates.append(gate_net)
                    if len(external_gates) != 1:
                        continue
                    role = self._force_role_from_output_polarity(core_net, forced_value, output_polarity_by_net)
                    if role is None:
                        continue
                    add_force_candidate(
                        role=role,
                        gate_net=external_gates[0],
                        active_level=active_level,
                        core_net=core_net,
                        path_depth=2,
                    )

        # Remove duplicates and avoid shadowing the regular phase controls with same net/level.
        deduped = self._unique_seq_entries(candidates)
        final_controls: list[dict] = []
        for candidate in deduped:
            candidate_net = str(candidate.get("net") or "")
            candidate_level = str((candidate.get("activation") or {}).get("level") or "")
            role = str(candidate.get("role") or "")
            if role not in {"set", "reset"}:
                continue
            conflict = False
            for existing in element.get("control_signals", []) or []:
                existing_net = str(existing.get("net") or "")
                existing_level = str((existing.get("activation") or {}).get("level") or "")
                existing_role = str(existing.get("role") or "")
                if candidate_net == existing_net and candidate_level == existing_level and existing_role in {"enable", "phase", "master_phase", "slave_phase", "hold_phase", "phase_complement", "master_hold_phase", "slave_hold_phase", "clock"}:
                    conflict = True
                    break
            if not conflict:
                final_controls.append(candidate)
        return final_controls
