"""logic_functions: detect topological latch candidates."""
from __future__ import annotations
from typing import TYPE_CHECKING


class DetectTopologicalLatchCandidatesMixin:
    def _detect_topological_latch_candidates(
        self,
        transistors: list[_Transistor],
        inverter_map: dict[tuple[str, str], dict],
        switches: list[dict],
        power_nets: set[str],
        ground_nets: set[str],
        nets_by_canonical: dict[str, list[str]],
        output_nets: set[str],
    ) -> list[dict]:
        terminal_degree = self._terminal_degree(transistors)
        cross_coupled_pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for in_net, out_net in inverter_map:
            if in_net == out_net or (out_net, in_net) not in inverter_map:
                continue
            pair = tuple(sorted((in_net, out_net), key=self._natural_key))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            cross_coupled_pairs.append(pair)
        cross_coupled_pairs.sort(key=lambda pair: (self._natural_key(pair[0]), self._natural_key(pair[1])))
        all_core_nets: set[str] = set()
        for pair in cross_coupled_pairs:
            all_core_nets.update(pair)

        latch_candidates: list[dict] = []
        for pair in cross_coupled_pairs:
            core_nets = set(pair)
            core_inverters = [inverter_map[(pair[0], pair[1])], inverter_map[(pair[1], pair[0])]]
            core_instances = self._unique(core_inverters[0]["instances"] + core_inverters[1]["instances"])

            attached_switches: list[dict] = []
            control_votes: dict[tuple[str, str], dict] = {}
            involved_instances = list(core_instances)
            involved_nets: set[str] = set(core_nets)

            for switch in switches:
                switch_nets = set(switch["nets"])
                touch_core = len(switch_nets & core_nets)
                if touch_core == 0:
                    continue
                attached_switches.append(switch)
                for instance_name in switch.get("instances", []):
                    if instance_name not in involved_instances:
                        involved_instances.append(instance_name)
                involved_nets.update(switch_nets)
                primary_phase = switch.get("primary_phase")
                if not isinstance(primary_phase, dict):
                    continue
                key = (str(primary_phase.get("net")), str(primary_phase.get("level")))
                if not key[0] or key[1] not in {"high", "low"}:
                    continue
                external_nets = [
                    net_name
                    for net_name in switch["nets"]
                    if net_name not in core_nets and net_name not in power_nets and net_name not in ground_nets
                ]
                noncore_external_nets = [net_name for net_name in external_nets if net_name not in all_core_nets]
                interstage_nets = [net_name for net_name in external_nets if net_name in all_core_nets]
                vote = control_votes.setdefault(
                    key,
                    {
                        "net": key[0],
                        "level": key[1],
                        "phase_kind": str(primary_phase.get("kind") or ""),
                        "switches": [],
                        "switch_count": 0,
                        "external_switch_count": 0,
                        "noncore_external_switch_count": 0,
                        "interstage_switch_count": 0,
                        "feedback_switch_count": 0,
                        "complement_nets": set(),
                    },
                )
                vote["switches"].append(switch)
                vote["switch_count"] += 1
                if touch_core == 1 and external_nets:
                    vote["external_switch_count"] += 1
                if touch_core == 1 and noncore_external_nets:
                    vote["noncore_external_switch_count"] += 1
                if touch_core == 1 and interstage_nets:
                    vote["interstage_switch_count"] += 1
                if touch_core == 2:
                    vote["feedback_switch_count"] += 1
                if primary_phase.get("kind") == "nmos":
                    opposite_gates = switch.get("pmos_gates", [])
                else:
                    opposite_gates = switch.get("nmos_gates", [])
                for gate_net in opposite_gates:
                    if self._nets_are_inverted(key[0], gate_net, inverter_map):
                        vote["complement_nets"].add(gate_net)
                        for instance_name in self._inverter_instances_between(key[0], gate_net, inverter_map):
                            if instance_name not in involved_instances:
                                involved_instances.append(instance_name)
                        involved_nets.add(gate_net)

            if not attached_switches or not control_votes:
                continue

            ranked_votes = sorted(
                control_votes.values(),
                key=lambda item: (
                    -int(item["noncore_external_switch_count"]),
                    -int(item["external_switch_count"]),
                    -int(item["interstage_switch_count"]),
                    -int(item["switch_count"]),
                    -int(item["feedback_switch_count"]),
                    0 if item.get("phase_kind") == "nmos" else 1,
                    self._natural_key(str(item.get("net", ""))),
                    str(item.get("level", "")),
                ),
            )
            primary_vote = ranked_votes[0]
            if int(primary_vote.get("external_switch_count", 0)) <= 0:
                continue

            primary_enable_net = str(primary_vote["net"])
            primary_enable_level = str(primary_vote["level"])
            complement_nets = sorted(primary_vote.get("complement_nets", set()), key=self._natural_key)
            involved_nets.add(primary_enable_net)
            involved_nets.update(complement_nets)

            data_nets: list[str] = []
            preferred_data_nets: list[str] = []
            fallback_interstage_data_nets: list[str] = []
            for switch in primary_vote.get("switches", []):
                switch_nets = set(switch["nets"])
                if len(switch_nets & core_nets) != 1:
                    continue
                for net_name in switch["nets"]:
                    if net_name in core_nets or net_name in power_nets or net_name in ground_nets:
                        continue
                    if net_name in all_core_nets:
                        if net_name not in fallback_interstage_data_nets:
                            fallback_interstage_data_nets.append(net_name)
                    else:
                        if net_name not in preferred_data_nets:
                            preferred_data_nets.append(net_name)
            data_nets = preferred_data_nets or fallback_interstage_data_nets
            data_nets.sort(key=self._natural_key)

            if not data_nets:
                continue

            def core_net_score(net_name: str) -> tuple[int, int, int]:
                external_touch = 0
                for switch in attached_switches:
                    switch_set = set(switch["nets"])
                    if net_name not in switch_set:
                        continue
                    if len(switch_set & core_nets) == 1 and any(other not in core_nets for other in switch["nets"]):
                        external_touch += 1
                return (
                    1 if net_name in output_nets else 0,
                    external_touch,
                    terminal_degree.get(net_name, 0),
                )

            ordered_core_nets = sorted(pair, key=lambda net_name: (core_net_score(net_name),), reverse=True)
            if len(ordered_core_nets) < 2:
                ordered_core_nets = list(pair)
            primary_output_net = ordered_core_nets[0]
            secondary_output_net = ordered_core_nets[1]

            control_signals: list[dict] = [
                {
                    "role": "enable",
                    "port": None,
                    "net": primary_enable_net,
                    "activation": {"mode": "level", "level": primary_enable_level},
                }
            ]
            for comp_net in complement_nets:
                control_signals.append(
                    {
                        "role": "phase_complement",
                        "port": None,
                        "net": comp_net,
                        "activation": {
                            "mode": "level",
                            "level": "low" if primary_enable_level == "high" else "high",
                        },
                    }
                )
            for vote in ranked_votes[1:]:
                net_name = str(vote.get("net", ""))
                level = str(vote.get("level", ""))
                if not net_name or level not in {"high", "low"}:
                    continue
                role = "phase" if self._nets_are_inverted(primary_enable_net, net_name, inverter_map) else "control"
                control_signals.append(
                    {
                        "role": role,
                        "port": None,
                        "net": net_name,
                        "activation": {"mode": "level", "level": level},
                    }
                )
                involved_nets.add(net_name)
            control_signals = self._unique_seq_entries(control_signals)

            output_entries = [
                {"port": None, "net": primary_output_net, "polarity": "non_inverted"},
                {"port": None, "net": secondary_output_net, "polarity": "inverted"},
            ]
            data_entries = [{"port": None, "net": net_name} for net_name in data_nets]

            control_nets = [item["net"] for item in control_signals if item.get("net")]
            output_nets_local = [item["net"] for item in output_entries if item.get("net")]
            data_nets_local = [item["net"] for item in data_entries if item.get("net")]
            involved_tx_indexes = sorted(
                self._unique(
                    core_inverters[0].get("transistor_indexes", [])
                    + core_inverters[1].get("transistor_indexes", [])
                    + [idx for switch in attached_switches for idx in switch.get("transistor_indexes", [])]
                )
            )

            expanded_all_nets = sorted(
                self._expanded_aliases(sorted(involved_nets, key=self._natural_key), nets_by_canonical),
                key=self._natural_key,
            )
            expanded_control_nets = sorted(self._expanded_aliases(control_nets, nets_by_canonical), key=self._natural_key)
            expanded_output_nets = sorted(self._expanded_aliases(output_nets_local, nets_by_canonical), key=self._natural_key)
            expanded_data_nets = sorted(self._expanded_aliases(data_nets_local, nets_by_canonical), key=self._natural_key)

            latch_candidates.append(
                {
                    "_core_nets": sorted(core_nets, key=self._natural_key),
                    "_enable": {
                        "net": primary_enable_net,
                        "level": primary_enable_level,
                        "complement_nets": complement_nets,
                    },
                    "_data_nets": data_nets_local,
                    "_output_nets": output_nets_local,
                    "_all_instances": sorted(involved_instances, key=self._natural_key),
                    "_all_nets_canonical": sorted(involved_nets, key=self._natural_key),
                    "_transistor_indexes": involved_tx_indexes,
                    "instance": None,
                    "cell": None,
                    "view": None,
                    "library": None,
                    "designator": None,
                    "kind": "latch",
                    "subtype": "D",
                    "outputs": output_entries,
                    "data_inputs": data_entries,
                    "control_signals": control_signals,
                    "triggering": {
                        "mode": "level",
                        "level": primary_enable_level,
                        "port": None,
                        "net": primary_enable_net,
                    },
                    "logic_path": {
                        "nets": expanded_all_nets,
                        "instances": sorted(involved_instances, key=self._natural_key),
                    },
                    "highlight": {
                        "nets": expanded_all_nets,
                        "instances": sorted(involved_instances, key=self._natural_key),
                        "control_nets": expanded_control_nets,
                        "data_nets": expanded_data_nets,
                        "output_nets": expanded_output_nets,
                    },
                }
            )

        return latch_candidates

if TYPE_CHECKING:
    from krona.application.logic_functions_parts.transistor import _Transistor
