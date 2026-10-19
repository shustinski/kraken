"""logic_functions: detect topological hybrid latch."""
from __future__ import annotations
from typing import TYPE_CHECKING


class DetectTopologicalHybridLatchMixin:
    def _detect_topological_hybrid_latch_candidates(
        self,
        transistors: list[_Transistor],
        inverter_map: dict[tuple[str, str], dict],
        tri_state_inverters: list[dict],
        switches: list[dict],
        power_nets: set[str],
        ground_nets: set[str],
        nets_by_canonical: dict[str, list[str]],
        output_nets: set[str],
    ) -> list[dict]:
        all_hybrid_core_nets: set[str] = set()
        for tri in tri_state_inverters:
            all_hybrid_core_nets.add(str(tri.get("input", "")))
            all_hybrid_core_nets.add(str(tri.get("output", "")))

        terminal_degree = self._terminal_degree(transistors)
        candidates: list[dict] = []
        seen_cores: set[tuple[str, str, str, str]] = set()

        for tri in tri_state_inverters:
            core_input = str(tri.get("input") or "")
            core_output = str(tri.get("output") or "")
            if not core_input or not core_output or core_input == core_output:
                continue

            static_inv = tri.get("static_inverter") or {}
            regen_phase = tri.get("phase") or {}
            regen_phase_complement = tri.get("phase_complement") or {}
            if not isinstance(regen_phase, dict):
                continue
            core_key = tuple(sorted((core_input, core_output), key=self._natural_key))
            dedup_key = (
                core_key[0],
                core_key[1],
                str(regen_phase.get("net") or ""),
                str(regen_phase.get("level") or ""),
            )
            if dedup_key in seen_cores:
                continue
            seen_cores.add(dedup_key)

            tri_tx_indexes = set(int(idx) for idx in tri.get("transistor_indexes", []))
            transparency_switches: list[dict] = []
            for switch in switches:
                switch_phase = switch.get("primary_phase")
                if self._phase_relation(switch_phase, regen_phase, inverter_map) != "opposite":
                    continue
                if tri_tx_indexes & set(int(idx) for idx in switch.get("transistor_indexes", [])):
                    continue
                switch_nets = set(switch.get("nets", ()))
                if not switch_nets & {core_input, core_output}:
                    continue
                external_nets = [
                    net_name
                    for net_name in switch.get("nets", ())
                    if net_name not in {core_input, core_output}
                    and net_name not in power_nets
                    and net_name not in ground_nets
                ]
                if not external_nets:
                    continue
                transparency_switches.append(switch)

            if not transparency_switches:
                continue

            def transparency_score(switch: dict) -> tuple[int, int, int, int]:
                ext_nets = [
                    net_name
                    for net_name in switch.get("nets", ())
                    if net_name not in {core_input, core_output}
                    and net_name not in power_nets
                    and net_name not in ground_nets
                ]
                noncore_ext = [net_name for net_name in ext_nets if net_name not in all_hybrid_core_nets]
                touches_output_net = any(net_name in output_nets for net_name in switch.get("nets", ()))
                return (
                    1 if noncore_ext else 0,
                    1 if switch.get("kind") == "transmission_gate" else 0,
                    1 if touches_output_net else 0,
                    len(ext_nets),
                )

            transparency_switches.sort(key=transparency_score, reverse=True)
            primary_transparency_switch = transparency_switches[0]
            enable_phase = dict(primary_transparency_switch.get("primary_phase") or {})
            if not enable_phase:
                continue

            data_nets: list[str] = []
            for switch in transparency_switches:
                for net_name in switch.get("nets", ()):
                    if net_name in {core_input, core_output} or net_name in power_nets or net_name in ground_nets:
                        continue
                    if net_name not in data_nets:
                        data_nets.append(net_name)
            if not data_nets:
                continue
            data_nets.sort(key=self._natural_key)

            core_nets = {core_input, core_output}
            involved_instances = self._unique(
                list(static_inv.get("instances", []))
                + list(tri.get("instances", []))
                + [
                    instance_name
                    for switch in transparency_switches
                    for instance_name in switch.get("instances", [])
                ]
            )
            involved_nets: set[str] = set(core_nets)
            involved_nets.update(data_nets)
            if str(enable_phase.get("net") or ""):
                involved_nets.add(str(enable_phase["net"]))
            if str(regen_phase.get("net") or ""):
                involved_nets.add(str(regen_phase["net"]))
            if str(regen_phase_complement.get("net") or ""):
                involved_nets.add(str(regen_phase_complement["net"]))
            for switch in transparency_switches:
                involved_nets.update(net_name for net_name in switch.get("nets", ()) if isinstance(net_name, str))
                for control in switch.get("controls", []):
                    control_net = str(control.get("net") or "")
                    if control_net:
                        involved_nets.add(control_net)

            def core_output_score(net_name: str) -> tuple[int, int, int]:
                touches_output = 1 if net_name in output_nets else 0
                touches_switch = 0
                for switch in transparency_switches:
                    if net_name in set(switch.get("nets", ())):
                        touches_switch += 1
                return (touches_output, touches_switch, terminal_degree.get(net_name, 0))

            ordered_core = sorted([core_output, core_input], key=lambda net_name: core_output_score(net_name), reverse=True)
            primary_output_net = ordered_core[0]
            secondary_output_net = ordered_core[1]
            output_entries = [
                {"port": None, "net": primary_output_net, "polarity": "non_inverted"},
                {"port": None, "net": secondary_output_net, "polarity": "inverted"},
            ]
            data_entries = [{"port": None, "net": net_name} for net_name in data_nets]

            control_signals: list[dict] = [
                {
                    "role": "enable",
                    "port": None,
                    "net": str(enable_phase.get("net") or ""),
                    "activation": {"mode": "level", "level": str(enable_phase.get("level") or "high")},
                },
                {
                    "role": "hold_phase",
                    "port": None,
                    "net": str(regen_phase.get("net") or ""),
                    "activation": {"mode": "level", "level": str(regen_phase.get("level") or "high")},
                },
            ]
            if str(regen_phase_complement.get("net") or ""):
                control_signals.append(
                    {
                        "role": "phase_complement",
                        "port": None,
                        "net": str(regen_phase_complement.get("net") or ""),
                        "activation": {"mode": "level", "level": str(regen_phase_complement.get("level") or "low")},
                    }
                )
            control_signals = self._unique_seq_entries(control_signals)

            control_nets = [item["net"] for item in control_signals if item.get("net")]
            output_nets_local = [item["net"] for item in output_entries if item.get("net")]
            data_nets_local = [item["net"] for item in data_entries if item.get("net")]
            involved_tx_indexes = sorted(
                self._unique(
                    list(static_inv.get("transistor_indexes", []))
                    + list(tri.get("transistor_indexes", []))
                    + [
                        idx
                        for switch in transparency_switches
                        for idx in switch.get("transistor_indexes", [])
                    ]
                )
            )
            all_nets_canonical = sorted(self._unique(list(involved_nets)), key=self._natural_key)
            expanded_all_nets = sorted(self._expanded_aliases(all_nets_canonical, nets_by_canonical), key=self._natural_key)
            expanded_control_nets = sorted(self._expanded_aliases(control_nets, nets_by_canonical), key=self._natural_key)
            expanded_output_nets = sorted(self._expanded_aliases(output_nets_local, nets_by_canonical), key=self._natural_key)
            expanded_data_nets = sorted(self._expanded_aliases(data_nets_local, nets_by_canonical), key=self._natural_key)

            candidates.append(
                {
                    "_core_nets": sorted(core_nets, key=self._natural_key),
                    "_enable": {
                        "net": str(enable_phase.get("net") or ""),
                        "level": str(enable_phase.get("level") or "high"),
                        "complement_nets": [
                            net_name
                            for net_name in [
                                str(regen_phase.get("net") or ""),
                                str(regen_phase_complement.get("net") or ""),
                            ]
                            if net_name and net_name != str(enable_phase.get("net") or "")
                        ],
                    },
                    "_data_nets": data_nets_local,
                    "_output_nets": output_nets_local,
                    "_all_instances": sorted(self._unique(involved_instances), key=self._natural_key),
                    "_all_nets_canonical": all_nets_canonical,
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
                        "level": str(enable_phase.get("level") or "high"),
                        "port": None,
                        "net": str(enable_phase.get("net") or ""),
                    },
                    "logic_path": {
                        "nets": expanded_all_nets,
                        "instances": sorted(self._unique(involved_instances), key=self._natural_key),
                    },
                    "highlight": {
                        "nets": expanded_all_nets,
                        "instances": sorted(self._unique(involved_instances), key=self._natural_key),
                        "control_nets": expanded_control_nets,
                        "data_nets": expanded_data_nets,
                        "output_nets": expanded_output_nets,
                    },
                }
            )

        return candidates

if TYPE_CHECKING:
    from krona.application.logic_functions_parts.transistor import _Transistor
