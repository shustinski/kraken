"""logic_functions: detect tristate cmos inverters."""
from __future__ import annotations
from typing import TYPE_CHECKING


class DetectTristateCmosInvertersMixin:
    def _detect_tristate_cmos_inverters(
        self,
        transistors: list[_Transistor],
        power_nets: set[str],
        ground_nets: set[str],
        inverter_map: dict[tuple[str, str], dict],
    ) -> list[dict]:
        records: list[dict] = []
        seen: set[tuple[str, str, str, str, tuple[int, ...]]] = set()

        for (static_input, static_output), static_inv in inverter_map.items():
            n_paths = self._two_transistor_paths_to_supply(
                transistors=transistors,
                kind="nmos",
                start_net=static_input,
                target_nets=ground_nets,
            )
            p_paths = self._two_transistor_paths_to_supply(
                transistors=transistors,
                kind="pmos",
                start_net=static_input,
                target_nets=power_nets,
            )
            if not n_paths or not p_paths:
                continue

            for n_path in n_paths:
                if n_path["gates"].count(static_output) != 1:
                    continue
                n_phase_gates = [gate for gate in n_path["gates"] if gate != static_output]
                if len(n_phase_gates) != 1:
                    continue
                n_phase_net = n_phase_gates[0]
                n_extra_state = self._constant_gate_conduction_state("nmos", n_phase_net, power_nets, ground_nets)
                if n_extra_state is False:
                    continue
                for p_path in p_paths:
                    if p_path["gates"].count(static_output) != 1:
                        continue
                    p_phase_gates = [gate for gate in p_path["gates"] if gate != static_output]
                    if len(p_phase_gates) != 1:
                        continue
                    p_phase_net = p_phase_gates[0]
                    p_extra_state = self._constant_gate_conduction_state("pmos", p_phase_net, power_nets, ground_nets)
                    if p_extra_state is False:
                        continue

                    phase: dict | None = None
                    phase_complement: dict | None = None
                    topology_style = "tristate"
                    if n_extra_state is None and p_extra_state is None:
                        if n_phase_net == p_phase_net:
                            phase = {"net": n_phase_net, "level": "high"}
                            phase_complement = {"net": p_phase_net, "level": "low"}
                            topology_style = "tristate_same_clock"
                        elif self._nets_are_inverted(n_phase_net, p_phase_net, inverter_map):
                            phase = {"net": n_phase_net, "level": "high"}
                            phase_complement = {"net": p_phase_net, "level": "low"}
                        else:
                            continue
                    elif n_extra_state is None and p_extra_state is True:
                        phase = {"net": n_phase_net, "level": "high"}
                        phase_complement = None
                        topology_style = "pseudo_tristate_n_phase"
                    elif n_extra_state is True and p_extra_state is None:
                        phase = {"net": p_phase_net, "level": "low"}
                        phase_complement = None
                        topology_style = "pseudo_tristate_p_phase"
                    else:
                        continue
                    if phase is None:
                        continue

                    transistor_indexes = sorted(
                        self._unique(n_path["transistor_indexes"] + p_path["transistor_indexes"])
                    )
                    key = (
                        static_output,
                        static_input,
                        str(phase.get("net") or ""),
                        str(phase.get("level") or ""),
                        tuple(transistor_indexes),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    instances = self._unique(n_path["instances"] + p_path["instances"])
                    records.append(
                        {
                            "input": static_output,
                            "output": static_input,
                            "phase": phase,
                            "phase_complement": phase_complement,
                            "topology_style": topology_style,
                            "static_inverter": static_inv,
                            "nmos_path": n_path,
                            "pmos_path": p_path,
                            "n_path_extra_gate": {
                                "net": n_phase_net,
                                "constant_state": None if n_extra_state is None else ("on" if n_extra_state else "off"),
                            },
                            "p_path_extra_gate": {
                                "net": p_phase_net,
                                "constant_state": None if p_extra_state is None else ("on" if p_extra_state else "off"),
                            },
                            "transistor_indexes": transistor_indexes,
                            "instances": instances,
                        }
                    )

        return sorted(
            records,
            key=lambda item: (
                self._natural_key(str(item.get("output", ""))),
                self._natural_key(str(item.get("input", ""))),
                self._natural_key(str(item.get("phase", {}).get("net", ""))),
            ),
        )

if TYPE_CHECKING:
    from krona.application.logic_functions_parts.transistor import _Transistor
