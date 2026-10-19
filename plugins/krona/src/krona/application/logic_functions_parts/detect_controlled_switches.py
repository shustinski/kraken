"""logic_functions: detect controlled switches."""
from __future__ import annotations
from typing import TYPE_CHECKING


class DetectControlledSwitchesMixin:
    def _detect_controlled_switches(
        self,
        transistors: list[_Transistor],
        power_nets: set[str],
        ground_nets: set[str],
    ) -> list[dict]:
        grouped: dict[tuple[str, str], dict] = {}
        for idx, transistor in enumerate(transistors):
            terminals = self._unique([transistor.source, transistor.drain])
            if len(terminals) != 2:
                continue
            if any(net in power_nets or net in ground_nets for net in terminals):
                continue
            left, right = sorted(terminals, key=self._natural_key)
            bucket = grouped.setdefault(
                (left, right),
                {
                    "nets": (left, right),
                    "nmos": [],
                    "pmos": [],
                },
            )
            bucket[transistor.kind].append(
                {
                    "index": idx,
                    "gate": transistor.gate,
                    "instance": transistor.instance_name,
                }
            )

        switches: list[dict] = []
        for bucket in grouped.values():
            nmos_items = list(bucket.get("nmos", []))
            pmos_items = list(bucket.get("pmos", []))
            if not nmos_items and not pmos_items:
                continue

            nmos_gates = sorted(self._unique([item["gate"] for item in nmos_items]), key=self._natural_key)
            pmos_gates = sorted(self._unique([item["gate"] for item in pmos_items]), key=self._natural_key)
            nmos_gate_states = {
                net_name: self._constant_gate_conduction_state("nmos", net_name, power_nets, ground_nets)
                for net_name in nmos_gates
            }
            pmos_gate_states = {
                net_name: self._constant_gate_conduction_state("pmos", net_name, power_nets, ground_nets)
                for net_name in pmos_gates
            }
            controllable_nmos_gates = [net_name for net_name in nmos_gates if nmos_gate_states.get(net_name) is None]
            controllable_pmos_gates = [net_name for net_name in pmos_gates if pmos_gate_states.get(net_name) is None]
            constant_on_nmos_gates = [net_name for net_name in nmos_gates if nmos_gate_states.get(net_name) is True]
            constant_on_pmos_gates = [net_name for net_name in pmos_gates if pmos_gate_states.get(net_name) is True]
            constant_off_nmos_gates = [net_name for net_name in nmos_gates if nmos_gate_states.get(net_name) is False]
            constant_off_pmos_gates = [net_name for net_name in pmos_gates if pmos_gate_states.get(net_name) is False]
            controls: list[dict] = []
            for net_name in nmos_gates:
                constant_state = nmos_gate_states.get(net_name)
                if constant_state is None:
                    controls.append(
                        {
                            "net": net_name,
                            "kind": "nmos",
                            "activation": {"mode": "level", "level": "high"},
                        }
                    )
                else:
                    controls.append(
                        {
                            "net": net_name,
                            "kind": "nmos",
                            "activation": {"mode": "constant", "state": "on" if constant_state else "off"},
                        }
                    )
            for net_name in pmos_gates:
                constant_state = pmos_gate_states.get(net_name)
                if constant_state is None:
                    controls.append(
                        {
                            "net": net_name,
                            "kind": "pmos",
                            "activation": {"mode": "level", "level": "low"},
                        }
                    )
                else:
                    controls.append(
                        {
                            "net": net_name,
                            "kind": "pmos",
                            "activation": {"mode": "constant", "state": "on" if constant_state else "off"},
                        }
                    )

            primary_phase: dict | None = None
            if len(controllable_nmos_gates) == 1:
                primary_phase = {"net": controllable_nmos_gates[0], "level": "high", "kind": "nmos"}
            elif len(controllable_nmos_gates) == 0 and len(controllable_pmos_gates) == 1:
                primary_phase = {"net": controllable_pmos_gates[0], "level": "low", "kind": "pmos"}

            switches.append(
                {
                    "nets": bucket["nets"],
                    "instances": sorted(
                        self._unique([item["instance"] for item in nmos_items + pmos_items]),
                        key=self._natural_key,
                    ),
                    "transistor_indexes": [item["index"] for item in nmos_items + pmos_items],
                    "nmos_gates": nmos_gates,
                    "pmos_gates": pmos_gates,
                    "controllable_nmos_gates": controllable_nmos_gates,
                    "controllable_pmos_gates": controllable_pmos_gates,
                    "constant_on_nmos_gates": constant_on_nmos_gates,
                    "constant_on_pmos_gates": constant_on_pmos_gates,
                    "constant_off_nmos_gates": constant_off_nmos_gates,
                    "constant_off_pmos_gates": constant_off_pmos_gates,
                    "controls": controls,
                    "primary_phase": primary_phase,
                    "kind": (
                        "transmission_gate"
                        if nmos_items and pmos_items
                        else ("pass_nmos" if nmos_items else "pass_pmos")
                    ),
                }
            )

        return sorted(switches, key=lambda item: (self._natural_key(item["nets"][0]), self._natural_key(item["nets"][1])))

if TYPE_CHECKING:
    from krona.application.logic_functions_parts.transistor import _Transistor
