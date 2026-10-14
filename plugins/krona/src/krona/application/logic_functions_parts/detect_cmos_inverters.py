"""logic_functions: detect cmos inverters."""
from __future__ import annotations


class DetectCmosInvertersMixin:
    def _two_transistor_paths_to_supply(
        self,
        transistors: list[_Transistor],
        kind: str,
        start_net: str,
        target_nets: set[str],
    ) -> list[dict]:
        items = [(idx, transistor) for idx, transistor in enumerate(transistors) if transistor.kind == kind]
        paths: list[dict] = []
        for first_idx, first in items:
            if start_net not in {first.source, first.drain}:
                continue
            mid = first.drain if first.source == start_net else first.source
            if mid in target_nets:
                continue
            for second_idx, second in items:
                if second_idx == first_idx:
                    continue
                if mid not in {second.source, second.drain}:
                    continue
                other = second.drain if second.source == mid else second.source
                if other not in target_nets:
                    continue
                paths.append(
                    {
                        "transistor_indexes": [first_idx, second_idx],
                        "gates": [first.gate, second.gate],
                        "instances": [first.instance_name, second.instance_name],
                        "mid_net": mid,
                    }
                )
        return paths

    @staticmethod
    def _constant_gate_conduction_state(
        transistor_kind: str,
        gate_net: str,
        power_nets: set[str],
        ground_nets: set[str],
    ) -> bool | None:
        if gate_net in power_nets:
            return transistor_kind == "nmos"
        if gate_net in ground_nets:
            return transistor_kind == "pmos"
        return None

    def _detect_cmos_inverters(
        self,
        transistors: list[_Transistor],
        power_nets: set[str],
        ground_nets: set[str],
    ) -> dict[tuple[str, str], dict]:
        grouped: dict[tuple[str, str], dict] = {}
        for idx, transistor in enumerate(transistors):
            terminals = [transistor.source, transistor.drain]
            if transistor.kind == "pmos":
                if (terminals[0] in power_nets) == (terminals[1] in power_nets):
                    continue
                output_net = terminals[1] if terminals[0] in power_nets else terminals[0]
            else:
                if (terminals[0] in ground_nets) == (terminals[1] in ground_nets):
                    continue
                output_net = terminals[1] if terminals[0] in ground_nets else terminals[0]
            key = (transistor.gate, output_net)
            bucket = grouped.setdefault(
                key,
                {
                    "input": transistor.gate,
                    "output": output_net,
                    "pmos_indexes": [],
                    "nmos_indexes": [],
                    "instances": [],
                },
            )
            bucket[f"{transistor.kind}_indexes"].append(idx)
            bucket["instances"].append(transistor.instance_name)

        out: dict[tuple[str, str], dict] = {}
        for key, bucket in grouped.items():
            if not bucket["pmos_indexes"] or not bucket["nmos_indexes"]:
                continue
            out[key] = {
                "input": bucket["input"],
                "output": bucket["output"],
                "pmos_indexes": list(bucket["pmos_indexes"]),
                "nmos_indexes": list(bucket["nmos_indexes"]),
                "transistor_indexes": list(bucket["pmos_indexes"]) + list(bucket["nmos_indexes"]),
                "instances": self._unique(list(bucket["instances"])),
            }
        return out
