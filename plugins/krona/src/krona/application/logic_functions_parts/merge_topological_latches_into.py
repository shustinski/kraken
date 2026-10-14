"""logic_functions: merge topological latches into."""
from __future__ import annotations


class MergeTopologicalLatchesIntoMixin:
    def _merge_topological_latches_into_flip_flops(
        self,
        latches: list[dict],
        inverter_map: dict[tuple[str, str], dict],
        nets_by_canonical: dict[str, list[str]],
        output_nets: set[str],
    ) -> tuple[list[dict], set[int]]:
        pair_options: list[dict] = []
        for left_idx in range(len(latches)):
            left = latches[left_idx]
            left_enable = left.get("_enable") or {}
            for right_idx in range(left_idx + 1, len(latches)):
                right = latches[right_idx]
                right_enable = right.get("_enable") or {}
                if not self._controls_are_opposite_phase(left_enable, right_enable, inverter_map):
                    continue

                left_to_right = self._nets_share_or_invert_path(
                    set(left.get("_output_nets", [])),
                    set(right.get("_data_nets", [])),
                    inverter_map,
                )
                right_to_left = self._nets_share_or_invert_path(
                    set(right.get("_output_nets", [])),
                    set(left.get("_data_nets", [])),
                    inverter_map,
                )
                if left_to_right == 0 and right_to_left == 0:
                    continue
                if left_to_right > right_to_left:
                    master_idx, slave_idx, direction_score = left_idx, right_idx, left_to_right
                elif right_to_left > left_to_right:
                    master_idx, slave_idx, direction_score = right_idx, left_idx, right_to_left
                else:
                    left_hits_output = bool(set(left.get("_output_nets", [])) & output_nets)
                    right_hits_output = bool(set(right.get("_output_nets", [])) & output_nets)
                    if right_hits_output and not left_hits_output:
                        master_idx, slave_idx, direction_score = left_idx, right_idx, left_to_right
                    elif left_hits_output and not right_hits_output:
                        master_idx, slave_idx, direction_score = right_idx, left_idx, right_to_left
                    else:
                        continue

                slave = latches[slave_idx]
                slave_enable = slave.get("_enable") or {}
                edge = "rising" if slave_enable.get("level") == "high" else "falling"
                pair_options.append(
                    {
                        "master_idx": master_idx,
                        "slave_idx": slave_idx,
                        "score": (
                            direction_score,
                            1 if set(slave.get("_output_nets", [])) & output_nets else 0,
                        ),
                        "edge": edge,
                    }
                )

        pair_options.sort(key=lambda item: item["score"], reverse=True)
        used: set[int] = set()
        ff_elements: list[dict] = []
        ff_counter = 1

        for option in pair_options:
            master_idx = int(option["master_idx"])
            slave_idx = int(option["slave_idx"])
            if master_idx in used or slave_idx in used:
                continue
            master = latches[master_idx]
            slave = latches[slave_idx]
            master_enable = master.get("_enable") or {}
            slave_enable = slave.get("_enable") or {}
            if not self._controls_are_opposite_phase(master_enable, slave_enable, inverter_map):
                continue

            clock_net = str(slave_enable.get("net") or master_enable.get("net") or "")
            if not clock_net:
                continue
            edge = str(option["edge"])

            union_instances = sorted(
                self._unique(list(master.get("_all_instances", [])) + list(slave.get("_all_instances", []))),
                key=self._natural_key,
            )
            union_nets_canonical = sorted(
                self._unique(list(master.get("_all_nets_canonical", [])) + list(slave.get("_all_nets_canonical", []))),
                key=self._natural_key,
            )
            union_transistor_indexes = sorted(
                self._unique(list(master.get("_transistor_indexes", [])) + list(slave.get("_transistor_indexes", [])))
            )
            expanded_all_nets = sorted(self._expanded_aliases(union_nets_canonical, nets_by_canonical), key=self._natural_key)

            ff_control_signals = [
                {
                    "role": "clock",
                    "port": None,
                    "net": clock_net,
                    "activation": {"mode": "edge", "edge": edge},
                },
                {
                    "role": "master_phase",
                    "port": None,
                    "net": str(master_enable.get("net") or ""),
                    "activation": {"mode": "level", "level": str(master_enable.get("level") or "high")},
                },
                {
                    "role": "slave_phase",
                    "port": None,
                    "net": str(slave_enable.get("net") or ""),
                    "activation": {"mode": "level", "level": str(slave_enable.get("level") or "high")},
                },
            ]

            # Keep additional non-enable controls from both latches.
            for source_label, latch in (("master", master), ("slave", slave)):
                for control in latch.get("control_signals", []):
                    role = str(control.get("role") or "")
                    if role == "enable":
                        continue
                    net_name = str(control.get("net") or "")
                    if not net_name:
                        continue
                    ff_control_signals.append(
                        {
                            "role": f"{source_label}_{role}",
                            "port": None,
                            "net": net_name,
                            "activation": dict(control.get("activation") or {}),
                        }
                    )
            ff_control_signals = self._unique_seq_entries(ff_control_signals)

            outputs = [dict(item) for item in (slave.get("outputs") or [])]
            data_inputs = [dict(item) for item in (master.get("data_inputs") or [])]

            ff_output_nets = [str(item.get("net")) for item in outputs if item.get("net")]
            ff_data_nets = [str(item.get("net")) for item in data_inputs if item.get("net")]
            ff_control_nets = [str(item.get("net")) for item in ff_control_signals if item.get("net")]

            ff_elements.append(
                {
                    "_core_nets": sorted(
                        self._unique(list(master.get("_core_nets", [])) + list(slave.get("_core_nets", []))),
                        key=self._natural_key,
                    ),
                    "_all_nets_canonical": union_nets_canonical,
                    "_transistor_indexes": union_transistor_indexes,
                    "instance": f"FF{ff_counter}",
                    "cell": None,
                    "view": None,
                    "library": None,
                    "designator": None,
                    "kind": "flip_flop",
                    "subtype": "D",
                    "outputs": outputs,
                    "data_inputs": data_inputs,
                    "control_signals": ff_control_signals,
                    "triggering": {
                        "mode": "edge",
                        "edge": edge,
                        "port": None,
                        "net": clock_net,
                    },
                    "logic_path": {
                        "nets": expanded_all_nets,
                        "instances": union_instances,
                    },
                    "highlight": {
                        "nets": expanded_all_nets,
                        "instances": union_instances,
                        "control_nets": sorted(self._expanded_aliases(ff_control_nets, nets_by_canonical), key=self._natural_key),
                        "data_nets": sorted(self._expanded_aliases(ff_data_nets, nets_by_canonical), key=self._natural_key),
                        "output_nets": sorted(self._expanded_aliases(ff_output_nets, nets_by_canonical), key=self._natural_key),
                    },
                }
            )
            ff_counter += 1
            used.add(master_idx)
            used.add(slave_idx)

        return ff_elements, used
