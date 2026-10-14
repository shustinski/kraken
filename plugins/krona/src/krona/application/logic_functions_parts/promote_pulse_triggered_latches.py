"""logic_functions: promote pulse triggered latches."""
from __future__ import annotations


class PromotePulseTriggeredLatchesMixin:
    def _promote_pulse_triggered_latches_to_flip_flops(
        self,
        latches: list[dict],
        used_latch_indexes: set[int],
        degenerate_transistor_markers: list[dict],
        power_nets: set[str],
        ground_nets: set[str],
        nets_by_canonical: dict[str, list[str]],
    ) -> tuple[list[dict], set[int]]:
        if not latches or not degenerate_transistor_markers:
            return [], set()

        markers_by_gate: dict[str, list[dict]] = {}
        for marker in degenerate_transistor_markers:
            gate_net = str(marker.get("gate") or "")
            if not gate_net:
                continue
            markers_by_gate.setdefault(gate_net, []).append(marker)

        promoted: list[dict] = []
        used: set[int] = set()
        ff_counter = 1
        for idx, latch in enumerate(latches):
            if idx in used_latch_indexes:
                continue
            if str(latch.get("kind") or "") != "latch" or str(latch.get("subtype") or "") != "D":
                continue

            enable = latch.get("_enable") or {}
            enable_net = str(enable.get("net") or "")
            enable_level = str(enable.get("level") or "")
            if not enable_net or enable_level not in {"high", "low"}:
                continue

            control_signals = [dict(item) for item in (latch.get("control_signals") or []) if isinstance(item, dict)]
            roles = {str(item.get("role") or ""): item for item in control_signals}
            enable_control = roles.get("enable")
            hold_control = roles.get("hold_phase")
            if not isinstance(enable_control, dict) or not isinstance(hold_control, dict):
                continue
            if str(enable_control.get("net") or "") != enable_net:
                continue
            if str(hold_control.get("net") or "") != enable_net:
                continue
            hold_activation = hold_control.get("activation") if isinstance(hold_control.get("activation"), dict) else {}
            hold_level = str(hold_activation.get("level") or "")
            if hold_level not in {"high", "low"} or hold_level == enable_level:
                continue

            markers = list(markers_by_gate.get(enable_net, []))
            if len(markers) < 2:
                continue
            has_rail_bridge = any(str(marker.get("style") or "") == "rail_bridge" for marker in markers)
            if not has_rail_bridge:
                continue

            local_shorted_markers = [
                marker
                for marker in markers
                if str(marker.get("style") or "") == "sd_shorted"
                and str(marker.get("node") or "")
                and str(marker.get("node") or "") not in power_nets
                and str(marker.get("node") or "") not in ground_nets
            ]
            if not local_shorted_markers:
                continue
            marker_kinds = {str(marker.get("kind") or "") for marker in markers if marker.get("kind")}
            if len(marker_kinds) < 2:
                continue

            edge = "rising" if enable_level == "high" else "falling"
            ff_control_signals = [
                {
                    "role": "clock",
                    "port": None,
                    "net": enable_net,
                    "activation": {"mode": "edge", "edge": edge},
                }
            ]
            ff_control_signals.extend(control_signals)
            ff_control_signals = self._unique_seq_entries(ff_control_signals)

            ff_element = {key: value for key, value in latch.items()}
            ff_element["instance"] = f"PULSE_FF{ff_counter}"
            ff_element["kind"] = "flip_flop"
            ff_element["subtype"] = "D"
            ff_element["control_signals"] = ff_control_signals
            ff_element["triggering"] = {
                "mode": "edge",
                "edge": edge,
                "port": None,
                "net": enable_net,
            }
            ff_element["topology"] = {
                "kind": "pulse_triggered_single_latch",
                "clock_net": enable_net,
                "pulse_level": enable_level,
                "evidence": {
                    "degenerate_mos_count": len(markers),
                    "local_clocked_mos_caps": [
                        {
                            "instance": str(marker.get("instance") or ""),
                            "kind": str(marker.get("kind") or ""),
                            "node": str(marker.get("node") or ""),
                        }
                        for marker in local_shorted_markers
                    ],
                    "rail_bridge_instances": sorted(
                        [
                            str(marker.get("instance") or "")
                            for marker in markers
                            if str(marker.get("style") or "") == "rail_bridge"
                        ],
                        key=self._natural_key,
                    ),
                },
            }

            marker_instances = [str(marker.get("instance") or "") for marker in markers if marker.get("instance")]
            marker_nets_canonical: list[str] = []
            for marker in markers:
                gate_net = str(marker.get("gate") or "")
                if gate_net:
                    marker_nets_canonical.append(gate_net)
                for net_name in marker.get("terminals", []) or []:
                    if isinstance(net_name, str) and net_name:
                        marker_nets_canonical.append(net_name)
            marker_nets_canonical = sorted(self._unique(marker_nets_canonical), key=self._natural_key)
            expanded_marker_nets = sorted(self._expanded_aliases(marker_nets_canonical, nets_by_canonical), key=self._natural_key)

            ff_element["_all_instances"] = sorted(
                self._unique([str(name) for name in (ff_element.get("_all_instances") or [])] + marker_instances),
                key=self._natural_key,
            )
            ff_element["_all_nets_canonical"] = sorted(
                self._unique([str(net) for net in (ff_element.get("_all_nets_canonical") or [])] + marker_nets_canonical),
                key=self._natural_key,
            )

            logic_path = ff_element.get("logic_path")
            if isinstance(logic_path, dict):
                logic_path["instances"] = sorted(
                    self._unique([str(name) for name in (logic_path.get("instances") or [])] + marker_instances),
                    key=self._natural_key,
                )
                logic_path["nets"] = sorted(
                    self._unique([str(net) for net in (logic_path.get("nets") or [])] + expanded_marker_nets),
                    key=self._natural_key,
                )

            highlight = ff_element.get("highlight")
            if isinstance(highlight, dict):
                highlight["instances"] = sorted(
                    self._unique([str(name) for name in (highlight.get("instances") or [])] + marker_instances),
                    key=self._natural_key,
                )
                highlight["nets"] = sorted(
                    self._unique([str(net) for net in (highlight.get("nets") or [])] + expanded_marker_nets),
                    key=self._natural_key,
                )
                highlight["control_nets"] = sorted(
                    self._unique([str(net) for net in (highlight.get("control_nets") or [])] + expanded_marker_nets),
                    key=self._natural_key,
                )

            promoted.append(ff_element)
            used.add(idx)
            ff_counter += 1

        return promoted, used
