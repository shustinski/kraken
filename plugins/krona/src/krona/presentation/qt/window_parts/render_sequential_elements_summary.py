"""window: render sequential elements summary."""
from __future__ import annotations


class RenderSequentialElementsSummaryMixin:
    def _render_sequential_elements_summary(self, elements: list[dict]) -> str:
        if not elements:
            return self._t("logic.label.none", "-")
        parts: list[str] = []
        for element in elements:
            instance_name = str(element.get("instance") or "?")
            kind_name = str(element.get("kind") or "sequential")
            subtype = str(element.get("subtype") or "").strip()
            if subtype and subtype != "unknown":
                type_label = f"{subtype}-{kind_name}"
            else:
                type_label = kind_name

            trigger_info = element.get("triggering")
            trigger_text = ""
            if isinstance(trigger_info, dict):
                trigger_mode = str(trigger_info.get("mode") or "").lower()
                trigger_net = str(trigger_info.get("net") or "")
                trigger_net_label = self._logic_display_net_name(trigger_net)
                trigger_function_text = self._logic_function_expression(trigger_info.get("function")) if isinstance(trigger_info.get("function"), dict) else None
                if trigger_function_text and (not trigger_net_label or trigger_net_label == trigger_net):
                    trigger_net_label = trigger_function_text
                if trigger_mode == "edge":
                    edge = self._format_logic_activation({"mode": "edge", "edge": trigger_info.get("edge")})
                    if trigger_net_label:
                        trigger_text = f", CLK={trigger_net_label} ({edge})"
                    else:
                        trigger_text = f", {edge}"
                elif trigger_mode == "level":
                    level = self._format_logic_activation({"mode": "level", "level": trigger_info.get("level")})
                    if trigger_net_label:
                        trigger_text = f", EN={trigger_net_label} ({level})"
                    else:
                        trigger_text = f", {level}"

            data_parts: list[str] = []
            for idx, data_input in enumerate(element.get("data_inputs", []), start=1):
                net_name = str(data_input.get("net") or "")
                if not net_name:
                    continue
                net_label = self._logic_display_net_name(net_name)
                function_expr = self._logic_function_expression(data_input.get("function"))
                signal_label = "D" if idx == 1 else f"D{idx}"
                if function_expr:
                    data_parts.append(f"{signal_label}={function_expr}")
                else:
                    data_parts.append(f"{signal_label}={net_label}")
            data_text = f", {'; '.join(data_parts)}" if data_parts else ""

            control_parts: list[str] = []
            for control in element.get("control_signals", []):
                role = str(control.get("role") or "")
                if role in {"clock", "enable"}:
                    continue
                net_name = str(control.get("net") or "")
                if not net_name:
                    continue
                net_label = self._logic_display_net_name(net_name)
                function_expr = self._logic_function_expression(control.get("function"))
                if function_expr and (not control.get("labels")) and net_label == net_name:
                    net_label = function_expr
                activation_text = self._format_logic_activation(control.get("activation"))
                control_parts.append(f"{role}={net_label} ({activation_text})")
            control_text = f", {'; '.join(control_parts)}" if control_parts else ""

            outputs: list[str] = []
            for item in element.get("outputs", []):
                if not item.get("net"):
                    continue
                mapped_name = self._logic_display_net_name(str(item.get("net") or ""), prefer_output_path=True)
                if mapped_name and mapped_name not in outputs:
                    outputs.append(mapped_name)
            output_text = f", Q={','.join(outputs)}" if outputs else ""

            parts.append(f"{instance_name}: {type_label}{trigger_text}{data_text}{control_text}{output_text}")
        return " | ".join(parts)
