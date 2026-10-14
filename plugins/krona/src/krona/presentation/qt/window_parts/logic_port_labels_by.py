"""window: logic port labels by."""
from __future__ import annotations


class LogicPortLabelsByMixin:
    def _format_logic_activation(self, activation: dict | None) -> str:
        if not isinstance(activation, dict):
            return self._t("logic.label.none", "-")
        mode = str(activation.get("mode") or "").lower()
        if mode == "edge":
            edge = str(activation.get("edge") or "unknown").lower()
            if edge == "rising":
                return self._t("logic.activation.rising_edge", "rising edge")
            if edge == "falling":
                return self._t("logic.activation.falling_edge", "falling edge")
            return self._t("logic.activation.edge_unknown", "edge")
        if mode == "level":
            level = str(activation.get("level") or "unknown").lower()
            if level == "high":
                return self._t("logic.activation.high_level", "high level")
            if level == "low":
                return self._t("logic.activation.low_level", "low level")
            return self._t("logic.activation.level_unknown", "level")
        return self._t("logic.label.none", "-")

    def _logic_port_labels_by_net(self) -> dict[str, list[str]]:
        if not isinstance(self._logic_data, dict):
            return {}
        labels_by_net: dict[str, list[str]] = {}

        def add(net_name: str, label: str) -> None:
            if not net_name or not label:
                return
            labels = labels_by_net.setdefault(str(net_name), [])
            if label not in labels:
                labels.append(label)

        meta = self._logic_data.get("meta", {})
        for input_name, net_name in (meta.get("input_nets", {}) or {}).items():
            add(str(net_name), str(input_name))
        outputs = self._logic_data.get("outputs", {})
        for output_name, output_data in outputs.items():
            if not isinstance(output_data, dict):
                continue
            add(str(output_data.get("net") or ""), str(output_name))
        return labels_by_net

    def _logic_output_path_labels_by_net(self) -> dict[str, list[str]]:
        if not isinstance(self._logic_data, dict):
            return {}
        labels_by_net: dict[str, list[str]] = {}
        outputs = self._logic_data.get("outputs", {})
        for output_name, output_data in outputs.items():
            if not isinstance(output_data, dict):
                continue
            logic_path = output_data.get("logic_path", {})
            if not isinstance(logic_path, dict):
                continue
            for net_name in logic_path.get("nets", []) or []:
                if not net_name:
                    continue
                labels = labels_by_net.setdefault(str(net_name), [])
                label = str(output_name)
                if label not in labels:
                    labels.append(label)
        return labels_by_net

    def _logic_display_net_name(self, net_name: str, *, prefer_output_path: bool = False) -> str:
        if not net_name:
            return net_name
        direct_labels = self._logic_port_labels_by_net().get(str(net_name), [])
        if direct_labels:
            return "/".join(direct_labels)
        if prefer_output_path:
            output_path_labels = self._logic_output_path_labels_by_net().get(str(net_name), [])
            if output_path_labels:
                return "/".join(output_path_labels) + "~"
        return net_name

    @staticmethod
    def _logic_function_expression(function_data: dict | None) -> str | None:
        if not isinstance(function_data, dict):
            return None
        mux = function_data.get("mux")
        if isinstance(mux, dict):
            mux_expr = str(mux.get("expression") or "").strip()
            if mux_expr:
                return mux_expr
        for key in ("expression", "simplified_expression", "sum_of_products"):
            value = str(function_data.get(key) or "").strip()
            if value:
                return value
        return None
