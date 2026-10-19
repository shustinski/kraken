"""logic_functions: annotate sequential elements."""
from __future__ import annotations
from typing import TYPE_CHECKING


class AnnotateSequentialElementsMixin:
    def _annotate_sequential_elements(
        self,
        elements: list[dict],
        transistors: list[_Transistor],
        power_nets: set[str],
        ground_nets: set[str],
        inverter_map: dict[tuple[str, str], dict],
        input_net_by_var: dict[str, str],
        output_net_by_name: dict[str, str],
    ) -> None:
        if not elements:
            return

        inputs = sorted(input_net_by_var.keys(), key=self._natural_key)
        input_names_by_net: dict[str, list[str]] = {}
        for name, net_name in input_net_by_var.items():
            input_names_by_net.setdefault(net_name, []).append(name)
        output_names_by_net: dict[str, list[str]] = {}
        for name, net_name in output_net_by_name.items():
            output_names_by_net.setdefault(net_name, []).append(name)

        function_cache: dict[str, dict] = {}
        for element in elements:
            async_controls = self._detect_async_force_controls_for_element(
                element=element,
                transistors=transistors,
                power_nets=power_nets,
                ground_nets=ground_nets,
            )
            if async_controls:
                existing = list(element.get("control_signals", []))
                element["control_signals"] = self._unique_seq_entries(existing + async_controls)
                control_nets = [str(item.get("net")) for item in element["control_signals"] if item.get("net")]
                highlight = element.get("highlight")
                if isinstance(highlight, dict):
                    current_control_nets = [str(net) for net in highlight.get("control_nets", [])]
                    merged_control = self._unique(current_control_nets + control_nets)
                    highlight["control_nets"] = merged_control
                    current_nets = [str(net) for net in highlight.get("nets", [])]
                    highlight["nets"] = self._unique(current_nets + merged_control)
                logic_path = element.get("logic_path")
                if isinstance(logic_path, dict):
                    current_nets = [str(net) for net in logic_path.get("nets", [])]
                    logic_path["nets"] = self._unique(current_nets + control_nets)

            for output_entry in element.get("outputs", []) or []:
                net_name = str(output_entry.get("net") or "")
                output_entry["labels"] = sorted(self._unique(output_names_by_net.get(net_name, [])), key=self._natural_key)
                output_entry["function"] = {
                    "kind": "state",
                    "role": "Qn" if output_entry.get("polarity") == "inverted" else "Q",
                }

            for data_entry in element.get("data_inputs", []) or []:
                net_name = str(data_entry.get("net") or "")
                data_entry["labels"] = sorted(self._unique(input_names_by_net.get(net_name, [])), key=self._natural_key)
                data_entry["function"] = self._analyze_combinational_net_function(
                    target_net=net_name,
                    inputs=inputs,
                    input_net_by_var=input_net_by_var,
                    output_net_by_name=output_net_by_name,
                    power_nets=power_nets,
                    ground_nets=ground_nets,
                    transistors=transistors,
                    cache=function_cache,
                    allow_mux=True,
                )

            for control_entry in element.get("control_signals", []) or []:
                net_name = str(control_entry.get("net") or "")
                control_entry["labels"] = sorted(self._unique(input_names_by_net.get(net_name, [])), key=self._natural_key)
                control_entry["function"] = self._analyze_combinational_net_function(
                    target_net=net_name,
                    inputs=inputs,
                    input_net_by_var=input_net_by_var,
                    output_net_by_name=output_net_by_name,
                    power_nets=power_nets,
                    ground_nets=ground_nets,
                    transistors=transistors,
                    cache=function_cache,
                    allow_mux=False,
                )
            triggering = element.get("triggering")
            if isinstance(triggering, dict):
                trigger_net = str(triggering.get("net") or "")
                if trigger_net:
                    trigger_labels = sorted(self._unique(input_names_by_net.get(trigger_net, [])), key=self._natural_key)
                    if trigger_labels:
                        triggering["labels"] = trigger_labels
                    trigger_function = self._analyze_combinational_net_function(
                        target_net=trigger_net,
                        inputs=inputs,
                        input_net_by_var=input_net_by_var,
                        output_net_by_name=output_net_by_name,
                        power_nets=power_nets,
                        ground_nets=ground_nets,
                        transistors=transistors,
                        cache=function_cache,
                        allow_mux=False,
                    )
                    triggering["function"] = trigger_function

            data_inputs = list(element.get("data_inputs", []) or [])
            if len(data_inputs) == 1:
                data_function = data_inputs[0].get("function")
                if isinstance(data_function, dict):
                    element["data_function"] = data_function
                    mux_info = data_function.get("mux")
                    if isinstance(mux_info, dict):
                        element["data_path"] = {"kind": "mux2", **mux_info}

            io_map_inputs: list[dict] = []
            for data_entry in element.get("data_inputs", []) or []:
                io_map_inputs.append(
                    {
                        "role": "data",
                        "net": data_entry.get("net"),
                        "labels": data_entry.get("labels", []),
                        "function": data_entry.get("function"),
                    }
                )
            for control_entry in element.get("control_signals", []) or []:
                io_map_inputs.append(
                    {
                        "role": control_entry.get("role"),
                        "net": control_entry.get("net"),
                        "labels": control_entry.get("labels", []),
                        "activation": control_entry.get("activation"),
                        "timing": control_entry.get("timing"),
                        "function": control_entry.get("function"),
                    }
                )
            io_map_outputs: list[dict] = []
            for output_entry in element.get("outputs", []) or []:
                io_map_outputs.append(
                    {
                        "role": "Qn" if output_entry.get("polarity") == "inverted" else "Q",
                        "net": output_entry.get("net"),
                        "labels": output_entry.get("labels", []),
                        "function": output_entry.get("function"),
                    }
                )
            element["io_map"] = {"inputs": io_map_inputs, "outputs": io_map_outputs}

if TYPE_CHECKING:
    from krona.application.logic_functions_parts.transistor import _Transistor
