"""logic_functions: analyze combinational net function."""
from __future__ import annotations
from itertools import product


class AnalyzeCombinationalNetFunctionMixin:
    def _analyze_combinational_net_function(
        self,
        target_net: str,
        inputs: list[str],
        input_net_by_var: dict[str, str],
        output_net_by_name: dict[str, str],
        power_nets: set[str],
        ground_nets: set[str],
        transistors: list[_Transistor],
        cache: dict[str, dict],
        allow_mux: bool = True,
    ) -> dict:
        target_net = str(target_net or "")
        if not target_net:
            return {"kind": "net_function", "expression": None}
        cache_key = f"{target_net}|mux={1 if allow_mux else 0}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        input_names = sorted([name for name, net_name in input_net_by_var.items() if net_name == target_net], key=self._natural_key)
        output_names = sorted([name for name, net_name in output_net_by_name.items() if net_name == target_net], key=self._natural_key)
        if input_names:
            expression = " | ".join(input_names) if len(input_names) > 1 else input_names[0]
            result = {
                "kind": "net_function",
                "expression": expression,
                "simplified_expression": expression,
                "sum_of_products": expression,
                "depends_on": list(input_names),
                "mux": None,
                "source_ports": {
                    "inputs": input_names,
                    "outputs": output_names,
                },
                "is_primary_input": True,
            }
            cache[cache_key] = result
            return result

        if len(inputs) > 12:
            result = {
                "kind": "net_function",
                "expression": None,
                "simplified_expression": None,
                "sum_of_products": None,
                "depends_on": [],
                "mux": None,
                "source_ports": {
                    "inputs": input_names,
                    "outputs": output_names,
                },
                "reason": "too_many_primary_inputs",
            }
            cache[cache_key] = result
            return result

        rows: list[dict] = []
        for bits in product([0, 1], repeat=len(inputs)):
            assignment = dict(zip(inputs, bits))
            value = self._evaluate_output_state(
                output_net=target_net,
                assignment=assignment,
                input_net_by_var=input_net_by_var,
                power_nets=power_nets,
                ground_nets=ground_nets,
                transistors=transistors,
            )
            rows.append({"inputs": assignment, "value": value})

        has_unknown = any(row.get("value") is None for row in rows)
        simplified = None if has_unknown else self._simplified_expression(inputs, rows)
        sop = None if has_unknown else self._sum_of_products(inputs, rows)
        mux = None if (has_unknown or not allow_mux) else self._detect_mux2_from_truth_table(inputs, rows)
        expression = mux.get("expression") if isinstance(mux, dict) and mux.get("expression") else simplified

        depends_on: list[str] = []
        if not has_unknown:
            for var in inputs:
                other_vars = [name for name in inputs if name != var]
                slices: dict[tuple[tuple[str, int], ...], dict[int, int | None]] = {}
                for row in rows:
                    key = tuple((name, int(row["inputs"].get(name, 0))) for name in other_vars)
                    slices.setdefault(key, {})[int(row["inputs"].get(var, 0))] = row.get("value")
                if any(values.get(0) != values.get(1) for values in slices.values() if 0 in values and 1 in values):
                    depends_on.append(var)

        result = {
            "kind": "net_function",
            "expression": expression,
            "simplified_expression": simplified,
            "sum_of_products": sop,
            "depends_on": depends_on,
            "mux": mux,
            "source_ports": {
                "inputs": input_names,
                "outputs": output_names,
            },
            "is_primary_input": False,
        }
        cache[cache_key] = result
        return result
