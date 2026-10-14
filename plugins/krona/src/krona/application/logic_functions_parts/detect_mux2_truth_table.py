"""logic_functions: detect mux2 truth table."""
from __future__ import annotations


class DetectMux2TruthTableMixin:
    def _detect_mux2_from_truth_table(self, inputs: list[str], truth_table: list[dict]) -> dict | None:
        if not inputs or any(row.get("value") is None for row in truth_table):
            return None

        best: dict | None = None
        best_score: tuple[int, int] | None = None
        for select in inputs:
            others = [name for name in inputs if name != select]
            rows_by_other: dict[tuple[tuple[str, int], ...], dict[int, int]] = {}
            valid = True
            for row in truth_table:
                value = row.get("value")
                if value not in {0, 1}:
                    valid = False
                    break
                inputs_map = row.get("inputs", {})
                if select not in inputs_map:
                    valid = False
                    break
                key = tuple((name, int(inputs_map.get(name, 0))) for name in others)
                branch = int(inputs_map.get(select, 0))
                bucket = rows_by_other.setdefault(key, {})
                bucket[branch] = int(value)
            if not valid or not rows_by_other:
                continue
            if any(0 not in bucket or 1 not in bucket for bucket in rows_by_other.values()):
                continue
            if all(bucket[0] == bucket[1] for bucket in rows_by_other.values()):
                continue

            rows0 = []
            rows1 = []
            for key, bucket in rows_by_other.items():
                assignment = {name: bit for name, bit in key}
                rows0.append({"inputs": assignment, "value": bucket[0]})
                rows1.append({"inputs": assignment, "value": bucket[1]})
            rows0.sort(key=lambda row: tuple(row["inputs"].get(name, 0) for name in others))
            rows1.sort(key=lambda row: tuple(row["inputs"].get(name, 0) for name in others))
            expr0 = self._simplified_expression(others, rows0) if others or rows0 else None
            expr1 = self._simplified_expression(others, rows1) if others or rows1 else None
            if expr0 is None or expr1 is None or expr0 == expr1:
                continue

            mux_expr = f"mux({select}, {expr0}, {expr1})"
            score = (int(select.startswith("IN")), -(len(mux_expr)))
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "kind": "mux2",
                    "select": select,
                    "when0": expr0,
                    "when1": expr1,
                    "expression": mux_expr,
                }
        return best
