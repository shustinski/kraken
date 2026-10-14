"""logic_functions: detect parity expression."""
from __future__ import annotations


class DetectParityExpressionMixin:
    @staticmethod
    def _sum_of_products(inputs: list[str], truth_table: list[dict]) -> str:
        ones = [row for row in truth_table if row["value"] == 1]
        if not ones:
            return "0"
        if len(ones) == len(truth_table):
            return "1"
        terms: list[str] = []
        for row in ones:
            literals = []
            for variable in inputs:
                value = row["inputs"][variable]
                literals.append(variable if value else f"!{variable}")
            terms.append(" & ".join(literals))
        return " | ".join(terms)

    @classmethod
    def _simplified_expression(cls, inputs: list[str], truth_table: list[dict]) -> str:
        one_terms: list[str] = []
        zero_terms: list[str] = []
        dc_terms: list[str] = []
        for row in truth_table:
            bits = "".join("1" if row["inputs"].get(name) == 1 else "0" for name in inputs)
            value = row.get("value")
            if value == 1:
                one_terms.append(bits)
            elif value == 0:
                zero_terms.append(bits)
            elif value is None:
                dc_terms.append(bits)

        if not one_terms:
            return "0"
        if len(one_terms) == len(truth_table):
            return "1"

        parity_expr = cls._detect_parity_expression(inputs, truth_table)
        if parity_expr is not None:
            return parity_expr

        direct_expr = cls._simplify_from_on_set(inputs, truth_table, one_terms, dc_terms)
        inverted_expr = cls._invert_candidate(
            cls._simplify_from_on_set(inputs, truth_table, zero_terms, dc_terms)
        )
        return cls._pick_better_expression(direct_expr, inverted_expr)

    @classmethod
    def _simplify_from_on_set(
        cls,
        inputs: list[str],
        truth_table: list[dict],
        one_terms: list[str],
        dc_terms: list[str],
    ) -> str:
        if not one_terms:
            return "0"
        prime_implicants = cls._prime_implicants(one_terms, dc_terms)
        if not prime_implicants:
            return cls._sum_of_products(inputs, truth_table)

        selected = cls._select_cover(one_terms, prime_implicants)
        if not selected:
            return cls._sum_of_products(inputs, truth_table)
        literal_terms = [cls._implicant_to_literals(term, inputs) for term in selected]
        return cls._render_expression(literal_terms)

    @staticmethod
    def _invert_candidate(expr: str) -> str:
        if expr == "0":
            return "1"
        if expr == "1":
            return "0"
        if expr.startswith("!(") and expr.endswith(")"):
            return expr[2:-1]
        if " " in expr or "|" in expr or "&" in expr or "^" in expr:
            return f"!({expr})"
        return f"!{expr}"

    @staticmethod
    def _pick_better_expression(direct_expr: str, inverted_expr: str) -> str:
        def score(expr: str) -> tuple[int, int]:
            op_count = expr.count("&") + expr.count("|") + expr.count("^")
            return op_count, len(expr)

        return inverted_expr if score(inverted_expr) < score(direct_expr) else direct_expr

    @classmethod
    def _detect_parity_expression(cls, inputs: list[str], truth_table: list[dict]) -> str | None:
        if not inputs:
            return None
        if any(row.get("value") is None for row in truth_table):
            return None

        var_indexes = list(range(len(inputs)))
        best_expr: str | None = None
        best_score: tuple[int, int] | None = None

        # Search F = parity(selected vars) xor const.
        for mask in range(1, 1 << len(inputs)):
            selected = [idx for idx in var_indexes if mask & (1 << idx)]
            for const in (0, 1):
                matches = True
                for row in truth_table:
                    inputs_map = row.get("inputs", {})
                    parity = const
                    for idx in selected:
                        parity ^= 1 if inputs_map.get(inputs[idx]) == 1 else 0
                    if parity != row.get("value"):
                        matches = False
                        break
                if not matches:
                    continue

                base = " ^ ".join(inputs[idx] for idx in selected)
                expr = f"!({base})" if const == 1 and len(selected) > 1 else (f"!{base}" if const == 1 else base)
                score = (expr.count("^"), len(expr))
                if best_score is None or score < best_score:
                    best_score = score
                    best_expr = expr

        return best_expr

    @staticmethod
    def _combine_implicants(a: str, b: str) -> str | None:
        diff = 0
        out: list[str] = []
        for ca, cb in zip(a, b):
            if ca == cb:
                out.append(ca)
            elif ca != "-" and cb != "-":
                diff += 1
                out.append("-")
            else:
                return None
            if diff > 1:
                return None
        return "".join(out) if diff == 1 else None

    @classmethod
    def _prime_implicants(cls, one_terms: list[str], dc_terms: list[str]) -> list[str]:
        current = sorted(set(one_terms + dc_terms))
        prime_implicants: set[str] = set()

        while current:
            used: set[str] = set()
            next_terms: set[str] = set()
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    combined = cls._combine_implicants(current[i], current[j])
                    if combined is not None:
                        used.add(current[i])
                        used.add(current[j])
                        next_terms.add(combined)
            for term in current:
                if term not in used:
                    prime_implicants.add(term)
            current = sorted(next_terms)
        return sorted(prime_implicants)

    @staticmethod
    def _implicant_covers(implicant: str, minterm: str) -> bool:
        for ci, cm in zip(implicant, minterm):
            if ci != "-" and ci != cm:
                return False
        return True
