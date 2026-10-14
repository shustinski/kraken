"""logic_functions: select cover."""
from __future__ import annotations


class SelectCoverMixin:
    @classmethod
    def _select_cover(cls, one_terms: list[str], prime_implicants: list[str]) -> list[str]:
        uncovered = set(one_terms)
        selected: list[str] = []

        coverage: dict[str, set[str]] = {
            implicant: {m for m in one_terms if cls._implicant_covers(implicant, m)} for implicant in prime_implicants
        }

        # Essential prime implicants
        for minterm in one_terms:
            covering = [implicant for implicant, covered in coverage.items() if minterm in covered]
            if len(covering) == 1 and covering[0] not in selected:
                selected.append(covering[0])
                uncovered -= coverage[covering[0]]

        # Greedy completion by best coverage, then by simplicity.
        while uncovered:
            best = None
            best_score: tuple[int, int, int, str] | None = None
            for implicant, covered in coverage.items():
                gain = len(covered & uncovered)
                if gain == 0:
                    continue
                score = (gain, implicant.count("-"), -implicant.count("1"), implicant)
                if best_score is None or score > best_score:
                    best = implicant
                    best_score = score
            if best is None:
                break
            if best not in selected:
                selected.append(best)
            uncovered -= coverage[best]

        return selected if not uncovered else []

    @staticmethod
    def _implicant_to_expr(implicant: str, inputs: list[str]) -> str:
        literals: list[str] = []
        for bit, variable in zip(implicant, inputs):
            if bit == "-":
                continue
            literals.append(variable if bit == "1" else f"!{variable}")
        if not literals:
            return "1"
        return " & ".join(literals)

    @staticmethod
    def _implicant_to_literals(implicant: str, inputs: list[str]) -> frozenset[tuple[str, bool]]:
        literals: set[tuple[str, bool]] = set()
        for bit, variable in zip(implicant, inputs):
            if bit == "-":
                continue
            is_negated = bit == "0"
            literals.add((variable, is_negated))
        return frozenset(literals)

    @staticmethod
    def _render_literal(literal: tuple[str, bool]) -> str:
        variable, is_negated = literal
        return f"!{variable}" if is_negated else variable

    @classmethod
    def _render_and_term(cls, term: frozenset[tuple[str, bool]]) -> str:
        if not term:
            return "1"
        ordered = sorted(term, key=lambda item: item[0])
        return " & ".join(cls._render_literal(literal) for literal in ordered)

    @staticmethod
    def _absorb_terms(terms: list[frozenset[tuple[str, bool]]]) -> list[frozenset[tuple[str, bool]]]:
        unique = sorted(set(terms), key=lambda term: (len(term), sorted(term)))
        out: list[frozenset[tuple[str, bool]]] = []
        for term in unique:
            if any(other.issubset(term) for other in unique if other is not term):
                continue
            out.append(term)
        return out

    @classmethod
    def _detect_xor_or_xnor(cls, terms: list[frozenset[tuple[str, bool]]]) -> str | None:
        if len(terms) != 2:
            return None
        t1, t2 = terms
        vars1 = {name for name, _ in t1}
        vars2 = {name for name, _ in t2}
        if vars1 != vars2 or len(vars1) != 2:
            return None

        ordered_vars = sorted(vars1)
        signs1: dict[str, bool] = {name: neg for name, neg in t1}
        signs2: dict[str, bool] = {name: neg for name, neg in t2}
        if any(signs1[var] == signs2[var] for var in ordered_vars):
            return None

        v1, v2 = ordered_vars
        xor_expr = f"{v1} ^ {v2}"
        is_xnor = signs1[v1] == signs1[v2]
        return f"!({xor_expr})" if is_xnor else xor_expr

    @classmethod
    def _render_expression(cls, terms: list[frozenset[tuple[str, bool]]]) -> str:
        terms = cls._absorb_terms(terms)
        if not terms:
            return "0"
        if any(len(term) == 0 for term in terms):
            return "1"
        if len(terms) == 1:
            return cls._render_and_term(terms[0])

        common = set(terms[0])
        for term in terms[1:]:
            common &= set(term)
        if common:
            common_term = frozenset(common)
            reduced_terms = [frozenset(set(term) - common) for term in terms]
            inner = cls._render_expression(reduced_terms)
            outer = cls._render_and_term(common_term)
            if inner == "1":
                return outer
            return f"{outer} & ({inner})"

        xor_or_xnor = cls._detect_xor_or_xnor(terms)
        if xor_or_xnor is not None:
            return xor_or_xnor

        rendered_terms = sorted(cls._render_and_term(term) for term in terms)
        if len(rendered_terms) == 1:
            return rendered_terms[0]
        return " | ".join(rendered_terms)
