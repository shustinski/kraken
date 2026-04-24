from __future__ import annotations

from .graphs import GraphBundle
from .model import GraphSummary, RecognizedStructure, StructuralAnalysisReport
from .patterns import PatternDatabase


class ReportingLayer:
    """
    Builds a structured JSON-friendly report from raw pattern/evidence output.
    """

    def build_report(
        self,
        *,
        design_name: str | None,
        source_format: str,
        graphs: GraphBundle,
        patterns: PatternDatabase,
        recognized: list[RecognizedStructure],
        metadata: dict | None = None,
    ) -> StructuralAnalysisReport:
        graph_summary = GraphSummary(
            influence_node_count=len(graphs.influence.nodes),
            influence_edge_count=len(graphs.influence.edges),
            conditional_edge_count=len(graphs.conduction.edges),
            scc_count=len(patterns.scc_summaries),
            storage_scc_count=sum(1 for item in patterns.scc_summaries if item.get("candidate_storage")),
        )

        features = self._aggregate_feature_summary(patterns, recognized)
        diagnostics = [*patterns.diagnostics]

        return StructuralAnalysisReport(
            design_name=design_name,
            source_format=source_format,
            graph=graph_summary,
            recognized_structures=recognized,
            clocks=patterns.clock_analysis.as_dict(),
            features=features,
            sccs=patterns.scc_summaries,
            diagnostics=diagnostics,
            method_limitations=self._method_limitations(),
            sat_extension_plan=self._sat_extension_plan(),
            dynamic_support_plan=self._dynamic_support_plan(),
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _aggregate_feature_summary(patterns: PatternDatabase, recognized: list[RecognizedStructure]) -> dict:
        feature_counts: dict[str, int] = {}
        for item in recognized:
            for feature in item.features:
                feature_counts[feature.value] = feature_counts.get(feature.value, 0) + 1

        return {
            "recognized_count": len(recognized),
            "clock_gating": patterns.clock_gating_features,
            "keeper_chains": patterns.keeper_features,
            "dynamic_precharge_evaluate": patterns.dynamic_features,
            "xor_toggle_feedback": patterns.xor_toggle_features,
            "toggle_chains": patterns.toggle_chains,
            "feature_counts": feature_counts,
        }

    @staticmethod
    def _method_limitations() -> list[str]:
        return [
            "Структурный анализ опирается на эвристику имен пинов/ячеек и может терять точность на экзотических библиотеках без стандартных обозначений.",
            "Анализ устойчивых состояний SCC реализован как приближенная проверка по безусловным ребрам известной полярности; сложные многовходовые функции требуют SAT/символьного решателя.",
            "Для EDIF с неполными joined-связями восстановление пинов выполняется по symbol/geometry и зависит от качества графической аннотации.",
            "Фазовый анализ ограничен перебором релевантных clock/reset сигналов и намеренно ограничивает число комбинаций для производительности.",
            "Детекторы dynamic/keeper/counter используют структурные признаки и не моделируют аналоговые эффекты, утечки и временныe параметры.",
        ]

    @staticmethod
    def _sat_extension_plan() -> list[str]:
        return [
            "Построить булеву transition relation для узлов хранения: Q(t+1)=F(Q(t), D, CLK, RESET, SET, EN).",
            "Кодировать условные ребра как импликации/guard-литералы и отдавать задачу в SAT/SMT (например, PySAT/Z3).",
            "Использовать SAT для проверки бистабильности (существуют >=2 фиксированные точки) и взаимной достижимости состояний под фазовыми ограничениями.",
            "Добавить UNSAT-core/trace для объяснимости: какие ребра/условия доказали или опровергли гипотезу о типе ячейки.",
            "Расширить до bounded model checking для отличия edge-triggered реализаций от level-sensitive с дополнительной логикой.",
        ]

    @staticmethod
    def _dynamic_support_plan() -> list[str]:
        return [
            "Добавить модель зарядового состояния узла (charged/discharged/unknown) и weak/strong drive strength в device abstraction layer.",
            "Расширить conditional conduction graph событиями precharge/evaluate и временными окнами фаз (non-overlap clocks, keeper contention).",
            "Поддержать паразитические емкости и floating nodes как отдельные объекты графа с правилами разряда/удержания.",
            "Ввести символьный временной анализ по фазам (phi1/phi2) с проверкой overlap, race-through и charge sharing.",
            "Интегрировать гибридный switch-level + event simulation backend для валидации структурных гипотез динамических latch/DOMINO-узлов.",
        ]
