"""patterns: detect bistable cores."""
from __future__ import annotations
from krona.analysis.structural.model import (
    BistableCoreEvidence,
    EdgeSemantic,
)
from krona.analysis.structural.scc import TarjanSCCDetector

from krona.analysis.structural.patterns_parts.storage_phase_profile import BistableCoreCandidate


class DetectBistableCoresMixin:
    def _detect_bistable_cores(self) -> tuple[list[dict], list[BistableCoreCandidate]]:
        scc_detector = TarjanSCCDetector(self._graphs.influence.adjacency)
        sccs = scc_detector.run()
        edge_set = self._graphs.influence.edges

        scc_summaries: list[dict] = []
        cores: list[BistableCoreCandidate] = []
        for component in sccs:
            nodes = tuple(sorted(component.nodes))
            internal_edges = [
                edge
                for edge in edge_set
                if edge.source in nodes and edge.target in nodes and edge.semantic == EdgeSemantic.INFLUENCE
            ]
            has_self_loop = any(edge.source == edge.target for edge in internal_edges)
            if component.size == 1 and not has_self_loop:
                scc_summaries.append(
                    {
                        "nodes": list(nodes),
                        "size": component.size,
                        "candidate_storage": False,
                        "positive_feedback": False,
                        "stable_state_count": 0,
                    }
                )
                continue

            positive_feedback = self._has_positive_feedback(nodes, internal_edges)
            stable_states = self._stable_states(nodes, internal_edges)
            cross_pairs = self._cross_coupled_inverters(nodes, internal_edges)
            candidate_storage = positive_feedback and len(stable_states) >= 2

            scc_summaries.append(
                {
                    "nodes": list(nodes),
                    "size": component.size,
                    "candidate_storage": candidate_storage,
                    "positive_feedback": positive_feedback,
                    "stable_state_count": len(stable_states),
                    "stable_states": stable_states,
                    "cross_coupled_inverters": [list(pair) for pair in cross_pairs],
                }
            )
            if not candidate_storage:
                continue

            incoming = tuple(edge for edge in edge_set if edge.target in nodes and edge.source not in nodes)
            outgoing = tuple(edge for edge in edge_set if edge.source in nodes and edge.target not in nodes)
            outputs = tuple(sorted({edge.source for edge in outgoing} | {node for node in nodes if self._has_outgoing_to_noncore(node, nodes)}))
            evidence = BistableCoreEvidence(
                scc_nodes=nodes,
                positive_feedback=positive_feedback,
                stable_states=stable_states,
                cross_coupled_inverters=cross_pairs,
            )
            cores.append(
                BistableCoreCandidate(
                    nodes=nodes,
                    evidence=evidence,
                    incoming_edges=incoming,
                    outgoing_edges=outgoing,
                    storage_outputs=outputs or nodes,
                    cross_coupled_pair=cross_pairs[0] if cross_pairs else None,
                )
            )
        return scc_summaries, cores
