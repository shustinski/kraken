"""patterns: detect toggle chains."""
from __future__ import annotations


class DetectToggleChainsMixin:
    def _detect_toggle_chains(self, xor_toggle: list[dict]) -> list[dict]:
        q_to_stage: dict[str, str] = {}
        for item in xor_toggle:
            q_to_stage[str(item.get("q_net"))] = str(item.get("ff_device"))
        links: list[tuple[str, str, str]] = []
        for item in xor_toggle:
            dst_ff = str(item.get("ff_device"))
            for toggle_in in item.get("toggle_inputs", []):
                src_ff = q_to_stage.get(str(toggle_in))
                if src_ff and src_ff != dst_ff:
                    links.append((src_ff, dst_ff, str(toggle_in)))
        if not links:
            return []
        adjacency: dict[str, set[str]] = {}
        edge_payload: dict[tuple[str, str], list[str]] = {}
        for src, dst, net in links:
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set())
            edge_payload.setdefault((src, dst), []).append(net)
        indegree = {node: 0 for node in adjacency}
        for dsts in adjacency.values():
            for dst in dsts:
                indegree[dst] = indegree.get(dst, 0) + 1
        starts = [node for node, deg in indegree.items() if deg == 0] or list(adjacency)
        chains: list[dict] = []
        seen_paths: set[tuple[str, ...]] = set()
        for start in starts:
            path = [start]
            current = start
            local_seen = {start}
            while adjacency.get(current):
                nxt = sorted(adjacency[current])[0]
                if nxt in local_seen:
                    break
                path.append(nxt)
                local_seen.add(nxt)
                current = nxt
            if len(path) < 2:
                continue
            key = tuple(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            chains.append(
                {
                    "kind": "toggle_chain",
                    "stages": path,
                    "links": [
                        {"src_stage": s, "dst_stage": d, "feedback_nets": sorted(edge_payload.get((s, d), []))}
                        for s, d in zip(path, path[1:])
                    ],
                }
            )
        return chains
