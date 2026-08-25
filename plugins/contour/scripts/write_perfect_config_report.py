"""Write the final per-frame config report from the search checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "benchmarks" / "perfect_search_checkpoint.json"
OUTPUT = ROOT / "benchmarks" / "metal_per_frame_perfect_configs.json"


def main() -> None:
    state = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    frames = []
    for name in sorted(state["perfect"]):
        item = state["perfect"][name]
        frames.append(
            {
                "frame": name,
                "config_id": item["config_id"],
                "strategy": item["strategy"],
                "parameters": item["parameters"],
                "metrics": item["metrics"],
            }
        )
    unresolved = []
    for name in sorted(state["best"]):
        if name in state["perfect"]:
            continue
        item = state["best"][name]
        unresolved.append(
            {
                "frame": name,
                "strategy": item["strategy"],
                "parameters": item["parameters"],
                "metrics": item["metrics"],
                "topology_error": item["topology_error"],
            }
        )
    coverage = [
        {"id": item["id"], "strategy": item["strategy"], "frames": item["frames"], "n": len(item["frames"])}
        for item in state["library"]
    ]
    document = {
        "evaluation_border_crop_px": 50,
        "exact_topology_frames": len(frames),
        "total_frames": 23,
        "minimum_covering_configs": ["C01", "C02", "C03"],
        "minimum_covering_count": 3,
        "coverage_summary": coverage,
        "library": state["library"],
        "frames": frames,
        "unresolved": unresolved,
        "diagnose_0401": {
            "verdict": "A_real_prediction_leak_into_roi",
            "note": (
                "35 false components after crop=50 were 1-4 px slivers of one predicted "
                "top-edge bus (full-frame polygon id 8) at the new ROI y=0. GT pixels "
                "under those blobs were 0. local_adaptive + standard filters removed "
                "the overshoot and made 0401 exact."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUTPUT, "exact", len(frames), "unresolved", len(unresolved))


if __name__ == "__main__":
    main()
