"""How well a whole run (benchmark_full_run.py) finds what scripts/perf_synth_set.py planted.

    python benchmarks/full_run_scorecard.py DATASET RUN.json [RUN2.json ...] [--markdown OUT.md]

Per run:
- dirt: share of planted dirt regions (large / small) that a dirt region of the layer touches, the
  same for uncertain-only patches, and how many found regions touch nothing planted (false);
- cells: share of planted cell defects (per kind) that a non-normal cell touches, and flagged cells
  away from anything planted;
- validation: ranking of frames by the validation score (AUC of frames with planted bad parts
  against clean frames of the array), and the share of planted bad parts a finding box touches.

Several runs side by side show whether an optimisation changed what is found.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SIZE = 2000
_NUMBER = re.compile(r"(\d+)(?=\D*$)")


def _index(key: str) -> int | None:
    match = _NUMBER.search(Path(str(key)).stem)
    return int(match.group(1)) if match else None


def _touch(a, b, pad: float = 0.0) -> bool:
    ax, ay, aw, ah = (float(v) for v in a[:4])
    bx, by, bw, bh = (float(v) for v in b[:4])
    return ax - pad < bx + bw and bx - pad < ax + aw and ay - pad < by + bh and by - pad < ay + ah


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    ordered = sorted([(value, 1) for value in positive] + [(value, 0) for value in negative])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[index][0]:
            end += 1
        rank = (index + end) / 2.0 + 1.0
        rank_sum += rank * sum(label for _value, label in ordered[index : end + 1])
        index = end + 1
    count = len(positive)
    return (rank_sum - count * (count + 1) / 2.0) / (count * len(negative))


def _layer_of_model(run: dict) -> dict[str, str]:
    """Model id -> ones/zeros (the run names its models by layer)."""

    names: dict[str, str] = {}
    for model_id in (run.get("cells", {}).get("defects") or {}):
        names[model_id] = model_id if model_id in ("ones", "zeros") else ("zeros" if "zeros" in model_id else "ones")
    for model_id in (run.get("dirt", {}).get("regions") or {}):
        names.setdefault(model_id, model_id if model_id in ("ones", "zeros") else "ones")
    return names


def score(answers: dict[int, dict], run: dict) -> dict:
    layer_of = _layer_of_model(run)
    out: dict = {}

    # Dirt.
    dirt = {"large": [0, 0], "small": [0, 0], "uncertain": [0, 0], "found": 0, "false": 0, "false_outside_array": 0}
    for model_id, frames in (run.get("dirt", {}).get("regions") or {}).items():
        found_by_index = {_index(key): regions for key, regions in frames.items()}
        for index, answer in answers.items():
            regions = found_by_index.get(index) or []
            for item in answer["dirt"]:
                dirt[item["kind"]][1] += 1
                dirt[item["kind"]][0] += any(_touch(region, item["box"]) for region in regions)
            for item in answer["uncertain"]:
                dirt["uncertain"][1] += 1
                dirt["uncertain"][0] += any(_touch(region, item["box"]) for region in regions)
            planted = [item["box"] for item in answer["dirt"] + answer["uncertain"]]
            planted += [item["box"] for item in answer["layers"].get(layer_of.get(model_id, ""), {}).get("defects", [])]
            planted += [item["box"] for item in answer["conflicts"]]
            for region in regions:
                dirt["found"] += 1
                unexplained = not any(_touch(region, box, pad=20) for box in planted)
                if unexplained and not answer["in_array"]:
                    # Stray blobs drawn on frames outside the array are not in the answers.
                    dirt["false_outside_array"] += 1
                elif unexplained:
                    dirt["false"] += 1
    out["dirt"] = {
        key: (round(value[0] / value[1], 3) if value[1] else None, value[1]) if isinstance(value, list) else value
        for key, value in dirt.items()
    }

    # Cells.
    kinds: dict[str, list[int]] = {}
    flagged = 0
    false_flags = 0
    for model_id, frames in (run.get("cells", {}).get("defects") or {}).items():
        layer = layer_of.get(model_id, "")
        by_index = {_index(key): boxes for key, boxes in frames.items()}
        for index, answer in answers.items():
            boxes = by_index.get(index) or []
            planted = answer["layers"].get(layer, {}).get("defects", [])
            for item in planted:
                entry = kinds.setdefault(item["kind"], [0, 0])
                entry[1] += 1
                entry[0] += any(_touch(box, item["box"], pad=2) for box in boxes)
            others = [item["box"] for item in answer["dirt"] + answer["conflicts"]] + [item["box"] for item in planted]
            for box in boxes:
                flagged += 1
                false_flags += not any(_touch(box, other, pad=6) for other in others)
    out["cells"] = {
        "recall": {kind: (round(hit / total, 3), total) for kind, (hit, total) in sorted(kinds.items())},
        "flagged": flagged,
        "away_from_planted": false_flags,
    }

    # Validation.
    validation = run.get("validation") or {}
    scores = validation.get("scores") or {}
    findings = validation.get("findings") or {}
    positive, negative = [], []
    parts = [0, 0]
    for key, value in scores.items():
        index = _index(key)
        answer = answers.get(index)
        if value is None or answer is None or not answer["in_array"]:
            continue
        bad = answer["dirt"] + answer["uncertain"] + answer["conflicts"]
        bad += [item for layer in answer["layers"].values() for item in layer["defects"]]
        (positive if bad else negative).append(float(value))
        boxes = [
            [b[0] * SIZE, b[1] * SIZE, b[2] * SIZE, b[3] * SIZE]
            for _kind, _detector, _model, _severity, b in findings.get(key, [])
            if b
        ]
        whole_frame = any(not item[4] for item in findings.get(key, []))
        for item in answer["dirt"] + answer["uncertain"] + answer["conflicts"]:
            parts[1] += 1
            parts[0] += whole_frame or any(_touch(box, item["box"], pad=10) for box in boxes)
    auc = _auc(positive, negative)
    out["validation"] = {
        "metric": validation.get("metric_key"),
        "auc_bad_vs_clean": None if auc is None else round(auc, 4),
        "frames_bad": len(positive),
        "frames_clean": len(negative),
        "bad_parts_touched_by_findings": (round(parts[0] / parts[1], 3) if parts[1] else None, parts[1]),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    answers = {}
    with (args.dataset / "answers.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            answers[int(item["index"])] = item
    results = {}
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        # Only the frames the run had (--limit / --stride runs take a part of the set).
        present = {
            _index(key)
            for groups in (run.get("cells", {}).get("digests") or {}).values()
            for key in groups
        } | {_index(key) for key in (run.get("validation", {}).get("scores") or {})}
        subset = {index: item for index, item in answers.items() if index in present}
        results[path.stem] = score(subset, run)
    text = json.dumps(results, ensure_ascii=False, indent=1)
    print(text)
    if args.markdown:
        args.markdown.write_text("```json\n" + text + "\n```\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
