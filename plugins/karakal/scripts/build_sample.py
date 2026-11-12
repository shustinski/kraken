"""Build a labelling sample from a validation run and export it with a manifest.

    python scripts/build_sample.py OUT_DIR --photos PHOTOS --model a=MASKS_A [--model b=MASKS_B]
        [--metric topology_risk] [--total 40] [--seed 0] [--exclude registry.txt] [--photo-check]
        [--worst N --per-kind N --diverse N --random N] [--prelabel a]

Confidence maps are taken from the model folders by the confidence suffix (``_confidence``).
The sample: the worst frames by the metric, frames of every kind of finding, frames spread over
the run and a random control part (see core/sampling.py); frames listed in ``--exclude`` are
skipped, and the exported ones are added to it, so the next sample does not repeat them.
The export (core/sample_export.py): photos, masks, the mask to correct, an overlay of the
findings, manifest.json / manifest.csv. When the corrected masks are back:

    python scripts/evaluate_with_ground_truth.py OUT_DIR TRUTH_DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

from karakal.core.analytics import collect_frame_records, compute_build_result_analytics  # noqa: E402
from karakal.core.domain import BuildOptions, FolderSpec, GeometryMode, ModelSpec  # noqa: E402
from karakal.core.sample_export import export_sample  # noqa: E402
from karakal.core.sampling import (  # noqa: E402
    STRATEGY_DIVERSE,
    STRATEGY_PER_KIND,
    STRATEGY_RANDOM,
    STRATEGY_WORST,
    SamplePlan,
    SampleRule,
    append_exclusions,
    build_sample,
    default_plan,
    read_exclusions,
)


def _models(values: list[str]) -> list[tuple[str, Path]]:
    models = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--model wants ID=FOLDER, got {value!r}")
        model_id, folder = value.split("=", 1)
        models.append((model_id.strip(), Path(folder.strip())))
    return models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--photos", type=Path, help="folder of the chip photos (the source)")
    parser.add_argument("--model", action="append", default=[], help="ID=FOLDER of a model's masks (and confidence maps)")
    parser.add_argument("--metric", default="topology_risk", help="metric of the 'worst' and 'diverse' parts")
    parser.add_argument("--total", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worst", type=int)
    parser.add_argument("--per-kind", type=int)
    parser.add_argument("--diverse", type=int)
    parser.add_argument("--random", type=int)
    parser.add_argument("--exclude", type=Path, help="registry of frames already labelled; the sample is added to it")
    parser.add_argument("--photo-check", action="store_true", help="also read the photos against the masks")
    parser.add_argument("--points", action="store_true", help="the layer is points (vias)")
    parser.add_argument("--prelabel", help="model whose mask the operator corrects (the first by default)")
    parser.add_argument("--confidence-suffix", default="_confidence")
    args = parser.parse_args(argv)
    models = _models(args.model)
    if not models:
        parser.error("at least one --model")
    specs = tuple(ModelSpec(model_id, model_id, folder, prob_folder=folder) for model_id, folder in models)
    options = BuildOptions(
        cache_enabled=False,
        photo_check=bool(args.photo_check),
        geometry_mode=GeometryMode.POINT if args.points else GeometryMode.MASK,
        confidence_suffix=args.confidence_suffix,
    )
    source = collect_frame_records(
        specs, options, original_folder=FolderSpec(args.photos, "photos") if args.photos else None
    )
    result = compute_build_result_analytics(source, metric_key=args.metric)
    exclude = read_exclusions(args.exclude)
    parts = (args.worst, args.per_kind, args.diverse, args.random)
    if any(part is not None for part in parts):
        rules = tuple(
            SampleRule(strategy, int(count or 0), metric_key=args.metric if strategy in (STRATEGY_WORST, STRATEGY_DIVERSE) else None)
            for strategy, count in zip((STRATEGY_WORST, STRATEGY_PER_KIND, STRATEGY_DIVERSE, STRATEGY_RANDOM), parts)
            if count
        )
        plan = SamplePlan(rules, seed=args.seed, exclude=exclude)
    else:
        plan = default_plan(args.metric, total=args.total, seed=args.seed, exclude=exclude)
    sample = build_sample(result, plan)
    manifest = export_sample(result, sample, args.out_dir, plan=plan, prelabel_model=args.prelabel)
    if args.exclude is not None:
        append_exclusions(args.exclude, (frame.key for frame in sample))
    by_strategy: dict[str, int] = {}
    for frame in sample:
        by_strategy[frame.strategy] = by_strategy.get(frame.strategy, 0) + 1
    print(f"{len(sample)} frames of {len(result.records)} -> {manifest}  {by_strategy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
