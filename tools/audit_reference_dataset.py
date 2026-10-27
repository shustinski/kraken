"""Read-only inventory of paired image/CIF references with content-safe splits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def indexed(directory, extensions):
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.is_symlink():
            raise ValueError(f"Symbolic source requires explicit review: {path}")
        key = path.stem.casefold()
        if key in result:
            raise ValueError(f"Duplicate reference stem: {path} and {result[key]}")
        result[key] = path
    return result


def build_splits(records):
    content_groups = defaultdict(list)
    for record in records:
        if 'image_sha256' in record:
            content_groups[record['image_sha256']].append(record)
    conflicts = [dict(image_sha256=content, keys=[record['key'] for record in group],
                      reason='identical_image_different_reference_bytes_requires_review')
                 for content, group in content_groups.items()
                 if len({record.get('vector_sha256') for record in group}) > 1]
    quarantined = {item['image_sha256'] for item in conflicts}
    split = dict(train=[], validation=[], heldout=[])
    assignments = {}
    for index, record in enumerate(records):
        content = record.get('image_sha256')
        if content is None or record['issues'] or content in quarantined:
            continue
        proposed = 'train' if index < len(records) * .6 else 'validation' if index < len(records) * .8 else 'heldout'
        selected = assignments.setdefault(content, proposed)
        split[selected].append(record['key'])
    return split, conflicts, sum(len(group)>1 for group in content_groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be in 1..16")
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root / "plugins/contour/src")]
    from PIL import Image
    from contour.serializers import load_polygons_cif

    images = indexed(args.images.resolve(), {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    vectors = indexed(args.vectors.resolve(), {".cif"})
    keys = sorted(images.keys() & vectors.keys())
    output = args.output.resolve()
    if output.is_relative_to(args.images.resolve()) or output.is_relative_to(args.vectors.resolve()):
        raise ValueError("Inventory output must be separate from source directories")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "pairs.jsonl"
    if manifest.exists():
        raise FileExistsError("Choose a fresh audit directory; existing evidence is preserved")
    started = time.monotonic()

    def inspect(key):
        image, vector = images[key], vectors[key]
        value = dict(key=key, image=str(image), vector=str(vector), image_bytes=image.stat().st_size,
                     vector_bytes=vector.stat().st_size, issues=[])
        try:
            value.update(image_sha256=digest(image), vector_sha256=digest(vector))
            with Image.open(image) as decoded:
                decoded.load()
                value.update(image_size=list(decoded.size), image_mode=decoded.mode, pixels_decoded=True)
            reference_name, reference_size, polygons = load_polygons_cif(vector)
            value.update(reference_image=reference_name, reference_size=reference_size,
                         polygon_count=len(polygons), point_count=sum(len(p.points) for p in polygons))
            if reference_size is not None and list(reference_size) != value["image_size"]:
                value["issues"].append("reference_image_dimension_mismatch")
            if reference_name and Path(reference_name).stem.casefold() != key:
                value["issues"].append("reference_image_name_mismatch")
            if not polygons:
                value["issues"].append("empty_reference_geometry_requires_review")
        except Exception as exc:
            value["issues"].append(f"decode_or_parse_error: {type(exc).__name__}: {exc}")
        return value

    records, shapes, issues = [], Counter(), Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor, manifest.open("x", encoding="utf-8") as stream:
        for index, value in enumerate(executor.map(inspect, keys), 1):
            records.append(value)
            shapes[str(value.get("image_size"))] += 1
            issues.update(value["issues"])
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            if index % 100 == 0:
                stream.flush()
                print(f"{index}/{len(keys)} paired references decoded, hashed and parsed", flush=True)
    split, conflicts, duplicate_groups = build_splits(records)
    summary = dict(schema_version=1, source_images=str(args.images.resolve()), source_vectors=str(args.vectors.resolve()),
        paired=len(keys), image_only=sorted(images.keys() - vectors.keys()), vector_only=sorted(vectors.keys() - images.keys()),
        shapes=dict(shapes), issues=dict(issues), source_bytes=sum(r["image_bytes"]+r["vector_bytes"] for r in records),
        duplicate_image_groups=duplicate_groups, conflicting_references=conflicts,
        manifest_sha256=digest(manifest), seconds=round(time.monotonic()-started,3),
        split_policy="contiguous ordinal blocks 60/20/20; identical image bytes stay in one split; issue records and differing-reference duplicate groups excluded",
        limits=["No physical tile adjacency inferred", "CIF is supplied reference annotation, not verified electrical ground truth",
                "Chip-level independence of the splits is not established"], split_counts={key:len(values) for key,values in split.items()})
    (output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output/"splits.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if issues or conflicts or summary["image_only"] or summary["vector_only"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
