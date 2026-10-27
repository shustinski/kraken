"""Read-only CIF exchange checks on real paired images, without model training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root / "plugins/contour/src")]
    from PIL import Image
    from contour.infrastructure.stitch_cif import load_stitch_cif
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.limit <= 5498:
        parser.error("limit must be 1..5498")
    args.output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    results = []
    files = sorted((args.corpus / "cif_metal").glob("*.cif"))[:args.limit]
    if not files:
        raise ValueError("No real CIF inputs found")
    for path in files:
        image_path = args.corpus / "jpg_metal" / (path.stem + ".jpg")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        with Image.open(image_path) as image:
            size = image.size
        result = {"cif": str(path), "image": str(image_path), "cif_sha256": digest, "image_sha256": image_digest, "image_size": size}
        try:
            geometries = load_stitch_cif(path, expected_sha256=digest, image_size=size)
            result.update(status="passed", features=len(geometries), holes=sum(len(g.interiors) for _, g in geometries))
        except (ValueError, RuntimeError) as exc:
            result.update(status="rejected", error=str(exc))
            result.update({key: getattr(exc, attribute) for key, attribute in
                (("error_code", "code"), ("polygon_id", "polygon_id"), ("command_index", "command_index")) if hasattr(exc, attribute)})
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == image_digest
        results.append(result)
        if len(results) % 250 == 0:
            print(json.dumps({"checked": len(results), "rejected": sum(r["status"] == "rejected" for r in results)}), flush=True)
    receipt = {"scope": "real CIF exchange parsing only; no physical registration or geometry quality acceptance",
        "training_started": False, "source_files_unchanged": True, "checked": len(results),
        "passed": sum(r["status"] == "passed" for r in results), "rejected": sum(r["status"] == "rejected" for r in results),
        "seconds": time.monotonic()-started, "files": results}
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}), flush=True)
    return int(receipt["rejected"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
