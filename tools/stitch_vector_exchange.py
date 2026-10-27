"""Export or verify an accepted Kraken vector mosaic, without recomputation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def inspect(path):
    from kraken_manager.infrastructure.vector_exchange import VectorExchangeReader
    counts = {"blocks": 0, "features": 0, "conflicts": 0}
    with VectorExchangeReader(path) as reader:
        for record in reader.blocks():
            counts["blocks"] += 1
            counts["features"] += len(record["features"])
            counts["conflicts"] += len(record["conflicts"])
        assert reader.verified
        metadata = {name: reader.header[name] for name in ("schema", "unit", "project_id", "layer_id", "job_id", "output_version_id", "output_sha256")}
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"status": "passed", "complete_stream_verified": True, **metadata, **counts, "file_sha256": digest,
        "training_started": False, "computation_started": False, "electrical_connectivity": "unverified"}


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--data-dir", type=Path, required=True)
    export.add_argument("--project-id", required=True)
    export.add_argument("--job-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        from kraken_hub.composition import EmbeddedProjectService
        from kraken_manager.infrastructure.filesystem import FilesystemEventStore
        source = EmbeddedProjectService(args.data_dir)
        events = FilesystemEventStore(source.catalog_root, args.project_id)
        before = events.verify()
        source.export_stitch_vectors(project_id=args.project_id, job_id=args.job_id, destination=args.output)
        assert events.verify() == before
        result = {**inspect(args.output), "source_events_unchanged": True}
    else:
        result = inspect(args.path)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
