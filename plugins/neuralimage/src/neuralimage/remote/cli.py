"""CLI for server operation and explicit result recovery."""

from __future__ import annotations
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from .store import JobStore


def main():
    parser = argparse.ArgumentParser(description="NeuralImage headless compute server")
    parser.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("NEURALIMAGE_DATA_DIR", "./neuralimage-data"))
    )
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--max-job-gib", type=int, default=100)
    serve.add_argument("--retention-days", type=int, default=0, help="0 retains completed jobs until explicit cleanup")
    sub.add_parser("jobs")
    clean = sub.add_parser("cleanup")
    clean.add_argument("--older-than-days", type=int, required=True)
    download = sub.add_parser("download")
    download.add_argument("--url", required=True)
    download.add_argument("--job", required=True)
    download.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .server import create_app

        uvicorn.run(
            create_app(args.data_dir, max_job_bytes=args.max_job_gib * 1024**3, retention_days=args.retention_days),
            host=args.host,
            port=args.port,
            workers=1,
        )
    elif args.command == "jobs":
        for row in JobStore(args.data_dir).list():
            print(json.dumps({key: row[key] for key in ("id", "status", "created", "error")}, ensure_ascii=False))
    elif args.command == "cleanup":
        from .ownership import server_ownership

        store = JobStore(args.data_dir)
        with server_ownership(store.root):
            cleanup(store, args.older_than_days)
    elif args.command == "download":
        from .client import RemoteClient

        print(RemoteClient(args.url).download(args.job, args.destination))


def cleanup(store: JobStore, days: int):
    if days < 0:
        raise ValueError("Retention days cannot be negative")
    cutoff = time.time() - days * 86400
    for row in store.list():
        if (
            row["status"] not in {"succeeded", "failed", "cancelled"}
            or row["finished"] is None
            or row["finished"] > cutoff
        ):
            continue
        folder = (store.root / row["id"]).resolve()
        if folder.parent != store.root or len(row["id"]) != 32:
            raise ValueError("Invalid job directory")
        if folder.exists():
            shutil.rmtree(folder)
        with store.connect() as db:
            db.execute("DELETE FROM events WHERE job=?", (row["id"],))
            db.execute("DELETE FROM answers WHERE job=?", (row["id"],))
            db.execute("DELETE FROM jobs WHERE id=?", (row["id"],))


if __name__ == "__main__":
    main()
