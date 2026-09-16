"""Persistent job state; mutations are serialized by the service lock."""

from __future__ import annotations

import json
from contextlib import contextmanager
import sqlite3
import time
import uuid
from pathlib import Path

from .contracts import validate_payload


class JobStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
                    queued REAL, error TEXT NOT NULL DEFAULT '', command TEXT NOT NULL DEFAULT '',
                    finished REAL, outcome TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL,
                    topic TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS answers (
                    job TEXT NOT NULL, question TEXT NOT NULL, value INTEGER NOT NULL,
                    PRIMARY KEY(job, question));
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.root / "jobs.sqlite3", timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, key: str, payload: dict) -> dict:
        validate_payload(payload)
        serialized = json.dumps(payload, sort_keys=True)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE request_key=?", (key,)).fetchone()
            if row:
                if row["payload"] != serialized:
                    raise ValueError("Request key already used for different inputs")
                return dict(row)
            job = uuid.uuid4().hex
            (self.root / job / "inputs").mkdir(parents=True)
            (self.root / job / "outputs").mkdir()
            db.execute(
                "INSERT INTO jobs(id,request_key,payload,status,created) VALUES(?,?,?,?,?)",
                (job, key, serialized, "uploading", time.time()),
            )
        return self.get(job)

    def get(self, job: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
        if row is None:
            raise KeyError(job)
        return dict(row)

    def list(self) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY created")]

    def update(self, job: str, status: str, error: str = "") -> None:
        finished = time.time() if status in {"succeeded", "failed", "cancelled", "interrupted"} else None
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?,error=?,finished=? WHERE id=?", (status, error, finished, job))

    def finish_compute(self, job: str, outcome: str, error: str = "") -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='finishing',outcome=?,error=? WHERE id=?", (outcome, error, job))

    def enqueue(self, job: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='queued',queued=?,command='' WHERE id=?", (time.time(), job))

    def claim(self) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY queued,created LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (row["id"],))
        return self.get(row["id"])

    def event(self, job: str, topic: str, payload) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO events(job,topic,payload) VALUES(?,?,?)",
                (job, topic, json.dumps(payload, ensure_ascii=False)),
            )

    def events(self, job: str, after: int) -> list[dict]:
        self.get(job)
        with self.connect() as db:
            return [
                {"seq": r["seq"], "topic": r["topic"], "payload": json.loads(r["payload"])}
                for r in db.execute("SELECT * FROM events WHERE job=? AND seq>? ORDER BY seq LIMIT 256", (job, after))
            ]

    def command(self, job: str, command: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET command=? WHERE id=?", (command, job))

    def answer(self, job: str, question: str, value: bool) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO answers VALUES(?,?,?)", (job, question, int(value)))

    def read_answer(self, job: str, question: str) -> bool | None:
        with self.connect() as db:
            row = db.execute("SELECT value FROM answers WHERE job=? AND question=?", (job, question)).fetchone()
        return bool(row[0]) if row else None
