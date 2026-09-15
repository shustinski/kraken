"""Single-process HTTP control plane with a spawned, serial GPU worker."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import signal
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, TypeAdapter

from .contracts import API_VERSION, CHUNK_SIZE, digest, safe_path
from .store import JobStore
from .worker import run_job


class CreateJob(BaseModel):
    request_key: str = Field(min_length=1, max_length=128)
    payload: dict


class Answer(BaseModel):
    question: str = Field(min_length=1, max_length=128)
    value: bool


class Scheduler:
    def __init__(self, store: JobStore, lock):
        self.store = store
        self.lock = lock
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.process = None

    def run(self):
        while not self.stop.wait(0.2):
            with self.lock:
                job = self.store.claim()
            if job is None:
                continue
            process = multiprocessing.get_context("spawn").Process(
                target=run_job, args=(str(self.store.root), job["id"])
            )
            self.process = process
            try:
                process.start()
                while process.is_alive():
                    if self.stop.is_set():
                        self.store.command(job["id"], "pause")
                    process.join(0.2)
                state = self.store.get(job["id"])
                if state["status"] == "finishing" and process.exitcode == 0:
                    self.store.update(job["id"], state["outcome"], state["error"])
                elif state["status"] == "running" or process.exitcode != 0:
                    self.store.update(job["id"], "failed", f"Compute process exited: {process.exitcode}")
            except Exception as error:
                self.store.update(job["id"], "failed", str(error))
            finally:
                self.process = None


def create_app(
    root: Path,
    *,
    require_cuda: bool = True,
    execute: bool = True,
    max_job_bytes: int = 100 * 1024**3,
    retention_days: int = 0,
) -> FastAPI:
    from neuralimage.application.dto import MainWindowState, SettingsState
    main_validator = TypeAdapter(MainWindowState)
    settings_validator = TypeAdapter(SettingsState)
    store = JobStore(root)
    lock = threading.RLock()
    scheduler = Scheduler(store, lock)
    preview_lock = asyncio.Semaphore(1)

    @asynccontextmanager
    async def lifespan(app):
        from .ownership import server_ownership

        with server_ownership(store.root):
            if require_cuda:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("NeuralImage requires an available NVIDIA CUDA device; check Docker GPU access")
            for job in store.list():
                if job["status"] in {"running", "finishing"}:
                    store.update(job["id"], "interrupted", "Server restarted during computation")
            if retention_days > 0:
                from .cli import cleanup

                cleanup(store, retention_days)
            if execute:
                scheduler.thread.start()
            yield
            scheduler.stop.set()
            if execute:
                await asyncio.to_thread(scheduler.thread.join, 60)
                if scheduler.thread.is_alive():
                    process = scheduler.process
                    if process is not None and process.is_alive():
                        if os.name == "nt":
                            await asyncio.to_thread(
                                subprocess.run,
                                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                check=True,
                                capture_output=True,
                            )
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    await asyncio.to_thread(scheduler.thread.join)

    app = FastAPI(title="NeuralImage compute server", lifespan=lifespan)
    app.state.store = store

    def get(job):
        try:
            return store.get(job)
        except KeyError:
            raise HTTPException(404, "Unknown job") from None

    def input_file(job, name):
        row = get(job)
        manifest = {entry["path"]: entry for entry in json.loads(row["payload"])["files"]}
        if name not in manifest:
            raise HTTPException(404, "Unknown input")
        return row, manifest[name], safe_path(store.root / job / "inputs", name)

    @app.get("/api/v1/capabilities")
    def capabilities():
        return {
            "version": API_VERSION,
            "queue": "fifo",
            "authentication": "none",
            "chunk_size": CHUNK_SIZE,
            "max_job_bytes": max_job_bytes,
        }

    @app.post("/api/v1/preview")
    async def preview(request: Request):
        from .preview import MAX_PREVIEW_BYTES, compute_preview

        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_PREVIEW_BYTES:
                raise HTTPException(413, "Preview request is too large")
            body.extend(chunk)
        try:
            payload = json.loads(body)
            # PreviewEngine is CPU-only. Serialize its RNG state changes; it does
            # not compete with the GPU process or require a Qt event loop.
            async with preview_lock:
                return await asyncio.to_thread(compute_preview, payload)
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/v1/jobs")
    def create(body: CreateJob):
        try:
            main_validator.validate_json(json.dumps(body.payload["main"]), strict=True)
            settings_validator.validate_json(json.dumps(body.payload["settings"]), strict=True)
            if sum(item["size"] for item in body.payload["files"]) > max_job_bytes:
                raise ValueError("Job exceeds upload limit")
            with lock:
                row = store.create(body.request_key, body.payload)
            return {"id": row["id"], "status": row["status"]}
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/v1/jobs")
    def jobs():
        return [{k: v for k, v in row.items() if k not in {"payload", "request_key"}} for row in store.list()]

    @app.get("/api/v1/jobs/{job}")
    def status(job: str):
        return {k: v for k, v in get(job).items() if k not in {"payload", "request_key"}}

    @app.get("/api/v1/jobs/{job}/inputs/{name:path}")
    def offset(job: str, name: str):
        _, _, path = input_file(job, name)
        return {"offset": path.stat().st_size if path.exists() else 0}

    @app.put("/api/v1/jobs/{job}/inputs/{name:path}")
    async def upload(job: str, name: str, request: Request, offset: int = 0):
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > CHUNK_SIZE:
                raise HTTPException(413, "Chunk too large")
            data.extend(chunk)

        def write():
            with lock:
                row, entry, path = input_file(job, name)
                if row["status"] != "uploading":
                    raise HTTPException(409, "Inputs are sealed")
                current = path.stat().st_size if path.exists() else 0
                if offset != current:
                    raise HTTPException(409, "Offset mismatch; query the current offset")
                if offset + len(data) > entry["size"]:
                    raise HTTPException(413, "Input exceeds declared size")
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with path.open("ab") as stream:
                        stream.write(data)
                        stream.flush()
                    if path.stat().st_size == entry["size"] and digest(path) != entry["sha256"]:
                        path.unlink()
                        raise HTTPException(422, "SHA-256 mismatch; upload again")
                except OSError as error:
                    raise HTTPException(507, str(error)) from error
                return {"offset": path.stat().st_size}

        return await asyncio.to_thread(write)

    @app.post("/api/v1/jobs/{job}/submit")
    def submit(job: str):
        with lock:
            row = get(job)
            if row["status"] != "uploading":
                return status(job)
            for entry in json.loads(row["payload"])["files"]:
                path = safe_path(store.root / job / "inputs", entry["path"])
                if not path.is_file() or path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
                    raise HTTPException(409, f"Missing or invalid input: {entry['path']}")
            store.enqueue(job)
        return status(job)

    @app.get("/api/v1/jobs/{job}/events")
    def events(job: str, after: int = 0):
        get(job)
        result = store.events(job, after)
        for event in result:
            if event["topic"] == "question" and store.read_answer(job, event["payload"]["id"]) is not None:
                event["topic"] = "logging"
                event["payload"] = "Server question already answered"
        return result

    @app.post("/api/v1/jobs/{job}/answer")
    def answer(job: str, body: Answer):
        get(job)
        store.answer(job, body.question, body.value)
        return {"ok": True}

    @app.post("/api/v1/jobs/{job}/{command}")
    def control(job: str, command: str):
        with lock:
            row = get(job)
            if command in {"pause", "cancel"}:
                if row["status"] == "running":
                    store.command(job, command)
                elif row["status"] in {"queued", "paused"} or (row["status"] == "uploading" and command == "cancel"):
                    store.update(job, "paused" if command == "pause" else "cancelled")
                else:
                    raise HTTPException(409, "Job is already finished")
            elif command == "resume":
                if row["status"] not in {"paused", "interrupted"}:
                    raise HTTPException(409, "Job is not paused or interrupted")
                if row["status"] == "interrupted":
                    output = store.root / job / "outputs"
                    if not any(output.rglob("*.ckpt")) and not any(
                        output.rglob(".neuralimage-recognition-progress.json")
                    ):
                        raise HTTPException(409, "No checkpoint or recognition progress available")
                store.enqueue(job)
            else:
                raise HTTPException(404, "Unknown command")
        return status(job)

    @app.get("/api/v1/jobs/{job}/artifacts")
    def artifacts(job: str):
        get(job)
        output = store.root / job / "outputs"
        return [
            {"path": p.relative_to(output).as_posix(), "size": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(output.rglob("*"))
            if p.is_file() and not p.is_symlink()
        ]

    @app.get("/api/v1/jobs/{job}/artifacts/{name:path}")
    def artifact(job: str, name: str):
        get(job)
        try:
            path = safe_path(store.root / job / "outputs", name)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if not path.is_file():
            raise HTTPException(404, "Unknown artifact")
        return FileResponse(path)

    return app
