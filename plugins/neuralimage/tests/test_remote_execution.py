from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.remote.contracts import package_inputs, safe_path
from neuralimage.remote.server import create_app
from neuralimage.remote.codec import encode, decode
from neuralimage.application.services.workflow_mapper import build_workflow_parameters


def payload():
    return {
        "version": 1,
        "main": asdict(MainWindowState(work_mode="train_only")),
        "settings": asdict(SettingsState()),
        "files": [],
    }


@pytest.fixture
def api(tmp_path):
    app = create_app(tmp_path / "server", require_cuda=False, execute=False)
    with TestClient(app) as client:
        yield client, app.state.store


def create(client, value=None, key="first"):
    response = client.post("/api/v1/jobs", json={"request_key": key, "payload": value or payload()})
    assert response.status_code == 200, response.text
    return response.json()["id"]


@pytest.mark.parametrize("path", ["../outside", "a/../../outside", "C:/secret", "a\\b", "/tmp/x", "a/./b"])
def test_rejects_escaping_paths(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)


def test_idempotency_and_manifest_validation(api):
    client, store = api
    job = create(client)
    assert create(client) == job
    changed = payload()
    changed["main"]["epochs"] = 2
    assert client.post("/api/v1/jobs", json={"request_key": "first", "payload": changed}).status_code == 422
    assert len(store.list()) == 1


def test_upload_resume_hash_and_seal(api):
    client, _ = api
    body = payload()
    body["files"] = [{"path": "images/frame.bin", "size": 6, "sha256": hashlib.sha256(b"abcdef").hexdigest()}]
    job = create(client, body)
    route = f"/api/v1/jobs/{job}/inputs/images/frame.bin"
    assert client.post(f"/api/v1/jobs/{job}/submit").status_code == 409
    assert client.put(route, content=b"abc").json() == {"offset": 3}
    assert client.put(route, content=b"abc").status_code == 409
    assert client.get(route).json()["offset"] == 3
    assert client.put(route + "?offset=3", content=b"xxx").status_code == 422
    assert client.get(route).json()["offset"] == 0
    assert client.put(route, content=b"abcdef").status_code == 200
    assert client.post(f"/api/v1/jobs/{job}/submit").json()["status"] == "queued"
    assert client.put(route + "?offset=6", content=b"").status_code == 409


def test_fifo_pause_cancel_and_events(api):
    client, store = api
    first = create(client, key="a")
    second = create(client, key="b")
    client.post(f"/api/v1/jobs/{first}/submit")
    client.post(f"/api/v1/jobs/{second}/submit")
    assert client.post(f"/api/v1/jobs/{first}/pause").json()["status"] == "paused"
    assert store.claim()["id"] == second
    client.post(f"/api/v1/jobs/{second}/cancel")
    assert store.get(second)["command"] == "cancel"
    assert store.get(second)["status"] == "running"
    store.event(second, "metrics", {"epoch": 1})
    event = client.get(f"/api/v1/jobs/{second}/events").json()[0]
    assert client.get(f"/api/v1/jobs/{second}/events?after={event['seq']}").json() == []
    assert client.post(f"/api/v1/jobs/{first}/resume").json()["status"] == "queued"


def test_restart_marks_only_running_interrupted(tmp_path):
    app = create_app(tmp_path, require_cuda=False, execute=False)
    store = app.state.store
    a = store.create("a", payload())["id"]
    b = store.create("b", payload())["id"]
    store.enqueue(a)
    store.enqueue(b)
    store.claim()
    with TestClient(create_app(tmp_path, require_cuda=False, execute=False)) as client:
        assert client.get(f"/api/v1/jobs/{a}").json()["status"] == "interrupted"
        assert client.get(f"/api/v1/jobs/{b}").json()["status"] == "queued"


@pytest.mark.parametrize(
    "mode", ["train_only", "train_and_recognition", "recognition_only", "continue_training", "further_training"]
)
def test_settings_round_trip(mode):
    _, training, recognition = build_workflow_parameters(MainWindowState(work_mode=mode), SettingsState())
    assert decode(encode(training)) == training
    assert decode(encode(recognition)) == recognition


def test_recognition_uploads_model_checkpoint_and_images_only(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame.png").write_bytes(b"image")
    model = tmp_path / "model.pth"
    model.write_bytes(b"model")
    model.with_suffix(".ckpt").write_bytes(b"checkpoint")
    main = MainWindowState(
        work_mode="recognition_only", source_folder=str(images), model_path=str(model), sample_folder="missing"
    )
    value, files = package_inputs(main, SettingsState())
    assert value["main"]["sample_folder"] == ""
    assert len(files) == 3
    assert all(not Path(entry["path"]).is_absolute() for entry in value["files"])


def test_result_download_is_http_file(api):
    client, store = api
    job = create(client)
    result = store.root / job / "outputs" / "result.txt"
    result.write_text("result", encoding="utf-8")
    entries = client.get(f"/api/v1/jobs/{job}/artifacts").json()
    assert entries[0]["sha256"] == hashlib.sha256(b"result").hexdigest()
    assert client.get(f"/api/v1/jobs/{job}/artifacts/result.txt").content == b"result"


def _delayed_compute(root, job):
    import time
    from neuralimage.remote.store import JobStore

    store = JobStore(Path(root))
    store.event(job, "started", None)
    store.finish_compute(job, "succeeded")
    time.sleep(0.7)


def test_scheduler_waits_for_process_exit(tmp_path, monkeypatch):
    import time
    import threading
    from neuralimage.remote import server
    from neuralimage.remote.store import JobStore

    store = JobStore(tmp_path)
    first = store.create("one", payload())["id"]
    second = store.create("two", payload())["id"]
    store.enqueue(first)
    store.enqueue(second)
    monkeypatch.setattr(server, "run_job", _delayed_compute)
    scheduler = server.Scheduler(store, threading.RLock())
    scheduler.thread.start()
    try:
        deadline = time.monotonic() + 20
        observed_finishing = False
        while time.monotonic() < deadline:
            a, b = store.get(first), store.get(second)
            if a["status"] == "finishing":
                observed_finishing = True
                assert b["status"] == "queued"
            if b["status"] == "succeeded":
                break
            time.sleep(0.02)
        assert observed_finishing
        assert store.get(first)["status"] == store.get(second)["status"] == "succeeded"
    finally:
        scheduler.stop.set()
        scheduler.thread.join(10)
    assert not scheduler.thread.is_alive()


def test_second_server_cannot_own_same_directory(tmp_path):
    from neuralimage.remote.ownership import server_ownership

    with server_ownership(tmp_path):
        with pytest.raises(RuntimeError, match="Another"):
            with server_ownership(tmp_path):
                pytest.fail("Acquired twice")


def test_real_http_client_recovers_lost_upload_ack_and_downloads(tmp_path, monkeypatch):
    import socket
    import threading
    import time
    import uvicorn
    from urllib.error import URLError
    from neuralimage.remote.client import RemoteClient
    app = create_app(tmp_path / "server", require_cuda=False, execute=False)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]))
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        client = RemoteClient(f"http://127.0.0.1:{port}")
        client.capabilities()
        source = tmp_path / "input.bin"
        source.write_bytes(b"abc" * (2 * 1024 * 1024))
        body = payload()
        body["files"] = [{"path": "data/input.bin", "size": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]
        job = client.request("POST", "/jobs", {"request_key": "network", "payload": body})["id"]
        original_request = client.request
        lost_ack = []
        def request(method, route, payload=None, **kwargs):
            value = original_request(method, route, payload, **kwargs)
            if method == "PUT" and not lost_ack:
                lost_ack.append(True)
                raise URLError("Acknowledgement lost after server committed chunk")
            return value
        monkeypatch.setattr(client, "request", request)
        client.upload(job, {"data/input.bin": source})
        assert lost_ack
        client.request("POST", f"/jobs/{job}/submit")
        result = app.state.store.root / job / "outputs" / "result.bin"
        result.write_bytes(b"result bytes")
        app.state.store.update(job, "succeeded")
        destination = client.download(job, tmp_path / "download")
        assert (destination / "result.bin").read_bytes() == b"result bytes"
        assert client.download(job, tmp_path / "download") == destination
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
    assert not thread.is_alive()
