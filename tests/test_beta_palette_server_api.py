"""Палитра W3 — серверные маршруты на живом стенде (KRAKEN_LIVE_WORKER_CONFIG). S05/S06/S07/S08."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from kraken_hub.application.remote_error import RemoteServerError
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.workspace import ImageConversionSettings
from tests.test_live_server_worker_acceptance import _connect, _protected_snapshot


@pytest.fixture
def live_palette(tmp_path):
    filename = os.environ.get("KRAKEN_LIVE_WORKER_CONFIG")
    if not filename:
        pytest.skip("Нужен KRAKEN_LIVE_WORKER_CONFIG выделенного beta-стенда")
    config = json.loads(Path(filename).read_text(encoding="utf-8-sig"))
    remote = _connect(config, tmp_path / "client")
    protected = _protected_snapshot(remote, config["protected_project_id"])
    active = {"queued", "staging", "running", "recovery_required"}
    assert not [job for job in remote.plugin_jobs() if job.state.value in active]
    created = []
    try:
        yield remote, created
    finally:
        owned = {str(project_id) for project_id in created}
        for job in remote.plugin_jobs():
            if str(job.project_id) in owned and job.state.value in active:
                remote.cancel_plugin_job(
                    principal=remote.auth.principal,
                    gateway=None,
                    job=job,
                    idempotency_key=str(uuid4()),
                )
        for project_id in created:
            current = remote.get_project(project_id)
            if current is not None and current.state.value != "archived":
                remote.archive_project(
                    principal=remote.auth.principal,
                    project=current,
                    idempotency_key=str(uuid4()),
                )
        assert not [job for job in remote.plugin_jobs() if job.state.value in active]
        assert _protected_snapshot(remote, config["protected_project_id"]) == protected


def test_s05_project_crud_and_revision_conflict(live_palette):
    remote, created = live_palette
    principal = remote.auth.principal
    project = remote.create_project(
        principal=principal,
        name=f"palette-s05-{uuid4().hex[:12]}",
        width=2,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key=str(uuid4()),
    )
    created.append(project.id)
    assert project.id in {item.id for item in remote.list_projects()}
    assert remote.get_project(project.id).name == project.name
    workspace = remote._call("GET", f"/api/v1/projects/{project.id}/workspace")
    assert workspace["project_id"] == str(project.id)

    renamed = remote.rename_project(
        principal=principal,
        project=project,
        name=project.name + "-renamed",
        idempotency_key=str(uuid4()),
    )
    assert renamed.name.endswith("-renamed")
    archived = remote.archive_project(principal=principal, project=renamed, idempotency_key=str(uuid4()))
    assert project.id not in {item.id for item in remote.list_projects()}
    assert project.id in {item.id for item in remote.list_projects(include_archived=True)}
    restored = remote.restore_project(principal=principal, project=archived, idempotency_key=str(uuid4()))
    assert project.id in {item.id for item in remote.list_projects()}

    with pytest.raises(RemoteServerError) as conflict:
        remote._call(
            "PATCH",
            f"/api/v1/projects/{project.id}",
            payload={"name": "stale rename"},
            idempotency_key=str(uuid4()),
            if_match=project.revision,
        )
    assert conflict.value.status == 409
    assert conflict.value.code == "revision.conflict"
    assert conflict.value.problem.get("actual_revision") == restored.revision

    remote.request_project_deletion(project=restored, reason="W3 acceptance")
    assert remote.project_deletion_requests(project.id)
    assert remote.get_project(project.id) is not None


def test_s06_layer_routes_full_surface(live_palette, tmp_path):
    remote, created = live_palette
    principal = remote.auth.principal
    project = remote.create_project(
        principal=principal,
        name=f"palette-s06-{uuid4().hex[:12]}",
        width=2,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key=str(uuid4()),
    )
    created.append(project.id)
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (16, 16), "white").save(source / "1_1.png")
    scan = remote.scan_layer_source(source, maximum_frames=2, project_id=project.id)
    assert scan.total_files == 1 and not scan.issues

    external, binding, _ = remote.create_external_layer(
        principal=principal,
        project=project,
        name="External",
        layer_type=LayerType.METAL,
        order=0,
        image_directory=source,
        ssc_directory=None,
        prv_directory=None,
        idempotency_key=str(uuid4()),
    )
    assert binding.image_directory
    imported, imported_binding, _ = remote.create_layer_from_disk(
        principal=principal,
        project=remote.get_project(project.id),
        name="Imported",
        layer_type=LayerType.METAL,
        order=1,
        scan=scan,
        conversion=ImageConversionSettings(target_format="png"),
        idempotency_key=str(uuid4()),
    )
    assert imported_binding.image_directory
    base = f"/api/v1/projects/{project.id}/layers"
    assert {item.id for item in remote.list_layers(project.id)} == {external.id, imported.id}
    imported = next(item for item in remote.list_layers(project.id) if item.id == imported.id)
    renamed = remote._call(
        "POST", f"{base}/{imported.id}/rename", payload={"name": "Imported renamed"},
        idempotency_key=str(uuid4()), if_match=imported.revision,
    )
    reordered = remote._call(
        "POST", f"{base}/{imported.id}/reorder", payload={"order": 0},
        idempotency_key=str(uuid4()), if_match=renamed["revision"],
    )
    assert reordered["order"] == 0
    external = next(item for item in remote.list_layers(project.id) if item.id == external.id)
    archived = remote._call(
        "POST", f"{base}/{external.id}/archive", payload={},
        idempotency_key=str(uuid4()), if_match=external.revision,
    )
    assert archived["state"] == "archived"
    remote._call(
        "POST", f"{base}/{imported.id}/delete", payload={"confirmation_name": "Imported renamed"},
        idempotency_key=str(uuid4()), if_match=reordered["revision"],
    )
    assert not remote.list_layers(project.id)
    with pytest.raises(RemoteServerError) as missing:
        remote._call("GET", f"{base}/{imported.id}/files")
    assert missing.value.status == 404
    assert any(event.event_type == "LayerDeletedV1" for event in remote.history(project.id))
    assert remote.get_project(project.id) is not None


def test_s07_representations_import_viewport_and_stale_upload(live_palette, tmp_path):
    remote, created = live_palette
    principal = remote.auth.principal
    project = remote.create_project(
        principal=principal,
        name=f"palette-s07-{uuid4().hex[:12]}",
        width=2,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key=str(uuid4()),
    )
    created.append(project.id)
    layer = remote.create_layer(
        principal=principal, project=project, name="Images", layer_type=LayerType.METAL,
        order=0, idempotency_key=str(uuid4()),
    )
    representation = remote.create_representation(
        principal=principal, project=remote.get_project(project.id), layer=layer,
        name="Source", kind=RepresentationKind.IMAGE, active=True,
        idempotency_key=str(uuid4()),
    )
    source = tmp_path / "images"
    source.mkdir()
    image = source / "1_1.png"
    Image.new("RGB", (16, 16), "white").save(image)
    plan = remote.plan_import_directory(project=project, directory=source, kind=RepresentationKind.IMAGE)
    assert len(plan.items) == 1 and plan.ready
    imported = remote.commit_managed_import(
        principal=principal, project=project, layer=layer, representation=representation,
        plan=plan, idempotency_key=str(uuid4()),
    )
    assert len(imported.versions) == 1
    linked = tmp_path / "linked"
    linked.mkdir()
    Image.new("RGB", (16, 16), "white").save(linked / "0000.jpg")
    external, _, external_representation = remote.create_external_layer(
        principal=principal, project=remote.get_project(project.id), name="Linked source",
        layer_type=LayerType.METAL, order=1, image_directory=linked,
        ssc_directory=None, prv_directory=None, idempotency_key=str(uuid4()),
    )
    assert external_representation is not None
    prepared = remote.materialize_representation_inputs(
        principal=principal, project=project, layer=external, representation=external_representation,
    )
    assert any(cell.x == 1 and cell.y == 1 and cell.sha256 for cell in prepared)
    base = f"/api/v1/projects/{project.id}"
    frames = remote._call(
        "GET", f"{base}/frames?layer_id={layer.id}&representation_id={representation.id}"
    )
    assert any(item["x"] == 1 and item["y"] == 1 for item in frames["items"])
    viewport = remote._call(
        "GET", f"{base}/viewport?layer_id={layer.id}&representation_id={representation.id}&x1=1&y1=1&x2=2&y2=1"
    )
    assert viewport["cells"]
    series = remote.list_artifact_series(project.id, layer_id=layer.id, representation_id=representation.id)
    assert series
    series_id = series[0].id
    assert remote.artifact_stream_revision(project.id, series_id) > 0
    with pytest.raises(RemoteServerError) as conflict:
        remote._call(
            "POST", f"{base}/artifacts/{series_id}/uploads",
            payload={
                "filename": image.name,
                "media_type": "image/png",
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "size_bytes": image.stat().st_size,
                "parameters": {},
            },
            idempotency_key=str(uuid4()), if_match=0,
        )
    assert conflict.value.status == 409
    assert conflict.value.code == "revision.conflict"


def _stand_config():
    filename = os.environ.get("KRAKEN_LIVE_WORKER_CONFIG")
    assert filename, "Нужен KRAKEN_LIVE_WORKER_CONFIG"
    return json.loads(Path(filename).read_text(encoding="utf-8-sig"))


def _register_account(server_url, directory, username):
    from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteHttpClient, RemoteServerProjectService
    from kraken_hub.infrastructure.remote.parts.principal_server_payload import _principal_from_server_payload

    http = RemoteHttpClient(server_url)
    password = f"Palette-{uuid4().hex}"
    session = http.request(
        "POST",
        "/api/v1/auth/accounts",
        token="register",
        payload={"username": username, "password": password, "display_name": "Палитра Наблюдатель"},
    )
    token = session["access_token"]
    principal = _principal_from_server_payload(http.request("GET", "/api/v1/session", token=token))
    return RemoteServerProjectService(
        server_url,
        auth=RemoteAuth(access_token=token, principal=principal),
        data_dir=directory,
        http=http,
    )


def _palette_project(remote, created, prefix):
    project = remote.create_project(
        principal=remote.auth.principal,
        name=f"{prefix}-{uuid4().hex[:8]}",
        width=2,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key=str(uuid4()),
    )
    created.append(project.id)
    return project


def _ws_url(server_url):
    if server_url.startswith("https://"):
        return "wss://" + server_url.removeprefix("https://") + "/api/v1/ws"
    return "ws://" + server_url.removeprefix("http://") + "/api/v1/ws"


def _ws_until(socket, predicate, timeout=10):
    import time

    deadline = time.monotonic() + timeout
    seen = []
    while time.monotonic() < deadline:
        socket.settimeout(max(0.1, deadline - time.monotonic()))
        try:
            message = json.loads(socket.recv())
        except Exception as exc:
            if type(exc).__name__ in {"WebSocketTimeoutException", "timeout", "TimeoutError"}:
                if seen and time.monotonic() >= deadline:
                    break
                continue
            raise
        seen.append(message)
        if predicate(message):
            return message, seen
    return None, seen


def test_s08_project_command_accepted_and_forbidden(live_palette, tmp_path):
    from kraken_manager.application.dto import CommandContext, CreateLayerCommand

    remote, created = live_palette
    project = _palette_project(remote, created, "palette-s08-cmd")
    before = {event.event_id for event in remote.history(project.id)}
    layer = remote.execute_project_command(
        CreateLayerCommand(
            context=CommandContext(actor=remote.auth.principal, idempotency_key=str(uuid4())),
            project_id=project.id,
            name="Commanded",
            type=LayerType.METAL,
            order=0,
            expected_project_revision=project.revision,
        )
    )
    assert layer.name == "Commanded"
    assert any(event.event_id not in before for event in remote.history(project.id))
    guest = _register_account(
        _stand_config()["server_url"], tmp_path / "guest", f"palette-norole-{uuid4().hex[:6]}"
    )
    assert guest.auth.principal.id != remote.auth.principal.id
    with pytest.raises(RemoteServerError) as denied:
        guest.execute_project_command(
            CreateLayerCommand(
                context=CommandContext(actor=guest.auth.principal, idempotency_key=str(uuid4())),
                project_id=project.id,
                name="Forbidden",
                type=LayerType.METAL,
                order=1,
                expected_project_revision=remote.get_project(project.id).revision,
            )
        )
    assert denied.value.status == 403


def test_s08_websocket_ping_subscribe_unsubscribe(live_palette, tmp_path):
    import websocket

    remote, created = live_palette
    config = _stand_config()
    project = _palette_project(remote, created, "palette-s08-ws")
    other = _connect(config, tmp_path / "second-http")
    socket = websocket.create_connection(
        _ws_url(config["server_url"]),
        header=[f"Authorization: Bearer {remote.auth.access_token}"],
        timeout=5,
    )
    try:
        assert json.loads(socket.recv())["type"] == "connected"
        socket.send(json.dumps({"type": "ping"}))
        assert json.loads(socket.recv())["type"] == "pong"
        socket.send(json.dumps({"type": "subscribe", "project_id": str(project.id), "catalog": True}))
        subscribed, _ = _ws_until(socket, lambda message: message.get("type") == "subscribed")
        assert subscribed is not None and subscribed["catalog"] is True
        other.rename_project(
            principal=other.auth.principal,
            project=other.get_project(project.id),
            name=project.name + "-ws",
            idempotency_key=str(uuid4()),
        )
        changed, _ = _ws_until(socket, lambda message: message.get("type") == "catalog_changed")
        assert changed is not None and changed["project_id"] == str(project.id)
        socket.send(json.dumps({"type": "unsubscribe", "project_id": str(project.id), "catalog": True}))
        unsubscribed, _ = _ws_until(socket, lambda message: message.get("type") == "unsubscribed")
        assert unsubscribed is not None
        _ws_until(socket, lambda _message: False, timeout=1.5)
        other.rename_project(
            principal=other.auth.principal,
            project=other.get_project(project.id),
            name=project.name + "-silent",
            idempotency_key=str(uuid4()),
        )
        leaked, _ = _ws_until(socket, lambda message: message.get("type") == "catalog_changed", timeout=2)
        assert leaked is None
    finally:
        socket.close()


def test_s08_notes_checks_package_return(live_palette, tmp_path):
    from tests.test_live_server_worker_acceptance import _upload_frame

    remote, created = live_palette
    principal = remote.auth.principal
    project = _palette_project(remote, created, "palette-s08-notes")
    layer = remote.create_layer(
        principal=principal, project=project, name="Checked", layer_type=LayerType.METAL,
        order=0, idempotency_key=str(uuid4()),
    )
    note = remote.create_note(
        principal=principal, project_id=project.id, body="заметка палитры",
        layer_id=layer.id, idempotency_key=str(uuid4()),
    )
    revised = remote.revise_note(
        principal=principal, note=note, body="заметка обновлена", idempotency_key=str(uuid4()),
    )
    assert revised.body == "заметка обновлена"
    image = remote.create_representation(
        principal=principal, project=remote.get_project(project.id), layer=layer,
        name="Source", kind=RepresentationKind.IMAGE, active=True, idempotency_key=str(uuid4()),
    )
    source = tmp_path / "1_1.png"
    Image.new("RGB", (32, 32), "white").save(source)
    _upload_frame(remote, project.id, layer.id, image.id, source, 1, 1)
    layer = next(item for item in remote.list_layers(project.id) if item.id == layer.id)
    vector = remote.create_representation(
        principal=principal, project=remote.get_project(project.id), layer=layer,
        name="Vectors", kind=RepresentationKind.VECTOR, source_image_representation_id=image.id,
        active=True, idempotency_key=str(uuid4()),
    )
    cif = tmp_path / "1_1.cif"
    cif.write_bytes(b"CIF palette\n")
    _store_frame(remote, project.id, layer.id, vector.id, cif, 1, 1, "application/x-cif")
    performer = next(
        item for item in remote.list_performers() if str(item.principal_id) == str(principal.id)
    )
    batch = remote.create_review_batch(
        principal=principal, project_id=project.id, layer_id=layer.id,
        image_representation_id=image.id, vector_representation_id=vector.id,
        coordinates=((1, 1),), assignee_id=performer.id, instructions="проверка",
        idempotency_key=str(uuid4()),
    )
    package = tmp_path / "check-package"
    remote.export_review_batch(
        principal=principal, batch=batch, destination=package, idempotency_key=str(uuid4()),
    )
    assert (package / "kraken-review.json").is_file()
    current, _plan = remote.review_return_preflight(
        principal=principal, source=package, idempotency_key=str(uuid4()),
    )
    committed = remote.commit_review_return(
        principal=principal, batch=current, source=package, idempotency_key=str(uuid4()),
    )
    assert committed is not None
    order = remote.issue_work(
        principal=principal, project_id=project.id, layer_id=layer.id, kind="vectorize",
        placement="server", assignee_id=performer.id, coordinates=((1, 1),),
        image_representation_id=str(image.id), vector_representation_id=str(vector.id),
        idempotency_key=str(uuid4()),
    )
    listed = remote._call("GET", f"/api/v1/projects/{project.id}/work-orders")
    assert any(item["id"] == str(order.id) for item in listed["items"])


def _store_frame(remote, project_id, layer_id, representation_id, source, x, y, media_type):
    from kraken_manager.domain.artifacts import deterministic_frame_series_id
    from kraken_manager.domain.project import FrameCoordinate

    frame_id = FrameCoordinate(x, y).frame_id(project_id)
    series = remote._call(
        "POST", f"/api/v1/projects/{project_id}/artifacts",
        payload={
            "name": source.stem, "scope": "frame_representation", "layer_id": str(layer_id),
            "representation_id": str(representation_id), "frame_id": str(frame_id),
            "series_id": str(deterministic_frame_series_id(representation_id, frame_id)),
        },
        idempotency_key=str(uuid4()),
    )
    metadata = {
        "filename": source.name, "media_type": media_type,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "size_bytes": source.stat().st_size,
        "parameters": {"x": x, "y": y, "mapping": "preflight"},
    }
    key = str(uuid4())
    revision = remote.artifact_stream_revision(project_id, series["id"])
    base = f"/api/v1/projects/{project_id}/artifacts/{series['id']}/uploads"
    transfer = remote._call("POST", base, payload=metadata, idempotency_key=key, if_match=revision)
    remote._http.upload_url(transfer["url"], token=transfer["token"], source=source)
    remote._call(
        "POST", base + "/complete", payload={**metadata, "upload_token": transfer["token"]},
        idempotency_key=key, if_match=revision,
    )


def test_s08_jobs_agent_protocol_error_branches(live_palette, tmp_path):
    import sys
    import threading
    from dataclasses import replace
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from kraken_agent.runner import PluginProcessSpec, PluginRegistry
    from kraken_server.configuration import unprotect_secret
    from tests.test_live_server_worker_acceptance import ObservedWorker, _upload_frame

    remote, created = live_palette
    config = _stand_config()
    principal = remote.auth.principal
    project = _palette_project(remote, created, "palette-s08-job")
    layer = remote.create_layer(
        principal=principal, project=project, name="Queued", layer_type=LayerType.METAL,
        order=0, idempotency_key=str(uuid4()),
    )
    image = remote.create_representation(
        principal=principal, project=project, layer=layer, name="Source",
        kind=RepresentationKind.IMAGE, active=True, idempotency_key=str(uuid4()),
    )
    source = tmp_path / "1_1.png"
    Image.new("RGB", (64, 64), "white").save(source)
    _upload_frame(remote, project.id, layer.id, image.id, source, 1, 1)
    layer = remote.list_layers(project.id)[0]
    vector = remote.create_representation(
        principal=principal, project=remote.get_project(project.id), layer=layer, name="Contour",
        kind=RepresentationKind.VECTOR, source_image_representation_id=image.id, active=True,
        idempotency_key=str(uuid4()),
    )
    job = remote.submit_plugin_job(
        principal=principal, gateway=None, project_id=project.id, layer_id=layer.id,
        source_representation_id=image.id, target_representation_id=vector.id,
        coordinates=[(1, 1)], capability="frames.vectorize.v1",
        parameters={"input_transport": "server", "execution_mode": "headless", "execution_host": "server"},
        idempotency_key=str(uuid4()),
    )
    assert job.state.value == "queued"
    with pytest.raises(RemoteServerError) as stale:
        remote.cancel_plugin_job(
            principal=principal, gateway=None, job=replace(job, revision=job.revision + 99),
            idempotency_key=str(uuid4()),
        )
    assert stale.value.status == 409
    assert next(item for item in remote.plugin_jobs() if item.id == job.id).state.value == "queued"
    denied = Request(
        config["server_url"] + "/api/v1/agent/lease",
        data=b'{"capabilities":["frames.vectorize.v1"]}',
        method="POST",
        headers={"Authorization": "Bearer garbage-palette-token", "Content-Type": "application/json"},
    )
    with pytest.raises(HTTPError) as unauthorized:
        urlopen(denied, timeout=10)
    assert unauthorized.value.code == 401

    def worker(name):
        created_worker = ObservedWorker(
            server_url=config["server_url"],
            token=unprotect_secret(Path(config["machine_secret"])),
            data_dir=tmp_path / name,
            registry=PluginRegistry({
                "frames.vectorize.v1": PluginProcessSpec(
                    "frames.vectorize.v1", (sys.executable, "-m", "contour.worker")
                )
            }),
        )
        created_worker.transitions = []
        created_worker.allowed_jobs = {str(job.id)}
        return created_worker

    first, second = worker("a"), worker("b")
    results = {}

    def claim(name, agent):
        try:
            results[name] = agent._request(
                "POST", f"/api/v1/agent/jobs/{job.id}/lease",
                {"capabilities": ["frames.vectorize.v1"], "lease_seconds": 60},
            )
        except Exception as exc:
            results[name] = exc

    try:
        threads = [
            threading.Thread(target=claim, args=("a", first)),
            threading.Thread(target=claim, args=("b", second)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        def won(value):
            return isinstance(value, dict) and isinstance(value.get("job"), dict)

        winners = [name for name, value in results.items() if won(value)]
        assert len(winners) == 1
        loser = results["b" if winners[0] == "a" else "a"]
        if isinstance(loser, HTTPError):
            assert loser.code == 409
        else:
            assert not won(loser)
        winner = first if winners[0] == "a" else second
        lease = results[winners[0]]["job"]
        winner.run_leased_job(lease)
        done = next(item for item in remote.plugin_jobs() if item.id == job.id)
        assert done.state.value == "succeeded", (done.state, done.error)
        winner.headers["X-Kraken-Lease-Claim"] = lease["lease_claim"]
        with pytest.raises(HTTPError) as duplicate:
            winner._request("POST", f"/api/v1/agent/jobs/{job.id}/complete", {})
        assert duplicate.value.code == 403
    finally:
        first.stop()
        second.stop()


def test_s08_plans_crud_and_revision_conflict(live_palette):
    remote, created = live_palette
    project = _palette_project(remote, created, "palette-s08-plan")
    base = f"/api/v1/projects/{project.id}"
    plan = remote._call(
        "POST", f"{base}/work-plans", payload={"title": "Палитра"}, idempotency_key=str(uuid4()),
    )
    assert plan["title"] == "Палитра"
    template = remote._call(
        "POST", f"{base}/work-plan-templates",
        payload={"title": "Шаблон", "tasks": [{"key": "scan", "title": "Скан", "multiplicity": "layer"}]},
        idempotency_key=str(uuid4()), if_match=0,
    )
    assert template["title"] == "Шаблон"
    analytics = remote._call("POST", f"{base}/work-planning/analytics", payload={})
    assert "items" in analytics or "current_compute_resources" in analytics
    added = remote._call(
        "POST", f"{base}/work-plans/{plan['id']}/commands",
        payload={"operation": "add_item", "arguments": {"item": {"title": "Сегмент"}}},
        idempotency_key=str(uuid4()), if_match=plan["revision"],
    )
    with pytest.raises(RemoteServerError) as conflict:
        remote._call(
            "POST", f"{base}/work-plans/{plan['id']}/commands",
            payload={"operation": "add_item", "arguments": {"item": {"title": "Повтор"}}},
            idempotency_key=str(uuid4()), if_match=plan["revision"],
        )
    assert conflict.value.status == 409
    assert added["revision"] != plan["revision"]


def test_s08_history_and_statistics(live_palette):
    from datetime import UTC, datetime, timedelta
    from urllib.parse import quote

    remote, created = live_palette
    project = _palette_project(remote, created, "palette-s08-history")
    remote.rename_project(
        principal=remote.auth.principal, project=project, name=project.name + "-renamed",
        idempotency_key=str(uuid4()),
    )
    history = remote._call("GET", f"/api/v1/projects/{project.id}/history?limit=50")
    assert any(item["event_type"].startswith("Project") for item in history["items"])
    start = quote((datetime.now(UTC) - timedelta(hours=2)).isoformat())
    end = quote((datetime.now(UTC) + timedelta(minutes=5)).isoformat())
    report = remote._call(
        "GET", f"/api/v1/projects/{project.id}/statistics?start={start}&end={end}&timezone=Europe/Moscow"
    )
    assert "metrics" in report and "series" in report
