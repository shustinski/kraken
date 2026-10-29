"""Exercise frozen Server/Admin/Gateway against a new disposable PG schema.

No Windows service is installed. The supplied database must be an audit database;
only this invocation's UUID schema is created and dropped. No credentials/tokens
are recorded in the report or subprocess command lines.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time
import uuid

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from kraken_server.configuration import write_config


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--diagnostic-trace", action="store_true")
    args = parser.parse_args()
    dsn = os.environ["KRAKEN_DISTRIBUTION_AUDIT_DSN"]
    distribution = args.distribution.resolve(strict=True)
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=False)
    executables = {
        name: (distribution / (name + ".exe")).resolve(strict=True)
        for name in ("KrakenServer", "KrakenAdmin", "KrakenBlobGateway")
    }
    evidence = dict(
        status="running",
        checks=[],
        binaries={
            name: dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
            for name, path in executables.items()
        },
    )
    schema = "distribution_audit_" + uuid.uuid4().hex
    evidence["schema"] = schema
    engine = create_engine(dsn)
    schema_engine = None
    process = None
    logs = []
    started = time.monotonic()

    def passed(name, **details):
        evidence["checks"].append(dict(name=name, **details))
        print(name, flush=True)

    def command(name, arguments, environment):
        with (report / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [str(executables["KrakenAdmin"]), *arguments],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=120,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see its log")
        passed(name, exit_code=result.returncode)

    def stop():
        nonlocal process
        if process is not None:
            if process.poll() is None:
                process.stdin.write("stop\n")
                process.stdin.flush()
            code = process.wait(timeout=45)
            process.stdin.close()
            process = None
            if code:
                raise RuntimeError(f"Packaged server exited {code}")

    try:
        with engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        url = make_url(dsn).update_query_dict({"options": "-csearch_path=" + schema})
        schema_dsn = url.render_as_string(hide_password=False)
        schema_engine = create_engine(schema_dsn)
        port, gateway_port = free_port(), free_port()
        config = write_config(
            report / "server.toml",
            database_url=schema_dsn,
            blob_root=report / "blobs",
            source_root=report / "source",
            derived_root=report / "derived",
            port=port,
            log_level="low",
            blob_gateway_public_url=f"http://127.0.0.1:{gateway_port}",
            blob_gateway_bind=f"127.0.0.1:{gateway_port}",
            blob_gateway_executable=executables["KrakenBlobGateway"],
        )
        environment = dict(os.environ)
        environment.pop("KRAKEN_SERVER_DEVELOPMENT", None)
        password = secrets.token_urlsafe(32)
        environment["KRAKEN_INITIAL_ADMIN_PASSWORD"] = password
        command("packaged_migrations", ["migrate", "--config", str(config.path)], environment)
        command(
            "packaged_bootstrap",
            [
                "bootstrap-admin",
                "--config",
                str(config.path),
                "--username",
                "packaged-audit",
                "--display-name",
                "Packaged distribution audit",
            ],
            environment,
        )
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base, timeout=15, trust_env=False) as client:

            def launch(ordinal):
                nonlocal process
                log = (report / f"server-{ordinal}.log").open("w", encoding="utf-8")
                logs.append(log)
                command_line = [str(executables["KrakenServer"]), "--config", str(config.path), "--admin-control"]
                if args.diagnostic_trace:
                    command_line += ["--diagnostic-trace", str(report / f"threads-{ordinal}.txt")]
                process = subprocess.Popen(
                    command_line,
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Packaged server stopped before health; see log")
                    try:
                        result = client.get("/api/v1/health")
                        if result.status_code == 200 and result.json().get("status") == "ok":
                            passed(f"packaged_http_start_{ordinal}", metadata=result.json().get("metadata"))
                            return
                    except httpx.TransportError:
                        pass
                    time.sleep(0.2)
                raise TimeoutError("Packaged server health did not become ready")

            def request(method, path, status=200, **options):
                request_started = time.monotonic()
                response = None
                try:
                    response = client.request(method, path, **options)
                finally:
                    evidence.setdefault("requests", []).append(
                        dict(
                            method=method,
                            path=path,
                            status=None if response is None else response.status_code,
                            seconds=round(time.monotonic() - request_started, 4),
                        )
                    )
                if response.status_code != status:
                    # Redact response bodies: authentication and ticket responses
                    # can contain bearer credentials.
                    raise RuntimeError(f"{method} {path}: expected {status}, got {response.status_code}")
                return response.json()

            launch(1)
            session = request(
                "POST", "/api/v1/auth/sessions", json={"username": "packaged-audit", "password": password}
            )
            client.headers["Authorization"] = "Bearer " + session["access_token"]
            passed("packaged_local_authentication")
            project = request(
                "POST",
                "/api/v1/projects",
                status=201,
                headers={"Idempotency-Key": "packaged-project"},
                json={"name": "Packaged distribution audit", "width": 2, "height": 2, "orientation": "y_down"},
            )
            project_id = project.get("project_id", project.get("id"))
            assert project_id
            prefix = f"/api/v1/projects/{project_id}"
            series = request(
                "POST",
                prefix + "/artifacts",
                status=201,
                headers={"Idempotency-Key": "packaged-series"},
                json={"name": "Storage acceptance", "scope": "project_attachment"},
            )
            series_id = series.get("id", series.get("series_id"))
            assert series_id
            payload = b"Kraken packaged storage acceptance\n" * 4096
            digest = hashlib.sha256(payload).hexdigest()
            upload = dict(
                filename="storage-probe.bin",
                media_type="application/octet-stream",
                sha256=digest,
                size_bytes=len(payload),
            )
            headers = {"Idempotency-Key": "packaged-upload", "If-Match": "1"}
            path = prefix + f"/artifacts/{series_id}/uploads"
            ticket = request("POST", path, headers=headers, json=upload)
            assert ticket["mode"] == "gateway"
            transfer = httpx.put(
                ticket["url"],
                content=payload,
                headers={"Authorization": "Bearer " + ticket["token"]},
                trust_env=False,
                timeout=20,
            )
            transfer.raise_for_status()
            completed = request(
                "POST",
                path + "/complete",
                status=201,
                headers=headers,
                json={**upload, "upload_token": ticket["token"]},
            )
            version_id = completed.get("id", completed.get("artifact_version_id"))
            assert version_id
            passed("packaged_gateway_upload", bytes=len(payload), sha256=digest)
            download_path = prefix + f"/artifacts/versions/{version_id}/download-ticket"

            def check_download(ordinal):
                ticket = request("GET", download_path)
                response = httpx.get(
                    ticket["url"],
                    headers={"Authorization": "Bearer " + ticket["token"], "Range": "bytes=7-1023"},
                    trust_env=False,
                    timeout=20,
                )
                assert response.status_code == 206
                assert response.content == payload[7:1024]
                passed(f"packaged_gateway_range_{ordinal}", bytes=len(response.content))

            check_download(1)
            # Exercise the actual SciPy operation inside the frozen Server, not
            # just availability in the source/build interpreter. This declared
            # tiny fixture carries no physical or electrical acceptance claim.
            from PIL import Image

            layer = request(
                "POST",
                prefix + "/layers",
                status=201,
                headers={"Idempotency-Key": "mask-layer", "If-Match": "1"},
                json={"name": "Declared synthetic runtime fixture", "type": "metal", "order": 0},
            )
            layer_id = layer.get("layer_id", layer.get("id"))
            # Seed only this invocation's UUID schema with the exact old
            # projection-only representation. The frozen HTTP backend must
            # capture it; the verifier never appends its canonical events.
            from kraken_manager.domain.identity import Performer
            from kraken_manager.domain.selection import FrameSelectionV1
            from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement
            from kraken_manager.infrastructure.postgres.performer_store import PostgresPerformerStore
            from kraken_manager.infrastructure.postgres.projection_store import PostgresProjectionStore

            performer = PostgresPerformerStore(schema_engine).create(
                Performer.create(name="Own deployment legacy fixture", color="#446688"))
            legacy_order = WorkOrder.create(project_id=project_id, layer_id=layer_id,
                kind=WorkKind.STITCH, placement=WorkPlacement.EXTERNAL, assignee_id=performer.id,
                created_by=session["principal"]["id"], destination=str(report / "absent-legacy-package"),
                frame_count=1, selection=FrameSelectionV1.rectangle(1, 1, 1, 1),
                instructions="Own deployment projection-only diagnostic")
            legacy_projection = PostgresProjectionStore(schema_engine)
            legacy_projection.save_work_order(legacy_order)
            legacy_order = legacy_order.mark_running()
            legacy_projection.save_work_order(legacy_order)
            migration_path = prefix + "/work-orders/legacy-migration"
            migration_plan = request("GET", migration_path)
            assert len(migration_plan["legacy"]) == 1 and len(migration_plan["legacy"][0]["observations"]) == 2
            migration_body = {"expected_plan_digest": migration_plan["digest"]}
            migration_headers = {"Idempotency-Key": "packaged-legacy-work-import"}
            migrated = request("POST", migration_path, headers=migration_headers, json=migration_body)
            assert migrated["migrated"] == 1
            assert request("POST", migration_path, headers=migration_headers, json=migration_body) == migrated
            assert legacy_projection.get_work_order(legacy_order.id) == legacy_order
            assert not (report / "absent-legacy-package").exists()
            assert request("GET", migration_path)["legacy"] == []
            passed("packaged_legacy_work_order_migration", imported=1, observations=2, idempotent_retry=True)
            mask_series = request(
                "POST",
                prefix + "/artifacts",
                status=201,
                headers={"Idempotency-Key": "mask-series"},
                json={"name": "Exact binary mask", "scope": "layer_attachment", "layer_id": layer_id},
            )
            mask_series_id = mask_series.get("id", mask_series.get("series_id"))
            image = Image.new("L", (8, 8), 0)
            image.putpixel((2, 2), 255)
            image.putpixel((3, 3), 255)
            encoded = BytesIO()
            image.save(encoded, format="PNG")
            mask_data = encoded.getvalue()
            mask_sha = hashlib.sha256(mask_data).hexdigest()
            upload = dict(
                filename="declared-mask.png", media_type="image/png", sha256=mask_sha, size_bytes=len(mask_data)
            )
            headers = {"Idempotency-Key": "mask-upload", "If-Match": "1"}
            mask_path = prefix + f"/artifacts/{mask_series_id}/uploads"
            ticket = request("POST", mask_path, headers=headers, json=upload)
            response = httpx.put(
                ticket["url"],
                content=mask_data,
                headers={"Authorization": "Bearer " + ticket["token"]},
                trust_env=False,
                timeout=20,
            )
            response.raise_for_status()
            completed = request(
                "POST",
                mask_path + "/complete",
                status=201,
                headers=headers,
                json={**upload, "upload_token": ticket["token"]},
            )
            mask_version_id = completed.get("id", completed.get("artifact_version_id"))

            def physical_command(operation, arguments, key):
                current = request("GET", prefix + "/multilayer")
                return request(
                    "POST",
                    prefix + "/multilayer/commands",
                    headers={"Idempotency-Key": key, "If-Match": str(current["model"]["revision"])},
                    json={"operation": operation, "arguments": arguments},
                )

            physical_command(
                "set_stack",
                {"layers": [dict(layer_id=layer_id, layer_type="metal", z_m=0, thickness_m=1e-7, conductive=True)]},
                "mask-stack",
            )
            physical_command(
                "calibrate",
                dict(
                    layer_id=layer_id,
                    artifact_version_id=mask_version_id,
                    pixel_size_m=[1e-6, 1e-6],
                    y_axis="down",
                    uncertainty_m=1e-9,
                ),
                "mask-calibration",
            )
            current = request("GET", prefix + "/electrical")
            electrical = request(
                "POST",
                prefix + "/electrical/commands",
                timeout=60,
                headers={"Idempotency-Key": "mask-extraction", "If-Match": str(current["electrical"]["revision"])},
                json={
                    "operation": "extract_conductors",
                    "expected_physical_revision": current["physical_model_revision"],
                    "arguments": dict(
                        layer_id=layer_id, mask_version_id=mask_version_id, connectivity=4, foreground_value=255
                    ),
                },
            )
            extraction = electrical["electrical"]["extractions"][-1]
            assert len(extraction["components"]) == 2
            assert not electrical["graph"]["nets"]
            passed(
                "packaged_scipy_binary_extraction",
                components=2,
                adjacency=4,
                fixture_sha256=mask_sha,
                electrical_acceptance="unknown",
            )
            stop()
            passed("packaged_graceful_stop_1", exit_code=0)
            launch(2)
            persisted = request("GET", prefix)
            assert persisted.get("project_id", persisted.get("id")) == project_id
            check_download(2)
            assert request("GET", prefix + "/electrical") == electrical
            assert request("GET", migration_path)["legacy"] == []
            assert legacy_projection.get_work_order(legacy_order.id) == legacy_order
            passed("packaged_legacy_work_restart_persistence")
            passed("packaged_electrical_restart_persistence")
            stop()
            passed("packaged_restart_persistence_and_stop", exit_code=0)
        evidence["status"] = "passed"
    except BaseException as exc:
        evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            stop()
        finally:
            for log in logs:
                log.close()
            if schema_engine is not None:
                schema_engine.dispose()
            with engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            engine.dispose()
            evidence.update(schema_removed=True, seconds=round(time.monotonic() - started, 3))
            (report / "report.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
