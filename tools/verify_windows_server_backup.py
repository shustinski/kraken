"""Exercise frozen backup/recovery on real files and two disposable databases.

Only this invocation's UUID databases are created/dropped. No model is trained.
The supplied corpus is read only; credentials and transfer tickets stay out of
reports and subprocess arguments. This is a file installation/process check,
not an SCM, reboot, native UI or physical topology acceptance check.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
from uuid import uuid4

import httpx
from psycopg import sql
from sqlalchemy.engine import make_url

from kraken_admin.server_backup import _connect
from kraken_core.media_types import artifact_media_type
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.project import deterministic_frame_id
from kraken_server.configuration import ServerConfig, write_config
from verify_windows_server_distribution import free_port


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--pg-bin-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.distribution.resolve(strict=True)
    binaries = {name: (root / (name + ".exe")).resolve(strict=True)
                for name in ("KrakenAdmin", "KrakenServer", "KrakenBlobGateway")}
    manifest = args.reference_manifest.resolve(strict=True)
    pairs = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not pairs:
        raise ValueError("A nonempty real reference manifest is required")
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=False)
    evidence = dict(status="running", training=False, fixture_kind="real JPEG/CIF pairs; declared catalog grid",
                    reference_manifest_sha256=sha(manifest), pairs=len(pairs), checks=[],
                    binaries={name: dict(sha256=sha(path), size_bytes=path.stat().st_size) for name, path in binaries.items()})
    started = time.monotonic()
    environment = dict(os.environ)
    environment.pop("KRAKEN_SERVER_DEVELOPMENT", None)
    environment["KRAKEN_INITIAL_ADMIN_PASSWORD"] = secrets.token_urlsafe(32)
    names = ["kraken_backup_deploy_" + uuid4().hex for _ in range(2)]
    parent_url = os.environ["KRAKEN_DISTRIBUTION_AUDIT_DSN"]
    urls = [make_url(parent_url).set(database=name).render_as_string(hide_password=False) for name in names]
    evidence["owned_databases"] = names
    created, logs, source_files, versions = [], [], {}, []
    process, failure = None, None

    def passed(name, **details):
        evidence["checks"].append(dict(name=name, **details))
        print(name, flush=True)

    def command(label, arguments):
        with (report / (label + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run([str(binaries["KrakenAdmin"]), *map(str, arguments)],
                env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=150,
                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise RuntimeError(f"{label} exited {result.returncode}; inspect its log")
        text = (report / (label + ".log")).read_text(encoding="utf-8-sig")
        passed(label, exit_code=0)
        return json.loads(text.splitlines()[-1]) if text.strip().endswith("}") else None

    def stop():
        nonlocal process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.stdin.write("stop\n")
                process.stdin.flush()
            code = process.wait(timeout=30)
            if code:
                raise RuntimeError(f"Server exited {code}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            process.stdin.close()
            process = None

    def launch(config, label):
        nonlocal process
        log = (report / (label + ".log")).open("w", encoding="utf-8")
        logs.append(log)
        process = subprocess.Popen([str(binaries["KrakenServer"]), "--config", str(config.path), "--admin-control"],
            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW)
        client = httpx.Client(base_url=f"http://127.0.0.1:{config.port}", trust_env=False, timeout=20)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{label} stopped before health; inspect its log")
                try:
                    response = client.get("/api/v1/health")
                    if response.status_code == 200 and response.json().get("status") == "ok":
                        break
                except httpx.TransportError:
                    pass
                time.sleep(.1)
            else:
                raise TimeoutError(f"{label} health timed out")
            response = client.post("/api/v1/auth/sessions", json={"username": "backup-audit",
                "password": environment["KRAKEN_INITIAL_ADMIN_PASSWORD"]})
            if response.status_code != 200:
                raise RuntimeError(f"{label} authentication: {response.status_code}")
            client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
            gateway = httpx.get(config.blob_gateway_public_url + "/health", trust_env=False, timeout=10)
            assert gateway.status_code == 200 and gateway.json()["storage_backup_barrier"] == 1
            passed(label + "_authenticated", storage_backup_barrier=1)
            return client
        except BaseException:
            client.close()
            raise

    def request(client, method, path, *, expected=200, **options):
        response = client.request(method, path, **options)
        if response.status_code != expected:
            # Bodies can contain tokens. Keep status/path only.
            raise RuntimeError(f"{method} {path}: expected {expected}, got {response.status_code}")
        return response.json()

    def verify_files(client, prefix, label):
        for item in versions:
            ticket = request(client, "GET", prefix + f"/artifacts/versions/{item['version_id']}/download-ticket")
            headers = {"Authorization": "Bearer " + ticket["token"]}
            response = httpx.get(ticket["url"], headers=headers, trust_env=False, timeout=20)
            assert response.status_code == 200
            assert hashlib.sha256(response.content).hexdigest() == item["sha256"]
            partial = httpx.get(ticket["url"], headers={**headers, "Range": "bytes=7-1023"}, trust_env=False, timeout=20)
            assert partial.status_code == 206 and partial.content == response.content[7:1024]
        passed(label, files=len(versions), full_sha256=True, range=True)

    try:
        for pair in pairs:
            for kind in ("image", "vector"):
                path = Path(pair[kind]).resolve(strict=True)
                expected_sha = pair[kind + "_sha256"]
                assert sha(path) == expected_sha, f"Reference bytes changed: {path}"
                source_files[path] = expected_sha
        with _connect(parent_url) as administrator:
            for name in names:
                administrator.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(name)))
                created.append(name)
        port, gateway_port = free_port(), free_port()
        config = write_config(report / "original-config" / "server.toml", database_url=urls[0],
            blob_root=report / "original-blobs", source_root=report / "original-source",
            derived_root=report / "original-derived", port=port, log_level="low",
            blob_gateway_public_url=f"http://127.0.0.1:{gateway_port}", blob_gateway_bind=f"127.0.0.1:{gateway_port}",
            blob_gateway_executable=binaries["KrakenBlobGateway"])
        command("packaged_backup_migrations", ["migrate", "--config", config.path])
        command("packaged_backup_bootstrap", ["bootstrap-admin", "--config", config.path,
            "--username", "backup-audit", "--display-name", "Own backup acceptance"])
        with closing(launch(config, "source")) as client:
            project = request(client, "POST", "/api/v1/projects", expected=201,
                headers={"Idempotency-Key": "backup-project"},
                json=dict(name="Real corpus recovery", width=4, height=(len(pairs) + 3) // 4, orientation="y_down"))
            pid = project.get("project_id", project.get("id"))
            prefix = f"/api/v1/projects/{pid}"
            layer = request(client, "POST", prefix + "/layers", expected=201,
                headers={"Idempotency-Key": "backup-layer", "If-Match": "1"}, json=dict(name="M1", type="metal", order=0))
            lid = layer.get("layer_id", layer.get("id"))
            representations = {}
            for kind in ("image", "vector"):
                layer_revision = next(item["revision"] for item in request(client, "GET", prefix + "/layers")["items"]
                                      if item.get("layer_id", item.get("id")) == lid)
                representation = request(client, "POST", prefix + f"/layers/{lid}/representations", expected=201,
                    headers={"Idempotency-Key": "backup-representation-" + kind, "If-Match": str(layer_revision)},
                    json=dict(name=kind, kind=kind, purpose="source" if kind == "image" else "vector", active=True,
                              source_image_representation_id=representations.get("image") if kind == "vector" else None))
                representations[kind] = representation.get("representation_id", representation.get("id"))
            for index, pair in enumerate(pairs):
                frame_id = str(deterministic_frame_id(pid, index % 4 + 1, index // 4 + 1))
                for kind in ("image", "vector"):
                    path = Path(pair[kind])
                    series = request(client, "POST", prefix + "/artifacts", expected=201,
                        headers={"Idempotency-Key": f"series-{index}-{kind}"},
                        json=dict(name=path.name, scope="frame_representation", layer_id=lid,
                            representation_id=representations[kind], frame_id=frame_id,
                            series_id=str(deterministic_frame_series_id(representations[kind], frame_id))))
                    sid = series.get("id", series.get("series_id"))
                    upload_path = prefix + f"/artifacts/{sid}/uploads"
                    revision = request(client, "GET", prefix + f"/artifacts/{sid}/revision")["revision"]
                    headers = {"Idempotency-Key": f"upload-{index}-{kind}", "If-Match": str(revision)}
                    payload = dict(filename=path.name, media_type=artifact_media_type(path.name),
                                   sha256=pair[kind + "_sha256"], size_bytes=path.stat().st_size,
                                   parameters=dict(x=index % 4 + 1, y=index // 4 + 1, mapping="declared_catalog_grid"))
                    ticket = request(client, "POST", upload_path, headers=headers, json=payload)
                    assert ticket["mode"] == "gateway"
                    with path.open("rb") as stream:
                        response = httpx.put(ticket["url"], content=stream, trust_env=False, timeout=30,
                            headers={"Authorization": "Bearer " + ticket["token"], "Content-Length": str(payload["size_bytes"])})
                    assert response.status_code == 200
                    completed = request(client, "POST", upload_path + "/complete", expected=201, headers=headers,
                        json={**payload, "upload_token": ticket["token"]})
                    versions.append(dict(kind=kind, frame_id=frame_id, sha256=payload["sha256"],
                        version_id=completed.get("id", completed.get("artifact_version_id"))))
            passed("real_pairs_uploaded", files=len(versions))
            matrices = {kind: request(client, "GET", prefix + "/frames", params=dict(layer_id=lid, representation_id=rid))
                        for kind, rid in representations.items()}
            evidence["source_matrices"] = matrices
            for kind, matrix in matrices.items():
                expected = {v["frame_id"]: v["sha256"] for v in versions if v["kind"] == kind}
                assert {v["frame_id"]: v["sha256"] for v in matrix["items"] if not v.get("missing")} == expected
            passed("source_real_pair_matrix", cells=len(pairs), representations=2)
            verify_files(client, prefix, "source_real_files")
            review_key = config.blob_root / ".server" / "review-ed25519.key"
            key_digest = sha(review_key)
            # Concrete files in all configured roots prove recovery covers more
            # than content-addressed objects. They are this invocation's bytes.
            (config.source_root / "own-source-note.txt").write_bytes(b"source tree acceptance")
            (config.derived_root / "own-derived-note.txt").write_bytes(b"derived tree acceptance")
            archive = report / "archive"
            command("packaged_backup_create", ["backup-server", "--config", config.path,
                "--destination", archive, "--pg-bin-dir", args.pg_bin_dir])
            command("packaged_backup_verify", ["backup-verify", "--backup", archive])
        stop()
        environment["KRAKEN_RESTORE_DATABASE_URL"] = urls[1]
        restore_arguments = ["restore-server", "--backup", archive, "--destination", report / "restored-data",
            "--config-dir", report / "restored-config", "--pg-bin-dir", args.pg_bin_dir,
            "--blob-gateway-executable", binaries["KrakenBlobGateway"]]
        result = command("packaged_backup_restore", restore_arguments)
        repeated = command("packaged_backup_restore_resume", [*restore_arguments, "--resume"])
        assert repeated["backup_id"] == result["backup_id"]
        recovered = ServerConfig.load(result["config"])
        assert sha(recovered.blob_root / ".server" / "review-ed25519.key") == key_digest
        assert (recovered.source_root / "own-source-note.txt").read_bytes() == b"source tree acceptance"
        assert (recovered.derived_root / "own-derived-note.txt").read_bytes() == b"derived tree acceptance"
        assert recovered.blob_gateway_secret_value != config.blob_gateway_secret_value
        passed("restored_roots_and_keys", review_key_preserved=True, gateway_key_rotated=True)
        for ordinal in (1, 2):
            with closing(launch(recovered, f"recovered-{ordinal}")) as client:
                assert request(client, "GET", prefix)["name"] == "Real corpus recovery"
                for kind, rid in representations.items():
                    matrix = request(client, "GET", prefix + "/frames", params=dict(layer_id=lid, representation_id=rid))
                    assert matrix == matrices[kind]
                verify_files(client, prefix, f"recovered_real_files_{ordinal}")
            stop()
        passed("restored_restart_persistence", restarts=1, matrix_unchanged=True)
        evidence.update(status="passed", versions=versions)
    except BaseException as exc:
        failure = exc
        evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            stop()
        except BaseException as exc:
            failure = failure or exc
            evidence.update(status="failed", stop_error=f"{type(exc).__name__}: {exc}")
        try:
            with _connect(parent_url) as administrator:
                for name in created:
                    assert name.startswith("kraken_backup_deploy_") and len(name) == len("kraken_backup_deploy_") + 32
                    administrator.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
            evidence["owned_databases_removed"] = len(created) == len(names)
            evidence["source_files_unchanged"] = bool(source_files) and all(sha(path) == digest for path, digest in source_files.items())
            if not evidence["source_files_unchanged"]:
                raise RuntimeError("Reference corpus preservation was not established")
        except BaseException as exc:
            failure = failure or exc
            evidence.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
        for log in logs:
            log.close()
        evidence["seconds"] = round(time.monotonic() - started, 3)
        (report / "report.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if failure:
        raise failure
    print(json.dumps(dict(status=evidence["status"], seconds=evidence["seconds"], pairs=len(pairs))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
