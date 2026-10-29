"""Upgrade only separately identified audit installations, retaining real data.

The two fixture installer versions are labels for the deployment test, not a
published Kraken release. Server uses only UUID databases created by this run;
the production AppId and service are never installed or modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
from uuid import UUID, uuid4

import httpx
from psycopg import sql
from sqlalchemy.engine import make_url

from kraken_admin.server_backup import _connect
from kraken_core.media_types import artifact_media_type
from kraken_server.configuration import ServerConfig, write_config
from verify_windows_installer import registration, service_hash
from verify_windows_server_backup import sha
from verify_windows_server_distribution import free_port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--update", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--pg-bin-dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    old, new = [json.loads(path.read_text(encoding="utf-8-sig")) for path in (args.baseline, args.update)]
    target, app_id = old["target"], old["audit_app_id"]
    if target not in {"Server", "Desktop"} or not app_id.startswith(f"Kraken{target}Audit-"):
        raise ValueError("Only owned audit identities may be upgraded")
    UUID(app_id.split("Audit-", 1)[1])
    if new["target"] != target or new["audit_app_id"] != app_id or old["variant"] != "audit" or new["variant"] != "audit":
        raise ValueError("Both installers must use the same explicit audit identity")
    if old["version"] == new["version"]:
        raise ValueError("Use distinct fixture installer versions")
    for fixture in (old, new):
        fixture["installer"] = Path(fixture["installer"]).resolve(strict=True)
        if sha(fixture["installer"]) != fixture["setup_sha256"]:
            raise ValueError("Installer changed after compilation")
    if registration(app_id) is not None:
        raise ValueError("Audit identity is already installed; preserve it")
    if target == "Server" and args.pg_bin_dir is None:
        raise ValueError("Server recovery requires PostgreSQL client tools")
    reference = args.reference_manifest.resolve(strict=True)
    pairs = [json.loads(line) for line in reference.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not pairs:
        raise ValueError("Real pairs are required")
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=False)
    install = report / "installed application"
    before_service = service_hash()
    evidence = dict(status="running", target=target, app_id=app_id, checks=[], training=False,
        fixture_versions=[old["version"], new["version"]], reference_manifest_sha256=sha(reference),
        service_before_sha256=before_service, installers=[dict(version=f["version"], sha256=f["setup_sha256"]) for f in (old, new)])
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    environment.pop("KRAKEN_SERVER_DEVELOPMENT", None)
    source_files, created, urls = {}, [], []
    parent_url, process, installed, failure = None, None, False, None
    started = time.monotonic()

    def passed(name, **details):
        evidence["checks"].append(dict(name=name, **details))
        print(name, flush=True)

    def run(command, label, *, expected=0, timeout=180):
        with (report / (label + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(list(map(str, command)), env=environment, stdout=log,
                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, timeout=timeout)
        if result.returncode != expected:
            raise RuntimeError(f"{label}: expected {expected}, got {result.returncode}; inspect its log")
        return result.returncode

    def owned_registration():
        record = registration(app_id)
        uninstaller = install / "unins000.exe"
        if (record is None or Path(record["InstallLocation"]).resolve() != install.resolve()
                or not uninstaller.is_file() or str(uninstaller).casefold() not in record["UninstallString"].casefold()):
            raise RuntimeError("Audit registration/uninstaller is not owned by this invocation")
        return uninstaller

    def setup(fixture, label, *, expected=0):
        code = run([fixture["installer"], "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER",
            f"/DIR={install}", "/TASKS=", f"/GROUP={app_id}", f"/LOG={report / (label + '-setup.log')}"],
            label, expected=expected)
        owned_registration()
        if service_hash() != before_service:
            raise RuntimeError("File installation changed an unrelated service")
        passed(label, exit_code=code)

    def expected_files(fixture):
        source = Path(fixture["build_root"]) / f"Kraken{target}"
        return {path.relative_to(source).as_posix(): sha(path) for path in source.rglob("*") if path.is_file()}

    def check_payload(expected, label):
        for relative, digest in expected.items():
            if sha(install / relative) != digest:
                raise RuntimeError(f"{label}: installed payload mismatch: {relative}")
        passed(label, files=len(expected))

    def admin(arguments, label):
        run([install / "KrakenAdmin.exe", *arguments], label)
        text = (report / (label + ".log")).read_text(encoding="utf-8-sig")
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
                raise RuntimeError(f"Owned process exited {code}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            if process.stdin:
                process.stdin.close()
            process = None

    def request(client, method, path, *, expected=200, **kwargs):
        response = client.request(method, path, **kwargs)
        if response.status_code != expected:
            raise RuntimeError(f"{method} {path}: expected {expected}, got {response.status_code}")
        return response.json()

    @contextmanager
    def server(config, label):
        nonlocal process
        with (report / (label + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(install / "KrakenServer.exe"), "--config", str(config.path), "--admin-control"],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True, env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{config.port}", timeout=20, trust_env=False) as client:
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError(f"{label} stopped before health")
                        try:
                            if client.get("/api/v1/health").status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(.1)
                    else:
                        raise TimeoutError(f"{label} health timed out")
                    auth = request(client, "POST", "/api/v1/auth/sessions", json=dict(username="upgrade-owner",
                        password=environment["KRAKEN_INITIAL_ADMIN_PASSWORD"]))
                    client.headers["Authorization"] = "Bearer " + auth["access_token"]
                    passed(label + "_authenticated")
                    yield client
            finally:
                stop()

    def files_readback(client, prefix, versions, label):
        for version in versions:
            ticket = request(client, "GET", prefix + f"/artifacts/versions/{version['id']}/download-ticket")
            headers = {"Authorization": "Bearer " + ticket["token"]}
            response = httpx.get(ticket["url"], headers=headers, timeout=20, trust_env=False)
            assert response.status_code == 200 and __import__("hashlib").sha256(response.content).hexdigest() == version["sha256"]
            response_range = httpx.get(ticket["url"], headers={**headers, "Range": "bytes=7-1023"}, timeout=20, trust_env=False)
            assert response_range.status_code == 206 and response_range.content == response.content[7:1024]
        passed(label, files=len(versions), sha256=True, range=True)

    try:
        for pair in pairs:
            for kind in ("image", "vector"):
                path = Path(pair[kind]).resolve(strict=True)
                assert sha(path) == pair[kind + "_sha256"]
                source_files[path] = pair[kind + "_sha256"]
        old_files, new_files = expected_files(old), expected_files(new)
        if old_files.get(f"Kraken{target}.exe") == new_files.get(f"Kraken{target}.exe"):
            raise ValueError("Baseline and update must contain different actual binaries")
        setup(old, "baseline_install")
        installed = True
        check_payload(old_files, "baseline_payload")
        user_note = install / "own-untracked-user-note.txt"
        user_note.write_bytes(b"This invocation's untracked data must survive update and uninstall.")
        note_sha = sha(user_note)

        if target == "Server":
            parent_url = environment["KRAKEN_DISTRIBUTION_AUDIT_DSN"]
            names = ["kraken_upgrade_audit_" + uuid4().hex for _ in range(2)]
            evidence["owned_databases"] = names
            urls = [make_url(parent_url).set(database=name).render_as_string(hide_password=False) for name in names]
            with _connect(parent_url) as administrator:
                for name in names:
                    administrator.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(name)))
                    created.append(name)
            environment["KRAKEN_INITIAL_ADMIN_PASSWORD"] = secrets.token_urlsafe(32)
            port, gateway_port = free_port(), free_port()
            config = write_config(report / "retained configuration" / "server.toml", database_url=urls[0],
                blob_root=report / "original-blobs", source_root=report / "original-source", derived_root=report / "original-derived",
                port=port, log_level="low", blob_gateway_public_url=f"http://127.0.0.1:{gateway_port}",
                blob_gateway_bind=f"127.0.0.1:{gateway_port}", blob_gateway_executable=install / "KrakenBlobGateway.exe")
            admin(["migrate", "--config", config.path], "baseline_migrations")
            admin(["bootstrap-admin", "--config", config.path, "--username", "upgrade-owner",
                   "--display-name", "Own upgrade acceptance"], "baseline_bootstrap")
            configuration_bytes = {p.name: sha(p) for p in config.path.parent.iterdir() if p.is_file()}
            versions = []
            with server(config, "baseline_server") as client:
                project = request(client, "POST", "/api/v1/projects", expected=201, headers={"Idempotency-Key": "upgrade-project"},
                    json=dict(name="Retained real corpus", width=4, height=(len(pairs) + 3) // 4, orientation="y_down"))
                prefix = "/api/v1/projects/" + project.get("project_id", project.get("id"))
                for index, pair in enumerate(pairs):
                    for kind in ("image", "vector"):
                        path = Path(pair[kind])
                        series = request(client, "POST", prefix + "/artifacts", expected=201,
                            headers={"Idempotency-Key": f"series-{index}-{kind}"},
                            json=dict(name=path.name, scope="project_attachment"))
                        sid = series.get("id", series.get("series_id"))
                        upload_path = prefix + f"/artifacts/{sid}/uploads"
                        headers = {"Idempotency-Key": f"upload-{index}-{kind}", "If-Match": "1"}
                        payload = dict(filename=path.name, media_type=artifact_media_type(path.name),
                            sha256=pair[kind + "_sha256"], size_bytes=path.stat().st_size)
                        ticket = request(client, "POST", upload_path, headers=headers, json=payload)
                        assert ticket["mode"] == "gateway"
                        with path.open("rb") as stream:
                            transfer = httpx.put(ticket["url"], content=stream, timeout=30, trust_env=False,
                                headers={"Authorization": "Bearer " + ticket["token"], "Content-Length": str(payload["size_bytes"])})
                        assert transfer.status_code == 200
                        completed = request(client, "POST", upload_path + "/complete", expected=201, headers=headers,
                            json={**payload, "upload_token": ticket["token"]})
                        versions.append(dict(id=completed.get("id", completed.get("artifact_version_id")), sha256=payload["sha256"]))
                history = request(client, "GET", prefix + "/history", params=dict(limit=500))
                files_readback(client, prefix, versions, "baseline_real_files")
                setup(new, "busy_update_refused", expected=7)
                assert process.poll() is None
                assert request(client, "GET", prefix)["name"] == "Retained real corpus"
                check_payload(old_files, "busy_refusal_preserved_payload")
                passed("busy_refusal_preserved_running_server")
            setup(new, "quiescent_update")
            check_payload(new_files, "updated_payload")
            assert {p.name: sha(p) for p in config.path.parent.iterdir() if p.is_file()} == configuration_bytes
            admin(["migrate", "--config", config.path], "updated_migrations")
            with server(config, "updated_server") as client:
                assert request(client, "GET", prefix)["name"] == "Retained real corpus"
                assert request(client, "GET", prefix + "/history", params=dict(limit=500)) == history
                files_readback(client, prefix, versions, "updated_real_files")
                run([owned_registration(), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                    f"/LOG={report / 'busy-uninstall.log'}"], "busy_uninstall_refused", expected=1)
                owned_registration()
                assert process.poll() is None
                check_payload(new_files, "busy_uninstall_preserved_payload")
                archive = report / "post-upgrade-backup"
                admin(["backup-server", "--config", config.path, "--destination", archive, "--pg-bin-dir", args.pg_bin_dir], "updated_backup")
                admin(["backup-verify", "--backup", archive], "updated_backup_verify")
            environment["KRAKEN_RESTORE_DATABASE_URL"] = urls[1]
            restored = admin(["restore-server", "--backup", archive, "--destination", report / "recovered-data",
                "--config-dir", report / "recovered-config", "--pg-bin-dir", args.pg_bin_dir,
                "--blob-gateway-executable", install / "KrakenBlobGateway.exe"], "updated_restore")
            with server(ServerConfig.load(restored["config"]), "recovered_server") as client:
                assert request(client, "GET", prefix)["name"] == "Retained real corpus"
                assert request(client, "GET", prefix + "/history", params=dict(limit=500)) == history
                files_readback(client, prefix, versions, "recovered_real_files")
            assert {p.name: sha(p) for p in config.path.parent.iterdir() if p.is_file()} == configuration_bytes
            passed("retained_configuration_history_and_recovery", versions=len(versions))
        else:
            acceptance = report / "baseline-desktop"
            with (report / "baseline-desktop.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([str(install / "KrakenDesktop.exe"), "--self-test-dir", str(acceptance),
                    "--self-test-reference-manifest", str(reference)], stdout=log, stderr=subprocess.STDOUT,
                    env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    assert process.poll() is None
                    setup(new, "busy_update_refused", expected=7)
                    assert process.poll() is None
                    check_payload(old_files, "busy_refusal_preserved_payload")
                    assert process.wait(timeout=120) == 0
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
                    process = None
            reports = list(acceptance.glob("*/report.json"))
            assert len(reports) == 1
            baseline = json.loads(reports[0].read_text(encoding="utf-8"))
            assert baseline["status"] == "passed" and baseline["source_files_unchanged"] and baseline["settings_isolated"]
            evidence["baseline_desktop_acceptance"] = baseline
            setup(new, "quiescent_update")
            check_payload(new_files, "updated_payload")
            run([install / "KrakenDesktop.exe", "--self-test-dir", report / "reopened-baseline",
                 "--self-test-reopen-report", reports[0]], "updated_desktop_reopens_baseline")
            reopen_reports = list((report / "reopened-baseline").glob("*/report.json"))
            assert len(reopen_reports) == 1
            reopened = json.loads(reopen_reports[0].read_text(encoding="utf-8"))
            if (reopened["status"] != "passed" or not reopened.get("old_account_authenticated")
                    or not reopened.get("canonical_data_unchanged") or not reopened.get("previous_settings_unchanged")
                    or not reopened.get("qt_matrix_and_thumbnails") or reopened.get("pairs") != len(pairs)):
                raise RuntimeError("Updated Desktop did not preserve/reopen the previous release's actual project")
            evidence["previous_desktop_reopened"] = reopened
            passed("updated_desktop_reopened_previous_account_project", pairs=reopened["pairs"])
            run([install / "KrakenDesktop.exe", "--self-test-dir", report / "updated-desktop",
                 "--self-test-reference-manifest", reference], "updated_desktop_acceptance")
            reports = list((report / "updated-desktop").glob("*/report.json"))
            assert len(reports) == 1
            updated = json.loads(reports[0].read_text(encoding="utf-8"))
            assert updated["status"] == "passed" and updated["source_files_unchanged"] and updated["settings_isolated"]
            evidence["updated_desktop_acceptance"] = updated
            passed("updated_desktop_real_pair_acceptance", pairs=len(pairs))
        assert sha(user_note) == note_sha
        obsolete = sorted(relative for relative in old_files.keys() - new_files.keys() if (install / relative).exists())
        evidence["obsolete_baseline_files"] = obsolete
        if obsolete:
            raise RuntimeError("Obsolete baseline runtime files remain after update")
        passed("untracked_user_file_preserved")
        evidence["status"] = "passed"
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
            if created:
                with _connect(parent_url) as administrator:
                    for name in created:
                        assert name.startswith("kraken_upgrade_audit_") and len(name) == len("kraken_upgrade_audit_") + 32
                        administrator.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
                evidence["owned_databases_removed"] = True
            if installed:
                run([owned_registration(), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                    f"/LOG={report / 'uninstall.log'}"], "audit_uninstall")
                evidence["registration_removed"] = registration(app_id) is None
                evidence["executable_removed"] = not (install / f"Kraken{target}.exe").exists()
                evidence["untracked_user_file_retained_after_uninstall"] = sha(user_note) == note_sha
                if not all(evidence[k] for k in ("registration_removed", "executable_removed", "untracked_user_file_retained_after_uninstall")):
                    raise RuntimeError("Audit uninstall did not preserve data/remove only its own application")
            evidence["source_files_unchanged"] = bool(source_files) and all(sha(path) == digest for path, digest in source_files.items())
            evidence["service_after_sha256"] = service_hash()
            if evidence["service_after_sha256"] != before_service or not evidence["source_files_unchanged"]:
                raise RuntimeError("Service ownership or reference corpus changed")
        except BaseException as exc:
            failure = failure or exc
            evidence.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
        evidence["seconds"] = round(time.monotonic() - started, 3)
        (report / "report.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if failure:
        raise failure
    print(json.dumps(dict(status=evidence["status"], target=target, seconds=evidence["seconds"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
